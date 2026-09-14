#!/bin/bash

################################################################################
# Confidence Engine (LisaSimpson Integration)
#
# Calculates task success probability (0.0-1.0) and maps to adaptive retry limits.
# Enables Ralph Wiggum retry-until-verified loops with intelligent backoff.
#
# Scoring Formula:
# - Historical success rate (40% weight): From decision-log.jsonl past outcomes
# - Task complexity (30% weight): Heuristic estimation from task description
# - Prerequisites availability (30% weight): Can we access needed files/resources?
#
# Retry Mapping:
# - High confidence (≥0.8) → 5 retry attempts
# - Medium confidence (0.5-0.79) → 3 retry attempts
# - Low confidence (<0.5) → 1 retry attempt
#
# Authors: LisaSimpson + Autonomy Team
# Created: 2025-01-08
################################################################################

set -euo pipefail

# Daemon root
DAEMON_ROOT="${DAEMON_ROOT:-.}"

################################################################################
# HISTORICAL SUCCESS RATE CALCULATION
################################################################################

# Calculate success rate for a task type from historical decision log
# Usage: get_historical_success_rate <task_description> [persona]
# Returns: Float 0.0-1.0
get_historical_success_rate() {
    local task_description="$1"
    local persona="${2:-}"

    # Extract task type/keywords from description
    local task_type
    task_type=$(extract_task_type "$task_description")

    # Query decision-log.jsonl for matching tasks
    local decision_log="${DAEMON_ROOT}/logs/decision-log.jsonl"

    if [ ! -f "$decision_log" ]; then
        # No historical data, assume neutral
        echo "0.5"
        return 0
    fi

    # Count total and successful outcomes
    local total_tasks=0
    local successful_tasks=0

    while IFS= read -r line; do
        [ -z "$line" ] && continue

        # Check if line matches task_type
        if echo "$line" | jq -e ".task_type == \"$task_type\" or .description | test(\"$task_type\"; \"i\")" >/dev/null 2>&1; then
            total_tasks=$((total_tasks + 1))

            # Check if successful
            if echo "$line" | jq -e ".outcome == \"success\"" >/dev/null 2>&1; then
                successful_tasks=$((successful_tasks + 1))
            fi
        fi
    done < "$decision_log"

    # Calculate rate
    if [ "$total_tasks" -eq 0 ]; then
        echo "0.5"
    else
        local rate
        rate=$(echo "scale=2; $successful_tasks / $total_tasks" | bc -l)
        # Ensure result is between 0.0 and 1.0
        if (( $(echo "$rate > 1" | bc -l) )); then
            echo "1.0"
        elif (( $(echo "$rate < 0" | bc -l) )); then
            echo "0.0"
        else
            echo "$rate"
        fi
    fi
}

# Extract task type from task description
# Returns: Task type keyword (write, analyze, refactor, debug, test, deploy, etc.)
extract_task_type() {
    local description="$1"

    # Simple heuristic: look for common task type keywords
    if echo "$description" | grep -qi "^write\|create\|draft\|compose"; then
        echo "write"
    elif echo "$description" | grep -qi "^analyze\|review\|evaluate\|assess"; then
        echo "analyze"
    elif echo "$description" | grep -qi "^refactor\|optimize\|improve\|enhance"; then
        echo "refactor"
    elif echo "$description" | grep -qi "^debug\|fix\|troubleshoot\|resolve"; then
        echo "debug"
    elif echo "$description" | grep -qi "^test\|verify\|validate\|check"; then
        echo "test"
    elif echo "$description" | grep -qi "^deploy\|release\|publish\|launch"; then
        echo "deploy"
    elif echo "$description" | grep -qi "^research\|investigate\|explore"; then
        echo "research"
    else
        echo "general"
    fi
}

################################################################################
# TASK COMPLEXITY ESTIMATION
################################################################################

# Estimate task complexity based on description
# Returns: Float 0.0-1.0 where 1.0 = very complex, 0.0 = trivial
estimate_task_complexity() {
    local description="$1"

    # Count complexity indicators
    local complexity_score=0.3  # Base complexity

    # Increase complexity for longer descriptions (indicates more work)
    local desc_length=${#description}
    if [ "$desc_length" -gt 500 ]; then
        complexity_score=$(echo "$complexity_score + 0.2" | bc -l)
    elif [ "$desc_length" -gt 200 ]; then
        complexity_score=$(echo "$complexity_score + 0.1" | bc -l)
    fi

    # Check for complexity keywords
    if echo "$description" | grep -qi "multiple\|several\|various\|many"; then
        complexity_score=$(echo "$complexity_score + 0.1" | bc -l)
    fi

    if echo "$description" | grep -qi "refactor\|redesign\|restructure\|rewrite"; then
        complexity_score=$(echo "$complexity_score + 0.15" | bc -l)
    fi

    if echo "$description" | grep -qi "integrate\|combine\|merge\|coordinate"; then
        complexity_score=$(echo "$complexity_score + 0.15" | bc -l)
    fi

    if echo "$description" | grep -qi "difficult\|challenging\|complex\|intricate"; then
        complexity_score=$(echo "$complexity_score + 0.2" | bc -l)
    fi

    # Check for simplicity indicators
    if echo "$description" | grep -qi "simple\|basic\|straightforward\|trivial"; then
        complexity_score=$(echo "$complexity_score - 0.2" | bc -l)
    fi

    # Clamp to 0.0-1.0
    if (( $(echo "$complexity_score > 1" | bc -l) )); then
        echo "1.0"
    elif (( $(echo "$complexity_score < 0" | bc -l) )); then
        echo "0.0"
    else
        echo "$complexity_score"
    fi
}

################################################################################
# PREREQUISITES AVAILABILITY CHECK
################################################################################

# Check if task prerequisites are available
# Usage: check_prerequisites_availability <goal_json> [success_criteria_json]
# Returns: Float 0.0-1.0 (availability score)
check_prerequisites_availability() {
    local goal_json="$1"
    local success_criteria="${2:-}"

    # Extract dependencies and blockers from goal
    local blockers
    blockers=$(echo "$goal_json" | jq '.blockers | length' 2>/dev/null || echo "0")

    local dependencies
    dependencies=$(echo "$goal_json" | jq '.dependencies | length' 2>/dev/null || echo "0")

    local availability_score=1.0

    # Reduce score based on blockers (each blocker reduces by 0.2)
    if [ "$blockers" -gt 0 ]; then
        local blocker_reduction
        blocker_reduction=$(echo "scale=2; $blockers * 0.2" | bc -l)
        availability_score=$(echo "$availability_score - $blocker_reduction" | bc -l)
    fi

    # Check if dependencies are satisfied
    if [ "$dependencies" -gt 0 ]; then
        # Would need to check goal status, for now reduce score slightly
        availability_score=$(echo "$availability_score - 0.1" | bc -l)
    fi

    # Check for critical files/resources mentioned
    local success_criteria_array
    success_criteria_array=$(echo "$goal_json" | jq '.success_criteria | length' 2>/dev/null || echo "0")

    local blocked_criteria=0
    if [ "$success_criteria_array" -gt 0 ]; then
        blocked_criteria=$(echo "$goal_json" | jq '[.success_criteria[] | select(.status == "blocked")] | length' 2>/dev/null || echo "0")
    fi

    if [ "$blocked_criteria" -gt 0 ]; then
        local blocked_reduction
        blocked_reduction=$(echo "scale=2; $blocked_criteria * 0.15" | bc -l)
        availability_score=$(echo "$availability_score - $blocked_reduction" | bc -l)
    fi

    # Clamp to 0.0-1.0
    if (( $(echo "$availability_score > 1" | bc -l) )); then
        echo "1.0"
    elif (( $(echo "$availability_score < 0" | bc -l) )); then
        echo "0.0"
    else
        echo "$availability_score"
    fi
}

################################################################################
# CONFIDENCE CALCULATION
################################################################################

# Calculate task success confidence
# Usage: calculate_task_confidence <task_description> <goal_json> [persona]
# Returns: JSON with confidence score and breakdown
calculate_task_confidence() {
    local task_description="$1"
    local goal_json="$2"
    local persona="${3:-}"

    # Component scores
    local historical_success
    historical_success=$(get_historical_success_rate "$task_description" "$persona")

    local complexity
    complexity=$(estimate_task_complexity "$task_description")

    local prerequisites
    prerequisites=$(check_prerequisites_availability "$goal_json")

    # Invert complexity: high complexity = low confidence for this component
    local complexity_component
    complexity_component=$(echo "scale=2; 1.0 - $complexity" | bc -l)

    # Weighted formula: 40% historical, 30% complexity, 30% prerequisites
    local confidence_score
    confidence_score=$(echo "scale=3; ($historical_success * 0.4) + ($complexity_component * 0.3) + ($prerequisites * 0.3)" | bc -l)

    # Ensure 0.0-1.0
    if (( $(echo "$confidence_score > 1" | bc -l) )); then
        confidence_score="1.0"
    elif (( $(echo "$confidence_score < 0" | bc -l) )); then
        confidence_score="0.0"
    fi

    # Build result JSON
    jq -n \
        --arg task_desc "$task_description" \
        --argjson historical "$historical_success" \
        --argjson complexity "$complexity_component" \
        --argjson prerequisites "$prerequisites" \
        --argjson total "$confidence_score" \
        --arg persona "$persona" \
        '{
            task_description: ($task_desc | .[0:50]),
            confidence_score: ($total | tonumber),
            components: {
                historical_success_rate: ($historical | tonumber),
                complexity_component: ($complexity | tonumber),
                prerequisites_availability: ($prerequisites | tonumber)
            },
            calculation: "40% historical + 30% (1-complexity) + 30% prerequisites",
            persona: $persona,
            timestamp: (now | floor | todate)
        }'
}

################################################################################
# RETRY LIMIT MAPPING
################################################################################

# Map confidence score to retry attempt limit
# Usage: map_confidence_to_retry_limit <confidence_score>
# Returns: Integer (1, 3, or 5)
map_confidence_to_retry_limit() {
    local confidence_score="$1"

    # High confidence (≥0.8) → 5 retries (trust the system)
    if (( $(echo "$confidence_score >= 0.8" | bc -l) )); then
        echo "5"
    # Medium confidence (0.5-0.79) → 3 retries (cautious)
    elif (( $(echo "$confidence_score >= 0.5" | bc -l) )); then
        echo "3"
    # Low confidence (<0.5) → 1 retry (fail fast)
    else
        echo "1"
    fi
}

# Get retry classification
# Usage: classify_confidence <confidence_score>
# Returns: "high", "medium", or "low"
classify_confidence() {
    local confidence_score="$1"

    if (( $(echo "$confidence_score >= 0.8" | bc -l) )); then
        echo "high"
    elif (( $(echo "$confidence_score >= 0.5" | bc -l) )); then
        echo "medium"
    else
        echo "low"
    fi
}

################################################################################
# CONFIDENCE CALIBRATION
################################################################################

# Calculate confidence accuracy (predicted vs actual success)
# Usage: calculate_confidence_accuracy <predictions_json> <actual_outcomes_json>
# Returns: JSON with accuracy metrics
calculate_confidence_accuracy() {
    local predictions="$1"
    local actual_outcomes="$2"

    # This would require historical paired data
    # For now, return a baseline structure
    jq -n '{
        total_predictions: 0,
        correct_predictions: 0,
        accuracy_percentage: 0,
        mean_error: 0,
        calibration_status: "insufficient_data",
        recommendation: "Collect more prediction-outcome pairs"
    }'
}

# Adjust confidence weights based on calibration feedback
# Usage: adjust_confidence_weights <calibration_result>
# Returns: Updated weight factors
adjust_confidence_weights() {
    local calibration_result="$1"

    # Extract accuracy from calibration
    local accuracy
    accuracy=$(echo "$calibration_result" | jq -r '.accuracy_percentage')

    # If accuracy < 70%, suggest weight adjustment
    if (( $(echo "$accuracy < 70" | bc -l) )); then
        echo "Weights need adjustment: historical=$(70) complexity=$(25) prerequisites=$(5)"
    else
        echo "Weights are well-calibrated"
    fi
}

################################################################################
# CONFIDENCE PERSISTENCE
################################################################################

# Save calculated confidence to goal
# Usage: save_confidence_to_goal <goal_json> <confidence_score>
# Returns: Updated goal JSON
save_confidence_to_goal() {
    local goal_json="$1"
    local confidence_score="$2"

    echo "$goal_json" | jq \
        --argjson confidence "$confidence_score" \
        '.confidence_score = $confidence'
}

# Get confidence from goal or calculate if missing
# Usage: get_or_calculate_confidence <goal_json> <task_description> [persona]
# Returns: Confidence score (0.0-1.0)
get_or_calculate_confidence() {
    local goal_json="$1"
    local task_description="$2"
    local persona="${3:-}"

    # Check if goal already has confidence score
    local existing_confidence
    existing_confidence=$(echo "$goal_json" | jq -r '.confidence_score // null')

    if [ "$existing_confidence" != "null" ] && [ "$existing_confidence" != "" ]; then
        echo "$existing_confidence"
    else
        # Calculate and return
        local confidence_json
        confidence_json=$(calculate_task_confidence "$task_description" "$goal_json" "$persona")
        echo "$confidence_json" | jq -r '.confidence_score'
    fi
}

################################################################################
# CONFIDENCE CACHING (PHASE 7D OPTIMIZATION)
################################################################################

# Initialize cache directory
_init_confidence_cache() {
    mkdir -p "${DAEMON_ROOT}/.cache" 2>/dev/null || true
}

# Calculate cache key from task description
_confidence_cache_key() {
    echo "$1" | md5sum | awk '{print $1}'
}

# Get cached confidence if available and fresh
_get_cached_confidence() {
    local task_description="$1"
    local cache_key
    cache_key=$(_confidence_cache_key "$task_description")

    local cache_file="${DAEMON_ROOT}/.cache/confidence_${cache_key}.json"

    if [ -f "$cache_file" ]; then
        # Check age (15-minute TTL = 900 seconds)
        local file_age=$(($(date +%s) - $(stat -c %Y "$cache_file" 2>/dev/null || stat -f %m "$cache_file" 2>/dev/null || echo 0)))
        if [ "$file_age" -lt 900 ]; then
            # Cache hit
            cat "$cache_file"
            return 0
        fi
    fi

    return 1
}

# Cache confidence result
_cache_confidence() {
    local task_description="$1"
    local result="$2"

    local cache_key
    cache_key=$(_confidence_cache_key "$task_description")

    local cache_file="${DAEMON_ROOT}/.cache/confidence_${cache_key}.json"

    echo "$result" > "$cache_file" 2>/dev/null || true
}

# Calculate confidence with caching
# Usage: calculate_task_confidence_cached <task_description> <goal_json> [persona]
# Returns: JSON with confidence score and breakdown (from cache or calculated)
calculate_task_confidence_cached() {
    local task_description="$1"
    local goal_json="$2"
    local persona="${3:-}"

    _init_confidence_cache

    # Try cache first
    if cached_result=$(_get_cached_confidence "$task_description" 2>/dev/null); then
        return 0
    fi

    # Cache miss - calculate
    local result
    result=$(calculate_task_confidence "$task_description" "$goal_json" "$persona")

    # Store in cache
    _cache_confidence "$task_description" "$result"

    echo "$result"
}

# Clean old cache entries (older than 24 hours)
_cleanup_confidence_cache() {
    local cache_dir="${DAEMON_ROOT}/.cache"
    if [ ! -d "$cache_dir" ]; then
        return
    fi

    find "$cache_dir" -name "confidence_*.json" -type f -mtime +1 -delete 2>/dev/null || true
}

################################################################################
# EXPORTS
################################################################################

export -f get_historical_success_rate
export -f extract_task_type
export -f estimate_task_complexity
export -f check_prerequisites_availability
export -f calculate_task_confidence
export -f map_confidence_to_retry_limit
export -f classify_confidence
export -f calculate_confidence_accuracy
export -f adjust_confidence_weights
export -f save_confidence_to_goal
export -f get_or_calculate_confidence
export -f calculate_task_confidence_cached

################################################################################
# SELF-TEST
################################################################################

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Running Confidence Engine Self-Tests..." >&2
    echo ""

    # Test 1: Extract task type
    echo "Test 1: Task type extraction" >&2
    echo "  'Write a chapter' → $(extract_task_type "Write a chapter")" >&2
    echo "  'Refactor the API' → $(extract_task_type "Refactor the API")" >&2
    echo "  'Debug the issue' → $(extract_task_type "Debug the issue")" >&2
    echo ""

    # Test 2: Complexity estimation
    echo "Test 2: Task complexity estimation" >&2
    echo "  Simple task: $(estimate_task_complexity "Fix typo")" >&2
    echo "  Complex task: $(estimate_task_complexity "Refactor entire authentication system to support OAuth and JWT while maintaining backward compatibility with multiple database backends and integrating with existing microservices")" >&2
    echo ""

    # Test 3: Confidence calculation
    echo "Test 3: Confidence calculation" >&2
    goal=$(jq -n '{goal_id: "test", blockers: [], dependencies: [], success_criteria: []}')
    confidence=$(calculate_task_confidence "Write a simple chapter" "$goal" "architect")
    echo "Simple write task confidence:" >&2
    echo "$confidence" | jq '.confidence_score' >&2
    echo ""

    # Test 4: Retry limit mapping
    echo "Test 4: Retry limit mapping" >&2
    echo "  High confidence (0.85) → $(map_confidence_to_retry_limit "0.85") retries" >&2
    echo "  Medium confidence (0.65) → $(map_confidence_to_retry_limit "0.65") retries" >&2
    echo "  Low confidence (0.35) → $(map_confidence_to_retry_limit "0.35") retries" >&2
    echo ""

    # Test 5: Confidence classification
    echo "Test 5: Confidence classification" >&2
    echo "  0.9 → $(classify_confidence "0.9")" >&2
    echo "  0.65 → $(classify_confidence "0.65")" >&2
    echo "  0.3 → $(classify_confidence "0.3")" >&2
    echo ""

    echo "✓ Confidence Engine self-tests completed!" >&2
fi
