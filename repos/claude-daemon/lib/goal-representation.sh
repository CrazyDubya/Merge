#!/bin/bash

################################################################################
# Goal Representation Library
#
# Provides schema and utilities for representing and managing goals in the
# autonomous daemon system.
#
# A goal represents "what should be accomplished" - distinct from tasks which
# are specific actionable items. Goals have success criteria, progress tracking,
# and can be hierarchically organized.
#
# Authors: Architect + Autonomy Rebuild Team
# Created: 2025-12-23
################################################################################

set -euo pipefail

# Daemon root - make sure it's set
DAEMON_ROOT="${DAEMON_ROOT:-.}"

################################################################################
# GOAL SCHEMA DEFINITION
################################################################################

# Goal structure (JSON):
# {
#   "goal_id": "unique_identifier",
#   "type": "project_completion|system_optimization|learning|creative_work",
#   "description": "Human-readable goal description",
#   "success_criteria": [
#     {
#       "criterion": "criterion_id",
#       "description": "What needs to be true for this criterion to pass",
#       "measurable": true|false,
#       "status": "not_started|in_progress|completed|blocked",
#       "evidence": "How we know this is true (file path, metric, etc)",
#       "verified": true|false
#     }
#   ],
#   "progress": 0-100,
#   "blockers": ["blocker_id"],
#   "dependencies": ["parent_goal_id"],
#   "sub_goals": ["sub_goal_id"],
#   "created_at": "ISO8601",
#   "updated_at": "ISO8601",
#   "target_completion": "ISO8601_or_null",
#   "completed_at": "ISO8601_or_null",
#   "metadata": {
#     "priority": "high|medium|low",
#     "effort_estimate_hours": number,
#     "effort_spent_hours": number,
#     "assigned_persona": "optimizer|architect|etc",
#     "tags": ["tag1", "tag2"]
#   },
#   "world_state": {  # LISASIMPSON: Optional - explicit state modeling
#     "pre_conditions": {},
#     "post_conditions": {},
#     "state_variables": {}
#   },
#   "confidence_score": 0.0-1.0,  # LISASIMPSON: Task success probability
#   "verification_plan": {  # LISASIMPSON: Auto-generated OUTPUT/VERIFY
#     "checks": []
#   }
# }

################################################################################
# GOAL TYPES
################################################################################

# Define valid goal types
if [ -z "${GOAL_TYPES:-}" ]; then
    GOAL_TYPES=(
        "project_completion"      # Complete a defined project (e.g., "novel publishable")
        "system_optimization"      # Improve system performance/reliability
        "learning"                 # Learn/understand something
        "creative_work"            # Create something new
        "maintenance"              # Ongoing maintenance/upkeep
        "research"                 # Research/investigation
    )
fi

# Check if a goal type is valid
is_valid_goal_type() {
    local type="$1"
    for valid_type in "${GOAL_TYPES[@]}"; do
        if [ "$type" = "$valid_type" ]; then
            return 0
        fi
    done
    return 1
}

################################################################################
# GOAL VALIDATION
################################################################################

# Validate goal JSON structure
# Returns 0 if valid, 1 if invalid
validate_goal_structure() {
    local goal_json="$1"

    # Check required fields
    if ! echo "$goal_json" | jq -e '.goal_id' >/dev/null 2>&1; then
        echo "ERROR: Missing required field: goal_id" >&2
        return 1
    fi

    if ! echo "$goal_json" | jq -e '.type' >/dev/null 2>&1; then
        echo "ERROR: Missing required field: type" >&2
        return 1
    fi

    if ! echo "$goal_json" | jq -e '.description' >/dev/null 2>&1; then
        echo "ERROR: Missing required field: description" >&2
        return 1
    fi

    if ! echo "$goal_json" | jq -e '.success_criteria' >/dev/null 2>&1; then
        echo "ERROR: Missing required field: success_criteria" >&2
        return 1
    fi

    # Validate type is legal
    local goal_type
    goal_type=$(echo "$goal_json" | jq -r '.type')
    if ! is_valid_goal_type "$goal_type"; then
        echo "ERROR: Invalid goal type: $goal_type" >&2
        return 1
    fi

    # Validate success_criteria is array
    if ! echo "$goal_json" | jq -e '.success_criteria | type == "array"' >/dev/null 2>&1; then
        echo "ERROR: success_criteria must be an array" >&2
        return 1
    fi

    # Validate each success criterion
    local criterion_count
    criterion_count=$(echo "$goal_json" | jq '.success_criteria | length')
    for ((i=0; i<criterion_count; i++)); do
        local criterion
        criterion=$(echo "$goal_json" | jq ".success_criteria[$i]")

        if ! echo "$criterion" | jq -e '.criterion' >/dev/null 2>&1; then
            echo "ERROR: success_criteria[$i] missing 'criterion' field" >&2
            return 1
        fi

        if ! echo "$criterion" | jq -e '.description' >/dev/null 2>&1; then
            echo "ERROR: success_criteria[$i] missing 'description' field" >&2
            return 1
        fi
    done

    return 0
}

################################################################################
# GOAL CREATION
################################################################################

# Create a new goal with validation
# Usage: create_goal <goal_id> <type> <description> <success_criteria_json> [target_completion] [world_state_json] [confidence_score] [verification_plan_json]
# Returns: Goal JSON
create_goal() {
    local goal_id="$1"
    local goal_type="$2"
    local description="$3"
    local success_criteria="$4"
    local target_completion="${5:-null}"
    local world_state="${6:-null}"
    local confidence_score="${7:-null}"
    local verification_plan="${8:-null}"

    # Validate type
    if ! is_valid_goal_type "$goal_type"; then
        echo "ERROR: Invalid goal type: $goal_type" >&2
        return 1
    fi

    # Validate success_criteria is valid JSON array
    if ! echo "$success_criteria" | jq -e '. | type == "array"' >/dev/null 2>&1; then
        echo "ERROR: success_criteria must be a JSON array" >&2
        return 1
    fi

    local now
    now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Build goal JSON with optional LISASIMPSON fields
    local goal_json
    if [ "$target_completion" = "null" ]; then
        goal_json=$(jq -n \
            --arg goal_id "$goal_id" \
            --arg type "$goal_type" \
            --arg description "$description" \
            --argjson success_criteria "$success_criteria" \
            --arg created_at "$now" \
            --arg updated_at "$now" \
            --argjson world_state "$world_state" \
            --argjson confidence_score "$confidence_score" \
            --argjson verification_plan "$verification_plan" \
            '{
                goal_id: $goal_id,
                type: $type,
                description: $description,
                success_criteria: $success_criteria,
                progress: 0,
                blockers: [],
                dependencies: [],
                sub_goals: [],
                created_at: $created_at,
                updated_at: $updated_at,
                target_completion: null,
                completed_at: null,
                metadata: {
                    priority: "medium",
                    effort_estimate_hours: 0,
                    effort_spent_hours: 0,
                    assigned_persona: null,
                    tags: []
                }
            } |
            if $world_state != null then . + {world_state: $world_state} else . end |
            if $confidence_score != null then . + {confidence_score: $confidence_score} else . end |
            if $verification_plan != null then . + {verification_plan: $verification_plan} else . end')
    else
        goal_json=$(jq -n \
            --arg goal_id "$goal_id" \
            --arg type "$goal_type" \
            --arg description "$description" \
            --argjson success_criteria "$success_criteria" \
            --arg created_at "$now" \
            --arg updated_at "$now" \
            --arg target_completion "$target_completion" \
            --argjson world_state "$world_state" \
            --argjson confidence_score "$confidence_score" \
            --argjson verification_plan "$verification_plan" \
            '{
                goal_id: $goal_id,
                type: $type,
                description: $description,
                success_criteria: $success_criteria,
                progress: 0,
                blockers: [],
                dependencies: [],
                sub_goals: [],
                created_at: $created_at,
                updated_at: $updated_at,
                target_completion: $target_completion,
                completed_at: null,
                metadata: {
                    priority: "medium",
                    effort_estimate_hours: 0,
                    effort_spent_hours: 0,
                    assigned_persona: null,
                    tags: []
                }
            } |
            if $world_state != null then . + {world_state: $world_state} else . end |
            if $confidence_score != null then . + {confidence_score: $confidence_score} else . end |
            if $verification_plan != null then . + {verification_plan: $verification_plan} else . end')
    fi

    # Validate the created goal
    if ! validate_goal_structure "$goal_json"; then
        return 1
    fi

    echo "$goal_json"
}

################################################################################
# GOAL PROGRESS CALCULATION
################################################################################

# Calculate goal progress based on success criteria status
# Progress = (completed_criteria / total_criteria) * 100
# Usage: calculate_goal_progress <goal_json>
# Returns: Progress percentage (0-100)
calculate_goal_progress() {
    local goal_json="$1"

    local total_criteria
    total_criteria=$(echo "$goal_json" | jq '.success_criteria | length')

    if [ "$total_criteria" -eq 0 ]; then
        echo 0
        return 0
    fi

    local completed_criteria
    completed_criteria=$(echo "$goal_json" | jq '[.success_criteria[] | select(.status == "completed")] | length')

    # Calculate percentage
    local progress=$((completed_criteria * 100 / total_criteria))
    echo "$progress"
}

################################################################################
# BLOCKER DETECTION
################################################################################

# Identify blockers preventing goal progress
# Returns list of blockers
identify_blockers() {
    local goal_json="$1"

    local blockers_json
    blockers_json=$(echo "$goal_json" | jq '[
        .success_criteria[] |
        select(.status == "blocked") |
        {
            criterion: .criterion,
            reason: (.reason // "No reason specified"),
            blocking_factor: (
                if .status == "blocked" then
                    .criterion
                else
                    empty
                end
            )
        }
    ]')

    echo "$blockers_json"
}

################################################################################
# GOAL STATUS QUERIES
################################################################################

# Get goal status summary
# Returns: JSON with status overview
get_goal_status() {
    local goal_json="$1"

    local total
    total=$(echo "$goal_json" | jq '.success_criteria | length')

    local completed
    completed=$(echo "$goal_json" | jq '[.success_criteria[] | select(.status == "completed")] | length')

    local in_progress
    in_progress=$(echo "$goal_json" | jq '[.success_criteria[] | select(.status == "in_progress")] | length')

    local blocked
    blocked=$(echo "$goal_json" | jq '[.success_criteria[] | select(.status == "blocked")] | length')

    local not_started
    not_started=$(echo "$goal_json" | jq '[.success_criteria[] | select(.status == "not_started")] | length')

    local progress
    progress=$(calculate_goal_progress "$goal_json")

    jq -n \
        --arg goal_id "$(echo "$goal_json" | jq -r '.goal_id')" \
        --arg description "$(echo "$goal_json" | jq -r '.description')" \
        --argjson total "$total" \
        --argjson completed "$completed" \
        --argjson in_progress "$in_progress" \
        --argjson blocked "$blocked" \
        --argjson not_started "$not_started" \
        --argjson progress "$progress" \
        '{
            goal_id: $goal_id,
            description: $description,
            progress: $progress,
            total_criteria: $total,
            completed: $completed,
            in_progress: $in_progress,
            blocked: $blocked,
            not_started: $not_started
        }'
}

# Check if goal is complete
# Returns 0 if complete, 1 if not
is_goal_complete() {
    local goal_json="$1"

    local progress
    progress=$(calculate_goal_progress "$goal_json")

    if [ "$progress" -eq 100 ]; then
        return 0
    fi
    return 1
}

# Check if goal is blocked
# Returns 0 if blocked, 1 if not
is_goal_blocked() {
    local goal_json="$1"

    local blocked_count
    blocked_count=$(echo "$goal_json" | jq '[.success_criteria[] | select(.status == "blocked")] | length')

    if [ "$blocked_count" -gt 0 ]; then
        return 0
    fi
    return 1
}

################################################################################
# GOAL UPDATES
################################################################################

# Update a success criterion status
# Usage: update_criterion_status <goal_json> <criterion_id> <status> <evidence> [reason]
# Returns: Updated goal JSON
update_criterion_status() {
    local goal_json="$1"
    local criterion_id="$2"
    local status="$3"
    local evidence="$4"
    local reason="${5:-}"

    local now
    now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Update the criterion and recalculate progress
    goal_json=$(echo "$goal_json" | jq \
        --arg criterion_id "$criterion_id" \
        --arg status "$status" \
        --arg evidence "$evidence" \
        --arg reason "$reason" \
        --arg updated_at "$now" \
        '.success_criteria |= map(
            if .criterion == $criterion_id then
                . + {
                    status: $status,
                    evidence: $evidence,
                    verified: true
                } + (if $reason != "" then {reason: $reason} else {} end)
            else
                .
            end
        ) |
        . + {updated_at: $updated_at}'
    )

    echo "$goal_json"
}

# Update goal progress (calculated field)
# Usage: update_goal_progress <goal_json>
# Returns: Goal JSON with updated progress
update_goal_progress() {
    local goal_json="$1"

    local progress
    progress=$(calculate_goal_progress "$goal_json")

    local now
    now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    echo "$goal_json" | jq \
        --argjson progress "$progress" \
        --arg updated_at "$now" \
        '.progress = $progress | .updated_at = $updated_at'
}

# Mark goal as complete
# Usage: complete_goal <goal_json>
# Returns: Updated goal JSON
complete_goal() {
    local goal_json="$1"

    local now
    now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    echo "$goal_json" | jq \
        --arg now "$now" \
        '.completed_at = $now | .progress = 100'
}

################################################################################
# GOAL METADATA
################################################################################

# Set goal priority
# Usage: set_goal_priority <goal_json> <priority>
# Returns: Updated goal JSON
set_goal_priority() {
    local goal_json="$1"
    local priority="$2"

    echo "$goal_json" | jq \
        --arg priority "$priority" \
        '.metadata.priority = $priority'
}

# Set goal effort estimate
# Usage: set_goal_effort_estimate <goal_json> <hours>
# Returns: Updated goal JSON
set_goal_effort_estimate() {
    local goal_json="$1"
    local hours="$2"

    echo "$goal_json" | jq \
        --argjson hours "$hours" \
        '.metadata.effort_estimate_hours = $hours'
}

# Add tag to goal
# Usage: add_goal_tag <goal_json> <tag>
# Returns: Updated goal JSON
add_goal_tag() {
    local goal_json="$1"
    local tag="$2"

    echo "$goal_json" | jq \
        --arg tag "$tag" \
        '.metadata.tags += [$tag] | .metadata.tags |= unique'
}

################################################################################
# LISASIMPSON: WORLD STATE, CONFIDENCE, AND VERIFICATION
################################################################################

# Set world state for a goal
# Usage: set_goal_world_state <goal_json> <world_state_json>
# Returns: Updated goal JSON
set_goal_world_state() {
    local goal_json="$1"
    local world_state="$2"

    echo "$goal_json" | jq \
        --argjson world_state "$world_state" \
        '.world_state = $world_state'
}

# Set confidence score for a goal
# Usage: set_goal_confidence <goal_json> <confidence_score>
# Returns: Updated goal JSON
set_goal_confidence() {
    local goal_json="$1"
    local confidence_score="$2"

    # Validate score is between 0.0 and 1.0
    if (( $(echo "$confidence_score < 0 || $confidence_score > 1" | bc -l) )); then
        echo "ERROR: Confidence score must be between 0.0 and 1.0" >&2
        return 1
    fi

    echo "$goal_json" | jq \
        --argjson confidence_score "$confidence_score" \
        '.confidence_score = $confidence_score'
}

# Set verification plan for a goal
# Usage: set_goal_verification_plan <goal_json> <verification_plan_json>
# Returns: Updated goal JSON
set_goal_verification_plan() {
    local goal_json="$1"
    local verification_plan="$2"

    echo "$goal_json" | jq \
        --argjson verification_plan "$verification_plan" \
        '.verification_plan = $verification_plan'
}

# Get world state from a goal
# Usage: get_goal_world_state <goal_json>
# Returns: World state JSON or empty object if not present
get_goal_world_state() {
    local goal_json="$1"

    echo "$goal_json" | jq '.world_state // {}'
}

# Get confidence score from a goal
# Usage: get_goal_confidence <goal_json>
# Returns: Confidence score (0.0-1.0) or null if not set
get_goal_confidence() {
    local goal_json="$1"

    echo "$goal_json" | jq '.confidence_score // null'
}

# Get verification plan from a goal
# Usage: get_goal_verification_plan <goal_json>
# Returns: Verification plan JSON or empty object if not present
get_goal_verification_plan() {
    local goal_json="$1"

    echo "$goal_json" | jq '.verification_plan // {}'
}

################################################################################
# GOAL DEPENDENCIES
################################################################################

# Add dependency relationship
# Usage: add_goal_dependency <goal_json> <parent_goal_id>
# Returns: Updated goal JSON
add_goal_dependency() {
    local goal_json="$1"
    local parent_goal_id="$2"

    echo "$goal_json" | jq \
        --arg parent "$parent_goal_id" \
        '.dependencies += [$parent] | .dependencies |= unique'
}

# Add sub-goal relationship
# Usage: add_sub_goal <goal_json> <sub_goal_id>
# Returns: Updated goal JSON
add_sub_goal() {
    local goal_json="$1"
    local sub_goal_id="$2"

    echo "$goal_json" | jq \
        --arg sub_goal "$sub_goal_id" \
        '.sub_goals += [$sub_goal] | .sub_goals |= unique'
}

################################################################################
# PRETTY PRINTING
################################################################################

# Pretty print goal summary
# Usage: print_goal_summary <goal_json>
print_goal_summary() {
    local goal_json="$1"

    local status
    status=$(get_goal_status "$goal_json")

    echo "=== Goal Summary ===" >&2
    echo "ID: $(echo "$goal_json" | jq -r '.goal_id')" >&2
    echo "Description: $(echo "$goal_json" | jq -r '.description')" >&2
    echo "Type: $(echo "$goal_json" | jq -r '.type')" >&2
    echo "Progress: $(echo "$status" | jq '.progress')%" >&2
    echo "Criteria: $(echo "$status" | jq '.completed')/$(echo "$status" | jq '.total_criteria') complete" >&2
    echo "Status: $(
        if echo "$goal_json" | jq -e '.completed_at != null' >/dev/null 2>&1; then
            echo "COMPLETED"
        elif is_goal_blocked "$goal_json" >/dev/null 2>&1; then
            echo "BLOCKED"
        else
            echo "IN PROGRESS"
        fi
    )" >&2
}

################################################################################
# EXPORTS
################################################################################

# If sourced, make functions available
export -f is_valid_goal_type
export -f validate_goal_structure
export -f create_goal
export -f calculate_goal_progress
export -f identify_blockers
export -f get_goal_status
export -f is_goal_complete
export -f is_goal_blocked
export -f update_criterion_status
export -f update_goal_progress
export -f complete_goal
export -f set_goal_priority
export -f set_goal_effort_estimate
export -f add_goal_tag
export -f set_goal_world_state
export -f set_goal_confidence
export -f set_goal_verification_plan
export -f get_goal_world_state
export -f get_goal_confidence
export -f get_goal_verification_plan
export -f add_goal_dependency
export -f add_sub_goal
export -f print_goal_summary

################################################################################
# SELF-TEST (run if executed directly)
################################################################################

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Running Goal Representation Self-Tests..."

    # Test 1: Create a goal
    echo "Test 1: Creating a goal..."
    test_goal=$(create_goal \
        "test_goal_1" \
        "creative_work" \
        "Test goal for validation" \
        '[
            {"criterion": "draft_complete", "description": "Write draft", "status": "not_started", "measurable": true},
            {"criterion": "review_complete", "description": "Get feedback", "status": "not_started", "measurable": true}
        ]')

    if validate_goal_structure "$test_goal"; then
        echo "✓ Goal structure is valid" >&2
    else
        echo "✗ Goal structure validation failed" >&2
        exit 1
    fi

    # Test 2: Calculate progress
    echo "Test 2: Calculating progress (should be 0%)..."
    progress=$(calculate_goal_progress "$test_goal")
    if [ "$progress" -eq 0 ]; then
        echo "✓ Initial progress is 0%" >&2
    else
        echo "✗ Initial progress should be 0%, got $progress" >&2
        exit 1
    fi

    # Test 3: Update criterion
    echo "Test 3: Updating criterion status..."
    test_goal=$(update_criterion_status "$test_goal" "draft_complete" "completed" "file:///path/to/draft.txt")
    progress=$(calculate_goal_progress "$test_goal")
    if [ "$progress" -eq 50 ]; then
        echo "✓ Progress updated to 50%" >&2
    else
        echo "✗ Progress should be 50%, got $progress" >&2
        exit 1
    fi

    # Test 4: Get status
    echo "Test 4: Getting goal status..."
    status=$(get_goal_status "$test_goal")
    echo "$status" | jq . >&2

    echo "" >&2
    echo "✓ All Goal Representation self-tests passed!" >&2
fi
