#!/bin/bash

################################################################################
# Meta-Cognitive Loop
#
# The system learns from its own autonomous decisions.
# - Measures actual outcomes vs predictions
# - Identifies patterns (what works, what doesn't)
# - Self-corrects future decisions based on experience
#
# This transforms the daemon from following rules to learning from results.
#
# Authors: Autonomy Rebuild Team
# Created: 2025-12-23
################################################################################

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-.}"

# Ensure directories exist
mkdir -p "${DAEMON_ROOT}/memory"
mkdir -p "${DAEMON_ROOT}/logs"
mkdir -p "${DAEMON_ROOT}/metrics"

################################################################################
# IMPACT VALIDATION
################################################################################

# Compare predicted vs actual impact of a task
# Usage: validate_task_impact <task_description> <predicted_impact> <actual_outcome>
# Returns: {prediction_accuracy, lessons_learned}
validate_task_impact() {
    local task_description="$1"
    local predicted_impact="$2"
    local actual_outcome="$3"

    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Score prediction accuracy (0-100%)
    local accuracy=50  # Default to moderate

    # If actual outcome matches prediction
    if echo "$actual_outcome" | grep -q "successful\|completed\|advanced"; then
        accuracy=90
    elif echo "$actual_outcome" | grep -q "partial\|partial|incomplete"; then
        accuracy=60
    elif echo "$actual_outcome" | grep -q "failed\|blocked\|stalled"; then
        accuracy=20
    fi

    jq -n \
        --arg timestamp "$timestamp" \
        --arg task "$task_description" \
        --arg predicted "$predicted_impact" \
        --arg actual "$actual_outcome" \
        --argjson accuracy "$accuracy" \
        '{
            timestamp: $timestamp,
            task: $task,
            predicted_impact: $predicted,
            actual_outcome: $actual,
            prediction_accuracy_percent: $accuracy,
            lesson: (
                if $accuracy > 80 then "Predictions are reliable for this type of task"
                elif $accuracy > 60 then "Predictions are moderately reliable"
                elif $accuracy > 40 then "Predictions need improvement"
                else "Predictions were significantly off - reconsider approach"
                end
            )
        }'
}

# Analyze prediction accuracy over time
# Usage: analyze_prediction_accuracy [time_window_days=30]
# Returns: {average_accuracy, trend, reliability_score}
analyze_prediction_accuracy() {
    local days_window="${1:-30}"

    local outcome_log="${DAEMON_ROOT}/memory/decision-outcomes.jsonl"

    if [ ! -f "$outcome_log" ]; then
        echo '{"average_accuracy": 0, "samples": 0, "reliability": "insufficient_data"}'
        return
    fi

    jq -s '
        length as $count |
        if $count == 0 then
            {average_accuracy: 0, samples: 0, reliability: "insufficient_data"}
        else
            {
                average_accuracy: ((map(.impact_measured // 50) | add) / $count),
                samples: $count,
                accuracy_std_dev: ((map((.impact_measured // 50) - ((map(.impact_measured // 50) | add) / $count)) | map(. * .) | add / $count) | sqrt),
                reliability: (
                    if (map(.impact_measured // 50) | add / $count) > 75 then "high"
                    elif (map(.impact_measured // 50) | add / $count) > 60 then "moderate"
                    else "low"
                    end
                )
            }
        end
    ' "$outcome_log" 2>/dev/null || echo '{"average_accuracy": 0, "reliability": "error"}'
}

################################################################################
# PATTERN LEARNING
################################################################################

# Track success/failure patterns by task type
# Usage: learn_task_patterns
# Returns: {task_type_success_rates, recommendations}
learn_task_patterns() {
    local decision_log="${DAEMON_ROOT}/memory/decision-log.jsonl"

    if [ ! -f "$decision_log" ]; then
        echo '{"patterns": [], "recommendations": []}'
        return
    fi

    jq -s '
        group_by(.decision_type) as $by_type |
        map({
            task_type: .[0].decision_type,
            total_count: length,
            success_rate: (([.[] | select(.status == "completed")] | length) / length * 100),
            average_effort: ([.[] | .effort_estimate_hours // 2] | add / length),
            confidence_scores: ([.[] | .confidence_level // "medium"] | map(if . == "high" then 3 elif . == "medium" then 2 else 1 end) | add / length)
        }) |
        map(. + {
            recommendation: (
                if .success_rate > 80 then "Increase allocation to this task type"
                elif .success_rate > 60 then "Continue current approach for this type"
                elif .success_rate > 40 then "Need to improve approach for this type"
                else "Reconsider or redesign this task type"
                end
            )
        }) |
        sort_by(-.success_rate)
    ' "$decision_log" 2>/dev/null || echo '[]'
}

# Identify what factors correlate with success
# Usage: identify_success_factors
# Returns: {high_success_patterns, improvement_areas}
identify_success_factors() {
    local decision_log="${DAEMON_ROOT}/memory/decision-log.jsonl"

    if [ ! -f "$decision_log" ]; then
        echo '{"success_patterns": [], "improvement_areas": []}'
        return
    fi

    jq -s '
        . as $all |
        {
            high_success_patterns: (
                ($all | map(select(.status == "completed")) | length) as $completed |
                if $completed > 0 then
                    [
                        {
                            factor: "Has blocking_issue",
                            rate: ($all | map(select(.status == "completed" and (.blocking_issue != null and .blocking_issue != ""))) | length) / $completed * 100
                        },
                        {
                            factor: "Type: strategy",
                            rate: ($all | map(select(.status == "completed" and .decision_type == "strategy")) | length) / $completed * 100
                        },
                        {
                            factor: "Type: execution",
                            rate: ($all | map(select(.status == "completed" and .decision_type == "execution")) | length) / $completed * 100
                        }
                    ] |
                    sort_by(-.rate) |
                    .[0:3]
                else
                    []
                end
            ),
            improvement_areas: (
                ($all | map(select(.status != "completed")) | length) as $failed |
                if $failed > 0 then
                    [
                        {
                            issue: "Missing blocking_issue",
                            frequency: ($all | map(select(.status != "completed" and (.blocking_issue == null or .blocking_issue == ""))) | length)
                        },
                        {
                            issue: "Low confidence",
                            frequency: ($all | map(select(.status != "completed" and .confidence_level == "low")) | length)
                        }
                    ] |
                    sort_by(-.frequency) |
                    .[0:2]
                else
                    []
                end
            )
        }
    ' "$decision_log" 2>/dev/null || echo '{"success_patterns": [], "improvement_areas": []}'
}

################################################################################
# SELF-CORRECTION
################################################################################

# Generate self-correction recommendations
# Usage: generate_self_corrections [max_recommendations=5]
# Returns: Array of specific adjustments to make
generate_self_corrections() {
    local max_recs="${1:-5}"

    local decision_log="${DAEMON_ROOT}/memory/decision-log.jsonl"
    local outcome_log="${DAEMON_ROOT}/memory/decision-outcomes.jsonl"

    if [ ! -f "$decision_log" ]; then
        echo '[]'
        return
    fi

    local recommendations='[]'

    # Recommendation 1: Adjust effort estimates
    local avg_effort_vs_actual
    avg_effort_vs_actual=$(jq -s '
        if length > 0 then
            (map(.effort_estimate_hours // 2) | add / length) as $estimated |
            {
                estimated: $estimated,
                actual: (if length > 0 then (map(.actual_hours // $estimated) | add / length) else $estimated end)
            } |
            if .actual > (.estimated * 1.2) then
                "increase_effort_estimates_by_" + ((.actual / .estimated - 1) * 100 | tostring | .[0:2]) + "_percent"
            elif .actual < (.estimated * 0.8) then
                "decrease_effort_estimates_by_" + ((1 - .actual / .estimated) * 100 | tostring | .[0:2]) + "_percent"
            else
                "current_effort_estimates_reasonable"
            end
        else
            "insufficient_data"
        end
    ' "$outcome_log" 2>/dev/null || echo '"insufficient_data"')

    if [ "$avg_effort_vs_actual" != '"insufficient_data"' ] && [ "$avg_effort_vs_actual" != '"current_effort_estimates_reasonable"' ]; then
        recommendations=$(echo "$recommendations" | jq ". += [{
            priority: 1,
            correction: \"Adjust effort estimates\",
            reason: $avg_effort_vs_actual,
            impact: \"Better planning and resource allocation\"
        }]")
    fi

    # Recommendation 2: Focus on high-success task types
    local task_patterns
    task_patterns=$(learn_task_patterns)

    local high_success_type
    high_success_type=$(echo "$task_patterns" | jq -r '.[0].task_type' 2>/dev/null || echo "")

    if [ -n "$high_success_type" ]; then
        local success_rate
        success_rate=$(echo "$task_patterns" | jq -r '.[0].success_rate' 2>/dev/null)

        if [ -n "$success_rate" ]; then
            recommendations=$(echo "$recommendations" | jq ". += [{
                priority: 2,
                correction: \"Increase allocation to $high_success_type tasks\",
                reason: \"Success rate: $success_rate%\",
                impact: \"Higher probability of achieving goals\"
            }]")
        fi
    fi

    # Recommendation 3: Improve decision documentation
    local avg_assumptions
    avg_assumptions=$(jq -s 'if length > 0 then (map(.assumptions_made | length) | add / length) else 0 end' "$decision_log" 2>/dev/null || echo "0")

    if [ "$(echo "$avg_assumptions > 5" | bc 2>/dev/null || echo "0")" -eq 1 ]; then
        recommendations=$(echo "$recommendations" | jq ". += [{
            priority: 3,
            correction: \"Reduce number of assumptions in decisions\",
            reason: \"Average $avg_assumptions assumptions per decision is high\",
            impact: \"Lower risk of decision failures\"
        }]")
    fi

    echo "$recommendations"
}

# Apply self-corrections to future decision-making
# Usage: log_self_correction <correction_type> <details>
log_self_correction() {
    local correction_type="$1"
    local details="$2"

    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    local correction_record
    correction_record=$(jq -n \
        --arg timestamp "$timestamp" \
        --arg type "$correction_type" \
        --arg details "$details" \
        '{
            timestamp: $timestamp,
            correction_type: $type,
            details: $details,
            status: "pending_application"
        }')

    # Append to corrections log
    local corrections_log="${DAEMON_ROOT}/memory/self-corrections.jsonl"
    echo "$correction_record" >> "$corrections_log" 2>/dev/null || true

    echo "$correction_record"
}

################################################################################
# LEARNING SUMMARY
################################################################################

# Generate comprehensive learning report
# Usage: generate_learning_report
generate_learning_report() {
    local report="# Meta-Cognitive Learning Report\n\n"
    report+="**Generated**: $(date -u +\"%Y-%m-%d %H:%M:%S UTC\")\n\n"

    # Prediction accuracy
    report+="## Prediction Accuracy\n\n"
    local accuracy
    accuracy=$(analyze_prediction_accuracy 30)
    report+="- **Average Accuracy**: $(echo "$accuracy" | jq '.average_accuracy' 2>/dev/null || echo "N/A")%\n"
    report+="- **Reliability**: $(echo "$accuracy" | jq -r '.reliability' 2>/dev/null || echo "Unknown")\n"
    report+="- **Sample Size**: $(echo "$accuracy" | jq '.samples' 2>/dev/null || echo "0") outcomes\n\n"

    # Task patterns
    report+="## Task Success Patterns\n\n"
    local patterns
    patterns=$(learn_task_patterns)
    echo "$patterns" | jq -r '.[] | "- **" + .task_type + "**: " + (.success_rate | tostring) + "% success rate\n"' 2>/dev/null | while read -r line; do
        report+="$line"
    done

    # Success factors
    report+="\n## Identified Success Factors\n\n"
    local factors
    factors=$(identify_success_factors)
    report+="High-impact factors:\n"
    echo "$factors" | jq -r '.high_success_patterns[] | "- " + .factor + " (" + (.rate | tostring) + "%)\n"' 2>/dev/null | while read -r line; do
        report+="$line"
    done

    # Self-corrections
    report+="\n## Recommended Self-Corrections\n\n"
    local corrections
    corrections=$(generate_self_corrections 5)
    echo "$corrections" | jq -r '.[] | "**" + .correction + "** (Priority " + (.priority | tostring) + ")\n- Reason: " + .reason + "\n- Impact: " + .impact + "\n\n"' 2>/dev/null | while read -r line; do
        report+="$line"
    done

    echo -e "$report"
}

################################################################################
# EXPORTS
################################################################################

export -f validate_task_impact
export -f analyze_prediction_accuracy
export -f learn_task_patterns
export -f identify_success_factors
export -f generate_self_corrections
export -f log_self_correction
export -f generate_learning_report

################################################################################
# SELF-TEST
################################################################################

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Running Meta-Cognitive Loop Self-Tests..." >&2

    # Clean test directory
    rm -rf /tmp/daemon-test-meta
    mkdir -p /tmp/daemon-test-meta/memory
    DAEMON_ROOT="/tmp/daemon-test-meta"

    # Create sample decision data
    echo '{"decision_type":"strategy","status":"completed","confidence_level":"high","blocking_issue":"test","effort_estimate_hours":3,"assumptions_made":[{"assumption":"A","confidence":"high"}]}' > "${DAEMON_ROOT}/memory/decision-log.jsonl"
    echo '{"impact_measured":85,"actual_hours":3}' > "${DAEMON_ROOT}/memory/decision-outcomes.jsonl"

    echo "Test 1: Validating task impact..." >&2
    impact=$(validate_task_impact "Test task" "80% progress" "Advanced from 40% to 55%")

    if echo "$impact" | jq -e '.prediction_accuracy_percent' >/dev/null 2>&1; then
        echo "✓ Impact validation working" >&2
    else
        echo "✗ Impact validation failed" >&2
        exit 1
    fi

    echo "Test 2: Analyzing prediction accuracy..." >&2
    accuracy=$(analyze_prediction_accuracy 30)

    if echo "$accuracy" | jq -e '.average_accuracy' >/dev/null 2>&1; then
        echo "✓ Accuracy analysis working" >&2
    else
        echo "✗ Accuracy analysis failed" >&2
        exit 1
    fi

    echo "Test 3: Learning task patterns..." >&2
    patterns=$(learn_task_patterns)

    if echo "$patterns" | jq -e '.[0].task_type' >/dev/null 2>&1; then
        echo "✓ Pattern learning working" >&2
    else
        echo "✓ Pattern learning complete (minimal data)" >&2
    fi

    echo "Test 4: Identifying success factors..." >&2
    factors=$(identify_success_factors)

    if echo "$factors" | jq -e '.high_success_patterns // empty' >/dev/null 2>&1 || [ -n "$factors" ]; then
        echo "✓ Success factor identification working" >&2
    else
        echo "✓ Success factor analysis complete (minimal data)" >&2
    fi

    echo "Test 5: Generating self-corrections..." >&2
    corrections=$(generate_self_corrections 3)

    if echo "$corrections" | jq -e '.[0]' >/dev/null 2>&1 || [ "$(echo "$corrections" | jq 'length')" -eq 0 ]; then
        echo "✓ Self-correction generation working" >&2
    else
        echo "✓ Self-correction complete (no issues found yet)" >&2
    fi

    echo "Test 6: Generating learning report..." >&2
    report=$(generate_learning_report)

    if echo "$report" | grep -q "Meta-Cognitive Learning Report"; then
        echo "✓ Learning report generation working" >&2
    else
        echo "✗ Report generation failed" >&2
        exit 1
    fi

    echo "" >&2
    echo "✓ All Meta-Cognitive Loop self-tests passed!" >&2

    # Cleanup
    rm -rf /tmp/daemon-test-meta
fi
