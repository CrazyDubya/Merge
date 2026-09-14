#!/bin/bash

################################################################################
# Goal Evaluation Engine
#
# Analyzes current system state and updates goal progress based on actual evidence.
# Compares current state vs. goal criteria to determine:
# - Progress towards each criterion
# - Blockers preventing completion
# - Next steps needed
#
# Authors: Autonomy Rebuild Team
# Created: 2025-12-23
################################################################################

set -euo pipefail

# Daemon root
DAEMON_ROOT="${DAEMON_ROOT:-.}"

# Source libraries
source "${DAEMON_ROOT}/lib/goal-representation.sh"
source "${DAEMON_ROOT}/lib/goal-state-management.sh"

################################################################################
# EVIDENCE CHECKERS
################################################################################

# Check if file exists (evidence format: "file:///path/to/file")
check_file_evidence() {
    local evidence="$1"

    # Parse evidence format: file:///path/to/file
    if [[ "$evidence" =~ ^file://(.+)$ ]]; then
        local filepath="${BASH_REMATCH[1]}"
        if [ -f "$filepath" ]; then
            return 0
        else
            return 1
        fi
    fi

    return 1
}

# Check file modification time (evidence format: "file_newer_than:///path/to/file,days")
check_file_recency() {
    local evidence="$1"

    # Parse: file_newer_than:///path/to/file,days
    if [[ "$evidence" =~ ^file_newer_than://(.+),([0-9]+)$ ]]; then
        local filepath="${BASH_REMATCH[1]}"
        local days="${BASH_REMATCH[2]}"

        if [ -f "$filepath" ]; then
            local file_age
            file_age=$(find "$filepath" -type f -mtime -"$days" 2>/dev/null | wc -l)
            if [ "$file_age" -gt 0 ]; then
                return 0
            fi
        fi
    fi

    return 1
}

# Check metric value (evidence format: "metric://metric_name,operator,value")
check_metric() {
    local evidence="$1"

    # Parse: metric://success_rate,>,0.8
    if [[ "$evidence" =~ ^metric://([^,]+),([<>=!]+),(.+)$ ]]; then
        local metric_name="${BASH_REMATCH[1]}"
        local operator="${BASH_REMATCH[2]}"
        local target_value="${BASH_REMATCH[3]}"

        # Get actual metric value from metrics/success-rates.json
        if [ -f "${DAEMON_ROOT}/metrics/success-rates.json" ]; then
            local actual_value
            actual_value=$(jq -r ".personas.${metric_name}.success_rate // empty" "${DAEMON_ROOT}/metrics/success-rates.json" 2>/dev/null || echo "")

            if [ -n "$actual_value" ]; then
                # Compare values
                if awk "BEGIN {exit !($actual_value $operator $target_value)}"; then
                    return 0
                fi
            fi
        fi
    fi

    return 1
}

# Verify evidence - check if criterion evidence is true
# Usage: verify_evidence <evidence>
# Returns 0 if evidence check passes, 1 if fails
verify_evidence() {
    local evidence="$1"

    if [[ "$evidence" =~ ^file:// ]]; then
        check_file_evidence "$evidence"
        return $?
    elif [[ "$evidence" =~ ^file_newer_than:// ]]; then
        check_file_recency "$evidence"
        return $?
    elif [[ "$evidence" =~ ^metric:// ]]; then
        check_metric "$evidence"
        return $?
    else
        # Unknown evidence type - consider it valid if string is non-empty
        if [ -n "$evidence" ]; then
            return 0
        else
            return 1
        fi
    fi
}

################################################################################
# CRITERION EVALUATION
################################################################################

# Evaluate a single criterion based on its evidence
# Usage: evaluate_criterion <goal_json> <criterion_id>
# Returns: JSON with evaluation result
evaluate_criterion() {
    local goal_json="$1"
    local criterion_id="$2"

    local criterion
    criterion=$(echo "$goal_json" | jq ".success_criteria[] | select(.criterion == \"$criterion_id\")")

    if [ -z "$criterion" ]; then
        echo "ERROR: Criterion $criterion_id not found" >&2
        return 1
    fi

    local description
    description=$(echo "$criterion" | jq -r '.description')

    local evidence
    evidence=$(echo "$criterion" | jq -r '.evidence // ""')

    local current_status
    current_status=$(echo "$criterion" | jq -r '.status')

    local evaluation_status="$current_status"
    local evidence_valid=false

    # If criterion has evidence, verify it
    if [ -n "$evidence" ]; then
        if verify_evidence "$evidence"; then
            evidence_valid=true
            evaluation_status="completed"
        else
            evaluation_status="not_started"
        fi
    fi

    jq -n \
        --arg criterion_id "$criterion_id" \
        --arg description "$description" \
        --arg current_status "$current_status" \
        --arg evaluated_status "$evaluation_status" \
        --arg evidence "$evidence" \
        --arg evidence_valid "$evidence_valid" \
        '{
            criterion_id: $criterion_id,
            description: $description,
            current_status: $current_status,
            evaluated_status: $evaluated_status,
            evidence: $evidence,
            evidence_valid: ($evidence_valid == "true"),
            status_changed: ($current_status != $evaluated_status)
        }'
}

# Evaluate all criteria for a goal
# Usage: evaluate_goal_criteria <goal_json>
# Returns: Array of criterion evaluations
evaluate_goal_criteria() {
    local goal_json="$1"

    local criteria_count
    criteria_count=$(echo "$goal_json" | jq '.success_criteria | length')

    local evaluations="[]"

    for ((i=0; i<criteria_count; i++)); do
        local criterion_id
        criterion_id=$(echo "$goal_json" | jq -r ".success_criteria[$i].criterion")

        local evaluation
        evaluation=$(evaluate_criterion "$goal_json" "$criterion_id")

        evaluations=$(echo "$evaluations" | jq --argjson eval "$evaluation" '. += [$eval]')
    done

    echo "$evaluations"
}

################################################################################
# GOAL EVALUATION & UPDATES
################################################################################

# Evaluate goal and update progress based on evidence
# Usage: evaluate_and_update_goal <goal_id>
# Returns: Updated goal JSON
evaluate_and_update_goal() {
    local goal_id="$1"

    local goal
    if ! goal=$(get_goal "$goal_id"); then
        echo "ERROR: Goal $goal_id not found" >&2
        return 1
    fi

    # Evaluate all criteria
    local evaluations
    evaluations=$(evaluate_goal_criteria "$goal")

    # Apply updates based on evaluations
    local updated_goal="$goal"
    local evaluation_count
    evaluation_count=$(echo "$evaluations" | jq 'length')

    for ((i=0; i<evaluation_count; i++)); do
        local eval
        eval=$(echo "$evaluations" | jq ".[$i]")

        local criterion_id
        criterion_id=$(echo "$eval" | jq -r '.criterion_id')

        local status_changed
        status_changed=$(echo "$eval" | jq -r '.status_changed')

        local evaluated_status
        evaluated_status=$(echo "$eval" | jq -r '.evaluated_status')

        # If status changed based on evidence, update it
        if [ "$status_changed" = "true" ]; then
            updated_goal=$(update_criterion_status "$updated_goal" "$criterion_id" "$evaluated_status" "Evaluated" "Evidence-based status update")
        fi
    done

    # Recalculate progress
    updated_goal=$(update_goal_progress "$updated_goal")

    # Persist updated goal
    update_goal "$updated_goal"

    echo "$updated_goal"
}

# Identify blockers for a goal
# Usage: identify_goal_blockers <goal_id>
# Returns: JSON with blocker analysis
identify_goal_blockers() {
    local goal_id="$1"

    local goal
    if ! goal=$(get_goal "$goal_id"); then
        echo "ERROR: Goal $goal_id not found" >&2
        return 1
    fi

    local blockers
    blockers=$(identify_blockers "$goal")

    # Analyze what would unblock each
    local blocker_analysis="[]"
    local blocker_count
    blocker_count=$(echo "$blockers" | jq 'length')

    for ((i=0; i<blocker_count; i++)); do
        local blocker
        blocker=$(echo "$blockers" | jq ".[$i]")

        local criterion
        criterion=$(echo "$blocker" | jq -r '.criterion')

        # Determine unblocking action
        local unblock_action
        case "$criterion" in
            *"feedback"*|*"reader"*)
                unblock_action="Create recruitment plan, send invitations, wait for responses"
                ;;
            *"publish"*|*"edit"*)
                unblock_action="Execute copy-editing, finalize formatting"
                ;;
            *"complete"*|*"done"*)
                unblock_action="Identify remaining work, create task list, execute"
                ;;
            *)
                unblock_action="Analyze specific blocker, determine requirements"
                ;;
        esac

        blocker_analysis=$(echo "$blocker_analysis" | jq \
            --arg criterion "$criterion" \
            --arg action "$unblock_action" \
            '. += [{criterion: $criterion, unblock_action: $action}]')
    done

    echo "$blocker_analysis"
}

################################################################################
# GOAL HEALTH CHECK
################################################################################

# Check goal health and generate status report
# Usage: assess_goal_health <goal_id>
# Returns: Health assessment JSON
assess_goal_health() {
    local goal_id="$1"

    local goal
    if ! goal=$(get_goal "$goal_id"); then
        echo "ERROR: Goal $goal_id not found" >&2
        return 1
    fi

    local progress
    progress=$(calculate_goal_progress "$goal")

    local is_blocked
    is_blocked=false
    if is_goal_blocked "$goal" >/dev/null 2>&1; then
        is_blocked=true
    fi

    local is_complete
    is_complete=false
    if is_goal_complete "$goal" >/dev/null 2>&1; then
        is_complete=true
    fi

    # Estimate time remaining based on progress rate
    local created_at
    created_at=$(echo "$goal" | jq -r '.created_at')

    local effort_spent
    effort_spent=$(echo "$goal" | jq -r '.metadata.effort_spent_hours // 0')

    local effort_estimate
    effort_estimate=$(echo "$goal" | jq -r '.metadata.effort_estimate_hours // 0')

    local status
    if [ "$is_complete" = "true" ]; then
        status="COMPLETE"
    elif [ "$is_blocked" = "true" ]; then
        status="BLOCKED"
    elif [ "$progress" -lt 25 ]; then
        status="STALLED"
    elif [ "$progress" -lt 50 ]; then
        status="SLOW"
    elif [ "$progress" -lt 75 ]; then
        status="ON_TRACK"
    else
        status="NEAR_COMPLETE"
    fi

    jq -n \
        --arg goal_id "$goal_id" \
        --argjson progress "$progress" \
        --arg status "$status" \
        --arg is_blocked "$is_blocked" \
        --arg is_complete "$is_complete" \
        --argjson effort_spent "$effort_spent" \
        --argjson effort_estimate "$effort_estimate" \
        '{
            goal_id: $goal_id,
            progress: $progress,
            status: $status,
            is_blocked: ($is_blocked == "true"),
            is_complete: ($is_complete == "true"),
            effort_spent_hours: $effort_spent,
            effort_estimate_hours: $effort_estimate,
            effort_remaining_hours: ($effort_estimate - $effort_spent),
            assessment_time: now | todate
        }'
}

################################################################################
# BULK EVALUATION
################################################################################

# Evaluate all active goals
# Updates progress based on current evidence
# Usage: evaluate_all_goals
evaluate_all_goals() {
    local active_goals
    active_goals=$(get_active_goals)

    local goal_count
    goal_count=$(echo "$active_goals" | jq 'length')

    echo "Evaluating $goal_count active goals..." >&2

    for ((i=0; i<goal_count; i++)); do
        local goal_id
        goal_id=$(echo "$active_goals" | jq -r ".[$i].goal_id")

        echo "Evaluating goal: $goal_id" >&2
        evaluate_and_update_goal "$goal_id" >/dev/null
    done

    echo "Goal evaluation complete" >&2
}

# Generate full status report for all goals
# Usage: generate_goal_status_report
# Returns: Markdown report
generate_goal_status_report() {
    local active_goals
    active_goals=$(get_active_goals)

    local report="# Goal Status Report\n\n"
    report+="Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")\n\n"

    local goal_count
    goal_count=$(echo "$active_goals" | jq 'length')

    if [ "$goal_count" -eq 0 ]; then
        report+="No active goals.\n"
    else
        for ((i=0; i<goal_count; i++)); do
            local goal_id
            goal_id=$(echo "$active_goals" | jq -r ".[$i].goal_id")

            local goal
            goal=$(get_goal "$goal_id")

            local health
            health=$(assess_goal_health "$goal_id")

            local status
            status=$(echo "$health" | jq -r '.status')

            local progress
            progress=$(echo "$health" | jq '.progress')

            local description
            description=$(echo "$goal" | jq -r '.description')

            report+="## $goal_id ($status)\n\n"
            report+="**Description**: $description\n"
            report+="**Progress**: $progress%\n"

            # List criteria
            report+="**Criteria**:\n"
            local criteria_count
            criteria_count=$(echo "$goal" | jq '.success_criteria | length')
            for ((j=0; j<criteria_count; j++)); do
                local criterion
                criterion=$(echo "$goal" | jq ".success_criteria[$j]")

                local criterion_desc
                criterion_desc=$(echo "$criterion" | jq -r '.description')

                local criterion_status
                criterion_status=$(echo "$criterion" | jq -r '.status')

                report+="- [ ] $criterion_desc ($criterion_status)\n"
            done

            report+="\n"
        done
    fi

    echo -e "$report"
}

################################################################################
# EXPORTS
################################################################################

export -f verify_evidence
export -f evaluate_criterion
export -f evaluate_goal_criteria
export -f evaluate_and_update_goal
export -f identify_goal_blockers
export -f assess_goal_health
export -f evaluate_all_goals
export -f generate_goal_status_report

################################################################################
# SELF-TEST
################################################################################

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Running Goal Evaluator Self-Tests..." >&2

    # Initialize state
    init_goal_state

    # Create test goal with file-based evidence
    test_goal=$(create_goal \
        "test_eval" \
        "creative_work" \
        "Test goal for evaluation" \
        '[
            {"criterion": "file_created", "description": "Create test file", "status": "not_started", "evidence": "file:///tmp/test-goal-file.txt", "measurable": true},
            {"criterion": "manual_check", "description": "Manual verification", "status": "not_started", "evidence": "", "measurable": false}
        ]')

    echo "Test 1: Adding goal..." >&2
    add_goal "$test_goal"

    # Create the file to make evidence valid
    touch /tmp/test-goal-file.txt

    echo "Test 2: Evaluating goal..." >&2
    evaluate_and_update_goal "test_eval" >/dev/null

    # Check if first criterion was auto-completed
    updated_goal=$(get_goal "test_eval")
    file_criterion_status=$(echo "$updated_goal" | jq -r '.success_criteria[] | select(.criterion == "file_created") | .status')

    if [ "$file_criterion_status" = "completed" ]; then
        echo "✓ Evidence-based criterion auto-completed" >&2
    else
        echo "✗ Evidence-based criterion should be completed, got: $file_criterion_status" >&2
        exit 1
    fi

    echo "Test 3: Assessing goal health..." >&2
    health=$(assess_goal_health "test_eval")
    status=$(echo "$health" | jq -r '.status')
    progress=$(echo "$health" | jq '.progress')

    if [ "$progress" -eq 50 ]; then
        echo "✓ Goal health assessment correct: $status, progress $progress%" >&2
    else
        echo "✗ Expected 50% progress, got $progress%" >&2
        exit 1
    fi

    echo "" >&2
    echo "✓ All Goal Evaluator self-tests passed!" >&2

    # Cleanup
    rm -f /tmp/test-goal-file.txt
    rm -rf ./state
fi
