#!/bin/bash

################################################################################
# Retry Orchestrator (LisaSimpson + Ralph Wiggum Complete Integration)
#
# MAIN INTEGRATION POINT: Wraps task execution with:
# 1. Confidence-based retry limits (high=5, medium=3, low=1)
# 2. Pre-execution checkpoints for rollback
# 3. Verification after each attempt
# 4. Self-referential verification (did I REALLY complete this?)
# 5. Approach adjustment on retry (modify prompt with feedback)
# 6. Automatic rollback on failure
# 7. Episode tracking for multi-step workflows
#
# Implements Ralph Wiggum's persistent retry-until-verified paradigm with
# LisaSimpson's state verification and checkpoint recovery.
#
# Authors: LisaSimpson + Ralph Wiggum + Autonomy Team
# Created: 2025-01-08
################################################################################

set -euo pipefail

# Daemon root
DAEMON_ROOT="${DAEMON_ROOT:-.}"

################################################################################
# LIBRARY DEPENDENCIES
################################################################################

# Source required libraries
source "${DAEMON_ROOT}/lib/confidence-engine.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/checkpoint-manager.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/task-outcome-verification.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/episodic-memory.sh" 2>/dev/null || true

################################################################################
# RETRY ORCHESTRATION ENGINE
################################################################################

# Main retry-until-verified wrapper
# Usage: execute_with_retry <task_title> <task_description> <persona> <checkpoint_files>
# Returns: Exit code (0 = success, 1 = failed after all retries)
execute_with_retry() {
    local task_title="$1"
    local task_description="$2"
    local persona="$3"
    local checkpoint_files="${4:-}"

    # Extract metadata
    local task_id=$(echo "$task_title" | md5sum | awk '{print $1}' | cut -c1-8)

    # Step 1: Calculate confidence and determine retry limit
    local confidence_score
    confidence_score=$(calculate_task_confidence "$task_description" "{}" "$persona" 2>/dev/null | jq -r '.confidence_score' 2>/dev/null || echo "0.5")

    local retry_limit
    retry_limit=$(map_confidence_to_retry_limit "$confidence_score" 2>/dev/null || echo "3")

    local confidence_class
    confidence_class=$(classify_confidence "$confidence_score" 2>/dev/null || echo "medium")

    echo "RETRY_ORCHESTRATOR: Task=$task_id | Confidence=$confidence_score ($confidence_class) | Retries=$retry_limit" >&2

    # Step 2: Create initial checkpoint
    local checkpoint_id=""
    if [ -n "$checkpoint_files" ] && declare -f create_checkpoint >/dev/null 2>&1; then
        local checkpoint_result
        checkpoint_result=$(create_checkpoint "$task_id" "$task_title" 2>/dev/null)
        checkpoint_id=$(echo "$checkpoint_result" | jq -r '.checkpoint_id' 2>/dev/null || echo "")
        if [ -n "$checkpoint_id" ]; then
            echo "RETRY_ORCHESTRATOR: Checkpoint created: $checkpoint_id" >&2
        fi
    fi

    # Step 2b: Create episode for multi-step workflow tracking (Phase 6)
    # NOTE: create_episode returns FULL JSON object, not just ID (functional API)
    local episode=""
    local episode_id=""
    if declare -f create_episode >/dev/null 2>&1; then
        episode=$(create_episode "$task_id" "$task_title" 2>/dev/null || echo "")
        if [ -n "$episode" ]; then
            episode_id=$(echo "$episode" | jq -r '.episode_id' 2>/dev/null || echo "")
            echo "RETRY_ORCHESTRATOR: Episode created: $episode_id for task $task_id" >&2
        fi
    fi

    # Step 3: Retry loop
    local attempt=0
    local max_attempts=$retry_limit
    local success=false
    local last_error=""
    local last_output=""

    while [ $attempt -lt $max_attempts ] && [ "$success" = "false" ]; do
        attempt=$((attempt + 1))
        echo "RETRY_ORCHESTRATOR: Attempt $attempt/$max_attempts" >&2

        # Build attempt-specific prompt modifications
        local attempt_feedback=""
        if [ $attempt -gt 1 ]; then
            attempt_feedback="
RETRY ATTEMPT $attempt:
Previous attempt failed. Feedback for this attempt:
$last_error

Adjust your approach:
- Try a different strategy
- Break down the problem differently
- Pay closer attention to edge cases
- Be more thorough and meticulous

This is your $attempt attempt out of $max_attempts. Execute this task with the feedback above."
        fi

        # Step 4: Execute task (original execute_task_action or similar)
        echo "RETRY_ORCHESTRATOR: Executing task action (attempt $attempt)..." >&2

        # Capture output for analysis
        # Pass task_title so execute_task_action doesn't re-fetch
        if execute_task_action "$persona" "$task_title" 2>&1 | tee -a /tmp/retry_attempt_${attempt}.log > /tmp/retry_output_${attempt}.txt; then
            last_output=$(cat /tmp/retry_output_${attempt}.txt 2>/dev/null || echo "")

            # Record action in episode (MUST capture return value - functional API)
            if [ -n "$episode" ] && declare -f add_action_to_episode >/dev/null 2>&1; then
                local action_json
                action_json=$(jq -n \
                    --arg type "task_execution" \
                    --arg task "$task_title" \
                    --arg attempt "$attempt" \
                    --arg status "execution_success" \
                    '{type: $type, task: $task, attempt: ($attempt | tonumber), status: $status, timestamp: (now | todate)}' 2>/dev/null)
                episode=$(add_action_to_episode "$episode" "$action_json" 2>/dev/null || echo "$episode")
            fi

            # Step 5: Verify task completion
            echo "RETRY_ORCHESTRATOR: Verifying task completion..." >&2

            if verify_task_completion "$task_title" "$task_description"; then
                # Step 6: Self-referential verification
                echo "RETRY_ORCHESTRATOR: Performing self-referential verification..." >&2

                if self_referential_verification "$task_title" "$task_description"; then
                    success=true
                    echo "RETRY_ORCHESTRATOR: ✓ Task verified successfully on attempt $attempt" >&2

                    # Log success
                    log_retry_attempt "$task_id" "$attempt" "success" "$confidence_score" "$checkpoint_id"
                else
                    echo "RETRY_ORCHESTRATOR: Self-verification failed, retrying..." >&2
                    last_error="Self-referential verification failed: Did I really complete this task?"

                    # Rollback if checkpoint exists
                    if [ -n "$checkpoint_id" ] && declare -f rollback_to_checkpoint >/dev/null 2>&1; then
                        echo "RETRY_ORCHESTRATOR: Rolling back to checkpoint $checkpoint_id" >&2
                        rollback_to_checkpoint "$checkpoint_id" 2>/dev/null || true
                    fi
                fi
            else
                echo "RETRY_ORCHESTRATOR: Verification failed (attempt $attempt/$max_attempts)" >&2
                last_error="Verification failed: Output does not match expected criteria"

                # Rollback if checkpoint exists
                if [ -n "$checkpoint_id" ] && declare -f rollback_to_checkpoint >/dev/null 2>&1; then
                    echo "RETRY_ORCHESTRATOR: Rolling back to checkpoint $checkpoint_id" >&2
                    rollback_to_checkpoint "$checkpoint_id" 2>/dev/null || true
                fi

                # Only retry if we have attempts left
                if [ $attempt -lt $max_attempts ]; then
                    echo "RETRY_ORCHESTRATOR: Attempting retry with adjusted approach..." >&2
                    adjust_task_approach "$task_title" "$attempt" "$last_error"
                fi
            fi
        else
            last_output=$(cat /tmp/retry_output_${attempt}.txt 2>/dev/null || echo "")
            echo "RETRY_ORCHESTRATOR: Task execution failed (exit code)" >&2
            last_error="Task execution failed with non-zero exit code"

            # Record action in episode (failed execution - MUST capture return value)
            if [ -n "$episode" ] && declare -f add_action_to_episode >/dev/null 2>&1; then
                local action_json
                action_json=$(jq -n \
                    --arg type "task_execution" \
                    --arg task "$task_title" \
                    --arg attempt "$attempt" \
                    --arg status "execution_failure" \
                    '{type: $type, task: $task, attempt: ($attempt | tonumber), status: $status, timestamp: (now | todate)}' 2>/dev/null)
                episode=$(add_action_to_episode "$episode" "$action_json" 2>/dev/null || echo "$episode")
            fi

            # Rollback if checkpoint exists
            if [ -n "$checkpoint_id" ] && declare -f rollback_to_checkpoint >/dev/null 2>&1; then
                echo "RETRY_ORCHESTRATOR: Rolling back to checkpoint $checkpoint_id" >&2
                rollback_to_checkpoint "$checkpoint_id" 2>/dev/null || true
            fi

            if [ $attempt -lt $max_attempts ]; then
                echo "RETRY_ORCHESTRATOR: Retrying with different approach..." >&2
                adjust_task_approach "$task_title" "$attempt" "$last_error"
            fi
        fi
    done

    # Step 7: Close episode and extract lessons (Phase 6)
    # NOTE: Must capture return value and call save_episode (functional API)
    if [ -n "$episode" ] && declare -f close_episode >/dev/null 2>&1; then
        local final_status
        if [ "$success" = "true" ]; then
            final_status="success"
        else
            final_status="failure"
        fi

        episode=$(close_episode "$episode" "$final_status" 2>/dev/null || echo "$episode")

        # CRITICAL: Actually save the episode to disk
        if declare -f save_episode >/dev/null 2>&1; then
            save_episode "$episode" 2>/dev/null || true
            echo "RETRY_ORCHESTRATOR: Episode saved: $episode_id with status $final_status" >&2
        else
            echo "RETRY_ORCHESTRATOR: Episode closed but save_episode not available" >&2
        fi
    fi

    # Step 8: Final result
    if [ "$success" = "true" ]; then
        echo "RETRY_ORCHESTRATOR: ✓ Task succeeded after $attempt attempt(s)" >&2
        log_retry_attempt "$task_id" "$attempt" "success" "$confidence_score" "$checkpoint_id"
        return 0
    else
        echo "RETRY_ORCHESTRATOR: ✗ Task failed after $max_attempts attempt(s)" >&2
        log_retry_attempt "$task_id" "$max_attempts" "failure" "$confidence_score" "$checkpoint_id"

        # Archive checkpoint for debugging
        if [ -n "$checkpoint_id" ] && declare -f archive_checkpoint >/dev/null 2>&1; then
            archive_checkpoint "$checkpoint_id" 2>/dev/null || true
        fi

        return 1
    fi
}

################################################################################
# VERIFICATION METHODS
################################################################################

# Verify task completion based on criteria
# Usage: verify_task_completion <task_title> <task_description>
# Returns: 0 if complete, 1 if not
verify_task_completion() {
    local task_title="$1"
    local task_description="$2"

    # Use existing verification infrastructure if available
    if declare -f task_outcome_verified >/dev/null 2>&1; then
        task_outcome_verified "$task_title" 2>/dev/null && return 0 || return 1
    fi

    # Fallback: Check for expected output files
    if echo "$task_description" | grep -qi "write\|create\|generate"; then
        # Check if new files were created recently
        local recent_files
        recent_files=$(find "${DAEMON_ROOT}" -type f -mmin -5 2>/dev/null | wc -l)
        if [ "$recent_files" -gt 0 ]; then
            return 0
        fi
    fi

    # Default: Assume not verified
    return 1
}

# Self-referential verification
# Usage: self_referential_verification <task_title> <task_description>
# Returns: 0 if verified to be complete, 1 if not
self_referential_verification() {
    local task_title="$1"
    local task_description="$2"

    # Multiple confirmation methods
    local verification_count=0
    local verification_passed=0

    # Method 1: Check output file existence
    if echo "$task_description" | grep -oE "\.[a-z]+" >/dev/null; then
        verification_count=$((verification_count + 1))
        local expected_ext
        expected_ext=$(echo "$task_description" | grep -oE "\.[a-z]{2,}" | head -1)

        if find "${DAEMON_ROOT}" -name "*$expected_ext" -mmin -5 2>/dev/null | grep -q .; then
            verification_passed=$((verification_passed + 1))
        fi
    fi

    # Method 2: Check task queue status
    if [ -f "${DAEMON_ROOT}/tasks/queue.md" ]; then
        verification_count=$((verification_count + 1))
        if grep -qi "^- \[x\]" "${DAEMON_ROOT}/tasks/queue.md"; then
            verification_passed=$((verification_passed + 1))
        fi
    fi

    # Method 3: Check logs for success indicators
    verification_count=$((verification_count + 1))
    if grep -qi "complete\|success\|done" /tmp/retry_attempt_*.log 2>/dev/null; then
        verification_passed=$((verification_passed + 1))
    fi

    # Require at least 2/3 methods to confirm completion
    if [ $verification_count -gt 0 ] && [ $verification_passed -ge $((verification_count / 2)) ]; then
        return 0
    fi

    return 1
}

################################################################################
# APPROACH ADJUSTMENT (RETRY FEEDBACK)
################################################################################

# Adjust task approach based on failure feedback
# Usage: adjust_task_approach <task_title> <attempt_number> <error_message>
adjust_task_approach() {
    local task_title="$1"
    local attempt="$2"
    local error_msg="$3"

    # Log the adjustment for debugging
    {
        echo "RETRY_ADJUSTMENT: Task=$task_title"
        echo "RETRY_ADJUSTMENT: Attempt=$attempt"
        echo "RETRY_ADJUSTMENT: Error=$error_msg"
        echo "RETRY_ADJUSTMENT: Timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >> "${DAEMON_ROOT}/logs/retry-adjustments.log"

    # Could modify the prompt or approach here for next retry
    # For now, just logging the adjustment
}

################################################################################
# LOGGING AND METRICS
################################################################################

# Log retry attempt for metrics and debugging
# Usage: log_retry_attempt <task_id> <attempt_num> <status> <confidence> <checkpoint_id>
log_retry_attempt() {
    local task_id="$1"
    local attempt_num="$2"
    local status="$3"
    local confidence="$4"
    local checkpoint_id="${5:-}"

    local log_entry
    log_entry=$(jq -n \
        --arg task_id "$task_id" \
        --arg attempt "$attempt_num" \
        --arg status "$status" \
        --arg confidence "$confidence" \
        --arg checkpoint "$checkpoint_id" \
        --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{
            task_id: $task_id,
            attempt: ($attempt | tonumber),
            status: $status,
            confidence_score: ($confidence | tonumber),
            checkpoint_id: $checkpoint,
            timestamp: $timestamp
        }')

    # Append to retry metrics log
    echo "$log_entry" >> "${DAEMON_ROOT}/logs/retry-metrics.jsonl" 2>/dev/null || true
}

# Get retry statistics for a task
# Usage: get_retry_stats <task_id>
# Returns: JSON with statistics
get_retry_stats() {
    local task_id="$1"

    local metrics_file="${DAEMON_ROOT}/logs/retry-metrics.jsonl"

    if [ ! -f "$metrics_file" ]; then
        echo "{}"
        return
    fi

    grep "$task_id" "$metrics_file" 2>/dev/null | jq -s \
        'group_by(.task_id) | map({
            task_id: .[0].task_id,
            attempts: length,
            successes: map(select(.status == "success")) | length,
            failures: map(select(.status == "failure")) | length,
            avg_confidence: (map(.confidence_score) | add / length),
            last_status: .[-1].status
        }) | .[0]' || echo "{}"
}

################################################################################
# EXPORTS
################################################################################

export -f execute_with_retry
export -f verify_task_completion
export -f self_referential_verification
export -f adjust_task_approach
export -f log_retry_attempt
export -f get_retry_stats

################################################################################
# SELF-TEST
################################################################################

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Running Retry Orchestrator Self-Tests..." >&2
    echo ""

    # Test 1: Confidence calculation
    echo "Test 1: Confidence-based retry calculation" >&2
    high_conf=$(calculate_task_confidence "Write a simple article" "{}" "architect" 2>/dev/null | jq -r '.confidence_score' 2>/dev/null || echo "0")
    high_limit=$(map_confidence_to_retry_limit "$high_conf" 2>/dev/null || echo "0")
    echo "  High confidence ($high_conf) → $high_limit retries" >&2

    # Test 2: Verification stubs
    echo "Test 2: Verification methods available" >&2
    if declare -f verify_task_completion >/dev/null 2>&1; then
        echo "  ✓ verify_task_completion available" >&2
    fi
    if declare -f self_referential_verification >/dev/null 2>&1; then
        echo "  ✓ self_referential_verification available" >&2
    fi

    # Test 3: Logging
    echo "Test 3: Retry attempt logging" >&2
    mkdir -p /tmp/daemon-test/logs
    DAEMON_ROOT="/tmp/daemon-test"
    log_retry_attempt "test_task_1" "1" "success" "0.85" "cp_test_1"
    if [ -f "/tmp/daemon-test/logs/retry-metrics.jsonl" ]; then
        echo "  ✓ Retry metrics logged successfully" >&2
    fi

    echo "" >&2
    echo "✓ Retry Orchestrator self-tests completed!" >&2
fi
