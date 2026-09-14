#!/bin/bash
#
# Task Recovery Orchestrator
# Manages retries, exponential backoff, and task quarantine for failed tasks
#
# Usage:
#   source "${DAEMON_ROOT}/lib/task-recovery.sh"
#   if should_retry_task "$task"; then
#       retry_task_with_backoff "$task"
#   fi
#

set -euo pipefail

# Initialize retry tracking files if needed
init_retry_tracking() {
    mkdir -p "$DAEMON_ROOT/metrics"
    touch "$DAEMON_ROOT/metrics/task-retries.log"
    touch "$DAEMON_ROOT/tasks/retry-queue.txt"
}

# Get number of retry attempts for a task
get_retry_count() {
    local task="$1"
    local clean_task=$(echo "$task" | sed 's/^\[.*\]\s*//' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')

    grep "^$clean_task" "$DAEMON_ROOT/metrics/task-retries.log" 2>/dev/null | wc -l
}

# Record a retry attempt
record_retry() {
    local task="$1"
    local attempt="$2"
    local reason="$3"
    local clean_task=$(echo "$task" | sed 's/^\[.*\]\s*//' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')

    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)|$clean_task|attempt:$attempt|reason:$reason" >> \
        "$DAEMON_ROOT/metrics/task-retries.log"

    log "INFO" "Recorded retry attempt #$attempt for: $task (reason: $reason)"
}

# Check if task should be retried based on attempt count
should_retry_task() {
    local task="$1"
    local max_retries=5
    local retry_count=$(get_retry_count "$task")

    if [ $retry_count -ge $max_retries ]; then
        log "ERROR" "Task exceeded max retries ($max_retries): $task"
        quarantine_task "$task" "Max retries exceeded ($max_retries)"
        return 1
    fi

    return 0
}

# Schedule task for retry with exponential backoff
retry_task_with_backoff() {
    local task="$1"
    local retry_count=$(get_retry_count "$task")

    # Exponential backoff: 60s, 120s, 240s, 480s, 900s (1, 2, 4, 8, 15 minutes)
    local backoff_delays=(60 120 240 480 900)
    local delay=${backoff_delays[$retry_count]:-900}

    # Calculate scheduled retry time
    local scheduled_time=$(date -d "+${delay} seconds" +%s)

    # Add task to delayed retry queue
    echo "$scheduled_time|$task" >> "$DAEMON_ROOT/tasks/retry-queue.txt"

    log "INFO" "Scheduled retry #$((retry_count + 1)) in ${delay}s for: $task"
    record_retry "$task" "$((retry_count + 1))" "Backoff ${delay}s scheduled"

    # Also alert if this is a retry
    if [ $retry_count -gt 0 ]; then
        log_timeline "task_retry_scheduled" "system" "Task: $task, Attempt: $((retry_count + 1)), Delay: ${delay}s"
    fi
}

# Move task to quarantine after max retries exceeded
quarantine_task() {
    local task="$1"
    local reason="$2"
    local retry_count=$(get_retry_count "$task")

    mkdir -p "$DAEMON_ROOT/tasks/quarantine"

    # Create quarantine record
    local quarantine_file="$DAEMON_ROOT/tasks/quarantine/$(date +%Y%m%d-%H%M%S).md"

    cat > "$quarantine_file" <<EOF
# Quarantined Task

**Timestamp**: $(date -u +%Y-%m-%dT%H:%M:%SZ)
**Reason**: $reason
**Task**: $task
**Retry Attempts**: $retry_count

## Details

Task failed repeatedly and exceeded maximum retry limit ($retry_count attempts).
A human should review this task to determine if it's valid or should be deleted.

## Resolution Steps

1. Review task: is it still needed?
2. Debug why it's failing (check logs)
3. Either:
   - Fix the underlying issue and re-add to queue
   - Delete task if no longer relevant
   - Contact human if task is critical

## Logs

Last 20 lines of activity.log related to this task:
\`\`\`
$(grep "$task" "$DAEMON_ROOT/logs/activity.log" 2>/dev/null | tail -20 || echo "No logs found")
\`\`\`
EOF

    log "ERROR" "Task quarantined: $task"
    log "ERROR" "Quarantine file: $quarantine_file"

    # Remove task from main queue
    remove_task_from_queue "$task"

    # Remove from retry queue if present
    if [ -f "$DAEMON_ROOT/tasks/retry-queue.txt" ]; then
        grep -v "$task" "$DAEMON_ROOT/tasks/retry-queue.txt" > /tmp/retry-queue.txt
        mv /tmp/retry-queue.txt "$DAEMON_ROOT/tasks/retry-queue.txt"
    fi

    # Log timeline event
    log_timeline "task_quarantined" "system" "Task: $task, Reason: $reason"

    # Send alert to human (high priority)
    send_human_alert "Task Quarantined: Too Many Failures" \
        "Task has been quarantined after $retry_count failed retry attempts.\n\nTask: $task\n\nReason: $reason\n\nSee: $quarantine_file" \
        "high"
}

# Check if any delayed retries are ready and re-add them to queue
process_retry_queue() {
    local retry_queue="$DAEMON_ROOT/tasks/retry-queue.txt"

    if [ ! -f "$retry_queue" ]; then
        return 0
    fi

    local now=$(date +%s)
    local temp_queue=$(mktemp)
    trap "rm -f '$temp_queue'" RETURN
    local retries_processed=0

    while IFS='|' read -r scheduled_time task; do
        # Skip empty lines
        if [ -z "$scheduled_time" ] || [ -z "$task" ]; then
            continue
        fi

        if [ "$now" -ge "$scheduled_time" ]; then
            log "INFO" "Retry time reached for task: $task"

            # Re-add to main queue
            echo "- [ ] $task" >> "$DAEMON_ROOT/tasks/queue.md"

            log "INFO" "Task re-added to queue for retry: $task"
            retries_processed=$((retries_processed + 1))

            # Log timeline event
            log_timeline "task_retry_ready" "system" "Task: $task"
        else
            # Keep this retry in queue (not ready yet)
            echo "$scheduled_time|$task" >> "$temp_queue"
        fi
    done < "$retry_queue"

    # Update retry queue with only future retries
    if [ -f "$temp_queue" ]; then
        mv "$temp_queue" "$retry_queue"
    else
        rm -f "$retry_queue"
    fi

    if [ $retries_processed -gt 0 ]; then
        log "INFO" "Processed $retries_processed ready retries from queue"
    fi
}

# Helper function to remove task from main queue
remove_task_from_queue() {
    local task="$1"
    local queue="$DAEMON_ROOT/tasks/queue.md"

    if [ ! -f "$queue" ]; then
        return 0
    fi

    # Escape special regex characters in task
    local escaped_task=$(printf '%s\n' "$task" | sed 's/[[\.*^$/]/\\&/g')

    # Remove task from queue (handles both checked and unchecked)
    sed -i "/^\- \[\( \|~\|x\|X\)\] .*$escaped_task/d" "$queue"

    log "INFO" "Task removed from queue: $task"
}

# Get retry count for a task
get_task_retry_stats() {
    local task="$1"
    local clean_task=$(echo "$task" | sed 's/^\[.*\]\s*//' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')

    local attempts=$(grep "^$clean_task" "$DAEMON_ROOT/metrics/task-retries.log" 2>/dev/null | wc -l)
    local last_attempt=$(grep "^$clean_task" "$DAEMON_ROOT/metrics/task-retries.log" 2>/dev/null | tail -1)

    echo "attempts=$attempts"
    if [ -n "$last_attempt" ]; then
        echo "last_attempt=$last_attempt"
    fi
}

# Clear retry history for a task (used after successful completion)
clear_retry_history() {
    local task="$1"
    local clean_task=$(echo "$task" | sed 's/^\[.*\]\s*//' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')

    grep -v "^$clean_task" "$DAEMON_ROOT/metrics/task-retries.log" > /tmp/task-retries.log
    mv /tmp/task-retries.log "$DAEMON_ROOT/metrics/task-retries.log"

    log "INFO" "Retry history cleared for: $task"
}

# Helper to send human alerts (creates inbox message for human review)
send_human_alert() {
    local subject="$1"
    local message="$2"
    local priority="${3:-medium}"

    # Create inbox message for human review
    local inbox_dir="${DAEMON_ROOT}/inbox/human/unread"
    mkdir -p "$inbox_dir"

    local msg_file="$inbox_dir/alert-$(date +%Y%m%d-%H%M%S).md"

    cat > "$msg_file" <<EOF
---
from: task-recovery-system
timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
priority: $priority
---

# $subject

$message
EOF

    # Note: only call log if it's defined
    if declare -f log > /dev/null 2>&1; then
        log "INFO" "Alert sent to human: $subject"
    fi
}

# Initialize on source
init_retry_tracking

export -f init_retry_tracking
export -f get_retry_count
export -f record_retry
export -f should_retry_task
export -f retry_task_with_backoff
export -f quarantine_task
export -f process_retry_queue
export -f remove_task_from_queue
export -f get_task_retry_stats
export -f clear_retry_history
export -f send_human_alert
