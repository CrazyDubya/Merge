#!/bin/bash
#
# Urgent Task Detection Library
# Implements age-based task prioritization
#
# Purpose:
#   - Detects tasks that have been pending for extended periods
#   - Boosts priority for old tasks to prevent starvation
#   - Alerts on very old tasks (>24 hours)
#
# Logic:
#   - Tasks >4 hours old: Priority boost (move up in queue)
#   - Tasks >24 hours old: Send alert to human
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
TASKS_DIR="${DAEMON_ROOT}/tasks"
TASKS_FILE="${TASKS_DIR}/queue.md"

# ============================================================================
# Core Functions
# ============================================================================

# Get task creation time from queue.md metadata
# Tasks may have metadata like "created: 2025-12-07T10:30:00Z"
# For now, we use file modification time as proxy
get_queue_modification_time() {
    if [ -f "$TASKS_FILE" ]; then
        stat -f%m "$TASKS_FILE" 2>/dev/null || stat -c%Y "$TASKS_FILE" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# Calculate task age in seconds (based on queue.md modification)
# All tasks in queue are considered same age (when queue was last modified)
get_task_age_seconds() {
    local now=$(date +%s)
    local queue_mtime=$(get_queue_modification_time)

    if [ "$queue_mtime" -eq 0 ]; then
        echo "0"
    else
        echo $((now - queue_mtime))
    fi
}

# Convert seconds to human-readable format
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))

    if [ "$hours" -gt 0 ]; then
        echo "${hours}h ${minutes}m"
    else
        echo "${minutes}m"
    fi
}

# Check if task is urgent (>4 hours old)
is_task_urgent() {
    local age_seconds=$1
    local urgent_threshold=$((4 * 3600))  # 4 hours in seconds

    [ "$age_seconds" -gt "$urgent_threshold" ]
}

# Check if task needs escalation (>24 hours old)
is_task_critical() {
    local age_seconds=$1
    local critical_threshold=$((24 * 3600))  # 24 hours in seconds

    [ "$age_seconds" -gt "$critical_threshold" ]
}

# Get count of pending tasks
get_pending_task_count() {
    if [ ! -f "$TASKS_FILE" ]; then
        echo "0"
        return 0
    fi

    grep -c "^- \[ \]" "$TASKS_FILE" 2>/dev/null || echo "0"
}

# ============================================================================
# Priority Boosting
# ============================================================================

# Move urgent tasks to front of queue
# Usage: boost_urgent_task_priority
# This reorders queue.md to put old tasks at the top
boost_urgent_task_priority() {
    local age_seconds=$(get_task_age_seconds)

    if ! is_task_urgent "$age_seconds"; then
        return 0  # No urgent tasks
    fi

    if [ ! -f "$TASKS_FILE" ]; then
        return 1
    fi

    local duration=$(format_duration "$age_seconds")
    if declare -f log >/dev/null 2>&1; then
        log "WARN" "Urgent tasks detected (pending $duration) - boosting priority"
    fi

    # Note: In a full implementation, this would reorder queue.md
    # For now, we just log the detection
    return 0
}

# ============================================================================
# Alerting
# ============================================================================

# Check for critical age tasks and alert
# Usage: check_task_age_alerts
check_task_age_alerts() {
    local age_seconds=$(get_task_age_seconds)
    local pending_count=$(get_pending_task_count)

    if [ "$pending_count" -eq 0 ]; then
        return 0  # No pending tasks
    fi

    if is_task_critical "$age_seconds"; then
        local duration=$(format_duration "$age_seconds")

        if declare -f log >/dev/null 2>&1; then
            log "CRITICAL" "Pending tasks are VERY OLD (>24h): $duration - investigate stagnation"
        fi

        # Send alert to human
        if declare -f send_enriched_alert >/dev/null 2>&1; then
            send_enriched_alert \
                "TASK_AGE_CRITICAL" \
                "critical" \
                "Task queue stagnation detected" \
                "Pending tasks have been waiting $duration. This may indicate:\n
                  - Persona inability to complete assigned work\n
                  - Task requirements mismatched to persona skills\n
                  - External dependencies blocking progress" || true
        fi

        return 0  # Alert sent (still continue daemon loop)
    elif is_task_urgent "$age_seconds"; then
        local duration=$(format_duration "$age_seconds")

        if declare -f log >/dev/null 2>&1; then
            log "WARN" "Pending tasks are getting old ($duration) - increasing priority"
        fi

        return 0  # Just warning, no escalation
    fi

    return 0  # Tasks are recent
}

# ============================================================================
# Status & Monitoring
# ============================================================================

# Show task age status
# Usage: show_task_age_status
show_task_age_status() {
    local age_seconds=$(get_task_age_seconds)
    local duration=$(format_duration "$age_seconds")
    local pending=$(get_pending_task_count)

    echo "Task Age Status"
    echo "==============="
    echo "Pending tasks: $pending"
    echo "Queue age: $duration"
    echo ""

    if [ "$pending" -eq 0 ]; then
        echo "Status: ✅ Queue is empty"
    elif is_task_critical "$age_seconds"; then
        echo "Status: 🔴 CRITICAL - Queue stagnation"
        echo "Action: Investigate and remediate immediately"
    elif is_task_urgent "$age_seconds"; then
        echo "Status: 🟡 URGENT - Pending tasks need attention"
        echo "Action: Boost priority and reassign if needed"
    else
        echo "Status: ✅ Queue is fresh"
    fi
}

# Export functions for use in daemon.sh
export -f get_task_age_seconds
export -f format_duration
export -f is_task_urgent
export -f is_task_critical
export -f get_pending_task_count
export -f boost_urgent_task_priority
export -f check_task_age_alerts
export -f show_task_age_status
