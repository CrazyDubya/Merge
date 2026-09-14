#!/bin/bash
#
# Dynamic Reflection Weight Library
# Adjusts reflection probability based on queue status
#
# Purpose:
#   When tasks are pending: reflection weight = 1% (focus on work)
#   When idle (no tasks): reflection weight = 10% (deeper thinking)
#
# This prevents the daemon from wasting time on reflection when
# there's real work to do, while allowing contemplation when idle.
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
TASKS_DIR="${DAEMON_ROOT}/tasks"
TASKS_FILE="${TASKS_DIR}/queue.md"

# ============================================================================
# Core Functions
# ============================================================================

# Count pending tasks (incomplete tasks, excluding in-progress)
count_pending_tasks() {
    if [ ! -f "$TASKS_FILE" ]; then
        echo "0"
        return 0
    fi

    # Count lines with [ ] (not started) and [x] is excluded
    # We count pending as: not started + in-progress (both are blocking work)
    local pending=0
    pending=$(grep -c "^- \[\\( \\|~\\)\\]" "$TASKS_FILE" 2>/dev/null || echo "0")

    echo "$pending"
}

# Check if daemon has any pending work
has_pending_work() {
    local pending=$(count_pending_tasks)
    [ "$pending" -gt 0 ]
    return $?
}

# Calculate dynamic reflection weight
calculate_reflection_weight() {
    local pending=$(count_pending_tasks)

    if [ "$pending" -gt 0 ]; then
        # Tasks pending: use 1% weight (minimal reflection)
        # This is calculated as: 1/100 = 0.01 in probability terms
        echo "0.01"
    else
        # No tasks: use 10% weight (deeper thinking)
        # This is calculated as: 10/100 = 0.10 in probability terms
        echo "0.10"
    fi
}

# Get human-readable reflection weight percentage
get_reflection_weight_percent() {
    local weight=$(calculate_reflection_weight)

    if [ "$weight" = "0.01" ]; then
        echo "1%"
    else
        echo "10%"
    fi
}

# Log reflection weight decision with context
log_reflection_decision() {
    local pending=$(count_pending_tasks)
    local weight=$(calculate_reflection_weight)
    local percent=$(get_reflection_weight_percent)

    if declare -f log >/dev/null 2>&1; then
        if [ "$pending" -gt 0 ]; then
            log "DEBUG" "Reflection weight: $percent (tasks pending: $pending, focus on work)"
        else
            log "DEBUG" "Reflection weight: $percent (no tasks, allow deeper thinking)"
        fi
    fi
}

# ============================================================================
# Integration with Action Selection
# ============================================================================

# Apply dynamic weight to action probabilities
# Usage: apply_dynamic_reflection_weight <current_weights_json>
# Returns: modified JSON with adjusted reflection weight
apply_dynamic_reflection_weight() {
    local weights_json="$1"
    local reflection_weight=$(calculate_reflection_weight)
    local pending=$(count_pending_tasks)

    # Use jq to update reflection weight in the weights JSON
    local temp_file=$(mktemp)
    trap "rm -f '$temp_file'" RETURN

    jq --arg refl_weight "$reflection_weight" \
       --argjson pending "$pending" \
       '.reflection = ($refl_weight | tonumber) |
        .pending_tasks = $pending' \
       <<< "$weights_json" > "$temp_file"

    cat "$temp_file"
}

# ============================================================================
# Debugging & Monitoring
# ============================================================================

# Show current reflection weight status
show_reflection_status() {
    local pending=$(count_pending_tasks)
    local weight=$(calculate_reflection_weight)
    local percent=$(get_reflection_weight_percent)

    echo "Reflection Weight Status:"
    echo "========================="
    echo "Pending Tasks: $pending"
    echo "Reflection Weight: $percent"
    echo "Weight Value: $weight"
    echo ""

    if [ "$pending" -gt 0 ]; then
        echo "Mode: TASK-FOCUSED"
        echo "Explanation: With pending work, daemon minimizes reflection (1%)"
        echo "             to focus on task completion."
    else
        echo "Mode: CONTEMPLATION"
        echo "Explanation: No pending work, daemon enables deeper reflection (10%)"
        echo "             for thoughtful analysis and planning."
    fi
}

# Export functions for use in daemon.sh
export -f count_pending_tasks
export -f has_pending_work
export -f calculate_reflection_weight
export -f get_reflection_weight_percent
export -f log_reflection_decision
export -f apply_dynamic_reflection_weight
export -f show_reflection_status
