#!/bin/bash
#
# Reflection Scheduler Library
# Differentiates between scheduled reflection and idle reflection
#
# Purpose:
#   - Scheduled reflection: Periodic deep thinking with cooldown enforcement
#   - Idle reflection: Opportunistic thinking when no tasks (no cooldown)
#
# Configuration:
#   - Load reflection schedule from triggers/reflection-schedule.json
#   - Format: {"scheduled_hours": [9, 14, 20], "reflection_duration_minutes": 30}
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
TRIGGERS_DIR="${DAEMON_ROOT}/triggers"
REFLECTION_SCHEDULE="${TRIGGERS_DIR}/reflection-schedule.json"

# ============================================================================
# Core Functions
# ============================================================================

# Get current hour (0-23)
get_current_hour() {
    date +%H | sed 's/^0//'  # Remove leading zero
}

# Check if current hour is in scheduled reflection times
is_scheduled_reflection_time() {
    if [ ! -f "$REFLECTION_SCHEDULE" ]; then
        return 1  # No schedule, treat all as idle reflection
    fi

    local current_hour=$(get_current_hour)
    local scheduled_hours

    scheduled_hours=$(jq -r '.scheduled_hours[]' "$REFLECTION_SCHEDULE" 2>/dev/null || echo "")

    if [ -z "$scheduled_hours" ]; then
        return 1  # No scheduled times
    fi

    # Check if current hour is in the scheduled list
    while read -r hour; do
        if [ "$hour" = "$current_hour" ]; then
            return 0  # Match found
        fi
    done <<< "$scheduled_hours"

    return 1  # Not in schedule
}

# Get reflection type (scheduled or idle)
# Usage: get_reflection_type
# Returns: "scheduled" or "idle"
get_reflection_type() {
    if is_scheduled_reflection_time; then
        echo "scheduled"
    else
        echo "idle"
    fi
}

# Check if persona is on cooldown for scheduled reflection
# Scheduled reflection has mandatory cooldown period
# Idle reflection does NOT enforce cooldown
is_reflection_cooldown_active() {
    local persona="$1"
    local reflection_type=$(get_reflection_type)

    # Only enforce cooldown for scheduled reflection
    if [ "$reflection_type" = "scheduled" ]; then
        if declare -f is_persona_on_cooldown >/dev/null 2>&1; then
            is_persona_on_cooldown "$persona"
            return $?
        fi
    fi

    # No cooldown for idle reflection
    return 1
}

# Get reflection cooldown duration based on type
# Scheduled: Longer cooldown to spread thinking across day
# Idle: No cooldown (take advantage of free time)
get_reflection_cooldown_hours() {
    local reflection_type=$(get_reflection_type)

    if [ "$reflection_type" = "scheduled" ]; then
        # 3-hour cooldown for scheduled reflection
        # Prevents reflection spam in narrow time windows
        echo "3"
    else
        # No cooldown for idle reflection
        echo "0"
    fi
}

# ============================================================================
# Logging & Monitoring
# ============================================================================

# Log reflection decision with type and reasoning
log_reflection_type() {
    local reflection_type=$(get_reflection_type)

    if declare -f log >/dev/null 2>&1; then
        if [ "$reflection_type" = "scheduled" ]; then
            log "INFO" "Reflection type: SCHEDULED (cooldown enforced, planned thinking time)"
        else
            log "INFO" "Reflection type: IDLE (no cooldown, opportunistic thinking)"
        fi
    fi
}

# Show reflection schedule status
show_reflection_schedule_status() {
    if [ ! -f "$REFLECTION_SCHEDULE" ]; then
        echo "Reflection Schedule: NOT CONFIGURED"
        echo ""
        echo "To enable scheduled reflection, create:"
        echo "  $REFLECTION_SCHEDULE"
        echo ""
        echo "Example content:"
        echo '  {"scheduled_hours": [9, 14, 20], "reflection_duration_minutes": 30}'
        return
    fi

    local current_hour=$(get_current_hour)
    local reflection_type=$(get_reflection_type)
    local cooldown_hours=$(get_reflection_cooldown_hours)

    echo "Reflection Schedule Status"
    echo "=========================="
    echo "Current hour: $current_hour"
    echo "Reflection type: $reflection_type"
    echo "Cooldown: $([ "$cooldown_hours" -eq 0 ] && echo "None (idle)" || echo "${cooldown_hours}h (scheduled)")"
    echo ""

    if jq . "$REFLECTION_SCHEDULE" 2>/dev/null; then
        true
    else
        echo "Invalid JSON in reflection schedule"
    fi
}

# Export functions for use in daemon.sh
export -f get_current_hour
export -f is_scheduled_reflection_time
export -f get_reflection_type
export -f is_reflection_cooldown_active
export -f get_reflection_cooldown_hours
export -f log_reflection_type
export -f show_reflection_schedule_status
