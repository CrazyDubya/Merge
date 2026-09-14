#!/bin/bash
#
# Intelligent Alert Manager
# Handles alert deduplication, enrichment, and priority-based routing
#
# Usage:
#   source "${DAEMON_ROOT}/lib/alert-manager.sh"
#   send_enriched_alert "Alert Subject" "high" "Alert message body"
#

set -euo pipefail

# Initialize alert state tracking
init_alert_manager() {
    mkdir -p "$DAEMON_ROOT/metrics"
    mkdir -p "$DAEMON_ROOT/inbox/human/unread"
    mkdir -p "$DAEMON_ROOT/monitoring-alerts"

    if [ ! -f "$DAEMON_ROOT/metrics/alert-state.json" ]; then
        echo '{"alerts": [], "last_deduplicated": null}' > "$DAEMON_ROOT/metrics/alert-state.json"
    fi
}

# Check if alert is duplicate within cooldown window
is_duplicate_alert() {
    local subject="$1"
    local priority="$2"
    local cooldown_minutes="${3:-60}"  # Default 1 hour cooldown

    local alert_state="$DAEMON_ROOT/metrics/alert-state.json"
    local since=$(date -d "-${cooldown_minutes} minutes" -u +%Y-%m-%dT%H:%M:%SZ)

    local duplicate=$(jq -r --arg subj "$subject" --arg prio "$priority" --arg since "$since" \
        '[.alerts[] | select(.subject == $subj and .priority == $prio and .timestamp > $since)] | length' \
        "$alert_state" 2>/dev/null || echo "0")

    if [ "$duplicate" -gt 0 ]; then
        return 0  # Is duplicate
    else
        return 1  # Not duplicate
    fi
}

# Increment counter for duplicate alert
increment_alert_counter() {
    local subject="$1"
    local priority="$2"

    local alert_state="$DAEMON_ROOT/metrics/alert-state.json"
    local temp_file=$(mktemp)
    trap "rm -f '$temp_file'" RETURN

    jq --arg subj "$subject" --arg prio "$priority" \
        '(.alerts[] | select(.subject == $subj and .priority == $prio) | .count) += 1' \
        "$alert_state" > "$temp_file"

    mv "$temp_file" "$alert_state"
}

# Get inbox directory based on priority
get_inbox_for_priority() {
    local priority="$1"

    case "$priority" in
        critical|high)
            echo "$DAEMON_ROOT/inbox/human/unread"
            ;;
        medium)
            echo "$DAEMON_ROOT/monitoring-alerts"
            ;;
        low)
            echo "$DAEMON_ROOT/monitoring-alerts"
            ;;
        *)
            echo "$DAEMON_ROOT/monitoring-alerts"
            ;;
    esac
}

# Create alert file with enrichment
create_alert_file() {
    local inbox_dir="$1"
    local subject="$2"
    local message="$3"
    local priority="$4"

    mkdir -p "$inbox_dir"

    local msg_file="${inbox_dir}/alert-$(date +%Y%m%d-%H%M%S).md"
    local message_id=$(uuidgen 2>/dev/null || echo "alert-$(date +%s)")

    # Enrich message with system context if available
    local enriched="$message"

    if [ -f "$DAEMON_ROOT/logs/activity.log" ]; then
        enriched="$enriched\n\n## Recent Activity (Last 20 lines)\n\`\`\`"
        enriched="$enriched\n$(tail -20 "$DAEMON_ROOT/logs/activity.log" | sed 's/```/\\```/g')"
        enriched="$enriched\n\`\`\`"
    fi

    if [ -f "$DAEMON_ROOT/personalities/state.json" ]; then
        enriched="$enriched\n\n## System State\n\`\`\`json"
        enriched="$enriched\n$(jq -c '.current_persona, (.personas | to_entries[] | .value.last_active)' "$DAEMON_ROOT/personalities/state.json" | head -5)"
        enriched="$enriched\n\`\`\`"
    fi

    # Write alert file
    cat > "$msg_file" <<EOF
---
from: anomaly-detection
to: human
timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
priority: $priority
message_id: $message_id
tags: ["alert", "monitoring"]
---

# $subject

$(echo -e "$enriched")
EOF

    if declare -f log > /dev/null 2>&1; then
        log "INFO" "Alert sent to $(basename "$inbox_dir"): $subject [$priority]"
    fi
}

# Record alert in state file
record_alert() {
    local subject="$1"
    local priority="$2"

    local alert_state="$DAEMON_ROOT/metrics/alert-state.json"
    local temp_file=$(mktemp)
    trap "rm -f '$temp_file'" RETURN

    local alert=$(jq -n \
        --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg subj "$subject" \
        --arg prio "$priority" \
        '{timestamp: $ts, subject: $subj, priority: $prio, count: 1}')

    jq --argjson alert "$alert" '.alerts += [$alert]' "$alert_state" > "$temp_file"

    # Keep only last 1000 alerts
    jq '.alerts = (.alerts | .[-1000:])' "$temp_file" > "$alert_state"

    rm -f "$temp_file"
}

# Send enriched alert (main function)
send_enriched_alert() {
    local subject="$1"
    local priority="${2:-medium}"
    local message="${3:-}"

    # Check for duplicate
    if is_duplicate_alert "$subject" "$priority" 3600; then
        increment_alert_counter "$subject" "$priority"
        if declare -f log > /dev/null 2>&1; then
            log "DEBUG" "Suppressed duplicate alert: $subject [$priority]"
        fi
        return 0
    fi

    # Enrich and route based on priority
    local inbox_dir=$(get_inbox_for_priority "$priority")
    create_alert_file "$inbox_dir" "$subject" "$message" "$priority"

    # Record in state
    record_alert "$subject" "$priority"

    return 0
}

# Log alert event to timeline (if available)
log_alert_timeline() {
    local subject="$1"
    local priority="$2"

    if declare -f log_timeline > /dev/null 2>&1; then
        log_timeline "alert_sent" "system" "Alert: $subject ($priority)"
    fi
}

# Initialize on source
init_alert_manager

export -f init_alert_manager
export -f is_duplicate_alert
export -f increment_alert_counter
export -f get_inbox_for_priority
export -f create_alert_file
export -f record_alert
export -f send_enriched_alert
export -f log_alert_timeline
