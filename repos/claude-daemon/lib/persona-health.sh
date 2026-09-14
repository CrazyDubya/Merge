#!/bin/bash
#
# Persona Health Monitoring System
# Tracks per-persona task success rates and health status
#
# Usage:
#   source "${DAEMON_ROOT}/lib/persona-health.sh"
#   record_persona_task_outcome "optimizer" "refactor-code" 0 45
#   calculate_persona_health "optimizer"
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
METRICS_DIR="${DAEMON_ROOT}/metrics"
PERSONA_HEALTH_DIR="${METRICS_DIR}/persona-health"
HEALTH_SCORES_FILE="${PERSONA_HEALTH_DIR}/health-scores.json"
COOLDOWNS_FILE="${METRICS_DIR}/persona-cooldowns.json"

# Initialize persona health directories
init_persona_health() {
    mkdir -p "$PERSONA_HEALTH_DIR"

    # Initialize cooldowns file if not exists
    if [ ! -f "$COOLDOWNS_FILE" ]; then
        echo '{"cooldowns": {}}' > "$COOLDOWNS_FILE"
    fi

    # Initialize health scores file if not exists
    if [ ! -f "$HEALTH_SCORES_FILE" ]; then
        jq -n '{
            "architect": 75,
            "optimizer": 75,
            "auditor": 75,
            "maintainer": 75,
            "skeptic": 75,
            "experimenter": 75,
            "last_updated": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"
        }' > "$HEALTH_SCORES_FILE"
    fi
}

# Record task outcome for specific persona
record_persona_task_outcome() {
    local persona="$1"
    local task="$2"
    local success="$3"  # 0=success, 1=failure
    local duration="$4"

    local log_file="${PERSONA_HEALTH_DIR}/${persona}-tasks.log"

    # Create persona log if not exists
    touch "$log_file"

    # Append task outcome
    # Format: timestamp|task|success|duration
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)|$task|$success|$duration" >> "$log_file"
}

# Calculate persona health score (rolling 24h window, 0-100)
calculate_persona_health() {
    local persona="$1"
    local window_hours="${2:-24}"

    local log_file="${PERSONA_HEALTH_DIR}/${persona}-tasks.log"

    if [ ! -f "$log_file" ]; then
        # No data yet, return neutral health
        echo "50"
        return 0
    fi

    # Get cutoff timestamp (N hours ago)
    local cutoff_timestamp=$(date -u -d "-${window_hours} hours" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -v-${window_hours}H +%Y-%m-%dT%H:%M:%SZ)

    # Count tasks and successes in single awk pass
    local stats
    stats=$(awk -F'|' -v cutoff="$cutoff_timestamp" '
        $1 > cutoff {
            total++
            if ($3 == "0") success++
        }
        END {
            print total+0 ":" success+0
        }
    ' "$log_file")

    local recent_tasks="${stats%%:*}"
    local successful_tasks="${stats##*:}"

    if [ "$recent_tasks" -lt 1 ]; then
        # No recent tasks, return 50 (neutral)
        echo "50"
        return 0
    fi

    # Calculate success rate
    local success_rate=$(( (successful_tasks * 100) / recent_tasks ))

    echo "$success_rate"
}

# Get persona health status
get_persona_health_status() {
    local persona="$1"
    local health=$(calculate_persona_health "$persona")

    if [ "$health" -ge 90 ]; then
        echo "HEALTHY"
    elif [ "$health" -ge 75 ]; then
        echo "DEGRADED"
    elif [ "$health" -ge 50 ]; then
        echo "UNHEALTHY"
    else
        echo "CRITICAL"
    fi
}

# Get persona failure streak (consecutive failures)
get_persona_failure_streak() {
    local persona="$1"
    local log_file="${PERSONA_HEALTH_DIR}/${persona}-tasks.log"

    if [ ! -f "$log_file" ]; then
        echo "0"
        return 0
    fi

    # Count consecutive failures from end of log
    local streak=0
    while IFS='|' read -r timestamp task success duration; do
        if [ "$success" = "1" ]; then
            break
        fi
        streak=$((streak + 1))
    done < <(tac "$log_file" | head -10)  # Check last 10 entries

    echo "$streak"
}

# Check if persona should be on cooldown
is_persona_on_cooldown() {
    local persona="$1"

    if [ ! -f "$COOLDOWNS_FILE" ]; then
        return 1  # Not on cooldown
    fi

    local cooldown_until=$(jq -r ".cooldowns.\"$persona\"//null" "$COOLDOWNS_FILE" 2>/dev/null)

    if [ -z "$cooldown_until" ] || [ "$cooldown_until" = "null" ]; then
        return 1  # Not on cooldown
    fi

    # Check if cooldown has expired
    local now=$(date -u +%s)
    local cooldown_epoch=$(date -u -d "$cooldown_until" +%s 2>/dev/null || date -u -j -f '%Y-%m-%dT%H:%M:%SZ' "$cooldown_until" +%s 2>/dev/null || echo "0")

    if [ "$cooldown_epoch" -le "$now" ]; then
        return 1  # Cooldown expired
    fi

    return 0  # Still on cooldown
}

# Get cooldown remaining time in hours
get_persona_cooldown_remaining() {
    local persona="$1"

    if [ ! -f "$COOLDOWNS_FILE" ]; then
        echo "0"
        return 0
    fi

    local cooldown_until=$(jq -r ".cooldowns.\"$persona\"//null" "$COOLDOWNS_FILE" 2>/dev/null)

    if [ -z "$cooldown_until" ] || [ "$cooldown_until" = "null" ]; then
        echo "0"
        return 0
    fi

    local now=$(date -u +%s)
    local cooldown_epoch=$(date -u -d "$cooldown_until" +%s 2>/dev/null || date -u -j -f '%Y-%m-%dT%H:%M:%SZ' "$cooldown_until" +%s 2>/dev/null || echo "0")
    local remaining_seconds=$((cooldown_epoch - now))

    if [ "$remaining_seconds" -le 0 ]; then
        echo "0"
        return 0
    fi

    # Convert to hours (round up)
    echo $(( (remaining_seconds + 3599) / 3600 ))
}

# Trigger cooldown for failing persona
trigger_persona_cooldown() {
    local persona="$1"
    local reason="$2"
    local duration_hours="${3:-6}"

    # Calculate cooldown expiration (current time + duration)
    local cooldown_until=$(date -u -d "+${duration_hours} hours" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -v+${duration_hours}H +%Y-%m-%dT%H:%M:%SZ)

    # Update cooldowns file
    local temp_file=$(mktemp)
    trap "rm -f '$temp_file'" RETURN
    jq --arg persona "$persona" --arg until "$cooldown_until" \
        '.cooldowns[$persona] = $until' \
        "$COOLDOWNS_FILE" > "$temp_file"
    mv "$temp_file" "$COOLDOWNS_FILE"

    # Alert to human inbox
    if declare -f send_enriched_alert >/dev/null 2>&1; then
        send_enriched_alert \
            "Persona Cooldown: $persona" \
            "high" \
            "Persona $persona placed on cooldown for $duration_hours hours (reason: $reason). Will resume activation at $cooldown_until."
    fi

    # Log to activity log if available
    if declare -f log >/dev/null 2>&1; then
        log "WARN" "Persona cooldown triggered: $persona ($reason, $duration_hours hours)"
    fi
}

# Expire persona cooldowns
expire_persona_cooldowns() {
    if [ ! -f "$COOLDOWNS_FILE" ]; then
        return 0
    fi

    local expired=()
    local now=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # Find expired cooldowns
    while IFS='=' read -r persona cooldown_until; do
        if [[ "$persona" == "cooldowns." ]]; then
            persona="${persona#cooldowns.}"
            if [ "$cooldown_until" \< "$now" ]; then
                expired+=("$persona")
            fi
        fi
    done < <(jq -r '.cooldowns | to_entries[] | "\(.key)=\(.value)"' "$COOLDOWNS_FILE" 2>/dev/null)

    # Remove expired cooldowns
    if [ ${#expired[@]} -gt 0 ]; then
        local temp_file=$(mktemp)
        trap "rm -f '$temp_file'" RETURN
        jq '.cooldowns |= with_entries(select((.value >= '"\"$now\""')))' \
            "$COOLDOWNS_FILE" > "$temp_file"
        mv "$temp_file" "$COOLDOWNS_FILE"

        # Log expiration
        for persona in "${expired[@]}"; do
            if declare -f log >/dev/null 2>&1; then
                log "INFO" "Persona cooldown expired: $persona"
            fi
        done
    fi

    return 0
}

# Update health scores (aggregate health for all personas)
update_health_scores() {
    local temp_file=$(mktemp)
    trap "rm -f '$temp_file'" RETURN
    local now=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    jq -n \
        --arg architect "$(calculate_persona_health architect)" \
        --arg optimizer "$(calculate_persona_health optimizer)" \
        --arg auditor "$(calculate_persona_health auditor)" \
        --arg maintainer "$(calculate_persona_health maintainer)" \
        --arg skeptic "$(calculate_persona_health skeptic)" \
        --arg experimenter "$(calculate_persona_health experimenter)" \
        --arg ts "$now" \
        '{
            "architect": ($architect | tonumber),
            "optimizer": ($optimizer | tonumber),
            "auditor": ($auditor | tonumber),
            "maintainer": ($maintainer | tonumber),
            "skeptic": ($skeptic | tonumber),
            "experimenter": ($experimenter | tonumber),
            "last_updated": $ts
        }' > "$temp_file"

    mv "$temp_file" "$HEALTH_SCORES_FILE"
}

# Get all persona health scores
get_all_persona_health_scores() {
    if [ ! -f "$HEALTH_SCORES_FILE" ]; then
        init_persona_health
    fi

    cat "$HEALTH_SCORES_FILE"
}

# Export functions for use in other scripts
export -f init_persona_health
export -f record_persona_task_outcome
export -f calculate_persona_health
export -f get_persona_health_status
export -f get_persona_failure_streak
export -f is_persona_on_cooldown
export -f get_persona_cooldown_remaining
export -f trigger_persona_cooldown
export -f expire_persona_cooldowns
export -f update_health_scores
export -f get_all_persona_health_scores

# Initialize on source
init_persona_health
