#!/bin/bash
#
# Remediation Engine Library
# Automatically fixes detected anomalies without human intervention
#
# Purpose: Transform "self-aware" (detects issues) to "self-healing" (fixes issues)
# Handles: Persona locks, health degradation, validation failures, reflection loops, etc.
#
# Usage:
#   source "${DAEMON_ROOT}/lib/remediation-engine.sh"
#   remediate_anomaly "persona_lock" "architect,optimizer"
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
STATE_DIR="${DAEMON_ROOT}/personalities"
STATE_FILE="${STATE_DIR}/state.json"
METRICS_DIR="${DAEMON_ROOT}/metrics"
LOGS_DIR="${DAEMON_ROOT}/logs"

# ============================================================================
# Core Remediation Dispatcher
# ============================================================================

# Map anomaly type to remediation handler
# Usage: remediate_anomaly "anomaly_type" "additional_context"
remediate_anomaly() {
    local anomaly_type="$1"
    local context="${2:-}"
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    echo "$(date '+%Y-%m-%d %H:%M:%S') [REMEDIATION] Processing $anomaly_type: $context" >> "$LOGS_DIR/remediation.log"

    case "$anomaly_type" in
        persona_lock)
            remediate_persona_lock "$context"
            ;;
        health_degradation)
            remediate_health_degradation "$context"
            ;;
        validation_failure_spike)
            remediate_validation_failures "$context"
            ;;
        reflection_loop)
            remediate_reflection_loop "$context"
            ;;
        api_health_degraded)
            remediate_api_health "$context"
            ;;
        queue_stagnation)
            remediate_queue_stagnation "$context"
            ;;
        *)
            log "WARN" "Unknown anomaly type: $anomaly_type"
            return 1
            ;;
    esac
}

# ============================================================================
# Remediation 1: Persona Lock Breaker
# ============================================================================

# Breaks 2-persona locks automatically
# Usage: remediate_persona_lock "architect,optimizer"
remediate_persona_lock() {
    local locked_personas="$1"
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    log "INFO" "[REMEDY] Breaking persona lock: $locked_personas"

    # Get all personas
    local all_personas=("architect" "optimizer" "auditor" "maintainer" "skeptic" "experimenter")
    local available_personas=()

    # Find personas not in the lock
    for persona in "${all_personas[@]}"; do
        if ! echo "$locked_personas" | grep -q "$persona"; then
            available_personas+=("$persona")
        fi
    done

    # If available personas exist, force switch to one
    if [ ${#available_personas[@]} -gt 0 ]; then
        local new_persona="${available_personas[$((RANDOM % ${#available_personas[@]}))]}"

        # Force immediate switch
        if declare -f state_become >/dev/null 2>&1; then
            state_become "$new_persona" "lock_break_remediation"
            log "INFO" "[REMEDY] ✅ Lock broken: Switched to $new_persona"
        fi

        # Record remediation
        record_remediation "persona_lock" "broken" "$locked_personas" "$new_persona"
        return 0
    fi

    log "WARN" "[REMEDY] ❌ No available personas to switch to"
    return 1
}

# ============================================================================
# Remediation 2: Health Degradation Handler
# ============================================================================

# Triggers cooldowns and alerts for unhealthy personas
# Usage: remediate_health_degradation "optimizer:35"
remediate_health_degradation() {
    local context="$1"  # "persona:health_score"
    local persona="${context%%:*}"
    local health_score="${context##*:}"

    log "INFO" "[REMEDY] Health degradation detected: $persona at $health_score%"

    # If health < 30%, trigger aggressive cooldown
    if [ "$health_score" -lt 30 ] && declare -f trigger_persona_cooldown >/dev/null 2>&1; then
        log "INFO" "[REMEDY] Critical health ($health_score%) - triggering 12h cooldown"
        trigger_persona_cooldown "$persona" "critical_health_degradation:${health_score}%" 12
        record_remediation "health_degradation" "cooldown_triggered" "$persona" "12h"
        return 0
    fi

    # If health 30-50%, trigger standard cooldown
    if [ "$health_score" -lt 50 ] && declare -f trigger_persona_cooldown >/dev/null 2>&1; then
        log "INFO" "[REMEDY] Low health ($health_score%) - triggering 6h cooldown"
        trigger_persona_cooldown "$persona" "low_health_degradation:${health_score}%" 6
        record_remediation "health_degradation" "cooldown_triggered" "$persona" "6h"
        return 0
    fi

    record_remediation "health_degradation" "monitoring" "$persona" "no_action"
    return 0
}

# ============================================================================
# Remediation 3: Validation Failure Handler
# ============================================================================

# Clears retry queue and resets validation state on spikes
# Usage: remediate_validation_failures "task-validation:15"
remediate_validation_failures() {
    local context="$1"  # "failure_type:count"
    local failure_type="${context%%:*}"
    local failure_count="${context##*:}"

    log "INFO" "[REMEDY] Validation failure spike: $failure_type ($failure_count failures)"

    # If spike > 10 failures, clear retry queue for fresh start
    if [ "$failure_count" -gt 10 ]; then
        log "INFO" "[REMEDY] High failure count ($failure_count) - clearing retry queue"

        # Initialize empty retry tracking
        if [ -f "${METRICS_DIR}/retry-tracking.json" ]; then
            echo '{"retries":{}}' > "${METRICS_DIR}/retry-tracking.json"
            record_remediation "validation_failures" "queue_cleared" "$failure_type" "$failure_count"
            return 0
        fi
    fi

    # If spike > 5 failures, alert but don't clear
    if [ "$failure_count" -gt 5 ]; then
        if declare -f send_enriched_alert >/dev/null 2>&1; then
            send_enriched_alert \
                "Validation Failure Spike: $failure_type" \
                "high" \
                "High validation failure rate detected: $failure_count failures. Monitoring for escalation."
        fi
        record_remediation "validation_failures" "alert_sent" "$failure_type" "$failure_count"
        return 0
    fi

    return 0
}

# ============================================================================
# Remediation 4: Reflection Loop Breaker
# ============================================================================

# Breaks infinite reflection cycles by forcing action
# Usage: remediate_reflection_loop "architect:12"
remediate_reflection_loop() {
    local context="$1"  # "persona:reflection_count"
    local persona="${context%%:*}"
    local reflection_count="${context##*:}"

    log "INFO" "[REMEDY] Reflection loop detected: $persona ($reflection_count reflections)"

    # If >15 consecutive reflections, force switch to action-oriented persona
    if [ "$reflection_count" -gt 15 ]; then
        log "INFO" "[REMEDY] Breaking reflection loop - forcing action persona"

        # Choose action-oriented persona
        local action_personas=("optimizer" "maintainer" "experimenter")
        local new_persona="${action_personas[$((RANDOM % ${#action_personas[@]}))]}"

        if declare -f state_become >/dev/null 2>&1; then
            state_become "$new_persona" "break_reflection_loop"
            log "INFO" "[REMEDY] ✅ Reflection loop broken: Switched to $new_persona"
        fi

        record_remediation "reflection_loop" "broken" "$persona" "$new_persona:$reflection_count"
        return 0
    fi

    return 0
}

# ============================================================================
# Remediation 5: API Health Recovery
# ============================================================================

# Implements circuit breaker and retry backoff for API issues
# Usage: remediate_api_health "claude_api:error_rate_25"
remediate_api_health() {
    local context="$1"  # "api_name:issue"
    local api_name="${context%%:*}"
    local issue="${context##*:}"

    log "INFO" "[REMEDY] API health issue: $api_name - $issue"

    # Initialize circuit breaker state if needed
    local breaker_state_file="${METRICS_DIR}/circuit-breaker-${api_name}.json"

    if [ ! -f "$breaker_state_file" ]; then
        echo '{"state":"closed","failures":0,"last_attempt":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}' > "$breaker_state_file"
    fi

    # Read current state
    local failures=$(jq -r '.failures // 0' "$breaker_state_file")

    # Increment failure count
    failures=$((failures + 1))

    # If >3 failures, open circuit breaker
    if [ "$failures" -gt 3 ]; then
        log "INFO" "[REMEDY] Opening circuit breaker for $api_name"

        # Update state to open
        jq '.state = "open" | .failures = '$failures' | .opened_at = "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"' \
            "$breaker_state_file" > "${breaker_state_file}.tmp"
        mv "${breaker_state_file}.tmp" "$breaker_state_file"

        record_remediation "api_health" "circuit_open" "$api_name" "$failures failures"
        return 0
    fi

    # Update failure count
    jq '.failures = '$failures' | .last_attempt = "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"' \
        "$breaker_state_file" > "${breaker_state_file}.tmp"
    mv "${breaker_state_file}.tmp" "$breaker_state_file"

    record_remediation "api_health" "retrying" "$api_name" "$failures failures"
    return 0
}

# ============================================================================
# Remediation 6: Queue Stagnation Handler
# ============================================================================

# Clears stalled tasks and resets queue
# Usage: remediate_queue_stagnation "24"
remediate_queue_stagnation() {
    local hours_stalled="$1"

    log "INFO" "[REMEDY] Queue stagnation detected: $hours_stalled hours"

    # If stalled >24h, archive old in-progress tasks and restart
    if [ "$hours_stalled" -gt 24 ]; then
        log "INFO" "[REMEDY] Archive stalled in-progress tasks"

        local queue_file="${DAEMON_ROOT}/tasks/queue.md"
        if [ -f "$queue_file" ]; then
            # Move all [~] in-progress tasks to archive
            local stalled_count=$(grep -c "^- \[~\]" "$queue_file" || echo "0")

            if [ "$stalled_count" -gt 0 ]; then
                # Create archive
                mkdir -p "${DAEMON_ROOT}/tasks/stalled"
                local archive_file="${DAEMON_ROOT}/tasks/stalled/stalled-$(date +%Y%m%d-%H%M%S).md"
                grep "^- \[~\]" "$queue_file" > "$archive_file"

                # Remove stalled tasks from queue
                sed -i '/^- \[~\]/d' "$queue_file"

                log "INFO" "[REMEDY] ✅ Archived $stalled_count stalled tasks"
                record_remediation "queue_stagnation" "archived" "in-progress tasks" "$stalled_count"
                return 0
            fi
        fi
    fi

    return 0
}

# ============================================================================
# Remediation Recording & Logging
# ============================================================================

# Record remediation action for audit trail
record_remediation() {
    local anomaly_type="$1"
    local action_taken="$2"
    local context="$3"
    local result="$4"
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # Ensure logs directory exists
    mkdir -p "$LOGS_DIR"

    # Log to remediation audit trail
    local remediation_entry=$(cat <<EOF
{
  "timestamp": "$timestamp",
  "anomaly_type": "$anomaly_type",
  "action": "$action_taken",
  "context": "$context",
  "result": "$result"
}
EOF
)

    if declare -f atomic_append >/dev/null 2>&1; then
        atomic_append "${LOGS_DIR}/remediation-audit.jsonl" "$remediation_entry"
    else
        echo "$remediation_entry" >> "${LOGS_DIR}/remediation-audit.jsonl"
    fi
}

# Get remediation statistics
get_remediation_stats() {
    local days_back="${1:-7}"
    local cutoff_time=$(date -u -d "$days_back days ago" +%Y-%m-%dT%H:%M:%SZ)

    if [ ! -f "${LOGS_DIR}/remediation-audit.jsonl" ]; then
        echo '{"total":0,"by_type":{},"success_rate":0}'
        return 0
    fi

    # Calculate stats from audit trail
    local total=$(wc -l < "${LOGS_DIR}/remediation-audit.jsonl")
    local successful=$(grep -c '"action":"broken"\|"action":"cooldown_triggered"\|"action":"circuit_open"' "${LOGS_DIR}/remediation-audit.jsonl" || echo "0")

    local success_rate=0
    if [ "$total" -gt 0 ]; then
        success_rate=$((successful * 100 / total))
    fi

    echo "{\"total\":$total,\"successful\":$successful,\"success_rate\":$success_rate}"
}

# ============================================================================
# Remediation Health Check
# ============================================================================

# Check if remediation system is healthy
check_remediation_health() {
    local status="healthy"
    local issues=()

    # Check for failed remediations (>50% failure rate)
    if [ -f "${LOGS_DIR}/remediation-audit.jsonl" ]; then
        local total=$(wc -l < "${LOGS_DIR}/remediation-audit.jsonl")
        if [ "$total" -gt 10 ]; then
            local failed=$(grep -c '"action":"failed"\|"action":"monitoring"\|"action":"alert_sent"' "${LOGS_DIR}/remediation-audit.jsonl" || echo "0")
            local failure_rate=$((failed * 100 / total))

            if [ "$failure_rate" -gt 50 ]; then
                status="warning"
                issues+=("High remediation failure rate: $failure_rate%")
            fi
        fi
    fi

    # Check for stuck circuit breakers
    for breaker_file in "${METRICS_DIR}"/circuit-breaker-*.json; do
        if [ -f "$breaker_file" ]; then
            local state=$(jq -r '.state' "$breaker_file" 2>/dev/null || echo "unknown")
            local opened_at=$(jq -r '.opened_at' "$breaker_file" 2>/dev/null)

            if [ "$state" = "open" ] && [ -n "$opened_at" ]; then
                local age_seconds=$(date -u +%s) - $(date -d "$opened_at" +%s 2>/dev/null || echo 0)
                if [ "$age_seconds" -gt 3600 ]; then  # Open for > 1 hour
                    status="warning"
                    issues+=("Circuit breaker stuck open: $(basename $breaker_file)")
                fi
            fi
        fi
    done

    # Report status
    echo "status=$status"
    if [ ${#issues[@]} -gt 0 ]; then
        echo "issues=${issues[*]}"
    fi
}

# Export functions
export -f remediate_anomaly
export -f remediate_persona_lock
export -f remediate_health_degradation
export -f remediate_validation_failures
export -f remediate_reflection_loop
export -f remediate_api_health
export -f remediate_queue_stagnation
export -f record_remediation
export -f get_remediation_stats
export -f check_remediation_health
