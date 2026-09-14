#!/bin/bash
#
# Self-Healing Orchestration Loop
# Runs every 15 minutes to detect anomalies and trigger autonomous remediation
#
# Purpose: Continuous detect→diagnose→remediate cycle
# Closes the feedback loop: anomalies → root causes → autonomous fixes
#
# Usage:
#   ./scripts/self-healing-loop.sh [--check|--heal|--diagnose]
#   Default (no args): Full cycle (detect + diagnose + remediate)
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
STATE_DIR="${DAEMON_ROOT}/personalities"
STATE_FILE="${STATE_DIR}/state.json"
METRICS_DIR="${DAEMON_ROOT}/metrics"
LOGS_DIR="${DAEMON_ROOT}/logs"

# Create logs directory if needed
mkdir -p "$LOGS_DIR"

# Source required libraries
if [ -f "${DAEMON_ROOT}/lib/remediation-engine.sh" ]; then
    source "${DAEMON_ROOT}/lib/remediation-engine.sh"
fi

if [ -f "${DAEMON_ROOT}/lib/root-cause-analysis.sh" ]; then
    source "${DAEMON_ROOT}/lib/root-cause-analysis.sh"
fi

if [ -f "${DAEMON_ROOT}/lib/alert-manager.sh" ]; then
    source "${DAEMON_ROOT}/lib/alert-manager.sh"
fi

if [ -f "${DAEMON_ROOT}/lib/persona-health.sh" ]; then
    source "${DAEMON_ROOT}/lib/persona-health.sh"
fi

# ============================================================================
# Main Orchestration Loop
# ============================================================================

run_self_healing_cycle() {
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    local cycle_log="${LOGS_DIR}/self-healing-cycle-$(date +%Y%m%d).log"

    echo "" >> "$cycle_log"
    echo "═══════════════════════════════════════════════════════════" >> "$cycle_log"
    echo "  Self-Healing Cycle - $timestamp" >> "$cycle_log"
    echo "═══════════════════════════════════════════════════════════" >> "$cycle_log"
    echo "" >> "$cycle_log"

    # Phase 1: Detect Anomalies
    echo "[1/4] DETECTION PHASE" >> "$cycle_log"
    local anomalies=$(detect_all_anomalies)
    local anomaly_count=$(echo "$anomalies" | jq -r 'length // 0' 2>/dev/null || echo "0")
    echo "Detected $anomaly_count anomalies" >> "$cycle_log"

    if [ "$anomaly_count" -eq 0 ]; then
        echo "✅ No anomalies detected - system healthy" >> "$cycle_log"
        echo "" >> "$cycle_log"
        return 0
    fi

    # Phase 2: Diagnose Root Causes
    echo "" >> "$cycle_log"
    echo "[2/4] DIAGNOSIS PHASE" >> "$cycle_log"
    local diagnoses=$(diagnose_all_anomalies "$anomalies")
    echo "$diagnoses" | jq -r '.[] | "  • \(.anomaly): \(.root_causes | join(", "))"' >> "$cycle_log" 2>/dev/null || true

    # Phase 3: Plan Remediations
    echo "" >> "$cycle_log"
    echo "[3/4] REMEDIATION PLANNING" >> "$cycle_log"
    local remediation_plan=$(plan_remediations "$diagnoses")
    echo "$remediation_plan" | jq -r '.[] | "  • \(.action) for \(.anomaly)"' >> "$cycle_log" 2>/dev/null || true

    # Phase 4: Execute Remediations
    echo "" >> "$cycle_log"
    echo "[4/4] REMEDIATION EXECUTION" >> "$cycle_log"
    local remediation_results=$(execute_remediations "$remediation_plan")
    echo "$remediation_results" | jq -r '.[] | "  • \(.action): \(.result)"' >> "$cycle_log" 2>/dev/null || true

    # Summary
    echo "" >> "$cycle_log"
    echo "═══════════════════════════════════════════════════════════" >> "$cycle_log"
    local remediated=$(echo "$remediation_results" | jq -r '[.[] | select(.result == "success")] | length' 2>/dev/null || echo "0")
    local total=$(echo "$remediation_results" | jq -r 'length // 0' 2>/dev/null || echo "0")
    echo "Cycle Summary: Detected=$anomaly_count, Remediated=$remediated/$total" >> "$cycle_log"
    echo "═══════════════════════════════════════════════════════════" >> "$cycle_log"

    # Update remediation health status
    update_healing_status "$anomaly_count" "$remediated" "$total"
}

# ============================================================================
# Phase 1: Anomaly Detection
# ============================================================================

detect_all_anomalies() {
    local anomalies=()

    # Check 1: Persona Lock Detection
    if declare -f detect_two_body_lock >/dev/null 2>&1; then
        if lock_info=$(detect_two_body_lock 2>/dev/null); then
            anomalies+=('{
                "type":"persona_lock",
                "context":"'$lock_info'",
                "severity":"high"
            }')
        fi
    fi

    # Check 2: Health Degradation
    if declare -f calculate_persona_health >/dev/null 2>&1; then
        for persona in architect optimizer auditor maintainer skeptic experimenter; do
            local health=$(calculate_persona_health "$persona" 2>/dev/null || echo "100")
            if [ "$health" -lt 50 ]; then
                anomalies+=('{
                    "type":"health_degradation",
                    "context":"'$persona:$health'",
                    "severity":"high"
                }')
            fi
        done
    fi

    # Check 3: Validation Failure Spike
    local recent_failures=$(grep -c "validation.*failed" "${LOGS_DIR}/activity.log" 2>/dev/null | tail -100 || echo "0")
    if [ "$recent_failures" -gt 5 ]; then
        anomalies+=('{
            "type":"validation_failures",
            "context":"validation_failure_spike:'$recent_failures'",
            "severity":"medium"
        }')
    fi

    # Check 4: Reflection Loop Detection
    for persona in architect auditor skeptic; do
        local reflection_count=$(grep -c "Starting reflection.*$persona" "${LOGS_DIR}/activity.log" 2>/dev/null | tail -50 || echo "0")
        if [ "$reflection_count" -gt 15 ]; then
            anomalies+=('{
                "type":"reflection_loop",
                "context":"'$persona:$reflection_count'",
                "severity":"medium"
            }')
        fi
    done

    # Check 5: API Health Issues
    local api_errors=$(grep -c "API error\|External.*failed" "${LOGS_DIR}/activity.log" 2>/dev/null | tail -100 || echo "0")
    if [ "$api_errors" -gt 3 ]; then
        anomalies+=('{
            "type":"api_degradation",
            "context":"claude_api:error_spike:'$api_errors'",
            "severity":"high"
        }')
    fi

    # Check 6: Queue Stagnation
    local last_completed=$(grep -l "Task completed" "${LOGS_DIR}/activity.log" 2>/dev/null | xargs -I{} stat -f%Sm -t%s {} 2>/dev/null | tail -1 || echo "0")
    local now=$(date +%s)
    local hours_since_last=$((($now - ${last_completed:-0}) / 3600))

    if [ "$hours_since_last" -gt 24 ]; then
        anomalies+=('{
            "type":"queue_stagnation",
            "context":"'$hours_since_last'",
            "severity":"critical"
        }')
    fi

    # Output as JSON array
    if [ ${#anomalies[@]} -eq 0 ]; then
        echo '[]'
    else
        echo '[' $(IFS=,; echo "${anomalies[*]}") ']'
    fi
}

# ============================================================================
# Phase 2: Root Cause Diagnosis
# ============================================================================

diagnose_all_anomalies() {
    local anomalies="$1"

    # For each anomaly, call diagnose_issue and collect results
    echo "$anomalies" | jq -r '.[] | @json' | while read -r anomaly_json; do
        local anomaly_type=$(echo "$anomaly_json" | jq -r '.type')
        local context=$(echo "$anomaly_json" | jq -r '.context')

        if declare -f diagnose_issue >/dev/null 2>&1; then
            diagnose_issue "$anomaly_type" "$context"
        fi
    done | jq -s '.'
}

# ============================================================================
# Phase 3: Remediation Planning
# ============================================================================

plan_remediations() {
    local diagnoses="$1"

    # For each diagnosis, determine best remediation action
    echo "$diagnoses" | jq -r '.[] | @json' | while read -r diagnosis_json; do
        local anomaly_type=$(echo "$diagnosis_json" | jq -r '.anomaly // .type')
        local confidence=$(echo "$diagnosis_json" | jq -r '.confidence_percent // 50')

        local action="unknown"
        local priority="normal"

        case "$anomaly_type" in
            persona_lock)
                action="break_lock"
                priority="high"
                ;;
            health_degradation)
                local severity=$(echo "$diagnosis_json" | jq -r '.severity')
                if [ "$severity" = "critical" ]; then
                    action="trigger_cooldown_12h"
                    priority="critical"
                else
                    action="trigger_cooldown_6h"
                    priority="high"
                fi
                ;;
            validation_failures)
                [ "$confidence" -gt 70 ] && action="clear_queue" || action="monitor"
                priority="medium"
                ;;
            reflection_loop)
                action="force_action_switch"
                priority="high"
                ;;
            api_degradation)
                action="open_circuit_breaker"
                priority="high"
                ;;
            queue_stagnation)
                action="archive_stalled_tasks"
                priority="critical"
                ;;
        esac

        echo '{
            "anomaly":"'$anomaly_type'",
            "action":"'$action'",
            "priority":"'$priority'",
            "confidence":'$confidence'
        }'
    done | jq -s '.'
}

# ============================================================================
# Phase 4: Remediation Execution
# ============================================================================

execute_remediations() {
    local plan="$1"

    # Execute remediations in priority order (critical → high → medium)
    # Sort by priority first
    echo "$plan" | jq -r '.[] | select(.priority == "critical") | @json' | while read -r action_json; do
        execute_single_remediation "$action_json"
    done

    echo "$plan" | jq -r '.[] | select(.priority == "high") | @json' | while read -r action_json; do
        execute_single_remediation "$action_json"
    done

    echo "$plan" | jq -r '.[] | select(.priority != "critical" and .priority != "high") | @json' | while read -r action_json; do
        execute_single_remediation "$action_json"
    done | jq -s '.'
}

execute_single_remediation() {
    local action_json="$1"
    local action=$(echo "$action_json" | jq -r '.action')
    local anomaly=$(echo "$action_json" | jq -r '.anomaly')

    local result="pending"

    case "$action" in
        break_lock)
            # Extract context if available
            if declare -f remediate_persona_lock >/dev/null 2>&1; then
                remediate_persona_lock "personas_from_context" && result="success" || result="failed"
            fi
            ;;
        trigger_cooldown_*)
            result="scheduled"  # Cooldown triggered by health monitor
            ;;
        clear_queue)
            if declare -f remediate_validation_failures >/dev/null 2>&1; then
                remediate_validation_failures "validation_failure_spike:high" && result="success" || result="failed"
            fi
            ;;
        force_action_switch)
            if declare -f remediate_reflection_loop >/dev/null 2>&1; then
                remediate_reflection_loop "persona_context" && result="success" || result="failed"
            fi
            ;;
        open_circuit_breaker)
            if declare -f remediate_api_health >/dev/null 2>&1; then
                remediate_api_health "claude_api:error_spike" && result="success" || result="failed"
            fi
            ;;
        archive_stalled_tasks)
            if declare -f remediate_queue_stagnation >/dev/null 2>&1; then
                remediate_queue_stagnation "24" && result="success" || result="failed"
            fi
            ;;
        monitor)
            result="monitoring"
            ;;
    esac

    echo '{
        "anomaly":"'$anomaly'",
        "action":"'$action'",
        "result":"'$result'"
    }'
}

# ============================================================================
# Status Management
# ============================================================================

update_healing_status() {
    local anomalies_detected="$1"
    local remediated="$2"
    local total_planned="$3"
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    local status_file="${METRICS_DIR}/healing-status.json"

    # Calculate success rate
    local success_rate=0
    if [ "$total_planned" -gt 0 ]; then
        success_rate=$((remediated * 100 / total_planned))
    fi

    cat > "$status_file" <<EOF
{
  "timestamp": "$timestamp",
  "last_cycle": {
    "anomalies_detected": $anomalies_detected,
    "remediated": $remediated,
    "planned": $total_planned,
    "success_rate_percent": $success_rate
  },
  "system_status": "$([ "$success_rate" -ge 80 ] && echo "healthy" || echo "degraded")"
}
EOF
}

get_healing_status() {
    local status_file="${METRICS_DIR}/healing-status.json"

    if [ -f "$status_file" ]; then
        cat "$status_file"
    else
        echo '{"status":"never_run"}'
    fi
}

# ============================================================================
# CLI Interface
# ============================================================================

case "${1:-}" in
    --check)
        echo "Detecting anomalies..."
        detect_all_anomalies | jq '.'
        ;;
    --diagnose)
        echo "Detecting and diagnosing..."
        local anomalies=$(detect_all_anomalies)
        diagnose_all_anomalies "$anomalies" | jq '.'
        ;;
    --heal)
        echo "Running full self-healing cycle..."
        run_self_healing_cycle
        ;;
    --status)
        get_healing_status | jq '.'
        ;;
    *)
        run_self_healing_cycle
        ;;
esac
