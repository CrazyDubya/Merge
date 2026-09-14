#!/bin/bash
#
# Anomaly Detection Script
# Identifies unusual patterns in daemon behavior that may indicate problems
#
# Runs via cron every hour and also called periodically from daemon main loop
#
# Usage:
#   ./detect-anomalies.sh [--verbose]
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
METRICS_DIR="${DAEMON_ROOT}/metrics"
LOGS_DIR="${DAEMON_ROOT}/logs"
ANOMALIES_FILE="${METRICS_DIR}/anomalies-$(date +%Y%m%d).json"

# Source helper libraries if available
if [ -f "${DAEMON_ROOT}/lib/performance-metrics.sh" ]; then
    source "${DAEMON_ROOT}/lib/performance-metrics.sh"
fi

# Helper functions
log_anomaly() {
    local severity="$1"  # warn, error, critical
    local message="$2"

    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "[${severity}] $message" >> "${LOGS_DIR}/anomaly-detection.log"

    # Also append to daily anomalies file
    echo "${timestamp}|${severity}|${message}" >> "$ANOMALIES_FILE"
}

# Check for task duration anomalies
check_task_duration_anomaly() {
    local current_avg=$(get_avg_duration 6)  # Last 6 hours
    local baseline=$(get_baseline "task_duration")

    # If no baseline or no data, skip
    if [ "$baseline" = "0" ] || [ "$current_avg" = "0" ]; then
        return 0
    fi

    # Flag if 3x slower than baseline
    if [ $(echo "$current_avg > ($baseline * 3)" | bc) -eq 1 ]; then
        log_anomaly "warn" "Task duration elevated: ${current_avg}s (baseline: ${baseline}s, +$(echo "scale=0; ($current_avg - $baseline) * 100 / $baseline" | bc)%)"
        return 1
    fi

    return 0
}

# Check for success rate degradation
check_success_rate_trend() {
    local success_24h=$(get_success_rate 24)
    local success_7d=$(get_success_rate 168)  # 7 days

    # Need at least some data
    if [ "$success_24h" = "0" ] && [ "$success_7d" = "0" ]; then
        return 0
    fi

    # If success rate drops >10% from 7d to 24h, flag as warning
    if [ "$success_7d" -gt 0 ] && [ $(echo "$success_7d - $success_24h > 10" | bc) -eq 1 ]; then
        local degradation=$(echo "$success_7d - $success_24h" | bc)
        log_anomaly "warn" "Success rate declining: 24h=${success_24h}% (7d avg=${success_7d}%, -${degradation}%)"
        return 1
    fi

    return 0
}

# Check for stagnation (no file modifications)
check_stagnation() {
    local modified=$(find "$DAEMON_ROOT" -type f -mmin -360 -not -path "*/logs/*" 2>/dev/null | wc -l)

    if [ "$modified" -eq 0 ]; then
        log_anomaly "error" "Stagnation detected: No files modified in 6 hours"
        return 1
    fi

    return 0
}

# Check for persona monopolization (two-body lock)
check_persona_lock() {
    local switch_file="${METRICS_DIR}/switch-history.jsonl"

    if [ ! -f "$switch_file" ]; then
        return 0
    fi

    # Get last 20 switches
    local recent_switches=$(tail -20 "$switch_file" 2>/dev/null)

    if [ -z "$recent_switches" ]; then
        return 0
    fi

    # Count unique persona pairs
    local unique_pairs=$(echo "$recent_switches" | \
        jq -r '[.from + "-" + .to] | unique | length' 2>/dev/null || echo "0")

    # If only 1-2 unique pairs out of 20 switches, flag persona lock
    if [ "$unique_pairs" -le 2 ]; then
        local dominant=$(echo "$recent_switches" | \
            jq -r '[.from, .to] | group_by(.) | max_by(length)[0]' 2>/dev/null)
        log_anomaly "warn" "Persona lock detected: Last 20 switches use only $unique_pairs persona pair(s)"
        return 1
    fi

    return 0
}

# Check for reflection-only loop
check_reflection_loop() {
    local activity_log="${LOGS_DIR}/activity.log"

    if [ ! -f "$activity_log" ]; then
        return 0
    fi

    # Get last 10 actions
    local recent_actions=$(grep -o "Action selected: [^ ]*" "$activity_log" 2>/dev/null | tail -10 | cut -d: -f2 | sort | uniq -c)

    # If all recent actions are "reflection", flag loop
    local reflection_count=$(echo "$recent_actions" | grep -c "reflection" || echo 0)
    local total_actions=$(echo "$recent_actions" | wc -l)

    if [ "$total_actions" -gt 0 ] && [ "$reflection_count" -eq "$total_actions" ]; then
        log_anomaly "warn" "Reflection-only loop detected: All last 10 actions are reflections (no productive work)"
        return 1
    fi

    return 0
}

# Check for task validation failures spike
check_validation_failures() {
    local metrics_file="${METRICS_DIR}/task-execution.log"

    if [ ! -f "$metrics_file" ]; then
        return 0
    fi

    # Last 6 hours validation failure rate
    local since=$(date -d "-6 hours" -u +%Y-%m-%dT%H:%M:%SZ)

    local stats=$(jq -s --arg since "$since" \
        'map(select(.timestamp > $since)) |
         if length < 5 then "insufficient"
         else
           {total: length,
            failed: (map(select(.validation_passed == 1)) | length)} |
           {rate: (.failed / .total * 100 | round)}
         end' "$metrics_file" 2>/dev/null || echo "{}")

    if echo "$stats" | grep -q "rate"; then
        local failure_rate=$(echo "$stats" | jq -r '.rate' 2>/dev/null || echo "0")

        # If >20% validation failures, flag as warning
        if [ "$failure_rate" -gt 20 ]; then
            log_anomaly "warn" "Validation failures elevated: ${failure_rate}% of tasks failed output validation (last 6h)"
            return 1
        fi
    fi

    return 0
}

# Check for CPU/memory/disk issues
check_resources() {
    local memory_usage=$(free | grep Mem | awk '{print int($3/$2 * 100)}')

    # Memory > 80% is warning
    if [ "$memory_usage" -gt 80 ]; then
        log_anomaly "warn" "Memory usage high: ${memory_usage}%"
        return 1
    fi

    # Disk usage
    local disk_usage=$(df "$DAEMON_ROOT" | tail -1 | awk '{print $5}' | sed 's/%//')

    if [ "$disk_usage" -gt 85 ]; then
        log_anomaly "warn" "Disk usage high: ${disk_usage}%"
        return 1
    fi

    return 0
}

# Check for API/Claude errors
check_api_health() {
    local activity_log="${LOGS_DIR}/activity.log"

    if [ ! -f "$activity_log" ]; then
        return 0
    fi

    # Count errors in last hour
    local since=$(date -d "-1 hour" +%Y-%m-%d\ %H:%M:%S)
    local errors=$(grep -c "ERROR\|FAILED" "$activity_log" 2>/dev/null | grep -oE '[0-9]+' | tail -1 || echo "0")

    # More than 10 errors per hour is concerning
    if [ "$errors" -gt 10 ]; then
        log_anomaly "warn" "High error rate: $errors errors in last hour"
        return 1
    fi

    return 0
}

# Main anomaly detection run
main() {
    local verbose="${1:-}"

    if [ -n "$verbose" ]; then
        echo "=== Anomaly Detection Run ==="
        echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo ""
    fi

    local anomalies_found=0

    # Run all checks
    if ! check_task_duration_anomaly; then
        anomalies_found=$((anomalies_found + 1))
    fi

    if ! check_success_rate_trend; then
        anomalies_found=$((anomalies_found + 1))
    fi

    if ! check_stagnation; then
        anomalies_found=$((anomalies_found + 1))
    fi

    if ! check_persona_lock; then
        anomalies_found=$((anomalies_found + 1))
    fi

    if ! check_reflection_loop; then
        anomalies_found=$((anomalies_found + 1))
    fi

    if ! check_validation_failures; then
        anomalies_found=$((anomalies_found + 1))
    fi

    if ! check_resources; then
        anomalies_found=$((anomalies_found + 1))
    fi

    if ! check_api_health; then
        anomalies_found=$((anomalies_found + 1))
    fi

    if [ -n "$verbose" ]; then
        if [ "$anomalies_found" -eq 0 ]; then
            echo "✅ No anomalies detected"
        else
            echo "⚠️ $anomalies_found anomalies detected (see $ANOMALIES_FILE)"
        fi
        echo ""
    fi

    return 0
}

# Run if sourced or executed
main "$@"
