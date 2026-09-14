#!/bin/bash
#
# Daemon Health Dashboard
# Generates comprehensive health status report with anomaly summary
#
# Usage:
#   ./health-dashboard.sh                 # Text output
#   ./health-dashboard.sh --json          # JSON output
#   ./health-dashboard.sh --watch         # Continuous monitoring
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
METRICS_DIR="${DAEMON_ROOT}/metrics"
LOGS_DIR="${DAEMON_ROOT}/logs"

# Source helper libraries
[ -f "${DAEMON_ROOT}/lib/performance-metrics.sh" ] && source "${DAEMON_ROOT}/lib/performance-metrics.sh" || true

# Generate text dashboard
generate_text_dashboard() {
    cat <<'EOF'

╔════════════════════════════════════════════════════════════════════╗
║            🤖 Daemon Health Dashboard                              ║
╚════════════════════════════════════════════════════════════════════╝

EOF

    echo "Generated: $(date -u +%Y-%m-%d\ %H:%M:%SZ\ UTC)"
    echo ""

    # Overall Status
    determine_overall_status
    echo ""

    # Task Execution Metrics (24h)
    echo "📊 Task Execution (Last 24h)"
    echo "─────────────────────────────────────────────"

    local success_rate=$(get_success_rate 24)
    local avg_duration=$(get_avg_duration 24)
    local task_count=$(grep -c "task_complete\|task_failed" "$LOGS_DIR/activity.log" 2>/dev/null || echo "0")

    echo "  Tasks: $task_count"
    echo "  Success Rate: ${success_rate}%"
    echo "  Avg Duration: ${avg_duration}s"
    echo ""

    # Persona Activity
    echo "🎭 Persona Activity (Last 24h)"
    echo "─────────────────────────────────────────────"
    get_persona_stats
    echo ""

    # Anomalies
    echo "⚠️  Anomalies Detected"
    echo "─────────────────────────────────────────────"
    if [ -f "${METRICS_DIR}/anomalies-$(date +%Y%m%d).json" ]; then
        local anomaly_count=$(wc -l < "${METRICS_DIR}/anomalies-$(date +%Y%m%d).json")
        echo "  Found $anomaly_count anomalies today"
        tail -5 "${METRICS_DIR}/anomalies-$(date +%Y%m%d).json" | \
            awk -F'|' '{print "    ⚡ " $3}'
    else
        echo "  ✅ No anomalies detected"
    fi
    echo ""

    # Resource Status
    echo "💾 Resource Status"
    echo "─────────────────────────────────────────────"
    show_resource_stats
    echo ""

    # Recent Alerts
    echo "🔔 Recent Alerts"
    echo "─────────────────────────────────────────────"
    if [ -d "${DAEMON_ROOT}/inbox/human/unread" ]; then
        local alert_count=$(ls "${DAEMON_ROOT}/inbox/human/unread" 2>/dev/null | wc -l)
        echo "  Unread alerts: $alert_count"
    fi
    echo ""

    # Recommendations
    echo "💡 Recommendations"
    echo "─────────────────────────────────────────────"
    generate_recommendations
}

# Generate JSON dashboard
generate_json_dashboard() {
    local success_rate=$(get_success_rate 24)
    local avg_duration=$(get_avg_duration 24)
    local task_count=$(grep -c "task_complete\|task_failed" "$LOGS_DIR/activity.log" 2>/dev/null || echo "0")

    jq -n \
        --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg success_rate "$success_rate" \
        --arg avg_duration "$avg_duration" \
        --arg task_count "$task_count" \
        '{
            timestamp: $ts,
            metrics: {
                tasks_24h: ($task_count | tonumber),
                success_rate_24h: ($success_rate | tonumber),
                avg_duration_seconds: ($avg_duration | tonumber)
            }
        }'
}

# Determine overall status
determine_overall_status() {
    local success_rate=$(get_success_rate 24)

    if [ "$success_rate" -ge 90 ]; then
        echo "✅ Overall Status: HEALTHY"
    elif [ "$success_rate" -ge 75 ]; then
        echo "⚠️  Overall Status: WARNING"
    else
        echo "❌ Overall Status: CRITICAL"
    fi
}

# Get persona activity stats
get_persona_stats() {
    if [ ! -f "${METRICS_DIR}/switch-history.jsonl" ]; then
        echo "  No persona data available"
        return
    fi

    local since=$(date -d "-24 hours" -u +%Y-%m-%dT%H:%M:%SZ)

    jq -s --arg since "$since" \
        'map(select(.timestamp > $since)) |
         group_by(.to) |
         map({persona: .[0].to, count: length}) |
         sort_by(.count) | reverse[] |
         "  \(.persona): \(.count) activations"' \
        "${METRICS_DIR}/switch-history.jsonl" 2>/dev/null | head -10 || echo "  Unable to load persona stats"
}

# Show resource statistics
show_resource_stats() {
    local memory=$(free | grep Mem | awk '{printf "%d/%d GB (%.0f%%)", $3/1024/1024, $2/1024/1024, $3/$2*100}')
    echo "  Memory: $memory"

    local disk=$(df "$DAEMON_ROOT" | tail -1 | awk '{printf "%d/%d GB (%.0f%%)", $3/1024/1024, $2/1024/1024, $5}' | sed 's/%//')
    echo "  Disk: $disk%"

    local activity_log_size=$(du -h "$LOGS_DIR/activity.log" 2>/dev/null | cut -f1)
    echo "  Activity Log: ${activity_log_size:-unknown}"
}

# Generate recommendations
generate_recommendations() {
    local issues=0

    local success_rate=$(get_success_rate 24)
    if [ "$success_rate" -lt 80 ]; then
        echo "  1. Success rate declining - check logs for error patterns"
        issues=$((issues + 1))
    fi

    local avg_duration=$(get_avg_duration 24)
    local baseline=$(get_baseline "task_duration")
    if [ "$baseline" != "0" ] && [ "$avg_duration" -gt $((baseline * 3)) ]; then
        echo "  $((issues + 1)). Task duration 3x slower than baseline - check system load"
        issues=$((issues + 1))
    fi

    if [ "$issues" -eq 0 ]; then
        echo "  ✅ System is performing normally"
    fi
}

# Watch mode - continuous refresh
watch_mode() {
    while true; do
        clear
        generate_text_dashboard
        echo "Refreshing in 30 seconds... (Ctrl+C to exit)"
        sleep 30
    done
}

# Main
main() {
    local mode="${1:-text}"

    case "$mode" in
        --json)
            generate_json_dashboard
            ;;
        --watch)
            watch_mode
            ;;
        --text|*)
            generate_text_dashboard
            ;;
    esac
}

main "$@"
