#!/bin/bash
#
# Task Analytics Dashboard
# Analyzes task execution metrics (real-time + historical)
#
# Usage:
#   ./task-analytics.sh --realtime       # Current queue snapshot
#   ./task-analytics.sh --historical     # Last 7 days of logs
#   ./task-analytics.sh --both           # Both realtime + historical
#   ./task-analytics.sh --format json    # JSON output instead of terminal
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
TASKS_DIR="${DAEMON_ROOT}/tasks"
TASKS_FILE="${TASKS_DIR}/queue.md"
ACTIVITY_LOG="${DAEMON_ROOT}/logs/activity.log"
COMPLETED_TASKS_DIR="${TASKS_DIR}/completed"

# ============================================================================
# Real-Time Analytics (Current Queue Status)
# ============================================================================

analyze_realtime() {
    if [ ! -f "$TASKS_FILE" ]; then
        echo "Tasks file not found"
        return 1
    fi

    # Count pending tasks
    local pending=$(grep -c "^- \[ \]" "$TASKS_FILE" 2>/dev/null || echo "0")
    local in_progress=$(grep -c "^- \[~\]" "$TASKS_FILE" 2>/dev/null || echo "0")

    # Count by persona
    local tasks_per_persona
    tasks_per_persona=$(grep "^- \[\( \|~\)\]" "$TASKS_FILE" | \
                        sed 's/.*\[//; s/\].*//' | \
                        sed 's/PRIMARY:.*//' | sed 's/BACKUP.*//' | \
                        sed 's/,.*//' | \
                        sort | uniq -c | sort -rn)

    echo "=== REAL-TIME TASK ANALYTICS ==="
    echo ""
    echo "QUEUE STATUS:"
    echo "  Pending tasks: $pending"
    echo "  In-progress: $in_progress"
    echo "  Total: $((pending + in_progress))"
    echo ""
    echo "TASKS BY PERSONA:"
    echo "$tasks_per_persona" | while read count persona; do
        printf "  %-15s %d tasks\n" "$persona" "$count"
    done
    echo ""

    # Check for urgent tasks
    local urgent_threshold=$((4 * 3600))  # 4 hours in seconds
    local queue_age=$(( $(date +%s) - $(stat -c%Y "$TASKS_FILE" 2>/dev/null || echo $(date +%s)) ))

    if [ "$queue_age" -gt "$urgent_threshold" ]; then
        echo "⚠️  URGENT: Queue has pending tasks >4 hours old"
        echo "   Age: $(( queue_age / 3600 ))h $(( (queue_age % 3600) / 60 ))m"
    fi
}

# ============================================================================
# Historical Analytics (Log Analysis)
# ============================================================================

analyze_historical() {
    local days="${1:-7}"

    if [ ! -f "$ACTIVITY_LOG" ]; then
        echo "Activity log not found"
        return 1
    fi

    echo "=== HISTORICAL TASK ANALYTICS (Last $days Days) ==="
    echo ""

    # Extract task completion events from activity log
    local total_completed=0
    local total_failed=0
    local avg_duration=0

    if grep -q "Task completed\|Task failed" "$ACTIVITY_LOG"; then
        total_completed=$(grep -c "Task completed" "$ACTIVITY_LOG" 2>/dev/null || echo "0")
        total_failed=$(grep -c "Task failed\|Task quarantined" "$ACTIVITY_LOG" 2>/dev/null || echo "0")
    fi

    echo "TASK COMPLETION METRICS:"
    echo "  Tasks completed: $total_completed"
    echo "  Tasks failed: $total_failed"
    if [ $((total_completed + total_failed)) -gt 0 ]; then
        local success_rate=$(( (total_completed * 100) / (total_completed + total_failed) ))
        echo "  Success rate: $success_rate%"
    fi
    echo ""

    # Persona-specific metrics
    echo "PERSONA PERFORMANCE:"
    echo ""

    for persona in architect optimizer auditor maintainer skeptic experimenter; do
        local persona_completed=$(grep -c "\\[$persona\\].*Task completed" "$ACTIVITY_LOG" 2>/dev/null || echo "0")
        local persona_failed=$(grep -c "\\[$persona\\].*Task failed" "$ACTIVITY_LOG" 2>/dev/null || echo "0")

        if [ $((persona_completed + persona_failed)) -gt 0 ]; then
            local p_success_rate=$(( (persona_completed * 100) / (persona_completed + persona_failed) ))
            printf "  %-12s: %d completed, %d failed, %d%% success\n" \
                   "${persona^}" "$persona_completed" "$persona_failed" "$p_success_rate"
        fi
    done

    echo ""
    echo "TASK TYPE ANALYSIS:"

    # Look for common task types in logs
    local code_review=$(grep -c "Code Review\|code-review" "$ACTIVITY_LOG" 2>/dev/null || echo "0")
    local bug_fix=$(grep -c "Bug Fix\|bug-fix" "$ACTIVITY_LOG" 2>/dev/null || echo "0")
    local refactoring=$(grep -c "Refactor\|refactor" "$ACTIVITY_LOG" 2>/dev/null || echo "0")
    local documentation=$(grep -c "Document\|documentation" "$ACTIVITY_LOG" 2>/dev/null || echo "0")

    [ "$code_review" -gt 0 ] && echo "  Code reviews completed: $code_review"
    [ "$bug_fix" -gt 0 ] && echo "  Bug fixes completed: $bug_fix"
    [ "$refactoring" -gt 0 ] && echo "  Refactoring tasks: $refactoring"
    [ "$documentation" -gt 0 ] && echo "  Documentation tasks: $documentation"
}

# ============================================================================
# Bottleneck Analysis
# ============================================================================

analyze_bottlenecks() {
    echo ""
    echo "BOTTLENECK IDENTIFICATION:"
    echo ""

    # Find personas with most pending tasks
    local busiest_persona=$(grep "^- \[\( \|~\)\]" "$TASKS_FILE" 2>/dev/null | \
                           sed 's/.*\[//; s/\].*//' | \
                           sed 's/PRIMARY:.*//' | sed 's/BACKUP.*//' | \
                           sed 's/,.*//' | \
                           sort | uniq -c | sort -rn | head -1 | awk '{print $2}')

    if [ -n "$busiest_persona" ]; then
        local busy_count=$(grep -c "\[$busiest_persona\]" "$TASKS_FILE" 2>/dev/null || echo "0")
        if [ "$busy_count" -gt 5 ]; then
            echo "🔴 OVERLOADED: $busiest_persona has $busy_count pending tasks"
            echo "   Consider reassigning to backup personas or spreading work"
        fi
    fi

    # Check for high failure rate task types
    if grep -q "failure\|failed" "$ACTIVITY_LOG" 2>/dev/null; then
        local recent_failures=$(tail -500 "$ACTIVITY_LOG" 2>/dev/null | grep -c "Task failed" 2>/dev/null || echo "0")
        if [ "$recent_failures" -gt 5 ]; then
            echo "⚠️  HIGH FAILURE RATE: $recent_failures task failures in recent activity"
            echo "   Review failure reasons and adjust task requirements"
        fi
    fi
}

# ============================================================================
# JSON Output
# ============================================================================

output_json() {
    local pending=$(grep -c "^- \[ \]" "$TASKS_FILE" 2>/dev/null || echo "0")
    local in_progress=$(grep -c "^- \[~\]" "$TASKS_FILE" 2>/dev/null || echo "0")

    cat <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "queue": {
    "pending": $pending,
    "in_progress": $in_progress,
    "total": $((pending + in_progress))
  }
}
EOF
}

# ============================================================================
# Main
# ============================================================================

main() {
    local mode="--both"
    local format="terminal"

    while [ $# -gt 0 ]; do
        case "$1" in
            --realtime)
                mode="realtime"
                shift
                ;;
            --historical)
                mode="historical"
                shift
                ;;
            --both)
                mode="both"
                shift
                ;;
            --format)
                format="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done

    if [ "$format" = "json" ]; then
        output_json
    else
        case "$mode" in
            realtime)
                analyze_realtime
                ;;
            historical)
                analyze_historical 7
                ;;
            both)
                analyze_realtime
                echo ""
                analyze_historical 7
                analyze_bottlenecks
                ;;
        esac
    fi
}

# Run if executed directly
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi
