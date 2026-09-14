#!/bin/bash
#
# Calculate Performance Baselines
# Processes task execution history to establish statistical baselines
#
# Run via cron daily to update baselines from rolling 7-day window
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"

# Source performance metrics library
source "${DAEMON_ROOT}/lib/performance-metrics.sh"

# Main baseline calculation
main() {
    echo "Calculating performance baselines..."
    echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo ""

    if calculate_baseline 7 10; then
        echo "✅ Baselines calculated successfully"
        echo ""
        echo "Updated baseline metrics:"
        jq '.metrics.task_duration.overall' "${DAEMON_ROOT}/metrics/baselines.json"
    else
        echo "⚠️ Could not calculate baselines (insufficient data)"
        echo "   Need at least 10 task executions in the last 7 days"
    fi
}

main "$@"
