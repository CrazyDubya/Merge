#!/bin/bash
#
# Persona Health Monitoring Script
# Runs hourly via cron to check persona health and trigger cooldowns if needed
#
# Usage:
#   ./monitor-persona-health.sh
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
METRICS_DIR="${DAEMON_ROOT}/metrics"

# Source required libraries
if [ -f "$DAEMON_ROOT/lib/persona-health.sh" ]; then
    source "$DAEMON_ROOT/lib/persona-health.sh"
fi

if [ -f "$DAEMON_ROOT/lib/alert-manager.sh" ]; then
    source "$DAEMON_ROOT/lib/alert-manager.sh"
fi

# Monitor persona health and trigger cooldowns
main() {
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    echo "═══════════════════════════════════════════════════════════"
    echo "  Persona Health Monitoring - $timestamp"
    echo "═══════════════════════════════════════════════════════════"
    echo ""

    # Update health scores
    update_health_scores

    # Check each persona's health
    local cooldowns_triggered=0

    for persona in architect optimizer auditor maintainer skeptic experimenter; do
        local health=$(calculate_persona_health "$persona")
        local status=$(get_persona_health_status "$persona")
        local failure_streak=$(get_persona_failure_streak "$persona")

        echo "Checking $persona: health=$health% status=$status failures=$failure_streak"

        # Trigger cooldown if health is critical and streak is high
        if [ "$health" -lt 50 ] && [ "$failure_streak" -ge 5 ]; then
            echo "  → Triggering cooldown for $persona (health=$health%, failures=$failure_streak)"

            trigger_persona_cooldown "$persona" "health_degradation:${health}%" 6

            ((cooldowns_triggered++))
        elif [ "$status" = "UNHEALTHY" ] && [ "$failure_streak" -ge 3 ]; then
            echo "  → Alert: $persona health degrading (health=$health%, failures=$failure_streak)"

            if declare -f send_enriched_alert >/dev/null 2>&1; then
                send_enriched_alert \
                    "Persona Health Degradation: $persona" \
                    "medium" \
                    "Persona $persona health is degrading. Health: $health%, Failure streak: $failure_streak"
            fi
        fi
    done

    echo ""
    echo "Summary:"
    echo "  Timestamp: $timestamp"
    echo "  Cooldowns triggered: $cooldowns_triggered"
    echo ""
    echo "═══════════════════════════════════════════════════════════"
}

main "$@"
