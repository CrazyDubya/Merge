#!/bin/bash
#
# Persona Cooldown Expiration Script
# Runs every 30 minutes via cron to expire old cooldowns
#
# Usage:
#   ./expire-persona-cooldowns.sh
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
COOLDOWNS_FILE="${DAEMON_ROOT}/metrics/persona-cooldowns.json"

# Source required libraries
if [ -f "$DAEMON_ROOT/lib/persona-health.sh" ]; then
    source "$DAEMON_ROOT/lib/persona-health.sh"
fi

# Expire old cooldowns
main() {
    if [ ! -f "$COOLDOWNS_FILE" ]; then
        return 0
    fi

    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    local now=$(date -u +%s)

    echo "═══════════════════════════════════════════════════════════"
    echo "  Cooldown Expiration Check - $timestamp"
    echo "═══════════════════════════════════════════════════════════"
    echo ""

    # Check for expired cooldowns
    local expired_count=0
    local active_cooldowns=0

    jq -r '.cooldowns | to_entries[] | "\(.key)=\(.value)"' "$COOLDOWNS_FILE" 2>/dev/null | while IFS='=' read -r persona cooldown_until; do
        if [ -z "$persona" ] || [ -z "$cooldown_until" ]; then
            continue
        fi

        # Parse cooldown time
        local cooldown_epoch=$(date -u -d "$cooldown_until" +%s 2>/dev/null || date -u -j -f '%Y-%m-%dT%H:%M:%SZ' "$cooldown_until" +%s 2>/dev/null || echo "0")

        if [ "$cooldown_epoch" -le "$now" ]; then
            echo "Cooldown expired: $persona (was until $cooldown_until)"
            ((expired_count++))
        else
            local remaining=$((cooldown_epoch - now))
            local remaining_hours=$(( (remaining + 1799) / 3600 ))
            echo "Cooldown active: $persona (${remaining_hours}h remaining)"
            ((active_cooldowns++))
        fi
    done

    # Expire cooldowns via persona-health.sh function
    if declare -f expire_persona_cooldowns >/dev/null 2>&1; then
        expire_persona_cooldowns
    fi

    echo ""
    echo "Summary:"
    echo "  Timestamp: $timestamp"
    echo "  Expired: $expired_count"
    echo "  Active: $active_cooldowns"
    echo ""
    echo "═══════════════════════════════════════════════════════════"
}

main "$@"
