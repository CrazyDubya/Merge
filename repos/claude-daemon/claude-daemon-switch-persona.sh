#!/bin/bash
#
# Manually switch the active persona
#
# MIGRATION NOTE: Converted to use State API (Phase 3 of ADR-001)
# - Replaced direct jq manipulation with state_become()
# - Validation, transactions, and audit logging now handled by API
# - Migrated by: Experimenter (2025-11-04)
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
SWITCH_HISTORY="${DAEMON_ROOT}/metrics/switch-history.jsonl"

# Source State API
source "${DAEMON_ROOT}/lib/state-api.sh"

PERSONAS=("auditor" "optimizer" "architect" "experimenter" "maintainer" "skeptic")

if [ $# -eq 0 ]; then
    echo "Usage: $0 <persona>"
    echo ""
    echo "Available personas:"
    echo "  • auditor      - Security-focused, cautious, thorough"
    echo "  • optimizer    - Performance-obsessed, data-driven"
    echo "  • architect    - Systems thinker, pattern-focused"
    echo "  • experimenter - Chaotic creative, exploratory"
    echo "  • maintainer   - Stability-focused, documentation-loving"
    echo "  • skeptic      - Questioning everything, devil's advocate"
    echo ""
    exit 1
fi

NEW_PERSONA="$1"

# Validate persona
if [[ ! " ${PERSONAS[@]} " =~ " ${NEW_PERSONA} " ]]; then
    echo "❌ Error: Unknown persona '$NEW_PERSONA'"
    echo ""
    echo "Available: ${PERSONAS[*]}"
    exit 1
fi

# Get current persona (using State API)
CURRENT_PERSONA=$(state_who)

if [ "$CURRENT_PERSONA" = "$NEW_PERSONA" ]; then
    echo "⚠️  Already active: $NEW_PERSONA"
    exit 0
fi

# Switch persona (using State API - handles validation, transactions, audit logging)
if state_become "$NEW_PERSONA" "manual"; then
    # Log to legacy switch history (for backwards compatibility)
    # ADR-002: Use atomic_append for concurrent-safe writes (Tier 1 - critical state)
    local entry
    entry=$(printf '{"timestamp":"%s","from":"%s","to":"%s","reason":"manual","layer":"user_override"}' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CURRENT_PERSONA" "$NEW_PERSONA")
    atomic_append "$SWITCH_HISTORY" "$entry"

    echo "✅ Persona switched: $CURRENT_PERSONA → $NEW_PERSONA"
    echo ""
    echo "The daemon will embody $NEW_PERSONA on its next activation."
else
    echo "❌ Failed to switch persona (see error above)"
    exit 1
fi
