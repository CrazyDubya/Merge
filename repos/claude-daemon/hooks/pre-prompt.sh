#!/bin/bash
#
# Pre-prompt hook for Claude Code
# This hook injects the current persona context before each prompt
#
# To enable: Add to ~/.claude/settings.json:
# {
#   "hooks": {
#     "pre-prompt": "~/.claude/daemon/hooks/pre-prompt.sh"
#   }
# }
#
# MIGRATION NOTE: Partially converted to use State API (Phase 3 of ADR-001)
# - Replaced current_persona read with state_who()
# - Traits and last_switch_time still use direct jq (no API function yet)
# - TODO: Add state_get_traits() and state_get_last_switch() to API
# - Migrated by: Experimenter (2025-11-04)
#

DAEMON_ROOT="${HOME}/.claude/daemon"
STATE_FILE="${DAEMON_ROOT}/personalities/state.json"

# Source State API
source "${DAEMON_ROOT}/lib/state-api.sh"

# Get current persona (using State API)
CURRENT_PERSONA=$(state_who 2>/dev/null || echo "")

if [ -z "$CURRENT_PERSONA" ] || [ "$CURRENT_PERSONA" = "null" ]; then
    # No active persona, pass through
    exit 0
fi

# Load persona definition
PERSONA_FILE="${DAEMON_ROOT}/personalities/archetypes/${CURRENT_PERSONA}.md"

if [ ! -f "$PERSONA_FILE" ]; then
    exit 0
fi

# Check if this is a daemon-initiated interaction
# (we don't want to inject persona for regular user sessions)
if [ "${CLAUDE_PERSONA:-}" != "$CURRENT_PERSONA" ]; then
    exit 0
fi

# Inject persona context
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎭 ACTIVE PERSONA: ${CURRENT_PERSONA^^}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Load CLAUDE.md if it exists
if [ -f "${HOME}/recursion/CLAUDE.md" ]; then
    echo "📋 Project context loaded from CLAUDE.md"
    echo ""
fi

# Show current traits
TRAITS=$(jq -r --arg p "$CURRENT_PERSONA" '.personas[$p].base_traits[]' "$STATE_FILE" 2>/dev/null | tr '\n' ',' | sed 's/,$//')
if [ -n "$TRAITS" ]; then
    echo "🧬 Active traits: $TRAITS"
    echo ""
fi

# Show recent activity summary
LAST_SWITCH=$(jq -r '.last_switch_time' "$STATE_FILE" 2>/dev/null || echo "unknown")
if [ "$LAST_SWITCH" != "null" ] && [ -n "$LAST_SWITCH" ]; then
    echo "⏰ Active since: $LAST_SWITCH"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

exit 0
