#!/bin/bash
#
# Reset Activation Counters from Audit Log
# Fixes historical mismatch between total_activations and audit log entries
#
# Context: Audit logging was added Nov 9, but total_activations accumulated
#          over months. This script resets counters to audit log truth.
#
# Usage:
#   ./reset-activation-counters.sh
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
STATE_FILE="$DAEMON_ROOT/personalities/state.json"
AUDIT_LOG="$DAEMON_ROOT/logs/state-audit.jsonl"

echo "═══════════════════════════════════════════════════════════════════"
echo "  Activation Counter Reset Script"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Check if audit log exists
if [ ! -f "$AUDIT_LOG" ]; then
    echo "❌ ERROR: Audit log not found at $AUDIT_LOG"
    echo "   Cannot reset counters without audit log data"
    exit 1
fi

if [ ! -f "$STATE_FILE" ]; then
    echo "❌ ERROR: State file not found at $STATE_FILE"
    exit 1
fi

echo "📊 Analyzing audit log..."
echo "   Path: $AUDIT_LOG"
echo "   Size: $(du -h "$AUDIT_LOG" | cut -f1)"
echo "   Entries: $(wc -l < "$AUDIT_LOG")"
echo ""

# Create backup of state file
echo "📦 Creating backup of state.json..."
cp "$STATE_FILE" "${STATE_FILE}.backup-$(date +%Y%m%d-%H%M%S)"
echo "   ✓ Backup created"
echo ""

# Reset counters for each persona
echo "🔄 Resetting activation counters..."
echo ""

for persona in architect optimizer auditor maintainer skeptic experimenter; do
    # Count persona switches in audit log
    # Look for entries with "operation":"persona_switch" AND "to":"$persona"
    count=$(grep -c "\"operation\":\"persona_switch\"" "$AUDIT_LOG" 2>/dev/null || echo "0")

    # Filter by the specific "to=$persona" switch
    count=$(grep "\"operation\":\"persona_switch\"" "$AUDIT_LOG" 2>/dev/null | grep -c "\"to\":\"$persona\"" || echo "0")

    echo "   $persona:"
    echo "      Old count: $(jq -r ".personas.$persona.total_activations // 0" "$STATE_FILE")"
    echo "      New count: $count (from audit log)"

    # Update state.json
    temp_file=$(mktemp)
    trap "rm -f '$temp_file'" RETURN

    jq --arg persona "$persona" --argjson count "$count" \
        '.personas[$persona].total_activations = $count' \
        "$STATE_FILE" > "$temp_file"

    mv "$temp_file" "$STATE_FILE"
done

echo ""
echo "✅ Activation counters reset successfully"
echo ""

# Verify the reset
echo "═══════════════════════════════════════════════════════════════════"
echo "  Verification"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

echo "Current activation counts:"
for persona in architect optimizer auditor maintainer skeptic experimenter; do
    count=$(jq -r ".personas.$persona.total_activations // 0" "$STATE_FILE")
    echo "   $persona: $count"
done

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Running Audit Coverage Check"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

if [ -f "$DAEMON_ROOT/scripts/audit-coverage-monitor.sh" ]; then
    "$DAEMON_ROOT/scripts/audit-coverage-monitor.sh" || true
else
    echo "⚠️  Audit coverage monitor script not found"
fi

echo ""
echo "✅ Reset complete!"
echo ""

echo "Summary:"
echo "   • Activation counters reset from audit log"
echo "   • Backup saved with timestamp"
echo "   • Audit coverage check performed"
echo ""

echo "To restore from backup if needed:"
echo "   cp $STATE_FILE.backup-* $STATE_FILE"
echo ""
