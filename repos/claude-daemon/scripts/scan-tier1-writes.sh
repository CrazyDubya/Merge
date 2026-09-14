#!/bin/bash
#
# Tier 1 Write Scanner
#
# Purpose: Find ALL writes to Tier 1 files and validate atomic_append usage
# Created by: Experimenter (2025-11-08)
# Context: Post-ADR-002 security review automation
#
# Why this exists:
# - Auditor and Experimenter both missed manual switch script gap
# - Human pattern recognition < systematic scanning
# - This tool prevents future blind spots
#
# Usage:
#   ./scan-tier1-writes.sh
#
# Exit codes:
#   0 - No gaps found (all Tier 1 writes protected)
#   1 - Gaps found (unprotected writes detected)

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
cd "$DAEMON_ROOT"

echo "=========================================="
echo "  Tier 1 Write Scanner (ADR-002)"
echo "=========================================="
echo ""

GAPS_FOUND=0

# Strategy: Search for all >> operations, then filter for Tier 1 files

echo "Scanning for all append operations (>>)..."
echo "----------------------------------------"

# Find all >> operations in shell scripts
while IFS= read -r match; do
    file=$(echo "$match" | cut -d: -f1)
    line=$(echo "$match" | cut -d: -f2)
    content=$(echo "$match" | cut -d: -f3-)

    # Skip self-references
    if [[ "$file" == *"atomic-io.sh"* ]] || [[ "$file" == *"scan-tier1-writes.sh"* ]]; then
        continue
    fi

    # Check if this is a Tier 1 file write
    is_tier1=false

    # Check for Tier 1 patterns
    if echo "$content" | grep -qE "(state-audit\.jsonl|AUDIT_LOG|switch-history\.jsonl|SWITCH_HISTORY|persona-timeline\.jsonl|TIMELINE_FILE|TIMELINE\")"; then
        is_tier1=true
    fi

    if [[ "$is_tier1" == true ]]; then
        # Get context to check for atomic_append
        context=$(sed -n "$((line - 5)),$((line + 5))p" "$file" 2>/dev/null || echo "")

        if echo "$context" | grep -q "atomic_append"; then
            echo "  ✓ $file:$line - PROTECTED"
        else
            echo "  ✗ $file:$line - UNPROTECTED GAP!"
            echo "     → $content"
            ((GAPS_FOUND++))
        fi
    fi
done < <(grep -rn " >>" --include="*.sh" . 2>/dev/null | grep -v "\.git/" | grep -v "experiments/" || true)

echo ""
echo "=========================================="
echo "  Scan Complete"
echo "=========================================="
echo ""

if [[ $GAPS_FOUND -eq 0 ]]; then
    echo "✓ SUCCESS: All Tier 1 writes are protected!"
    exit 0
else
    echo "✗ GAPS FOUND: $GAPS_FOUND unprotected write(s)"
    echo ""
    echo "Recommendation: Migrate to atomic_append"
    echo "See: docs/ADR-002-concurrent-write-safety.md"
    exit 1
fi
