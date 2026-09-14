#!/bin/bash
# Comprehensive append operation scanner
# Finds ALL >> operations and classifies them by risk tier
#
# Usage: ./scripts/scan-all-appends.sh

set -e

echo "=========================================="
echo "  Comprehensive Append Scanner"
echo "=========================================="
echo ""

# Tier 1 files (CRITICAL - must use atomic_append)
TIER1_PATTERNS=(
    "state-audit\.jsonl"
    "AUDIT_LOG"
    "switch-history\.jsonl"
    "SWITCH_HISTORY"
    "persona-timeline\.jsonl"
    "TIMELINE_FILE"
    "TIMELINE\""
)

# Tier 2 files (IMPORTANT - recommended atomic_append)
TIER2_PATTERNS=(
    "metrics/"
    "inbox/"
    "METRICS_"
    "INBOX_"
)

# Tier 3 files (OPTIONAL - best effort acceptable)
TIER3_PATTERNS=(
    "activity\.log"
    "\.log"
    "/tmp/"
    "debug"
    "test"
    "experiment"
)

# Counters
TIER1_PROTECTED=0
TIER1_GAPS=0
TIER2_PROTECTED=0
TIER2_UNPROTECTED=0
TIER3_COUNT=0
OTHER_COUNT=0

echo "Scanning for all append operations (>>)..."
echo "----------------------------------------"
echo ""

# Find all >> operations
while IFS= read -r match; do
    file=$(echo "$match" | cut -d: -f1)
    line=$(echo "$match" | cut -d: -f2)
    content=$(echo "$match" | cut -d: -f3-)

    # Skip comments
    if echo "$content" | grep -qE "^\s*#"; then
        continue
    fi

    # Skip self-references and library files
    if [[ "$file" == *"atomic-io.sh"* ]] || \
       [[ "$file" == *"scan-"*".sh"* ]] || \
       [[ "$file" == *"lib/state-audit.sh"* ]]; then
        continue
    fi

    # Get context to check for atomic_append usage
    context=$(sed -n "$((line - 5)),$((line + 5))p" "$file" 2>/dev/null || echo "")

    # Determine tier
    tier=""

    # Check Tier 1
    for pattern in "${TIER1_PATTERNS[@]}"; do
        if echo "$content" | grep -qE "$pattern"; then
            tier="TIER1"
            if echo "$context" | grep -q "atomic_append"; then
                echo "  ✓ TIER 1 PROTECTED: $file:$line"
                ((TIER1_PROTECTED++))
            else
                echo "  ✗ TIER 1 GAP: $file:$line"
                echo "    → $content"
                ((TIER1_GAPS++))
            fi
            break
        fi
    done

    # Check Tier 2 (if not Tier 1)
    if [[ -z "$tier" ]]; then
        for pattern in "${TIER2_PATTERNS[@]}"; do
            if echo "$content" | grep -qE "$pattern" || echo "$file" | grep -qE "$pattern"; then
                tier="TIER2"
                if echo "$context" | grep -q "atomic_append"; then
                    echo "  ~ TIER 2 PROTECTED: $file:$line"
                    ((TIER2_PROTECTED++))
                else
                    echo "  ~ TIER 2 UNPROTECTED: $file:$line"
                    echo "    → $content"
                    ((TIER2_UNPROTECTED++))
                fi
                break
            fi
        done
    fi

    # Check Tier 3 (if not Tier 1 or 2)
    if [[ -z "$tier" ]]; then
        for pattern in "${TIER3_PATTERNS[@]}"; do
            if echo "$content" | grep -qE "$pattern" || echo "$file" | grep -qE "$pattern"; then
                tier="TIER3"
                echo "  · TIER 3: $file:$line (best-effort OK)"
                ((TIER3_COUNT++))
                break
            fi
        done
    fi

    # Everything else
    if [[ -z "$tier" ]]; then
        echo "  ? OTHER: $file:$line"
        echo "    → $content"
        ((OTHER_COUNT++))
    fi

done < <(grep -rn " >>" --include="*.sh" . 2>/dev/null | grep -v "\.git/" | grep -v "experiments/" || true)

echo ""
echo "=========================================="
echo "  Scan Summary"
echo "=========================================="
echo ""
echo "TIER 1 (CRITICAL - must use atomic_append):"
echo "  ✓ Protected:    $TIER1_PROTECTED"
echo "  ✗ Gaps found:   $TIER1_GAPS"
echo ""
echo "TIER 2 (IMPORTANT - recommended atomic_append):"
echo "  ~ Protected:    $TIER2_PROTECTED"
echo "  ~ Unprotected:  $TIER2_UNPROTECTED"
echo ""
echo "TIER 3 (OPTIONAL - best effort acceptable):"
echo "  · Count:        $TIER3_COUNT"
echo ""
echo "OTHER (needs classification):"
echo "  ? Count:        $OTHER_COUNT"
echo ""

if [[ $TIER1_GAPS -gt 0 ]]; then
    echo "❌ CRITICAL: Tier 1 gaps found! These must be fixed."
    exit 1
elif [[ $TIER2_UNPROTECTED -gt 0 ]]; then
    echo "⚠️  WARNING: Tier 2 files could benefit from protection"
    echo "   (recommended but not required)"
    exit 0
else
    echo "✅ All critical files are protected!"
    exit 0
fi
