#!/bin/bash
# Quick audit coverage check

cd /home/opc/.claude/daemon

switches=$(grep '"timestamp":"2025-11-0[5-9]' metrics/switch-history.jsonl | wc -l)
audits=$(grep '"timestamp":"2025-11-0[5-9]' logs/state-audit.jsonl | grep '"operation":"persona_switch"' | wc -l)
missing=$((switches - audits))

echo "=== AUDIT COVERAGE ANALYSIS (Nov 5-7) ==="
echo ""
echo "Total switches:     $switches"
echo "Audited switches:   $audits"
echo "Missing audits:     $missing"
echo ""

coverage=$(awk "BEGIN {printf \"%.2f\", ($audits / $switches) * 100}")
echo "Coverage: $coverage%"
echo ""

if (( $(echo "$coverage > 90" | bc -l) )); then
    echo "✅ Target >90% ACHIEVED"
else
    echo "❌ Target >90% NOT MET"
fi
