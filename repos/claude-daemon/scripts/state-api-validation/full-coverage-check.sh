#!/bin/bash
# TRUE audit coverage check (all entries, not unique timestamps)

cd /home/opc/.claude/daemon

switches_total=$(grep '"timestamp":"2025-11-0[5-9]' metrics/switch-history.jsonl | wc -l)
audits_total=$(grep '"timestamp":"2025-11-0[5-9]' logs/state-audit.jsonl | grep '"operation":"persona_switch"' | wc -l)

echo "=== TRUE AUDIT COVERAGE (All Entries) ==="
echo ""
echo "Total switch-history entries: $switches_total"
echo "Total audit log entries:      $audits_total"
echo "Missing:                      $((switches_total - audits_total))"
echo ""

coverage=$(awk "BEGIN {printf \"%.2f\", ($audits_total / $switches_total) * 100}")
echo "Coverage: $coverage%"
echo ""

if (( $(echo "$coverage > 90" | bc -l) )); then
    echo "✅ Target >90% ACHIEVED"
else
    echo "❌ Target >90% NOT MET"
fi

echo ""
echo "=== THRASHING ANALYSIS ==="
echo "Nov 5 noon: $(grep "2025-11-05T12:" metrics/switch-history.jsonl | wc -l) switch-history, $(grep "2025-11-05T12:" logs/state-audit.jsonl | grep '"operation":"persona_switch"' | wc -l) audit"
echo "Nov 6 noon: $(grep "2025-11-06T12:" metrics/switch-history.jsonl | wc -l) switch-history, $(grep "2025-11-06T12:" logs/state-audit.jsonl | grep '"operation":"persona_switch"' | wc -l) audit"
echo "Nov 7 noon: $(grep "2025-11-07T12:" metrics/switch-history.jsonl | wc -l) switch-history, $(grep "2025-11-07T12:" logs/state-audit.jsonl | grep '"operation":"persona_switch"' | wc -l) audit"
