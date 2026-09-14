#!/bin/bash
# Find the ACTUAL 46 missing entries

cd /home/opc/.claude/daemon

# Create sorted lists of ALL entries with line numbers to preserve duplicates
awk -F'"timestamp":"' '/"timestamp":"2025-11-0[5-9]/{print NR":"$2}' metrics/switch-history.jsonl | cut -d'"' -f1 > /tmp/switch-lines.txt
awk -F'"timestamp":"' '/"timestamp":"2025-11-0[5-9].*"operation":"persona_switch"/{print NR":"$2}' logs/state-audit.jsonl | cut -d'"' -f1 > /tmp/audit-lines.txt

# Count them
switches=$(wc -l < /tmp/switch-lines.txt)
audits=$(wc -l < /tmp/audit-lines.txt)

echo "Switches: $switches"
echo "Audits: $audits"
echo "Missing: $((switches - audits))"
echo ""

# Find timestamps that appear MORE in switch-history than audit
echo "Analyzing timestamp frequencies..."
cut -d':' -f2 /tmp/switch-lines.txt | sort | uniq -c | sort -rn > /tmp/switch-freq.txt
cut -d':' -f2 /tmp/audit-lines.txt | sort | uniq -c | sort -rn > /tmp/audit-freq.txt

echo ""
echo "Top 10 timestamps with mismatches:"
join -1 2 -2 2 <(sort -k2 /tmp/switch-freq.txt) <(sort -k2 /tmp/audit-freq.txt) | \
  awk '$2 != $3 {printf "%s: %d switch-history, %d audit (diff: %d)\n", $1, $2, $3, $2-$3}' | \
  head -10
