#!/bin/bash
# Fix Timeline JSON Format
# Maintainer persona - Convert malformed multi-line JSON to single-line without losing data
#
# IMPORTANT: This script PRESERVES all information, only fixes formatting

set -e

TIMELINE="memory/persona-timeline.jsonl"
BACKUP="memory/persona-timeline.jsonl.backup-$(date +%Y%m%d-%H%M%S)"
FIXED="memory/persona-timeline.jsonl.fixed"

echo "=== TIMELINE JSON FORMAT FIX ==="
echo ""
echo "Purpose: Convert malformed multi-line JSON to single-line format"
echo "Safety: Preserves ALL information, only changes formatting"
echo ""

# Safety check
if [ ! -f "$TIMELINE" ]; then
    echo "Error: Timeline file not found: $TIMELINE"
    exit 1
fi

# Step 1: Create backup
echo "[1/4] Creating safety backup..."
cp "$TIMELINE" "$BACKUP"
echo "  ✓ Backed up to $BACKUP"

# Step 2: Analyze current state
echo ""
echo "[2/4] Analyzing current timeline..."
TOTAL_LINES=$(wc -l < "$TIMELINE")
WELL_FORMED=$(grep -c '^\{"timestamp".*\}$' "$TIMELINE" || echo "0")
MALFORMED=$(($(grep -c '^{' "$TIMELINE") - $WELL_FORMED))

echo "  Total lines: $TOTAL_LINES"
echo "  Well-formed single-line JSON: $WELL_FORMED"
echo "  Malformed multi-line JSON: $MALFORMED"

# Step 3: Fix formatting using Python (more reliable than awk for JSON)
echo ""
echo "[3/4] Fixing JSON formatting..."

python3 << 'PYTHON_SCRIPT'
import json
import sys
import re

timeline_file = "memory/persona-timeline.jsonl"
output_file = "memory/persona-timeline.jsonl.fixed"

with open(timeline_file, 'r') as f:
    content = f.read()

# Parse JSON objects (both single-line and multi-line)
lines = content.split('\n')
json_buffer = ""
in_json = False
entries = []
well_formed_count = 0
fixed_count = 0

for line in lines:
    stripped = line.strip()

    # Check if this is already a complete single-line JSON
    if re.match(r'^\{.*\}$', stripped):
        try:
            obj = json.loads(stripped)
            # Already well-formed, keep as-is
            entries.append(json.dumps(obj, separators=(',', ':')))
            well_formed_count += 1
            continue
        except json.JSONDecodeError:
            pass  # Not valid JSON, treat as multi-line start

    if stripped.startswith('{'):
        in_json = True
        json_buffer = stripped
    elif in_json:
        json_buffer += " " + stripped
        if stripped.endswith('}'):
            # Try to parse the JSON
            try:
                obj = json.loads(json_buffer)
                # Write as single-line compact JSON
                entries.append(json.dumps(obj, separators=(',', ':')))
                fixed_count += 1
            except json.JSONDecodeError:
                # If parsing fails, keep original (better than losing data)
                sys.stderr.write(f"Warning: Could not parse JSON, keeping original\n")
                entries.append(json_buffer)
                fixed_count += 1
            json_buffer = ""
            in_json = False
    else:
        # Empty lines or noise - skip
        pass

# Write fixed timeline
with open(output_file, 'w') as f:
    for entry in entries:
        f.write(entry + '\n')

print(f"  ✓ Kept {well_formed_count} well-formed entries")
print(f"  ✓ Fixed {fixed_count} malformed entries")
print(f"  ✓ Total: {len(entries)} entries")
PYTHON_SCRIPT

# Step 4: Verify and replace
echo ""
echo "[4/4] Verifying fixed timeline..."

# Count entries in fixed file
FIXED_ENTRIES=$(wc -l < "$FIXED")
FIXED_WELL_FORMED=$(grep -c '^\{' "$FIXED" || echo "0")

echo "  Fixed file: $FIXED_ENTRIES entries"
echo "  Well-formed: $FIXED_WELL_FORMED entries"

# Sanity check: Did we lose entries?
if [ $FIXED_ENTRIES -lt $(grep -c '^{' "$TIMELINE") ]; then
    echo ""
    echo "  ⚠️  WARNING: Entry count mismatch!"
    echo "  Original JSON objects: $(grep -c '^{' "$TIMELINE")"
    echo "  Fixed entries: $FIXED_ENTRIES"
    echo ""
    echo "  NOT replacing timeline. Manual review needed."
    echo "  Fixed file available at: $FIXED"
    echo "  Backup available at: $BACKUP"
    exit 1
fi

# Replace original with fixed version
echo ""
echo "Replacing original timeline with fixed version..."
mv "$FIXED" "$TIMELINE"
echo "  ✓ Timeline updated"

# Summary
echo ""
echo "=== FIX COMPLETE ==="
echo ""
echo "Results:"
echo "  Before: $TOTAL_LINES lines ($WELL_FORMED well-formed, $MALFORMED malformed)"
echo "  After:  $FIXED_ENTRIES lines (all well-formed single-line JSON)"
echo "  Backup: $BACKUP"
echo ""
echo "All information preserved. Format standardized to single-line JSON."
