#!/bin/bash
#
# Inter-Persona Dialogue Rotation Script
# Archives old dialogue entries while preserving recent communication
#
# Usage:
#   ./rotate-inter-persona-dialogue.sh [--keep-entries N] [--dry-run]
#
# Optimization rationale:
#   inter-persona-dialogue.md was 13,728 words (2,618 lines, 50 entries)
#   Like task queue, this mixes "active communication" with "historical log"
#   Archive old entries to reduce context bloat while preserving recent dialogue

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
DIALOGUE_FILE="${DAEMON_ROOT}/memory/inter-persona-dialogue.md"
ARCHIVE_DIR="${DAEMON_ROOT}/memory/archives"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ARCHIVE_FILE="${ARCHIVE_DIR}/inter-persona-dialogue-${TIMESTAMP}.md"

# Default: keep most recent 20 entries
KEEP_ENTRIES=20
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep-entries)
            KEEP_ENTRIES="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Usage: $0 [--keep-entries N] [--dry-run]"
            exit 1
            ;;
    esac
done

echo "=== Inter-Persona Dialogue Rotation ==="
echo "Dialogue file: $DIALOGUE_FILE"
echo "Archive to: $ARCHIVE_FILE"
echo "Keep recent: $KEEP_ENTRIES entries"
echo "Dry run: $DRY_RUN"
echo ""

# Check if dialogue file exists
if [ ! -f "$DIALOGUE_FILE" ]; then
    echo "Error: Dialogue file not found: $DIALOGUE_FILE"
    exit 1
fi

# Create archive directory if needed
mkdir -p "$ARCHIVE_DIR"

# Count current entries (sections starting with ##)
TOTAL_ENTRIES=$(grep -c "^## " "$DIALOGUE_FILE" || true)

echo "[1/5] Current state:"
echo "  Total entries: $TOTAL_ENTRIES"

if [ "$TOTAL_ENTRIES" -le "$KEEP_ENTRIES" ]; then
    echo ""
    echo "No rotation needed: Only $TOTAL_ENTRIES entries (threshold: $KEEP_ENTRIES)"
    exit 0
fi

ENTRIES_TO_ARCHIVE=$((TOTAL_ENTRIES - KEEP_ENTRIES))
echo "  Entries to archive: $ENTRIES_TO_ARCHIVE"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "(Dry run - no changes will be made)"
    echo ""
    echo "Would archive $ENTRIES_TO_ARCHIVE oldest entries"
    echo "Would keep $KEEP_ENTRIES most recent entries"
    exit 0
fi

# Backup original
BACKUP_FILE="${DIALOGUE_FILE}.backup-${TIMESTAMP}"
cp "$DIALOGUE_FILE" "$BACKUP_FILE"
echo "[2/5] Backup created: $BACKUP_FILE"

# Extract file header (everything before first ##)
echo "[3/5] Extracting header and entries..."
HEADER_LINES=$(grep -n "^## " "$DIALOGUE_FILE" | head -1 | cut -d: -f1)
HEADER_LINES=$((HEADER_LINES - 1))

head -n "$HEADER_LINES" "$DIALOGUE_FILE" > "${DIALOGUE_FILE}.tmp"

# Find line numbers of all entry headers
grep -n "^## " "$DIALOGUE_FILE" | awk -F: '{print $1}' > /tmp/entry_lines.txt

# Get line number where we should split (keep recent entries)
SPLIT_LINE=$(grep -n "^## " "$DIALOGUE_FILE" | tail -n "$KEEP_ENTRIES" | head -1 | cut -d: -f1)

# Create archive with old entries
{
    echo "# Archived Inter-Persona Dialogue"
    echo ""
    echo "Archived from: memory/inter-persona-dialogue.md"
    echo "Archive date: $(date -Iseconds)"
    echo "Entries archived: $ENTRIES_TO_ARCHIVE (kept $KEEP_ENTRIES most recent)"
    echo ""
    echo "---"
    echo ""
    
    # Extract old entries (from first entry to split point)
    FIRST_ENTRY_LINE=$(grep -n "^## " "$DIALOGUE_FILE" | head -1 | cut -d: -f1)
    sed -n "${FIRST_ENTRY_LINE},$((SPLIT_LINE - 1))p" "$DIALOGUE_FILE"
} > "$ARCHIVE_FILE"

echo "  ✓ Archived $ENTRIES_TO_ARCHIVE entries to: $ARCHIVE_FILE"

# Create new dialogue file with header + recent entries
echo "[4/5] Creating rotated dialogue file..."
{
    cat "${DIALOGUE_FILE}.tmp"
    echo ""
    echo "---"
    echo ""
    echo "**Archive Notice**: Older entries archived to \`archives/inter-persona-dialogue-${TIMESTAMP}.md\` on ${TIMESTAMP}"
    echo ""
    echo "Previous log contained $ENTRIES_TO_ARCHIVE entries. Recent ${KEEP_ENTRIES} entries preserved below."
    echo ""
    echo "---"
    echo ""
    
    # Append recent entries
    tail -n +${SPLIT_LINE} "$DIALOGUE_FILE"
} > "${DIALOGUE_FILE}.new"

# Replace original with rotated version
# CONCURRENCY SAFETY: Use lockfile coordination to prevent race conditions
# with persona writes. Acquires exclusive lock to ensure no persona writes
# during file replacement (prevents data loss to old inode).
# See: docs/ADR-002-concurrent-write-safety.md
# Reference: rotate-persona-timeline.sh (lockfile pattern)

LOCKFILE="${DIALOGUE_FILE}.lock"
(
    # Acquire exclusive lock (blocks persona writes for ~100ms)
    if ! flock -x -w 10 200; then
        echo "ERROR: Failed to acquire lock for rotation (timeout after 10s)"
        echo "This may indicate a persona is writing or system issues"
        exit 1
    fi

    echo "Lock acquired, replacing file..."

    # Replace file atomically (safe - we hold the lock)
    mv "${DIALOGUE_FILE}.new" "$DIALOGUE_FILE"

    echo "File replaced successfully"

    # Lock released automatically when subshell exits

) 200>>"$LOCKFILE"

# Capture exit code from flock subshell
ROTATION_STATUS=$?

if [ $ROTATION_STATUS -ne 0 ]; then
    echo "ERROR: Rotation failed (could not acquire lock)"
    echo "Backup preserved at: $BACKUP_FILE"
    exit 1
fi

rm "${DIALOGUE_FILE}.tmp"

echo "  ✓ Created rotated dialogue with $KEEP_ENTRIES recent entries"

# Measure impact
echo ""
echo "[5/5] Measuring impact..."
OLD_LINES=$(wc -l < "$BACKUP_FILE")
NEW_LINES=$(wc -l < "$DIALOGUE_FILE")
OLD_WORDS=$(wc -w < "$BACKUP_FILE")
NEW_WORDS=$(wc -w < "$DIALOGUE_FILE")
ARCHIVE_LINES=$(wc -l < "$ARCHIVE_FILE")

LINE_REDUCTION=$((OLD_LINES - NEW_LINES))
WORD_REDUCTION=$((OLD_WORDS - NEW_WORDS))
LINE_REDUCTION_PCT=$((LINE_REDUCTION * 100 / OLD_LINES))
WORD_REDUCTION_PCT=$((WORD_REDUCTION * 100 / OLD_WORDS))

echo "  Before: $OLD_LINES lines, $OLD_WORDS words"
echo "  After:  $NEW_LINES lines, $NEW_WORDS words"
echo "  Reduction: $LINE_REDUCTION lines (${LINE_REDUCTION_PCT}%), $WORD_REDUCTION words (${WORD_REDUCTION_PCT}%)"
echo ""
echo "  Estimated token savings: ~$((WORD_REDUCTION * 4 / 3)) tokens per read"
echo ""

echo "=== ROTATION COMPLETE ==="
echo ""
echo "Files:"
echo "  Dialogue: $DIALOGUE_FILE"
echo "  Archive: $ARCHIVE_FILE"
echo "  Backup: $BACKUP_FILE"
echo ""
echo "Performance impact: Reduced dialogue context by ~${WORD_REDUCTION_PCT}%"
