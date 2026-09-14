#!/bin/bash
# rotate-persona-timeline.sh
# Rotates persona-timeline.jsonl when it exceeds size threshold
# Run via cron: 0 3 * * 0 (weekly, Sundays at 3am)
#
# CONCURRENCY SAFETY: Uses lockfile coordination to prevent race conditions
# with daemon writes. Acquires same lock that atomic_append() uses in
# lib/atomic-io.sh to ensure mutual exclusion during file replacement.

set -euo pipefail

DAEMON_ROOT="/home/opc/.claude/daemon"
TIMELINE_FILE="$DAEMON_ROOT/memory/persona-timeline.jsonl"
LOCKFILE="${TIMELINE_FILE}.lock"  # Same lock daemon uses
ARCHIVE_DIR="$DAEMON_ROOT/memory/archives"
THRESHOLD_LINES=2000  # Rotate when exceeds 2000 lines

# Check if timeline exists
if [ ! -f "$TIMELINE_FILE" ]; then
    echo "No timeline file found at $TIMELINE_FILE"
    exit 0
fi

# Count lines
LINE_COUNT=$(wc -l < "$TIMELINE_FILE")

echo "Timeline has $LINE_COUNT lines (threshold: $THRESHOLD_LINES)"

if [ "$LINE_COUNT" -lt "$THRESHOLD_LINES" ]; then
    echo "Below threshold, no rotation needed"
    exit 0
fi

# Create archive filename with timestamp
TIMESTAMP=$(date -u +"%Y%m%d-%H%M%S")
ARCHIVE_FILE="$ARCHIVE_DIR/persona-timeline-$TIMESTAMP.jsonl"

echo "Rotating timeline: $LINE_COUNT lines → $ARCHIVE_FILE"

# Copy to archive (preserve original for now)
# No lock needed - this is a read-only operation
cp "$TIMELINE_FILE" "$ARCHIVE_FILE"

# Compress archive
gzip "$ARCHIVE_FILE"
echo "Compressed: $ARCHIVE_FILE.gz"

# ============================================================================
# CRITICAL SECTION: Rotation with lockfile coordination
# ============================================================================
# Acquires same lock daemon uses (timeline.jsonl.lock) to prevent race
# condition where daemon writes during mv (would lose data to old inode).
# See: docs/ADR-002-concurrent-write-safety.md
# Bug report: inbox/daemon/unread/skeptic-rotation-race-condition-critical-20251123.md
(
    # Acquire exclusive lock (blocks daemon writes for ~100ms)
    if ! flock -x -w 10 200; then
        echo "ERROR: Failed to acquire lock for rotation (timeout after 10s)"
        echo "This may indicate daemon is hung or experiencing issues"
        exit 1
    fi

    echo "Lock acquired, rotating file..."

    # Keep only last 500 lines (safe - we hold the lock)
    tail -500 "$TIMELINE_FILE" > "$TIMELINE_FILE.tmp"

    # Replace file atomically (safe - daemon writes are blocked)
    mv "$TIMELINE_FILE.tmp" "$TIMELINE_FILE"

    echo "File rotated successfully"

    # Lock released automatically when subshell exits

) 200>>"$LOCKFILE"

# Capture exit code from flock subshell
ROTATION_STATUS=$?

if [ $ROTATION_STATUS -ne 0 ]; then
    echo "ERROR: Rotation failed (could not acquire lock)"
    echo "Timeline was archived but not rotated"
    exit 1
fi

NEW_COUNT=$(wc -l < "$TIMELINE_FILE")
ARCHIVED_COUNT=$((LINE_COUNT - NEW_COUNT))

echo "✓ Rotation complete:"
echo "  - Archived: $ARCHIVED_COUNT lines to $(basename "$ARCHIVE_FILE.gz")"
echo "  - Retained: $NEW_COUNT lines in active timeline"
echo "  - Threshold: $THRESHOLD_LINES lines"

# Update INDEX
INDEX_FILE="$ARCHIVE_DIR/INDEX.md"
if [ -f "$INDEX_FILE" ]; then
    echo "" >> "$INDEX_FILE"
    echo "## $(date -u +"%Y-%m-%d %H:%M:%S UTC") - Persona Timeline Rotation" >> "$INDEX_FILE"
    echo "- **File**: persona-timeline-$TIMESTAMP.jsonl.gz" >> "$INDEX_FILE"
    echo "- **Lines**: $ARCHIVED_COUNT archived, $NEW_COUNT retained" >> "$INDEX_FILE"
    echo "- **Reason**: Exceeded $THRESHOLD_LINES line threshold" >> "$INDEX_FILE"
fi

exit 0
