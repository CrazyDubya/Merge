#!/bin/bash
#
# Log Rotation Script for persona-voice.log
#
# Purpose: Rotate the persona-voice log when it exceeds size threshold
# Context: CRITICAL - persona-voice.log was 50,742 lines with NO rotation
# Found by: Schizo Mode audit - unbounded growth to GB risk
#
# Usage:
#   ./rotate-persona-voice.sh              # Check size and rotate if needed
#   ./rotate-persona-voice.sh --force      # Force rotation regardless of size
#   ./rotate-persona-voice.sh --dry-run    # Show what would happen without doing it
#

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

DAEMON_ROOT="${HOME}/.claude/daemon"
LOGS_DIR="${DAEMON_ROOT}/logs"
LOG_FILE="${LOGS_DIR}/persona-voice.log"
ARCHIVE_DIR="${LOGS_DIR}/archives"
TEMP_DIR="/tmp/persona-voice-rotation-$$"

# Size threshold in MB (100MB = ~2-3 weeks of activity at high volume)
SIZE_THRESHOLD_MB=100

# Colors for output
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' NC=''
fi

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

cleanup() {
    if [ -d "$TEMP_DIR" ]; then
        log_info "Cleaning up temporary directory..."
        rm -rf "$TEMP_DIR"
    fi
}

trap cleanup EXIT INT TERM

get_file_size_mb() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "0"
        return
    fi
    local size_bytes
    size_bytes=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
    echo $((size_bytes / 1024 / 1024))
}

get_file_lines() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "0"
        return
    fi
    wc -l < "$file"
}

# ============================================================================
# MAIN ROTATION LOGIC
# ============================================================================

mkdir -p "$ARCHIVE_DIR"
mkdir -p "$TEMP_DIR"

# Parse arguments
DRY_RUN=0
FORCE_ROTATION=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --force) FORCE_ROTATION=1; shift ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    log_warning "Log file does not exist: $LOG_FILE"
    exit 0
fi

# Get current file size
CURRENT_SIZE=$(get_file_size_mb "$LOG_FILE")
CURRENT_LINES=$(get_file_lines "$LOG_FILE")

log_info "Current size: ${CURRENT_SIZE}MB (${CURRENT_LINES} lines)"

# Check if rotation is needed
if [ "$FORCE_ROTATION" -eq 0 ] && [ "$CURRENT_SIZE" -lt "$SIZE_THRESHOLD_MB" ]; then
    log_info "Size is below threshold (${SIZE_THRESHOLD_MB}MB), no rotation needed"
    exit 0
fi

if [ "$DRY_RUN" -eq 1 ]; then
    log_info "DRY RUN: Would rotate ${CURRENT_SIZE}MB log with ${CURRENT_LINES} lines"
    log_info "Archive would be: ${ARCHIVE_DIR}/persona-voice.jsonl.$(date +%Y%m%d-%H%M%S).gz"
    exit 0
fi

log_info "Rotating persona-voice.log (${CURRENT_SIZE}MB)..."

# Create backup
BACKUP_FILE="${LOG_FILE}.backup-$(date +%Y%m%d-%H%M%S)"
cp "$LOG_FILE" "$BACKUP_FILE"
log_info "Backup created: $BACKUP_FILE"

# Create archive with timestamp
ARCHIVE_FILE="${ARCHIVE_DIR}/persona-voice.log.$(date +%Y%m%d-%H%M%S).gz"

# Use flock for atomic operation
(
    if ! flock -x -w 30 200; then
        log_error "Could not acquire lock on log file"
        exit 1
    fi

    # Compress the log
    gzip -c "$LOG_FILE" > "$ARCHIVE_FILE"
    if [ ! -f "$ARCHIVE_FILE" ]; then
        log_error "Failed to create archive: $ARCHIVE_FILE"
        exit 1
    fi

    # Truncate the log file (keeps inode, same as all other rotations)
    > "$LOG_FILE"

    log_success "Log rotated: $ARCHIVE_FILE"
    log_success "Log file reset to 0 bytes"

) 200>>"${LOG_FILE}.lock"

# Verify rotation worked
NEW_SIZE=$(get_file_size_mb "$LOG_FILE")
if [ "$NEW_SIZE" -gt 0 ]; then
    log_error "Warning: Log file still has content after rotation"
    exit 1
fi

log_success "Rotation completed successfully"
log_info "Archive size: $(du -h "$ARCHIVE_FILE" | cut -f1)"

# Clean up backup if rotation succeeded
rm -f "$BACKUP_FILE"

exit 0
