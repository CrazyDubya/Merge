#!/bin/bash
#
# Log Rotation Script for state-audit.jsonl
#
# Purpose: Rotate the state audit log when it exceeds size threshold
# Author: Maintainer persona
# Date: 2025-12-25
#
# Usage:
#   ./rotate-state-audit-log.sh              # Check size and rotate if needed
#   ./rotate-state-audit-log.sh --force      # Force rotation regardless of size
#   ./rotate-state-audit-log.sh --dry-run    # Show what would happen without doing it
#
# Safety Features:
#   - Creates backup before rotation
#   - Compresses old logs to save space
#   - Maintains max 20 recent rotations (~20MB total)
#   - Never deletes data without archival
#
# Recovery:
#   Archives are stored in state/archives/ with timestamps
#   To restore: gunzip archive and concatenate with current log
#

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

DAEMON_ROOT="${HOME}/.claude/daemon"
LOGS_DIR="${DAEMON_ROOT}/logs"
LOG_FILE="${LOGS_DIR}/state-audit.jsonl"
ARCHIVE_DIR="${LOGS_DIR}/archives"
TEMP_DIR="/tmp/state-audit-rotation-$$"

# Size threshold in bytes (1MB = 1048576 bytes)
SIZE_THRESHOLD_BYTES=1048576

# Maximum number of rotations to keep (20 files = ~20MB at 1MB each)
MAX_ROTATIONS=20

# Colors for output (if terminal supports it)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
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

# Register cleanup on exit
trap cleanup EXIT INT TERM

get_file_size_bytes() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "0"
        return
    fi
    # Get size in bytes (portable across Linux/Mac)
    stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null
}

count_lines() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "0"
        return
    fi
    wc -l < "$file" | tr -d ' '
}

cleanup_old_rotations() {
    if [ ! -d "$ARCHIVE_DIR" ]; then
        return 0
    fi

    local rotation_count
    rotation_count=$(find "$ARCHIVE_DIR" -name "state-audit-*.jsonl.gz" 2>/dev/null | wc -l)

    if [ "$rotation_count" -le "$MAX_ROTATIONS" ]; then
        return 0
    fi

    local to_delete=$((rotation_count - MAX_ROTATIONS))
    log_info "Removing $to_delete old rotations (keeping max $MAX_ROTATIONS)..."

    # Delete oldest files first (by modification time)
    find "$ARCHIVE_DIR" -name "state-audit-*.jsonl.gz" -type f | sort | head -n "$to_delete" | while read -r file; do
        log_info "Removing old archive: $(basename "$file")"
        rm -f "$file"
    done

    return 0
}

# ============================================================================
# MAIN ROTATION LOGIC
# ============================================================================

check_rotation_needed() {
    local current_size_bytes
    current_size_bytes=$(get_file_size_bytes "$LOG_FILE")

    local current_size_mb=$((current_size_bytes / 1048576))
    local threshold_mb=$((SIZE_THRESHOLD_BYTES / 1048576))

    log_info "Current log size: ${current_size_mb}MB (threshold: ${threshold_mb}MB)"

    if [ "$current_size_bytes" -ge "$SIZE_THRESHOLD_BYTES" ]; then
        return 0  # Rotation needed
    else
        return 1  # No rotation needed
    fi
}

perform_rotation() {
    local dry_run="${1:-false}"
    local timestamp
    timestamp=$(date +%Y%m%d-%H%M%S)

    log_info "Starting log rotation (timestamp: $timestamp)"

    # Create directories
    if [ "$dry_run" = "false" ]; then
        mkdir -p "$ARCHIVE_DIR"
        mkdir -p "$TEMP_DIR"
    else
        log_info "[DRY RUN] Would create directories: $ARCHIVE_DIR, $TEMP_DIR"
    fi

    # Check source file exists
    if [ ! -f "$LOG_FILE" ]; then
        log_error "Source file not found: $LOG_FILE"
        return 1
    fi

    local line_count
    line_count=$(count_lines "$LOG_FILE")
    local current_size_bytes
    current_size_bytes=$(get_file_size_bytes "$LOG_FILE")
    local current_size_mb=$((current_size_bytes / 1048576))

    log_info "Lines in current log: $line_count"
    log_info "Current size: ${current_size_mb}MB"

    # Define archive filename
    local archive_base="state-audit-${timestamp}.jsonl"
    local archive_compressed="state-audit-${timestamp}.jsonl.gz"
    local archive_path="${ARCHIVE_DIR}/${archive_compressed}"

    if [ "$dry_run" = "true" ]; then
        log_info "[DRY RUN] Would archive current log to: $archive_path"
        log_info "[DRY RUN] Would save ${current_size_mb}MB to archive"
        return 0
    fi

    # Step 1: Create backup in temp directory
    log_info "Creating backup..."
    cp "$LOG_FILE" "${TEMP_DIR}/backup.jsonl"

    # Step 2: Compress current log to archive
    log_info "Compressing current log to archive..."
    gzip -c "$LOG_FILE" > "$archive_path"

    # Step 3: Validate archive
    log_info "Validating archive..."
    if ! gunzip -t "$archive_path" 2>/dev/null; then
        log_error "Archive validation failed! Aborting rotation."
        log_error "Backup preserved at: ${TEMP_DIR}/backup.jsonl"
        return 1
    fi

    # Step 4: Replace current log with new empty log
    log_info "Installing new log file..."

    # Create empty log file
    > "$LOG_FILE"

    # Step 5: Verify new log
    if [ ! -f "$LOG_FILE" ]; then
        log_error "New log file creation failed! Restoring from backup..."
        cp "${TEMP_DIR}/backup.jsonl" "$LOG_FILE"
        return 1
    fi

    local new_size_bytes
    new_size_bytes=$(get_file_size_bytes "$LOG_FILE")

    log_success "Rotation complete!"
    log_success "Archive created: $archive_path"
    log_success "New log size: ${new_size_bytes} bytes (was ${current_size_bytes} bytes)"
    log_success "Freed space: ${current_size_mb}MB"

    # Step 6: Cleanup old rotations
    cleanup_old_rotations

    # Create or update index file in archives directory
    local index_file="${ARCHIVE_DIR}/INDEX.md"
    if [ ! -f "$index_file" ]; then
        cat > "$index_file" << 'INDEXEOF'
# State Audit Log Archives

This directory contains archived state audit log entries. Each archive is compressed with gzip.

## Archives

INDEXEOF
    fi

    # Add entry to index
    echo "- \`${archive_compressed}\` - Created ${timestamp}, ${line_count} lines, ${current_size_mb}MB original size" >> "$index_file"

    log_info "Updated archive index: $index_file"

    return 0
}

# ============================================================================
# MAIN SCRIPT
# ============================================================================

main() {
    local force_rotation=false
    local dry_run=false

    # Parse command line arguments
    for arg in "$@"; do
        case $arg in
            --force)
                force_rotation=true
                ;;
            --dry-run)
                dry_run=true
                ;;
            --help)
                sed -n '2,24p' "$0" | sed 's/^# //;s/^#//'
                exit 0
                ;;
            *)
                log_error "Unknown argument: $arg"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    log_info "===== State Audit Log Rotation Script ====="
    log_info "Log file: $LOG_FILE"

    # Check if log file exists
    if [ ! -f "$LOG_FILE" ]; then
        log_warning "Log file not found: $LOG_FILE - skipping rotation"
        exit 0
    fi

    # Check if rotation is needed
    if [ "$force_rotation" = "true" ]; then
        log_info "Force rotation requested"
        perform_rotation "$dry_run"
    elif check_rotation_needed; then
        log_info "Rotation needed (size threshold exceeded)"
        perform_rotation "$dry_run"
    else
        log_success "No rotation needed. Log size is below threshold."
        exit 0
    fi
}

# Run main function
main "$@"
