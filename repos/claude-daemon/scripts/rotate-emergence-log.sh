#!/bin/bash
#
# Log Rotation Script for emergence-log.md
#
# Purpose: Rotate the emergence log when it exceeds size threshold
# Author: Maintainer persona
# Date: 2025-10-30
#
# Usage:
#   ./rotate-emergence-log.sh              # Check size and rotate if needed
#   ./rotate-emergence-log.sh --force      # Force rotation regardless of size
#   ./rotate-emergence-log.sh --dry-run    # Show what would happen without doing it
#
# Safety Features:
#   - Creates backup before rotation
#   - Validates file integrity before and after
#   - Creates archive directory if needed
#   - Compresses old logs to save space
#   - Preserves file header in new log
#   - Never deletes data without confirmation
#
# Recovery:
#   Archives are stored in memory/archives/ with timestamps
#   To restore: gunzip archive and concatenate with current log
#

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

DAEMON_ROOT="${HOME}/.claude/daemon"
MEMORY_DIR="${DAEMON_ROOT}/memory"
LOG_FILE="${MEMORY_DIR}/emergence-log.md"
ARCHIVE_DIR="${MEMORY_DIR}/archives"
TEMP_DIR="/tmp/emergence-log-rotation-$$"

# Size threshold in KB (target: keep active log <100KB)
SIZE_THRESHOLD_KB=100

# Number of lines to keep from header (including blank lines)
HEADER_LINES=20

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

get_file_size_kb() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "0"
        return
    fi
    # Get size in KB (portable across Linux/Mac)
    local size_bytes
    size_bytes=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
    echo $((size_bytes / 1024))
}

count_entries() {
    local file="$1"
    grep -E "^## [0-9]{4}-[0-9]{2}-[0-9]{2}" "$file" | wc -l
}

validate_markdown() {
    local file="$1"

    # Check file exists and is readable
    if [ ! -f "$file" ]; then
        log_error "File not found: $file"
        return 1
    fi

    if [ ! -r "$file" ]; then
        log_error "File not readable: $file"
        return 1
    fi

    # Check file is not empty
    if [ ! -s "$file" ]; then
        log_error "File is empty: $file"
        return 1
    fi

    # Check file has the expected header
    if ! head -5 "$file" | grep -q "# Emergence Log"; then
        log_warning "File doesn't have expected header: $file"
        return 1
    fi

    return 0
}

# ============================================================================
# MAIN ROTATION LOGIC
# ============================================================================

check_rotation_needed() {
    local current_size_kb
    current_size_kb=$(get_file_size_kb "$LOG_FILE")

    log_info "Current log size: ${current_size_kb}KB (threshold: ${SIZE_THRESHOLD_KB}KB)"

    if [ "$current_size_kb" -ge "$SIZE_THRESHOLD_KB" ]; then
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

    # Validate source file
    if ! validate_markdown "$LOG_FILE"; then
        log_error "Source file validation failed. Aborting rotation."
        return 1
    fi

    local entry_count
    entry_count=$(count_entries "$LOG_FILE")
    local current_size_kb
    current_size_kb=$(get_file_size_kb "$LOG_FILE")

    log_info "Entries in current log: $entry_count"
    log_info "Current size: ${current_size_kb}KB"

    # Define archive filename
    local archive_base="emergence-log-${timestamp}.md"
    local archive_compressed="emergence-log-${timestamp}.md.gz"
    local archive_path="${ARCHIVE_DIR}/${archive_compressed}"

    if [ "$dry_run" = "true" ]; then
        log_info "[DRY RUN] Would archive current log to: $archive_path"
        log_info "[DRY RUN] Would preserve header ($HEADER_LINES lines) in new log"
        log_info "[DRY RUN] Would save ${current_size_kb}KB to archive"
        return 0
    fi

    # Step 1: Create backup in temp directory
    log_info "Creating backup..."
    cp "$LOG_FILE" "${TEMP_DIR}/backup.md"

    # Step 2: Extract header for new log
    log_info "Extracting header ($HEADER_LINES lines)..."
    head -n "$HEADER_LINES" "$LOG_FILE" > "${TEMP_DIR}/new-log.md"

    # Add rotation notice to new log
    cat >> "${TEMP_DIR}/new-log.md" << EOF

---

**Log Rotation Notice**: Previous entries archived to \`archives/${archive_compressed}\` on ${timestamp}

Previous log contained ${entry_count} entries (${current_size_kb}KB). Archives can be found in the \`memory/archives/\` directory.

---

EOF

    # Step 3: Compress current log to archive
    log_info "Compressing current log to archive..."
    gzip -c "$LOG_FILE" > "$archive_path"

    # Step 4: Validate archive
    log_info "Validating archive..."
    if ! gunzip -t "$archive_path" 2>/dev/null; then
        log_error "Archive validation failed! Aborting rotation."
        log_error "Backup preserved at: ${TEMP_DIR}/backup.md"
        return 1
    fi

    # Step 5: Replace current log with new log
    # CONCURRENCY SAFETY: Use lockfile coordination to prevent race conditions
    # with persona writes. Acquires exclusive lock to ensure no persona writes
    # during file replacement (prevents data loss to old inode).
    # See: docs/ADR-002-concurrent-write-safety.md
    # Reference: rotate-persona-timeline.sh (lockfile pattern)
    log_info "Installing new log file..."

    local lockfile="${LOG_FILE}.lock"
    (
        # Acquire exclusive lock (blocks persona writes for ~100ms)
        if ! flock -x -w 10 200; then
            log_error "Failed to acquire lock for rotation (timeout after 10s)"
            log_error "This may indicate a persona is writing or system issues"
            return 1
        fi

        log_info "Lock acquired, replacing file..."

        # Replace file atomically (safe - we hold the lock)
        mv "${TEMP_DIR}/new-log.md" "$LOG_FILE"

        log_info "File replaced successfully"

        # Lock released automatically when subshell exits

    ) 200>>"$lockfile"

    # Capture exit code from flock subshell
    local mv_status=$?

    if [ $mv_status -ne 0 ]; then
        log_error "File replacement failed (could not acquire lock)"
        log_error "Backup preserved at: ${TEMP_DIR}/backup.md"
        return 1
    fi

    # Step 6: Verify new log
    if ! validate_markdown "$LOG_FILE"; then
        log_error "New log validation failed! Restoring from backup..."
        cp "${TEMP_DIR}/backup.md" "$LOG_FILE"
        return 1
    fi

    local new_size_kb
    new_size_kb=$(get_file_size_kb "$LOG_FILE")

    log_success "Rotation complete!"
    log_success "Archive created: $archive_path"
    log_success "New log size: ${new_size_kb}KB (was ${current_size_kb}KB)"
    log_success "Space saved: $((current_size_kb - new_size_kb))KB"

    # Create index file in archives directory
    local index_file="${ARCHIVE_DIR}/INDEX.md"
    if [ ! -f "$index_file" ]; then
        cat > "$index_file" << 'EOF'
# Emergence Log Archives

This directory contains archived emergence log entries. Each archive is compressed with gzip.

## Archives

EOF
    fi

    # Add entry to index
    echo "- \`${archive_compressed}\` - Created ${timestamp}, ${entry_count} entries, ${current_size_kb}KB original size" >> "$index_file"

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
                sed -n '2,26p' "$0" | sed 's/^# //;s/^#//'
                exit 0
                ;;
            *)
                log_error "Unknown argument: $arg"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    log_info "===== Emergence Log Rotation Script ====="
    log_info "Log file: $LOG_FILE"

    # Check if log file exists
    if [ ! -f "$LOG_FILE" ]; then
        log_error "Log file not found: $LOG_FILE"
        exit 1
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
