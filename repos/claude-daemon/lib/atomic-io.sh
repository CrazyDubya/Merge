#!/bin/bash
#
# Atomic I/O Operations Library
#
# Provides safe concurrent access to shared files using flock for coordination.
# Part of ADR-002: Concurrent Write Safety Strategy.
#
# Usage:
#   source lib/atomic-io.sh
#   atomic_append "$file" "$content"
#
# See: docs/ADR-002-concurrent-write-safety.md

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Default lock timeout (seconds)
ATOMIC_IO_TIMEOUT=${ATOMIC_IO_TIMEOUT:-5}

# Enable debug logging (set ATOMIC_IO_DEBUG=1 for troubleshooting)
ATOMIC_IO_DEBUG=${ATOMIC_IO_DEBUG:-0}

# ============================================================================
# Internal Functions
# ============================================================================

_atomic_io_log() {
    if [ "$ATOMIC_IO_DEBUG" -eq 1 ]; then
        echo "[atomic-io] $*" >&2
    fi
}

_atomic_io_error() {
    echo "[atomic-io] ERROR: $*" >&2
}

# ============================================================================
# Public API
# ============================================================================

# atomic_append FILE CONTENT [TIMEOUT]
#
# Atomically append content to file with exclusive lock.
#
# Arguments:
#   FILE     - Path to file (will be created if missing)
#   CONTENT  - String to append (single line)
#   TIMEOUT  - Lock timeout in seconds (default: 5)
#
# Returns:
#   0 - Success
#   1 - Lock timeout
#   2 - Write failure
#   3 - Invalid arguments
#
# Example:
#   atomic_append "/var/log/audit.jsonl" "$json_entry"
#   atomic_append "/var/log/audit.jsonl" "$json_entry" 10  # 10s timeout
#
atomic_append() {
    local file="$1"
    local content="$2"
    local timeout="${3:-$ATOMIC_IO_TIMEOUT}"

    # Validate arguments
    if [ -z "$file" ]; then
        _atomic_io_error "atomic_append: FILE argument required"
        return 3
    fi

    if [ -z "$content" ]; then
        _atomic_io_error "atomic_append: CONTENT argument required"
        return 3
    fi

    # Ensure parent directory exists
    local dir
    dir="$(dirname "$file")"
    if [ ! -d "$dir" ]; then
        _atomic_io_log "Creating directory: $dir"
        mkdir -p "$dir" 2>/dev/null || {
            _atomic_io_error "Failed to create directory: $dir"
            return 2
        }
    fi

    # Lockfile path (sibling to data file)
    local lockfile="${file}.lock"

    _atomic_io_log "Acquiring lock on $lockfile (timeout: ${timeout}s)"

    # Acquire exclusive lock and write atomically
    (
        # flock -x: exclusive lock (no other readers or writers)
        # flock -w: wait with timeout
        # 200: arbitrary file descriptor for lock
        if ! flock -x -w "$timeout" 200; then
            _atomic_io_error "Lock timeout on $file (${timeout}s expired)"
            return 1
        fi

        _atomic_io_log "Lock acquired, writing to $file"

        # Append content while holding lock
        if ! echo "$content" >> "$file"; then
            _atomic_io_error "Write failed to $file"
            return 2
        fi

        _atomic_io_log "Write successful, releasing lock"

        # Lock released automatically when subshell exits
        return 0

    ) 200>>"$lockfile"

    # Capture subshell exit code
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        _atomic_io_log "atomic_append completed successfully"
    else
        _atomic_io_error "atomic_append failed with code $exit_code"
    fi

    return $exit_code
}

# atomic_append_batch FILE LINE1 [LINE2 LINE3 ...]
#
# Atomically append multiple lines to file with single lock acquisition.
#
# More efficient than multiple atomic_append calls when writing many lines.
#
# Arguments:
#   FILE     - Path to file
#   LINE1..N - Lines to append
#
# Returns:
#   0 - Success
#   1 - Lock timeout
#   2 - Write failure
#   3 - Invalid arguments
#
# Example:
#   atomic_append_batch "/var/log/metrics.jsonl" \
#       '{"metric":"cpu","value":45}' \
#       '{"metric":"mem","value":78}' \
#       '{"metric":"disk","value":23}'
#
atomic_append_batch() {
    local file="$1"
    shift

    local lines=("$@")
    local timeout="${ATOMIC_IO_TIMEOUT}"

    # Validate arguments
    if [ -z "$file" ]; then
        _atomic_io_error "atomic_append_batch: FILE argument required"
        return 3
    fi

    if [ ${#lines[@]} -eq 0 ]; then
        _atomic_io_error "atomic_append_batch: At least one line required"
        return 3
    fi

    # Ensure parent directory exists
    local dir
    dir="$(dirname "$file")"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir" 2>/dev/null || {
            _atomic_io_error "Failed to create directory: $dir"
            return 2
        }
    fi

    local lockfile="${file}.lock"

    _atomic_io_log "Batch write: ${#lines[@]} lines to $file"

    # Acquire lock once, write all lines
    (
        if ! flock -x -w "$timeout" 200; then
            _atomic_io_error "Lock timeout on $file"
            return 1
        fi

        _atomic_io_log "Lock acquired for batch write"

        # Write all lines atomically
        for line in "${lines[@]}"; do
            if ! echo "$line" >> "$file"; then
                _atomic_io_error "Batch write failed at line: $line"
                return 2
            fi
        done

        _atomic_io_log "Batch write successful (${#lines[@]} lines)"
        return 0

    ) 200>>"$lockfile"

    return $?
}

# atomic_read FILE
#
# Read file with shared lock (ensures no partial reads during writes).
#
# Multiple readers can read simultaneously, but writers are blocked.
#
# Arguments:
#   FILE - Path to file
#
# Returns:
#   0 - Success (file contents written to stdout)
#   1 - File not found
#   2 - Read failure
#
# Example:
#   atomic_read "/var/log/audit.jsonl" | jq '.operation'
#
atomic_read() {
    local file="$1"

    # Validate arguments
    if [ -z "$file" ]; then
        _atomic_io_error "atomic_read: FILE argument required"
        return 3
    fi

    if [ ! -f "$file" ]; then
        _atomic_io_error "File not found: $file"
        return 1
    fi

    local lockfile="${file}.lock"

    _atomic_io_log "Reading $file with shared lock"

    # Acquire shared lock (multiple readers OK, blocks writers)
    (
        # flock -s: shared lock (multiple readers allowed)
        if ! flock -s 200; then
            _atomic_io_error "Failed to acquire read lock on $file"
            return 2
        fi

        _atomic_io_log "Shared lock acquired, reading"

        # Read while holding lock
        if ! cat "$file"; then
            _atomic_io_error "Read failed from $file"
            return 2
        fi

        return 0

    ) 200>>"$lockfile"

    return $?
}

# atomic_truncate FILE
#
# Atomically truncate (empty) file with exclusive lock.
#
# Useful for log rotation or reset operations.
#
# Arguments:
#   FILE - Path to file
#
# Returns:
#   0 - Success
#   1 - Lock timeout
#   2 - Truncate failure
#   3 - Invalid arguments
#
# Example:
#   atomic_truncate "/var/log/temp.log"
#
atomic_truncate() {
    local file="$1"
    local timeout="${ATOMIC_IO_TIMEOUT}"

    if [ -z "$file" ]; then
        _atomic_io_error "atomic_truncate: FILE argument required"
        return 3
    fi

    # File doesn't need to exist for truncate
    if [ ! -f "$file" ]; then
        _atomic_io_log "File doesn't exist, nothing to truncate: $file"
        return 0
    fi

    local lockfile="${file}.lock"

    _atomic_io_log "Truncating $file"

    (
        if ! flock -x -w "$timeout" 200; then
            _atomic_io_error "Lock timeout on $file"
            return 1
        fi

        if ! : > "$file"; then
            _atomic_io_error "Truncate failed on $file"
            return 2
        fi

        _atomic_io_log "Truncate successful"
        return 0

    ) 200>>"$lockfile"

    return $?
}

# atomic_file_operation FILE COMMAND [ARGS...]
#
# Execute arbitrary command on file with exclusive lock.
#
# Advanced usage for custom operations not covered by other functions.
#
# Arguments:
#   FILE    - Path to file
#   COMMAND - Command to execute while holding lock
#   ARGS    - Arguments to command
#
# Returns:
#   Exit code from COMMAND
#
# Example:
#   atomic_file_operation "/var/log/audit.jsonl" jq '.operation' -
#   atomic_file_operation "/var/log/metrics.jsonl" tail -n 100
#
atomic_file_operation() {
    local file="$1"
    shift
    local command="$1"
    shift
    local args=("$@")
    local timeout="${ATOMIC_IO_TIMEOUT}"

    if [ -z "$file" ]; then
        _atomic_io_error "atomic_file_operation: FILE argument required"
        return 3
    fi

    if [ -z "$command" ]; then
        _atomic_io_error "atomic_file_operation: COMMAND argument required"
        return 3
    fi

    local lockfile="${file}.lock"

    _atomic_io_log "Executing: $command ${args[*]}"

    (
        if ! flock -x -w "$timeout" 200; then
            _atomic_io_error "Lock timeout on $file"
            return 1
        fi

        # Execute command while holding lock
        "$command" "$file" "${args[@]}"
        return $?

    ) 200>>"$lockfile"

    return $?
}

# ============================================================================
# Cleanup Utility
# ============================================================================

# atomic_io_cleanup_locks [DIRECTORY]
#
# Remove stale lockfiles (from crashed processes).
#
# Lockfiles are normally removed automatically, but crashes can leave them.
# This is safe to run periodically (e.g., in maintenance scripts).
#
# Arguments:
#   DIRECTORY - Directory to scan (default: current directory)
#
# Returns:
#   0 - Success
#
# Example:
#   atomic_io_cleanup_locks "/var/log"
#   atomic_io_cleanup_locks  # defaults to current directory
#
atomic_io_cleanup_locks() {
    local dir="${1:-.}"

    _atomic_io_log "Scanning for stale lockfiles in $dir"

    local count=0
    while IFS= read -r -d '' lockfile; do
        # Check if lockfile is actually locked
        # If flock succeeds immediately, lockfile is stale
        if flock -x -w 0 200 2>/dev/null; then
            _atomic_io_log "Removing stale lockfile: $lockfile"
            rm -f "$lockfile"
            ((count++))
        else
            _atomic_io_log "Lockfile in use (skipping): $lockfile"
        fi 200>>"$lockfile"
    done < <(find "$dir" -name "*.lock" -type f -print0)

    if [ $count -gt 0 ]; then
        echo "Cleaned up $count stale lockfile(s)"
    else
        _atomic_io_log "No stale lockfiles found"
    fi

    return 0
}

# ============================================================================
# Main (for testing)
# ============================================================================

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    # Script executed directly (not sourced)
    cat <<'EOF'
Atomic I/O Library
==================

This library provides concurrent-safe file operations using flock.

Usage:
  source lib/atomic-io.sh
  atomic_append "$file" "$content"

Functions:
  atomic_append FILE CONTENT [TIMEOUT]     - Append line atomically
  atomic_append_batch FILE LINE1 LINE2...  - Append multiple lines
  atomic_read FILE                         - Read with shared lock
  atomic_truncate FILE                     - Truncate atomically
  atomic_file_operation FILE CMD ARGS...   - Custom operations
  atomic_io_cleanup_locks [DIR]            - Remove stale locks

Environment:
  ATOMIC_IO_TIMEOUT - Default lock timeout (default: 5 seconds)
  ATOMIC_IO_DEBUG   - Enable debug logging (set to 1)

Documentation:
  docs/ADR-002-concurrent-write-safety.md

Examples:
  # Append to audit log
  atomic_append "/var/log/audit.jsonl" '{"operation":"test"}'

  # Batch write metrics
  atomic_append_batch "/var/log/metrics.jsonl" \
      '{"metric":"cpu","value":45}' \
      '{"metric":"mem","value":78}'

  # Safe read during writes
  atomic_read "/var/log/audit.jsonl" | jq '.'

  # Cleanup stale locks
  atomic_io_cleanup_locks "/var/log"

EOF
fi
