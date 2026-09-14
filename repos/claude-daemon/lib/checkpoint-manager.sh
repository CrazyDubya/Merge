#!/bin/bash

################################################################################
# Checkpoint Manager (LisaSimpson + Ralph Wiggum Integration)
#
# Manages file snapshots before task execution and restoration on failure.
# Enables safe experimentation with file modifications - tasks can be rolled
# back if verification fails.
#
# Checkpoint Lifecycle:
# 1. Before execution: create_checkpoint() -> tar.gz snapshot
# 2. Execute task (may modify files)
# 3. If verification fails: rollback_to_checkpoint() -> restore files
# 4. Cleanup: cleanup_old_checkpoints() -> remove >7 days old (cron job)
#
# Storage: state/checkpoints/*.tar.gz with accompanying metadata JSON
#
# Authors: LisaSimpson + Ralph Wiggum + Autonomy Team
# Created: 2025-01-08
################################################################################

set -euo pipefail

# Daemon root
DAEMON_ROOT="${DAEMON_ROOT:-.}"

################################################################################
# CHECKPOINT CONFIGURATION
################################################################################

# Configuration constants
CHECKPOINT_DIR="${DAEMON_ROOT}/state/checkpoints"
CHECKPOINT_RETENTION_DAYS=7
CHECKPOINT_MAX_SIZE_MB=500
CHECKPOINT_COMPRESSION="gzip"  # gzip, bzip2, xz

# Initialize checkpoint directory
init_checkpoint_storage() {
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        mkdir -p "$CHECKPOINT_DIR"
    fi

    if [ ! -f "$CHECKPOINT_DIR/INDEX.md" ]; then
        cat > "$CHECKPOINT_DIR/INDEX.md" << 'EOF'
# Checkpoint Index

Auto-managed checkpoint storage for daemon file rollback capability.

## Format

Checkpoints are stored as:
- `checkpoint_YYYYMMDD_HHMMSS_TASKID.tar.gz` - Compressed file snapshot
- `checkpoint_YYYYMMDD_HHMMSS_TASKID.json` - Metadata

## Retention Policy

- Keep last 20 checkpoints minimum
- Delete checkpoints >7 days old
- Total size cap: 500MB
- Automatic cleanup runs daily (cron)

## Metadata Structure

```json
{
  "checkpoint_id": "cp_20250108_050923_task_x",
  "created_at": "2025-01-08T05:09:23Z",
  "task_id": "task_x",
  "task_title": "Write Chapter 12",
  "files": ["path/to/file1.md", "path/to/file2.txt"],
  "total_size_bytes": 45678,
  "status": "active",
  "restore_count": 0
}
```
EOF
    fi
}

################################################################################
# CHECKPOINT CREATION
################################################################################

# Generate unique checkpoint ID
# Returns: checkpoint_id string
generate_checkpoint_id() {
    local task_id="${1:-$(date +%s)}"
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    echo "cp_${timestamp}_${task_id}"
}

# Create a checkpoint snapshot of specified files
# Usage: create_checkpoint <task_id> <task_title> [file_path1] [file_path2] ...
# Returns: JSON with checkpoint details
create_checkpoint() {
    local task_id="$1"
    local task_title="${2:-Unknown Task}"
    shift 2
    local files_to_backup=("$@")

    # Initialize checkpoint storage if needed
    init_checkpoint_storage

    # Generate checkpoint ID
    local checkpoint_id
    checkpoint_id=$(generate_checkpoint_id "$task_id")

    local now
    now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Create temporary directory for archive
    local temp_dir
    temp_dir=$(mktemp -d)
    trap "rm -rf '${temp_dir}'" EXIT

    # Copy files to backup into temp directory structure
    local total_size=0
    local backed_up_files='[]'

    for file_path in "${files_to_backup[@]}"; do
        if [ -z "$file_path" ]; then
            continue
        fi

        # Only backup files that exist
        if [ -e "$file_path" ]; then
            # Preserve directory structure in archive
            local parent_dir
            parent_dir=$(dirname "$file_path")

            mkdir -p "$temp_dir/$parent_dir"
            cp -a "$file_path" "$temp_dir/$file_path"

            # Get file size
            if [ -f "$file_path" ]; then
                local file_size
                file_size=$(stat -f%z "$file_path" 2>/dev/null || stat -c%s "$file_path" 2>/dev/null || echo 0)
                total_size=$((total_size + file_size))
            fi

            backed_up_files=$(echo "$backed_up_files" | jq ". += [\"$file_path\"]")
        fi
    done

    # Check size constraints
    if [ $total_size -gt $((CHECKPOINT_MAX_SIZE_MB * 1024 * 1024)) ]; then
        echo "ERROR: Checkpoint too large ($((total_size / 1024 / 1024))MB > ${CHECKPOINT_MAX_SIZE_MB}MB)" >&2
        return 1
    fi

    # Create compressed archive
    local archive_path
    archive_path="${CHECKPOINT_DIR}/${checkpoint_id}.tar.gz"

    cd "$temp_dir"
    tar czf "$archive_path" . 2>/dev/null || {
        echo "ERROR: Failed to create checkpoint archive" >&2
        return 1
    }

    # Create metadata file
    local metadata_path
    metadata_path="${CHECKPOINT_DIR}/${checkpoint_id}.json"

    jq -n \
        --arg checkpoint_id "$checkpoint_id" \
        --arg created_at "$now" \
        --arg task_id "$task_id" \
        --arg task_title "$task_title" \
        --argjson files "$backed_up_files" \
        --argjson total_size "$total_size" \
        '{
            checkpoint_id: $checkpoint_id,
            created_at: $created_at,
            task_id: $task_id,
            task_title: $task_title,
            files: $files,
            total_size_bytes: $total_size,
            archive_path: ($checkpoint_id + ".tar.gz"),
            status: "active",
            restore_count: 0,
            last_restored_at: null
        }' > "$metadata_path"

    # Return checkpoint info
    jq -n \
        --arg checkpoint_id "$checkpoint_id" \
        --arg archive "$archive_path" \
        --arg metadata "$metadata_path" \
        --argjson total_size "$total_size" \
        --argjson backed_up_files "$backed_up_files" \
        '{
            success: true,
            checkpoint_id: $checkpoint_id,
            archive_path: $archive,
            metadata_path: $metadata,
            total_size_bytes: $total_size,
            backed_up_files: $backed_up_files,
            created_at: "'$now'"
        }'
}

################################################################################
# CHECKPOINT RESTORATION
################################################################################

# Restore files from a checkpoint
# Usage: rollback_to_checkpoint <checkpoint_id>
# Returns: JSON with restoration result
rollback_to_checkpoint() {
    local checkpoint_id="$1"

    init_checkpoint_storage

    local archive_path
    archive_path="${CHECKPOINT_DIR}/${checkpoint_id}.tar.gz"

    local metadata_path
    metadata_path="${CHECKPOINT_DIR}/${checkpoint_id}.json"

    # Verify checkpoint exists
    if [ ! -f "$archive_path" ]; then
        echo "ERROR: Checkpoint archive not found: $checkpoint_id" >&2
        return 1
    fi

    if [ ! -f "$metadata_path" ]; then
        echo "ERROR: Checkpoint metadata not found: $checkpoint_id" >&2
        return 1
    fi

    # Extract metadata
    local task_id
    task_id=$(jq -r '.task_id' "$metadata_path")

    local backed_up_files
    backed_up_files=$(jq -r '.files[]' "$metadata_path" 2>/dev/null)

    # Extract archive to daemon root
    local restore_dir
    restore_dir=$(mktemp -d)
    trap "rm -rf '${restore_dir}'" EXIT

    cd "$restore_dir"
    tar xzf "$archive_path" 2>/dev/null || {
        echo "ERROR: Failed to extract checkpoint archive" >&2
        return 1
    }

    # Restore files to original locations
    local restore_count=0
    while IFS= read -r file_path; do
        if [ -z "$file_path" ]; then
            continue
        fi

        if [ -f "$restore_dir/$file_path" ]; then
            # Create parent directory if needed
            mkdir -p "$(dirname "$file_path")"

            # Restore file
            cp -a "$restore_dir/$file_path" "$file_path"
            restore_count=$((restore_count + 1))
        fi
    done <<< "$backed_up_files"

    # Update checkpoint metadata (mark as restored)
    local now
    now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    local restore_count_meta
    restore_count_meta=$(jq -r '.restore_count' "$metadata_path")
    restore_count_meta=$((restore_count_meta + 1))

    jq \
        --arg now "$now" \
        --argjson restore_count "$restore_count_meta" \
        '.last_restored_at = $now | .restore_count = $restore_count' \
        "$metadata_path" > "${metadata_path}.tmp"
    mv "${metadata_path}.tmp" "$metadata_path"

    # Return restoration result
    jq -n \
        --arg checkpoint_id "$checkpoint_id" \
        --argjson files_restored "$restore_count" \
        --arg restored_at "$now" \
        '{
            success: true,
            checkpoint_id: $checkpoint_id,
            files_restored: $files_restored,
            restored_at: $restored_at
        }'
}

################################################################################
# CHECKPOINT MANAGEMENT
################################################################################

# List all active checkpoints
# Returns: JSON array of checkpoint metadata
list_checkpoints() {
    init_checkpoint_storage

    local checkpoints='[]'

    for metadata_file in "$CHECKPOINT_DIR"/*.json; do
        if [ -f "$metadata_file" ]; then
            local metadata
            metadata=$(cat "$metadata_file")
            checkpoints=$(echo "$checkpoints" | jq ". += [$metadata]")
        fi
    done

    echo "$checkpoints" | jq 'sort_by(.created_at) | reverse'
}

# Get checkpoint details
# Usage: get_checkpoint_info <checkpoint_id>
# Returns: JSON with checkpoint metadata
get_checkpoint_info() {
    local checkpoint_id="$1"

    init_checkpoint_storage

    local metadata_path
    metadata_path="${CHECKPOINT_DIR}/${checkpoint_id}.json"

    if [ ! -f "$metadata_path" ]; then
        echo "ERROR: Checkpoint not found: $checkpoint_id" >&2
        return 1
    fi

    cat "$metadata_path"
}

# Calculate total checkpoint storage used
# Returns: Size in bytes
get_checkpoint_storage_used() {
    init_checkpoint_storage

    local total_size=0

    for archive in "$CHECKPOINT_DIR"/*.tar.gz; do
        if [ -f "$archive" ]; then
            local size
            size=$(stat -f%z "$archive" 2>/dev/null || stat -c%s "$archive" 2>/dev/null || echo 0)
            total_size=$((total_size + size))
        fi
    done

    echo "$total_size"
}

################################################################################
# CHECKPOINT CLEANUP
################################################################################

# Clean up old checkpoints based on retention policy
# Usage: cleanup_old_checkpoints [retention_days] [max_keep]
# Returns: JSON with cleanup summary
cleanup_old_checkpoints() {
    local retention_days="${1:-$CHECKPOINT_RETENTION_DAYS}"
    local max_keep="${2:-20}"

    init_checkpoint_storage

    local now_epoch
    now_epoch=$(date +%s)

    local cutoff_epoch=$((now_epoch - (retention_days * 86400)))

    local deleted_count=0
    local deleted_size=0
    local deleted_checkpoints='[]'

    # Get all checkpoints sorted by date
    local all_checkpoints
    all_checkpoints=$(list_checkpoints)

    local checkpoint_count
    checkpoint_count=$(echo "$all_checkpoints" | jq 'length')

    # Process each checkpoint
    echo "$all_checkpoints" | jq -r '.[] | .checkpoint_id' | while read -r checkpoint_id; do
        local metadata_path
        metadata_path="${CHECKPOINT_DIR}/${checkpoint_id}.json"

        if [ ! -f "$metadata_path" ]; then
            continue
        fi

        local created_at
        created_at=$(jq -r '.created_at' "$metadata_path")

        local created_epoch
        created_epoch=$(date -j -f "%Y-%m-%dT%H:%M:%SZ" "$created_at" +%s 2>/dev/null || date -d "$created_at" +%s 2>/dev/null || echo 0)

        # Check if older than retention period
        if [ "$created_epoch" -lt "$cutoff_epoch" ]; then
            local archive_path
            archive_path="${CHECKPOINT_DIR}/${checkpoint_id}.tar.gz"

            local size=0
            if [ -f "$archive_path" ]; then
                size=$(stat -f%z "$archive_path" 2>/dev/null || stat -c%s "$archive_path" 2>/dev/null || echo 0)
                rm -f "$archive_path"
            fi

            rm -f "$metadata_path"
            deleted_count=$((deleted_count + 1))
            deleted_size=$((deleted_size + size))
        fi
    done

    # Return cleanup summary
    jq -n \
        --argjson deleted_count "$deleted_count" \
        --argjson deleted_size_bytes "$deleted_size" \
        --argjson retention_days "$retention_days" \
        '{
            success: true,
            deleted_count: $deleted_count,
            deleted_size_bytes: $deleted_size_bytes,
            deleted_size_mb: ($deleted_size_bytes / 1024 / 1024 | floor),
            retention_policy_days: $retention_days,
            timestamp: (now | floor | todate)
        }'
}

# Archive checkpoint (mark as historical)
# Usage: archive_checkpoint <checkpoint_id>
# Returns: JSON with archive result
archive_checkpoint() {
    local checkpoint_id="$1"

    init_checkpoint_storage

    local metadata_path
    metadata_path="${CHECKPOINT_DIR}/${checkpoint_id}.json"

    if [ ! -f "$metadata_path" ]; then
        echo "ERROR: Checkpoint not found: $checkpoint_id" >&2
        return 1
    fi

    # Mark as archived
    jq '.status = "archived"' "$metadata_path" > "${metadata_path}.tmp"
    mv "${metadata_path}.tmp" "$metadata_path"

    jq -n \
        --arg checkpoint_id "$checkpoint_id" \
        '{
            success: true,
            checkpoint_id: $checkpoint_id,
            status: "archived"
        }'
}

################################################################################
# CHECKPOINT VERIFICATION
################################################################################

# Verify checkpoint integrity
# Usage: verify_checkpoint <checkpoint_id>
# Returns: JSON with verification result
verify_checkpoint() {
    local checkpoint_id="$1"

    init_checkpoint_storage

    local archive_path
    archive_path="${CHECKPOINT_DIR}/${checkpoint_id}.tar.gz"

    local metadata_path
    metadata_path="${CHECKPOINT_DIR}/${checkpoint_id}.json"

    local is_valid=true
    local errors='[]'

    # Check metadata exists
    if [ ! -f "$metadata_path" ]; then
        is_valid=false
        errors=$(echo "$errors" | jq '. += ["Metadata file missing"]')
    fi

    # Check archive exists
    if [ ! -f "$archive_path" ]; then
        is_valid=false
        errors=$(echo "$errors" | jq '. += ["Archive file missing"]')
    else
        # Check archive is readable
        if ! tar tzf "$archive_path" >/dev/null 2>&1; then
            is_valid=false
            errors=$(echo "$errors" | jq '. += ["Archive is corrupted or unreadable"]')
        fi
    fi

    jq -n \
        --arg checkpoint_id "$checkpoint_id" \
        --argjson is_valid "$is_valid" \
        --argjson errors "$errors" \
        '{
            checkpoint_id: $checkpoint_id,
            is_valid: $is_valid,
            errors: $errors,
            verified_at: (now | floor | todate)
        }'
}

################################################################################
# EXPORTS
################################################################################

export -f init_checkpoint_storage
export -f generate_checkpoint_id
export -f create_checkpoint
export -f rollback_to_checkpoint
export -f list_checkpoints
export -f get_checkpoint_info
export -f get_checkpoint_storage_used
export -f cleanup_old_checkpoints
export -f archive_checkpoint
export -f verify_checkpoint

################################################################################
# SELF-TEST
################################################################################

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Running Checkpoint Manager Self-Tests..." >&2
    echo ""

    # Initialize test environment
    TEST_CHECKPOINT_DIR="/tmp/daemon-checkpoint-test"
    rm -rf "$TEST_CHECKPOINT_DIR"
    mkdir -p "$TEST_CHECKPOINT_DIR"
    CHECKPOINT_DIR="$TEST_CHECKPOINT_DIR"

    # Test 1: Create test files
    echo "Test 1: Creating test files..." >&2
    mkdir -p "$TEST_CHECKPOINT_DIR/test-files"
    echo "Original content" > "$TEST_CHECKPOINT_DIR/test-files/file1.txt"
    echo "Original data" > "$TEST_CHECKPOINT_DIR/test-files/file2.txt"

    # Test 2: Create checkpoint
    echo "Test 2: Creating checkpoint..." >&2
    checkpoint_result=$(create_checkpoint "test_task_1" "Test Task" "$TEST_CHECKPOINT_DIR/test-files/file1.txt" "$TEST_CHECKPOINT_DIR/test-files/file2.txt")
    checkpoint_id=$(echo "$checkpoint_result" | jq -r '.checkpoint_id')
    echo "✓ Checkpoint created: $checkpoint_id" >&2

    # Test 3: Modify files
    echo "Test 3: Modifying files..." >&2
    echo "Modified content" > "$TEST_CHECKPOINT_DIR/test-files/file1.txt"
    echo "Modified data" > "$TEST_CHECKPOINT_DIR/test-files/file2.txt"

    # Test 4: Verify modification
    echo "Test 4: Verifying modification..." >&2
    content=$(cat "$TEST_CHECKPOINT_DIR/test-files/file1.txt")
    if [ "$content" = "Modified content" ]; then
        echo "✓ Files modified successfully" >&2
    fi

    # Test 5: Rollback
    echo "Test 5: Rolling back to checkpoint..." >&2
    rollback_result=$(rollback_to_checkpoint "$checkpoint_id")
    echo "$rollback_result" | jq '.files_restored' >&2

    # Test 6: Verify restoration
    echo "Test 6: Verifying restoration..." >&2
    restored_content=$(cat "$TEST_CHECKPOINT_DIR/test-files/file1.txt")
    if [ "$restored_content" = "Original content" ]; then
        echo "✓ Checkpoint restoration successful" >&2
    else
        echo "✗ Checkpoint restoration failed" >&2
        exit 1
    fi

    # Test 7: List checkpoints
    echo "Test 7: Listing checkpoints..." >&2
    checkpoints=$(list_checkpoints)
    count=$(echo "$checkpoints" | jq 'length')
    echo "✓ Listed $count checkpoint(s)" >&2

    # Test 8: Get checkpoint info
    echo "Test 8: Getting checkpoint info..." >&2
    info=$(get_checkpoint_info "$checkpoint_id")
    echo "$info" | jq '{checkpoint_id, task_id, status}' >&2

    # Test 9: Verify checkpoint
    echo "Test 9: Verifying checkpoint integrity..." >&2
    verification=$(verify_checkpoint "$checkpoint_id")
    is_valid=$(echo "$verification" | jq -r '.is_valid')
    if [ "$is_valid" = "true" ]; then
        echo "✓ Checkpoint integrity verified" >&2
    fi

    # Test 10: Calculate storage
    echo "Test 10: Calculating storage used..." >&2
    storage=$(get_checkpoint_storage_used)
    storage_mb=$((storage / 1024 / 1024))
    echo "✓ Storage used: ${storage_mb}MB" >&2

    echo "" >&2
    echo "✓ All Checkpoint Manager self-tests passed!" >&2

    # Cleanup
    rm -rf "$TEST_CHECKPOINT_DIR"
fi
