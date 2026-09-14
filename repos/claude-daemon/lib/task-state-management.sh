#!/bin/bash
# Task State Management Library
# Created by Architect persona for task routing improvements
#
# This library extends queue.md task management to support:
# - In-progress task tracking with [~]
# - Persona-task compatibility matching
# - Task state transitions with ownership tracking

# Escape special regex characters for sed pattern matching
# Consolidates 8 individual sed pipes into a single sed call
# Characters escaped: \ [ ] . * ^ $ /
# Usage: escaped=$(escape_for_sed "$raw_string")
escape_for_sed() {
    printf "%s" "$1" | sed 's/[\\[\/.*^$]/\\&/g; s/\]/\\]/g'
}

# Mark task as in-progress
# Usage: mark_task_in_progress "$task_description" "$persona_name"
mark_task_in_progress() {
    local task="$1"
    local persona="$2"
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # Create backup
    local backup_file="${TASKS_DIR}/queue.md.backup-$(date +%Y%m%d-%H%M%S)"
    if ! cp "$TASKS_DIR/queue.md" "$backup_file"; then
        echo "ERROR: Failed to create backup of queue.md" >&2
        return 1
    fi

    local escaped_task
    escaped_task=$(escape_for_sed "$task")

    # Mark task as in-progress with metadata
    local temp_file
    temp_file=$(mktemp)

    # Ensure temp file is cleaned up on exit (trap-based cleanup)
    trap "rm -f '$temp_file'" RETURN

    sed "0,/^- \[ \] ${escaped_task}/s//- [~] ${escaped_task} (in-progress: $persona, started: $timestamp)/" \
        "$TASKS_DIR/queue.md" > "$temp_file"

    # Validation
    local original_size=$(wc -l < "$TASKS_DIR/queue.md")
    local new_size=$(wc -l < "$temp_file")
    local min_size=$((original_size * 80 / 100))

    if [ "$new_size" -lt "$min_size" ]; then
        echo "ERROR: mark_task_in_progress produced corrupted file" >&2
        return 1
    fi

    # Apply changes
    mv "$temp_file" "$TASKS_DIR/queue.md"
    log "INFO" "Task marked in-progress: $task (persona: $persona)"

    return 0
}

# Extract persona tag from task description
# Usage: persona=$(extract_persona_tag "$task_line")
# Returns: persona name (lowercase) or empty string if no tag
extract_persona_tag() {
    local task_line="$1"

    # Match [PERSONA] at start of task description (case-insensitive)
    if echo "$task_line" | grep -qiE '^\[([A-Za-z]+)\]'; then
        echo "$task_line" | sed -E 's/^.*\[([A-Za-z]+)\].*/\1/' | tr '[:upper:]' '[:lower:]'
    else
        echo ""
    fi
}

# Get next task that matches current persona
# Usage: task=$(get_next_task_for_persona "$persona_name")
# Falls back to untagged tasks if no persona-specific tasks found
get_next_task_for_persona() {
    local current_persona="$1"

    # Get all incomplete tasks (both [ ] and [~] are considered incomplete for assignment)
    # MAINTAINER FIX: Updated to match actual queue.md sections (ACTIVE, In Progress, etc)
    local all_tasks=$(sed -n '/^## \(ACTIVE\|In Progress\)/,/^## \(Examples\|Task Format\|Completed\|Completed Tasks\)/p' "$TASKS_DIR/queue.md" | \
                      grep "^- \[\( \|~\)\]")

    # Phase 5: First pass - Use cross-persona assignment logic (PRIMARY/BACKUP support)
    # Check if persona is eligible (primary, backup, or untagged)
    while IFS= read -r task_line; do
        if [ -z "$task_line" ]; then
            continue
        fi

        # Extract task description (everything after "- [ ] " or "- [~] ")
        local task_desc=$(echo "$task_line" | sed 's/^- \[\( \|~\)\] //')

        # Phase 5: Use can_persona_take_task() for PRIMARY/BACKUP assignment logic
        # This supports: [PRIMARY:persona,BACKUP:persona], [persona], and untagged
        if can_persona_take_task "$current_persona" "$task_line"; then
            # Return the task line (persona is eligible)
            echo "$task_line"
            return 0
        fi
    done <<< "$all_tasks"

    # Second pass: Look for untagged tasks (available to anyone) - fallback
    while IFS= read -r task_line; do
        if [ -z "$task_line" ]; then
            continue
        fi

        local task_desc=$(echo "$task_line" | sed 's/^- \[\( \|~\)\] //')
        local task_persona=$(extract_persona_tag "$task_desc")

        # Match if no persona tag (untagged tasks anyone can do)
        if [ -z "$task_persona" ]; then
            echo "$task_line"
            return 0
        fi
    done <<< "$all_tasks"

    # No matching tasks found
    echo ""
    return 1
}

# Check if current persona has in-progress tasks
# Usage: has_in_progress_work "$persona_name"
# Returns: 0 (true) if persona has [~] tasks, 1 (false) otherwise
has_in_progress_work() {
    local persona="$1"

    # Check if TASKS_DIR is set (defensive check, fixes integration bug)
    if [ -z "${TASKS_DIR:-}" ]; then
        return 1
    fi

    if [ ! -f "$TASKS_DIR/queue.md" ]; then
        return 1
    fi

    # Check for any [~] tasks belonging to this persona
    # Note: The grep pattern MUST check for the persona name in the "(in-progress: persona)" metadata
    # This ensures the persona parameter is actually used (was previously ignored)
    if grep -q "^- \[~\].*in-progress: $persona" "$TASKS_DIR/queue.md"; then
        return 0  # Has in-progress work
    else
        return 1  # No in-progress work
    fi
}

# Get all in-progress tasks for a persona
# Usage: tasks=$(get_in_progress_tasks "$persona_name")
get_in_progress_tasks() {
    local persona="$1"

    # MAINTAINER FIX: Updated to match actual queue.md sections (ACTIVE, In Progress, etc)
    sed -n '/^## \(ACTIVE\|In Progress\)/,/^## \(Completed\|Examples\|Task Format\)/p' "$TASKS_DIR/queue.md" | \
        grep "^- \[~\].*in-progress: $persona"
}

# Enhanced mark_task_completed that handles both [ ] and [~] tasks
# Usage: mark_task_completed_enhanced "$task_description" "$persona_name"
mark_task_completed_enhanced() {
    local task="$1"
    local persona="$2"
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # Create backup
    local backup_file="${TASKS_DIR}/queue.md.backup-$(date +%Y%m%d-%H%M%S)"
    if ! cp "$TASKS_DIR/queue.md" "$backup_file"; then
        echo "ERROR: Failed to create backup of queue.md" >&2
        return 1
    fi

    local escaped_task
    escaped_task=$(escape_for_sed "$task")

    # Mark task as completed (handle both [ ] and [~] states)
    local temp_file
    temp_file=$(mktemp)

    # Ensure temp file is cleaned up on exit (trap-based cleanup)
    trap "rm -f '$temp_file'" RETURN

    sed "0,/^- \[\( \|~\)\] ${escaped_task}/s//- [x] ${escaped_task} (completed: $timestamp, by: $persona)/" \
        "$TASKS_DIR/queue.md" > "$temp_file"

    # Validation
    local original_size=$(wc -l < "$TASKS_DIR/queue.md")
    local new_size=$(wc -l < "$temp_file")
    local min_size=$((original_size * 80 / 100))

    if [ "$new_size" -lt "$min_size" ]; then
        echo "ERROR: mark_task_completed_enhanced produced corrupted file" >&2
        return 1
    fi

    # Apply changes
    mv "$temp_file" "$TASKS_DIR/queue.md"
    log "INFO" "Task marked complete: $task (persona: $persona)"

    return 0
}

# ARCHITECT NOTES FOR FUTURE MAINTAINERS:
#
# 1. TASK STATE TRANSITIONS:
#    [ ] (not started) → [~] (in-progress) → [x] (completed)
#
# 2. IN-PROGRESS FORMAT:
#    - [~] Task description (in-progress: persona-name, started: ISO-8601)
#
# 3. COMPLETED FORMAT:
#    - [x] Task description (completed: ISO-8601, by: persona-name)
#
# 4. PERSONA TAGS:
#    Format: [PERSONA] at start of task description
#    Example: "[OPTIMIZER] Complete performance optimization"
#    Case-insensitive matching (converted to lowercase)
#
# 5. ROUTING LOGIC:
#    Priority 1: Tasks tagged for current persona
#    Priority 2: Untagged tasks (available to all)
#    Priority 3: Return empty (no suitable work)
#
# 6. BACKWARD COMPATIBILITY:
#    - Old [ ] tasks continue to work
#    - Old mark_task_completed() still functional
#    - New functions are additive, not replacing
#
# 7. ERROR HANDLING:
#    - All functions create backups before modifications
#    - File size validation prevents corruption
#    - Failures return non-zero exit codes
#    - All operations logged for auditing
#
# 8. TESTING:
#    Test file: scripts/test-task-state-management.sh
#    - Test state transitions ([ ] → [~] → [x])
#    - Test persona matching (tagged, untagged, mismatch)
#    - Test in-progress detection
#    - Test error handling (corruption, missing files)
#
# 9. INTEGRATION:
#    To use these functions in daemon.sh:
#    source "${DAEMON_ROOT}/lib/task-state-management.sh"
#
# 10. MONITORING:
#     Watch for these log messages:
#     - "Task marked in-progress" (normal operation)
#     - "Task marked complete" (normal operation)
#     - "ERROR: mark_task_* produced corrupted file" (investigate immediately)
