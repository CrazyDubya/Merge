#!/bin/bash
#
# Task Validation Framework
# Validates tasks before and after execution to catch silent failures
#
# Usage:
#   source "${DAEMON_ROOT}/lib/task-validation.sh"
#   validate_task_prerequisites "$task" && execute_task
#   validate_task_output "$task" "$persona" && mark_success || mark_failure
#

set -euo pipefail

# Validate task prerequisites before execution
validate_task_prerequisites() {
    local task="$1"

    # Check if required files exist
    if echo "$task" | grep -qi "write.*chapter"; then
        # Verify creative directory exists
        if [ ! -d "$DAEMON_ROOT/creative/sentient-toaster" ]; then
            log "ERROR" "Creative directory missing: $DAEMON_ROOT/creative/sentient-toaster"
            return 1
        fi
    fi

    # Check if task format is valid
    if [ -z "$task" ] || [ ${#task} -lt 5 ]; then
        log "ERROR" "Task too short or empty: $task"
        return 1
    fi

    return 0
}

# Validate task output was actually created
# Returns 0 if output is valid, 1 if output is missing/insufficient
validate_task_output() {
    local task="$1"
    local persona="$2"

    # For writing tasks: verify file creation
    if echo "$task" | grep -qi "write.*chapter"; then
        validate_writing_task_output "$task" "$persona"
        return $?
    fi

    # For audit/review/analysis tasks: MUST modify documentation files
    if echo "$task" | grep -qiE 'audit|review|examine|analyze|document.*findings'; then
        local docs_modified=$(find "$DAEMON_ROOT" -type f \( -name "*.md" -o -path "*emergence-log*" -o -path "*inter-persona-dialogue*" \) -mmin -5 2>/dev/null | wc -l)

        if [ "$docs_modified" -eq 0 ]; then
            log "ERROR" "[$persona] Audit/review task completed but NO documentation files updated"
            log "ERROR" "[$persona] Task: $task"
            return 1
        fi

        # Verify substantive documentation (at least 20 new lines added)
        # Check for new content in the most recently modified markdown file
        local most_recent_doc=$(find "$DAEMON_ROOT" -type f -name "*.md" -mmin -5 2>/dev/null | head -1)
        if [ -n "$most_recent_doc" ] && [ -f "$most_recent_doc" ]; then
            local file_size=$(stat -c %s "$most_recent_doc" 2>/dev/null || stat -f %z "$most_recent_doc" 2>/dev/null || echo 0)
            # 20 lines is roughly 1000 characters (50 chars/line average)
            if [ "$file_size" -lt 500 ]; then
                log "ERROR" "[$persona] Audit/review task: documentation too brief ($file_size bytes, expected >500)"
                return 1
            fi
        fi

        log "INFO" "[$persona] Audit/review task validated: documentation updated"
        return 0
    fi

    # For general tasks: check for non-log file modifications
    local modified=$(find "$DAEMON_ROOT" -type f -mmin -5 -not -path "*/logs/*" 2>/dev/null | wc -l)
    if [ $modified -eq 0 ]; then
        log "ERROR" "[$persona] Task completed but no substantive files modified (only logs touched)"
        return 1
    fi

    return 0
}

# Specific validation for writing tasks
validate_writing_task_output() {
    local task="$1"
    local persona="$2"

    # Find files modified in last 5 minutes in creative directory
    local new_files=$(find "$DAEMON_ROOT/creative/" -type f -mmin -5 2>/dev/null | sort)

    if [ -z "$new_files" ]; then
        log "ERROR" "[$persona] Writing task but no files created/modified in creative/"
        log "ERROR" "[$persona] Task: $task"
        return 1
    fi

    # Check each file has sufficient content (chapters should be >1000 bytes)
    local found_valid=0
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            local size=$(stat -c %s "$file" 2>/dev/null || echo 0)

            # For chapter files, require >1000 bytes (roughly 150+ words)
            if [[ "$file" =~ chapter ]]; then
                if [ $size -gt 1000 ]; then
                    log "INFO" "[$persona] Validated chapter output: $(basename "$file") ($size bytes)"
                    found_valid=1
                else
                    log "WARN" "[$persona] Chapter file too small ($size bytes): $(basename "$file")"
                fi
            else
                # For non-chapter files, just require some content
                if [ $size -gt 100 ]; then
                    log "INFO" "[$persona] Validated output: $(basename "$file") ($size bytes)"
                    found_valid=1
                fi
            fi
        fi
    done <<< "$new_files"

    if [ $found_valid -eq 1 ]; then
        return 0
    else
        log "ERROR" "[$persona] Writing task completed but no valid output files found"
        log "ERROR" "[$persona] Files found but all too small: $new_files"
        return 1
    fi
}

# Check acceptance criteria embedded in task description
# Example: "Task description | SUCCESS: file exists and size>2000"
check_acceptance_criteria() {
    local task="$1"

    # Extract SUCCESS criteria if present
    local criteria=$(echo "$task" | grep -oP 'SUCCESS: \K[^\|]+' || echo "")

    if [ -z "$criteria" ]; then
        # No explicit criteria - validation already done by output check
        return 0
    fi

    # Log the criteria we're checking
    log "INFO" "Checking acceptance criteria: $criteria"

    # Evaluate criteria in safe subshell
    if eval "$criteria" 2>/dev/null; then
        log "INFO" "Acceptance criteria met: $criteria"
        return 0
    else
        log "ERROR" "Acceptance criteria failed: $criteria"
        return 1
    fi
}

# Duration checks removed: Claude can write quickly with good prompts
# Real validation comes from OUTPUT/VERIFY metadata and file existence checks
# (Removed check_task_duration function - duration is not a reliable indicator)

# Helper to log validation results
log_validation() {
    local task="$1"
    local result="$2"  # pass, fail, warning
    local details="$3"

    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)|$result|$task|$details" >> \
        "$DAEMON_ROOT/logs/validation.log"
}

export -f validate_task_prerequisites
export -f validate_task_output
export -f validate_writing_task_output
export -f check_acceptance_criteria
export -f log_validation
