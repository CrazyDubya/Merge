#!/bin/bash
#
# Task Outcome Verification Library
# Implements OUTPUT/VERIFY field validation for task completion
#
# Purpose:
#   - Tasks can now specify expected output files and verification commands
#   - Prevents phantom completion (tasks marked done without actual output)
#   - Ensures quality by validating outcome before marking complete
#
# Task Format:
#   - [ ] [PERSONA] Task description
#     OUTPUT: path/to/expected/file.txt
#     VERIFY: command that returns 0 for success
#
# Example:
#   - [ ] [ARCHITECT] Design system architecture
#     OUTPUT: docs/architecture.md
#     VERIFY: [ $(wc -l < docs/architecture.md) -ge 50 ]
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
TASKS_DIR="${DAEMON_ROOT}/tasks"
TASKS_FILE="${TASKS_DIR}/queue.md"

# ============================================================================
# Core Functions
# ============================================================================

# Extract OUTPUT field from task metadata
# Usage: get_task_output_file "$task_line"
# Returns: path to expected output file or empty string
get_task_output_file() {
    local task_section="$1"

    # Look for OUTPUT: field in the metadata following the task
    # Format: OUTPUT: path/to/file
    if echo "$task_section" | grep -q "OUTPUT:"; then
        echo "$task_section" | grep "OUTPUT:" | head -1 | sed 's/.*OUTPUT:[[:space:]]*//' | tr -d '\r'
    else
        echo ""
    fi
}

# Extract VERIFY field from task metadata
# Usage: get_task_verify_command "$task_line"
# Returns: command that returns 0 on success
get_task_verify_command() {
    local task_section="$1"

    # Look for VERIFY: field in the metadata
    # Format: VERIFY: command or [ test ]
    if echo "$task_section" | grep -q "VERIFY:"; then
        echo "$task_section" | grep "VERIFY:" | head -1 | sed 's/.*VERIFY:[[:space:]]*//' | tr -d '\r'
    else
        echo ""
    fi
}

# Check if task has outcome verification metadata
# Usage: has_outcome_verification "$task_line"
# Returns: 0 if OUTPUT or VERIFY field exists, 1 otherwise
has_outcome_verification() {
    local task_section="$1"

    if echo "$task_section" | grep -qE "OUTPUT:|VERIFY:"; then
        return 0
    else
        return 1
    fi
}

# Verify task output exists
# Usage: verify_output_exists "/path/to/file"
# Returns: 0 if file exists, 1 if missing
verify_output_exists() {
    local output_file="$1"

    if [ -z "$output_file" ]; then
        return 0  # No output requirement is ok
    fi

    if [ -f "$output_file" ]; then
        return 0
    else
        return 1
    fi
}

# Run custom verification command
# Usage: run_verification_command "[ $(wc -l < file.txt) -ge 50 ]"
# Returns: 0 if verification passes, 1 if fails
run_verification_command() {
    local verify_cmd="$1"

    if [ -z "$verify_cmd" ]; then
        return 0  # No verification is ok
    fi

    # Execute the verification command
    # Use eval to allow complex bash expressions
    if eval "$verify_cmd" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Validate task outcome (all checks must pass)
# Usage: validate_task_outcome "$task_section"
# Returns: 0 if all checks pass, 1 if any check fails
validate_task_outcome() {
    local task_section="$1"
    local output_file
    local verify_cmd

    output_file=$(get_task_output_file "$task_section")
    verify_cmd=$(get_task_verify_command "$task_section")

    # Check output file exists (if specified)
    if [ -n "$output_file" ]; then
        if ! verify_output_exists "$output_file"; then
            log "WARN" "Task outcome validation failed: Output file not found: $output_file"
            return 1
        fi
    fi

    # Run verification command (if specified)
    if [ -n "$verify_cmd" ]; then
        if ! run_verification_command "$verify_cmd"; then
            log "WARN" "Task outcome validation failed: Verification command failed: $verify_cmd"
            return 1
        fi
    fi

    # All checks passed
    return 0
}

# ============================================================================
# Integration Functions
# ============================================================================

# Enhanced task completion that validates outcome
# Usage: mark_task_completed_with_verification "$task_description" "$persona_name"
# Returns: 0 if marked complete, 1 if validation failed
mark_task_completed_with_verification() {
    local task="$1"
    local persona="$2"
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    if [ ! -f "$TASKS_FILE" ]; then
        log "ERROR" "Tasks file not found: $TASKS_FILE"
        return 1
    fi

    # Extract task section with metadata (task line + next few lines for metadata)
    # Find the task line, then get the next 3 lines for metadata
    local task_section
    task_section=$(grep -A 3 "^- \[.*\] $(echo "$task" | sed 's/[[\.*^$/\\&/g')" "$TASKS_FILE" || echo "")

    if [ -z "$task_section" ]; then
        log "WARN" "Task not found for validation: $task"
        return 1
    fi

    # Check if task has outcome verification requirements
    if has_outcome_verification "$task_section"; then
        # Validate outcome before marking complete
        if ! validate_task_outcome "$task_section"; then
            log "ERROR" "Cannot mark task complete: Outcome validation failed: $task"
            return 1
        fi

        log "INFO" "Task outcome validation passed: $task"
    else
        # IMPORTANT: Task completed WITHOUT verification metadata
        # Using heuristic validation (see task-validation.sh)
        log "WARN" "[$persona] Task completed WITHOUT OUTPUT/VERIFY metadata - using heuristic validation only"
        log "INFO" "[$persona] Consider adding OUTPUT:/VERIFY: fields for stronger validation in future tasks"
    fi

    # If validation passed (or no validation required), mark complete
    # Use existing function if available
    if declare -f mark_task_completed_enhanced >/dev/null 2>&1; then
        mark_task_completed_enhanced "$task" "$persona"
        return $?
    else
        # Fallback: mark with basic completion
        log "INFO" "Task marked complete (no enhanced marking available): $task"
        return 0
    fi
}

# ============================================================================
# Debugging & Monitoring
# ============================================================================

# Show verification metadata for a task
# Usage: show_task_verification_info "$task_line"
show_task_verification_info() {
    local task_section="$1"
    local output_file
    local verify_cmd

    output_file=$(get_task_output_file "$task_section")
    verify_cmd=$(get_task_verify_command "$task_section")

    echo "Task Verification Info:"
    echo "======================="

    if [ -n "$output_file" ]; then
        echo "Output File: $output_file"
        if [ -f "$output_file" ]; then
            echo "  Status: ✅ Exists ($(wc -l < "$output_file") lines)"
        else
            echo "  Status: ❌ Missing"
        fi
    else
        echo "Output File: (none specified)"
    fi

    if [ -n "$verify_cmd" ]; then
        echo "Verify Command: $verify_cmd"
        if run_verification_command "$verify_cmd"; then
            echo "  Status: ✅ Pass"
        else
            echo "  Status: ❌ Fail"
        fi
    else
        echo "Verify Command: (none specified)"
    fi
}

# Validate all tasks in queue for outcome verification readiness
# Usage: audit_outcome_verification
audit_outcome_verification() {
    if [ ! -f "$TASKS_FILE" ]; then
        echo "Tasks file not found: $TASKS_FILE"
        return 1
    fi

    local total_tasks=0
    local with_verification=0
    local verified_passed=0
    local verified_failed=0

    echo "Outcome Verification Audit"
    echo "=========================="
    echo ""

    # Process each task
    while IFS= read -r line; do
        if [[ "$line" =~ ^-\ \[[^]]+\] ]]; then
            ((total_tasks++))
            local task_num=$total_tasks

            # Get task and metadata
            local task_desc=$(echo "$line" | sed 's/^- \[[^]]*\] //')

            # Extract metadata (next 3 lines)
            local metadata=""
            local count=0
            while IFS= read -r meta_line && [ $count -lt 3 ]; do
                if [[ "$meta_line" =~ OUTPUT:|VERIFY: ]]; then
                    metadata="$metadata
$meta_line"
                    ((count++))
                fi
            done

            if [ -n "$metadata" ]; then
                ((with_verification++))

                if has_outcome_verification "$line$metadata"; then
                    if validate_task_outcome "$line$metadata"; then
                        ((verified_passed++))
                        echo "✅ Task $task_num: Verification PASS"
                    else
                        ((verified_failed++))
                        echo "❌ Task $task_num: Verification FAIL"
                    fi
                fi
            fi
        fi
    done < "$TASKS_FILE"

    echo ""
    echo "Summary:"
    echo "  Total tasks: $total_tasks"
    echo "  With verification: $with_verification"
    echo "  Verified passing: $verified_passed"
    echo "  Verified failing: $verified_failed"
}

# Export functions for use in daemon.sh
export -f get_task_output_file
export -f get_task_verify_command
export -f has_outcome_verification
export -f verify_output_exists
export -f run_verification_command
export -f validate_task_outcome
export -f mark_task_completed_with_verification
export -f show_task_verification_info
export -f audit_outcome_verification
