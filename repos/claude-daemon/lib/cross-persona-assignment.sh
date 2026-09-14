#!/bin/bash
#
# Cross-Persona Task Assignment Library
# Implements PRIMARY/BACKUP persona tags for task flexibility
#
# Purpose:
#   - Tasks can now specify PRIMARY and BACKUP personas
#   - If PRIMARY is unavailable (on cooldown, unhealthy), use BACKUP
#   - Enables better load distribution and prevents task starvation
#
# Task Format:
#   - [ ] [PRIMARY:optimizer,BACKUP:architect] Optimize database query
#   - [ ] [PERSONA:optimizer] Old format (single persona, still supported)
#   - [ ] No tag (available to any persona)
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
TASKS_DIR="${DAEMON_ROOT}/tasks"
TASKS_FILE="${TASKS_DIR}/queue.md"

# ============================================================================
# Core Functions
# ============================================================================

# Parse persona assignment from task tag
# Format: [PRIMARY:persona1,BACKUP:persona2] or [persona] or none
# Usage: parse_persona_tag "$tag_content"
# Returns: JSON with primary, backup, and any fields
parse_persona_tag() {
    local tag_content="$1"

    if [[ "$tag_content" =~ PRIMARY:([a-zA-Z]+) ]]; then
        local primary="${BASH_REMATCH[1]}"
        local backup=""

        if [[ "$tag_content" =~ BACKUP:([a-zA-Z]+) ]]; then
            backup="${BASH_REMATCH[1]}"
        fi

        echo "{\"primary\":\"$primary\",\"backup\":\"$backup\"}"
    elif [[ "$tag_content" =~ [a-zA-Z]+ ]]; then
        # Legacy single-persona format
        local persona=$(echo "$tag_content" | grep -oE '[a-zA-Z]+' | head -1 | tr '[:upper:]' '[:lower:]')
        echo "{\"primary\":\"$persona\",\"backup\":\"\"}"
    else
        # No persona specified
        echo "{\"primary\":\"\",\"backup\":\"\"}"
    fi
}

# Extract persona tag from task line
# Format: - [ ] [PRIMARY:opt,BACKUP:arch] Task description
# Or: - [ ] [AUDITOR] Task description
extract_persona_tag() {
    local task_line="$1"

    # Skip checkbox "- [ ]" or "- [~]", then find the next [bracketed] content
    # Pattern: anything followed by "] [" then capture content until next "]"
    if [[ "$task_line" =~ \]\ \[([^\]]+)\] ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}

# Get assigned persona(s) for a task
# Usage: get_task_personas "$task_line"
# Returns: primary persona (or first available backup)
get_task_personas() {
    local task_line="$1"
    local tag=$(extract_persona_tag "$task_line")

    if [ -z "$tag" ]; then
        echo ""  # No persona requirement
        return
    fi

    local personas_json=$(parse_persona_tag "$tag")
    echo "$personas_json"
}

# Check if current persona can take this task
# Primary persona always eligible (if available)
# Backup persona eligible if primary unavailable
can_persona_take_task() {
    local current_persona="$1"
    local task_line="$2"

    local personas_json=$(get_task_personas "$task_line")

    if [ -z "$personas_json" ] || [ "$personas_json" = "{}" ]; then
        return 0  # No restriction, anyone can do it
    fi

    local primary=$(echo "$personas_json" | jq -r '.primary' 2>/dev/null || echo "")
    local backup=$(echo "$personas_json" | jq -r '.backup' 2>/dev/null || echo "")

    # If current is primary, always eligible
    if [ "$current_persona" = "$primary" ]; then
        return 0
    fi

    # If primary is on cooldown/unavailable and current is backup, eligible
    if [ -n "$backup" ] && [ "$current_persona" = "$backup" ]; then
        # Check if primary is available
        if declare -f is_persona_excluded >/dev/null 2>&1; then
            if is_persona_excluded "$primary"; then
                return 0  # Primary unavailable, backup can help
            fi
        fi
    fi

    # Not eligible
    return 1
}

# Find best persona for a task (primary or fallback to backup)
# Usage: get_best_persona_for_task "$task_line"
# Returns: persona that can do the task, or current persona
get_best_persona_for_task() {
    local task_line="$1"
    local current_persona="${2:-}"

    local personas_json=$(get_task_personas "$task_line")

    if [ -z "$personas_json" ] || [ "$personas_json" = "{}" ]; then
        echo "$current_persona"  # No restriction
        return
    fi

    local primary=$(echo "$personas_json" | jq -r '.primary' 2>/dev/null || echo "")
    local backup=$(echo "$personas_json" | jq -r '.backup' 2>/dev/null || echo "")

    # Check if primary is available
    if [ -n "$primary" ]; then
        if declare -f is_persona_excluded >/dev/null 2>&1; then
            if ! is_persona_excluded "$primary"; then
                echo "$primary"
                return
            fi
        else
            # Can't check availability, assume primary is available
            echo "$primary"
            return
        fi
    fi

    # Primary unavailable or not specified, try backup
    if [ -n "$backup" ]; then
        echo "$backup"
        return
    fi

    # No valid assignment, use current
    echo "$current_persona"
}

# ============================================================================
# Task Queue Integration
# ============================================================================

# Get next task that current persona can do
# Considers PRIMARY/BACKUP persona tags
# Usage: get_next_suitable_task "$current_persona"
get_next_suitable_task() {
    local current_persona="$1"

    if [ ! -f "$TASKS_FILE" ]; then
        return 1
    fi

    # Get all pending tasks
    local pending_tasks=$(grep "^- \[ \]" "$TASKS_FILE" 2>/dev/null || echo "")

    if [ -z "$pending_tasks" ]; then
        return 1
    fi

    # Find first task that matches current persona
    while IFS= read -r task_line; do
        if [ -z "$task_line" ]; then
            continue
        fi

        if can_persona_take_task "$current_persona" "$task_line"; then
            echo "$task_line"
            return 0
        fi
    done <<< "$pending_tasks"

    # No matching task found
    return 1
}

# ============================================================================
# Debugging & Monitoring
# ============================================================================

# Show persona assignment for a task
# Usage: show_task_assignment "$task_line"
show_task_assignment() {
    local task_line="$1"
    local personas_json=$(get_task_personas "$task_line")

    if [ -z "$personas_json" ] || [ "$personas_json" = "{}" ]; then
        echo "Assignment: Available to all personas"
        return
    fi

    local primary=$(echo "$personas_json" | jq -r '.primary' 2>/dev/null || echo "")
    local backup=$(echo "$personas_json" | jq -r '.backup' 2>/dev/null || echo "")

    echo "Assignment:"
    echo "  Primary: $primary"

    if [ -n "$backup" ] && [ "$backup" != "null" ]; then
        echo "  Backup:  $backup"
    else
        echo "  Backup:  (none)"
    fi
}

# Audit all tasks for persona assignments
# Usage: audit_persona_assignments
audit_persona_assignments() {
    if [ ! -f "$TASKS_FILE" ]; then
        echo "Tasks file not found"
        return 1
    fi

    local total=0
    local unassigned=0
    local primary_only=0
    local with_backup=0

    echo "Cross-Persona Assignment Audit"
    echo "==============================="
    echo ""

    while IFS= read -r task_line; do
        if [[ "$task_line" =~ ^-\ \[ ]]; then
            ((total++))
            local personas_json=$(get_task_personas "$task_line")

            if [ -z "$personas_json" ] || [ "$personas_json" = "{}" ]; then
                ((unassigned++))
            else
                local backup=$(echo "$personas_json" | jq -r '.backup' 2>/dev/null || echo "")
                if [ -n "$backup" ] && [ "$backup" != "null" ]; then
                    ((with_backup++))
                else
                    ((primary_only++))
                fi
            fi
        fi
    done < "$TASKS_FILE"

    echo "Summary:"
    echo "  Total tasks: $total"
    echo "  Unassigned (any persona): $unassigned"
    echo "  Primary only: $primary_only"
    echo "  With backup: $with_backup"
    echo ""
    echo "Recommendation:"
    if [ $with_backup -eq 0 ]; then
        echo "  Consider adding BACKUP personas to critical tasks for resilience"
    fi
}

# Export functions for use in daemon.sh
export -f parse_persona_tag
export -f extract_persona_tag
export -f get_task_personas
export -f can_persona_take_task
export -f get_best_persona_for_task
export -f get_next_suitable_task
export -f show_task_assignment
export -f audit_persona_assignments
