#!/bin/bash
#
# Task Template Engine
# Creates tasks from reusable templates with parameter substitution
#
# Usage:
#   ./create-task-from-template.sh --list
#   ./create-task-from-template.sh write-chapter chapter_number=5 min_words=2500
#
# Template Format (in tasks/templates/*.md):
#   ---
#   TEMPLATE: write-chapter
#   DESCRIPTION: Create a task to write a book chapter
#   PARAMETERS:
#     - chapter_number: Chapter number (required)
#     - min_words: Minimum word count (required)
#     - persona: Persona to assign (optional, default: EXPERIMENTER)
#   ---
#
#   - [ ] [{{persona}}] Write Chapter {{chapter_number}}
#     TASK_ID: task-{{timestamp}}-write-chapter-{{chapter_number}}
#     OUTPUT: chapters/chapter-{{chapter_number}}.md
#     VERIFY: [ $(wc -w < chapters/chapter-{{chapter_number}}.md) -ge {{min_words}} ]
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
TEMPLATES_DIR="${DAEMON_ROOT}/tasks/templates"
TASKS_FILE="${DAEMON_ROOT}/tasks/queue.md"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

# ============================================================================
# Template Discovery & Listing
# ============================================================================

# List all available templates
list_templates() {
    if [ ! -d "$TEMPLATES_DIR" ]; then
        echo "Templates directory not found: $TEMPLATES_DIR"
        return 1
    fi

    echo "Available Templates:"
    echo "===================="
    echo ""

    local count=0
    for template_file in "$TEMPLATES_DIR"/*.md; do
        if [ -f "$template_file" ]; then
            local template_name
            template_name=$(basename "$template_file" .md)

            local description=""
            if [ -f "$template_file" ]; then
                description=$(sed -n 's/^DESCRIPTION:[[:space:]]*//' p "$template_file" | head -1)
            fi

            if [ -z "$description" ]; then
                description="(no description)"
            fi

            echo "  • $template_name"
            echo "    $description"
            echo ""

            ((count++))
        fi
    done

    if [ $count -eq 0 ]; then
        echo "No templates found in $TEMPLATES_DIR"
        return 1
    fi

    echo "To create a task from a template:"
    echo "  ./create-task-from-template.sh <template-name> <param1=value1> [param2=value2] ..."
}

# Get template description
get_template_description() {
    local template_name="$1"
    local template_file="$TEMPLATES_DIR/${template_name}.md"

    if [ ! -f "$template_file" ]; then
        return 1
    fi

    sed -n 's/^DESCRIPTION:[[:space:]]*//' p "$template_file" | head -1
}

# Get required parameters from template
get_template_parameters() {
    local template_name="$1"
    local template_file="$TEMPLATES_DIR/${template_name}.md"

    if [ ! -f "$template_file" ]; then
        return 1
    fi

    # Extract PARAMETERS section and parse required/optional
    sed -n '/^PARAMETERS:/,/^---/p' "$template_file" | \
        grep -E "^\s*-\s+" | \
        sed 's/^[[:space:]]*-[[:space:]]*//g'
}

# ============================================================================
# Parameter Substitution
# ============================================================================

# Substitute template variables with parameter values
# Usage: substitute_parameters "$template_content" param1=value1 param2=value2
substitute_parameters() {
    local content="$1"
    shift
    local params=("$@")

    # Add timestamp automatically
    local timestamp
    timestamp=$(date +%Y%m%d-%H%M%S)
    content=$(echo "$content" | sed "s/{{timestamp}}/$timestamp/g")

    # Substitute each parameter
    for param in "${params[@]}"; do
        local key="${param%=*}"
        local value="${param#*=}"

        # Escape special characters for sed
        value=$(printf '%s\n' "$value" | sed -e 's/[\/&]/\\&/g')

        # Replace {{key}} with value
        content=$(echo "$content" | sed "s/{{$key}}/$value/g")
    done

    echo "$content"
}

# Validate all required parameters are provided
# Usage: if ! validate_required_parameters "template_name" param1=value1 param2=value2; then
validate_required_parameters() {
    local template_name="$1"
    shift
    local provided_params=("$@")

    local template_file="$TEMPLATES_DIR/${template_name}.md"

    if [ ! -f "$template_file" ]; then
        echo -e "${RED}ERROR: Template not found: $template_name${NC}" >&2
        return 1
    fi

    # Extract all {{placeholder}} patterns from template
    local required_placeholders
    required_placeholders=$(grep -oE '{{[^}]+}}' "$template_file" | \
                           sed 's/{{//' | sed 's/}}//' | \
                           sort -u | \
                           grep -v timestamp)  # timestamp is auto-provided

    # Check each placeholder is provided (unless it's a known auto-substitution)
    while read -r placeholder; do
        if [ -z "$placeholder" ]; then
            continue
        fi

        local found=0
        for param in "${provided_params[@]}"; do
            local key="${param%=*}"
            if [ "$key" = "$placeholder" ]; then
                found=1
                break
            fi
        done

        if [ $found -eq 0 ]; then
            echo -e "${RED}ERROR: Missing required parameter: $placeholder${NC}" >&2
            return 1
        fi
    done <<< "$required_placeholders"

    return 0
}

# ============================================================================
# Task Creation
# ============================================================================

# Create task from template and add to queue
# Usage: create_task_from_template "template_name" param1=value1 param2=value2
create_task_from_template() {
    local template_name="$1"
    shift
    local provided_params=("$@")

    local template_file="$TEMPLATES_DIR/${template_name}.md"

    # Validate template exists
    if [ ! -f "$template_file" ]; then
        echo -e "${RED}ERROR: Template not found: $template_name${NC}" >&2
        return 1
    fi

    # Validate required parameters
    if ! validate_required_parameters "$template_name" "${provided_params[@]}"; then
        return 1
    fi

    # Extract template content (everything after ---)
    local template_content
    template_content=$(sed -n '/^---$/,$ { /^---$/! p }' "$template_file")

    # Substitute parameters
    local task_text
    task_text=$(substitute_parameters "$template_content" "${provided_params[@]}")

    # Append to queue.md
    if [ ! -f "$TASKS_FILE" ]; then
        echo -e "${RED}ERROR: Tasks file not found: $TASKS_FILE${NC}" >&2
        return 1
    fi

    # Find the Pending Tasks section and append before next section
    if grep -q "^## Pending Tasks" "$TASKS_FILE"; then
        # Create backup
        cp "$TASKS_FILE" "${TASKS_FILE}.backup.$(date +%s)"

        # Append task to pending section
        # Find the last task in pending section and add after it
        local temp_file
        temp_file=$(mktemp)
        trap "rm -f '$temp_file'" RETURN

        echo "$task_text" >> "$TASKS_FILE"

        echo -e "${GREEN}✓ Task created from template: $template_name${NC}"
        echo "Task added to: $TASKS_FILE"

        # Extract and display task ID if present
        local task_id
        task_id=$(echo "$task_text" | grep "TASK_ID:" | sed 's/.*TASK_ID:[[:space:]]*//' | head -1 | tr -d '\r')
        if [ -n "$task_id" ]; then
            echo "Task ID: $task_id"
        fi

        return 0
    else
        echo -e "${RED}ERROR: 'Pending Tasks' section not found in $TASKS_FILE${NC}" >&2
        return 1
    fi
}

# ============================================================================
# Main
# ============================================================================

main() {
    if [ $# -eq 0 ]; then
        echo "Usage: $(basename "$0") <command> [arguments]"
        echo ""
        echo "Commands:"
        echo "  --list                           List all available templates"
        echo "  <template_name> [params...]      Create task from template"
        echo ""
        echo "Examples:"
        echo "  $(basename "$0") --list"
        echo "  $(basename "$0") write-chapter chapter_number=5 min_words=2500"
        return 1
    fi

    case "$1" in
        --list)
            list_templates
            ;;
        *)
            # Create from template
            local template_name="$1"
            shift
            create_task_from_template "$template_name" "$@"
            ;;
    esac
}

# Only run main if script is executed directly (not sourced)
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi
