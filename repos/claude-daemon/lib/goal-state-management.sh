#!/bin/bash

################################################################################
# Goal State Management Library
#
# Provides persistent state management for goals using atomic operations.
# Goals are stored in state/goals.json and include:
# - Active goals (in progress, not started)
# - Completed goals (achieved)
# - Goal history (creation, updates, completions)
#
# All operations use atomic writes (mktemp + mv) to prevent corruption.
#
# Authors: Autonomy Rebuild Team
# Created: 2025-12-23
################################################################################

set -euo pipefail

# Daemon root
DAEMON_ROOT="${DAEMON_ROOT:-.}"
# Use GOALS_STATE_DIR to avoid collision with STATE_DIR from state-api.sh
# which uses STATE_DIR for personalities/state.json
GOALS_STATE_DIR="${DAEMON_ROOT}/state"
GOALS_FILE="${GOALS_STATE_DIR}/goals.json"
GOALS_HISTORY_FILE="${GOALS_STATE_DIR}/goals-history.jsonl"
LOCK_FILE="${GOALS_FILE}.lock"

# Source goal representation library
source "${DAEMON_ROOT}/lib/goal-representation.sh"

################################################################################
# INITIALIZATION
################################################################################

# Initialize goals state directory and files
init_goal_state() {
    mkdir -p "$GOALS_STATE_DIR"

    # Create empty goals file if it doesn't exist
    if [ ! -f "$GOALS_FILE" ]; then
        local initial_state
        initial_state=$(jq -n '{
            active_goals: [],
            completed_goals: [],
            last_updated: now | todate
        }')
        echo "$initial_state" > "$GOALS_FILE"
    fi

    # Create history file if it doesn't exist
    if [ ! -f "$GOALS_HISTORY_FILE" ]; then
        touch "$GOALS_HISTORY_FILE"
    fi
}

################################################################################
# ATOMIC READ/WRITE OPERATIONS
################################################################################

# Read goals state from disk
# Returns: Complete goals state JSON
read_goals_state() {
    if [ ! -f "$GOALS_FILE" ]; then
        init_goal_state
    fi

    cat "$GOALS_FILE"
}

# Write goals state to disk (atomic)
# Usage: write_goals_state <goals_json>
write_goals_state() {
    local goals_json="$1"

    local temp_file
    temp_file=$(mktemp)
    trap "rm -f '$temp_file'" RETURN

    # Write to temp file
    echo "$goals_json" > "$temp_file"

    # Atomic move
    if mv "$temp_file" "$GOALS_FILE" 2>/dev/null; then
        return 0
    else
        echo "ERROR: Failed to write goals file" >&2
        return 1
    fi
}

# Log goal state change to history
# Usage: log_goal_history <action> <goal_id> <goal_json> [details]
log_goal_history() {
    local action="$1"
    local goal_id="$2"
    local goal_json="$3"
    local details="${4:-}"

    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    local history_entry
    history_entry=$(jq -n \
        --arg timestamp "$timestamp" \
        --arg action "$action" \
        --arg goal_id "$goal_id" \
        --argjson goal "$goal_json" \
        --arg details "$details" \
        '{
            timestamp: $timestamp,
            action: $action,
            goal_id: $goal_id,
            goal: $goal,
            details: $details
        }')

    # Append to history
    echo "$history_entry" >> "$GOALS_HISTORY_FILE"
}

################################################################################
# GOAL CRUD OPERATIONS
################################################################################

# Get all active goals
# Returns: Array of goal JSON objects
get_active_goals() {
    local state
    state=$(read_goals_state)

    echo "$state" | jq '.active_goals'
}

# Get all completed goals
# Returns: Array of goal JSON objects
get_completed_goals() {
    local state
    state=$(read_goals_state)

    echo "$state" | jq '.completed_goals'
}

# Get specific goal by ID
# Usage: get_goal <goal_id>
# Returns: Goal JSON or null if not found
get_goal() {
    local goal_id="$1"
    local state
    state=$(read_goals_state)

    # Try to find in active goals first
    local goal
    goal=$(echo "$state" | jq ".active_goals[] | select(.goal_id == \"$goal_id\")")

    if [ -n "$goal" ]; then
        echo "$goal"
        return 0
    fi

    # Try completed goals
    goal=$(echo "$state" | jq ".completed_goals[] | select(.goal_id == \"$goal_id\")")

    if [ -n "$goal" ]; then
        echo "$goal"
        return 0
    fi

    return 1
}

# Add new goal
# Usage: add_goal <goal_json>
add_goal() {
    local goal_json="$1"

    # Validate goal structure
    if ! validate_goal_structure "$goal_json"; then
        return 1
    fi

    local goal_id
    goal_id=$(echo "$goal_json" | jq -r '.goal_id')

    # Check if goal already exists
    if get_goal "$goal_id" >/dev/null 2>&1; then
        echo "ERROR: Goal $goal_id already exists" >&2
        return 1
    fi

    local state
    state=$(read_goals_state)

    # Add to active goals
    local now_date
    now_date=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    state=$(echo "$state" | jq \
        --argjson new_goal "$goal_json" \
        --arg now_date "$now_date" \
        '.active_goals += [$new_goal] | .last_updated = $now_date')

    write_goals_state "$state"

    # Log to history
    log_goal_history "created" "$goal_id" "$goal_json" "New goal created"

    echo "Goal $goal_id added successfully"
}

# Update existing goal
# Usage: update_goal <goal_json>
update_goal() {
    local goal_json="$1"

    # Validate goal structure
    if ! validate_goal_structure "$goal_json"; then
        return 1
    fi

    local goal_id
    goal_id=$(echo "$goal_json" | jq -r '.goal_id')

    # Check if goal exists
    if ! get_goal "$goal_id" >/dev/null 2>&1; then
        echo "ERROR: Goal $goal_id not found" >&2
        return 1
    fi

    local state
    state=$(read_goals_state)

    # Update in active goals
    local now_date
    now_date=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    state=$(echo "$state" | jq \
        --arg goal_id "$goal_id" \
        --argjson updated_goal "$goal_json" \
        --arg now_date "$now_date" \
        '.active_goals |= map(
            if .goal_id == $goal_id then
                $updated_goal
            else
                .
            end
        ) | .last_updated = $now_date')

    write_goals_state "$state"

    # Log to history
    log_goal_history "updated" "$goal_id" "$goal_json" "Goal updated"

    echo "Goal $goal_id updated successfully"
}

# Mark goal as complete and move to completed
# Usage: complete_goal_in_state <goal_id>
complete_goal_in_state() {
    local goal_id="$1"

    # Get the goal
    local goal
    if ! goal=$(get_goal "$goal_id"); then
        echo "ERROR: Goal $goal_id not found" >&2
        return 1
    fi

    # Mark as complete
    goal=$(complete_goal "$goal")

    local state
    state=$(read_goals_state)

    # Move from active to completed
    local now_date
    now_date=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    state=$(echo "$state" | jq \
        --arg goal_id "$goal_id" \
        --argjson completed_goal "$goal" \
        --arg now_date "$now_date" \
        '.active_goals |= map(
            select(.goal_id != $goal_id)
        ) |
        .completed_goals += [$completed_goal] |
        .last_updated = $now_date')

    write_goals_state "$state"

    # Log to history
    log_goal_history "completed" "$goal_id" "$goal" "Goal marked complete"

    echo "Goal $goal_id marked as complete"
}

################################################################################
# GOAL QUERIES
################################################################################

# Get all goals with status (active or completed)
# Returns: JSON object with both active and completed goals
get_all_goals() {
    read_goals_state
}

# Get goal progress
# Usage: get_goal_progress <goal_id>
# Returns: Progress percentage (0-100)
get_goal_progress() {
    local goal_id="$1"

    local goal
    if ! goal=$(get_goal "$goal_id"); then
        echo "ERROR: Goal $goal_id not found" >&2
        return 1
    fi

    calculate_goal_progress "$goal"
}

# Get goals by type
# Usage: get_goals_by_type <type>
# Returns: Array of goal JSON objects matching type
get_goals_by_type() {
    local goal_type="$1"
    local state
    state=$(read_goals_state)

    echo "$state" | jq ".active_goals[] | select(.type == \"$goal_type\")"
}

# Get goals by status
# Usage: get_goals_by_status <status>
# Returns: Array of goals matching status (for any criteria)
get_goals_by_status() {
    local status="$1"
    local state
    state=$(read_goals_state)

    # Return goals that have at least one criterion with this status
    echo "$state" | jq --arg status "$status" '.active_goals[] | select(.success_criteria[] | select(.status == $status))'
}

# Get blocked goals
# Returns: Array of goals that are currently blocked
get_blocked_goals() {
    local state
    state=$(read_goals_state)

    echo "$state" | jq '.active_goals[] | select((.success_criteria[] | select(.status == "blocked")) | length > 0)'
}

# Get goals due soon
# Usage: get_goals_due_soon [days_threshold]
# Returns: Array of goals with target completion approaching
get_goals_due_soon() {
    local days_threshold="${1:-7}"
    local state
    state=$(read_goals_state)

    local now_timestamp
    now_timestamp=$(date -u +%s)

    local threshold_timestamp
    threshold_timestamp=$((now_timestamp + (days_threshold * 86400)))

    echo "$state" | jq --arg threshold "$threshold_timestamp" '.active_goals[] | select(
        .target_completion != null and
        (.target_completion | fromdateiso8601) < ($threshold | tonumber)
    )'
}

################################################################################
# GOAL CRITERION OPERATIONS
################################################################################

# Update criterion status in goal
# Usage: update_criterion_in_goal <goal_id> <criterion_id> <status> <evidence> [reason]
update_criterion_in_goal() {
    local goal_id="$1"
    local criterion_id="$2"
    local status="$3"
    local evidence="$4"
    local reason="${5:-}"

    local goal
    if ! goal=$(get_goal "$goal_id"); then
        echo "ERROR: Goal $goal_id not found" >&2
        return 1
    fi

    # Update criterion in goal
    goal=$(update_criterion_status "$goal" "$criterion_id" "$status" "$evidence" "$reason")

    # Recalculate progress
    goal=$(update_goal_progress "$goal")

    # Check if now complete
    if is_goal_complete "$goal"; then
        complete_goal_in_state "$goal_id"
    else
        update_goal "$goal"
    fi

    log_goal_history "criterion_updated" "$goal_id" "$goal" "Criterion $criterion_id updated to $status"
}

################################################################################
# GOAL DISPLAY/REPORTING
################################################################################

# Display all goals summary
# Shows active and completed goals with their progress
print_goals_summary() {
    local state
    state=$(read_goals_state)

    echo "=== Active Goals ===" >&2
    local active_count
    active_count=$(echo "$state" | jq '.active_goals | length')

    if [ "$active_count" -eq 0 ]; then
        echo "No active goals" >&2
    else
        echo "$state" | jq -r '.active_goals[] | "\(.goal_id) (\(.type)): \(.progress)% complete"' >&2
    fi

    echo "" >&2
    echo "=== Completed Goals ===" >&2
    local completed_count
    completed_count=$(echo "$state" | jq '.completed_goals | length')

    if [ "$completed_count" -eq 0 ]; then
        echo "No completed goals" >&2
    else
        echo "$state" | jq -r '.completed_goals[] | "\(.goal_id) (completed: \(.completed_at))"' >&2
    fi
}

# Export goal summary for human reading
# Usage: export_goal_summary
# Returns: Markdown-formatted goal summary
export_goal_summary() {
    local state
    state=$(read_goals_state)

    local output
    output="# Goal Status Summary\n\n"
    output+="Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")\n\n"

    output+="## Active Goals\n\n"
    local active_count
    active_count=$(echo "$state" | jq '.active_goals | length')

    if [ "$active_count" -eq 0 ]; then
        output+="No active goals.\n\n"
    else
        output+="$(echo "$state" | jq -r '.active_goals[] | "- **\(.goal_id)** (\(.type))\n  - Progress: \(.progress)%\n  - Description: \(.description)\n"')\n"
    fi

    output+="## Completed Goals\n\n"
    local completed_count
    completed_count=$(echo "$state" | jq '.completed_goals | length')

    if [ "$completed_count" -eq 0 ]; then
        output+="No completed goals yet.\n\n"
    else
        output+="$(echo "$state" | jq -r '.completed_goals[] | "- **\(.goal_id)** (completed: \(.completed_at))\n"')\n"
    fi

    echo -e "$output"
}

################################################################################
# EXPORTS
################################################################################

export -f init_goal_state
export -f read_goals_state
export -f write_goals_state
export -f log_goal_history
export -f get_active_goals
export -f get_completed_goals
export -f get_goal
export -f add_goal
export -f update_goal
export -f complete_goal_in_state
export -f get_all_goals
export -f get_goal_progress
export -f get_goals_by_type
export -f get_goals_by_status
export -f get_blocked_goals
export -f get_goals_due_soon
export -f update_criterion_in_goal
export -f print_goals_summary
export -f export_goal_summary

################################################################################
# SELF-TEST
################################################################################

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Running Goal State Management Self-Tests..." >&2

    # Initialize
    init_goal_state

    # Test 1: Add a goal
    echo "Test 1: Adding a goal..." >&2
    test_goal=$(create_goal \
        "test_publishable" \
        "creative_work" \
        "Make novel publishable" \
        '[
            {"criterion": "chapters_complete", "description": "All 25 chapters written", "status": "completed", "measurable": true},
            {"criterion": "copy_edit", "description": "Copy editing complete", "status": "in_progress", "measurable": true},
            {"criterion": "beta_feedback", "description": "Beta reader feedback incorporated", "status": "not_started", "measurable": true}
        ]')

    if add_goal "$test_goal"; then
        echo "✓ Goal added successfully" >&2
    else
        echo "✗ Failed to add goal" >&2
        exit 1
    fi

    # Test 2: Retrieve goal
    echo "Test 2: Retrieving goal..." >&2
    retrieved=$(get_goal "test_publishable")
    if [ -n "$retrieved" ]; then
        echo "✓ Goal retrieved successfully" >&2
    else
        echo "✗ Failed to retrieve goal" >&2
        exit 1
    fi

    # Test 3: Check progress
    echo "Test 3: Checking progress..." >&2
    progress=$(get_goal_progress "test_publishable")
    if [ "$progress" -eq 33 ]; then
        echo "✓ Progress correctly calculated: $progress%" >&2
    else
        echo "✗ Progress incorrect: expected 33%, got $progress%" >&2
        exit 1
    fi

    # Test 4: Update criterion
    echo "Test 4: Updating criterion..." >&2
    update_criterion_in_goal "test_publishable" "copy_edit" "completed" "file:///path" 2>/dev/null
    new_progress=$(get_goal_progress "test_publishable")
    if [ "$new_progress" -eq 66 ]; then
        echo "✓ Criterion updated, progress now: $new_progress%" >&2
    else
        echo "✗ Progress update failed: expected 66%, got $new_progress%" >&2
        exit 1
    fi

    # Test 5: List all goals
    echo "Test 5: Listing all goals..." >&2
    print_goals_summary

    echo "" >&2
    echo "✓ All Goal State Management self-tests passed!" >&2

    # Cleanup test data
    rm -f "$GOALS_FILE" "$GOALS_HISTORY_FILE"
fi
