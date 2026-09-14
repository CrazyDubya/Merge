#!/bin/bash

################################################################################
# World State Library (LisaSimpson Integration)
#
# Provides explicit state variable modeling for goals and tasks. Tracks
# pre-conditions and post-conditions, enabling:
# - State-based verification (did conditions change as expected?)
# - Rollback capability (what was state before?)
# - Learning (which actions caused state transitions?)
#
# Implements key LisaSimpson pattern: "WorldState modeling with explicit
# pre/post-condition tracking as state variables"
#
# Authors: LisaSimpson + Autonomy Team
# Created: 2025-01-08
################################################################################

set -euo pipefail

# Daemon root
DAEMON_ROOT="${DAEMON_ROOT:-.}"

################################################################################
# STATE VARIABLE REGISTRY
################################################################################

# Define state variable types
declare -A STATE_VAR_TYPES=(
    [file_exists]="boolean"        # Does file exist?
    [file_size]="integer"          # File size in bytes
    [file_hash]="string"           # SHA256 hash of file
    [file_line_count]="integer"    # Line count
    [file_word_count]="integer"    # Word count
    [dir_file_count]="integer"     # Number of files in directory
    [command_output]="string"      # Output of a command
    [metric_value]="integer"       # Named metric value
    [goal_progress]="integer"      # Goal progress (0-100)
    [goal_status]="string"         # Goal status
    [text_contains]="boolean"      # Does text contain substring?
    [json_field]="any"             # Value of JSON field
)

################################################################################
# PRE-CONDITION CAPTURE
################################################################################

# Capture state before action execution
# Usage: capture_pre_conditions <goal_id> [state_var_json]
# Returns: Pre-conditions JSON snapshot with timestamp
capture_pre_conditions() {
    local goal_id="$1"
    local state_vars="${2:-{}}"

    local now
    now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Snapshot current values of all state variables
    local pre_conditions
    pre_conditions=$(echo "$state_vars" | jq \
        --arg goal_id "$goal_id" \
        --arg timestamp "$now" \
        '{
            goal_id: $goal_id,
            timestamp: $timestamp,
            snapshots: {},
            variables: .
        }')

    # For each state variable, capture actual value
    local var_count
    var_count=$(echo "$state_vars" | jq 'keys | length')

    for ((i=0; i<var_count; i++)); do
        local var_name
        var_name=$(echo "$state_vars" | jq -r "keys[$i]")

        local var_config
        var_config=$(echo "$state_vars" | jq ".\"$var_name\"")

        # Capture the actual value based on var type
        local value
        value=$(capture_state_variable "$var_name" "$var_config")

        pre_conditions=$(echo "$pre_conditions" | jq \
            --arg var_name "$var_name" \
            --argjson value "$value" \
            ".snapshots[\$var_name] = \$value")
    done

    echo "$pre_conditions"
}

# Capture a single state variable
# Supports: file existence, size, hash, line count, word count, dir file count
# Usage: capture_state_variable <var_name> <var_config_json>
# Returns: JSON with captured value
capture_state_variable() {
    local var_name="$1"
    local var_config="$2"

    local var_type
    var_type=$(echo "$var_config" | jq -r '.type // "unknown"')

    local value=""
    local success="false"

    case "$var_type" in
        file_exists)
            local filepath
            filepath=$(echo "$var_config" | jq -r '.path')
            if [ -f "$filepath" ]; then
                value="true"
                success="true"
            else
                value="false"
                success="true"
            fi
            ;;
        file_size)
            local filepath
            filepath=$(echo "$var_config" | jq -r '.path')
            if [ -f "$filepath" ]; then
                value=$(stat -f%z "$filepath" 2>/dev/null || stat -c%s "$filepath" 2>/dev/null || echo 0)
                success="true"
            fi
            ;;
        file_hash)
            local filepath
            filepath=$(echo "$var_config" | jq -r '.path')
            if [ -f "$filepath" ]; then
                value=$(sha256sum "$filepath" 2>/dev/null | awk '{print $1}' || echo "")
                success="true"
            fi
            ;;
        file_line_count)
            local filepath
            filepath=$(echo "$var_config" | jq -r '.path')
            if [ -f "$filepath" ]; then
                value=$(wc -l < "$filepath" 2>/dev/null || echo 0)
                success="true"
            fi
            ;;
        file_word_count)
            local filepath
            filepath=$(echo "$var_config" | jq -r '.path')
            if [ -f "$filepath" ]; then
                value=$(wc -w < "$filepath" 2>/dev/null || echo 0)
                success="true"
            fi
            ;;
        dir_file_count)
            local dirpath
            dirpath=$(echo "$var_config" | jq -r '.path')
            if [ -d "$dirpath" ]; then
                value=$(find "$dirpath" -type f | wc -l)
                success="true"
            fi
            ;;
        command_output)
            local cmd
            cmd=$(echo "$var_config" | jq -r '.command')
            value=$(eval "$cmd" 2>/dev/null || echo "")
            success="true"
            ;;
        metric_value)
            local metric_name
            metric_name=$(echo "$var_config" | jq -r '.name')
            value=$(get_metric_value "$metric_name" 2>/dev/null || echo "0")
            success="true"
            ;;
        goal_progress)
            local target_goal_id
            target_goal_id=$(echo "$var_config" | jq -r '.goal_id')
            value=$(get_goal_progress_value "$target_goal_id" 2>/dev/null || echo "0")
            success="true"
            ;;
        goal_status)
            local target_goal_id
            target_goal_id=$(echo "$var_config" | jq -r '.goal_id')
            value=$(get_goal_status_value "$target_goal_id" 2>/dev/null || echo "unknown")
            success="true"
            ;;
        text_contains)
            local filepath
            filepath=$(echo "$var_config" | jq -r '.path')
            local substring
            substring=$(echo "$var_config" | jq -r '.text')
            if [ -f "$filepath" ] && grep -q "$substring" "$filepath" 2>/dev/null; then
                value="true"
                success="true"
            else
                value="false"
                success="true"
            fi
            ;;
        *)
            value="unsupported_type"
            ;;
    esac

    jq -n \
        --arg var_name "$var_name" \
        --arg var_type "$var_type" \
        --arg value "$value" \
        --arg success "$success" \
        '{
            var_name: $var_name,
            type: $var_type,
            value: $value,
            success: ($success == "true")
        }'
}

################################################################################
# POST-CONDITION CAPTURE
################################################################################

# Capture state after action execution
# Usage: capture_post_conditions <goal_id> <pre_conditions_json>
# Returns: Post-conditions JSON snapshot with timestamp
capture_post_conditions() {
    local goal_id="$1"
    local pre_conditions="$2"

    local now
    now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Extract state variables from pre-conditions
    local state_vars
    state_vars=$(echo "$pre_conditions" | jq '.variables')

    # Capture new values
    local post_conditions
    post_conditions=$(echo "$pre_conditions" | jq \
        --arg timestamp "$now" \
        '.timestamp = $timestamp |
         .snapshots = {} |
         .post_snapshots = {}')

    # Re-snapshot all variables
    local var_count
    var_count=$(echo "$state_vars" | jq 'keys | length')

    for ((i=0; i<var_count; i++)); do
        local var_name
        var_name=$(echo "$state_vars" | jq -r "keys[$i]")

        local var_config
        var_config=$(echo "$state_vars" | jq ".\"$var_name\"")

        local value
        value=$(capture_state_variable "$var_name" "$var_config")

        post_conditions=$(echo "$post_conditions" | jq \
            --arg var_name "$var_name" \
            --argjson value "$value" \
            ".post_snapshots[\$var_name] = \$value")
    done

    echo "$post_conditions"
}

################################################################################
# STATE DIFF GENERATION
################################################################################

# Compare pre-conditions with post-conditions
# Usage: generate_state_diff <pre_conditions_json> <post_conditions_json>
# Returns: JSON with changes detected
generate_state_diff() {
    local pre_conditions="$1"
    local post_conditions="$2"

    local pre_snapshots
    pre_snapshots=$(echo "$pre_conditions" | jq '.snapshots')

    local post_snapshots
    post_snapshots=$(echo "$post_conditions" | jq '.post_snapshots')

    # Compare each variable
    local diff
    diff=$(jq -n \
        --argjson pre "$pre_snapshots" \
        --argjson post "$post_snapshots" \
        '{
            changes: [],
            unchanged: [],
            failed_captures: []
        }')

    # Find all variables
    local all_vars
    all_vars=$(echo "$pre_snapshots" | jq -r 'keys[]' 2>/dev/null || true)

    while IFS= read -r var_name; do
        [ -z "$var_name" ] && continue

        local pre_val
        pre_val=$(echo "$pre_snapshots" | jq ".\"$var_name\".value" 2>/dev/null || echo "null")

        local post_val
        post_val=$(echo "$post_snapshots" | jq ".\"$var_name\".value" 2>/dev/null || echo "null")

        if [ "$pre_val" != "$post_val" ]; then
            diff=$(echo "$diff" | jq \
                --arg var "$var_name" \
                --arg pre "$pre_val" \
                --arg post "$post_val" \
                '.changes += [{var_name: $var, pre_value: $pre, post_value: $post, changed: true}]')
        else
            diff=$(echo "$diff" | jq \
                --arg var "$var_name" \
                '.unchanged += [$var]')
        fi
    done <<< "$all_vars"

    echo "$diff"
}

################################################################################
# STATE TRANSITION VALIDATION
################################################################################

# Validate that state transition is valid
# Usage: validate_state_transition <goal_id> <pre_conditions> <post_conditions> <expected_changes_json>
# Returns: JSON with validation result
validate_state_transition() {
    local goal_id="$1"
    local pre_conditions="$2"
    local post_conditions="$3"
    local expected_changes="${4:-{}}"

    # Generate actual diff
    local actual_diff
    actual_diff=$(generate_state_diff "$pre_conditions" "$post_conditions")

    # Count changes
    local changes_count
    changes_count=$(echo "$actual_diff" | jq '.changes | length')

    local expected_count
    expected_count=$(echo "$expected_changes" | jq 'length')

    # Basic validation: did we see some state change?
    local is_valid="true"
    local validation_reason="State transition verified"

    if [ "$changes_count" -eq 0 ] && [ "$expected_count" -gt 0 ]; then
        is_valid="false"
        validation_reason="Expected state changes but none detected"
    fi

    jq -n \
        --arg goal_id "$goal_id" \
        --arg is_valid "$is_valid" \
        --arg reason "$validation_reason" \
        --argjson actual_diff "$actual_diff" \
        --argjson expected_changes "$expected_changes" \
        '{
            goal_id: $goal_id,
            is_valid: ($is_valid == "true"),
            reason: $reason,
            actual_diff: $actual_diff,
            expected_changes: $expected_changes,
            changes_detected: ($actual_diff.changes | length),
            timestamp: now | floor | todate
        }'
}

################################################################################
# GOAL STATE QUERIES
################################################################################

# Get current goal progress value
get_goal_progress_value() {
    local goal_id="$1"
    # Query active goals and extract progress
    if command -v get_active_goals &> /dev/null; then
        get_active_goals 2>/dev/null | jq -r ".[] | select(.goal_id == \"$goal_id\") | .progress" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# Get current goal status value
get_goal_status_value() {
    local goal_id="$1"
    # Query active goals and extract primary status
    if command -v get_active_goals &> /dev/null; then
        local status
        status=$(get_active_goals 2>/dev/null | jq -r ".[] | select(.goal_id == \"$goal_id\") |
            if .completed_at != null then \"completed\"
            elif (.success_criteria[] | select(.status == \"blocked\") | .criterion) != null then \"blocked\"
            else \"in_progress\"
            end" 2>/dev/null)
        echo "${status:-unknown}"
    else
        echo "unknown"
    fi
}

# Get metric value (stub for integration with metrics system)
get_metric_value() {
    local metric_name="$1"
    # Would integrate with metrics/success-rates.json or similar
    echo "0"
}

################################################################################
# WORLD STATE PERSISTENCE
################################################################################

# Save world state snapshot to disk
# Usage: save_world_state <goal_id> <state_snapshot_json>
save_world_state() {
    local goal_id="$1"
    local state_snapshot="$2"

    local state_file="${DAEMON_ROOT}/state/world-state.json"

    # Ensure state file exists
    if [ ! -f "$state_file" ]; then
        echo '{}' > "$state_file"
    fi

    # Update state file with new snapshot
    local updated_state
    updated_state=$(jq \
        --arg goal_id "$goal_id" \
        --argjson snapshot "$state_snapshot" \
        '.[$goal_id] = $snapshot' \
        "$state_file")

    echo "$updated_state" > "$state_file"
}

# Load world state for goal
# Usage: load_world_state <goal_id>
# Returns: World state JSON or empty object if not found
load_world_state() {
    local goal_id="$1"
    local state_file="${DAEMON_ROOT}/state/world-state.json"

    if [ ! -f "$state_file" ]; then
        echo '{}'
        return 0
    fi

    jq ".\"$goal_id\" // {}" "$state_file"
}

################################################################################
# PREDICATE EVALUATION
################################################################################

# Evaluate a state predicate (assertion about current state)
# Usage: evaluate_predicate <predicate_json>
# Returns: JSON with evaluation result
evaluate_predicate() {
    local predicate="$1"

    local predicate_type
    predicate_type=$(echo "$predicate" | jq -r '.type // "unknown"')

    local result="false"

    case "$predicate_type" in
        file_exists)
            local filepath
            filepath=$(echo "$predicate" | jq -r '.path')
            if [ -f "$filepath" ]; then
                result="true"
            fi
            ;;
        file_contains)
            local filepath
            filepath=$(echo "$predicate" | jq -r '.path')
            local substring
            substring=$(echo "$predicate" | jq -r '.text')
            if [ -f "$filepath" ] && grep -q "$substring" "$filepath" 2>/dev/null; then
                result="true"
            fi
            ;;
        command_succeeds)
            local cmd
            cmd=$(echo "$predicate" | jq -r '.command')
            if eval "$cmd" &>/dev/null; then
                result="true"
            fi
            ;;
        goal_completed)
            local goal_id
            goal_id=$(echo "$predicate" | jq -r '.goal_id')
            local progress
            progress=$(get_goal_progress_value "$goal_id")
            if [ "$progress" -eq 100 ]; then
                result="true"
            fi
            ;;
        *)
            result="false"
            ;;
    esac

    jq -n \
        --arg type "$predicate_type" \
        --arg result "$result" \
        '{
            predicate_type: $type,
            evaluation_result: ($result == "true"),
            timestamp: now | floor | todate
        }'
}

################################################################################
# EXPORTS
################################################################################

export -f capture_pre_conditions
export -f capture_post_conditions
export -f capture_state_variable
export -f generate_state_diff
export -f validate_state_transition
export -f get_goal_progress_value
export -f get_goal_status_value
export -f get_metric_value
export -f save_world_state
export -f load_world_state
export -f evaluate_predicate

################################################################################
# SELF-TEST
################################################################################

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Running WorldState Self-Tests..." >&2

    # Test 1: Create test state variables
    echo "Test 1: Defining state variables..." >&2
    state_vars=$(jq -n '{
        "file_present": {
            "type": "file_exists",
            "path": "/tmp/test-worldstate-file.txt"
        },
        "file_lines": {
            "type": "file_line_count",
            "path": "/tmp/test-worldstate-file.txt"
        }
    }')

    # Create test file
    echo "line 1" > /tmp/test-worldstate-file.txt
    echo "line 2" >> /tmp/test-worldstate-file.txt

    # Test 2: Capture pre-conditions
    echo "Test 2: Capturing pre-conditions..." >&2
    pre=$(capture_pre_conditions "test_goal" "$state_vars")
    echo "$pre" | jq . >&2

    # Modify file
    echo "line 3" >> /tmp/test-worldstate-file.txt

    # Test 3: Capture post-conditions
    echo "Test 3: Capturing post-conditions..." >&2
    post=$(capture_post_conditions "test_goal" "$pre")
    echo "$post" | jq . >&2

    # Test 4: Generate diff
    echo "Test 4: Generating state diff..." >&2
    diff=$(generate_state_diff "$pre" "$post")
    echo "$diff" | jq . >&2

    # Cleanup
    rm -f /tmp/test-worldstate-file.txt

    echo "" >&2
    echo "✓ WorldState self-tests completed!" >&2
fi
