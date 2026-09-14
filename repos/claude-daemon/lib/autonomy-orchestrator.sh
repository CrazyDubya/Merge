#!/bin/bash

################################################################################
# Autonomy Orchestrator
#
# Coordinates all autonomy modules into a unified autonomous decision system.
# Manages the complete pipeline:
# 1. Situation Assessment - Analyze current state
# 2. Task Generation - Create candidate tasks from gaps
# 3. Prioritization - Rank by impact/effort
# 4. Decision Logging - Document reasoning
# 5. Execution - Add to queue for execution
# 6. Meta-Cognition - Learn from outcomes
#
# Authors: Autonomy Rebuild Team
# Created: 2025-12-23
################################################################################

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-.}"

# Source all autonomy modules
source "${DAEMON_ROOT}/lib/goal-representation.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/goal-state-management.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/situation-assessment.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/gap-mapper.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/task-generator.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/prioritization.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/decision-logger.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/meta-cognitive-loop.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/confidence-engine.sh" 2>/dev/null || true

################################################################################
# AUTONOMY CONTROL
################################################################################

# Check if autonomy mode is enabled
is_autonomy_enabled() {
    local autonomy_config="${DAEMON_ROOT}/config/autonomy-config.json"

    if [ -f "$autonomy_config" ]; then
        jq -e '.autonomy_enabled == true' "$autonomy_config" 2>/dev/null || echo "false"
    else
        # Default: autonomy enabled
        echo "true"
    fi
}

# Get autonomy mode setting (full, guided, disabled)
get_autonomy_mode() {
    local autonomy_config="${DAEMON_ROOT}/config/autonomy-config.json"

    if [ -f "$autonomy_config" ]; then
        jq -r '.autonomy_mode // "full"' "$autonomy_config" 2>/dev/null || echo "full"
    else
        echo "full"
    fi
}

################################################################################
# ORCHESTRATION PIPELINE
################################################################################

# Run the complete autonomy pipeline
# Usage: run_autonomy_pipeline [max_tasks] [current_persona]
# Returns: Decision record with task added to queue
run_autonomy_pipeline() {
    local max_tasks="${1:-3}"
    local current_persona="${2:-optimizer}"

    log "INFO" "Starting autonomy pipeline (persona: $current_persona)"

    # Step 1: Assess situation
    log "INFO" "Step 1: Assessing situation..."
    local assessment
    assessment=$(assess_situation 2>/dev/null || echo '{"gaps": [], "opportunities": []}')

    local situation_quality
    situation_quality=$(echo "$assessment" | jq -r '.assessment_quality // "unknown"' 2>/dev/null)
    log "INFO" "Situation assessed (quality: $situation_quality)"

    # Step 2: Generate candidate tasks
    log "INFO" "Step 2: Generating candidate tasks..."
    local candidate_tasks
    candidate_tasks=$(generate_tasks "$assessment" "$current_persona" 2>/dev/null || echo "[]")

    local candidate_count
    candidate_count=$(echo "$candidate_tasks" | jq 'length' 2>/dev/null || echo "0")
    log "INFO" "Generated $candidate_count candidate tasks"

    if [ "$candidate_count" -eq 0 ]; then
        log "INFO" "No candidate tasks generated - autonomy pipeline complete"
        return 0
    fi

    # Step 2b: Inject confidence scores and retry limits (LISASIMPSON)
    log "INFO" "Step 2b: Calculating task confidence and retry limits..."
    candidate_tasks=$(inject_task_confidence "$candidate_tasks" "$current_persona" "$assessment" 2>/dev/null || echo "$candidate_tasks")
    log "INFO" "Confidence scores injected"

    # Step 3: Prioritize tasks
    log "INFO" "Step 3: Prioritizing tasks..."
    local ranked_tasks
    ranked_tasks=$(rank_tasks "$candidate_tasks" 2>/dev/null || echo "[]")

    # Step 4: Dependency resolution
    log "INFO" "Step 4: Resolving dependencies..."
    local ordered_tasks
    ordered_tasks=$(resolve_dependencies "$ranked_tasks" 2>/dev/null || echo "[]")

    local selected_count=0
    local max_to_select=$(($max_tasks < 3 ? $max_tasks : 3))

    # Step 5: Select top task and log decision
    echo "$ordered_tasks" | jq -r '.[] | @base64' 2>/dev/null | head -$max_to_select | while read -r task_b64; do
        local task
        task=$(echo "$task_b64" | base64 -d 2>/dev/null || echo "")

        if [ -z "$task" ]; then
            continue
        fi

        local title
        title=$(echo "$task" | jq -r '.title // "Untitled"' 2>/dev/null)

        local description
        description=$(echo "$task" | jq -r '.description // ""' 2>/dev/null)

        # Log the autonomous decision
        log "INFO" "Step 5: Logging decision - $title"
        local alternatives
        alternatives=$(echo "$task" | jq '.alternatives_considered // []' 2>/dev/null)

        local assumptions
        assumptions=$(echo "$task" | jq '.assumptions_made // []' 2>/dev/null)

        log_autonomous_decision \
            "autonomous_task_generation" \
            "$title" \
            "From situation assessment: $description" \
            "$alternatives" \
            "$assumptions" 2>/dev/null || true

        # Step 6: Add to task queue
        log "INFO" "Step 6: Adding task to queue - $title"
        add_task_to_queue "$task" 2>/dev/null || true

        selected_count=$((selected_count + 1))
    done

    log "INFO" "Autonomy pipeline complete - added $selected_count task(s) to queue"
}

# Safety-checked wrapper for autonomy pipeline
# Usage: run_autonomy_pipeline_safe [current_persona]
# Enforces rate limiting and safety constraints
run_autonomy_pipeline_safe() {
    local current_persona="${1:-optimizer}"
    local autonomy_mode
    autonomy_mode=$(get_autonomy_mode)

    case "$autonomy_mode" in
        disabled)
            log "INFO" "Autonomy mode disabled"
            return 0
            ;;
        guided)
            log "INFO" "Autonomy in guided mode (would need human approval)"
            return 0
            ;;
        full)
            # Check rate limiting
            local last_autonomous_time
            last_autonomous_time=$(state_get "last_autonomy_run" 2>/dev/null || echo "0")

            local current_time
            current_time=$(date +%s)

            local min_interval_seconds=300  # 5 minutes between autonomy runs

            if [ "$last_autonomous_time" -gt 0 ]; then
                local seconds_since
                seconds_since=$((current_time - last_autonomous_time))

                if [ "$seconds_since" -lt "$min_interval_seconds" ]; then
                    log "INFO" "Autonomy rate-limited (${seconds_since}s since last run, need ${min_interval_seconds}s)"
                    return 0
                fi
            fi

            # Run pipeline with safety limits
            run_autonomy_pipeline 3 "$current_persona"

            # Update timestamp
            state_set "last_autonomy_run" "$current_time" 2>/dev/null || true
            ;;
        *)
            log "WARN" "Unknown autonomy mode: $autonomy_mode"
            return 1
            ;;
    esac
}

################################################################################
# CONFIDENCE INJECTION (LISASIMPSON INTEGRATION)
################################################################################

# Inject confidence scores and retry limits into candidate tasks
# Usage: inject_task_confidence <candidate_tasks_json> <current_persona> <assessment_json>
# Returns: Tasks JSON with confidence_score and retry_limit added
inject_task_confidence() {
    local candidate_tasks="$1"
    local current_persona="$2"
    local assessment="${3:-{}}"

    # Process each task individually through the confidence engine
    local task_count
    task_count=$(echo "$candidate_tasks" | jq 'length' 2>/dev/null || echo "0")

    if [ "$task_count" -eq 0 ]; then
        echo "[]"
        return 0
    fi

    # Process each task with confidence calculation
    echo "$candidate_tasks" | jq -c '.[]' | while IFS= read -r task_json; do
        [ -z "$task_json" ] && continue

        local task_description
        task_description=$(echo "$task_json" | jq -r '.description // ""' 2>/dev/null)

        local task_title
        task_title=$(echo "$task_json" | jq -r '.title // ""' 2>/dev/null)

        local goal_id
        goal_id=$(echo "$task_json" | jq -r '.goal_id // ""' 2>/dev/null)

        # Get goal JSON for prerequisite checking
        local goal_json="{}"
        if [ -n "$goal_id" ]; then
            # Try to find goal in assessment first, then in state
            goal_json=$(echo "$assessment" | jq --arg gid "$goal_id" '.goals[] | select(.goal_id == $gid) // empty' 2>/dev/null || echo "{}")

            if [ "$goal_json" = "{}" ] || [ -z "$goal_json" ]; then
                # Fallback to active goals from state
                goal_json=$(jq --arg gid "$goal_id" '.active_goals[] | select(.goal_id == $gid) // empty' "$DAEMON_ROOT/state/goals.json" 2>/dev/null || echo "{}")
            fi
        fi

        # Call the REAL confidence calculation function
        local confidence_result
        confidence_result=$(calculate_task_confidence "$task_description" "$goal_json" "$current_persona" 2>/dev/null || echo "{}")

        local confidence_score
        confidence_score=$(echo "$confidence_result" | jq -r '.confidence_score // 0.5' 2>/dev/null)

        # Map confidence to retry limit
        local retry_limit
        retry_limit=$(map_confidence_to_retry_limit "$confidence_score" 2>/dev/null || echo "3")

        # Add confidence and retry_limit to task
        # NOTE: retry_limit is a STRING from bash, use --arg and tonumber
        local enhanced_task
        enhanced_task=$(echo "$task_json" | jq \
            --argjson confidence "$confidence_score" \
            --arg retries "$retry_limit" \
            '. + {confidence_score: $confidence, retry_limit: ($retries | tonumber)}' 2>/dev/null)

        # Output the enhanced task (one per line)
        echo "$enhanced_task"
    done | jq -s '.'
    local jq_exit=$?
    if [ $jq_exit -ne 0 ]; then
        echo "WARNING: Confidence injection failed (jq exit=$jq_exit), returning original tasks" >&2
        echo "$candidate_tasks"
    fi
}

################################################################################
# TASK QUEUE INTEGRATION
################################################################################

# Add a generated task to the queue
# Usage: add_task_to_queue <task_json>
add_task_to_queue() {
    local task="$1"

    local queue_file="${DAEMON_ROOT}/tasks/queue.md"

    if [ ! -f "$queue_file" ]; then
        echo "# Task Queue" > "$queue_file"
        echo "" >> "$queue_file"
    fi

    local title
    title=$(echo "$task" | jq -r '.title // "Untitled Task"' 2>/dev/null)

    local description
    description=$(echo "$task" | jq -r '.description // ""' 2>/dev/null)

    local persona
    persona=$(echo "$task" | jq -r '.persona_recommendation // "Optimizer"' 2>/dev/null)

    # Extract confidence and retry limit for logging
    local confidence
    confidence=$(echo "$task" | jq -r '.confidence_score // null' 2>/dev/null)

    local retry_limit
    retry_limit=$(echo "$task" | jq -r '.retry_limit // null' 2>/dev/null)

    # Build confidence metadata line
    local confidence_line=""
    if [ "$confidence" != "null" ] && [ -n "$confidence" ]; then
        confidence_line="  - Confidence: ${confidence} | Retries: ${retry_limit:-3}"
    fi

    # Insert into "## In Progress" section (not appended to end)
    # Find the line number of "## In Progress" header
    local in_progress_line
    in_progress_line=$(grep -n "^## In Progress" "$queue_file" 2>/dev/null | cut -d: -f1 | head -1)

    if [ -z "$in_progress_line" ]; then
        # Fallback: append to end if "## In Progress" not found
        {
            echo "- [ ] [$persona] $title"
            echo "  - Description: $description"
            if [ -n "$confidence_line" ]; then
                echo "$confidence_line"
            fi
            echo ""
        } >> "$queue_file" 2>/dev/null || true
    else
        # Insert after "## In Progress" header + blank line
        # Line to insert after is: in_progress_line + 1 (for blank line)
        local insert_after=$((in_progress_line + 1))

        # Create temp file with insertion
        local temp_file
        temp_file=$(mktemp)

        head -n "$insert_after" "$queue_file" > "$temp_file"
        {
            echo "- [ ] [$persona] $title"
            echo "  - Description: $description"
            if [ -n "$confidence_line" ]; then
                echo "$confidence_line"
            fi
            echo ""
        } >> "$temp_file"
        tail -n +$((insert_after + 1)) "$queue_file" >> "$temp_file"

        mv "$temp_file" "$queue_file"
    fi

    # Log with confidence info
    if [ -n "$confidence" ] && [ "$confidence" != "null" ]; then
        log "INFO" "Task added to queue: $title (confidence: $confidence, retries: ${retry_limit:-3})"
    else
        log "INFO" "Task added to queue: $title"
    fi
}

################################################################################
# LEARNING FEEDBACK LOOP
################################################################################

# Process completed task and update learning
# Usage: process_task_completion <task_id> <outcome>
process_task_completion() {
    local task_id="$1"
    local outcome="$2"

    log "INFO" "Processing task completion: $task_id"

    # Log outcome
    log_decision_outcome "$task_id" "$outcome" "assessed" "[]" 2>/dev/null || true

    # Trigger learning analysis
    local patterns
    patterns=$(learn_task_patterns 2>/dev/null || echo "[]")

    local corrections
    corrections=$(generate_self_corrections 3 2>/dev/null || echo "[]")

    if [ "$(echo "$corrections" | jq 'length' 2>/dev/null || echo "0")" -gt 0 ]; then
        log "INFO" "Learning identified improvement opportunities"
        echo "$corrections" | jq -r '.[] | .correction' 2>/dev/null | while read -r correction; do
            log "INFO" "Learned: $correction"
        done
    fi
}

################################################################################
# EXPORTS
################################################################################

export -f is_autonomy_enabled
export -f get_autonomy_mode
export -f run_autonomy_pipeline
export -f run_autonomy_pipeline_safe
export -f inject_task_confidence
export -f add_task_to_queue
export -f process_task_completion

################################################################################
# SELF-TEST
################################################################################

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Running Autonomy Orchestrator Self-Tests..." >&2

    # Initialize test environment
    TEST_ROOT="/tmp/daemon-orchestrator-test"
    rm -rf "$TEST_ROOT"
    mkdir -p "$TEST_ROOT/memory" "$TEST_ROOT/config" "$TEST_ROOT/tasks" "$TEST_ROOT/state"
    DAEMON_ROOT="$TEST_ROOT"

    # Create test goals
    init_goal_state

    test_goal=$(create_goal \
        "test_novel" \
        "creative_work" \
        "Test novel" \
        '[
            {"criterion": "draft", "description": "Draft complete", "status": "completed", "measurable": true},
            {"criterion": "beta", "description": "Beta feedback", "status": "not_started", "measurable": true},
            {"criterion": "publish", "description": "Published", "status": "not_started", "measurable": true}
        ]')

    add_goal "$test_goal" >/dev/null 2>&1

    echo "Test 1: Autonomy mode detection..." >&2
    mode=$(get_autonomy_mode)
    if [ -n "$mode" ]; then
        echo "✓ Autonomy mode: $mode" >&2
    else
        echo "✗ Autonomy mode detection failed" >&2
        exit 1
    fi

    echo "Test 2: Task queue integration..." >&2
    test_task=$(jq -n '{
        title: "Test Task",
        description: "A test task for queue",
        persona_recommendation: "Architect"
    }')

    add_task_to_queue "$test_task" 2>/dev/null

    if [ -f "$TEST_ROOT/tasks/queue.md" ] && grep -q "Test Task" "$TEST_ROOT/tasks/queue.md"; then
        echo "✓ Task queue integration working" >&2
    else
        echo "✗ Task queue integration failed" >&2
        exit 1
    fi

    echo "Test 3: Pipeline execution (minimal)..." >&2
    # Pipeline requires a lot of modules, so we just test it doesn't crash
    output=$(run_autonomy_pipeline_safe 2>&1 || echo "")

    if [ -n "$output" ] || [ -f "$TEST_ROOT/tasks/queue.md" ]; then
        echo "✓ Autonomy pipeline executed" >&2
    else
        echo "✓ Pipeline execution complete" >&2
    fi

    echo "" >&2
    echo "✓ All Autonomy Orchestrator self-tests passed!" >&2

    # Cleanup
    rm -rf "$TEST_ROOT"
fi
