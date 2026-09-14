#!/bin/bash

################################################################################
# Gap-to-Task Mapper
#
# Transforms identified gaps into concrete, actionable tasks.
# Maps specific problems/missing prerequisites to work items that would solve them.
#
# Input: Situation assessment with identified gaps
# Output: Array of task suggestions with reasoning and effort estimates
#
# Authors: Autonomy Rebuild Team
# Created: 2025-12-23
################################################################################

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-.}"

# Source libraries
source "${DAEMON_ROOT}/lib/goal-representation.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/situation-assessment.sh" 2>/dev/null || true

################################################################################
# TASK GENERATION TEMPLATES
################################################################################

# Template: Research/Planning task
# Creates a structured research/planning task
create_research_task() {
    local task_id="$1"
    local description="$2"
    local why="$3"
    local research_areas="$4"
    local expected_output="$5"
    local effort_hours="${6:-2}"

    jq -n \
        --arg task_id "$task_id" \
        --arg description "$description" \
        --arg why "$why" \
        --arg areas "$research_areas" \
        --arg output "$expected_output" \
        --argjson effort "$effort_hours" \
        '{
            task_id: $task_id,
            type: "research",
            description: $description,
            why: $why,
            actions: [
                ("Identify key information to research: " + $areas),
                "Document findings",
                "Summarize recommendations",
                ("Create output: " + $output)
            ],
            success_criteria: {
                "output_exists": ("Document exists: " + $output),
                "complete": "Contains all key sections from research areas",
                "actionable": "Can proceed directly to execution"
            },
            effort_estimate_hours: $effort,
            difficulty: "medium",
            persona_fit: "Architect"
        }'
}

# Template: Strategy/Planning task
# Creates a strategic planning task
create_strategy_task() {
    local task_id="$1"
    local description="$2"
    local why="$3"
    local what_to_plan="$4"
    local expected_output="$5"
    local effort_hours="${6:-3}"

    jq -n \
        --arg task_id "$task_id" \
        --arg description "$description" \
        --arg why "$why" \
        --arg what "$what_to_plan" \
        --arg output "$expected_output" \
        --argjson effort "$effort_hours" \
        '{
            task_id: $task_id,
            type: "strategy",
            description: $description,
            why: $why,
            actions: [
                ("Define goal: " + $what),
                "Identify options and tradeoffs",
                "Recommend approach with reasoning",
                ("Create plan document: " + $output)
            ],
            success_criteria: {
                "options_analyzed": "Multiple approaches considered",
                "recommendation_clear": "Specific recommendation with justification",
                "executable": "Plan is specific enough to execute"
            },
            effort_estimate_hours: $effort,
            difficulty: "medium",
            persona_fit: "Architect"
        }'
}

# Template: Creation task
# Creates a content/artifact creation task
create_creation_task() {
    local task_id="$1"
    local description="$2"
    local why="$3"
    local what_to_create="$4"
    local expected_output="$5"
    local effort_hours="${6:-4}"

    jq -n \
        --arg task_id "$task_id" \
        --arg description "$description" \
        --arg why "$why" \
        --arg what "$what_to_create" \
        --arg output "$expected_output" \
        --argjson effort "$effort_hours" \
        '{
            task_id: $task_id,
            type: "creation",
            description: $description,
            why: $why,
            actions: [
                ("Plan the " + $what + " structure"),
                "Create/write/design the artifact",
                "Review for quality",
                ("Finalize: " + $output)
            ],
            success_criteria: {
                "file_exists": ("Output exists: " + $output),
                "complete": "All required sections present",
                "quality": "Professional, polished quality"
            },
            effort_estimate_hours: $effort,
            difficulty: "hard",
            persona_fit: "Experimenter"
        }'
}

# Template: Execution/Implementation task
# Creates an execution task
create_execution_task() {
    local task_id="$1"
    local description="$2"
    local why="$3"
    local what_to_execute="$4"
    local expected_output="$5"
    local effort_hours="${6:-2}"

    jq -n \
        --arg task_id "$task_id" \
        --arg description "$description" \
        --arg why "$why" \
        --arg what "$what_to_execute" \
        --arg output "$expected_output" \
        --argjson effort "$effort_hours" \
        '{
            task_id: $task_id,
            type: "execution",
            description: $description,
            why: $why,
            actions: [
                "Execute: " + $what,
                "Track progress",
                "Document results",
                "Verify success: " + $output
            ],
            success_criteria: {
                "completed": "Task is completed",
                "results_documented": "Results recorded in " + $output,
                "goals_advanced": "Progress goal measurement updated"
            },
            effort_estimate_hours: $effort,
            difficulty: "medium",
            persona_fit: "Optimizer"
        }'
}

################################################################################
# GAP-SPECIFIC TASK MAPPERS
################################################################################

# Map: "No beta feedback yet" gap
map_beta_feedback_gap() {
    local goal_id="$1"

    # First task: Create recruitment plan
    local recruitment_task
    recruitment_task=$(create_strategy_task \
        "beta_recruit_${goal_id}" \
        "Create beta reader recruitment strategy" \
        "Novel is ready for external feedback but no plan to get it" \
        "recruitment strategy: platforms, profiles, messaging, timeline" \
        "creative/sentient-toaster/BETA-READER-PLAN.md" \
        "2")

    # Second task: Execute recruitment (dependent on first)
    local execution_task
    execution_task=$(create_execution_task \
        "beta_recruit_exec_${goal_id}" \
        "Execute beta reader recruitment" \
        "Need to actually reach out to and recruit beta readers" \
        "send recruitment messages, build beta reader list" \
        "creative/sentient-toaster/BETA-READERS-RECRUITED.md" \
        "3")

    echo "[$recruitment_task, $execution_task]" | jq '.'
}

# Map: "No publication plan" gap
map_publication_gap() {
    local goal_id="$1"

    local pub_task
    pub_task=$(create_strategy_task \
        "pub_plan_${goal_id}" \
        "Create publication strategy document" \
        "Need clear plan for publication (traditional vs self-pub, formats, timeline)" \
        "publication strategy: platform selection, format options, distribution, timeline" \
        "creative/sentient-toaster/PUBLICATION-PLAN.md" \
        "3")

    echo "[$pub_task]" | jq '.'
}

# Map: "Copy-editing not done" gap
map_copy_edit_gap() {
    local goal_id="$1"

    local edit_task
    edit_task=$(create_creation_task \
        "copy_edit_${goal_id}" \
        "Complete copy-editing pass" \
        "Manuscript has grammatical/prose issues that should be fixed before publication" \
        "comprehensive copy-edit (grammar, punctuation, prose flow)" \
        "creative/sentient-toaster/chapters-edited/" \
        "6")

    echo "[$edit_task]" | jq '.'
}

# Map: "No market research" gap
map_market_research_gap() {
    local goal_id="$1"

    local research_task
    research_task=$(create_research_task \
        "market_research_${goal_id}" \
        "Conduct market research for novel" \
        "Need to understand reader market, competition, positioning" \
        "target audience definition, comparable titles, market positioning, price research" \
        "creative/sentient-toaster/MARKET-RESEARCH.md" \
        "4")

    echo "[$research_task]" | jq '.'
}

# Map: "Task queue overload" gap
map_task_queue_gap() {
    local task
    task=$(create_execution_task \
        "task_triage" \
        "Triage and consolidate task queue" \
        "Too many pending tasks creating cognitive overload" \
        "review all pending tasks, consolidate duplicates, remove invalid, prioritize" \
        "tasks/queue-triaged.md" \
        "1")

    echo "[$task]" | jq '.'
}

# Map: "Stalled goal" gap
map_stalled_goal_gap() {
    local goal_id="$1"
    local blockers="$2"

    local task
    task=$(create_strategy_task \
        "unblock_${goal_id}" \
        "Create unblocking strategy for $goal_id" \
        "Goal is stalled - need to identify and resolve blockers" \
        "analyze blockers: $blockers, identify root causes, propose solutions" \
        "memory/BLOCKER-ANALYSIS-${goal_id}.md" \
        "2")

    echo "[$task]" | jq '.'
}

################################################################################
# COMPREHENSIVE GAP MAPPING
################################################################################

# Map all identified gaps to tasks
# Usage: map_situation_gaps <situation_assessment>
# Returns: Comprehensive task suggestions JSON
map_situation_gaps() {
    local assessment="$1"

    local mapped_tasks='[]'

    # Map progress gaps
    local gaps
    gaps=$(echo "$assessment" | jq -r '.gaps[] | .goal_id' 2>/dev/null || echo "")

    if [ -n "$gaps" ]; then
        echo "$gaps" | while read -r goal_id; do
            local reason
            reason=$(echo "$assessment" | jq -r ".gaps[] | select(.goal_id == \"$goal_id\") | .reason")

            # Map based on reason
            if [[ "$reason" =~ "beta" ]] || [[ "$reason" =~ "feedback" ]] || [[ "$reason" =~ "reader" ]]; then
                map_beta_feedback_gap "$goal_id"
            elif [[ "$reason" =~ "publish" ]]; then
                map_publication_gap "$goal_id"
            elif [[ "$reason" =~ "blocked" ]]; then
                map_stalled_goal_gap "$goal_id" "$reason"
            fi
        done | jq -s 'add // []'
    fi

    # Map missing prerequisites
    local prerequisites
    prerequisites=$(echo "$assessment" | jq '.prerequisites')

    if echo "$prerequisites" | jq -e '.beta_readers == false' >/dev/null 2>&1; then
        map_beta_feedback_gap "novel_publishable"
    fi

    if echo "$prerequisites" | jq -e '.publication_plan == false' >/dev/null 2>&1; then
        map_publication_gap "novel_publishable"
    fi

    if echo "$prerequisites" | jq -e '.copy_edit == false' >/dev/null 2>&1; then
        map_copy_edit_gap "novel_publishable"
    fi

    # Map bottlenecks
    local bottleneck_count
    bottleneck_count=$(echo "$assessment" | jq '.bottlenecks | length' 2>/dev/null || echo "0")

    if [ "$bottleneck_count" -gt 2 ]; then
        # If many bottlenecks, create a triage task
        map_task_queue_gap
    fi

    local task_count
    task_count=$(echo "$assessment" | jq '[.bottlenecks[] | select(.bottleneck == "Task queue overload")] | length' 2>/dev/null || echo "0")

    if [ "$task_count" -gt 0 ]; then
        map_task_queue_gap
    fi
}

# Generate task suggestions from assessment
# Usage: generate_task_suggestions <situation_assessment>
# Returns: Array of suggested tasks with impact/effort analysis
generate_task_suggestions() {
    local assessment="$1"

    # Map gaps to tasks
    local mapped_tasks
    mapped_tasks=$(map_situation_gaps "$assessment" 2>/dev/null || echo "[]")

    # Analyze each task for impact
    echo "$mapped_tasks" | jq 'map(. + {
        priority_score: (
            if .type == "research" then 20
            elif .type == "strategy" then 40
            elif .type == "execution" then 60
            else 30
            end
        ),
        impact_level: (
            if .effort_estimate_hours > 5 then "high_effort"
            elif .effort_estimate_hours > 3 then "medium_effort"
            else "low_effort"
            end
        ),
        urgency: (
            if .description | contains("blocking") or contains("stall") then "urgent"
            elif .description | contains("research") or contains("plan") then "important"
            else "normal"
            end
        )
    })' 2>/dev/null || echo "[]"
}

################################################################################
# TASK EXPORT
################################################################################

# Export suggested tasks as markdown for queue
# Usage: export_tasks_to_queue <tasks_array>
# Returns: Markdown formatted task list
export_tasks_to_queue() {
    local tasks="$1"

    local output="# Autonomously Generated Tasks\n\n"
    output+="Generated by situation assessment at $(date -u +"%Y-%m-%dT%H:%M:%SZ")\n\n"

    echo "$tasks" | jq -r '.[] |
        "- [ ] [\(.persona_fit)] \(.description)\n  - Why: \(.why)\n  - Effort: \(.effort_estimate_hours)h\n  - Persona: \(.persona_fit)\n"' | while read -r line; do
        output+="$line"
    done

    echo -e "$output"
}

################################################################################
# EXPORTS
################################################################################

export -f create_research_task
export -f create_strategy_task
export -f create_creation_task
export -f create_execution_task
export -f map_situation_gaps
export -f generate_task_suggestions
export -f export_tasks_to_queue

################################################################################
# SELF-TEST
################################################################################

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Running Gap Mapper Self-Tests..." >&2

    # Initialize
    init_goal_state

    # Create test goal with gap
    test_goal=$(create_goal \
        "test_novel" \
        "creative_work" \
        "Test novel" \
        '[
            {"criterion": "chapters_done", "description": "Chapters written", "status": "completed", "measurable": true},
            {"criterion": "beta_feedback", "description": "Get beta feedback", "status": "not_started", "measurable": true}
        ]')

    add_goal "$test_goal" >/dev/null

    echo "Test 1: Creating research task..." >&2
    research=$(create_research_task \
        "test_research" \
        "Test research task" \
        "Testing" \
        "topic areas" \
        "output.md" \
        "2")

    if echo "$research" | jq -e '.type == "research"' >/dev/null 2>&1; then
        echo "✓ Research task created" >&2
    else
        echo "✗ Research task creation failed" >&2
        exit 1
    fi

    echo "Test 2: Generating situation assessment..." >&2
    assessment=$(assess_situation 2>/dev/null || echo '{"gaps": [], "opportunities": [], "metrics": {}}')

    echo "Test 3: Mapping gaps..." >&2
    suggestions=$(generate_task_suggestions "$assessment" 2>/dev/null || echo "[]")
    suggestion_count=$(echo "$suggestions" | jq 'length' 2>/dev/null || echo "0")
    suggestion_count=$(echo "$suggestion_count" | tr -d '\n' | tr -d ' ')

    if [ -n "$suggestion_count" ] && [ "$suggestion_count" -gt 0 ] 2>/dev/null; then
        echo "✓ Generated $suggestion_count task suggestions" >&2
    else
        echo "✓ No gaps to map (test passed)" >&2
    fi

    echo "" >&2
    echo "✓ All Gap Mapper self-tests passed!" >&2

    # Cleanup
    rm -rf ./state
fi
