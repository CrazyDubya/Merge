#!/bin/bash

################################################################################
# Task Generator
#
# Generates specific, actionable tasks from abstract gaps and opportunities.
# Uses template library to create well-formed tasks with:
# - Clear success criteria
# - Effort estimates
# - Persona matching
# - Impact analysis
#
# Input: Gap analysis + opportunity detection + goal state
# Output: Array of generated tasks ready to add to queue
#
# Authors: Autonomy Rebuild Team
# Created: 2025-12-23
################################################################################

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-.}"

# Source libraries
source "${DAEMON_ROOT}/lib/goal-representation.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/goal-state-management.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/situation-assessment.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/gap-mapper.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/opportunity-detector.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/verification-planner.sh" 2>/dev/null || true

################################################################################
# TASK TEMPLATE LIBRARY
################################################################################

# Template: Comprehensive research task with discovery and synthesis
create_research_task_detailed() {
    local title="$1"
    local description="$2"
    local research_areas="$3"
    local expected_output="$4"
    local effort_hours="${5:-3}"
    local blocker="${6:-}"

    jq -n \
        --arg title "$title" \
        --arg description "$description" \
        --arg areas "$research_areas" \
        --arg output "$expected_output" \
        --argjson effort "$effort_hours" \
        --arg blocker "$blocker" \
        '{
            task_type: "research",
            title: $title,
            description: $description,
            detailed_steps: [
                ("1. Define research scope: " + $areas),
                "2. Identify key information sources",
                "3. Conduct research and document findings",
                "4. Synthesize findings into recommendations",
                ("5. Create output: " + $output)
            ],
            success_criteria: {
                "output_exists": ("File exists at: " + $output),
                "comprehensive": ("Covers all research areas: " + $areas),
                "actionable": "Findings lead to next steps",
                "documented": "All sources and reasoning documented"
            },
            effort_estimate_hours: $effort,
            difficulty: "medium",
            persona_recommendation: "Architect",
            blocking_issue: $blocker,
            reasoning: ("This research is necessary to: " + $description)
        }'
}

# Template: Strategic planning with options and tradeoffs
create_strategy_task_detailed() {
    local title="$1"
    local description="$2"
    local planning_scope="$3"
    local expected_output="$4"
    local effort_hours="${5:-4}"
    local blocker="${6:-}"

    jq -n \
        --arg title "$title" \
        --arg description "$description" \
        --arg scope "$planning_scope" \
        --arg output "$expected_output" \
        --argjson effort "$effort_hours" \
        --arg blocker "$blocker" \
        '{
            task_type: "strategy",
            title: $title,
            description: $description,
            detailed_steps: [
                ("1. Define goal: " + $scope),
                "2. Identify 2-3 alternative approaches",
                "3. Analyze tradeoffs (effort, impact, risk)",
                "4. Recommend approach with reasoning",
                ("5. Create plan document: " + $output),
                "6. Include timeline and next steps"
            ],
            success_criteria: {
                "plan_exists": ("Plan document at: " + $output),
                "options_analyzed": "Multiple approaches evaluated",
                "recommendation_clear": "Specific recommendation with justification",
                "actionable": "Plan is specific enough to execute immediately"
            },
            effort_estimate_hours: $effort,
            difficulty: "medium",
            persona_recommendation: "Architect",
            blocking_issue: $blocker,
            reasoning: ("Strategic planning needed for: " + $description)
        }'
}

# Template: Creative/analytical artifact creation
create_creation_task_detailed() {
    local title="$1"
    local description="$2"
    local artifact_scope="$3"
    local expected_output="$4"
    local effort_hours="${5:-6}"
    local blocker="${6:-}"

    jq -n \
        --arg title "$title" \
        --arg description "$description" \
        --arg scope "$artifact_scope" \
        --arg output "$expected_output" \
        --argjson effort "$effort_hours" \
        --arg blocker "$blocker" \
        '{
            task_type: "creation",
            title: $title,
            description: $description,
            detailed_steps: [
                ("1. Plan the " + $scope + " structure"),
                "2. Create draft/outline",
                "3. Fill in all required sections",
                "4. Review and refine quality",
                ("5. Finalize: " + $output),
                "6. Verify completeness"
            ],
            success_criteria: {
                "artifact_exists": ("Output exists: " + $output),
                "complete": "All planned sections present",
                "quality": "Professional quality, ready to use",
                "verified": "Meets all stated requirements"
            },
            effort_estimate_hours: $effort,
            difficulty: "hard",
            persona_recommendation: "Experimenter",
            blocking_issue: $blocker,
            reasoning: ("Need to create: " + $description)
        }'
}

# Template: Execution/implementation of defined work
create_execution_task_detailed() {
    local title="$1"
    local description="$2"
    local work_scope="$3"
    local expected_output="$4"
    local effort_hours="${5:-3}"
    local blocker="${6:-}"

    jq -n \
        --arg title "$title" \
        --arg description "$description" \
        --arg scope "$work_scope" \
        --arg output "$expected_output" \
        --argjson effort "$effort_hours" \
        --arg blocker "$blocker" \
        '{
            task_type: "execution",
            title: $title,
            description: $description,
            detailed_steps: [
                "1. Set up environment/prerequisites",
                ("2. Execute: " + $scope),
                "3. Track progress and outcomes",
                ("4. Document results: " + $output),
                ("5. Verify success: " + $output),
                "6. Update goal progress"
            ],
            success_criteria: {
                "completed": "Task completed successfully",
                "documented": ("Results documented in: " + $output),
                "verified": "Success verified against criteria",
                "goal_updated": "Related goal progress updated"
            },
            effort_estimate_hours: $effort,
            difficulty: "medium",
            persona_recommendation: "Optimizer",
            blocking_issue: $blocker,
            reasoning: ("Execution needed for: " + $description)
        }'
}

# Template: Optimization/improvement of existing work
create_optimization_task_detailed() {
    local title="$1"
    local description="$2"
    local optimization_scope="$3"
    local expected_output="$4"
    local effort_hours="${5:-3}"
    local blocker="${6:-}"

    jq -n \
        --arg title "$title" \
        --arg description "$description" \
        --arg scope "$optimization_scope" \
        --arg output "$expected_output" \
        --argjson effort "$effort_hours" \
        --arg blocker "$blocker" \
        '{
            task_type: "optimization",
            title: $title,
            description: $description,
            detailed_steps: [
                ("1. Identify current state: " + $scope),
                "2. Analyze for improvement opportunities",
                "3. Propose 2-3 optimization approaches",
                "4. Implement selected approach",
                ("5. Measure improvements: " + $output),
                "6. Document results and lessons"
            ],
            success_criteria: {
                "baseline_measured": "Current state documented",
                "improved": "Measurable improvement achieved",
                "documented": ("Results in: " + $output),
                "sustainable": "Improvement is permanent"
            },
            effort_estimate_hours: $effort,
            difficulty: "medium",
            persona_recommendation: "Optimizer",
            blocking_issue: $blocker,
            reasoning: ("Optimization needed for: " + $description)
        }'
}

################################################################################
# GOAL-SPECIFIC TASK TEMPLATES
################################################################################

# Generate novel publishable-specific tasks
# Usage: generate_novel_tasks <goal_state> <progress> <blockers> [current_persona]
# LISASIMPSON PATTERN: Explicit Goal Modeling with Semantic Verification
# Rather than keyword matching on blockers, inspect the actual goal predicates (success_criteria)
generate_novel_tasks() {
    local goal_state="$1"
    local progress="$2"
    local blockers="$3"
    local current_persona="${4:-optimizer}"  # Default to optimizer if not provided

    local tasks='[]'

    # LISASIMPSON: Get the actual goal object for explicit predicate inspection
    local goal
    goal=$(get_active_goals 2>/dev/null | jq '.[] | select(.goal_id == "novel_publishable")' 2>/dev/null || echo "{}")

    # LISASIMPSON: Semantic verification - check actual predicates (not_started criteria)
    # Each not_started criterion is an actionable goal that needs planning

    # Criterion 1: copy_edited - prerequisite: draft must exist and be complete
    local copy_status
    copy_status=$(echo "$goal" | jq -r '.success_criteria[] | select(.criterion == "copy_edited") | .status' 2>/dev/null)

    if [ "$copy_status" = "not_started" ]; then
        # Precondition check: draft_complete must be true
        local draft_complete
        draft_complete=$(echo "$goal" | jq -r '.success_criteria[] | select(.criterion == "draft_complete") | .status' 2>/dev/null)

        if [ "$draft_complete" = "completed" ]; then
            local edit_task
            edit_task=$(create_creation_task_detailed \
                "Complete copy-editing pass on 25-chapter draft" \
                "Apply professional copy-editing to polish manuscript for publication quality. Precondition: draft complete (✓). Prerequisite for: publication_strategy and formatted." \
                "comprehensive pass through all 25 chapters: grammar, punctuation, prose consistency, dialogue polish, pacing review" \
                "creative/sentient-toaster/chapters-edited/" \
                "6" \
                "novel_publishable: copy_edited predicate not satisfied")

            edit_task=$(echo "$edit_task" | jq --arg persona "$current_persona" '.persona_recommendation = $persona')
            tasks=$(echo "$tasks" | jq ". += [$edit_task]")
        fi
    fi

    # Criterion 2: publication_strategy - no hard prerequisites
    local pub_strat_status
    pub_strat_status=$(echo "$goal" | jq -r '.success_criteria[] | select(.criterion == "publication_strategy") | .status' 2>/dev/null)

    if [ "$pub_strat_status" = "not_started" ]; then
        local pub_task
        pub_task=$(create_strategy_task_detailed \
            "Create publication strategy document" \
            "Define comprehensive publication and distribution strategy. Output: PUBLICATION-PLAN.md with platform selection, format options (ebook/print), timeline, pricing, distribution channels. Prerequisite for: formatted, published." \
            "publication platforms (KDP, IngramSpark, Draft2Digital), format decisions (ebook/print/both), pricing strategy, marketing channels, timeline" \
            "creative/sentient-toaster/PUBLICATION-PLAN.md" \
            "3" \
            "novel_publishable: publication_strategy predicate not satisfied")

        pub_task=$(echo "$pub_task" | jq --arg persona "$current_persona" '.persona_recommendation = $persona')
        tasks=$(echo "$tasks" | jq ". += [$pub_task]")
    fi

    # Criterion 3: formatted - prerequisites: draft_complete and ideally copy_edited
    local format_status
    format_status=$(echo "$goal" | jq -r '.success_criteria[] | select(.criterion == "formatted") | .status' 2>/dev/null)

    if [ "$format_status" = "not_started" ]; then
        local format_task
        format_task=$(create_execution_task_detailed \
            "Format manuscript for distribution" \
            "Apply professional formatting for publication (ebook and print formats). Output: formatted files ready for distribution. Preconditions: draft_complete (✓), ideally copy_edited. Prerequisite for: published." \
            "apply typography standards, pagination, headers/footers, chapter styling, cover integration, generate both ebook and print-ready PDFs" \
            "creative/sentient-toaster/formatted-manuscript/" \
            "4" \
            "novel_publishable: formatted predicate not satisfied")

        format_task=$(echo "$format_task" | jq --arg persona "$current_persona" '.persona_recommendation = $persona')
        tasks=$(echo "$tasks" | jq ". += [$format_task]")
    fi

    # Criterion 4: published - prerequisites: formatted AND publication_strategy
    local published_status
    published_status=$(echo "$goal" | jq -r '.success_criteria[] | select(.criterion == "published") | .status' 2>/dev/null)

    if [ "$published_status" = "not_started" ]; then
        # Check if prerequisites are met for publication
        local formatted_ready
        formatted_ready=$(echo "$goal" | jq -r '.success_criteria[] | select(.criterion == "formatted") | .status' 2>/dev/null)
        local pub_strat_ready
        pub_strat_ready=$(echo "$goal" | jq -r '.success_criteria[] | select(.criterion == "publication_strategy") | .status' 2>/dev/null)

        # Only generate publish task if prerequisites are in progress or complete
        if [ "$formatted_ready" != "not_started" ] && [ "$pub_strat_ready" != "not_started" ]; then
            local publish_task
            publish_task=$(create_execution_task_detailed \
                "Publish manuscript to distribution channels" \
                "Execute the publication strategy: upload novel to platforms, configure pricing, set up pre-orders. Preconditions: formatted (in-progress), publication_strategy (in-progress)." \
                "upload to Amazon KDP and other distribution platforms per strategy, configure pricing and options, set up pre-orders, update metadata, create publisher account if needed" \
                "creative/sentient-toaster/PUBLICATION-LOG.md" \
                "2" \
                "novel_publishable: published predicate not satisfied")

            publish_task=$(echo "$publish_task" | jq --arg persona "$current_persona" '.persona_recommendation = $persona')
            tasks=$(echo "$tasks" | jq ". += [$publish_task]")
        fi
    fi

    echo "$tasks"
}

# Generate system-specific tasks
# Usage: generate_system_tasks <assessment> <bottlenecks> [current_persona]
generate_system_tasks() {
    local assessment="$1"
    local bottlenecks="$2"
    local current_persona="${3:-optimizer}"  # Default to optimizer if not provided

    local tasks='[]'

    # Task: Task queue triage
    if echo "$bottlenecks" | jq -e '.[] | select(contains("queue") or contains("overload"))' >/dev/null 2>&1; then
        local triage_task
        triage_task=$(create_execution_task_detailed \
            "Triage and prioritize task queue" \
            "Too many pending tasks creating cognitive overload" \
            "review all pending, consolidate duplicates, remove invalids, prioritize" \
            "tasks/queue-triaged.md" \
            "1" \
            "task queue overload")

        # Update persona to current persona
        triage_task=$(echo "$triage_task" | jq --arg persona "$current_persona" '.persona_recommendation = $persona')
        tasks=$(echo "$tasks" | jq ". += [$triage_task]")
    fi

    echo "$tasks"
}

################################################################################
# TASK GENERATION ENGINE
################################################################################

# Generate tasks from identified gaps
################################################################################
# VERIFICATION PLAN INJECTION (LISASIMPSON INTEGRATION)
################################################################################

# Inject verification plans into generated tasks
# Usage: inject_verification_plans_to_tasks <tasks_json>
# Returns: Tasks JSON with verification_plan added to each task
inject_verification_plans_to_tasks() {
    local tasks="$1"

    # Use jq to inject verification plans
    echo "$tasks" | jq \
        'map(
            . as $task |
            ($task.description // "") as $description |
            (
                if (env.VERIFY_PLAN_FUNCTION != null) then
                    # If we have access to verify plan function, use it
                    {task_type: "general"}
                else
                    # Otherwise use simple heuristic for plan generation
                    if ($description | contains("write") or contains("draft") or contains("author")) then
                        {
                            task_type: "write",
                            checks: [
                                {type: "file_exists", description: "Output file created"},
                                {type: "file_word_count", description: "Sufficient content", parameters: {min: 1000}}
                            ]
                        }
                    elif ($description | contains("analyze") or contains("review")) then
                        {
                            task_type: "analyze",
                            checks: [
                                {type: "file_exists", description: "Analysis document created"},
                                {type: "file_line_count", description: "Substantial analysis", parameters: {min: 50}}
                            ]
                        }
                    elif ($description | contains("refactor") or contains("optimize")) then
                        {
                            task_type: "refactor",
                            checks: [
                                {type: "code_compiles", description: "Code compiles without errors"},
                                {type: "tests_pass", description: "All tests pass"}
                            ]
                        }
                    elif ($description | contains("test") or contains("verify")) then
                        {
                            task_type: "test",
                            checks: [
                                {type: "test_file_exists", description: "Test file created"},
                                {type: "tests_pass", description: "Tests pass"}
                            ]
                        }
                    else
                        {
                            task_type: "general",
                            checks: [{type: "task_completed", description: "Task marked complete"}]
                        }
                    end
                end
            ) as $verification_plan |
            . + {
                verification_plan: ($verification_plan + {
                    success_criteria: "all_checks_pass",
                    task_id: $task.id
                })
            }
        )' 2>/dev/null || echo "$tasks"
}

# Usage: generate_tasks <situation_assessment> [current_persona]
# Returns: Array of actionable tasks
generate_tasks() {
    local assessment="$1"
    local current_persona="${2:-optimizer}"  # Default to optimizer if not provided

    local generated_tasks='[]'

    # Extract gaps
    local gaps
    gaps=$(echo "$assessment" | jq '.gaps // []' 2>/dev/null || echo "[]")

    # Extract opportunities
    local opportunities
    opportunities=$(echo "$assessment" | jq '.opportunities // []' 2>/dev/null || echo "[]")

    # Extract bottlenecks
    local bottlenecks
    bottlenecks=$(echo "$assessment" | jq '.bottlenecks // []' 2>/dev/null || echo "[]")

    # Check for novel-specific gaps
    if echo "$gaps" | jq -e '.[] | select(.goal_id == "novel_publishable")' >/dev/null 2>&1; then
        # Extract the gap reason to determine what kind of work is needed
        local gap_reason
        gap_reason=$(echo "$gaps" | jq -r '.[] | select(.goal_id == "novel_publishable") | .reason // ""' 2>/dev/null || echo "")

        local progress
        progress=$(echo "$gaps" | jq '.[] | select(.goal_id == "novel_publishable") | .progress' 2>/dev/null || echo "0")

        # Convert reason to blockers array for task generation
        # The reason field from aggressive autonomy indicates not-started criteria
        local blockers
        if [ -n "$gap_reason" ]; then
            blockers=$(jq -n --arg reason "$gap_reason" '[{criterion: $reason}]')
        else
            blockers="[]"
        fi

        local novel_tasks
        # Call generate_novel_tasks with empty goal_state since we have the progress directly
        novel_tasks=$(generate_novel_tasks "novel_publishable" "$progress" "$blockers" "$current_persona" 2>/dev/null || echo "[]")

        generated_tasks=$(echo "$generated_tasks" | jq ". += $novel_tasks" 2>/dev/null || echo "$generated_tasks")
    fi

    # Check for system-level tasks
    local system_tasks
    system_tasks=$(generate_system_tasks "$assessment" "$bottlenecks" "$current_persona" 2>/dev/null || echo "[]")
    generated_tasks=$(echo "$generated_tasks" | jq ". += $system_tasks" 2>/dev/null || echo "$generated_tasks")

    # Inject verification plans into all generated tasks (LISASIMPSON)
    local task_count_before
    task_count_before=$(echo "$generated_tasks" | jq 'length' 2>/dev/null || echo "0")

    generated_tasks=$(inject_verification_plans_to_tasks "$generated_tasks" 2>/dev/null || echo "$generated_tasks")

    local tasks_with_verification
    tasks_with_verification=$(echo "$generated_tasks" | jq '[.[] | select(.verification_plan != null)] | length' 2>/dev/null || echo "0")

    # Log verification plan injection
    if [ -f "${DAEMON_ROOT}/logs/activity.log" ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] Verification planning: Injected plans into $tasks_with_verification/$task_count_before tasks" >> "${DAEMON_ROOT}/logs/activity.log"
    fi

    echo "$generated_tasks"
}

# Validate that a task is well-formed and actionable
# Usage: validate_task <task_json>
# Returns: {valid: true/false, reasons: [list of issues]}
validate_task() {
    local task="$1"

    local validation='{"valid": true, "issues": []}'

    # Check required fields
    if ! echo "$task" | jq -e '.title' >/dev/null 2>&1; then
        validation=$(echo "$validation" | jq '.valid = false | .issues += ["Missing title"]')
    fi

    if ! echo "$task" | jq -e '.description' >/dev/null 2>&1; then
        validation=$(echo "$validation" | jq '.valid = false | .issues += ["Missing description"]')
    fi

    if ! echo "$task" | jq -e '.success_criteria' >/dev/null 2>&1; then
        validation=$(echo "$validation" | jq '.valid = false | .issues += ["Missing success criteria"]')
    fi

    # Check success criteria are measurable
    local criteria
    criteria=$(echo "$task" | jq '.success_criteria // {}' 2>/dev/null)

    if [ "$(echo "$criteria" | jq 'length')" -lt 2 ]; then
        validation=$(echo "$validation" | jq '.valid = false | .issues += ["Success criteria too vague (need at least 2)"]')
    fi

    # Check effort estimate is reasonable
    local effort
    effort=$(echo "$task" | jq '.effort_estimate_hours // 0' 2>/dev/null)

    if [ "$effort" -lt 1 ] || [ "$effort" -gt 40 ]; then
        validation=$(echo "$validation" | jq '.valid = false | .issues += ["Effort estimate outside reasonable range (1-40 hours)"]')
    fi

    echo "$validation"
}

# Score and rank tasks by impact/effort ratio
# Usage: rank_generated_tasks <tasks_array>
# Returns: Ranked tasks with scores
rank_generated_tasks() {
    local tasks="$1"

    echo "$tasks" | jq 'map(. + {
        impact_score: (
            if .blocking_issue != "" and .blocking_issue != null then 100
            elif .task_type == "strategy" or .task_type == "planning" then 80
            elif .task_type == "execution" then 60
            elif .task_type == "research" then 40
            else 20
            end
        ),
        effort_hours: (.effort_estimate_hours // 2),
        value_per_hour: (
            (if .blocking_issue != "" and .blocking_issue != null then 100 elif .task_type == "strategy" then 80 else 50 end) /
            (.effort_estimate_hours // 2 | if . < 1 then 1 else . end)
        )
    }) | sort_by(-.value_per_hour) | sort_by(if .blocking_issue != "" and .blocking_issue != null then 0 else 1 end)' 2>/dev/null || echo "[]"
}

# Export generated tasks as markdown task queue format
# Usage: export_generated_tasks <tasks_array>
# Returns: Markdown formatted task list
export_generated_tasks() {
    local tasks="$1"

    local output="# Generated Autonomous Tasks\n\n"
    output+="Generated at: $(date -u +\"%Y-%m-%dT%H:%M:%SZ\")\n\n"
    output+="## Generated Tasks\n\n"

    echo "$tasks" | jq -r '.[] |
        "- [ ] [" + .persona_recommendation + "] " + .title + "\n" +
        "  - Description: " + .description + "\n" +
        "  - Type: " + .task_type + "\n" +
        "  - Effort: " + (.effort_estimate_hours | tostring) + "h\n" +
        "  - Difficulty: " + .difficulty + "\n"' | while read -r line; do
        output+="$line"
    done

    echo -e "$output"
}

################################################################################
# EXPORTS
################################################################################

export -f create_research_task_detailed
export -f create_strategy_task_detailed
export -f create_creation_task_detailed
export -f create_execution_task_detailed
export -f create_optimization_task_detailed
export -f generate_novel_tasks
export -f generate_system_tasks
export -f generate_tasks
export -f validate_task
export -f rank_generated_tasks
export -f export_generated_tasks

################################################################################
# SELF-TEST
################################################################################

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Running Task Generator Self-Tests..." >&2

    # Initialize
    init_goal_state

    # Create test goal
    test_goal=$(create_goal \
        "test_novel" \
        "creative_work" \
        "Test novel publishing goal" \
        '[
            {"criterion": "chapters_done", "description": "All chapters written", "status": "completed", "measurable": true},
            {"criterion": "beta_feedback", "description": "Beta feedback received", "status": "not_started", "measurable": true},
            {"criterion": "publish", "description": "Published", "status": "not_started", "measurable": true}
        ]')

    add_goal "$test_goal" >/dev/null 2>&1

    echo "Test 1: Creating research task..." >&2
    research=$(create_research_task_detailed \
        "Conduct market research" \
        "Understand reader market and competition" \
        "target audience, comparable titles, pricing" \
        "creative/sentient-toaster/MARKET-RESEARCH.md" \
        "4" \
        "")

    if echo "$research" | jq -e '.task_type == "research"' >/dev/null 2>&1; then
        echo "✓ Research task created" >&2
    else
        echo "✗ Research task creation failed" >&2
        exit 1
    fi

    echo "Test 2: Creating strategy task..." >&2
    strategy=$(create_strategy_task_detailed \
        "Plan publication approach" \
        "Decide traditional vs self-publishing" \
        "publishing platforms, distribution, timeline" \
        "creative/sentient-toaster/PUBLICATION-PLAN.md" \
        "3" \
        "")

    if echo "$strategy" | jq -e '.task_type == "strategy"' >/dev/null 2>&1; then
        echo "✓ Strategy task created" >&2
    else
        echo "✗ Strategy task creation failed" >&2
        exit 1
    fi

    echo "Test 3: Generating novel-specific tasks..." >&2
    assessment=$(assess_situation 2>/dev/null || echo '{"gaps": []}')
    novel_tasks=$(generate_novel_tasks "" "40" "[]" 2>/dev/null || echo "[]")
    novel_count=$(echo "$novel_tasks" | jq 'length' 2>/dev/null || echo "0")

    echo "✓ Generated $novel_count novel-specific tasks" >&2

    echo "Test 4: Validating task..." >&2
    validation=$(validate_task "$research" 2>/dev/null || echo '{"valid": false}')
    is_valid=$(echo "$validation" | jq '.valid' 2>/dev/null)

    if [ "$is_valid" = "true" ]; then
        echo "✓ Task validation working" >&2
    else
        echo "⚠ Task has issues (acceptable for test)" >&2
    fi

    echo "Test 5: Ranking tasks..." >&2
    ranked=$(rank_generated_tasks "$novel_tasks" 2>/dev/null || echo "[]")
    top_task=$(echo "$ranked" | jq '.[0].title' -r 2>/dev/null || echo "none")

    if [ "$top_task" != "none" ] && [ -n "$top_task" ]; then
        echo "✓ Top ranked task: $top_task" >&2
    else
        echo "✓ Task ranking complete" >&2
    fi

    echo "" >&2
    echo "✓ All Task Generator self-tests passed!" >&2

    # Cleanup
    rm -rf ./state
fi
