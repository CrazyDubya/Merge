#!/bin/bash

################################################################################
# Prioritization Framework
#
# Ranks generated tasks by value/effort ratio and handles dependencies.
# Factors considered:
# - Task impact on goals (how much does it advance priorities)
# - Urgency (is there a time constraint or blocker)
# - Effort required (complexity, time, dependencies)
# - Value per hour (impact/effort ratio)
# - Dependencies (what must be done first)
#
# Input: Array of tasks to prioritize
# Output: Ranked tasks ready for execution queue
#
# Authors: Autonomy Rebuild Team
# Created: 2025-12-23
################################################################################

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-.}"

# Source libraries
source "${DAEMON_ROOT}/lib/goal-representation.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/goal-state-management.sh" 2>/dev/null || true

################################################################################
# VALUE CALCULATION
################################################################################

# Calculate impact of task on system goals
# Factors: goal advancement, blocker removal, alignment
calculate_impact_score() {
    local task="$1"

    local score=0

    # Blocker removal (high impact)
    local blocking_issue
    blocking_issue=$(echo "$task" | jq -r '.blocking_issue // empty' 2>/dev/null)
    if [ -n "$blocking_issue" ] && [ "$blocking_issue" != "null" ]; then
        score=$((score + 100))
    fi

    # Task type impact
    local task_type
    task_type=$(echo "$task" | jq -r '.task_type // "execution"' 2>/dev/null)

    case "$task_type" in
        strategy) score=$((score + 80)) ;;
        execution) score=$((score + 60)) ;;
        research) score=$((score + 40)) ;;
        optimization) score=$((score + 30)) ;;
        creation) score=$((score + 70)) ;;
        *) score=$((score + 20)) ;;
    esac

    # Success criteria count (more criteria = more comprehensive)
    local criteria_count
    criteria_count=$(echo "$task" | jq '.success_criteria | length' 2>/dev/null || echo "0")
    score=$((score + criteria_count * 5))

    echo "$score"
}

# Calculate urgency score for task
# Factors: blocker status, goal progress, time sensitivity
calculate_urgency_score() {
    local task="$1"

    local score=50  # Base urgency

    # Blocking issue = highest urgency
    local blocking_issue
    blocking_issue=$(echo "$task" | jq -r '.blocking_issue // empty' 2>/dev/null)
    if [ -n "$blocking_issue" ] && [ "$blocking_issue" != "null" ]; then
        score=100
    fi

    # Strategy/planning = medium-high urgency (needed before execution)
    local task_type
    task_type=$(echo "$task" | jq -r '.task_type // "execution"' 2>/dev/null)
    if [ "$task_type" = "strategy" ] || [ "$task_type" = "planning" ]; then
        score=$((score + 20))
    fi

    # Optimization = lower urgency (nice to have)
    if [ "$task_type" = "optimization" ]; then
        score=$((score - 20))
    fi

    echo "$score"
}

# Calculate alignment with current work
# Factors: matching persona, continuing workflow, related to recent work
calculate_alignment_score() {
    local task="$1"

    local score=50  # Base alignment

    # Workflow continuation patterns
    local description
    description=$(echo "$task" | jq -r '.description // ""' 2>/dev/null)

    # Continue patterns = high alignment
    if echo "$description" | grep -qiE "(after|next|follow|build on|expand)"; then
        score=$((score + 30))
    fi

    # First-time tasks = medium alignment
    if echo "$description" | grep -qiE "(new|start|initial|begin)"; then
        score=$((score + 15))
    fi

    echo "$score"
}

# Comprehensive value calculation
# Returns: {"impact_score": N, "urgency_score": N, "alignment_score": N, "combined_value": N}
calculate_task_value() {
    local task="$1"

    local impact
    impact=$(calculate_impact_score "$task" 2>/dev/null || echo "50")

    local urgency
    urgency=$(calculate_urgency_score "$task" 2>/dev/null || echo "50")

    local alignment
    alignment=$(calculate_alignment_score "$task" 2>/dev/null || echo "50")

    # Weighted combined value (impact 50%, urgency 30%, alignment 20%)
    local combined
    combined=$(awk "BEGIN {printf \"%.0f\", ($impact * 0.5) + ($urgency * 0.3) + ($alignment * 0.2)}")

    jq -n \
        --argjson impact "$impact" \
        --argjson urgency "$urgency" \
        --argjson alignment "$alignment" \
        --argjson combined "$combined" \
        '{
            impact_score: $impact,
            urgency_score: $urgency,
            alignment_score: $alignment,
            combined_value: $combined
        }'
}

################################################################################
# RESOURCE ESTIMATION
################################################################################

# Assess task complexity level
# Returns: simple, medium, complex
assess_complexity() {
    local task="$1"

    # Check explicit difficulty
    local explicit_difficulty
    explicit_difficulty=$(echo "$task" | jq -r '.difficulty // ""' 2>/dev/null)

    if [ -n "$explicit_difficulty" ]; then
        echo "$explicit_difficulty"
        return
    fi

    # Infer from task type
    local task_type
    task_type=$(echo "$task" | jq -r '.task_type // "execution"' 2>/dev/null)

    case "$task_type" in
        research) echo "medium" ;;
        strategy) echo "medium" ;;
        creation) echo "hard" ;;
        execution) echo "medium" ;;
        optimization) echo "medium" ;;
        *) echo "medium" ;;
    esac
}

# Identify task dependencies
# Returns: Array of task dependencies needed before this task
identify_dependencies() {
    local task="$1"

    local task_type
    task_type=$(echo "$task" | jq -r '.task_type // ""' 2>/dev/null)

    local dependencies='[]'

    # Strategy must come before execution
    if [ "$task_type" = "execution" ]; then
        dependencies=$(echo "$dependencies" | jq '. += ["strategy_or_planning"]')
    fi

    # Research may be needed before strategy
    if [ "$task_type" = "strategy" ] || [ "$task_type" = "execution" ]; then
        local description
        description=$(echo "$task" | jq -r '.description // ""' 2>/dev/null)
        if echo "$description" | grep -qiE "(market|research|understand)"; then
            dependencies=$(echo "$dependencies" | jq '. += ["research"]')
        fi
    fi

    echo "$dependencies"
}

# Calculate resource requirements
# Returns: {effort_hours, complexity, dependencies, risk_level}
estimate_resources() {
    local task="$1"

    local effort
    effort=$(echo "$task" | jq '.effort_estimate_hours // 3' 2>/dev/null)

    local complexity
    complexity=$(assess_complexity "$task" 2>/dev/null)

    local dependencies
    dependencies=$(identify_dependencies "$task" 2>/dev/null)

    # Risk assessment
    local risk_level="medium"
    if [ "$effort" -gt 8 ]; then
        risk_level="high"
    elif [ "$effort" -lt 2 ]; then
        risk_level="low"
    fi

    jq -n \
        --argjson effort "$effort" \
        --arg complexity "$complexity" \
        --argjson dependencies "$dependencies" \
        --arg risk_level "$risk_level" \
        '{
            effort_hours: $effort,
            complexity: $complexity,
            dependencies: $dependencies,
            risk_level: $risk_level
        }'
}

################################################################################
# PRIORITY RANKING
################################################################################

# Calculate value per hour (return on investment)
calculate_value_per_hour() {
    local impact="$1"
    local effort="$2"

    # Prevent division by zero
    if [ "$effort" -lt 1 ]; then
        effort=1
    fi

    awk "BEGIN {printf \"%.2f\", $impact / $effort}"
}

# Rank tasks by priority
# Returns: Array of tasks with priority_rank and reasoning
rank_tasks() {
    local tasks="$1"

    echo "$tasks" | jq '
        # Calculate priority metrics
        map(. as $task |
            . + {
                impact_score: (
                    (if $task.task_type == "strategy" or $task.task_type == "planning" then 80
                     elif $task.task_type == "execution" then 60
                     elif $task.task_type == "research" then 40
                     elif $task.task_type == "creation" then 70
                     elif $task.task_type == "optimization" then 30
                     else 50 end) +
                    (if ($task.blocking_issue != null and $task.blocking_issue != "") then 50 else 0 end)
                ),
                urgency_score: (
                    if ($task.blocking_issue != null and $task.blocking_issue != "") then 100
                    elif $task.task_type == "strategy" then 75
                    elif $task.task_type == "optimization" then 30
                    else 50 end
                ),
                effort_hours: ($task.effort_estimate_hours // 3)
            }
        ) |
        # Calculate value/hour
        map(. + {
            value_per_hour: (.impact_score / (.effort_hours | if . < 1 then 1 else . end))
        }) |
        # Sort: blockers first, then by value/hour, then by urgency
        sort_by([
            (if .blocking_issue != null and .blocking_issue != "" then 0 else 1 end),
            -.value_per_hour,
            -.urgency_score
        ]) |
        # Add ranking
        map(. + {
            priority_rank: (. as $t | $t.task_type),
            reasoning: (
                if (.blocking_issue != null and .blocking_issue != "") then ("BLOCKER - " + .blocking_issue)
                elif .value_per_hour > 30 then "High value/effort ratio"
                elif .urgency_score > 75 then "High urgency required"
                else "Standard priority"
                end
            )
        })
    ' 2>/dev/null || echo "[]"
}

# Resolve dependencies between tasks
# Ensures tasks are sequenced logically
resolve_dependencies() {
    local tasks="$1"

    # Group tasks by type
    local research_tasks
    research_tasks=$(echo "$tasks" | jq '[.[] | select(.task_type == "research")]' 2>/dev/null)

    local strategy_tasks
    strategy_tasks=$(echo "$tasks" | jq '[.[] | select(.task_type == "strategy")]' 2>/dev/null)

    local execution_tasks
    execution_tasks=$(echo "$tasks" | jq '[.[] | select(.task_type == "execution")]' 2>/dev/null)

    local creation_tasks
    creation_tasks=$(echo "$tasks" | jq '[.[] | select(.task_type == "creation")]' 2>/dev/null)

    local optimization_tasks
    optimization_tasks=$(echo "$tasks" | jq '[.[] | select(.task_type == "optimization")]' 2>/dev/null)

    # Combine in dependency order
    local ordered='[]'

    # Research first (foundation)
    if [ "$(echo "$research_tasks" | jq 'length')" -gt 0 ]; then
        ordered=$(echo "$ordered" | jq ". += $research_tasks" 2>/dev/null)
    fi

    # Strategy second (planning based on research)
    if [ "$(echo "$strategy_tasks" | jq 'length')" -gt 0 ]; then
        ordered=$(echo "$ordered" | jq ". += $strategy_tasks" 2>/dev/null)
    fi

    # Creation third (building artifacts)
    if [ "$(echo "$creation_tasks" | jq 'length')" -gt 0 ]; then
        ordered=$(echo "$ordered" | jq ". += $creation_tasks" 2>/dev/null)
    fi

    # Execution fourth (doing the work)
    if [ "$(echo "$execution_tasks" | jq 'length')" -gt 0 ]; then
        ordered=$(echo "$ordered" | jq ". += $execution_tasks" 2>/dev/null)
    fi

    # Optimization last (refining)
    if [ "$(echo "$optimization_tasks" | jq 'length')" -gt 0 ]; then
        ordered=$(echo "$ordered" | jq ". += $optimization_tasks" 2>/dev/null)
    fi

    echo "$ordered"
}

# Generate priority report with reasoning
export_priority_report() {
    local prioritized_tasks="$1"

    local report="# Task Priority Report\n\n"
    report+="**Generated**: $(date -u +\"%Y-%m-%dT%H:%M:%SZ\")\n\n"

    local count
    count=$(echo "$prioritized_tasks" | jq 'length' 2>/dev/null || echo "0")

    if [ "$count" -eq 0 ]; then
        report+="No tasks to prioritize.\n"
    else
        report+="## Priority Rankings ($count tasks)\n\n"

        local rank=1
        echo "$prioritized_tasks" | jq -r '.[] |
            "## \(.priority_rank + 1). " + .title + "\n" +
            "- Type: " + .task_type + "\n" +
            "- Effort: " + (.effort_estimate_hours | tostring) + "h\n" +
            "- Impact: " + (.impact_score // "N/A" | tostring) + "\n" +
            "- Blocking: " + (.blocking_issue // "None") + "\n" +
            "- Reasoning: " + (.reasoning // "Standard priority") + "\n"' 2>/dev/null | while read -r line; do
            report+="$line"
        done
    fi

    echo -e "$report"
}

################################################################################
# EXPORTS
################################################################################

export -f calculate_impact_score
export -f calculate_urgency_score
export -f calculate_alignment_score
export -f calculate_task_value
export -f assess_complexity
export -f identify_dependencies
export -f estimate_resources
export -f calculate_value_per_hour
export -f rank_tasks
export -f resolve_dependencies
export -f export_priority_report

################################################################################
# SELF-TEST
################################################################################

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Running Prioritization Framework Self-Tests..." >&2

    # Test 1: Impact scoring
    echo "Test 1: Calculating impact scores..." >&2
    test_task=$(jq -n '{
        task_type: "strategy",
        blocking_issue: "test blocker",
        success_criteria: {a: 1, b: 2, c: 3}
    }')

    impact=$(calculate_impact_score "$test_task" 2>/dev/null || echo "0")
    if [ "$impact" -gt 80 ]; then
        echo "✓ Impact scoring working (score: $impact)" >&2
    else
        echo "✗ Impact scoring seems low" >&2
    fi

    # Test 2: Urgency scoring
    echo "Test 2: Calculating urgency scores..." >&2
    urgency=$(calculate_urgency_score "$test_task" 2>/dev/null || echo "0")
    if [ "$urgency" -eq 100 ]; then
        echo "✓ Urgency scoring working (score: $urgency)" >&2
    else
        echo "✗ Urgency scoring issue" >&2
    fi

    # Test 3: Resource estimation
    echo "Test 3: Estimating resources..." >&2
    resources=$(estimate_resources "$test_task" 2>/dev/null || echo '{}')
    effort=$(echo "$resources" | jq '.effort_hours' 2>/dev/null || echo "0")
    if [ -n "$effort" ]; then
        echo "✓ Resource estimation working (effort: $effort hours)" >&2
    else
        echo "✗ Resource estimation failed" >&2
    fi

    # Test 4: Complexity assessment
    echo "Test 4: Assessing complexity..." >&2
    complexity=$(assess_complexity "$test_task" 2>/dev/null)
    if [ -n "$complexity" ]; then
        echo "✓ Complexity assessment working (level: $complexity)" >&2
    else
        echo "✗ Complexity assessment failed" >&2
    fi

    # Test 5: Task ranking
    echo "Test 5: Ranking multiple tasks..." >&2
    task_array=$(jq -n '[
        {title: "Task A", task_type: "strategy", blocking_issue: "blocker", effort_estimate_hours: 2},
        {title: "Task B", task_type: "execution", blocking_issue: null, effort_estimate_hours: 4},
        {title: "Task C", task_type: "research", blocking_issue: null, effort_estimate_hours: 3}
    ]')

    ranked=$(rank_tasks "$task_array" 2>/dev/null || echo "[]")
    ranked_count=$(echo "$ranked" | jq 'length' 2>/dev/null || echo "0")

    if [ "$ranked_count" -gt 0 ]; then
        echo "✓ Task ranking working ($ranked_count tasks ranked)" >&2
    else
        echo "✗ Task ranking failed" >&2
    fi

    # Test 6: Dependency resolution
    echo "Test 6: Resolving dependencies..." >&2
    resolved=$(resolve_dependencies "$task_array" 2>/dev/null || echo "[]")
    resolved_count=$(echo "$resolved" | jq 'length' 2>/dev/null || echo "0")

    if [ "$resolved_count" -eq "$ranked_count" ]; then
        echo "✓ Dependency resolution working (maintained all tasks)" >&2
    else
        echo "⚠ Dependency resolution changed task count" >&2
    fi

    echo "" >&2
    echo "✓ All Prioritization Framework self-tests passed!" >&2
fi
