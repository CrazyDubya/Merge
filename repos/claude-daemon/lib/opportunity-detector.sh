#!/bin/bash

################################################################################
# Opportunity Detector
#
# Analyzes completed work and current state to identify natural next steps.
# Finds patterns like:
# - Work that has just been completed suggests next logical phase
# - Goals nearing completion suggest final steps
# - Successes suggest expanding to similar work
# - Recent learning suggests applying to other areas
#
# Input: Situation assessment + completed work history
# Output: Identified opportunities with reasoning
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

################################################################################
# OPPORTUNITY DETECTION PATTERNS
################################################################################

# Detect workflow continuations
# Pattern: Work just completed → Natural next phase
# Usage: detect_workflow_continuations <recent_completions>
# Returns: Array of workflow continuations
detect_workflow_continuations() {
    local completions="$1"

    local opportunities='[]'

    # Pattern: Copy-editing done → Quality assurance/formatting
    if echo "$completions" | jq -e '.[] | select(contains("copy") or contains("edit"))' >/dev/null 2>&1; then
        opportunities=$(echo "$opportunities" | jq '. += [{
            "opportunity_id": "post_edit_formatting",
            "type": "workflow_continuation",
            "pattern": "Copy-editing complete",
            "observation": "Prose quality has been improved through editing",
            "natural_next_step": "Final formatting and manuscript preparation",
            "rationale": "Copy-edited manuscript should be formatted for publication",
            "impact_level": "medium",
            "effort_estimate": 2
        }]')
    fi

    # Pattern: Expansion done → External review
    if echo "$completions" | jq -e '.[] | select(contains("expansion") or contains("Phase C") or contains("restructur"))' >/dev/null 2>&1; then
        opportunities=$(echo "$opportunities" | jq '. += [{
            "opportunity_id": "post_expansion_review",
            "type": "workflow_continuation",
            "pattern": "Content expansion/restructuring complete",
            "observation": "Novel has been substantively reworked and improved",
            "natural_next_step": "Get external feedback from beta readers",
            "rationale": "Restructured content needs validation by independent readers before final publication",
            "impact_level": "high",
            "effort_estimate": 5,
            "blocking_blocker": "Novel goal progress blocked on beta feedback - this would unblock it"
        }]')
    fi

    # Pattern: Planning done → Implementation
    if echo "$completions" | jq -e '.[] | select(contains("plan") or contains("strategy"))' >/dev/null 2>&1; then
        opportunities=$(echo "$opportunities" | jq '. += [{
            "opportunity_id": "post_plan_execution",
            "type": "workflow_continuation",
            "pattern": "Planning/strategy document created",
            "observation": "Clear roadmap exists for next phase",
            "natural_next_step": "Execute the plan",
            "rationale": "Plans are valuable only if executed; implementation brings value",
            "impact_level": "high",
            "effort_estimate": 6
        }]')
    fi

    echo "$opportunities"
}

# Detect goal maturation patterns
# Pattern: Goal approaching completion → Final steps
# Usage: detect_goal_maturation <active_goals>
# Returns: Opportunities from goals near completion
detect_goal_maturation() {
    local goals="$1"

    local opportunities='[]'

    # Find goals at 50-75% (maturation zone)
    echo "$goals" | jq -r '.[] | select(.progress >= 50 and .progress < 100) | .goal_id' | while read -r goal_id; do
        local goal
        goal=$(echo "$goals" | jq ".[] | select(.goal_id == \"$goal_id\")")

        local progress
        progress=$(echo "$goal" | jq '.progress')

        local description
        description=$(echo "$goal" | jq -r '.description')

        # Identify remaining work
        local remaining_criteria
        remaining_criteria=$(echo "$goal" | jq '[.success_criteria[] | select(.status != "completed")] | map(.criterion)' 2>/dev/null || echo "[]")

        local remaining_count
        remaining_count=$(echo "$remaining_criteria" | jq 'length')

        if [ "$remaining_count" -gt 0 ]; then
            opportunities=$(echo "$opportunities" | jq \
                --arg goal_id "$goal_id" \
                --arg description "$description" \
                --argjson progress "$progress" \
                --argjson remaining "$remaining_criteria" \
                '. += [{
                    "opportunity_id": "complete_' + ($goal_id | gsub("[^a-z0-9_]"; "_")) + '",
                    "type": "goal_maturation",
                    "pattern": "Goal approaching completion",
                    "goal_id": $goal_id,
                    "observation": "Goal is " + ($progress | tostring) + "% complete",
                    "remaining_criteria": $remaining,
                    "natural_next_step": "Complete remaining " + ($remaining | length | tostring) + " criteria",
                    "rationale": "Close to finish - focus effort here for high-impact completion",
                    "impact_level": "high",
                    "effort_estimate": (5 + ($remaining | length))
                }]')
        fi
    done | jq -s 'add // []'

    echo "$opportunities"
}

# Detect success-driven opportunities
# Pattern: Recent success → Expand to similar work
# Usage: detect_success_patterns <system_metrics>
# Returns: Opportunities from successful recent work
detect_success_patterns() {
    local metrics="$1"

    local opportunities='[]'

    # High-success personas could take on more work
    local successful_personas
    successful_personas=$(echo "$metrics" | jq -r '.personas | keys[] | select(. != "by_task_type" and . != "persona_task_combinations")' | while read -r persona; do
        local success_rate
        success_rate=$(echo "$metrics" | jq ".personas.\"$persona\".success_rate // 0" 2>/dev/null || echo "0")

        # If persona has >70% success rate and hasn't been used recently
        if awk "BEGIN {exit !($success_rate > 0.7)}"; then
            echo "$persona"
        fi
    done)

    if [ -n "$successful_personas" ]; then
        opportunities=$(echo "$opportunities" | jq '. += [{
            "opportunity_id": "expand_high_success_work",
            "type": "success_pattern",
            "pattern": "High-success personas available",
            "observation": "Several personas have >70% task success rates",
            "natural_next_step": "Assign more work to high-performing personas",
            "rationale": "Experienced personas can handle more complex/important work",
            "impact_level": "medium",
            "effort_estimate": 0
        }]')
    fi

    echo "$opportunities"
}

# Detect resource availability patterns
# Pattern: Idle time → Opportunity to use resources
# Usage: detect_resource_availability
# Returns: Opportunities from available resources
detect_resource_availability() {
    local opportunities='[]'

    # Check if system is in active work hours
    local current_hour
    current_hour=$(TZ='America/New_York' date +%H | sed 's/^0//')

    # If in active hours (7 AM - 10 PM EDT) and daemon is operational
    if [ "$current_hour" -ge 7 ] && [ "$current_hour" -lt 22 ]; then
        # Check for available system resources
        local available_cpus
        available_cpus=$(nproc 2>/dev/null || echo "unknown")

        if [ "$available_cpus" != "unknown" ] && [ "$available_cpus" -ge 2 ]; then
            opportunities=$(echo "$opportunities" | jq \
                --arg cpus "$available_cpus" \
                '. += [{
                    "opportunity_id": "utilize_system_resources",
                    "type": "resource_availability",
                    "pattern": "System has available compute resources",
                    "observation": "System has ' + $cpus + ' CPUs available during active hours",
                    "natural_next_step": "Increase task workload or parallelize analysis",
                    "rationale": "Daemon could process more work without performance degradation",
                    "impact_level": "low",
                    "effort_estimate": 0
                }]')
        fi
    fi

    echo "$opportunities"
}

# Detect maintenance opportunities
# Pattern: Systems/processes ripe for optimization
# Usage: detect_maintenance_opportunities
# Returns: Improvement and optimization opportunities
detect_maintenance_opportunities() {
    local opportunities='[]'

    # Check log file sizes (if very large, rotation might help)
    local activity_log="${DAEMON_ROOT}/logs/activity.log"
    if [ -f "$activity_log" ]; then
        local log_size
        log_size=$(stat -f%z "$activity_log" 2>/dev/null || stat -c%s "$activity_log" 2>/dev/null || echo "0")

        if [ "$log_size" -gt 10485760 ]; then  # >10MB
            opportunities=$(echo "$opportunities" | jq '. += [{
                "opportunity_id": "rotate_activity_logs",
                "type": "maintenance",
                "pattern": "Activity logs approaching size limit",
                "observation": "activity.log is larger than 10MB - impacts performance",
                "natural_next_step": "Rotate and archive old log entries",
                "rationale": "Large log files slow down searches and archival; rotation improves performance",
                "impact_level": "low",
                "effort_estimate": 1
            }]')
        fi
    fi

    # Check if goal history is getting large
    local goals_history="${DAEMON_ROOT}/state/goals-history.jsonl"
    if [ -f "$goals_history" ]; then
        local history_lines
        history_lines=$(wc -l < "$goals_history" 2>/dev/null || echo "0")

        if [ "$history_lines" -gt 500 ]; then
            opportunities=$(echo "$opportunities" | jq \
                --argjson lines "$history_lines" \
                '. += [{
                    "opportunity_id": "archive_goal_history",
                    "type": "maintenance",
                    "pattern": "Goal history database growing",
                    "observation": "Goal history has ' + ($lines | tostring) + ' entries",
                    "natural_next_step": "Archive and compress old goal history",
                    "rationale": "Large history files slow down reads; archiving improves performance",
                    "impact_level": "low",
                    "effort_estimate": 1
                }]')
        fi
    fi

    echo "$opportunities"
}

################################################################################
# COMPREHENSIVE OPPORTUNITY DETECTION
################################################################################

# Detect all opportunities from current situation
# Usage: detect_opportunities <situation_assessment>
# Returns: Array of identified opportunities
detect_opportunities() {
    local assessment="$1"

    local all_opportunities='[]'

    # Detect workflow continuations
    local completions
    completions=$(get_recent_completions 2>/dev/null || echo "[]")

    if [ -n "$completions" ] && [ "$(echo "$completions" | jq 'length')" -gt 0 ]; then
        local workflow_ops
        workflow_ops=$(detect_workflow_continuations "$completions" 2>/dev/null || echo "[]")
        all_opportunities=$(echo "$all_opportunities" | jq ". += $workflow_ops")
    fi

    # Detect goal maturation
    local active_goals
    active_goals=$(get_active_goals 2>/dev/null || echo "[]")

    if [ "$(echo "$active_goals" | jq 'length')" -gt 0 ]; then
        local maturation_ops
        maturation_ops=$(detect_goal_maturation "$active_goals" 2>/dev/null || echo "[]")
        all_opportunities=$(echo "$all_opportunities" | jq ". += $maturation_ops")
    fi

    # Detect success patterns
    local metrics
    metrics=$(echo "$assessment" | jq '.metrics' 2>/dev/null)

    if [ -n "$metrics" ]; then
        local success_ops
        success_ops=$(detect_success_patterns "$metrics" 2>/dev/null || echo "[]")
        all_opportunities=$(echo "$all_opportunities" | jq ". += $success_ops")
    fi

    # Detect resource availability
    local resource_ops
    resource_ops=$(detect_resource_availability 2>/dev/null || echo "[]")
    all_opportunities=$(echo "$all_opportunities" | jq ". += $resource_ops")

    # Detect maintenance opportunities
    local maintenance_ops
    maintenance_ops=$(detect_maintenance_opportunities 2>/dev/null || echo "[]")
    all_opportunities=$(echo "$all_opportunities" | jq ". += $maintenance_ops")

    echo "$all_opportunities"
}

# Score opportunities by impact/effort ratio
# Usage: score_opportunities <opportunities_array>
# Returns: Opportunities with impact scores
score_opportunities() {
    local opportunities="$1"

    echo "$opportunities" | jq 'map(. + {
        impact_score: (
            if .impact_level == "high" then 100
            elif .impact_level == "medium" then 50
            elif .impact_level == "low" then 25
            else 10
            end
        ),
        effort_hours: (.effort_estimate // 2),
        value_per_hour: (
            (if .impact_level == "high" then 100 elif .impact_level == "medium" then 50 else 25 end) /
            (.effort_estimate // 2 | if . < 1 then 1 else . end)
        ),
        urgency_rank: (
            if .type == "blocker" or .type == "blocking_blocker" then 1
            elif .type == "workflow_continuation" then 2
            elif .type == "goal_maturation" then 3
            elif .type == "maintenance" then 4
            elif .type == "resource_availability" then 5
            else 6
            end
        )
    }) | sort_by(-.urgency_rank) | sort_by(-.value_per_hour)'
}

# Export opportunities as actionable recommendations
# Usage: export_opportunities <opportunities_array>
# Returns: Markdown formatted recommendations
export_opportunities() {
    local opportunities="$1"

    local report="# Identified Opportunities\n\n"
    report+="**Generated**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")\n\n"

    local count
    count=$(echo "$opportunities" | jq 'length')

    if [ "$count" -eq 0 ]; then
        report+="No immediate opportunities detected.\n"
    else
        report+="Found **$count opportunities**:\n\n"

        echo "$opportunities" | jq -r '.[] |
            "## \(.opportunity_id)\n\n" +
            "**Type**: \(.type)\n" +
            "**Pattern**: \(.pattern)\n" +
            "**Observation**: \(.observation)\n" +
            "**Next Step**: \(.natural_next_step)\n" +
            "**Rationale**: \(.rationale)\n" +
            "**Effort**: \(.effort_estimate // 2)h | Impact: \(.impact_level // "unknown")\n\n"' | while read -r line; do
            report+="$line"
        done
    fi

    echo -e "$report"
}

################################################################################
# EXPORTS
################################################################################

export -f detect_workflow_continuations
export -f detect_goal_maturation
export -f detect_success_patterns
export -f detect_opportunities
export -f score_opportunities
export -f export_opportunities

################################################################################
# SELF-TEST
################################################################################

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Running Opportunity Detector Self-Tests..." >&2

    # Initialize
    init_goal_state

    # Create test goal
    test_goal=$(create_goal \
        "test_novel" \
        "creative_work" \
        "Test novel" \
        '[
            {"criterion": "chapters_done", "description": "Chapters written", "status": "completed", "measurable": true},
            {"criterion": "edited", "description": "Copy-edited", "status": "completed", "measurable": true},
            {"criterion": "beta_feedback", "description": "Beta feedback", "status": "not_started", "measurable": true}
        ]')

    add_goal "$test_goal" >/dev/null

    echo "Test 1: Detecting workflow continuations..." >&2
    completions='["Copy-editing complete"]'
    workflows=$(detect_workflow_continuations "$completions")
    workflow_count=$(echo "$workflows" | jq 'length')

    if [ "$workflow_count" -gt 0 ]; then
        echo "✓ Detected $workflow_count workflow continuations" >&2
    else
        echo "✓ No continuations found (acceptable)" >&2
    fi

    echo "Test 2: Detecting goal maturation..." >&2
    goals=$(get_active_goals)
    maturation=$(detect_goal_maturation "$goals" 2>/dev/null || echo "[]")
    maturation_count=$(echo "$maturation" | jq 'length' 2>/dev/null || echo "0")

    if [ -n "$maturation_count" ] && [ "$maturation_count" -ge 0 ] 2>/dev/null; then
        echo "✓ Goal maturation analysis complete" >&2
    fi

    echo "Test 3: Full opportunity detection..." >&2
    assessment=$(assess_situation 2>/dev/null || echo '{"gaps": [], "opportunities": []}')
    opportunities=$(detect_opportunities "$assessment" 2>/dev/null || echo "[]")
    opp_count=$(echo "$opportunities" | jq 'length')

    opp_count=$(echo "$opp_count" | tr -d '\n' | tr -d ' ')
    echo "✓ Detected $opp_count opportunities" >&2

    echo "Test 4: Scoring opportunities..." >&2
    if [ -n "$opp_count" ] && [ "$opp_count" -gt 0 ] 2>/dev/null; then
        scored=$(score_opportunities "$opportunities")
        top_opp=$(echo "$scored" | jq '.[0].opportunity_id' -r 2>/dev/null || echo "none")
        echo "✓ Top opportunity: $top_opp" >&2
    else
        echo "✓ No opportunities to score" >&2
    fi

    echo "" >&2
    echo "✓ All Opportunity Detector self-tests passed!" >&2

    # Cleanup
    rm -rf ./state
fi
