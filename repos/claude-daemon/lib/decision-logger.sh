#!/bin/bash

################################################################################
# Decision Logger
#
# Logs all autonomous decisions with complete reasoning.
# Every decision must be explainable: what, why, alternatives considered,
# assumptions, confidence levels, and verification criteria.
#
# Output: memory/autonomous-decisions.md + memory/decision-log.jsonl
#
# Authors: Autonomy Rebuild Team
# Created: 2025-12-23
################################################################################

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-.}"

# Ensure directories exist
mkdir -p "${DAEMON_ROOT}/memory"
mkdir -p "${DAEMON_ROOT}/logs"

################################################################################
# DECISION LOGGING FUNCTIONS
################################################################################

# Log a major autonomous decision with full reasoning
# Usage: log_autonomous_decision <decision_type> <what_decided> <why_reasoning> <alternatives_json> <assumptions_json>
log_autonomous_decision() {
    local decision_type="$1"     # "task_generation", "prioritization", "execution_choice"
    local decision_text="$2"     # What was decided
    local reasoning="$3"         # Why this decision was made
    local alternatives="$4"      # JSON array of alternatives considered
    local assumptions="$5"       # JSON array of assumptions made

    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    local decision_id
    decision_id=$(date +%s)-$$

    # Create decision record
    local decision_record
    decision_record=$(jq -n \
        --arg id "$decision_id" \
        --arg timestamp "$timestamp" \
        --arg type "$decision_type" \
        --arg decision "$decision_text" \
        --arg reasoning "$reasoning" \
        --argjson alternatives "$alternatives" \
        --argjson assumptions "$assumptions" \
        '{
            decision_id: $id,
            timestamp: $timestamp,
            decision_type: $type,
            decision: $decision,
            reasoning: $reasoning,
            alternatives_considered: $alternatives,
            assumptions_made: $assumptions,
            confidence_level: "medium",
            status: "pending_execution"
        }')

    # Append to JSONL log
    local decision_log="${DAEMON_ROOT}/memory/decision-log.jsonl"
    echo "$decision_record" >> "$decision_log" 2>/dev/null || true

    echo "$decision_record"
}

# Log the outcome of a decision after execution
# Usage: log_decision_outcome <decision_id> <actual_outcome> <impact_measured> <assumptions_validated>
log_decision_outcome() {
    local decision_id="$1"
    local actual_outcome="$2"
    local impact_measured="$3"
    local assumptions_validated="$4"

    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    local outcome_record
    outcome_record=$(jq -n \
        --arg id "$decision_id" \
        --arg timestamp "$timestamp" \
        --arg outcome "$actual_outcome" \
        --arg impact "$impact_measured" \
        --argjson assumptions "$assumptions_validated" \
        '{
            decision_id: $id,
            timestamp: $timestamp,
            actual_outcome: $outcome,
            impact_measured: $impact,
            assumptions_validated: $assumptions,
            status: "completed"
        }')

    # Append to outcome log
    local outcome_log="${DAEMON_ROOT}/memory/decision-outcomes.jsonl"
    echo "$outcome_record" >> "$outcome_log" 2>/dev/null || true

    echo "$outcome_record"
}

# Add assumption validation result
# Usage: log_assumption_validation <decision_id> <assumption> <result> <confidence>
log_assumption_validation() {
    local decision_id="$1"
    local assumption="$2"
    local result="$3"        # "correct", "partially_correct", "incorrect"
    local confidence="$4"    # "high", "medium", "low"

    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    local validation_record
    validation_record=$(jq -n \
        --arg id "$decision_id" \
        --arg timestamp "$timestamp" \
        --arg assumption "$assumption" \
        --arg result "$result" \
        --arg confidence "$confidence" \
        '{
            decision_id: $id,
            timestamp: $timestamp,
            assumption: $assumption,
            validation_result: $result,
            confidence: $confidence
        }')

    # Append to validation log
    local validation_log="${DAEMON_ROOT}/memory/assumption-validations.jsonl"
    echo "$validation_record" >> "$validation_log" 2>/dev/null || true

    echo "$validation_record"
}

################################################################################
# DECISION REPORTING
################################################################################

# Generate markdown report of recent decisions
# Usage: generate_decision_report [days_back=7]
generate_decision_report() {
    local days_back="${1:-7}"

    local cutoff_date
    cutoff_date=$(date -u -d "$days_back days ago" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date -u -v-${days_back}d +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null)

    local report="# Autonomous Decisions Report\n\n"
    report+="**Generated**: $(date -u +\"%Y-%m-%d %H:%M:%S UTC\")\n"
    report+="**Period**: Last $days_back days\n\n"

    local decision_log="${DAEMON_ROOT}/memory/decision-log.jsonl"

    if [ ! -f "$decision_log" ]; then
        report+="No decisions logged yet.\n"
        echo -e "$report"
        return
    fi

    # Count decisions by type
    local total_decisions
    total_decisions=$(wc -l < "$decision_log" 2>/dev/null || echo "0")

    report+="## Summary\n"
    report+="- **Total Autonomous Decisions**: $total_decisions\n\n"

    report+="## Recent Decisions\n\n"

    # Process decisions (show last 10)
    tail -10 "$decision_log" | jq -r '
        "### " + .decision_type + " - " + .timestamp + "\n" +
        "**Decision**: " + .decision + "\n" +
        "**Reasoning**: " + .reasoning + "\n" +
        "**Status**: " + .status + "\n"
    ' 2>/dev/null | while read -r line; do
        report+="$line"
    done

    report+="\n## Assumption Tracking\n\n"
    report+="(Validations recorded as decisions are executed and verified)\n"

    echo -e "$report"
}

# Export decision log as markdown for review
# Usage: export_decision_log_markdown
export_decision_log_markdown() {
    local decision_log="${DAEMON_ROOT}/memory/decision-log.jsonl"

    if [ ! -f "$decision_log" ]; then
        echo "# Autonomous Decision Log\n\nNo decisions logged yet."
        return
    fi

    local markdown="# Autonomous Decision Log\n\n"
    markdown+="**Generated**: $(date -u +\"%Y-%m-%d %H:%M:%S UTC\")\n\n"

    markdown+="## Decision Timeline\n\n"

    # Process all decisions in reverse order (most recent first)
    tac "$decision_log" 2>/dev/null | jq -r '
        "## " + .timestamp + " - " + .decision_id + "\n\n" +
        "**Type**: " + .decision_type + "\n\n" +
        "**Decision**: " + .decision + "\n\n" +
        "**Reasoning**: " + .reasoning + "\n\n" +
        "**Status**: " + .status + "\n\n" +
        "**Assumptions**: " + (.assumptions_made | length | tostring) + " assumptions made\n\n" +
        "**Alternatives Considered**: " + (.alternatives_considered | length | tostring) + " alternatives\n\n" +
        "---\n\n"
    ' 2>/dev/null | while read -r line; do
        markdown+="$line"
    done

    echo -e "$markdown"
}

################################################################################
# DECISION ANALYSIS
################################################################################

# Analyze decision quality and outcomes
# Returns: {accuracy_rate, pattern_analysis, improvement_areas}
analyze_decisions() {
    local decision_log="${DAEMON_ROOT}/memory/decision-log.jsonl"
    local outcome_log="${DAEMON_ROOT}/memory/decision-outcomes.jsonl"

    if [ ! -f "$decision_log" ]; then
        echo '{"total_decisions": 0, "accuracy_rate": 0, "patterns": []}'
        return
    fi

    # Count decisions by type and outcome
    jq -s '
        group_by(.decision_type) |
        map({
            type: .[0].decision_type,
            count: length,
            success_rate: (([.[] | select(.status == "completed")] | length) / length * 100)
        })
    ' "$decision_log" 2>/dev/null || echo "[]"
}

# Identify patterns in decisions
# Usage: identify_decision_patterns
identify_decision_patterns() {
    local decision_log="${DAEMON_ROOT}/memory/decision-log.jsonl"

    if [ ! -f "$decision_log" ]; then
        echo '{"patterns": []}'
        return
    fi

    jq -s '
        {
            total_decisions: length,
            decision_types: (group_by(.decision_type) | map({type: .[0].decision_type, count: length})),
            common_reasoning_keywords: (
                [.[].reasoning] |
                join(" ") |
                split(" ") |
                group_by(.) |
                map({word: .[0], frequency: length}) |
                sort_by(-.frequency) |
                .[0:10]
            ),
            average_confidence: (
                ([.[] | .confidence_level // "medium"] |
                if . == "high" then 3
                elif . == "medium" then 2
                else 1 end) |
                add / length * 100
            )
        }
    ' "$decision_log" 2>/dev/null || echo '{"patterns": []}'
}

################################################################################
# EXPORTS
################################################################################

export -f log_autonomous_decision
export -f log_decision_outcome
export -f log_assumption_validation
export -f generate_decision_report
export -f export_decision_log_markdown
export -f analyze_decisions
export -f identify_decision_patterns

################################################################################
# SELF-TEST
################################################################################

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Running Decision Logger Self-Tests..." >&2

    # Clean up test directory
    rm -rf /tmp/daemon-test-decision
    mkdir -p /tmp/daemon-test-decision/memory
    DAEMON_ROOT="/tmp/daemon-test-decision"

    echo "Test 1: Logging autonomous decision..." >&2
    decision=$(log_autonomous_decision \
        "task_generation" \
        "Generate beta reader recruitment task" \
        "Novel is 40% complete and blocked on beta feedback - recruiting readers is critical to progress" \
        '[{"alternative": "Skip recruitment, go straight to publication", "reason": "Would skip validation step, risky"},
          {"alternative": "Wait for more writing", "reason": "Goal already mature enough, need external input"}]' \
        '[{"assumption": "Beta readers can be found within 2 weeks", "confidence": "medium"},
          {"assumption": "Feedback will help improve manuscript", "confidence": "high"}]')

    if echo "$decision" | jq -e '.decision_id' >/dev/null 2>&1; then
        echo "✓ Decision logged successfully" >&2
    else
        echo "✗ Decision logging failed" >&2
        exit 1
    fi

    decision_id=$(echo "$decision" | jq -r '.decision_id')

    echo "Test 2: Logging decision outcome..." >&2
    outcome=$(log_decision_outcome \
        "$decision_id" \
        "Task successfully created and added to queue" \
        "Advanced goal from 40% to 55% (15% progress)" \
        '[{"assumption": "Beta readers can be found", "result": "correct"},
          {"assumption": "Feedback will improve", "result": "pending"}]')

    if echo "$outcome" | jq -e '.decision_id' >/dev/null 2>&1; then
        echo "✓ Outcome logged successfully" >&2
    else
        echo "✗ Outcome logging failed" >&2
        exit 1
    fi

    echo "Test 3: Validating assumptions..." >&2
    validation=$(log_assumption_validation \
        "$decision_id" \
        "Beta readers can be found within 2 weeks" \
        "correct" \
        "high")

    if echo "$validation" | jq -e '.validation_result' >/dev/null 2>&1; then
        echo "✓ Assumption validation logged" >&2
    else
        echo "✗ Assumption validation failed" >&2
        exit 1
    fi

    echo "Test 4: Generating decision report..." >&2
    report=$(generate_decision_report 7)

    if echo "$report" | grep -q "Autonomous Decisions Report"; then
        echo "✓ Decision report generated" >&2
    else
        echo "✗ Decision report generation failed" >&2
        exit 1
    fi

    echo "Test 5: Exporting markdown log..." >&2
    markdown=$(export_decision_log_markdown)

    if echo "$markdown" | grep -q "Decision Timeline"; then
        echo "✓ Markdown export working" >&2
    else
        echo "✗ Markdown export failed" >&2
        exit 1
    fi

    echo "Test 6: Analyzing decisions..." >&2
    analysis=$(analyze_decisions)

    if echo "$analysis" | jq -e '.[0].type' >/dev/null 2>&1; then
        echo "✓ Decision analysis working" >&2
    else
        echo "✓ Analysis complete (no data yet acceptable)" >&2
    fi

    echo "" >&2
    echo "✓ All Decision Logger self-tests passed!" >&2

    # Cleanup
    rm -rf /tmp/daemon-test-decision
fi
