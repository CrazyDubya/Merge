#!/bin/bash
#
# Root Cause Analysis Library
# Diagnoses WHY anomalies occur, not just WHAT was detected
#
# Purpose: Go beyond anomaly detection to understand root causes
# Enables intelligent remediation decisions
#
# Usage:
#   source "${DAEMON_ROOT}/lib/root-cause-analysis.sh"
#   diagnose_issue "persona_lock" "architect,optimizer"
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
STATE_DIR="${DAEMON_ROOT}/personalities"
STATE_FILE="${STATE_DIR}/state.json"
METRICS_DIR="${DAEMON_ROOT}/metrics"
LOGS_DIR="${DAEMON_ROOT}/logs"

# ============================================================================
# Diagnostic Engine
# ============================================================================

# Diagnose root cause of an anomaly
# Usage: diagnose_issue "anomaly_type" "context"
# Returns: JSON with root_causes, severity, confidence
diagnose_issue() {
    local anomaly_type="$1"
    local context="${2:-}"

    case "$anomaly_type" in
        persona_lock)
            diagnose_persona_lock "$context"
            ;;
        health_degradation)
            diagnose_health_degradation "$context"
            ;;
        validation_failures)
            diagnose_validation_failures "$context"
            ;;
        reflection_loop)
            diagnose_reflection_loop "$context"
            ;;
        api_degradation)
            diagnose_api_degradation "$context"
            ;;
        queue_stagnation)
            diagnose_queue_stagnation "$context"
            ;;
        *)
            echo '{"error":"unknown_anomaly_type","type":"'$anomaly_type'"}'
            return 1
            ;;
    esac
}

# ============================================================================
# Diagnosis 1: Persona Lock Analysis
# ============================================================================

# Diagnose why a persona lock occurred
# Returns: Root causes (circadian misalignment, chaos override, task affinity, etc.)
diagnose_persona_lock() {
    local locked_personas="$1"  # "architect,optimizer"
    local diagnosis_json=""

    # Parse locked personas
    local persona1="${locked_personas%%,*}"
    local persona2="${locked_personas##*,}"

    # Check for recent switches - how many switches total?
    local recent_switches=$(tail -20 "${METRICS_DIR}/switch-history.jsonl" 2>/dev/null | \
        grep -E "\"from\":\"$persona1\"|\"to\":\"$persona1\"|\"from\":\"$persona2\"|\"to\":\"$persona2\"" | \
        wc -l)

    # Check circadian preferences
    local p1_pref=$(jq -r ".circadian.preferences.$persona1 // \"unknown\"" "${DAEMON_ROOT}/triggers/circadian.json" 2>/dev/null || echo "unknown")
    local p2_pref=$(jq -r ".circadian.preferences.$persona2 // \"unknown\"" "${DAEMON_ROOT}/triggers/circadian.json" 2>/dev/null || echo "unknown")

    # Check emotional state triggers
    local emotional_frustration=$(jq -r '.frustration_level // 0' "${DAEMON_ROOT}/triggers/emotional.json" 2>/dev/null || echo "0")
    local emotional_threshold=$(jq -r '.thresholds.switch_on_frustration // 3' "${DAEMON_ROOT}/triggers/emotional.json" 2>/dev/null || echo "3")

    # Analyze root causes
    local root_causes=()
    local confidence=0
    local severity="high"

    # Cause 1: Circadian window alignment
    local current_hour=$(date -u +%H)
    local in_circadian_window=0

    if [[ "$p1_pref" == *"$current_hour"* ]] || [[ "$p2_pref" == *"$current_hour"* ]]; then
        root_causes+=("Circadian alignment: Both personas preferred during current window")
        ((confidence += 30))
        ((in_circadian_window = 1))
    fi

    # Cause 2: Emotional state overrides
    if [ "$emotional_frustration" -ge "$emotional_threshold" ]; then
        root_causes+=("Emotional override: Frustration level ($emotional_frustration) exceeds switch threshold ($emotional_threshold)")
        ((confidence += 25))
    fi

    # Cause 3: Activation floor conflict
    local p1_last_active=$(jq -r ".personas.$persona1.last_active_timestamp // \"unknown\"" "$STATE_FILE" 2>/dev/null || echo "unknown")
    local p2_last_active=$(jq -r ".personas.$persona2.last_active_timestamp // \"unknown\"" "$STATE_FILE" 2>/dev/null || echo "unknown")

    if [ "$p1_last_active" != "unknown" ] && [ "$p2_last_active" != "unknown" ]; then
        local p1_age=$(($(date +%s) - $(date -d "$p1_last_active" +%s 2>/dev/null || echo 0)))
        local p2_age=$(($(date +%s) - $(date -d "$p2_last_active" +%s 2>/dev/null || echo 0)))

        if [ "$p1_age" -lt 3600 ] && [ "$p2_age" -lt 3600 ]; then
            root_causes+=("Activation floor cycling: Both $persona1 ($((p1_age/60))m) and $persona2 ($((p2_age/60))m) recently active")
            ((confidence += 20))
        fi
    fi

    # Cause 4: Task affinity oscillation
    local p1_task_count=$(grep -c "in-progress: $persona1" "${DAEMON_ROOT}/tasks/queue.md" 2>/dev/null || echo "0")
    local p2_task_count=$(grep -c "in-progress: $persona2" "${DAEMON_ROOT}/tasks/queue.md" 2>/dev/null || echo "0")

    if [ "$p1_task_count" -gt 0 ] && [ "$p2_task_count" -gt 0 ]; then
        root_causes+=("Task affinity conflict: Both have in-progress work ($persona1: $p1_task_count, $persona2: $p2_task_count)")
        ((confidence += 15))
    fi

    # Cause 5: Chaos injection cycle
    local chaos_weight=$(jq -r '.triggers.chaos.weight // 0.1' "${DAEMON_ROOT}/triggers/chaos-config.json" 2>/dev/null || echo "0.1")
    if (( $(echo "$chaos_weight > 0.05" | bc -l) )); then
        root_causes+=("Chaos oscillation: Chaos weight ($chaos_weight) high enough to override normal switches")
        ((confidence += 10))
    fi

    # Cap confidence at 100
    [ "$confidence" -gt 100 ] && confidence=100

    # Build diagnostic JSON
    cat <<EOF
{
  "anomaly": "persona_lock",
  "locked_personas": "$locked_personas",
  "recent_switch_count": $recent_switches,
  "root_causes": [$(printf '"%s"' "${root_causes[@]}" | paste -sd, -)],
  "confidence_percent": $confidence,
  "severity": "$severity",
  "circadian_info": {
    "$persona1": "$p1_pref",
    "$persona2": "$p2_pref",
    "current_hour": "$current_hour"
  },
  "emotional_state": {
    "frustration": $emotional_frustration,
    "switch_threshold": $emotional_threshold
  }
}
EOF
}

# ============================================================================
# Diagnosis 2: Health Degradation Analysis
# ============================================================================

diagnose_health_degradation() {
    local context="$1"  # "persona:health_score"
    local persona="${context%%:*}"
    local health_score="${context##*:}"

    local root_causes=()
    local confidence=0

    # Cause 1: High task failure rate
    local recent_failures=$(tail -100 "${LOGS_DIR}/activity.log" 2>/dev/null | \
        grep "Task failed.*$persona" | wc -l)
    local recent_tasks=$(tail -100 "${LOGS_DIR}/activity.log" 2>/dev/null | \
        grep "Starting task.*$persona" | wc -l)

    if [ "$recent_tasks" -gt 0 ]; then
        local failure_rate=$((recent_failures * 100 / recent_tasks))
        if [ "$failure_rate" -gt 50 ]; then
            root_causes+=("High task failure rate: $failure_rate% ($recent_failures/$recent_tasks)")
            ((confidence += 40))
        fi
    fi

    # Cause 2: Incompatible task assignment
    local current_task=$(grep "Starting task" "${LOGS_DIR}/activity.log" 2>/dev/null | tail -1 | sed 's/.*Task: //' | cut -d' ' -f1)
    local persona_strengths=$(jq -r ".personas.$persona.strengths // \"\"" "${DAEMON_ROOT}/personalities/archetypes/$persona.md.json" 2>/dev/null || echo "")

    if [ -n "$current_task" ] && [ -n "$persona_strengths" ]; then
        if ! echo "$persona_strengths" | grep -qi "$current_task"; then
            root_causes+=("Task-persona mismatch: $persona handling $current_task (weak match)")
            ((confidence += 25))
        fi
    fi

    # Cause 3: API/external dependency failures
    local api_errors=$(grep -c "API error\|External.*failed" "${LOGS_DIR}/activity.log" 2>/dev/null | tail -20 || echo "0")
    if [ "$api_errors" -gt 3 ]; then
        root_causes+=("External API failures: $api_errors errors in recent activity")
        ((confidence += 20))
    fi

    # Cause 4: Excessive reflection (analysis paralysis)
    local reflection_count=$(grep -c "Starting reflection" "${LOGS_DIR}/activity.log" 2>/dev/null | tail -50 || echo "0")
    if [ "$reflection_count" -gt 10 ]; then
        root_causes+=("Analysis paralysis: $reflection_count reflections without action")
        ((confidence += 15))
    fi

    [ "$confidence" -gt 100 ] && confidence=100

    cat <<EOF
{
  "anomaly": "health_degradation",
  "persona": "$persona",
  "health_score": $health_score,
  "root_causes": [$(printf '"%s"' "${root_causes[@]}" | paste -sd, -)],
  "confidence_percent": $confidence,
  "severity": $([ "$health_score" -lt 30 ] && echo "\"critical\"" || echo "\"high\""),
  "metrics": {
    "recent_failures": $recent_failures,
    "recent_tasks": $recent_tasks,
    "failure_rate_percent": $((recent_failures * 100 / (recent_tasks + 1))),
    "api_errors": $api_errors,
    "reflection_cycles": $reflection_count
  }
}
EOF
}

# ============================================================================
# Diagnosis 3: Validation Failure Analysis
# ============================================================================

diagnose_validation_failures() {
    local context="$1"  # "failure_type:count"
    local failure_type="${context%%:*}"
    local failure_count="${context##*:}"

    local root_causes=()
    local confidence=0

    # Cause 1: Schema mismatch
    local schema_errors=$(grep -c "schema validation\|validation.*error" "${LOGS_DIR}/activity.log" 2>/dev/null | tail -50 || echo "0")
    if [ "$schema_errors" -gt 3 ]; then
        root_causes+=("Schema validation failures: $schema_errors errors detected")
        ((confidence += 35))
    fi

    # Cause 2: Task output missing files
    local missing_files=$(grep -c "expected.*not found\|file.*not.*created" "${LOGS_DIR}/activity.log" 2>/dev/null | tail -50 || echo "0")
    if [ "$missing_files" -gt 3 ]; then
        root_causes+=("Missing output files: $missing_files tasks missing expected outputs")
        ((confidence += 30))
    fi

    # Cause 3: Timeout or incomplete execution
    local timeouts=$(grep -c "timeout\|deadline exceeded" "${LOGS_DIR}/activity.log" 2>/dev/null | tail -50 || echo "0")
    if [ "$timeouts" -gt 2 ]; then
        root_causes+=("Execution timeouts: $timeouts timeouts detected")
        ((confidence += 25))
    fi

    [ "$confidence" -gt 100 ] && confidence=100

    cat <<EOF
{
  "anomaly": "validation_failures",
  "failure_type": "$failure_type",
  "failure_count": $failure_count,
  "root_causes": [$(printf '"%s"' "${root_causes[@]}" | paste -sd, -)],
  "confidence_percent": $confidence,
  "severity": $([ "$failure_count" -gt 10 ] && echo "\"critical\"" || echo "\"high\""),
  "error_analysis": {
    "schema_errors": $schema_errors,
    "missing_files": $missing_files,
    "timeouts": $timeouts
  }
}
EOF
}

# ============================================================================
# Diagnosis 4: Reflection Loop Analysis
# ============================================================================

diagnose_reflection_loop() {
    local context="$1"  # "persona:count"
    local persona="${context%%:*}"
    local reflection_count="${context##*:}"

    local root_causes=()
    local confidence=0

    # Cause 1: No clear decision framework
    local decision_logs=$(grep -c "undecided\|unclear\|cannot decide" "${LOGS_DIR}/activity.log" 2>/dev/null | tail -30 || echo "0")
    if [ "$decision_logs" -gt 3 ]; then
        root_causes+=("Unclear decision criteria: $decision_logs instances of indecision")
        ((confidence += 40))
    fi

    # Cause 2: Conflicting task requirements
    local conflicting_tasks=$(grep -c "conflict\|incompatible\|contradictory" "${LOGS_DIR}/activity.log" 2>/dev/null | tail -30 || echo "0")
    if [ "$conflicting_tasks" -gt 2 ]; then
        root_causes+=("Conflicting requirements: $conflicting_tasks task conflicts detected")
        ((confidence += 30))
    fi

    # Cause 3: Analysis paralysis (perfectionism)
    if [ "$persona" = "auditor" ] || [ "$persona" = "skeptic" ]; then
        root_causes+=("Persona tendency: $persona prone to deep analysis cycles")
        ((confidence += 25))
    fi

    [ "$confidence" -gt 100 ] && confidence=100

    cat <<EOF
{
  "anomaly": "reflection_loop",
  "persona": "$persona",
  "reflection_count": $reflection_count,
  "root_causes": [$(printf '"%s"' "${root_causes[@]}" | paste -sd, -)],
  "confidence_percent": $confidence,
  "severity": "high",
  "analysis": {
    "decision_indecision_instances": $decision_logs,
    "task_conflicts": $conflicting_tasks,
    "persona_tendency": "$([ "$persona" = "auditor" ] || [ "$persona" = "skeptic" ] && echo "true" || echo "false")"
  }
}
EOF
}

# ============================================================================
# Diagnosis 5: API Degradation Analysis
# ============================================================================

diagnose_api_degradation() {
    local context="$1"  # "api_name:issue"
    local api_name="${context%%:*}"
    local issue="${context##*:}"

    local root_causes=()
    local confidence=0

    # Check API error logs
    local http_5xx=$(grep -c "5[0-9][0-9]" "${LOGS_DIR}/activity.log" 2>/dev/null | tail -100 || echo "0")
    local http_429=$(grep -c "429\|rate.*limit" "${LOGS_DIR}/activity.log" 2>/dev/null | tail -100 || echo "0")
    local timeout_errors=$(grep -c "timeout\|deadline" "${LOGS_DIR}/activity.log" 2>/dev/null | tail -100 || echo "0")

    if [ "$http_5xx" -gt 2 ]; then
        root_causes+=("Server errors: $http_5xx 5xx errors detected")
        ((confidence += 35))
    fi

    if [ "$http_429" -gt 1 ]; then
        root_causes+=("Rate limiting: Hitting API rate limits")
        ((confidence += 40))
    fi

    if [ "$timeout_errors" -gt 2 ]; then
        root_causes+=("Latency issues: $timeout_errors timeout errors detected")
        ((confidence += 30))
    fi

    [ "$confidence" -gt 100 ] && confidence=100

    cat <<EOF
{
  "anomaly": "api_degradation",
  "api": "$api_name",
  "issue": "$issue",
  "root_causes": [$(printf '"%s"' "${root_causes[@]}" | paste -sd, -)],
  "confidence_percent": $confidence,
  "severity": $([ "$http_5xx" -gt 5 ] && echo "\"critical\"" || echo "\"high\""),
  "http_diagnostics": {
    "server_errors_5xx": $http_5xx,
    "rate_limit_429": $http_429,
    "timeout_errors": $timeout_errors
  }
}
EOF
}

# ============================================================================
# Diagnosis 6: Queue Stagnation Analysis
# ============================================================================

diagnose_queue_stagnation() {
    local hours_stalled="$1"

    local root_causes=()
    local confidence=0

    # Check task queue state
    local inprogress_count=$(grep -c "^\- \[~\]" "${DAEMON_ROOT}/tasks/queue.md" 2>/dev/null || echo "0")
    local pending_count=$(grep -c "^\- \[ \]" "${DAEMON_ROOT}/tasks/queue.md" 2>/dev/null || echo "0")
    local completed_count=$(grep -c "^\- \[x\]" "${DAEMON_ROOT}/tasks/queue.md" 2>/dev/null || echo "0")

    # Cause 1: Stalled in-progress tasks
    if [ "$inprogress_count" -gt 3 ]; then
        root_causes+=("Many stalled in-progress tasks: $inprogress_count tasks stuck in [~] state")
        ((confidence += 40))
    fi

    # Cause 2: No task progress
    if [ "$pending_count" -gt 10 ] && [ "$completed_count" -eq 0 ]; then
        root_causes+=("No task completion: $pending_count pending but zero completions")
        ((confidence += 35))
    fi

    # Cause 3: Persona unavailability
    local unavailable_personas=$(jq -r '.personas | to_entries[] | select(.value.on_cooldown == true) | .key' "$STATE_FILE" 2>/dev/null | wc -l)
    if [ "$unavailable_personas" -ge 3 ]; then
        root_causes+=("Persona unavailability: $unavailable_personas personas on cooldown")
        ((confidence += 30))
    fi

    [ "$confidence" -gt 100 ] && confidence=100

    cat <<EOF
{
  "anomaly": "queue_stagnation",
  "hours_stalled": $hours_stalled,
  "root_causes": [$(printf '"%s"' "${root_causes[@]}" | paste -sd, -)],
  "confidence_percent": $confidence,
  "severity": "critical",
  "queue_state": {
    "in_progress_count": $inprogress_count,
    "pending_count": $pending_count,
    "completed_count": $completed_count,
    "unavailable_personas": $unavailable_personas
  }
}
EOF
}

# ============================================================================
# Diagnostic Utilities
# ============================================================================

# Combine diagnoses for compound anomalies
diagnose_compound_issue() {
    local primary_anomaly="$1"
    local secondary_anomaly="$2"
    local context="$3"

    echo '{"compound":true,"primary":"'$primary_anomaly'","secondary":"'$secondary_anomaly'","interaction":"analysis_pending"}'
}

# Get historical diagnosis patterns
get_diagnosis_patterns() {
    local days_back="${1:-7}"

    # Analyze historical patterns from activity logs
    local anomaly_frequency=$(grep -c "ANOMALY" "${LOGS_DIR}/activity.log" 2>/dev/null | tail -100 || echo "0")

    echo "{\"analysis_period_days\":$days_back,\"anomalies_detected\":$anomaly_frequency,\"pattern_status\":\"monitoring\"}"
}

# Export functions
export -f diagnose_issue
export -f diagnose_persona_lock
export -f diagnose_health_degradation
export -f diagnose_validation_failures
export -f diagnose_reflection_loop
export -f diagnose_api_degradation
export -f diagnose_queue_stagnation
export -f diagnose_compound_issue
export -f get_diagnosis_patterns
