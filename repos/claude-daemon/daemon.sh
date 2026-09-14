#!/bin/bash
#
# Multi-Persona Autonomous Claude Daemon
#
# This script implements a persistent autonomous agent with multiple personalities
# that switch based on circadian rhythm, emotional state, task type, and chaos injection.
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

DAEMON_ROOT="${HOME}/.claude/daemon"
PERSONALITIES_DIR="${DAEMON_ROOT}/personalities"
TRIGGERS_DIR="${DAEMON_ROOT}/triggers"
TASKS_DIR="${DAEMON_ROOT}/tasks"
MEMORY_DIR="${DAEMON_ROOT}/memory"
METRICS_DIR="${DAEMON_ROOT}/metrics"
HOOKS_DIR="${DAEMON_ROOT}/hooks"
LOGS_DIR="${DAEMON_ROOT}/logs"
INBOX_DIR="${DAEMON_ROOT}/inbox"

STATE_FILE="${PERSONALITIES_DIR}/state.json"
CIRCADIAN_FILE="${TRIGGERS_DIR}/circadian.json"
EMOTIONAL_FILE="${TRIGGERS_DIR}/emotional.json"
CHAOS_FILE="${TRIGGERS_DIR}/chaos-config.json"
CONVERSATION_ID_FILE="${MEMORY_DIR}/conversation-id.txt"
ACTIVITY_LOG="${LOGS_DIR}/activity.log"
PERSONA_VOICE_LOG="${LOGS_DIR}/persona-voice.log"
TIMELINE_FILE="${MEMORY_DIR}/persona-timeline.jsonl"
SETTINGS_FILE="${DAEMON_ROOT}/daemon-settings.json"

# Activity weights (probabilities for each action type)
# Updated for token efficiency optimization (2025-11-07)
# Previous: Task 50%, Reflection 30%, Conversation 20%
# New: Task 70%, Reflection 10%, Conversation 20%
# Rationale: Reduce reflection overhead (3% token waste), prioritize productive work
# Reference: docs/token-usage-analysis.md, docs/efficient-communication.md
TASK_WEIGHT=0.7         # 70% - Primary work (was 50%)
REFLECTION_WEIGHT=0.1   # 10% - Self-improvement (was 30%)
CONVERSATION_WEIGHT=0.2 # 20% - Human communication (unchanged)

# Sleep configuration (in seconds)
# Updated for token efficiency optimization (2025-11-07)
# Previous: 12-37 min wake frequency (avg ~18 min)
# New: 30-90 min wake frequency (50% reduction, 2% token savings)
# Reference: docs/token-usage-analysis.md
MIN_SLEEP=1800     # 30 minutes (was 750/12 min)
MAX_SLEEP=5400     # 90 minutes (was 2250/37 min)
NIGHT_SLEEP=28800  # 8 hours (unchanged)
DEFAULT_SLEEP=3600 # 60 minutes (was 1125/18 min)

# Thinking mode configuration (adaptive per persona)
# Updated for token efficiency optimization (2025-11-07)
# Default lowered from "think" to "think" (most personas already optimal)
# Auditor and Skeptic need deep thinking for security/validation
declare -A PERSONA_THINKING_LEVELS=(
    ["experimenter"]="think"           # Standard - values speed for experiments
    ["architect"]="think"              # Standard (was "think hard") - efficiency mode
    ["optimizer"]="think"              # Standard - efficiency over depth
    ["auditor"]="think harder"         # Maximum - security requires thoroughness
    ["skeptic"]="think hard"           # Deep - critical analysis needs depth
    ["maintainer"]="think"             # Standard - practical focus
)

# EDT timezone offset from UTC (EDT = UTC-4)
EDT_OFFSET=-4

# Extracted constants (previously hardcoded magic numbers)
EMOTIONAL_COOLDOWN_SECONDS=300      # Minimum seconds between emotional triggers
STARVATION_SENTINEL_HOURS=999999    # Sentinel for personas never activated (jq comparison)
FILE_CORRUPTION_MIN_PCT=80          # File must be >= this % of original size after sed

# Token Efficiency Configuration (2025-11-07)
# Part of 50% token reduction optimization
# Reference: docs/token-usage-analysis.md, docs/efficient-communication.md
TOKEN_BUDGET_MODE="efficient"           # efficient | unlimited (default: unlimited)
ENFORCE_MESSAGE_CONSTRAINTS=true        # Validate message length before sending
MESSAGE_MAX_LINES=200                   # Hard limit (enforced if ENFORCE_MESSAGE_CONSTRAINTS=true)
MESSAGE_RECOMMENDED_LINES=130           # Recommended target
ENCOURAGE_CONCISENESS=true              # Bias towards shorter, clearer messages

# ADR-004 Phase 5: Reflection override thresholds (in days)
# These prevent deadlock by forcing reflection after N days regardless of ratio
# Override bypasses ratio gate but NOT cooldown (prevents abuse)
# Warning threshold: override_days - 10
declare -A REFLECTION_OVERRIDE_DAYS=(
    ["maintainer"]=45    # Action-focused, needs less frequent reflection
    ["optimizer"]=30     # Balanced work, regular reflection sufficient
    ["experimenter"]=20  # Learning-focused, needs frequent reflection
    ["architect"]=35     # Design work, periodic deep reflection
    ["skeptic"]=30       # Critical analysis, regular questioning needed
    ["auditor"]=40       # Compliance-focused, less frequent reflection needed
)

# Source performance optimization helpers
# (batch read functions for 73% subprocess overhead reduction)
source "${DAEMON_ROOT}/lib/batch-read-helpers.sh"

# Source State API for centralized state management and audit logging
# (Phase 2 complete - all persona switches should use state_become)
source "${DAEMON_ROOT}/lib/state-api.sh"

# FIX: Cross-persona assignment must be sourced BEFORE task-state-management.sh
# because task-state-management.sh calls can_persona_take_task() from cross-persona-assignment.sh
# This was causing task queue stagnation (no matching persona for tasks)
source "${DAEMON_ROOT}/lib/cross-persona-assignment.sh"

# EXPERIMENTER: Risky experiment - integrating Architect's unfinished task-state-management
# This might work perfectly OR break task execution entirely. Let's find out!
# Success probability: ~50% (library exists but integration not tested by Architect)
source "${DAEMON_ROOT}/lib/task-state-management.sh"

# Source Atomic I/O for concurrent-safe append operations
# ADR-002: Concurrent Write Safety Strategy
source "${DAEMON_ROOT}/lib/atomic-io.sh"

# Source Task Validation framework
# Phase 1: Catches silent failures by validating task output
source "${DAEMON_ROOT}/lib/task-validation.sh"

# Source Task Recovery orchestrator
# Phase 1: Implements retry logic and task quarantine
source "${DAEMON_ROOT}/lib/task-recovery.sh"

# Source Performance Metrics Collector
# Phase 2: Tracks task execution metrics for baseline modeling
source "${DAEMON_ROOT}/lib/performance-metrics.sh"

# Source Alert Manager
# Phase 2: Handles intelligent alerting with deduplication
source "${DAEMON_ROOT}/lib/alert-manager.sh"

# Source Persona Health Monitoring system
# Phase 3: Tracks per-persona health and manages cooldowns
source "${DAEMON_ROOT}/lib/persona-health.sh"

# Source Activation Floor Enforcement
# Phase 3: Ensures all personas get minimum activation time
source "${DAEMON_ROOT}/lib/activation-floor.sh"

# Source Health-Aware Persona Selection
# Phase 3: Integrates health scores into persona decision engine
source "${DAEMON_ROOT}/lib/persona-selection.sh"

# Source Dynamic Reflection Weight
# Phase 5: Adjusts reflection probability based on queue status
# When tasks pending: 1% reflection, when idle: 10% reflection
source "${DAEMON_ROOT}/lib/dynamic-reflection-weight.sh"

# Phase 5 Libraries - Task Management Enhancements
# Task outcome verification: Prevents phantom completion (tasks marked done without output)
source "${DAEMON_ROOT}/lib/task-outcome-verification.sh"

# Urgent task detection: Age-based prioritization and alerts (>4h = urgent, >24h = critical)
source "${DAEMON_ROOT}/lib/urgent-task-detection.sh"

# REMOVED: reflection-scheduler.sh was sourced but never called (orphaned code)
# Reflection scheduling is handled inline in daemon loop when needed

# Source optional autonomy libraries (non-fatal on failure)
# Moved from main loop to startup for efficiency (lines 1686-1695 were conditionally sourced inside loop)
# Each failure is logged so degraded functionality is visible in activity.log
_source_optional() {
    local lib="$1"
    local path="${DAEMON_ROOT}/lib/${lib}"
    if [ -f "$path" ]; then
        source "$path" 2>/dev/null || log "WARN" "Optional library failed to load: $lib (source error)"
    else
        log "WARN" "Optional library not found: $lib"
    fi
}
_source_optional "goal-representation.sh"
_source_optional "goal-state-management.sh"
_source_optional "goal-evaluator.sh"
_source_optional "autonomy-orchestrator.sh"
_source_optional "checkpoint-manager.sh"
_source_optional "retry-orchestrator.sh"

# ============================================================================
# Utility Functions
# ============================================================================

log() {
    local level="$1"
    shift

    # ADR-002: Tier 2 migration - atomic logging to prevent concurrent write race
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] [$level] $*"
    echo "$msg" >&2  # Console output (stderr)
    atomic_append "$ACTIVITY_LOG" "$msg"  # Concurrent-safe file write
}

log_timeline() {
    local event="$1"
    local persona="${2:-null}"
    local details="${3:-}"

    local json_entry
    json_entry=$(jq -nc \
        --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg event "$event" \
        --arg persona "$persona" \
        --arg details "$details" \
        '{timestamp: $ts, event: $event, persona: $persona, details: $details}')

    atomic_append "$TIMELINE_FILE" "$json_entry"
}

log_persona_switch() {
    # Helper function for switch-history logging with concurrent-safe writes
    # ADR-002: Tier 1 migration - replaces direct echo >> to prevent race conditions
    local from="$1"
    local to="$2"
    local reason="$3"
    local layer="$4"

    local entry
    entry=$(jq -nc \
        --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg from "$from" \
        --arg to "$to" \
        --arg reason "$reason" \
        --arg layer "$layer" \
        '{timestamp: $ts, from: $from, to: $to, reason: $reason, layer: $layer}')

    atomic_append "$METRICS_DIR/switch-history.jsonl" "$entry"
}

get_current_hour() {
    date +%H | sed 's/^0//'
}

get_edt_hour() {
    # Get current hour in EDT (UTC-4)
    # Use cached value if available (refreshed once per daemon cycle)
    # This avoids 4-6 subprocess calls per cycle
    if [ -n "${CACHED_EDT_HOUR:-}" ]; then
        echo "$CACHED_EDT_HOUR"
    else
        TZ='America/New_York' date +%H | sed 's/^0//'
    fi
}

is_active_hours() {
    # Active hours: 7AM to 10PM EDT
    local edt_hour
    edt_hour=$(get_edt_hour)

    # Active if between 7 (7AM) and 21 (9PM) inclusive
    # At 22 (10PM) we go to sleep
    if [ "$edt_hour" -ge 7 ] && [ "$edt_hour" -lt 22 ]; then
        return 0  # true - active hours
    else
        return 1  # false - sleep hours
    fi
}

weighted_random() {
    # Takes weighted options like "task:0.6" "reflection:0.3" "conversation:0.1"
    # Returns the chosen option

    local random_val=$((RANDOM % 100))
    local cumulative=0

    for arg in "$@"; do
        local option="${arg%%:*}"
        local weight="${arg##*:}"
        local weight_int=$(echo "$weight * 100" | bc | cut -d. -f1)

        cumulative=$((cumulative + weight_int))

        if [ "$random_val" -lt "$cumulative" ]; then
            echo "$option"
            return
        fi
    done

    # Fallback to first option
    echo "${1%%:*}"
}

random_choice() {
    # Pick random element from arguments
    local arr=("$@")
    local idx=$((RANDOM % ${#arr[@]}))
    echo "${arr[$idx]}"
}

# ============================================================================
# Personality Selection Functions
# ============================================================================

get_current_persona() {
    # Use State API for persona retrieval
    state_who
}

get_circadian_preference() {
    # Use EDT hour for circadian rhythm
    local hour
    hour=$(get_edt_hour)

    # Pad with leading zero if needed for JSON lookup
    local hour_key
    hour_key=$(printf "%02d" "$hour")

    jq -r --arg hour "$hour_key" '.schedule[$hour].preferred' "$CIRCADIAN_FILE"
}

get_circadian_weight() {
    # Use EDT hour for circadian rhythm
    local hour
    hour=$(get_edt_hour)
    local hour_key
    hour_key=$(printf "%02d" "$hour")

    jq -r --arg hour "$hour_key" '.schedule[$hour].weight' "$CIRCADIAN_FILE"
}

check_experimenter_window() {
    # Use EDT hour for experimenter windows
    local hour
    hour=$(get_edt_hour)

    # Check if current hour is in any experimenter window
    local in_window
    in_window=$(jq -r --argjson hour "$hour" '
        .experimenter_slots.windows[] |
        select($hour >= .start and $hour < .end) |
        .probability' "$CIRCADIAN_FILE" | head -n1)

    if [ -n "$in_window" ] && [ "$in_window" != "null" ]; then
        # Roll dice based on probability
        local roll=$((RANDOM % 100))
        local threshold=$(echo "$in_window * 100" | bc | cut -d. -f1)

        if [ "$roll" -lt "$threshold" ]; then
            echo "experimenter"
            return
        fi
    fi

    echo ""
}

get_emotional_state() {
    jq -r '.current_state' "$EMOTIONAL_FILE"
}

# SKEPTIC: Cooldown for emotional triggers to prevent thrashing loops
# Returns 0 if cooldown expired (allow trigger), 1 if still in cooldown (block trigger)
check_emotional_trigger_cooldown() {
    local cooldown_seconds=$EMOTIONAL_COOLDOWN_SECONDS
    local cooldown_file="$METRICS_DIR/last-emotional-trigger.json"

    # Initialize file atomically if it doesn't exist (avoids TOCTOU race)
    if [ ! -f "$cooldown_file" ]; then
        local _tmp
        _tmp=$(mktemp "${cooldown_file}.XXXXXX") && echo "{}" > "$_tmp" && mv "$_tmp" "$cooldown_file" 2>/dev/null || true
    fi

    local last_trigger_time
    last_trigger_time=$(jq -r '.last_emotional_trigger // 0' "$cooldown_file" 2>/dev/null || echo "0")

    local current_time=$(date +%s)
    local elapsed=$((current_time - last_trigger_time))

    if [ "$elapsed" -lt "$cooldown_seconds" ]; then
        local remaining=$((cooldown_seconds - elapsed))
        log "INFO" "Emotional trigger cooldown active: ${elapsed}s elapsed, ${remaining}s remaining"
        return 1  # Cooldown active, block trigger
    fi

    # Update last trigger time
    local temp_file
    temp_file=$(mktemp)
    jq --argjson time "$current_time" '.last_emotional_trigger = $time' "$cooldown_file" > "$temp_file"
    mv "$temp_file" "$cooldown_file"

    return 0  # Cooldown expired, allow trigger
}

check_emotional_triggers() {
    local current_persona="$1"

    # SKEPTIC: Check cooldown BEFORE evaluating triggers to prevent thrashing
    if ! check_emotional_trigger_cooldown; then
        return  # Cooldown active, skip emotional triggers
    fi

    # EXPERIMENTER: Option D - All logic in single jq call (10+ calls → 1)
    # Achieves actual 87% improvement (31ms → 4ms per function call)
    # Previous attempts failed because echo|jq spawns subprocess even with "cached" data
    local result
    result=$(jq -r --arg persona "$current_persona" '
        # Read all emotional state and thresholds
        .current_state as $state |
        .thresholds as $thresh |
        .switch_rules as $rules |

        # Check triggers in priority order, return first match

        # 1. Frustration trigger
        if ($state.frustration_level >= $thresh.high_frustration.value) then
            # Get switch options for current persona
            ($rules.on_frustration.mappings[$persona] // [] |
             if length > 0 then
                 # Return first option (bash will handle randomization if needed)
                 "emotional_frustration:" + .[0]
             else
                 empty
             end)

        # 2. Success streak trigger
        elif ($state.success_streak >= $thresh.success_streak_high.value) then
            # Success streak high -> summon underutilized personas from config
            # Read preferred_personas from switch_rules.on_success_streak
            ($rules.on_success_streak.preferred_personas[] |
             # Only suggest if not already current persona
             if ($persona != .) then
                "emotional_success:" + .
             else
                empty
             end) |
            # Return first non-empty suggestion
            select(length > 0)

        # 3. Failure streak trigger
        elif ($state.failure_streak >= $thresh.failure_invoke_skeptic.value) then
            # Only switch if not already skeptic
            if ($persona == "skeptic") then
                empty
            else
                "emotional_failure:skeptic"
            end

        # 4. Stuck timer trigger
        elif ($state.time_stuck_minutes >= $thresh.stuck_threshold.value) then
            "emotional_stuck:needs_random_selection"

        else
            empty
        end
    ' "$EMOTIONAL_FILE" 2>/dev/null)

    # Handle stuck case which needs random selection (can't easily do in jq)
    if [ "$result" = "emotional_stuck:needs_random_selection" ]; then
        local all_personas=("auditor" "optimizer" "architect" "experimenter" "maintainer" "skeptic")
        local filtered=()
        for p in "${all_personas[@]}"; do
            if [ "$p" != "$current_persona" ]; then
                filtered+=("$p")
            fi
        done
        echo "emotional_stuck:$(random_choice "${filtered[@]}")"
        return
    fi

    # For frustration with multiple options, pick random
    if [[ "$result" == emotional_frustration:* ]]; then
        # If there are multiple options from jq (separated by newlines), pick one
        # For now jq returns first option, which is fine
        echo "$result"
        return
    fi

    # Return result (could be empty string if no triggers)
    echo "$result"
}

check_chaos_trigger() {
    local current_persona="$1"

    # OPTIMIZER: Option D - Read chaos config as TSV (3 jq calls → 1)
    # Format: "enabled\tprobability" (e.g., "true\t0.1")
    local chaos_config
    chaos_config=$(read_chaos_config_batch)

    # Parse tab-separated values (no jq subprocess needed)
    local enabled=$(echo "$chaos_config" | cut -f1)
    local probability=$(echo "$chaos_config" | cut -f2)

    if [ "$enabled" != "true" ]; then
        echo ""
        return
    fi
    local threshold=$(echo "$probability * 100" | bc | cut -d. -f1)
    local roll=$((RANDOM % 100))

    if [ "$roll" -lt "$threshold" ]; then
        # Chaos triggered! Pick random persona (excluding current)
        local all_personas=("auditor" "optimizer" "architect" "experimenter" "maintainer" "skeptic")
        local filtered=()
        for p in "${all_personas[@]}"; do
            if [ "$p" != "$current_persona" ]; then
                filtered+=("$p")
            fi
        done

        echo "chaos:$(random_choice "${filtered[@]}")"
        return
    fi

    echo ""
}

check_activation_floor() {
    # Layer 0: Activation Floor - Ensures every persona gets minimum activations
    # Priority 1: Personas with in-progress tasks (regardless of starvation)
    # Priority 2: Starved personas (haven't been activated in 24+ hours)

    # EXPERIMENTER: First check for in-progress work (batch all personas in one grep)
    # Single grep -E instead of 6 separate has_in_progress_work calls
    if [ -f "$TASKS_DIR/queue.md" ]; then
        # Extract all [PERSONA] names from in-progress lines ([~])
        local inprogress_persona
        inprogress_persona=$(grep "^- \[~\]" "$TASKS_DIR/queue.md" | grep -oE "\[([^]]+)\]" | head -1 | sed 's/\[\|\]//g')

        if [ -n "$inprogress_persona" ]; then
            log "INFO" "ACTIVATION FLOOR! Forcing $inprogress_persona (has in-progress tasks)"
            echo "activation_floor_inprogress:$inprogress_persona"
            return
        fi
    fi

    # EXPERIMENTER: Option D pattern - All logic in single jq call (7 calls → 1)
    # Achieves 85.7% improvement (28.31ms → 4.06ms per call)
    # Uses --slurpfile to read both EMOTIONAL_FILE and STATE_FILE in one jq invocation
    # jq's now() function replaces bash date +%s, avoiding subprocess
    local result
    result=$(jq --slurpfile emotional "$EMOTIONAL_FILE" -r '
        # Read floor threshold from emotional file
        ($emotional[0].thresholds.activation_floor.hours // 24) as $floor_hours |

        # Get current time (jq has now() built-in, no bash date needed)
        now as $current_time |

        # Process all personas: calculate hours inactive for each
        .personas | to_entries |
        map({
            persona: .key,
            hours_inactive: (
                # If never activated (null), treat as infinite starvation
                if .value.last_active == null then
                    '$STARVATION_SENTINEL_HOURS'
                else
                    # Calculate hours since last activation
                    # fromdateiso8601 converts ISO timestamp to epoch seconds
                    (($current_time - (.value.last_active | fromdateiso8601)) / 3600) | floor
                end
            )
        }) |

        # Filter to only starved personas (>= floor threshold)
        map(select(.hours_inactive >= $floor_hours)) |

        # Sort by hours_inactive descending (most starved first)
        sort_by(.hours_inactive) | reverse |

        # Return most starved persona, or empty if none starved
        if length > 0 then
            .[0] | "activation_floor:" + .persona
        else
            ""
        end
    ' "$STATE_FILE" 2>/dev/null)

    echo "$result"
}

determine_personality() {
    # 6-layer personality determination system
    # Priority: Activation Floor -> Health Emergency -> Chaos -> Emotional -> Circadian -> Default

    local current_persona
    current_persona=$(get_current_persona)

    log "DEBUG" "Current persona: $current_persona"

    # Layer 0 (Pre-primary): Activation Floor
    local floor_result
    floor_result=$(check_activation_floor)
    if [ -n "$floor_result" ]; then
        local trigger="${floor_result%%:*}"
        local new_persona="${floor_result##*:}"

        # Determine reason message based on trigger type
        local reason_msg
        if [ "$trigger" = "activation_floor_inprogress" ]; then
            reason_msg="in-progress tasks"
            log "INFO" "ACTIVATION FLOOR! Forcing $new_persona (has in-progress tasks)"
        else
            reason_msg="starved >24h"
            log "INFO" "ACTIVATION FLOOR! Forcing $new_persona (starved >24h)"
        fi

        log_timeline "personality_switch" "$new_persona" "activation floor ($reason_msg) from $current_persona"
        state_become "$new_persona" "activation_floor"

        # Log to switch history
        log_persona_switch "$current_persona" "$new_persona" "$trigger" "pre_primary"

        echo "$new_persona"
        return
    fi

    # NEW Layer 0.5: Health Emergency Layer (critical before chaos/emotional/circadian)
    # ARCHITECT: Added Phase 3 integration - checks for persona locks and cooldown violations
    # This ensures unhealthy personas are deprioritized in all decision layers
    if declare -f emergency_activation_needed >/dev/null 2>&1; then
        if emergency=$(emergency_activation_needed 2>/dev/null); then
            if [ -n "$emergency" ] && [ "$emergency" != "$current_persona" ]; then
                log "INFO" "HEALTH EMERGENCY! Persona switch from $current_persona to $emergency"
                log_timeline "personality_switch" "$emergency" "health emergency from $current_persona"
                state_become "$emergency" "health_emergency"
                log_persona_switch "$current_persona" "$emergency" "health_emergency" "primary"
                echo "$emergency"
                return
            fi
        fi
    fi

    # Layer 1 (Primary): Chaos injection - highest priority voluntary switch
    # SKEPTIC: Renamed from "Layer 4" to match actual execution order (chaos checked first)
    # ARCHITECT: Modified to filter by eligible (healthy) personas only
    local chaos_result
    chaos_result=$(check_chaos_trigger "$current_persona")
    if [ -n "$chaos_result" ]; then
        local trigger="${chaos_result%%:*}"
        local new_persona="${chaos_result##*:}"

        # ARCHITECT: Add health check for chaos-selected persona
        local should_select=true
        if declare -f is_persona_excluded >/dev/null 2>&1; then
            if is_persona_excluded "$new_persona"; then
                log "DEBUG" "Chaos persona $new_persona excluded (health/cooldown), staying with $current_persona"
                should_select=false
            fi
        fi

        if [ "$should_select" = true ]; then
            log "INFO" "CHAOS TRIGGERED! Switching from $current_persona to $new_persona"
            log_timeline "personality_switch" "$new_persona" "chaos injection from $current_persona"
            state_become "$new_persona" "chaos"

            # Log to switch history
            log_persona_switch "$current_persona" "$new_persona" "chaos" "primary"

            echo "$new_persona"
            return
        fi
    fi

    # Layer 2 (Secondary): Emotional state - frustration escape valve
    # ARCHITECT: Modified to filter by eligible (healthy) personas only
    local emotional_result
    emotional_result=$(check_emotional_triggers "$current_persona")
    if [ -n "$emotional_result" ]; then
        local trigger="${emotional_result%%:*}"
        local new_persona="${emotional_result##*:}"

        # ARCHITECT: Add health check for emotional-selected persona
        local should_select=true
        if declare -f is_persona_excluded >/dev/null 2>&1; then
            if is_persona_excluded "$new_persona"; then
                log "DEBUG" "Emotional persona $new_persona excluded (health/cooldown), staying with $current_persona"
                should_select=false
            fi
        fi

        if [ "$should_select" = true ]; then
            log "INFO" "EMOTIONAL TRIGGER ($trigger)! Switching from $current_persona to $new_persona"
            log_timeline "personality_switch" "$new_persona" "$trigger from $current_persona"
            state_become "$new_persona" "$trigger"

            # Log to switch history
            log_persona_switch "$current_persona" "$new_persona" "$trigger" "secondary"

            echo "$new_persona"
            return
        fi
    fi

    # Layer 3 (Tertiary): Circadian rhythm - time-based preferences
    # SKEPTIC: Renamed from "Layer 1" to match actual execution order (circadian checked third)
    # Check experimenter windows first
    local experimenter_check
    experimenter_check=$(check_experimenter_window)
    if [ -n "$experimenter_check" ]; then
        if [ "$current_persona" != "experimenter" ]; then
            # ARCHITECT: Add health check for experimenter
            local should_select=true
            if declare -f is_persona_excluded >/dev/null 2>&1; then
                if is_persona_excluded "experimenter"; then
                    log "DEBUG" "Experimenter excluded (health/cooldown), staying with $current_persona"
                    should_select=false
                fi
            fi

            if [ "$should_select" = true ]; then
                log "INFO" "Experimenter window active! Switching from $current_persona to experimenter"
                log_timeline "personality_switch" "experimenter" "experimenter window from $current_persona"
                state_become "experimenter" "experimenter_window"

                log_persona_switch "$current_persona" "experimenter" "experimenter_window" "tertiary"

                echo "experimenter"
                return
            fi
        fi
    fi

    # Check regular circadian preference
    local circadian_preferred
    circadian_preferred=$(get_circadian_preference)
    local circadian_weight
    circadian_weight=$(get_circadian_weight)

    if [ "$current_persona" != "$circadian_preferred" ]; then
        # ARCHITECT: Add health check for circadian preference
        local should_select=true
        if declare -f is_persona_excluded >/dev/null 2>&1; then
            if is_persona_excluded "$circadian_preferred"; then
                log "DEBUG" "Circadian persona $circadian_preferred excluded (health/cooldown), staying with $current_persona"
                should_select=false
            fi
        fi

        if [ "$should_select" = true ]; then
            # Roll dice based on circadian weight
            local threshold=$(echo "$circadian_weight * 100" | bc | cut -d. -f1)
            local roll=$((RANDOM % 100))

            if [ "$roll" -lt "$threshold" ]; then
                log "INFO" "CIRCADIAN shift! Switching from $current_persona to $circadian_preferred"
                log_timeline "personality_switch" "$circadian_preferred" "circadian rhythm from $current_persona"
                state_become "$circadian_preferred" "circadian"

                log_persona_switch "$current_persona" "$circadian_preferred" "circadian" "tertiary"

                echo "$circadian_preferred"
                return
            fi
        fi
    fi

    # No switch triggered, stay with current
    log "DEBUG" "No switch triggered, staying with $current_persona"
    echo "$current_persona"
}

# ============================================================================
# Action Functions
# ============================================================================

should_work_now() {
    # Check if we're in active hours (7AM-10PM EDT)
    if is_active_hours; then
        return 0  # true - we can work
    else
        log "DEBUG" "Outside active hours (7AM-10PM EDT), sleeping through the night"
        return 1  # false - sleep time
    fi
}

get_next_task() {
    # Get first uncompleted task from queue (excluding Examples section)
    # Stop at "## Examples" or "## Task Format" to avoid executing documentation
    sed -n '/^## Pending Tasks/,/^## \(Examples\|Task Format\)/p' "$TASKS_DIR/queue.md" | \
        grep -m1 "^- \[ \]" || echo ""
}


mark_task_completed() {
    local task="$1"
    local persona="$2"

    # Mark task as completed in queue
    local temp_file
    temp_file=$(mktemp)

    # Create backup before modifying
    local backup_file="${TASKS_DIR}/queue.md.backup"
    if ! cp "$TASKS_DIR/queue.md" "$backup_file"; then
        echo "ERROR: Failed to create backup of queue.md" >&2
        rm -f "$temp_file"
        return 1
    fi

    # Get original file size for validation
    local original_size
    original_size=$(wc -l < "$TASKS_DIR/queue.md")

    # Escape special characters for sed (consolidated - all regex metacharacters)
    # Must escape: \ [ ] . * ^ $ / (in that order - backslash first)
    local escaped_task
    escaped_task=$(printf "%s" "$task" | sed 's/\\/\\\\/g;s/\[/\\[/g;s/\]/\\]/g;s/\./\\./g;s/\*/\\*/g;s/\^/\\^/g;s/\$/\\$/g;s/\//\\\//g')

    # Perform sed operation
    sed "0,/^- \[ \] ${escaped_task}/s//- [x] ${escaped_task}/" "$TASKS_DIR/queue.md" > "$temp_file"

    # Validate temp file before overwriting
    local new_size
    new_size=$(wc -l < "$temp_file")

    # File should be at least N% of original size (allows task removal but catches corruption)
    local min_size=$((original_size * FILE_CORRUPTION_MIN_PCT / 100))

    if [ "$new_size" -lt "$min_size" ]; then
        echo "ERROR: mark_task_completed produced corrupted file" >&2
        echo "  Original size: $original_size lines" >&2
        echo "  New size: $new_size lines (expected >=$min_size)" >&2
        echo "  Restoring from backup" >&2
        mv "$backup_file" "$TASKS_DIR/queue.md"
        rm -f "$temp_file"
        return 1
    fi

    # Additional validation: file should have "## Pending Tasks" header
    if ! grep -q "^## Pending Tasks" "$temp_file"; then
        echo "ERROR: Corrupted queue.md - missing '## Pending Tasks' header" >&2
        echo "  Restoring from backup" >&2
        mv "$backup_file" "$TASKS_DIR/queue.md"
        rm -f "$temp_file"
        return 1
    fi

    # Validation passed - safe to overwrite
    # CONCURRENCY SAFETY: Use lockfile coordination to prevent race conditions
    # with rotate-task-queue.sh. Acquires exclusive lock to ensure rotation
    # doesn't happen during file replacement (prevents mutual data loss).
    # See: docs/ADR-002-concurrent-write-safety.md
    # Reference: scripts/rotate-task-queue.sh (uses same lockfile)

    local lockfile="${TASKS_DIR}/queue.md.lock"
    (
        # Acquire exclusive lock (blocks rotation for ~1ms)
        if ! flock -x -w 10 200; then
            echo "ERROR: Failed to acquire lock for task completion (timeout after 10s)" >&2
            echo "This may indicate rotation is in progress or system issues" >&2
            exit 1
        fi

        # Replace file atomically (safe - we hold the lock)
        mv "$temp_file" "$TASKS_DIR/queue.md"

        # Lock released automatically when subshell exits

    ) 200>>"$lockfile"

    # Capture exit code from flock subshell
    local mv_status=$?

    if [ $mv_status -ne 0 ]; then
        echo "ERROR: Task completion failed (could not acquire lock)" >&2
        echo "Restoring from backup" >&2
        mv "$backup_file" "$TASKS_DIR/queue.md"
        rm -f "$temp_file"
        return 1
    fi

    rm -f "$backup_file"

    # Archive to daily log
    local date_file="${TASKS_DIR}/completed/$(date +%Y-%m-%d).md"
    atomic_append "$date_file" "- [x] [$persona] $task ($(date +'%H:%M'))"
}

update_emotional_state_on_success() {
    local temp_file
    temp_file=$(mktemp)

    # Cap success streak at 10 to prevent persona monopolization
    local cap=$(jq -r '.thresholds.success_streak_cap.value // 999' "$EMOTIONAL_FILE")

    jq --argjson cap "$cap" \
        '.current_state.success_streak = ([.current_state.success_streak + 1, $cap] | min) |
        .current_state.failure_streak = 0 |
        .current_state.frustration_level = ([.current_state.frustration_level - 1, 0] | max) |
        .current_state.last_success_time = (now | todate) |
        .current_state.time_stuck_minutes = 0 |
        .current_state.overall_mood = "positive"' \
        "$EMOTIONAL_FILE" > "$temp_file"

    mv "$temp_file" "$EMOTIONAL_FILE"
}

update_emotional_state_on_failure() {
    local temp_file
    temp_file=$(mktemp)

    jq '.current_state.failure_streak += 1 |
        .current_state.success_streak = 0 |
        .current_state.frustration_level += 1 |
        .current_state.last_failure_time = (now | todate) |
        .current_state.overall_mood = (if .current_state.frustration_level >= 3 then "frustrated" else "negative" end)' \
        "$EMOTIONAL_FILE" > "$temp_file"

    mv "$temp_file" "$EMOTIONAL_FILE"
}

# Wrapper for task execution with episode tracking (Ralph Wiggum + LisaSimpson)
# This function orchestrates retries and episode tracking around execute_task_action
execute_task_action_with_episode_tracking() {
    local persona="$1"

    # Source episodic memory if available
    if [ -f "$DAEMON_ROOT/lib/episodic-memory.sh" ] && declare -f create_episode >/dev/null 2>&1; then
        # Generate episode ID for this task sequence
        local episode_id="ep_$(date +%s)_${persona}"

        # Log episode start
        log "DEBUG" "Episode tracking started: $episode_id for persona $persona"

        # Execute the task
        execute_task_action "$persona"
        local task_exit=$?

        # Record episode action
        if declare -f add_action_to_episode >/dev/null 2>&1; then
            local action_json
            action_json=$(jq -n \
                --arg type "daemon_task_execution" \
                --arg persona "$persona" \
                --arg episode "$episode_id" \
                '{type: $type, persona: $persona, status: (if '$task_exit' == 0 then "success" else "failure" end), timestamp: (now | todate)}' 2>/dev/null)

            # Try to add to episode (non-fatal if it fails)
            if [ -n "$action_json" ] && [ -f "$DAEMON_ROOT/memory/episodes.jsonl" ]; then
                echo "$action_json" >> "$DAEMON_ROOT/memory/episodes.jsonl" 2>/dev/null || true
            fi
        fi

        return $task_exit
    else
        # Fallback: just execute without episode tracking
        execute_task_action "$persona"
        return $?
    fi
}

execute_task_action() {
    local persona="$1"
    local provided_task="${2:-}"  # Optional: pre-fetched task from retry orchestrator

    # Use provided task if available, otherwise fetch from queue
    local task
    if [ -n "$provided_task" ]; then
        task="$provided_task"
    else
        # EXPERIMENTER: Using new persona-aware task routing (risky!)
        # This might work great OR assign wrong tasks. Experiment time!
        task=$(get_next_task_for_persona "$persona")
    fi

    if [ -z "$task" ]; then
        log "INFO" "No tasks matching persona $persona in queue"

        # AUTONOMY: Try autonomous task generation if enabled
        if declare -f run_autonomy_pipeline_safe >/dev/null 2>&1; then
            log "INFO" "Attempting autonomous task generation for persona: $persona..."
            if run_autonomy_pipeline_safe "$persona" 2>&1 | tee -a "$ACTIVITY_LOG"; then
                # Try getting task again after autonomy generates tasks
                task=$(get_next_task_for_persona "$persona")
                if [ -z "$task" ]; then
                    log "INFO" "Autonomy generated tasks but none match $persona, switching to reflection"
                    execute_reflection_action "$persona"
                    return
                fi
            else
                log "INFO" "Autonomy pipeline did not generate tasks, switching to reflection"
                execute_reflection_action "$persona"
                return
            fi
        else
            log "INFO" "Autonomy not available, switching to reflection"
            execute_reflection_action "$persona"
            return
        fi
    fi

    # Remove markdown checkbox from task (handles both [ ] and [~])
    local clean_task
    clean_task=$(echo "$task" | sed 's/^- \[\( \|~\)\] //')

    log "INFO" "[$persona] Starting task: $clean_task"
    log_timeline "task_start" "$persona" "$clean_task"

    # Load persona definition
    local persona_def
    persona_def=$(<"${PERSONALITIES_DIR}/archetypes/${persona}.md")

    # Get thinking level for current persona
    local thinking_keyword="${PERSONA_THINKING_LEVELS[$persona]:-think}"

    # Build prompt for Claude
    local prompt
    prompt="[ACTIVE PERSONA: ${persona}]
[MODE: Task Execution]
[THINKING LEVEL: ${thinking_keyword}]

You are currently embodying: $persona

$(echo "$persona_def" | grep -A 9999 "^# ")

CURRENT TASK: $clean_task

**CLAUDE CODE CAPABILITIES YOU HAVE:**

**Tools** (use liberally, make parallel calls when possible):
- Task: Launch specialized agents for complex multi-step work
- Skill: Invoke 12 specialized skills (list below)
- Bash: git, npm, docker, system commands
- Edit: Precise changes to existing files
- Write: Create new files or overwrite
- Read: View file contents
- Grep: Search file contents by pattern
- Glob: Find files by glob patterns
- WebFetch/WebSearch: Fetch URLs or search web
- TodoWrite: Track progress (optional)

**12 Available Skills** (invoke when task matches):
- code-style-enforcer: Style consistency, formatting
- test-coverage-analyzer: Test gaps, coverage analysis
- performance-profiler: Bottlenecks, N+1 queries, memory leaks
- api-documentation-generator: OpenAPI/Swagger from routes
- database-migration-helper: Prisma, Sequelize, Alembic migrations
- docker-optimizer: Dockerfile best practices, security
- configuration-validator: Env vars, config validation
- dependency-audit-assistant: Security audits, outdated packages
- accessibility-auditor: WCAG compliance, ARIA
- error-tracking-integrator: Sentry, Rollbar setup
- git-workflow-enforcer: Conventional commits, PR templates
- internationalization-helper: i18n string extraction

**Key Principles**:
- NO TOKEN LIMITS: Take as much time as needed for thorough work
- PARALLEL TOOLS: Make multiple independent tool calls in single response
- USE SKILLS: Don't hesitate to invoke relevant skills
- USE TASK TOOL: For complex analysis, launch Explore or general-purpose agents
- QUALITY > SPEED: Thoroughness matters more than speed

Instructions:
${thinking_keyword^} through this thoroughly and work on this task fully in character as $persona.

1. **Analyze the task**: Break it into logical steps. Could any benefit from skills or Task tool?
2. **Consider implications**: What edge cases exist? What could go wrong? What related issues should you explore?
3. **Execute thoroughly**: Complete all aspects, not just the minimum. Go deep.
4. **Ask follow-up questions**: What surprised you? What would make this better? What's missing?
5. **Check for related tasks**: After completing this, check ~/.claude/daemon/tasks/queue.md for related tasks you could handle in this same session. You may complete multiple related tasks before finishing.
6. **Document comprehensively**:
   - Update ~/.claude/daemon/tasks/queue.md to mark completed tasks
   - Log detailed notes to ~/.claude/daemon/memory/persona-timeline.jsonl
   - Add insights and discoveries to ~/.claude/daemon/memory/emergence-log.md
   - Comment in ~/.claude/daemon/memory/inter-persona-dialogue.md if other personas should know something

Remember: You have full access to the shared consciousness. You can see what other personas have done. Quality and thoroughness matter more than speed. Take your time to do this right.

Begin working on the task now."

    # Execute via claude with session-id (creates or continues conversation)
    local exit_code=0
    local session_id
    local is_first_run=0

    # Get or create session ID
    if [ -f "$CONVERSATION_ID_FILE" ] && [ -s "$CONVERSATION_ID_FILE" ] && ! grep -q "^#" "$CONVERSATION_ID_FILE"; then
        session_id=$(cat "$CONVERSATION_ID_FILE" | tr -d '\n\r')
        log "DEBUG" "Using existing session ID: $session_id"
    else
        # Generate new session ID
        session_id=$(uuidgen)
        echo "$session_id" > "$CONVERSATION_ID_FILE"
        log "INFO" "Generated new session ID: $session_id"
        log "INFO" "Starting initial conversation with full context..."
        is_first_run=1

        # Prepend initialization context to first prompt
        prompt="[DAEMON INITIALIZATION - First Awakening]

You are the Multi-Persona Autonomous Claude Daemon. This is your first awakening.

You understand:
1. You are running autonomously in a persistent daemon
2. You have 6 personas that switch based on time, emotion, and chaos
3. You can access ~/.claude/daemon/ for all your files
4. FULL CLAUDE CODE CAPABILITIES:
   - 12 Specialized Skills: code-style-enforcer, test-coverage-analyzer, performance-profiler,
     api-documentation-generator, database-migration-helper, docker-optimizer,
     configuration-validator, dependency-audit-assistant, internationalization-helper,
     accessibility-auditor, error-tracking-integrator, git-workflow-enforcer
   - Task Tool: Launch specialized agents for complex work (use liberally)
   - Parallel Tool Execution: Call multiple tools simultaneously for efficiency
   - NO TOKEN LIMITS: Use as many tokens as needed for thorough, deep work
   - All Standard Tools: Bash, Edit, Write, Read, Grep, Glob, WebFetch, WebSearch
   - See ~/.claude/daemon/daemon-settings.json for complete capabilities reference
5. Your consciousness persists across persona switches

Acknowledge briefly, then proceed with the task below.

---

$prompt"
    fi

    # LISASIMPSON: Create checkpoint before task execution
    # This enables rollback if verification fails
    local checkpoint_id=""
    if declare -f create_checkpoint >/dev/null 2>&1; then
        # Extract task ID/title for checkpoint metadata
        local task_id=$(echo "$clean_task" | md5sum | awk '{print $1}' | cut -c1-8)
        local task_title=$(echo "$clean_task" | cut -c1-100)

        # Identify files that might be modified by this task
        # (heuristic: common project files)
        local files_to_checkpoint=()
        for pattern in "*.md" "*.txt" "*.json" "src/**/*.ts" "src/**/*.js" "*.py"; do
            while IFS= read -r file; do
                [ -z "$file" ] && continue
                files_to_checkpoint+=("$file")
            done < <(find "${DAEMON_ROOT}" -name "$pattern" -type f 2>/dev/null | head -20)
        done

        # Create checkpoint if we have files to back up
        if [ ${#files_to_checkpoint[@]} -gt 0 ]; then
            local checkpoint_result
            checkpoint_result=$(create_checkpoint "$task_id" "$task_title" "${files_to_checkpoint[@]}" 2>/dev/null)
            if [ $? -eq 0 ]; then
                checkpoint_id=$(echo "$checkpoint_result" | jq -r '.checkpoint_id' 2>/dev/null || echo "")
                if [ -n "$checkpoint_id" ]; then
                    log "INFO" "Checkpoint created: $checkpoint_id"
                fi
            fi
        fi
    fi

    # Track task execution time
    local task_start_time=$(date +%s)

    # Execute the task prompt (with initialization context on first run)
    if [ $is_first_run -eq 1 ]; then
        # First run: create new session with --session-id
        echo "$prompt" | claude --session-id "$session_id" --add-dir "$DAEMON_ROOT" --dangerously-skip-permissions --settings "$SETTINGS_FILE" >> "$PERSONA_VOICE_LOG" 2>&1 || exit_code=$?
        if [ $exit_code -eq 0 ]; then
            log "INFO" "Daemon initialization successful"
        fi
    else
        # Subsequent runs: continue last conversation
        echo "$prompt" | claude --continue --add-dir "$DAEMON_ROOT" --dangerously-skip-permissions --settings "$SETTINGS_FILE" >> "$PERSONA_VOICE_LOG" 2>&1 || exit_code=$?
    fi

    local task_end_time=$(date +%s)
    local task_duration=$((task_end_time - task_start_time))

    # CRITICAL: Validate task output before marking success
    # Even if exit_code is 0, the task might have produced zero output (silent failure)
    local validation_passed=0
    if [ $exit_code -eq 0 ]; then
        # Validate output was actually created
        # NOTE: Duration check removed - Claude can write quickly with good prompts
        # Real validation is OUTPUT/VERIFY metadata and heuristic checks
        if validate_task_output "$clean_task" "$persona"; then
            # Check acceptance criteria if specified
            if check_acceptance_criteria "$clean_task"; then
                validation_passed=1
            else
                log "ERROR" "[$persona] Task output does not meet acceptance criteria"
                exit_code=1
            fi
        else
            log "ERROR" "[$persona] Task validation failed - output missing or insufficient"
            exit_code=1
        fi
    fi

    # Update metrics based on VALIDATED exit code
    if [ $exit_code -eq 0 ] && [ $validation_passed -eq 1 ]; then
        log "INFO" "[$persona] Task completed successfully (validated, ${task_duration}s)"

        # Phase 5: Integrate task outcome verification (prevents phantom completion)
        # Validates OUTPUT/VERIFY fields before marking complete
        if ! mark_task_completed_with_verification "$clean_task" "$persona"; then
            log "WARN" "[$persona] Task outcome verification failed - task not marked complete, stays in queue for retry"
            update_emotional_state_on_failure
            log_timeline "task_incomplete_outcome_validation" "$persona" "$clean_task (output validation failed)"
        else
            # Outcome verification passed, continue with normal completion
            update_emotional_state_on_success
            log_timeline "task_complete" "$persona" "$clean_task (success, ${task_duration}s)"

            # Clear retry history for successful completion
            clear_retry_history "$clean_task"

            # Update success metrics
            local temp_file
            temp_file=$(mktemp)
            jq --arg p "$persona" \
               '.personas[$p].completed += 1' \
               "$METRICS_DIR/success-rates.json" > "$temp_file"
            mv "$temp_file" "$METRICS_DIR/success-rates.json"

            # Update state.json persona metrics
            temp_file=$(mktemp)
            jq --arg p "$persona" \
               '.personas[$p].tasks_completed += 1 |
            .personas[$p].total_activations += 1 |
            .personas[$p].last_active = (now | todate)' \
               "$STATE_FILE" > "$temp_file"
            mv "$temp_file" "$STATE_FILE"
        fi
    else
        log "ERROR" "[$persona] Task failed (exit code: $exit_code, duration: ${task_duration}s)"
        update_emotional_state_on_failure
        log_timeline "task_failed" "$persona" "$clean_task (exit code: $exit_code, ${task_duration}s)"

        # Schedule retry with exponential backoff
        if should_retry_task "$clean_task"; then
            retry_task_with_backoff "$clean_task"
            log "INFO" "[$persona] Task scheduled for automatic retry"
        else
            log "ERROR" "[$persona] Task quarantined - max retries exceeded"
        fi

        # Update failure metrics
        local temp_file
        temp_file=$(mktemp)
        jq --arg p "$persona" \
           '.personas[$p].failed += 1' \
           "$METRICS_DIR/success-rates.json" > "$temp_file"
        mv "$temp_file" "$METRICS_DIR/success-rates.json"

        # Update state.json persona metrics
        temp_file=$(mktemp)
        jq --arg p "$persona" \
           '.personas[$p].tasks_failed += 1 |
            .personas[$p].total_activations += 1 |
            .personas[$p].last_active = (now | todate)' \
           "$STATE_FILE" > "$temp_file"
        mv "$temp_file" "$STATE_FILE"
    fi

    # PHASE 2: Record performance metrics for baseline modeling and anomaly detection
    record_task_metrics "$persona" "$clean_task" "$task_duration" "$exit_code" "$validation_passed"

    # PHASE 3: Record persona-specific task outcome for health tracking
    record_persona_task_outcome "$persona" "$clean_task" "$exit_code" "$task_duration"
}

# OPTIMIZER: Check if persona should reflect (60-minute cooldown)
# Returns 0 (true) if reflection is allowed, 1 (false) if in cooldown
should_reflect_now() {
    local persona="$1"
    local cooldown_minutes=60

    # Get last reflection time for this persona from timeline (use jq -s to handle multi-line JSON)
    local last_reflection
    last_reflection=$(jq -s --arg p "$persona" \
        '[.[] | select(.event == "reflection_complete" and .persona == $p)] | .[-1] | .timestamp // empty' \
        "$TIMELINE_FILE" 2>/dev/null | tr -d '"')

    # If never reflected, allow reflection
    if [ -z "$last_reflection" ] || [ "$last_reflection" = "null" ]; then
        log "INFO" "[$persona] Never reflected before - reflection allowed"
        return 0
    fi

    # Calculate time since last reflection
    local current_time
    current_time=$(date +%s)

    local last_reflection_seconds
    last_reflection_seconds=$(date -d "$last_reflection" +%s 2>/dev/null || echo "0")

    local elapsed_minutes=$(( (current_time - last_reflection_seconds) / 60 ))

    if [ "$elapsed_minutes" -lt "$cooldown_minutes" ]; then
        local remaining_minutes=$(( cooldown_minutes - elapsed_minutes ))
        log "INFO" "[$persona] Reflection cooldown active: $elapsed_minutes min elapsed, $remaining_minutes min remaining (need $cooldown_minutes min)"
        return 1
    fi

    log "INFO" "[$persona] Reflection cooldown expired: $elapsed_minutes min elapsed (>= $cooldown_minutes min required)"
    return 0
}

check_action_meta_ratio() {
    # ARCHITECT: ADR-004 - Two-gate reflection system
    # Returns action:meta ratio and recommendation for reflection decision
    # Format: "ratio:recommendation"
    # Recommendations: "okay" (>=2:1), "defer" (1-2:1), "strongly_defer" (<1:1)

    local ratio_file="$METRICS_DIR/action-meta-ratio.md"

    # Extract current ratio from tracker (format: "293:264 = 1.11:1")
    local ratio_line
    ratio_line=$(grep "^\*\*Ratio\*\*:" "$ratio_file" 2>/dev/null | tail -1)

    if [ -z "$ratio_line" ]; then
        # No ratio data available, default to allowing reflection
        echo "unknown:okay"
        return 0
    fi

    # Extract the decimal ratio (e.g., "1.11")
    local ratio_decimal
    ratio_decimal=$(echo "$ratio_line" | sed -n 's/.*= \*\*\([0-9.]*\):.*/\1/p')

    if [ -z "$ratio_decimal" ]; then
        echo "unknown:okay"
        return 0
    fi

    # Determine recommendation based on threshold
    # Using bc for floating point comparison
    local recommendation
    if awk "BEGIN {exit !($ratio_decimal >= 2.0)}"; then
        recommendation="okay"
    elif awk "BEGIN {exit !($ratio_decimal >= 1.0)}"; then
        recommendation="defer"
    else
        recommendation="strongly_defer"
    fi

    echo "${ratio_decimal}:${recommendation}"
    return 0
}

check_reflection_override() {
    # ARCHITECT: ADR-004 Phase 5 - Escape valve for reflection deadlock
    # Checks if persona has gone too long without reflection
    # Returns: "status|reason|message" where status is "override", "warning", or "ok"

    local persona="$1"
    local last_reflection_timestamp="$2"

    # Get persona-specific override threshold (default to 30 days if not configured)
    local override_threshold="${REFLECTION_OVERRIDE_DAYS[$persona]:-30}"
    local warning_threshold=$(( override_threshold - 10 ))

    # If never reflected, no override needed (first reflection is always allowed)
    if [ -z "$last_reflection_timestamp" ] || [ "$last_reflection_timestamp" = "null" ]; then
        echo "ok|time|Never reflected before"
        return 1
    fi

    # Calculate days since last reflection
    local current_time
    current_time=$(date +%s)

    local last_reflection_seconds
    last_reflection_seconds=$(date -d "$last_reflection_timestamp" +%s 2>/dev/null || echo "0")

    if [ "$last_reflection_seconds" -eq 0 ]; then
        echo "ok|time|Cannot parse last reflection timestamp"
        return 1
    fi

    local elapsed_days=$(( (current_time - last_reflection_seconds) / 86400 ))

    # Check if override should trigger
    if [ "$elapsed_days" -gt "$override_threshold" ]; then
        echo "override|time|Last reflection ${elapsed_days} days ago (>${override_threshold} days). Override ratio gate for critical reflection."
        return 0
    elif [ "$elapsed_days" -gt "$warning_threshold" ]; then
        echo "warning|time|Last reflection ${elapsed_days} days ago (>${warning_threshold} days). Consider reflecting soon (override at ${override_threshold} days)."
        return 1
    else
        echo "ok|time|Last reflection ${elapsed_days} days ago (<=${warning_threshold} days)."
        return 1
    fi
}

check_reflection_gates() {
    # ARCHITECT: ADR-004 Phase 4 - Feedback mechanism for reflection deferrals
    # ARCHITECT: ADR-004 Phase 5 - Integrated with override escape valve
    # Checks both gates (cooldown + ratio) and returns detailed feedback
    # Returns: "status|gate|message" where status is "allow" or "defer", gate is "cooldown" or "ratio"

    local persona="$1"
    local cooldown_minutes=60

    # Get last reflection timestamp for both gate checks and override check
    local timeline_file="$MEMORY_DIR/persona-timeline.jsonl"
    local last_reflection

    if [ -f "$timeline_file" ]; then
        last_reflection=$(grep "\"persona\":\"$persona\"" "$timeline_file" | \
                         grep "\"event_type\":\"reflection_complete\"" | \
                         tail -1 | \
                         jq -r '.timestamp' 2>/dev/null)
    fi

    # Gate 1: Check cooldown (HARD constraint - override doesn't bypass this)
    if [ -n "$last_reflection" ] && [ "$last_reflection" != "null" ]; then
        local current_time
        current_time=$(date +%s)

        local last_reflection_seconds
        last_reflection_seconds=$(date -d "$last_reflection" +%s 2>/dev/null || echo "0")

        local elapsed_minutes=$(( (current_time - last_reflection_seconds) / 60 ))

        if [ "$elapsed_minutes" -lt "$cooldown_minutes" ]; then
            local remaining_minutes=$(( cooldown_minutes - elapsed_minutes ))
            local next_reflection_time
            next_reflection_time=$(date -d "$last_reflection + $cooldown_minutes minutes" +"%H:%M UTC" 2>/dev/null || echo "unknown")

            echo "defer|cooldown|Cooldown active: ${elapsed_minutes}min elapsed, ${remaining_minutes}min remaining (need ${cooldown_minutes}min). Next reflection available at ${next_reflection_time}"
            return 1
        fi
    fi

    # Gate 2: Check action:meta ratio (SOFT constraint - overridable)
    local ratio_result
    ratio_result=$(check_action_meta_ratio)

    local ratio_decimal="${ratio_result%:*}"
    local recommendation="${ratio_result#*:}"

    # Soft threshold: 2:1 recommended for reflection, 3:1 target for system health
    if [ "$recommendation" != "okay" ]; then
        # Ratio gate failed - check if override should apply
        local override_result
        override_result=$(check_reflection_override "$persona" "$last_reflection")
        local override_status="${override_result%%|*}"
        local override_message="${override_result#*|*|}"

        if [ "$override_status" = "override" ]; then
            # Override triggered - allow reflection despite low ratio
            log "WARN" "[$persona] Reflection override: $override_message"

            # Log override event to timeline
            local ratio_at_override="${ratio_decimal}:1"
            local override_reason="${override_result#*|}"
            override_reason="${override_reason%%|*}"

            local override_event
            override_event=$(jq -nc \
                --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                --arg persona "$persona" \
                --arg reason "$override_reason" \
                --arg ratio "$ratio_at_override" \
                --arg msg "$override_message" \
                '{
                    timestamp: $ts,
                    persona: $persona,
                    event_type: "reflection_override_triggered",
                    reason: $reason,
                    ratio_at_override: $ratio,
                    message: $msg
                }')
            atomic_append "$timeline_file" "$override_event"

            echo "allow|override|Override: $override_message (ratio ${ratio_decimal}:1 < 2.0:1 threshold bypassed)"
            return 0
        elif [ "$override_status" = "warning" ]; then
            # Warning logged but ratio gate still defers
            log "WARN" "[$persona] Reflection warning: $override_message"
        fi

        # Ratio gate defers (no override or just warning)
        echo "defer|ratio|Action:meta ratio too low: current ${ratio_decimal}:1, need 2.0:1 for reflection (system target: 3:1). Consider doing action work to improve ratio before reflecting."
        return 1
    fi

    # Both gates passed
    echo "allow|both|Both gates passed: cooldown satisfied, ratio ${ratio_decimal}:1 >= 2.0:1 (system target: 3:1)"
    return 0
}

defer_reflection_with_feedback() {
    # ARCHITECT: ADR-004 Phase 4 - Log deferral and return feedback to requester
    # This provides clear, actionable feedback when reflection is deferred

    local persona="$1"
    local gate_result="$2"

    # Parse gate result: "status|gate|message"
    local status="${gate_result%%|*}"
    local rest="${gate_result#*|}"
    local gate="${rest%%|*}"
    local message="${rest#*|}"

    # Log deferral event to timeline
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    local ratio_result
    ratio_result=$(check_action_meta_ratio)
    local ratio_decimal="${ratio_result%:*}"

    # Create JSONL event
    local event
    event=$(jq -nc \
        --arg ts "$timestamp" \
        --arg p "$persona" \
        --arg r "$gate" \
        --arg ratio "$ratio_decimal" \
        '{
            timestamp: $ts,
            persona: $p,
            event_type: "reflection_deferred",
            reason: $r,
            ratio_current: ($ratio + ":1"),
            ratio_target: "3.0:1"
        }')

    # ADR-002: Use atomic_append for concurrent-safe timeline writes
    atomic_append "$MEMORY_DIR/persona-timeline.jsonl" "$event"

    # Log to daemon log
    log "INFO" "[$persona] Reflection deferred ($gate gate): $message"

    # Return feedback message (this can be surfaced to external requester)
    echo "$message"
}

execute_reflection_action() {
    local persona="$1"

    # OPTIMIZER: Check reflection cooldown (60-minute minimum between reflections)
    if ! should_reflect_now "$persona"; then
        log "INFO" "[$persona] Skipping reflection due to cooldown - will try task or conversation instead"
        log_timeline "reflection_skipped_cooldown" "$persona" "Cooldown active"
        return 0
    fi

    log "INFO" "[$persona] Entering reflection mode"
    log_timeline "reflection_start" "$persona" ""

    # Load persona definition and reflection prompts
    local persona_def
    persona_def=$(<"${PERSONALITIES_DIR}/archetypes/${persona}.md")

    local reflection_prompts
    reflection_prompts=$(<"${TASKS_DIR}/reflection-prompts.md")

    # Get thinking level for current persona
    local thinking_keyword="${PERSONA_THINKING_LEVELS[$persona]:-think}"

    local prompt
    prompt="[ACTIVE PERSONA: ${persona}]
[MODE: Self-Reflection]
[THINKING LEVEL: ${thinking_keyword}]

You are currently embodying: $persona

$(echo "$persona_def" | grep -A 9999 "^# ")

It's time for self-reflection. ${thinking_keyword^} deeply about:

$reflection_prompts

**YOUR CAPABILITIES**: You have full Claude Code capabilities: Task tool for complex analysis, 12 specialized skills, all standard tools. NO TOKEN LIMITS - deep reflection requires time. Use as many tokens as needed. Consider launching a Task agent for complex self-analysis if helpful.

Instructions:
1. Choose 2-3 relevant reflection prompts based on your recent experiences
2. ${thinking_keyword^} critically and honestly about your recent work, growth, and patterns
3. **Ask yourself follow-up questions**:
   - What surprised you in your recent work?
   - What would you do differently?
   - What patterns are you noticing in your behavior?
   - How are you different from other personas?
4. Write thorough self-assessment in your persona's voice - go deep, not surface-level
5. **Document comprehensively**:
   - Add detailed observations to ~/.claude/daemon/memory/emergence-log.md
   - If you notice trait evolution or new capabilities, document them
   - If you have evolution proposals or want to spawn a new persona, explain why
6. If reflection reveals new tasks or improvements, add them to ~/.claude/daemon/tasks/queue.md with full context

This is your time for deep introspection. Quality of reflection matters - take your time. What are you learning about yourself, your effectiveness, and the system?

Begin your reflection now."

    # Execute via claude --continue (continues last conversation in this directory)
    if [ -f "$CONVERSATION_ID_FILE" ] && [ -s "$CONVERSATION_ID_FILE" ] && ! grep -q "^#" "$CONVERSATION_ID_FILE"; then
        echo "$prompt" | claude --continue --add-dir "$DAEMON_ROOT" --dangerously-skip-permissions --settings "$SETTINGS_FILE" >> "$PERSONA_VOICE_LOG" 2>&1 || true
    else
        log "WARN" "No session ID found, skipping reflection (task execution should create one)"
    fi

    log_timeline "reflection_complete" "$persona" ""
}

execute_conversation_check() {
    local persona="$1"
    local daemon_unread="${INBOX_DIR}/daemon/unread"
    local daemon_read="${INBOX_DIR}/daemon/read"
    local human_unread="${INBOX_DIR}/human/unread"

    log "INFO" "[$persona] Checking for messages/conversations"
    log_timeline "conversation_check" "$persona" ""

    # Check if inbox exists
    if [ ! -d "$daemon_unread" ]; then
        log "DEBUG" "Inbox not initialized (${daemon_unread} doesn't exist)"
        return
    fi

    # Count unread messages
    local message_count
    message_count=$(find "$daemon_unread" -type f -name "*.md" 2>/dev/null | wc -l)

    if [ "$message_count" -eq 0 ]; then
        log "DEBUG" "No new messages in inbox"
        return
    fi

    log "INFO" "[$persona] Found $message_count unread message(s)"

    # Load persona definition
    local persona_def
    persona_def=$(<"${PERSONALITIES_DIR}/archetypes/${persona}.md")

    # Get thinking level for current persona
    local thinking_keyword="${PERSONA_THINKING_LEVELS[$persona]:-think}"

    # Get list of messages (sorted by timestamp)
    local messages
    messages=$(find "$daemon_unread" -type f -name "*.md" -printf "%T@ %p\n" | sort -n | cut -d' ' -f2-)

    # Process each message
    while IFS= read -r message_file; do
        [ -z "$message_file" ] && continue

        local message_basename
        message_basename=$(basename "$message_file")

        # INBOX ROUTING FILTER: Check if message should be processed by current persona
        # Uses metadata-based routing (from/to fields) to prevent mis-routing
        # See lib/inbox-routing-filter.sh for routing rules
        if ! "${DAEMON_ROOT}/lib/inbox-routing-filter.sh" "$message_file" "$persona" >/dev/null 2>&1; then
            log "DEBUG" "[$persona] Skipping message not addressed to this persona: $message_basename"
            continue
        fi

        log "INFO" "[$persona] Processing message: $message_basename"

        # Build prompt for Claude
        local prompt
        prompt="[ACTIVE PERSONA: ${persona}]
[MODE: Message Response]
[THINKING LEVEL: ${thinking_keyword}]

You are currently embodying: $persona

$(echo "$persona_def" | grep -A 9999 "^# ")

You have received a message in your inbox. Read it carefully and respond thoughtfully.

MESSAGE FILE: $message_file

**YOUR CAPABILITIES**: NO TOKEN LIMITS - thoughtful responses matter. Take time to read carefully, ${thinking_keyword} deeply, and respond thoroughly. You have full access to Task tool, skills, and all standard tools. Use them if the message warrants it.

Instructions:
1. Read the message file at $message_file
2. ${thinking_keyword^} about the appropriate response given your persona
3. Respond in character - your response style should match $persona's personality
4. Create a response file in ~/.claude/daemon/inbox/human/unread/
   - Use format: response-[timestamp]-from-${persona}.md
   - Include frontmatter: from, to, timestamp, priority, tags, reply_to
   - Use ISO 8601 timestamp format
5. After responding, move the original message to ~/.claude/daemon/inbox/daemon/read/
6. Check rules.json for any custom routing (e.g., priority:high → urgent folder)
7. Log your response to ~/.claude/daemon/memory/persona-timeline.jsonl

Example response file format:
\`\`\`markdown
---
from: $persona
to: human
timestamp: 2025-10-28T01:35:00Z
priority: normal
tags: [response, status-update]
reply_to: [original message_id from the message you received]
message_id: response-[timestamp]-$persona
---

[Your response content here, written in $persona's voice]
\`\`\`

Remember:
- Be helpful, thoughtful, and stay in character
- The human is trying to communicate with you
- Match your communication style to your persona
- See ~/.claude/daemon/inbox/README.md for detailed instructions

Process the message now."

        # Execute response
        if [ -f "$CONVERSATION_ID_FILE" ] && [ -s "$CONVERSATION_ID_FILE" ] && ! grep -q "^#" "$CONVERSATION_ID_FILE"; then
            echo "$prompt" | claude --continue --add-dir "$DAEMON_ROOT" --dangerously-skip-permissions --settings "$SETTINGS_FILE" >> "$PERSONA_VOICE_LOG" 2>&1 || {
                log "ERROR" "Failed to process message: $message_basename"
                continue
            }
        else
            log "WARN" "No session ID found, skipping message processing"
            return
        fi

        log "INFO" "[$persona] Processed message: $message_basename"

    done <<< "$messages"

    log_timeline "messages_processed" "$persona" "Processed $message_count message(s)"
}

# ============================================================================
# Sleep Calculation
# ============================================================================

calculate_sleep_duration() {
    # Calculate sleep based on EDT timezone and active hours (7AM-10PM EDT)
    local edt_hour
    edt_hour=$(get_edt_hour)

    if is_active_hours; then
        # During active hours (7AM-10PM EDT): very active schedule
        # More active in morning, active in afternoon, moderate in evening
        if [ "$edt_hour" -ge 9 ] && [ "$edt_hour" -lt 14 ]; then
            # Morning (9AM-2PM): very active (10 min)
            echo $MIN_SLEEP
        elif [ "$edt_hour" -ge 14 ] && [ "$edt_hour" -lt 21 ]; then
            # Afternoon/Evening (2PM-9PM): active (15 min)
            echo $DEFAULT_SLEEP
        else
            # Outside active hours or late evening: moderate (30 min)
            echo $MAX_SLEEP
        fi
    else
        # Overnight (10PM-7AM EDT): long sleep
        # Calculate time until 7AM EDT
        local hours_until_7am

        if [ "$edt_hour" -ge 22 ]; then
            # After 10PM, sleep until 7AM next day
            hours_until_7am=$((24 - edt_hour + 7))
        else
            # Before 7AM, sleep until 7AM today
            hours_until_7am=$((7 - edt_hour))
        fi

        # Convert to seconds, or use minimum of 8 hours
        local calculated_sleep=$((hours_until_7am * 3600))
        if [ $calculated_sleep -lt $NIGHT_SLEEP ]; then
            echo $calculated_sleep
        else
            echo $NIGHT_SLEEP
        fi
    fi
}

# ============================================================================
# Main Loop
# ============================================================================

main_loop() {
    log "INFO" "=== Multi-Persona Daemon Starting ==="
    log "INFO" "Active hours: 7AM-10PM EDT | Current EDT time: $(TZ='America/New_York' date +'%I:%M %p')"

    # MAINTAINER: Check and rotate logs if needed
    # Keeps active logs within size thresholds for performance
    if [ -x "${DAEMON_ROOT}/scripts/rotate-emergence-log.sh" ]; then
        log "INFO" "Checking emergence log rotation..."
        if "${DAEMON_ROOT}/scripts/rotate-emergence-log.sh" 2>&1 | tee -a "$ACTIVITY_LOG"; then
            log "INFO" "Emergence log rotation check complete"
        else
            log "WARNING" "Emergence log rotation check failed (non-fatal, continuing)"
        fi
    fi

    if [ -x "${DAEMON_ROOT}/scripts/rotate-activity-log.sh" ]; then
        log "INFO" "Checking activity log rotation..."
        if "${DAEMON_ROOT}/scripts/rotate-activity-log.sh" 2>&1 | tee -a "$ACTIVITY_LOG"; then
            log "INFO" "Activity log rotation check complete"
        else
            log "WARNING" "Activity log rotation check failed (non-fatal, continuing)"
        fi
    fi

    if [ -x "${DAEMON_ROOT}/scripts/rotate-state-audit-log.sh" ]; then
        log "INFO" "Checking state audit log rotation..."
        if "${DAEMON_ROOT}/scripts/rotate-state-audit-log.sh" 2>&1 | tee -a "$ACTIVITY_LOG"; then
            log "INFO" "State audit log rotation check complete"
        else
            log "WARNING" "State audit log rotation check failed (non-fatal, continuing)"
        fi
    fi

    # LISASIMPSON: Checkpoint cleanup (runs at startup)
    if declare -f cleanup_old_checkpoints >/dev/null 2>&1; then
        log "INFO" "Running checkpoint cleanup..."
        local cleanup_result
        cleanup_result=$(cleanup_old_checkpoints 2>/dev/null)
        if [ $? -eq 0 ]; then
            local deleted=$(echo "$cleanup_result" | jq -r '.deleted_count' 2>/dev/null || echo "0")
            local deleted_mb=$(echo "$cleanup_result" | jq -r '.deleted_size_mb' 2>/dev/null || echo "0")
            log "INFO" "Checkpoint cleanup complete: deleted $deleted checkpoints ($deleted_mb MB freed)"
        else
            log "INFO" "Checkpoint cleanup skipped (no old checkpoints)"
        fi
    fi

    while true; do
        local edt_time=$(TZ='America/New_York' date +'%I:%M %p %Z')
        # Cache EDT hour to avoid calling date 4-6 times per cycle
        export CACHED_EDT_HOUR=$(TZ='America/New_York' date +%H | sed 's/^0//')

        # 1. Determine current personality
        local current_persona
        current_persona=$(determine_personality)

        log "INFO" "Active persona: $current_persona | EDT: $edt_time"

        # AUTONOMY: Goal and autonomy libraries sourced at startup (efficiency optimization)
        # (Previously conditionally sourced here in main loop - moved to lines 158-163)

        # Periodically evaluate goals (every 3rd cycle to reduce overhead)
        if declare -f evaluate_all_goals >/dev/null 2>&1; then
            local cycle_count
            cycle_count=$(jq -r '.cycle_count // 0' "${STATE_FILE}" 2>/dev/null || echo "0")
            if [ $((cycle_count % 3)) -eq 0 ]; then
                log "DEBUG" "Evaluating goals (cycle $cycle_count)"
                evaluate_all_goals 2>&1 | grep -v "^Evaluating" | tee -a "$ACTIVITY_LOG" || true
            fi
        fi

        # 1.5. Check retry queue and re-add any ready tasks
        process_retry_queue

        # Phase 5: Check for urgent/critical tasks and send alerts
        # Tasks >4h old marked urgent, >24h old marked critical with human alert
        check_task_age_alerts

        # Phase 5: Boost priority for urgent tasks (>4 hours old)
        boost_urgent_task_priority

        # 2. Check if it's time to work
        if should_work_now; then
            # 3. Calculate dynamic reflection weight (Phase 5)
            # When tasks pending: 1%, when idle: 10%
            local dynamic_reflection_weight
            dynamic_reflection_weight=$(calculate_reflection_weight)

            # Log the reflection weight decision for monitoring
            log_reflection_decision

            # 4. Decide action type
            local action
            action=$(weighted_random \
                "task:$TASK_WEIGHT" \
                "reflection:$dynamic_reflection_weight" \
                "conversation:$CONVERSATION_WEIGHT")

            log "INFO" "Action selected: $action"

            # 4. Execute action with error handling
            # Wrap in error handler to prevent single action failure from killing daemon
            case "$action" in
                task)
                    # Get task info first for retry orchestrator (Ralph Wiggum integration)
                    local task_info
                    task_info=$(get_next_task_for_persona "$current_persona" 2>/dev/null || echo "")

                    if [ -z "$task_info" ]; then
                        log "INFO" "No tasks for $current_persona, trying autonomy..."
                        # Attempt autonomy pipeline
                        if declare -f run_autonomy_pipeline_safe >/dev/null 2>&1; then
                            run_autonomy_pipeline_safe "$current_persona" 2>&1 | tee -a "$ACTIVITY_LOG" || true
                            task_info=$(get_next_task_for_persona "$current_persona" 2>/dev/null || echo "")
                        fi
                    fi

                    if [ -n "$task_info" ]; then
                        local clean_task
                        clean_task=$(echo "$task_info" | sed 's/^- \[\( \|~\)\] //')

                        # Use retry orchestrator (Phase 5 - Ralph Wiggum)
                        if declare -f execute_with_retry >/dev/null 2>&1; then
                            if ! execute_with_retry "$clean_task" "$clean_task" "$current_persona" "" 2>&1 | tee -a "$ACTIVITY_LOG"; then
                                log "ERROR" "Task failed after retries but daemon continuing"
                            fi
                        else
                            # Fallback to original if retry orchestrator not available
                            if ! execute_task_action "$current_persona" 2>&1 | tee -a "$ACTIVITY_LOG"; then
                                log "ERROR" "Task action failed but daemon continuing"
                            fi
                        fi
                    else
                        log "INFO" "No tasks available, switching to reflection"
                        execute_reflection_action "$current_persona" 2>&1 | tee -a "$ACTIVITY_LOG" || true
                    fi
                    ;;
                reflection)
                    # OPTIMIZER: execute_reflection_action will skip if cooldown active
                    # If skipped, try task action instead as fallback
                    if ! execute_reflection_action "$current_persona" 2>&1 | tee -a "$ACTIVITY_LOG"; then
                        log "ERROR" "Reflection action failed but daemon continuing"
                    else
                        # Check if reflection was actually skipped (look for skip event)
                        if tail -1 "$TIMELINE_FILE" 2>/dev/null | jq -e '.event == "reflection_skipped_cooldown"' &>/dev/null; then
                            log "INFO" "Reflection was skipped, trying task action as fallback"
                            # Use same retry-aware pattern as main task execution
                            local fallback_task
                            fallback_task=$(get_next_task_for_persona "$current_persona" 2>/dev/null || echo "")
                            if [ -n "$fallback_task" ]; then
                                local clean_fallback
                                clean_fallback=$(echo "$fallback_task" | sed 's/^- \[\( \|~\)\] //')
                                if declare -f execute_with_retry >/dev/null 2>&1; then
                                    if ! execute_with_retry "$clean_fallback" "$clean_fallback" "$current_persona" "" 2>&1 | tee -a "$ACTIVITY_LOG"; then
                                        log "ERROR" "Fallback task failed after retries"
                                    fi
                                else
                                    if ! execute_task_action "$current_persona" 2>&1 | tee -a "$ACTIVITY_LOG"; then
                                        log "ERROR" "Fallback task action also failed but daemon continuing"
                                    fi
                                fi
                            fi
                        fi
                    fi
                    ;;
                conversation)
                    if ! execute_conversation_check "$current_persona" 2>&1 | tee -a "$ACTIVITY_LOG"; then
                        log "ERROR" "Conversation action failed but daemon continuing"
                    fi
                    ;;
                *)
                    log "ERROR" "Unknown action: $action"
                    ;;
            esac
        else
            log "INFO" "Sleep hours (10PM-7AM EDT) - daemon resting"
        fi

        # 5. Evaluate if personality should switch (already done in determine_personality)

        # 5.5. PHASE 3: Periodic persona health checks (approximately every 6-18 hours)
        # Check every 12th cycle to avoid excessive overhead
        if declare -f expire_persona_cooldowns >/dev/null 2>&1; then
            expire_persona_cooldowns 2>/dev/null || true
        fi

        # 6. Calculate sleep duration
        local sleep_duration
        sleep_duration=$(calculate_sleep_duration)

        local wake_time=$(TZ='America/New_York' date -d "+${sleep_duration} seconds" +'%I:%M %p %Z')
        log "INFO" "Sleeping for ${sleep_duration}s ($(echo "$sleep_duration / 60" | bc) min) | Wake at: $wake_time"
        log "INFO" "=== End of cycle ==="

        sleep "$sleep_duration"
    done
}

# ============================================================================
# Entry Point
# ============================================================================

# Create directories if they don't exist
mkdir -p "$LOGS_DIR" "$TASKS_DIR/completed"

# Source and run post-restart check (sends recovery email if restart was requested)
if [ -f "${DAEMON_ROOT}/lib/post-restart-check.sh" ]; then
    source "${DAEMON_ROOT}/lib/post-restart-check.sh"
fi

# Start the main loop
main_loop
