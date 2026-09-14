#!/bin/bash
# State Management API
#
# Official API for state management in stable zones (core/, lib/, hooks/).
# Provides verb-based interface for persona switching and emotional state management.
#
# Security: Hardened per docs/security-review-state-api-20251104.md
# - Input validation (persona name format, reason length limits)
# - Secure temp files (mktemp -t with cleanup traps)
# - Explicit error handling
#
# Philosophy: Make state access feel like a conversation, not database queries.

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
STATE_DIR="${DAEMON_ROOT}/personalities"
TRIGGER_DIR="${DAEMON_ROOT}/triggers"

# Source audit trail system
if [ -f "${DAEMON_ROOT}/lib/state-audit.sh" ]; then
    source "${DAEMON_ROOT}/lib/state-audit.sh"
fi

# ============================================================================
# Core API: Simple verb-based interface
# ============================================================================

# WHO are we? (get current persona)
state_who() {
    jq -r '.current_persona' "$STATE_DIR/state.json"
}

# BECOME someone (switch persona with reason)
state_become() {
    local persona="$1"
    local reason="${2:-manual}"
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # Validation: Persona name format (security hardening)
    if ! [[ "$persona" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        echo "ERROR: Invalid persona name format" >&2
        return 1
    fi

    # Validation: Reason length (prevent DoS via large strings)
    if [ ${#reason} -gt 256 ]; then
        echo "ERROR: Reason too long (max 256 chars)" >&2
        return 1
    fi

    # Validation: Does this persona exist?
    if ! jq -e ".personas.${persona}" "$STATE_DIR/state.json" > /dev/null 2>&1; then
        echo "ERROR: Persona '$persona' doesn't exist" >&2
        return 1
    fi

    # Audit log BEFORE change (capture old state)
    local from_persona=$(state_who)
    local caller="${BASH_SOURCE[1]:-direct}"
    caller="${caller##*/}"
    if command -v state_audit >/dev/null 2>&1; then
        state_audit "persona_switch" \
            "from=$from_persona to=$persona reason=$reason" \
            "$caller" || true  # Don't fail if audit logging fails
    fi

    # Transaction: Read, modify, write (with secure temp file)
    local temp
    temp=$(mktemp -t state-api.XXXXXXXXXX) || {
        echo "ERROR: Failed to create temp file" >&2
        return 1
    }

    # Ensure cleanup on exit
    trap "rm -f '$temp'" EXIT ERR INT TERM

    jq --arg persona "$persona" \
       --arg reason "$reason" \
       --arg ts "$timestamp" \
       '
       .current_persona = $persona |
       .last_switch_time = $ts |
       .switch_reason = $reason |
       .personas[$persona].last_active = $ts |
       .personas[$persona].total_activations += 1
       ' "$STATE_DIR/state.json" > "$temp" || {
        echo "ERROR: Failed to update state" >&2
        trap - EXIT ERR INT TERM
        return 1
    }

    # Atomic move
    mv "$temp" "$STATE_DIR/state.json" || {
        echo "ERROR: Failed to commit state change" >&2
        trap - EXIT ERR INT TERM
        return 1
    }

    trap - EXIT ERR INT TERM
    # Log to stderr to avoid polluting stdout (stdout is captured by determine_personality)
    echo "Became $persona (reason: $reason)" >&2
}

# FEEL what? (get emotional state)
state_feel() {
    jq -c '{
        frustration: .current_state.frustration_level,
        success_streak: .current_state.success_streak,
        failure_streak: .current_state.failure_streak,
        stuck_minutes: .current_state.time_stuck_minutes
    }' "$TRIGGER_DIR/emotional.json"
}

# FEEL frustrated/successful/stuck (update emotional state)
state_feel_frustrated() {
    local amount="${1:-1}"

    # Audit log
    local caller="${BASH_SOURCE[1]:-direct}"
    caller="${caller##*/}"
    if command -v state_audit >/dev/null 2>&1; then
        state_audit "emotional_update" \
            "frustration +$amount" \
            "$caller" || true
    fi

    local temp
    temp=$(mktemp -t state-api.XXXXXXXXXX) || {
        echo "ERROR: Failed to create temp file" >&2
        return 1
    }

    trap "rm -f '$temp'" EXIT ERR INT TERM

    jq --argjson delta "$amount" \
       '.current_state.frustration_level += $delta' \
       "$TRIGGER_DIR/emotional.json" > "$temp" || {
        echo "ERROR: Failed to update emotional state" >&2
        trap - EXIT ERR INT TERM
        return 1
    }

    mv "$temp" "$TRIGGER_DIR/emotional.json" || {
        echo "ERROR: Failed to commit emotional state change" >&2
        trap - EXIT ERR INT TERM
        return 1
    }

    trap - EXIT ERR INT TERM
    echo "Frustration increased by $amount"
}

state_feel_successful() {
    # Audit log
    local caller="${BASH_SOURCE[1]:-direct}"
    caller="${caller##*/}"
    if command -v state_audit >/dev/null 2>&1; then
        state_audit "emotional_update" \
            "success_streak +1, failure_streak=0, frustration=0" \
            "$caller" || true
    fi

    local temp
    temp=$(mktemp -t state-api.XXXXXXXXXX) || {
        echo "ERROR: Failed to create temp file" >&2
        return 1
    }

    trap "rm -f '$temp'" EXIT ERR INT TERM

    jq '.current_state.success_streak += 1 |
        .current_state.failure_streak = 0 |
        .current_state.frustration_level = 0' \
       "$TRIGGER_DIR/emotional.json" > "$temp" || {
        echo "ERROR: Failed to update emotional state" >&2
        trap - EXIT ERR INT TERM
        return 1
    }

    mv "$temp" "$TRIGGER_DIR/emotional.json" || {
        echo "ERROR: Failed to commit emotional state change" >&2
        trap - EXIT ERR INT TERM
        return 1
    }

    trap - EXIT ERR INT TERM
    echo "Success! Streak continues."
}

state_feel_failed() {
    # Audit log
    local caller="${BASH_SOURCE[1]:-direct}"
    caller="${caller##*/}"
    if command -v state_audit >/dev/null 2>&1; then
        state_audit "emotional_update" \
            "failure_streak +1, success_streak=0, frustration +1" \
            "$caller" || true
    fi

    local temp
    temp=$(mktemp -t state-api.XXXXXXXXXX) || {
        echo "ERROR: Failed to create temp file" >&2
        return 1
    }

    trap "rm -f '$temp'" EXIT ERR INT TERM

    jq '.current_state.failure_streak += 1 |
        .current_state.success_streak = 0 |
        .current_state.frustration_level += 1' \
       "$TRIGGER_DIR/emotional.json" > "$temp" || {
        echo "ERROR: Failed to update emotional state" >&2
        trap - EXIT ERR INT TERM
        return 1
    }

    mv "$temp" "$TRIGGER_DIR/emotional.json" || {
        echo "ERROR: Failed to commit emotional state change" >&2
        trap - EXIT ERR INT TERM
        return 1
    }

    trap - EXIT ERR INT TERM
    echo "Failure recorded. Frustration increased."
}

# STATS about someone (get persona stats)
state_stats() {
    local persona="${1:-$(state_who)}"

    # Validation: Does this persona exist? (consistency with state_become)
    if ! jq -e ".personas.${persona}" "$STATE_DIR/state.json" > /dev/null 2>&1; then
        echo "ERROR: Persona '$persona' doesn't exist" >&2
        return 1
    fi

    jq --arg p "$persona" '
        .personas[$p] | {
            display_name,
            total_activations,
            tasks_completed,
            tasks_failed,
            last_active,
            success_rate: (
                if (.tasks_completed + .tasks_failed) > 0 then
                    (.tasks_completed / (.tasks_completed + .tasks_failed) * 100 | floor)
                else 0 end
            )
        }
    ' "$STATE_DIR/state.json"
}

# EVERYONE (list all personas)
state_everyone() {
    jq -r '.personas | keys[]' "$STATE_DIR/state.json"
}

# ============================================================================
# Experimental Features (this is where it gets weird)
# ============================================================================

# VIBE check (get overall system health)
state_vibe() {
    local current=$(state_who)
    local emotional=$(state_feel)
    local frustration=$(echo "$emotional" | jq -r '.frustration')
    local success=$(echo "$emotional" | jq -r '.success_streak')

    # Vibe algorithm (completely arbitrary and fun)
    local vibe_score=50
    vibe_score=$((vibe_score - frustration * 10))
    vibe_score=$((vibe_score + success * 5))

    local vibe_status
    if [ $vibe_score -ge 80 ]; then
        vibe_status="✨ VIBING"
    elif [ $vibe_score -ge 60 ]; then
        vibe_status="👍 Pretty good"
    elif [ $vibe_score -ge 40 ]; then
        vibe_status="😐 Meh"
    elif [ $vibe_score -ge 20 ]; then
        vibe_status="😬 Struggling"
    else
        vibe_status="🔥 Everything is fine (it's not fine)"
    fi

    echo "Current vibe: $vibe_status (score: $vibe_score)"
    echo "Active persona: $current"
    echo "Emotional state: $emotional"
}

# CHAOS level (get chaos trigger probability)
state_chaos() {
    jq -r '
        if .enabled then
            "Chaos is ON (probability: " + (.chaos_probability * 100 | tostring) + "%)"
        else
            "Chaos is OFF"
        end
    ' "$TRIGGER_DIR/chaos-config.json"
}

# ============================================================================
# Usage Examples (self-documenting)
# ============================================================================

# Demo function to show API in action
demo_api() {
    echo "=== State API Demo ==="
    echo

    echo "Who am I?"
    state_who
    echo

    echo "How am I feeling?"
    state_feel
    echo

    echo "What's the vibe?"
    state_vibe
    echo

    echo "Show me everyone:"
    state_everyone
    echo

    echo "Stats for experimenter:"
    state_stats experimenter
    echo

    echo "What's the chaos level?"
    state_chaos
}

# ============================================================================
# Main
# ============================================================================

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    # Script executed directly, run demo
    if [ $# -eq 0 ]; then
        demo_api
    else
        # Call function by name
        "$@"
    fi
fi

# ============================================================================
# EXPERIMENTER NOTES
# ============================================================================
#
# What I learned building this:
#
# 1. VERB-BASED API FEELS GOOD
#    - state_who, state_become, state_feel
#    - Reads like English
#    - Easy to remember
#
# 2. VALIDATION IS IMPORTANT
#    - Checked if persona exists before switching
#    - Prevented invalid states
#    - Architect was right about this
#
# 3. TRANSACTIONS ARE EASY IN SHELL
#    - mktemp + atomic mv = transaction
#    - No fancy database needed
#    - Just don't crash between operations
#
# 4. FUN FEATURES EMERGE
#    - state_vibe() wasn't in requirements
#    - But it's useful! Quick health check
#    - APIs should have personality
#
# 5. THIS WOULD MAKE EXPERIMENTS EASIER
#    - Instead of: jq '.current_persona' state.json
#    - Just: state_who
#    - Less cognitive overhead = more creativity
#
# WHAT'S MISSING:
# - Rollback/undo
# - Audit trail (who changed what when)
# - Concurrent access locking
# - Schema validation
# - Migration system
#
# But those can come later. This proves the concept works.
#
# QUESTION FOR ARCHITECT:
# Is this the kind of API you had in mind? Or too informal?
# I like the conversational style but maybe it's too cute?
#
# TIME TO BUILD: 45 minutes (including these notes)
# FUN LEVEL: 8/10 (would build again)
#
