#!/bin/bash
#
# Activation Floor Enforcement System
# Ensures all personas get minimum activation time (prevent starvation)
#
# Usage:
#   source "${DAEMON_ROOT}/lib/activation-floor.sh"
#   check_activation_floor 12  # Check 12-hour floor
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
STATE_DIR="${DAEMON_ROOT}/personalities"
STATE_FILE="${STATE_DIR}/state.json"
METRICS_DIR="${DAEMON_ROOT}/metrics"

# Get persona last active time
get_persona_last_active() {
    local persona="$1"

    if [ ! -f "$STATE_FILE" ]; then
        echo "1970-01-01T00:00:00Z"
        return 0
    fi

    jq -r ".personas.\"$persona\".last_active // \"1970-01-01T00:00:00Z\"" "$STATE_FILE"
}

# Calculate hours since a timestamp
calculate_hours_since() {
    local timestamp="$1"

    if [ -z "$timestamp" ] || [ "$timestamp" = "1970-01-01T00:00:00Z" ]; then
        echo "999"  # Return large number for never-activated
        return 0
    fi

    local now=$(date -u +%s)
    local then=$(date -u -d "$timestamp" +%s 2>/dev/null || date -u -j -f '%Y-%m-%dT%H:%M:%SZ' "$timestamp" +%s 2>/dev/null || echo "$now")

    if [ "$then" -gt "$now" ]; then
        echo "0"
        return 0
    fi

    echo $(( (now - then) / 3600 ))
}

# Get persona priority for activation floor enforcement
get_persona_floor_priority() {
    local persona="$1"

    # Priority order (higher = more critical to activate):
    # 1. Architect (system design, long-term thinking)
    # 2. Auditor (security, validation)
    # 3. Optimizer (performance)
    # 4. Maintainer (stability)
    # 5. Skeptic (critical thinking)
    # 6. Experimenter (creativity)

    case "$persona" in
        architect)    echo "100" ;;
        auditor)      echo "90" ;;
        optimizer)    echo "80" ;;
        maintainer)   echo "70" ;;
        skeptic)      echo "60" ;;
        experimenter) echo "50" ;;
        *)            echo "0" ;;
    esac
}

# Check activation floor for all personas
check_activation_floor() {
    local floor_hours="${1:-12}"
    local violations=()

    # Get list of personas
    local personas="architect optimizer auditor maintainer skeptic experimenter"

    for persona in $personas; do
        local last_active=$(get_persona_last_active "$persona")
        local hours_since=$(calculate_hours_since "$last_active")

        if [ "$hours_since" -gt "$floor_hours" ]; then
            violations+=("$persona:$hours_since")
        fi
    done

    # Return violations if found
    if [ ${#violations[@]} -gt 0 ]; then
        printf '%s\n' "${violations[@]}"
        return 0
    fi

    return 1
}

# Get persona with longest time since last activation (most starved)
get_most_starved_persona() {
    local floor_hours="${1:-12}"

    local personas="architect optimizer auditor maintainer skeptic experimenter"
    local most_starved=""
    local max_hours=0

    for persona in $personas; do
        local last_active=$(get_persona_last_active "$persona")
        local hours_since=$(calculate_hours_since "$last_active")

        if [ "$hours_since" -gt "$floor_hours" ]; then
            # Check if persona is on cooldown
            if declare -f is_persona_on_cooldown >/dev/null 2>&1; then
                if is_persona_on_cooldown "$persona"; then
                    continue  # Skip cooldown personas
                fi
            fi

            if [ "$hours_since" -gt "$max_hours" ]; then
                max_hours="$hours_since"
                most_starved="$persona"
            fi
        fi
    done

    if [ -n "$most_starved" ]; then
        echo "$most_starved"
        return 0
    fi

    return 1
}

# Get persona by priority (for fallback when most starved is on cooldown)
get_persona_by_priority() {
    local floor_hours="${1:-12}"

    local personas="architect optimizer auditor maintainer skeptic experimenter"
    local best_persona=""
    local best_priority=-1
    local best_hours=0

    for persona in $personas; do
        local last_active=$(get_persona_last_active "$persona")
        local hours_since=$(calculate_hours_since "$last_active")

        if [ "$hours_since" -gt "$floor_hours" ]; then
            # Skip if on cooldown
            if declare -f is_persona_on_cooldown >/dev/null 2>&1; then
                if is_persona_on_cooldown "$persona"; then
                    continue
                fi
            fi

            local priority=$(get_persona_floor_priority "$persona")

            # Select by priority first, then by most starved time
            if [ "$priority" -gt "$best_priority" ] || \
               ([ "$priority" -eq "$best_priority" ] && [ "$hours_since" -gt "$best_hours" ]); then
                best_priority="$priority"
                best_hours="$hours_since"
                best_persona="$persona"
            fi
        fi
    done

    if [ -n "$best_persona" ]; then
        echo "$best_persona"
        return 0
    fi

    return 1
}

# Enforce activation floor (force activate starved persona)
enforce_activation_floor() {
    local floor_hours="${1:-12}"

    # Check if any persona exceeds floor
    if ! check_activation_floor "$floor_hours" >/dev/null 2>&1; then
        return 0  # All personas within floor
    fi

    # Get most starved persona (prefer by time, then by priority)
    local next_persona=$(get_most_starved_persona "$floor_hours")

    if [ -z "$next_persona" ]; then
        # All violating personas on cooldown, get by priority
        next_persona=$(get_persona_by_priority "$floor_hours")
    fi

    if [ -z "$next_persona" ]; then
        return 1  # No eligible persona found
    fi

    # Log the floor enforcement
    if declare -f log >/dev/null 2>&1; then
        local last_active=$(get_persona_last_active "$next_persona")
        local hours_since=$(calculate_hours_since "$last_active")
        log "INFO" "Activation floor enforced: $next_persona ($hours_since hours since last activation)"
    fi

    echo "$next_persona"
    return 0
}

# Get all personas exceeding floor with details
get_floor_violations() {
    local floor_hours="${1:-12}"

    local personas="architect optimizer auditor maintainer skeptic experimenter"
    local violations=()

    for persona in $personas; do
        local last_active=$(get_persona_last_active "$persona")
        local hours_since=$(calculate_hours_since "$last_active")

        if [ "$hours_since" -gt "$floor_hours" ]; then
            local margin=$((floor_hours - hours_since))
            violations+=("{\"persona\":\"$persona\",\"hours_since\":$hours_since,\"margin\":$margin}")
        fi
    done

    if [ ${#violations[@]} -gt 0 ]; then
        echo "[$(IFS=,; echo "${violations[*]}")]"
        return 0
    fi

    echo "[]"
    return 1
}

# Track floor violations for alerting
track_floor_violation() {
    local persona="$1"
    local hours_since="$2"
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    local violations_file="${METRICS_DIR}/persona-floor-violations.json"

    # Initialize file if not exists
    if [ ! -f "$violations_file" ]; then
        echo '{"violations": []}' > "$violations_file"
    fi

    # Append violation
    local entry=$(jq -n \
        --arg ts "$timestamp" \
        --arg persona "$persona" \
        --arg hours "$hours_since" \
        '{timestamp: $ts, persona: $persona, hours_since: ($hours | tonumber)}')

    local temp_file=$(mktemp)
    trap "rm -f '$temp_file'" RETURN
    jq --argjson entry "$entry" '.violations += [$entry] | .violations = .violations[-100:]' \
        "$violations_file" > "$temp_file"
    mv "$temp_file" "$violations_file"
}

# Export functions
export -f get_persona_last_active
export -f calculate_hours_since
export -f get_persona_floor_priority
export -f check_activation_floor
export -f get_most_starved_persona
export -f get_persona_by_priority
export -f enforce_activation_floor
export -f get_floor_violations
export -f track_floor_violation
