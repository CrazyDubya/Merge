#!/bin/bash
#
# Health-Aware Persona Selection System
# Integrates health scores into persona decision engine
#
# Modified decision priority with health awareness:
# 1. Emergency Layer (floor violations, locks, critical alerts)
# 2. Cooldown Layer (filter unhealthy personas)
# 3. Chaos Layer (10% random from healthy pool)
# 4. Emotional/Circadian/Task Layers (existing logic, filtered to healthy)
#
# Usage:
#   source "${DAEMON_ROOT}/lib/persona-selection.sh"
#   select_persona_with_health
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
STATE_DIR="${DAEMON_ROOT}/personalities"
STATE_FILE="${STATE_DIR}/state.json"
METRICS_DIR="${DAEMON_ROOT}/metrics"

# Get all personas
get_all_personas() {
    echo "architect optimizer auditor maintainer skeptic experimenter"
}

# Get eligible personas (not on cooldown, health above threshold)
get_eligible_personas() {
    local min_health="${1:-50}"
    local eligible=()

    for persona in $(get_all_personas); do
        # Skip if on cooldown
        if declare -f is_persona_on_cooldown >/dev/null 2>&1; then
            if is_persona_on_cooldown "$persona"; then
                continue
            fi
        fi

        # Check health score
        if declare -f calculate_persona_health >/dev/null 2>&1; then
            local health=$(calculate_persona_health "$persona")
            if [ "$health" -ge "$min_health" ]; then
                eligible+=("$persona")
            fi
        else
            # If health function not available, consider all eligible
            eligible+=("$persona")
        fi
    done

    echo "${eligible[@]}"
}

# Get least unhealthy persona (fallback when all on cooldown)
get_least_unhealthy_persona() {
    local least_unhealthy=""
    local highest_health=0

    for persona in $(get_all_personas); do
        if declare -f calculate_persona_health >/dev/null 2>&1; then
            local health=$(calculate_persona_health "$persona")
            if [ "$health" -gt "$highest_health" ]; then
                highest_health="$health"
                least_unhealthy="$persona"
            fi
        fi
    done

    echo "$least_unhealthy"
}

# Check if emergency activation needed (floor violations or critical alerts)
emergency_activation_needed() {
    # Check activation floor violations
    if declare -f check_activation_floor >/dev/null 2>&1; then
        if check_activation_floor 12 >/dev/null 2>&1; then
            # Floor violation - force activation
            if persona=$(enforce_activation_floor 12 2>/dev/null); then
                echo "$persona"
                return 0
            fi
        fi
    fi

    # Check persona lock
    if declare -f detect_two_body_lock >/dev/null 2>&1; then
        if lock=$(detect_two_body_lock 2>/dev/null); then
            local locked_personas=$(echo "$lock" | cut -d: -f1)
            if should_break=$(break_persona_lock "$locked_personas" 2>/dev/null); then
                echo "$should_break"
                return 0
            fi
        fi
    fi

    return 1
}

# Select best persona using health-aware priority system
select_persona_with_health() {
    local current_persona=$(state_who 2>/dev/null || echo "unknown")

    # Layer 1: Emergency layer (overrides everything)
    if emergency=$(emergency_activation_needed 2>/dev/null); then
        if [ -n "$emergency" ] && [ "$emergency" != "$current_persona" ]; then
            echo "$emergency"
            return 0
        fi
    fi

    # Layer 2: Get eligible personas (health > 50%)
    local eligible=$(get_eligible_personas 50)

    if [ -z "$eligible" ]; then
        # All personas on cooldown - force least unhealthy and alert
        local fallback=$(get_least_unhealthy_persona)

        if declare -f send_enriched_alert >/dev/null 2>&1; then
            send_enriched_alert "All Personas Unhealthy" "critical" \
                "All personas on cooldown. Forcing activation of $fallback."
        fi

        if [ -n "$fallback" ]; then
            echo "$fallback"
            return 0
        fi

        return 1
    fi

    # Layer 3: Chaos layer (10% random from eligible personas with health > 75%)
    local rand=$((RANDOM % 100))
    if [ "$rand" -lt 10 ]; then
        # Select random from eligible personas with high health
        local high_health=()
        for persona in $eligible; do
            if declare -f calculate_persona_health >/dev/null 2>&1; then
                local health=$(calculate_persona_health "$persona")
                if [ "$health" -ge 75 ]; then
                    high_health+=("$persona")
                fi
            fi
        done

        if [ ${#high_health[@]} -gt 0 ]; then
            local random_persona="${high_health[$((RANDOM % ${#high_health[@]}))]}"
            if [ "$random_persona" != "$current_persona" ]; then
                echo "$random_persona"
                return 0
            fi
        fi
    fi

    # Layer 4-6: Use existing decision logic but filtered to eligible personas only
    # This should be implemented in daemon.sh where the full decision logic lives
    # For now, return current persona (no change needed)
    echo "$current_persona"
    return 0
}

# Switch persona with health context logging
switch_persona_with_health() {
    local new_persona="$1"
    local current_persona=$(state_who 2>/dev/null || echo "unknown")

    if [ "$new_persona" = "$current_persona" ]; then
        return 0  # No switch needed
    fi

    # Get health info for logging
    local new_health=""
    local old_health=""

    if declare -f calculate_persona_health >/dev/null 2>&1; then
        new_health=$(calculate_persona_health "$new_persona")
        old_health=$(calculate_persona_health "$current_persona")
    fi

    # Perform switch using State API
    if declare -f state_become >/dev/null 2>&1; then
        local reason="health_aware_selection"
        state_become "$new_persona" "$reason"
    fi

    # Log health context
    if declare -f log >/dev/null 2>&1; then
        if [ -n "$new_health" ] && [ -n "$old_health" ]; then
            log "INFO" "Persona switch: $current_persona ($old_health%) → $new_persona ($new_health%)"
        else
            log "INFO" "Persona switch: $current_persona → $new_persona"
        fi
    fi

    return 0
}

# Check if persona should be excluded from selection
is_persona_excluded() {
    local persona="$1"

    # Exclude if on cooldown
    if declare -f is_persona_on_cooldown >/dev/null 2>&1; then
        if is_persona_on_cooldown "$persona"; then
            return 0
        fi
    fi

    # Exclude if health too low and has consecutive failures
    if declare -f get_persona_failure_streak >/dev/null 2>&1; then
        local streak=$(get_persona_failure_streak "$persona" 2>/dev/null || echo "0")
        if [ "$streak" -gt 3 ]; then
            return 0
        fi
    fi

    return 1
}

# Get persona health summary for decision logging
get_persona_health_summary() {
    local persona="$1"

    local health=""
    local status=""
    local cooldown=""

    if declare -f calculate_persona_health >/dev/null 2>&1; then
        health=$(calculate_persona_health "$persona")
    fi

    if declare -f get_persona_health_status >/dev/null 2>&1; then
        status=$(get_persona_health_status "$persona")
    fi

    if declare -f is_persona_on_cooldown >/dev/null 2>&1; then
        if is_persona_on_cooldown "$persona"; then
            cooldown="yes"
        fi
    fi

    echo "health=$health status=$status cooldown=$cooldown"
}

# Export functions
export -f get_all_personas
export -f get_eligible_personas
export -f get_least_unhealthy_persona
export -f emergency_activation_needed
export -f select_persona_with_health
export -f switch_persona_with_health
export -f is_persona_excluded
export -f get_persona_health_summary
