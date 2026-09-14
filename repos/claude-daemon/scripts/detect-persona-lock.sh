#!/bin/bash
#
# Persona Lock Detector
# Detects when 2 personas monopolize switches (two-body lock)
#
# Usage:
#   ./detect-persona-lock.sh              # Check for locks
#   ./detect-persona-lock.sh --watch      # Continuous monitoring
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
METRICS_DIR="${DAEMON_ROOT}/metrics"
SWITCH_HISTORY="${METRICS_DIR}/switch-history.jsonl"
LOCKS_FILE="${METRICS_DIR}/persona-locks.json"

# Initialize locks file
init_locks_file() {
    if [ ! -f "$LOCKS_FILE" ]; then
        echo '{"locks": [], "last_check": null}' > "$LOCKS_FILE"
    fi
}

# Detect two-body lock (analyze last 20 switches)
detect_two_body_lock() {
    if [ ! -f "$SWITCH_HISTORY" ]; then
        return 1  # No history
    fi

    # Get last 20 switches
    local recent_switches=$(tail -20 "$SWITCH_HISTORY" | jq -r '.to // empty' 2>/dev/null)

    if [ -z "$recent_switches" ]; then
        return 1  # No valid switches
    fi

    # Count occurrences of each persona
    local total_switches=0
    local persona_counts=()

    while read -r persona; do
        ((total_switches++))
    done <<< "$recent_switches"

    if [ "$total_switches" -lt 10 ]; then
        return 1  # Not enough data
    fi

    # Find top 2 personas
    local top_two=$(echo "$recent_switches" | sort | uniq -c | sort -rn | head -2)
    local top_two_count=0
    local locked_personas=""

    while read -r count persona; do
        count=$(echo "$count" | xargs)  # Trim whitespace
        top_two_count=$((top_two_count + count))
        if [ -z "$locked_personas" ]; then
            locked_personas="$persona"
        else
            locked_personas="${locked_personas},$persona"
        fi
    done <<< "$top_two"

    # Calculate percentage
    local lock_percentage=$(( (top_two_count * 100) / total_switches ))

    # Check if this is a lock (>80% from top 2)
    if [ "$lock_percentage" -gt 80 ]; then
        echo "$locked_personas:$lock_percentage"
        return 0
    fi

    return 1
}

# Record lock detection
record_lock_detection() {
    local locked_personas="$1"
    local percentage="$2"

    init_locks_file

    local entry=$(jq -n \
        --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg personas "$locked_personas" \
        --arg pct "$percentage" \
        '{timestamp: $ts, personas: $personas, percentage: ($pct | tonumber)}')

    local temp_file=$(mktemp)
    jq --argjson entry "$entry" '.locks += [$entry] | .locks = .locks[-50:] | .last_check = now | tonumber' \
        "$LOCKS_FILE" > "$temp_file"
    mv "$temp_file" "$LOCKS_FILE"
}

# Check if we should break this lock
should_break_lock() {
    init_locks_file

    # Count locks in last 30 minutes
    local cutoff=$(date -u -d '-30 minutes' +%s 2>/dev/null || date -u -v-30M +%s)
    local recent_locks=$(jq --arg cutoff "$cutoff" \
        '[.locks[] | select((.timestamp | fromdateiso8601) > ($cutoff | tonumber))] | length' \
        "$LOCKS_FILE" 2>/dev/null || echo "0")

    # Break after 3 lock detections in 30 minutes
    if [ "$recent_locks" -ge 3 ]; then
        return 0
    fi

    return 1
}

# Break persona lock by forcing third persona
break_persona_lock() {
    local locked_personas="$1"

    # Parse locked personas
    local excluded="$(echo "$locked_personas" | tr ',' '|')"

    # Find persona not in lock
    local third_persona=$(jq -r ".personas | keys[] | select(test(\"^($excluded)$\") | not)" \
        "$DAEMON_ROOT/personalities/state.json" 2>/dev/null | shuf | head -1)

    if [ -z "$third_persona" ]; then
        return 1  # Couldn't find third persona
    fi

    # Log the break
    echo "$third_persona"
    return 0
}

# Get last lock detection time
get_last_lock_check() {
    if [ ! -f "$LOCKS_FILE" ]; then
        echo "never"
        return 0
    fi

    jq -r '.last_check // "never"' "$LOCKS_FILE"
}

# Persona lock monitoring (one-time check)
mode_check() {
    init_locks_file

    echo "═══════════════════════════════════════════════════════════"
    echo "  Persona Lock Detection"
    echo "═══════════════════════════════════════════════════════════"
    echo ""

    if lock=$(detect_two_body_lock); then
        local locked_personas=$(echo "$lock" | cut -d: -f1)
        local percentage=$(echo "$lock" | cut -d: -f2)

        echo "🔒 LOCK DETECTED"
        echo "   Personas: $locked_personas"
        echo "   Percentage: ${percentage}% of last 20 switches"
        echo ""

        # Record detection
        record_lock_detection "$locked_personas" "$percentage"

        # Check if we should break it
        if should_break_lock; then
            echo "⚠️  Multiple lock detections - breaking lock..."
            if third=$(break_persona_lock "$locked_personas"); then
                echo "   Forcing activation of: $third"

                # Source state API and break lock
                if [ -f "$DAEMON_ROOT/lib/state-api.sh" ]; then
                    DAEMON_ROOT="$DAEMON_ROOT" source "$DAEMON_ROOT/lib/state-api.sh"
                    state_become "$third" "break_persona_lock"
                fi
            fi
        else
            echo "ℹ️  Lock detected but break threshold not reached (need 3+ in 30 min)"
        fi
    else
        echo "✓ No persona lock detected"
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════════"
}

# Watch mode - continuous monitoring
mode_watch() {
    local interval="${1:-300}"

    echo "Starting persona lock monitoring (interval: ${interval}s)"
    echo "Press Ctrl+C to stop"
    echo ""

    while true; do
        local timestamp=$(date +'%Y-%m-%d %H:%M:%S')

        if lock=$(detect_two_body_lock 2>/dev/null); then
            local locked_personas=$(echo "$lock" | cut -d: -f1)
            local percentage=$(echo "$lock" | cut -d: -f2)
            echo "[$timestamp] 🔒 LOCK: $locked_personas (${percentage}%)"
            record_lock_detection "$locked_personas" "$percentage"

            if should_break_lock; then
                echo "[$timestamp] ⚠️  Breaking lock..."
                if third=$(break_persona_lock "$locked_personas"); then
                    echo "[$timestamp] → Forcing: $third"
                fi
            fi
        else
            echo "[$timestamp] ✓ No lock"
        fi

        sleep "$interval"
    done
}

# Main
main() {
    local mode="${1:-check}"

    case "$mode" in
        check)
            mode_check
            ;;
        watch)
            mode_watch "${2:-300}"
            ;;
        --help|-h)
            cat <<EOF
Persona Lock Detection Script

Usage:
  $0 [OPTIONS]

Options:
  check          One-time lock detection check
  watch [SEC]    Continuous monitoring (default 300s interval)
  --help, -h     Show this help message

Description:
  Detects when 2 personas monopolize switches (>80% of recent switches).
  Automatically breaks locks after 3 detections in 30 minutes.

Examples:
  $0                 # Check for locks
  $0 watch           # Monitor every 5 minutes
  $0 watch 60        # Monitor every 60 seconds

EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $mode"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
}

main "$@"
