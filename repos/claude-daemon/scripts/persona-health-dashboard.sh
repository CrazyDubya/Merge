#!/bin/bash
#
# Persona Health Dashboard
# Single-view health status for all personas
#
# Usage:
#   ./persona-health-dashboard.sh              # Text output
#   ./persona-health-dashboard.sh --json       # JSON output
#   ./persona-health-dashboard.sh --watch      # Watch mode
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
METRICS_DIR="${DAEMON_ROOT}/metrics"
STATE_FILE="${DAEMON_ROOT}/personalities/state.json"

# Source required libraries
if [ -f "$DAEMON_ROOT/lib/persona-health.sh" ]; then
    source "$DAEMON_ROOT/lib/persona-health.sh"
fi

if [ -f "$DAEMON_ROOT/lib/activation-floor.sh" ]; then
    source "$DAEMON_ROOT/lib/activation-floor.sh"
fi

# Get persona last active time
get_persona_time_since_active() {
    local persona="$1"

    if [ ! -f "$STATE_FILE" ]; then
        echo "never"
        return 0
    fi

    local last_active=$(jq -r ".personas.\"$persona\".last_active // \"1970-01-01T00:00:00Z\"" "$STATE_FILE" 2>/dev/null)

    if [ "$last_active" = "1970-01-01T00:00:00Z" ]; then
        echo "never"
        return 0
    fi

    local now=$(date -u +%s)
    local then=$(date -u -d "$last_active" +%s 2>/dev/null || date -u -j -f '%Y-%m-%dT%H:%M:%SZ' "$last_active" +%s 2>/dev/null || echo "$now")

    if [ "$then" -gt "$now" ]; then
        echo "now"
        return 0
    fi

    local diff=$((now - then))

    if [ "$diff" -lt 3600 ]; then
        echo "$((diff / 60))m ago"
    elif [ "$diff" -lt 86400 ]; then
        echo "$((diff / 3600))h ago"
    else
        echo "$((diff / 86400))d ago"
    fi
}

# Generate text dashboard
generate_text_dashboard() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════════╗"
    echo "║          🤖 Persona Health Dashboard                                  ║"
    echo "╚═══════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Generated: $(date -u +%Y-%m-%d\ %H:%M:%SZ\ UTC)"
    echo ""

    # Update health scores
    if declare -f update_health_scores >/dev/null 2>&1; then
        update_health_scores
    fi

    # Get health scores
    if [ -f "$METRICS_DIR/persona-health/health-scores.json" ]; then
        local health_scores=$(cat "$METRICS_DIR/persona-health/health-scores.json")
    else
        local health_scores='{}'
    fi

    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  Per-Persona Health (Last 24h)"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo ""

    printf "%-15s %8s %10s %6s %8s %12s %12s\n" \
        "Persona" "Health" "Status" "Tasks" "Success" "Avg Duration" "Last Active"
    echo "───────────────────────────────────────────────────────────────────────"

    for persona in architect optimizer auditor maintainer skeptic experimenter; do
        local health=$(echo "$health_scores" | jq -r ".\"$persona\" // 50" 2>/dev/null || echo "50")
        local status=""

        if [ "$health" -ge 90 ]; then
            status="✓ HEALTHY"
        elif [ "$health" -ge 75 ]; then
            status="⚠ DEGRADED"
        elif [ "$health" -ge 50 ]; then
            status="⚠ UNHEALTHY"
        else
            status="✗ CRITICAL"
        fi

        # Count tasks
        local task_count=0
        if [ -f "$METRICS_DIR/persona-health/${persona}-tasks.log" ]; then
            task_count=$(wc -l < "$METRICS_DIR/persona-health/${persona}-tasks.log" 2>/dev/null || echo "0")
        fi

        # Calculate success rate
        local success_rate="-"
        if [ "$task_count" -gt 0 ]; then
            local successful=$(grep -c '|0|' "$METRICS_DIR/persona-health/${persona}-tasks.log" 2>/dev/null || echo "0")
            success_rate="$((successful * 100 / task_count))%"
        fi

        # Average duration
        local avg_duration="-"
        if [ "$task_count" -gt 5 ]; then
            avg_duration=$(awk -F'|' '{sum+=$4; count++} END {if(count>0) print int(sum/count)"s"}' \
                "$METRICS_DIR/persona-health/${persona}-tasks.log" 2>/dev/null || echo "-")
        fi

        # Last active
        local last_active=$(get_persona_time_since_active "$persona")

        printf "%-15s %7d%% %10s %6s %8s %12s %12s\n" \
            "$persona" "$health" "$status" "$task_count" "$success_rate" "$avg_duration" "$last_active"
    done

    echo ""

    # Active cooldowns
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  Active Cooldowns"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo ""

    if [ -f "$METRICS_DIR/persona-cooldowns.json" ]; then
        local cooldown_count=0
        jq -r '.cooldowns | to_entries[] | select(.value != null) | "  \(.key): expires at \(.value)"' \
            "$METRICS_DIR/persona-cooldowns.json" 2>/dev/null | while read -r line; do
            echo "$line"
            ((cooldown_count++))
        done

        if [ "$cooldown_count" -eq 0 ]; then
            echo "  ✓ No active cooldowns"
        fi
    else
        echo "  ✓ No active cooldowns"
    fi

    echo ""

    # Activation floor violations
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  Activation Floor Violations (12-hour floor)"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo ""

    local floor_violations=0
    for persona in architect optimizer auditor maintainer skeptic experimenter; do
        local last_active=$(get_persona_last_active "$persona")
        local hours_since=$(calculate_hours_since "$last_active")

        if [ "$hours_since" -gt 12 ]; then
            local margin=$((hours_since - 12))
            echo "  ⚠️  $persona: $hours_since hours since last active (exceeded by $margin hours)"
            ((floor_violations++))
        fi
    done

    if [ "$floor_violations" -eq 0 ]; then
        echo "  ✓ All personas within activation floor"
    fi

    echo ""

    # Recommendations
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  Recommendations"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo ""

    local recommendations=0

    # Check for critical health
    for persona in architect optimizer auditor maintainer skeptic experimenter; do
        local health=$(echo "$health_scores" | jq -r ".\"$persona\" // 50" 2>/dev/null || echo "50")
        if [ "$health" -lt 50 ]; then
            echo "  $((++recommendations)). $persona is in critical health ($health%) - may need investigation"
        fi
    done

    # Check for floor violations
    for persona in architect optimizer auditor maintainer skeptic experimenter; do
        local last_active=$(get_persona_last_active "$persona" 2>/dev/null)
        local hours_since=$(calculate_hours_since "$last_active" 2>/dev/null || echo "0")
        if [ "$hours_since" -gt 12 ] && [ "$hours_since" -lt 14 ]; then
            echo "  $((++recommendations)). $persona approaching floor violation ($(14 - hours_since) hours until critical)"
        fi
    done

    if [ "$recommendations" -eq 0 ]; then
        echo "  ✓ System is healthy - no immediate recommendations"
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════════════════════"
}

# Generate JSON dashboard
generate_json_dashboard() {
    update_health_scores 2>/dev/null || true

    local health_scores='{}'
    if [ -f "$METRICS_DIR/persona-health/health-scores.json" ]; then
        health_scores=$(cat "$METRICS_DIR/persona-health/health-scores.json")
    fi

    jq -n \
        --argjson health "$health_scores" \
        --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{
            timestamp: $ts,
            health_scores: $health,
            personas: {}
        }' > /tmp/persona-dashboard.json

    cat /tmp/persona-dashboard.json
}

# Watch mode
watch_mode() {
    local interval="${1:-60}"

    echo "Starting persona health watch mode (refreshing every ${interval}s)"
    echo "Press Ctrl+C to stop"
    echo ""

    while true; do
        clear
        generate_text_dashboard
        echo ""
        echo "Refreshing in ${interval}s... (Ctrl+C to exit)"
        sleep "$interval"
    done
}

# Main
main() {
    local mode="${1:-text}"

    case "$mode" in
        --json)
            generate_json_dashboard
            ;;
        --watch)
            watch_mode "${2:-60}"
            ;;
        --text|*)
            generate_text_dashboard
            ;;
    esac
}

main "$@"
