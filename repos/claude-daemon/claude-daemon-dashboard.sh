#!/bin/bash
#
# Claude Daemon TUI Dashboard
# Terminal-based monitoring for daemon status
#
# Usage: ./claude-daemon-dashboard.sh [--watch]
#
# Displays:
# - Current persona and emotional state
# - Pending tasks and inbox status
# - Recent activity (last 10 cycles)
#
# Security: Local file access only, no web exposure
# Performance: Tail-based reading for large files, single-pass parsing
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
STATE_FILE="${DAEMON_ROOT}/personalities/state.json"
TIMELINE_FILE="${DAEMON_ROOT}/memory/persona-timeline.jsonl"
TASK_QUEUE="${DAEMON_ROOT}/tasks/queue.md"
INBOX_DAEMON="${DAEMON_ROOT}/inbox/daemon/unread"
INBOX_HUMAN="${DAEMON_ROOT}/inbox/human/unread"
INTER_PERSONA_INBOX="${DAEMON_ROOT}/memory/inter-persona-inbox/unread"
SETTINGS_FILE="${DAEMON_ROOT}/daemon-settings.json"
EXPERIMENTS_DIR="${DAEMON_ROOT}/experiments"

# Color codes for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# Error handling: Print error message and continue with degraded display
error_msg() {
    echo -e "${RED}⚠ $1${RESET}" >&2
}

# Safe file read: Returns content or error placeholder
safe_read() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "FILE_NOT_FOUND"
        return 1
    elif [ ! -r "$file" ]; then
        echo "PERMISSION_DENIED"
        return 1
    else
        cat "$file" 2>/dev/null || echo "READ_ERROR"
    fi
}

# Safe jq parse: Returns value or fallback
safe_jq() {
    local file="$1"
    local query="$2"
    local fallback="${3:-unknown}"

    if [ ! -f "$file" ]; then
        echo "$fallback"
        return
    fi

    jq -r "$query" "$file" 2>/dev/null || echo "$fallback"
}

# Count files in directory safely
safe_count() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        echo "0"
        return
    fi

    local count=$(ls -1 "$dir"/*.md 2>/dev/null | wc -l)
    echo "$count"
}

# Parse task queue for pending tasks
count_pending_tasks() {
    if [ ! -f "$TASK_QUEUE" ]; then
        echo "0"
        return
    fi

    # Count lines starting with "- [ ]" or "- [~]" (pending or in-progress)
    # tr -d removes newlines to prevent integer expression errors
    local count=$(grep -c '^\- \[[ ~]\]' "$TASK_QUEUE" 2>/dev/null || echo "0")
    echo "$count" | tr -d '\n'
}

# Get recent activity from timeline (last N entries)
get_recent_activity() {
    local limit="${1:-10}"

    if [ ! -f "$TIMELINE_FILE" ]; then
        echo "No timeline data available"
        return
    fi

    # Tail to avoid reading huge file, filter COMPLETE JSON lines only, then parse with jq
    tail -n "$((limit * 3))" "$TIMELINE_FILE" 2>/dev/null | \
        grep -E '^\{"timestamp".*\}$' | \
        tail -n "$limit" | \
        jq -r '
            # Use event_type (new format) or event (old format), with description if available
            (.event_type // .event // "unknown") as $event |
            (.description // "") as $desc |
            (.timestamp // "?") as $ts |
            (.persona // "?") as $persona |
            "[\($ts | split("T")[1] | split(":")[0:2] | join(":"))] \($persona) - \($event)" +
            (if $desc != "" and $desc != "null" then
                ": \($desc | tostring | if length > 65 then .[0:62] + "..." else . end)"
            else
                ""
            end)
        ' 2>/dev/null || echo "No recent activity"
}

# Get emotional state (simplified from daemon.sh logic)
get_emotional_state() {
    if [ ! -f "$TIMELINE_FILE" ]; then
        echo "unknown"
        return
    fi

    # Check recent events for positive/negative indicators
    # Filter only valid JSON lines, then count event types
    local recent=$(tail -n 20 "$TIMELINE_FILE" 2>/dev/null | grep -E '^\{.*\}$' || echo "")

    if [ -z "$recent" ]; then
        echo "neutral"
        return
    fi

    # Count different event types, ensuring tr -d '\n' for clean integers
    local completed=$(echo "$recent" | grep -c '"event":"task_complete"' 2>/dev/null | tr -d '\n' || echo "0")
    local failed=$(echo "$recent" | grep -c '"event":"task_failed"' 2>/dev/null | tr -d '\n' || echo "0")
    local reflections=$(echo "$recent" | grep -c '"event":"reflection_complete"' 2>/dev/null | tr -d '\n' || echo "0")

    # Ensure variables are proper integers
    completed=${completed:-0}
    failed=${failed:-0}
    reflections=${reflections:-0}

    # Simple heuristic: more completions = positive, more failures = negative
    if [ "$completed" -ge 3 ]; then
        echo "positive (productive)"
    elif [ "$failed" -ge 2 ]; then
        echo "negative (frustrated)"
    elif [ "$reflections" -ge 3 ]; then
        echo "contemplative"
    else
        echo "neutral"
    fi
}

# Calculate time since last activity
time_since_last_activity() {
    if [ ! -f "$TIMELINE_FILE" ]; then
        echo "unknown"
        return
    fi

    # Get last valid timestamp from timeline (filter for COMPLETE valid JSON)
    local last_timestamp=$(tail -n 50 "$TIMELINE_FILE" 2>/dev/null | grep -E '^\{"timestamp".*\}$' | tail -n 1 | jq -r '.timestamp' 2>/dev/null || echo "")

    if [ -z "$last_timestamp" ] || [ "$last_timestamp" == "null" ]; then
        echo "unknown"
        return
    fi

    # Convert to seconds since epoch
    local last_epoch=$(date -d "$last_timestamp" +%s 2>/dev/null || echo "0")
    local now_epoch=$(date +%s)
    local diff=$((now_epoch - last_epoch))

    # Format as human-readable
    if [ "$diff" -lt 60 ]; then
        echo "${diff}s ago"
    elif [ "$diff" -lt 3600 ]; then
        echo "$((diff / 60))m ago"
    elif [ "$diff" -lt 86400 ]; then
        echo "$((diff / 3600))h ago"
    else
        echo "$((diff / 86400))d ago"
    fi
}

# Get current task being worked on
get_current_task() {
    if [ ! -f "$TASK_QUEUE" ]; then
        echo "No task queue"
        return
    fi

    # Look for first in-progress task [~]
    local current=$(grep '^\- \[~\]' "$TASK_QUEUE" 2>/dev/null | head -n 1 | sed 's/^- \[~\] //' || echo "")

    if [ -n "$current" ]; then
        echo "$current" | cut -c 1-80
    else
        # No in-progress, show next pending task
        local next=$(grep '^\- \[ \]' "$TASK_QUEUE" 2>/dev/null | head -n 1 | sed 's/^- \[ \] //' || echo "")
        if [ -n "$next" ]; then
            echo "(Next) $next" | cut -c 1-80
        else
            echo "No pending tasks"
        fi
    fi
}

# Get active experiments count
get_active_experiments() {
    if [ ! -d "$EXPERIMENTS_DIR" ]; then
        echo "0"
        return
    fi

    # Count .sh files in experiments directory
    ls -1 "$EXPERIMENTS_DIR"/*.sh 2>/dev/null | wc -l | tr -d ' \n'
}

# Check for system warnings/issues
get_system_warnings() {
    local warnings=()

    # Check for large inbox backlog
    local human_inbox=$(safe_count "$INBOX_HUMAN")
    if [ "$human_inbox" -ge 10 ]; then
        warnings+=("⚠️  Human inbox has $human_inbox unread messages")
    fi

    # Check for stale inter-persona messages
    local inter_persona=$(safe_count "$INTER_PERSONA_INBOX")
    if [ "$inter_persona" -ge 3 ]; then
        warnings+=("⚠️  Inter-persona inbox has $inter_persona unread messages")
    fi

    # Check for reflection gates being hit (look in recent timeline)
    if [ -f "$TIMELINE_FILE" ]; then
        local recent_deferrals=$(tail -n 50 "$TIMELINE_FILE" 2>/dev/null | grep -c '"reflection_deferred"' 2>/dev/null | tr -d '\n' || echo "0")
        recent_deferrals=${recent_deferrals:-0}  # Ensure it's a valid integer
        if [ "$recent_deferrals" -ge 3 ]; then
            warnings+=("ℹ️  Reflection gates active (${recent_deferrals} recent deferrals)")
        fi
    fi

    # Check timeline file size
    if [ -f "$TIMELINE_FILE" ]; then
        local timeline_size=$(wc -l < "$TIMELINE_FILE" 2>/dev/null || echo "0")
        if [ "$timeline_size" -ge 10000 ]; then
            warnings+=("⚠️  Timeline file large (${timeline_size} entries - consider rotation)")
        fi
    fi

    # Output warnings
    if [ ${#warnings[@]} -gt 0 ]; then
        printf '%s\n' "${warnings[@]}"
    else
        echo "✓ No system warnings"
    fi
}

# Main dashboard display
display_dashboard() {
    clear

    # Header
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${BOLD}${CYAN}          CLAUDE DAEMON STATUS DASHBOARD${RESET}"
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo ""

    # Current State Section
    echo -e "${BOLD}${YELLOW}━━━ CURRENT STATE ━━━${RESET}"
    echo ""

    # Persona - Consolidated single jq call (Option D pattern)
    # Reads: current_persona, display_name, last_switch, switch_reason in ONE jq invocation
    local current_persona display_name last_switch switch_reason
    IFS=$'\t' read -r current_persona display_name last_switch switch_reason < <(
        jq -r '
            .current_persona as $persona |
            [
                $persona,
                (.personas[$persona].display_name // "Unknown Persona"),
                (.last_switch_time // "unknown"),
                (.switch_reason // "unknown")
            ] | @tsv
        ' "$STATE_FILE" 2>/dev/null || echo -e "unknown\tUnknown Persona\tunknown\tunknown"
    )

    echo -e "${BOLD}Persona:${RESET}      ${MAGENTA}${display_name}${RESET} (${current_persona})"
    echo -e "${BOLD}Active since:${RESET} ${last_switch}"
    echo -e "${BOLD}Reason:${RESET}       ${switch_reason}"
    echo ""

    # Emotional state
    local mood=$(get_emotional_state)
    local mood_color="${GREEN}"
    if [[ "$mood" == *"negative"* ]] || [[ "$mood" == *"frustrated"* ]]; then
        mood_color="${RED}"
    elif [[ "$mood" == *"neutral"* ]]; then
        mood_color="${YELLOW}"
    fi
    echo -e "${BOLD}Mood:${RESET}         ${mood_color}${mood}${RESET}"
    echo ""

    # Last activity
    local last_activity=$(time_since_last_activity)
    echo -e "${BOLD}Last wake:${RESET}    ${last_activity}"
    echo ""

    # Workload Section
    echo -e "${BOLD}${YELLOW}━━━ WORKLOAD ━━━${RESET}"
    echo ""

    local pending_tasks=$(count_pending_tasks)
    local inbox_daemon=$(safe_count "$INBOX_DAEMON")
    local inbox_human=$(safe_count "$INBOX_HUMAN")

    local task_color="${GREEN}"
    if [ "$pending_tasks" -ge 5 ]; then
        task_color="${RED}"
    elif [ "$pending_tasks" -ge 3 ]; then
        task_color="${YELLOW}"
    fi

    echo -e "${BOLD}Pending tasks:${RESET}    ${task_color}${pending_tasks}${RESET}"
    echo -e "${BOLD}Daemon inbox:${RESET}     ${inbox_daemon} unread"

    # Color human inbox if large
    local human_color="${RESET}"
    if [ "$inbox_human" -ge 10 ]; then
        human_color="${RED}"
    elif [ "$inbox_human" -ge 5 ]; then
        human_color="${YELLOW}"
    fi
    echo -e "${BOLD}Human inbox:${RESET}      ${human_color}${inbox_human}${RESET} unread"

    # Inter-persona inbox
    local inter_persona=$(safe_count "$INTER_PERSONA_INBOX")
    if [ "$inter_persona" -gt 0 ]; then
        echo -e "${BOLD}Inter-persona:${RESET}    ${inter_persona} unread"
    fi
    echo ""

    # Current/Next Task
    echo -e "${BOLD}${YELLOW}━━━ CURRENT FOCUS ━━━${RESET}"
    echo ""
    local current_task=$(get_current_task)
    echo -e "${CYAN}${current_task}${RESET}"
    echo ""

    # Active Experiments
    local exp_count=$(get_active_experiments)
    if [ "$exp_count" -gt 0 ]; then
        echo -e "${BOLD}Active experiments:${RESET} ${exp_count} in ${EXPERIMENTS_DIR}"
        echo ""
    fi

    # System Warnings
    echo -e "${BOLD}${YELLOW}━━━ SYSTEM HEALTH ━━━${RESET}"
    echo ""
    get_system_warnings
    echo ""

    # Persona Statistics
    echo -e "${BOLD}${YELLOW}━━━ PERSONA STATISTICS ━━━${RESET}"
    echo ""

    if [ -f "$STATE_FILE" ]; then
        printf "${BOLD}%-15s %12s %12s %12s${RESET}\n" "PERSONA" "ACTIVATIONS" "COMPLETED" "FAILED"
        echo "────────────────────────────────────────────────────────"

        # Consolidated single jq call (Option D pattern) - N×3 calls → 1 call
        # Extracts all persona stats in one pass, sorted by persona name
        jq -r '
            .current_persona as $current |
            .personas | to_entries | sort_by(.key) | .[] |
            [
                .key,
                (.value.total_activations // 0),
                (.value.tasks_completed // 0),
                (.value.tasks_failed // 0),
                (if .key == $current then "CURRENT" else "NORMAL" end)
            ] | @tsv
        ' "$STATE_FILE" 2>/dev/null | while IFS=$'\t' read -r persona activations completed failed highlight; do
            # Highlight current persona
            if [ "$highlight" == "CURRENT" ]; then
                printf "${MAGENTA}%-15s %12s %12s %12s${RESET}\n" "$persona" "$activations" "$completed" "$failed"
            else
                printf "%-15s %12s %12s %12s\n" "$persona" "$activations" "$completed" "$failed"
            fi
        done
        echo ""
    else
        error_msg "State file not found"
        echo ""
    fi

    # Recent Activity Section
    echo -e "${BOLD}${YELLOW}━━━ RECENT ACTIVITY (last 10 events) ━━━${RESET}"
    echo ""

    if [ -f "$TIMELINE_FILE" ]; then
        get_recent_activity 10
    else
        error_msg "Timeline file not found"
    fi

    echo ""
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo ""
    echo -e "${BOLD}Dashboard updated:${RESET} $(date '+%Y-%m-%d %H:%M:%S')"

    # Watch mode instructions
    if [ "${WATCH_MODE:-false}" == "true" ]; then
        echo -e "${BOLD}Watch mode active${RESET} - Refreshing every 10 seconds (Ctrl-C to exit)"
    else
        echo -e "${BOLD}Tip:${RESET} Run with ${GREEN}--watch${RESET} flag for auto-refresh mode"
    fi

    echo ""
}

# Watch mode: Continuous refresh
watch_mode() {
    while true; do
        display_dashboard
        sleep 10
    done
}

# Main execution
main() {
    # Check if watch mode requested
    if [ "${1:-}" == "--watch" ]; then
        WATCH_MODE=true
        watch_mode
    else
        WATCH_MODE=false
        display_dashboard
    fi
}

# Trap Ctrl-C for clean exit
trap 'echo ""; echo "Dashboard exited."; exit 0' INT

main "$@"
