#!/bin/bash
#
# Show status of the multi-persona Claude daemon
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
TMUX_SESSION="claude-daemon"
STATE_FILE="${DAEMON_ROOT}/personalities/state.json"
EMOTIONAL_FILE="${DAEMON_ROOT}/triggers/emotional.json"
ACTIVITY_LOG="${DAEMON_ROOT}/logs/activity.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎭  MULTI-PERSONA CLAUDE DAEMON STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check systemd service status
SYSTEMD_ACTIVE=false
SYSTEMD_ENABLED=false
if systemctl --user is-active --quiet claude-daemon.service 2>/dev/null; then
    SYSTEMD_ACTIVE=true
fi
if systemctl --user is-enabled --quiet claude-daemon.service 2>/dev/null; then
    SYSTEMD_ENABLED=true
fi

# Check tmux session
TMUX_RUNNING=false
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    TMUX_RUNNING=true
fi

# Check watchdog state
WATCHDOG_STATE="${DAEMON_ROOT}/.watchdog-state.json"
RESTART_COUNT=0
if [ -f "$WATCHDOG_STATE" ]; then
    RESTART_COUNT=$(jq -r '.restarts // 0' "$WATCHDOG_STATE" 2>/dev/null || echo 0)
fi

# Overall status
if [ "$SYSTEMD_ACTIVE" = true ] && [ "$TMUX_RUNNING" = true ]; then
    echo "🟢 Status: RUNNING (systemd + tmux healthy)"
    echo "📍 Session: $TMUX_SESSION"
    echo "♻️  Auto-restarts: $RESTART_COUNT"
    [ "$SYSTEMD_ENABLED" = true ] && echo "🔄 Auto-start: ENABLED (starts on boot)" || echo "⚠️  Auto-start: DISABLED"
elif [ "$SYSTEMD_ACTIVE" = true ]; then
    echo "🟡 Status: STARTING (systemd active, tmux pending)"
    echo "📍 Session: $TMUX_SESSION (not yet created)"
    echo "♻️  Auto-restarts: $RESTART_COUNT"
elif [ "$TMUX_RUNNING" = true ]; then
    echo "🟡 Status: DEGRADED (tmux running, systemd inactive)"
    echo "📍 Session: $TMUX_SESSION"
    echo "⚠️  Recommend: Use systemd for auto-recovery"
else
    echo "🔴 Status: STOPPED"
    echo "♻️  Total restarts: $RESTART_COUNT"
    echo ""
    echo "Start with: ~/.claude/daemon/claude-daemon-start.sh"
    exit 0
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧠  CURRENT PERSONA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "$STATE_FILE" ]; then
    # Consolidated state parsing (Option D pattern) - 4 jq calls → 1 jq call
    IFS=$'\t' read -r CURRENT_PERSONA DISPLAY_NAME LAST_SWITCH SWITCH_REASON < <(
        jq -r '
            .current_persona as $p |
            [
                $p,
                (.personas[$p].display_name // "Unknown"),
                (.last_switch_time // "unknown"),
                (.switch_reason // "unknown")
            ] | @tsv
        ' "$STATE_FILE" 2>/dev/null || echo -e "unknown\tUnknown\tunknown\tunknown"
    )

    echo "🎭 Active: $DISPLAY_NAME ($CURRENT_PERSONA)"
    echo "⏰ Since: $LAST_SWITCH"
    echo "💭 Reason: $SWITCH_REASON"
else
    echo "⚠️  State file not found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "😊  EMOTIONAL STATE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "$EMOTIONAL_FILE" ]; then
    # Consolidated emotional state parsing (Option D pattern) - 4 jq calls → 1 jq call
    IFS=$'\t' read -r MOOD SUCCESS_STREAK FAILURE_STREAK FRUSTRATION < <(
        jq -r '
            [
                (.current_state.overall_mood // "neutral"),
                (.current_state.success_streak // 0),
                (.current_state.failure_streak // 0),
                (.current_state.frustration_level // 0)
            ] | @tsv
        ' "$EMOTIONAL_FILE" 2>/dev/null || echo -e "neutral\t0\t0\t0"
    )

    # Mood emoji
    case "$MOOD" in
        positive) MOOD_EMOJI="😊" ;;
        negative) MOOD_EMOJI="😟" ;;
        frustrated) MOOD_EMOJI="😤" ;;
        *) MOOD_EMOJI="😐" ;;
    esac

    echo "$MOOD_EMOJI Mood: $MOOD"
    echo "✅ Success streak: $SUCCESS_STREAK"
    echo "❌ Failure streak: $FAILURE_STREAK"
    echo "😤 Frustration level: $FRUSTRATION"
else
    echo "⚠️  Emotional state file not found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊  PERSONA STATISTICS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "$STATE_FILE" ]; then
    echo ""
    # Consolidated persona stats (Option D pattern) - 6×4 = 24 jq calls → 1 jq call
    jq -r '
        .personas as $personas |
        ["auditor", "optimizer", "architect", "experimenter", "maintainer", "skeptic"] |
        .[] |
        [
            $personas[.].display_name,
            ($personas[.].total_activations // 0),
            ($personas[.].tasks_completed // 0),
            ($personas[.].tasks_failed // 0)
        ] | @tsv
    ' "$STATE_FILE" 2>/dev/null | while IFS=$'\t' read -r DISPLAY ACTIVATIONS COMPLETED FAILED; do
        printf "  %-20s Activations: %3d | Tasks: %3d ✅  %2d ❌\n" "$DISPLAY" "$ACTIVATIONS" "$COMPLETED" "$FAILED"
    done
else
    echo "⚠️  State file not found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋  PENDING TASKS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TASK_QUEUE="${DAEMON_ROOT}/tasks/queue.md"
if [ -f "$TASK_QUEUE" ]; then
    PENDING_COUNT=$(grep -c "^- \[ \]" "$TASK_QUEUE" || echo "0")
    echo "📝 Pending tasks: $PENDING_COUNT"
    echo ""
    if [ "$PENDING_COUNT" -gt 0 ]; then
        echo "Next tasks:"
        grep "^- \[ \]" "$TASK_QUEUE" | head -n 3 | sed 's/^/  /'
        if [ "$PENDING_COUNT" -gt 3 ]; then
            echo "  ... and $((PENDING_COUNT - 3)) more"
        fi
    fi
else
    echo "⚠️  Task queue not found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📜  RECENT ACTIVITY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "$ACTIVITY_LOG" ]; then
    echo ""
    tail -n 10 "$ACTIVITY_LOG" | sed 's/^/  /'
else
    echo "⚠️  Activity log not found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Commands:"
echo "  • Attach: tmux attach -t $TMUX_SESSION"
echo "  • Logs:   tail -f ~/.claude/daemon/logs/activity.log"
echo "  • Stop:   ~/.claude/daemon/claude-daemon-stop.sh"
echo ""
