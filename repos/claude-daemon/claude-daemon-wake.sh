#!/bin/bash
#
# Wake the daemon immediately (disturb its sleep)
#

set -euo pipefail

TMUX_SESSION="claude-daemon"

# Check if daemon is running
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "❌ Daemon is not running"
    echo "Start with: ~/.claude/daemon/claude-daemon-start.sh"
    exit 1
fi

echo "⏰ Waking daemon..."

# Send Ctrl+C to interrupt sleep, then Enter to resume
tmux send-keys -t "$TMUX_SESSION" C-c 2>/dev/null || true

echo "✅ Daemon awakened!"
echo ""
echo "The daemon will process its next cycle immediately."
echo "Check activity: tail -f ~/.claude/daemon/logs/activity.log"
