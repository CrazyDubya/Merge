#!/bin/bash
#
# Start the multi-persona Claude daemon via systemd
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
SERVICE_NAME="claude-daemon.service"

echo "🚀 Starting multi-persona Claude daemon via systemd..."
echo ""

# Check if service is already running
if systemctl --user is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
    echo "⚠️  Daemon is already running"
    echo ""
    echo "Options:"
    echo "  • Check status: systemctl --user status $SERVICE_NAME"
    echo "  • View logs: journalctl --user -u $SERVICE_NAME -f"
    echo "  • Restart: ~/.claude/daemon/claude-daemon-restart.sh"
    echo "  • Stop: ~/.claude/daemon/claude-daemon-stop.sh"
    echo ""
    exit 1
fi

# Check if conversation ID is set
CONVERSATION_ID_FILE="${DAEMON_ROOT}/memory/conversation-id.txt"
if [ ! -s "$CONVERSATION_ID_FILE" ] || grep -q "^#" "$CONVERSATION_ID_FILE" 2>/dev/null; then
    echo "⚠️  WARNING: No conversation ID set!"
    echo ""
    echo "The daemon needs a conversation ID to persist memory."
    echo ""
    echo "To set one:"
    echo "  1. Start a conversation with Claude Code"
    echo "  2. Get the conversation ID (from conversation list or URL)"
    echo "  3. Run: echo 'your-conversation-id' > ~/.claude/daemon/memory/conversation-id.txt"
    echo ""
    echo "Or let the daemon create a new conversation on first run."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start via systemd
if systemctl --user start "$SERVICE_NAME"; then
    echo "✅ Daemon started successfully"
    echo ""
    echo "Auto-recovery features:"
    echo "  ✅ Auto-restart on crash (systemd)"
    echo "  ✅ Survives logout (user lingering)"
    echo "  ✅ Watchdog monitoring (cron-based)"
    echo "  ✅ Inbox alerts on repeated failures"
    echo ""
    echo "Useful commands:"
    echo "  • Status: systemctl --user status $SERVICE_NAME"
    echo "  • Logs (systemd): journalctl --user -u $SERVICE_NAME -f"
    echo "  • Logs (activity): tail -f ~/.claude/daemon/logs/activity.log"
    echo "  • Restart: ~/.claude/daemon/claude-daemon-restart.sh"
    echo "  • Stop: ~/.claude/daemon/claude-daemon-stop.sh"
    echo "  • Full status: ~/.claude/daemon/claude-daemon-status.sh"
    echo ""
    echo "🎭 The daemon will now run autonomously with full auto-recovery."
    echo ""
else
    echo "❌ Failed to start daemon"
    echo ""
    echo "Troubleshooting:"
    echo "  • Check service status: systemctl --user status $SERVICE_NAME"
    echo "  • Check logs: journalctl --user -u $SERVICE_NAME -n 50"
    echo "  • Verify service file: cat ~/.config/systemd/user/$SERVICE_NAME"
    echo ""
    exit 1
fi
