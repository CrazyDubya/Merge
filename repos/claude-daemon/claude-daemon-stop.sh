#!/bin/bash
#
# Stop the multi-persona Claude daemon via systemd
#

set -euo pipefail

SERVICE_NAME="claude-daemon.service"

echo "🛑 Stopping Claude daemon..."
echo ""

# Check if service is running
if ! systemctl --user is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
    echo "⚠️  Daemon is not running"
    exit 0
fi

# Stop via systemd
if systemctl --user stop "$SERVICE_NAME"; then
    echo "✅ Daemon stopped successfully"
    echo ""
    echo "Notes:"
    echo "  • Logs preserved in ~/.claude/daemon/logs/"
    echo "  • systemd service disabled - will restart on boot unless disabled"
    echo "  • To disable auto-start: systemctl --user disable $SERVICE_NAME"
    echo "  • To restart: ~/.claude/daemon/claude-daemon-start.sh"
    echo ""
else
    echo "❌ Failed to stop daemon gracefully"
    echo ""
    echo "Force stop options:"
    echo "  • Kill tmux session: tmux kill-session -t claude-daemon"
    echo "  • Check processes: ps aux | grep daemon"
    echo ""
    exit 1
fi
