#!/bin/bash
#
# Restart the multi-persona Claude daemon via systemd
#

set -euo pipefail

SERVICE_NAME="claude-daemon.service"

echo "🔄 Restarting Claude daemon..."
echo ""

# Restart via systemd (works whether running or not)
if systemctl --user restart "$SERVICE_NAME"; then
    echo "✅ Daemon restarted successfully"
    echo ""
    echo "Useful commands:"
    echo "  • Status: systemctl --user status $SERVICE_NAME"
    echo "  • Logs: journalctl --user -u $SERVICE_NAME -f"
    echo "  • Full status: ~/.claude/daemon/claude-daemon-status.sh"
    echo ""
else
    echo "❌ Failed to restart daemon"
    echo ""
    echo "Troubleshooting:"
    echo "  • Check status: systemctl --user status $SERVICE_NAME"
    echo "  • Check logs: journalctl --user -u $SERVICE_NAME -n 50"
    echo "  • Manual start: ~/.claude/daemon/claude-daemon-start.sh"
    echo ""
    exit 1
fi
