#!/bin/bash
#
# Simple HTTP server for the daemon dashboard
# EXPERIMENTER NOTE: This is intentionally simple - just serves files
# ARCHITECT can improve this later with proper routing, caching, etc.
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
PORT="${1:-8080}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎭  Starting Multi-Persona Daemon Dashboard Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📍 Root: $DAEMON_ROOT"
echo "🌐 Port: $PORT"
echo "🔗 URL:  http://localhost:$PORT/dashboard.html"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd "$DAEMON_ROOT"

# Check if Python 3 is available
if command -v python3 &> /dev/null; then
    python3 -m http.server "$PORT"
elif command -v python &> /dev/null; then
    python -m SimpleHTTPServer "$PORT"
else
    echo "❌ Error: Python not found. Install Python to run the dashboard server."
    exit 1
fi
