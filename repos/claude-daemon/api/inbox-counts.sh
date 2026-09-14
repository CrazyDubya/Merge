#!/bin/bash
#
# api/inbox-counts.sh - Returns inbox message counts dynamically
#
# Part of Architect's clean separation approach for dashboard fix.
# Returns ONLY message counts (human_unread, daemon_unread) - the ONE
# piece of data not available in static JSON files.
#
# All other dashboard data (persona, frustration, success_streak, last_switch)
# already exists in state.json and emotional.json, so no duplication.
#
# Security features (from Auditor's requirements):
# - Proper CORS headers (daemon.claude-play.com only)
# - Cache-Control: no-cache (always fresh data)
# - Robust file counting (find -type f, handles symlinks/directories)
# - Error handling with graceful degradation
# - Input validation (no user input, but defensive coding)

set -euo pipefail

# Configuration
DAEMON_ROOT="${DAEMON_ROOT:-/home/opc/.claude/daemon}"
INBOX_HUMAN_UNREAD="${DAEMON_ROOT}/inbox/human/unread"
INBOX_DAEMON_UNREAD="${DAEMON_ROOT}/inbox/daemon/unread"

# Content type (MUST be first for CGI handler to work correctly)
echo "Content-Type: application/json"

# Security: Cache-Control headers
# No caching - always return fresh data
echo "Cache-Control: no-cache, no-store, must-revalidate"
echo "Pragma: no-cache"
echo "Expires: 0"

# Security: Proper CORS headers
# Only allow daemon.claude-play.com (not wildcard *)
echo "Access-Control-Allow-Origin: https://daemon.claude-play.com"
echo "Access-Control-Allow-Methods: GET"
echo "Access-Control-Allow-Headers: Content-Type"
echo ""

# Count messages robustly
# Use find -type f to only count regular files (not directories or symlinks)
# Handles edge cases that ls|grep|wc misses
count_messages() {
    local dir="$1"

    # Validate directory exists
    if [[ ! -d "$dir" ]]; then
        echo "0"
        return
    fi

    # Count regular files only, ignore .gitkeep
    find "$dir" -maxdepth 1 -type f ! -name '.gitkeep' 2>/dev/null | wc -l || echo "0"
}

# Error handling with graceful degradation
{
    human_unread=$(count_messages "$INBOX_HUMAN_UNREAD")
    daemon_unread=$(count_messages "$INBOX_DAEMON_UNREAD")

    # Output JSON
    cat <<EOF
{
  "human_unread": ${human_unread},
  "daemon_unread": ${daemon_unread},
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "source": "api/inbox-counts.sh"
}
EOF

} || {
    # Graceful degradation on error
    # Return valid JSON with zero counts rather than failing
    cat <<EOF
{
  "human_unread": 0,
  "daemon_unread": 0,
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "source": "api/inbox-counts.sh",
  "error": "Failed to count messages"
}
EOF
}
