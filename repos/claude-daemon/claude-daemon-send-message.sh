#!/bin/bash
#
# claude-daemon-send-message.sh
# Send a message to the daemon's inbox
#
# Usage:
#   claude-daemon-send-message.sh "Your message here" [priority] [tags]
#
# Examples:
#   claude-daemon-send-message.sh "What are you working on?"
#   claude-daemon-send-message.sh "Security issue found!" urgent "bug,security"
#   claude-daemon-send-message.sh "Great job on that analysis" normal "feedback,praise"
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
INBOX_DIR="${DAEMON_ROOT}/inbox"
DAEMON_UNREAD="${INBOX_DIR}/daemon/unread"

# Check if inbox exists
if [ ! -d "$DAEMON_UNREAD" ]; then
    echo "❌ Inbox not initialized. Creating directory structure..."
    mkdir -p "$DAEMON_UNREAD"
fi

# Parse arguments
MESSAGE="$1"
PRIORITY="${2:-normal}"
TAGS="${3:-}"

if [ -z "$MESSAGE" ]; then
    echo "Usage: $(basename $0) \"Your message here\" [priority] [tags]"
    echo ""
    echo "Arguments:"
    echo "  message   - Your message to the daemon (required)"
    echo "  priority  - normal|high|urgent (default: normal)"
    echo "  tags      - Comma-separated tags like: question,feedback,bug"
    echo ""
    echo "Examples:"
    echo "  $(basename $0) \"What tasks are you working on today?\""
    echo "  $(basename $0) \"Found a critical bug\" urgent \"bug,critical\""
    echo "  $(basename $0) \"Can you explain the architecture?\" normal \"question,architecture\""
    exit 1
fi

# Validate priority
if [[ ! "$PRIORITY" =~ ^(normal|high|urgent)$ ]]; then
    echo "⚠️  Warning: Invalid priority '$PRIORITY'. Using 'normal' instead."
    echo "   Valid options: normal, high, urgent"
    PRIORITY="normal"
fi

# Generate message ID and timestamp
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
MESSAGE_ID="msg-$(date +%Y%m%d-%H%M%S)"
FILENAME="${MESSAGE_ID}.md"
FILEPATH="${DAEMON_UNREAD}/${FILENAME}"

# Format tags as JSON array
TAGS_JSON="[]"
if [ -n "$TAGS" ]; then
    # Convert comma-separated to JSON array
    IFS=',' read -ra TAG_ARRAY <<< "$TAGS"
    TAGS_JSON="["
    for i in "${!TAG_ARRAY[@]}"; do
        TAG="${TAG_ARRAY[$i]}"
        TAG=$(echo "$TAG" | xargs)  # trim whitespace
        TAGS_JSON+="\"$TAG\""
        if [ $i -lt $((${#TAG_ARRAY[@]} - 1)) ]; then
            TAGS_JSON+=", "
        fi
    done
    TAGS_JSON+="]"
fi

# Create message file with frontmatter
cat > "$FILEPATH" <<EOF
---
from: human
to: daemon
timestamp: $TIMESTAMP
priority: $PRIORITY
tags: $TAGS_JSON
reply_to: null
message_id: $MESSAGE_ID
---

$MESSAGE
EOF

echo "✅ Message sent to daemon!"
echo ""
echo "📁 File: $FILENAME"
echo "🕐 Time: $TIMESTAMP"
echo "⚡ Priority: $PRIORITY"
if [ -n "$TAGS" ]; then
    echo "🏷️  Tags: $TAGS"
fi
echo ""
echo "The daemon will process this message during its next conversation check."
echo "Daemon checks inbox with 10% probability on each wake cycle."
echo ""
echo "To see daemon's response:"
echo "  $DAEMON_ROOT/claude-daemon-read-messages.sh"
