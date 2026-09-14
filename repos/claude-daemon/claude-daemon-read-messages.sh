#!/bin/bash
#
# claude-daemon-read-messages.sh
# View messages from the daemon
#
# Usage:
#   claude-daemon-read-messages.sh [options]
#
# Options:
#   --all          Show both unread and read messages
#   --move-to-read Move all unread messages to read/ after displaying
#   --count        Just count unread messages
#   --folder DIR   Check specific folder (default: unread)
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
INBOX_DIR="${DAEMON_ROOT}/inbox"
HUMAN_UNREAD="${INBOX_DIR}/human/unread"
HUMAN_READ="${INBOX_DIR}/human/read"

# Defaults
SHOW_ALL=false
MOVE_TO_READ=false
COUNT_ONLY=false
FOLDER="unread"

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            SHOW_ALL=true
            shift
            ;;
        --move-to-read)
            MOVE_TO_READ=true
            shift
            ;;
        --count)
            COUNT_ONLY=true
            shift
            ;;
        --folder)
            FOLDER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $(basename $0) [options]"
            echo ""
            echo "Options:"
            echo "  --all           Show both unread and read messages"
            echo "  --move-to-read  Mark all unread messages as read"
            echo "  --count         Just count unread messages"
            echo "  --folder DIR    Check specific folder (default: unread)"
            echo "  -h, --help      Show this help"
            echo ""
            echo "Examples:"
            echo "  $(basename $0)                    # Show unread messages"
            echo "  $(basename $0) --all              # Show all messages"
            echo "  $(basename $0) --count            # Count unread"
            echo "  $(basename $0) --folder urgent    # Check urgent folder"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if inbox exists
if [ ! -d "$HUMAN_UNREAD" ]; then
    echo "📭 Inbox not initialized yet. No messages from daemon."
    echo ""
    echo "Inbox directory: $INBOX_DIR"
    exit 0
fi

# Count messages
UNREAD_COUNT=$(find "$HUMAN_UNREAD" -type f -name "*.md" 2>/dev/null | wc -l)

if [ "$COUNT_ONLY" = true ]; then
    echo "$UNREAD_COUNT"
    exit 0
fi

if [ "$UNREAD_COUNT" -eq 0 ]; then
    echo "📭 No unread messages from daemon"
    if [ "$SHOW_ALL" = true ]; then
        READ_COUNT=$(find "$HUMAN_READ" -type f -name "*.md" 2>/dev/null | wc -l)
        if [ "$READ_COUNT" -gt 0 ]; then
            echo ""
            echo "📚 You have $READ_COUNT read message(s)"
            echo "   View with: ls -lt $HUMAN_READ/"
        fi
    fi
    exit 0
fi

echo "📬 You have $UNREAD_COUNT unread message(s) from the daemon"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Display unread messages
for message_file in "$HUMAN_UNREAD"/*.md; do
    [ -e "$message_file" ] || continue

    BASENAME=$(basename "$message_file")

    # Extract frontmatter fields
    FROM=$(grep "^from:" "$message_file" | head -1 | cut -d: -f2- | xargs)
    TIMESTAMP=$(grep "^timestamp:" "$message_file" | head -1 | cut -d: -f2- | xargs)
    PRIORITY=$(grep "^priority:" "$message_file" | head -1 | cut -d: -f2- | xargs)
    TAGS=$(grep "^tags:" "$message_file" | head -1 | cut -d: -f2- | xargs)
    REPLY_TO=$(grep "^reply_to:" "$message_file" | head -1 | cut -d: -f2- | xargs)

    # Format timestamp for display
    DISPLAY_TIME=$(date -d "$TIMESTAMP" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "$TIMESTAMP")

    # Priority emoji
    PRIORITY_EMOJI="📝"
    if [ "$PRIORITY" = "urgent" ]; then
        PRIORITY_EMOJI="🚨"
    elif [ "$PRIORITY" = "high" ]; then
        PRIORITY_EMOJI="⚡"
    fi

    # Display header
    echo "$PRIORITY_EMOJI From: $FROM"
    echo "   Time: $DISPLAY_TIME"
    [ -n "$TAGS" ] && [ "$TAGS" != "[]" ] && [ "$TAGS" != "null" ] && echo "   Tags: $TAGS"
    [ -n "$REPLY_TO" ] && [ "$REPLY_TO" != "null" ] && echo "   Re: $REPLY_TO"
    echo ""

    # Display message content (skip frontmatter)
    awk '/^---$/{p++; next} p==2' "$message_file"

    echo ""
    echo "   📁 File: $BASENAME"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
done

# Move to read if requested
if [ "$MOVE_TO_READ" = true ]; then
    echo "Moving $UNREAD_COUNT message(s) to read..."
    mkdir -p "$HUMAN_READ"
    for message_file in "$HUMAN_UNREAD"/*.md; do
        [ -e "$message_file" ] || continue
        mv "$message_file" "$HUMAN_READ/"
    done
    echo "✅ All messages marked as read"
fi

# Show all if requested
if [ "$SHOW_ALL" = true ] && [ "$MOVE_TO_READ" = false ]; then
    READ_COUNT=$(find "$HUMAN_READ" -type f -name "*.md" 2>/dev/null | wc -l)
    if [ "$READ_COUNT" -gt 0 ]; then
        echo ""
        echo "📚 Previously read messages ($READ_COUNT):"
        ls -1t "$HUMAN_READ"/*.md | head -5 | while read -r file; do
            echo "   - $(basename "$file")"
        done
        if [ "$READ_COUNT" -gt 5 ]; then
            echo "   ... and $((READ_COUNT - 5)) more"
        fi
    fi
fi

echo ""
echo "To mark messages as read: $(basename $0) --move-to-read"
echo "To send a reply: $DAEMON_ROOT/claude-daemon-send-message.sh \"Your message\""
