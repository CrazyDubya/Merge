#!/bin/bash
#
# Add a task to the daemon's queue
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
TASK_QUEUE="${DAEMON_ROOT}/tasks/queue.md"

if [ $# -eq 0 ]; then
    echo "Usage: $0 \"Task description\""
    echo ""
    echo "Examples:"
    echo "  $0 \"Review security vulnerabilities in auth module\""
    echo "  $0 \"⚠️ Fix critical bug in payment processing\""
    echo "  $0 \"Experiment: Can we optimize the database queries?\""
    exit 1
fi

TASK="$1"

# Add task to queue (insert after the "## Pending Tasks" section)
if [ -f "$TASK_QUEUE" ]; then
    # Create temp file
    TEMP_FILE=$(mktemp)

    # Insert task after "## Pending Tasks" header
    awk -v task="$TASK" '
        /^## Pending Tasks/ {
            print
            print ""
            print "- [ ] " task
            added=1
            next
        }
        { print }
    ' "$TASK_QUEUE" > "$TEMP_FILE"

    mv "$TEMP_FILE" "$TASK_QUEUE"

    echo "✅ Task added to queue:"
    echo "   $TASK"
    echo ""
    echo "View queue: cat ~/.claude/daemon/tasks/queue.md"
else
    echo "❌ Error: Task queue not found at $TASK_QUEUE"
    exit 1
fi
