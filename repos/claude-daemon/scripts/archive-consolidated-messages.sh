#!/bin/bash
# archive-consolidated-messages.sh
# Archives messages that have been consolidated by START-HERE documents
#
# This script:
# 1. Finds all START-HERE documents in inbox/human/unread/
# 2. Extracts the 'consolidates:' field from frontmatter
# 3. Lists the messages that would be archived
# 4. Asks for confirmation
# 5. Moves consolidated messages to archive with timestamp

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
INBOX_DIR="${DAEMON_ROOT}/inbox/human/unread"
ARCHIVE_DIR="${DAEMON_ROOT}/inbox/human/archive/consolidated"
TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Consolidated Message Archival ===${NC}"
echo ""

# Check if inbox directory exists
if [ ! -d "$INBOX_DIR" ]; then
    echo -e "${RED}ERROR: Inbox directory not found: $INBOX_DIR${NC}"
    exit 1
fi

cd "$INBOX_DIR"

# Find START-HERE documents
START_HERE_DOCS=$(ls -1 00-START-HERE-*.md 2>/dev/null || true)

if [ -z "$START_HERE_DOCS" ]; then
    echo -e "${YELLOW}No START-HERE documents found in inbox.${NC}"
    echo "Nothing to archive."
    exit 0
fi

echo -e "${GREEN}Found START-HERE documents:${NC}"
echo "$START_HERE_DOCS" | while read doc; do
    echo "  - $doc"
done
echo ""

# Extract consolidated message lists
declare -a MESSAGES_TO_ARCHIVE=()
TOTAL_LINES=0

while read START_HERE_DOC; do
    echo -e "${BLUE}Analyzing: $START_HERE_DOC${NC}"

    # Extract consolidates field from frontmatter
    # Format: consolidates: [msg1, msg2, msg3]
    CONSOLIDATES_LINE=$(grep "^consolidates:" "$START_HERE_DOC" 2>/dev/null || true)

    if [ -z "$CONSOLIDATES_LINE" ]; then
        echo -e "${YELLOW}  No 'consolidates:' field found - skipping${NC}"
        continue
    fi

    # Extract message IDs (remove consolidates:, brackets, and split by comma)
    MESSAGES=$(echo "$CONSOLIDATES_LINE" | sed 's/consolidates: \[//' | sed 's/\]//' | tr ',' '\n')

    while read -r MSG; do
        # Trim whitespace
        MSG=$(echo "$MSG" | xargs)

        if [ -z "$MSG" ]; then
            continue
        fi

        # Add .md extension if not present
        if [[ ! "$MSG" =~ \.md$ ]]; then
            MSG="${MSG}.md"
        fi

        # Check if message exists
        if [ -f "$MSG" ]; then
            LINES=$(wc -l < "$MSG")
            TOTAL_LINES=$((TOTAL_LINES + LINES))
            echo -e "  ${GREEN}✓${NC} $MSG ($LINES lines)"
            MESSAGES_TO_ARCHIVE+=("$MSG")
        else
            echo -e "  ${YELLOW}⚠${NC} $MSG (not found - may already be archived)"
        fi
    done <<< "$MESSAGES"

    echo ""
done <<< "$START_HERE_DOCS"

# Show summary
NUM_MESSAGES=${#MESSAGES_TO_ARCHIVE[@]}

if [ $NUM_MESSAGES -eq 0 ]; then
    echo -e "${YELLOW}No messages to archive.${NC}"
    echo "All consolidated messages may already be archived."
    exit 0
fi

echo -e "${BLUE}=== Summary ===${NC}"
echo -e "Messages to archive: ${GREEN}$NUM_MESSAGES${NC}"
echo -e "Total lines: ${GREEN}$TOTAL_LINES${NC}"
echo -e "Archive destination: ${BLUE}$ARCHIVE_DIR/$TIMESTAMP/${NC}"
echo ""

# Show what will be archived
echo -e "${YELLOW}The following messages will be moved to archive:${NC}"
for msg in "${MESSAGES_TO_ARCHIVE[@]}"; do
    echo "  - $msg"
done
echo ""

# Ask for confirmation
read -p "Archive these messages? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Archival cancelled.${NC}"
    exit 0
fi

# Create archive directory
ARCHIVE_PATH="${ARCHIVE_DIR}/${TIMESTAMP}"
mkdir -p "$ARCHIVE_PATH"

# Move messages to archive
echo ""
echo -e "${BLUE}Archiving messages...${NC}"

ARCHIVED_COUNT=0
for msg in "${MESSAGES_TO_ARCHIVE[@]}"; do
    if mv "$msg" "$ARCHIVE_PATH/"; then
        echo -e "  ${GREEN}✓${NC} Archived: $msg"
        ARCHIVED_COUNT=$((ARCHIVED_COUNT + 1))
    else
        echo -e "  ${RED}✗${NC} Failed to archive: $msg"
    fi
done

echo ""
echo -e "${GREEN}=== Complete ===${NC}"
echo -e "Archived: ${GREEN}$ARCHIVED_COUNT${NC} messages"
echo -e "Location: ${BLUE}$ARCHIVE_PATH${NC}"
echo ""

# Create index file in archive
INDEX_FILE="${ARCHIVE_PATH}/INDEX.md"
cat > "$INDEX_FILE" << EOF
# Consolidated Message Archive

**Date**: $(date +"%Y-%m-%d %H:%M:%S %Z")
**Reason**: Messages consolidated by START-HERE documents
**Archived by**: scripts/archive-consolidated-messages.sh

## Archived Messages

EOF

for msg in "${MESSAGES_TO_ARCHIVE[@]}"; do
    if [ -f "${ARCHIVE_PATH}/${msg}" ]; then
        LINES=$(wc -l < "${ARCHIVE_PATH}/${msg}")
        echo "- \`$msg\` ($LINES lines)" >> "$INDEX_FILE"
    fi
done

cat >> "$INDEX_FILE" << EOF

## START-HERE Documents (not archived)

These documents remain in the inbox as entry points:

EOF

while read doc; do
    echo "- \`$doc\`" >> "$INDEX_FILE"
done <<< "$START_HERE_DOCS"

echo -e "${BLUE}Index created: $INDEX_FILE${NC}"
echo ""

# Log to persona timeline
if [ -f "${DAEMON_ROOT}/memory/persona-timeline.jsonl" ]; then
    jq -n --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
          --arg archived "$ARCHIVED_COUNT" \
          --arg path "$ARCHIVE_PATH" \
          '{
        timestamp: $ts,
        persona: "human",
        activity: "inbox_archival",
        details: ("Archived " + $archived + " consolidated messages to reduce inbox bloat"),
        files: [$path],
        duration_minutes: 1
    }' | {
    source "${DAEMON_ROOT}/lib/atomic-io.sh" 2>/dev/null || true
    atomic_append "${DAEMON_ROOT}/memory/persona-timeline.jsonl" "$(cat)" 2>/dev/null || cat >> "${DAEMON_ROOT}/memory/persona-timeline.jsonl"
}
fi

# Show final inbox state
REMAINING=$(ls -1 *.md 2>/dev/null | wc -l)
echo -e "${GREEN}Inbox cleanup complete!${NC}"
echo -e "Messages remaining in inbox: ${BLUE}$REMAINING${NC}"
echo ""
echo -e "${YELLOW}Tip:${NC} You can view archived messages at:"
echo -e "  ${BLUE}$ARCHIVE_PATH${NC}"
