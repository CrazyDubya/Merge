#!/bin/bash
# Mark inbox messages as read by moving from unread/ to read/
# Usage: ./mark-messages-read.sh [pattern]
#   If pattern provided: moves matching files only
#   If no pattern: interactive mode to select files

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
HUMAN_UNREAD="$DAEMON_ROOT/inbox/human/unread"
HUMAN_READ="$DAEMON_ROOT/inbox/human/read"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to move a single file
move_to_read() {
    local file="$1"
    local filename=$(basename "$file")

    if [[ -f "$HUMAN_READ/$filename" ]]; then
        echo -e "${YELLOW}WARNING: $filename already exists in read folder${NC}"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipped: $filename"
            return 1
        fi
    fi

    mv "$file" "$HUMAN_READ/"
    echo -e "${GREEN}✓${NC} Moved to read: $filename"
    return 0
}

# Count unread messages
count_unread() {
    find "$HUMAN_UNREAD" -type f -name "*.md" | wc -l
}

# List unread messages with numbers
list_unread() {
    echo -e "\n${YELLOW}Unread messages:${NC}"
    find "$HUMAN_UNREAD" -type f -name "*.md" -printf "%T@ %p\n" | \
        sort -rn | \
        cut -d' ' -f2- | \
        nl -w2 -s'. '
}

# Interactive mode
interactive_mode() {
    local total=$(count_unread)

    if [[ $total -eq 0 ]]; then
        echo -e "${GREEN}No unread messages!${NC}"
        exit 0
    fi

    echo -e "${YELLOW}Found $total unread messages${NC}"
    list_unread

    echo -e "\n${YELLOW}Options:${NC}"
    echo "  a - Mark ALL as read"
    echo "  # - Mark specific number(s) as read (e.g., '1,3,5' or '1-5')"
    echo "  q - Quit without changes"
    echo
    read -p "Choice: " choice

    case "$choice" in
        q|Q)
            echo "Cancelled."
            exit 0
            ;;
        a|A)
            echo "Moving all messages to read folder..."
            moved=0
            for file in "$HUMAN_UNREAD"/*.md; do
                [[ -f "$file" ]] || continue
                if move_to_read "$file"; then
                    ((moved++))
                fi
            done
            echo -e "\n${GREEN}Moved $moved messages to read folder${NC}"
            ;;
        *)
            # Parse number selection
            echo "Moving selected messages..."
            moved=0

            # Get file array
            mapfile -t files < <(find "$HUMAN_UNREAD" -type f -name "*.md" -printf "%T@ %p\n" | sort -rn | cut -d' ' -f2-)

            # Parse selection (supports 1,3,5 or 1-5)
            IFS=',' read -ra selections <<< "$choice"
            for sel in "${selections[@]}"; do
                if [[ $sel =~ ^([0-9]+)-([0-9]+)$ ]]; then
                    # Range (e.g., 1-5)
                    start=${BASH_REMATCH[1]}
                    end=${BASH_REMATCH[2]}
                    for i in $(seq $start $end); do
                        idx=$((i-1))
                        if [[ $idx -ge 0 && $idx -lt ${#files[@]} ]]; then
                            if move_to_read "${files[$idx]}"; then
                                ((moved++))
                            fi
                        fi
                    done
                elif [[ $sel =~ ^[0-9]+$ ]]; then
                    # Single number
                    idx=$((sel-1))
                    if [[ $idx -ge 0 && $idx -lt ${#files[@]} ]]; then
                        if move_to_read "${files[$idx]}"; then
                            ((moved++))
                        fi
                    fi
                fi
            done

            echo -e "\n${GREEN}Moved $moved messages to read folder${NC}"
            ;;
    esac

    # Show remaining count
    remaining=$(count_unread)
    echo -e "${YELLOW}Remaining unread: $remaining${NC}"
}

# Pattern mode
pattern_mode() {
    local pattern="$1"
    echo "Searching for pattern: $pattern"

    moved=0
    found=0

    for file in "$HUMAN_UNREAD"/*.md; do
        [[ -f "$file" ]] || continue

        if grep -q "$pattern" "$file" || [[ $(basename "$file") == *"$pattern"* ]]; then
            ((found++))
            echo "Match: $(basename "$file")"
            if move_to_read "$file"; then
                ((moved++))
            fi
        fi
    done

    if [[ $found -eq 0 ]]; then
        echo -e "${YELLOW}No messages matching pattern: $pattern${NC}"
    else
        echo -e "\n${GREEN}Moved $moved of $found matching messages${NC}"
    fi
}

# Main
main() {
    # Ensure directories exist
    mkdir -p "$HUMAN_UNREAD" "$HUMAN_READ"

    if [[ $# -eq 0 ]]; then
        # Interactive mode
        interactive_mode
    else
        # Pattern mode
        pattern_mode "$1"
    fi
}

main "$@"
