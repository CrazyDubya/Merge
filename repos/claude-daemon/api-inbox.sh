#!/bin/bash
#
# Inbox API Endpoint - CGI Script
# Returns JSON list of inbox messages with metadata
#
# Performance optimizations:
# - Parallel file reads using xargs
# - Single-pass YAML parsing (no multiple grep calls)
# - Bounded output (max 100 messages to prevent DoS)
# - File size limits (skip files >1MB)
#
# Security:
# - Path validation (no directory traversal)
# - File size limits
# - Output escaping for JSON
# - Read-only operations
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"

# EXPERIMENTER: Security - Validate authentication token
# Expected token hash (same as dashboard.html)
EXPECTED_TOKEN="79a8427e7b4d26669df2b08c90aae85667ee979fe74efb1f1a975570747f3371"

# Read Authorization header (CGI provides HTTP_* variables)
AUTH_HEADER="${HTTP_AUTHORIZATION:-}"

# Extract Bearer token
if [[ "$AUTH_HEADER" =~ ^Bearer\ (.+)$ ]]; then
    PROVIDED_TOKEN="${BASH_REMATCH[1]}"
else
    PROVIDED_TOKEN=""
fi

# Validate token
if [[ "$PROVIDED_TOKEN" != "$EXPECTED_TOKEN" ]]; then
    echo "Content-Type: application/json"
    echo "HTTP/1.1 401 Unauthorized"
    echo ""
    echo '{"error": "Unauthorized - invalid or missing authentication token"}'
    exit 0
fi

# Query parameter: folder (unread|read, defaults to unread)
QUERY_STRING="${QUERY_STRING:-}"
FOLDER="unread"

if [[ "$QUERY_STRING" == *"folder=read"* ]]; then
    FOLDER="read"
fi

# Security: Validate folder parameter (whitelist only)
case "$FOLDER" in
    unread|read) ;;
    *) FOLDER="unread" ;;
esac

INBOX_DIR="${DAEMON_ROOT}/inbox/human/${FOLDER}"

# CGI Headers
echo "Content-Type: application/json"
echo "Cache-Control: no-cache, no-store, must-revalidate"
echo "Access-Control-Allow-Origin: *"
echo ""

# Check if inbox directory exists
if [ ! -d "$INBOX_DIR" ]; then
    echo '{"error": "Inbox directory not found", "messages": []}'
    exit 0
fi

# Parse YAML frontmatter from a single message file
# Returns JSON object with metadata
parse_message_metadata() {
    local file="$1"
    local filename=$(basename "$file")

    # Security: Check file size (skip if >1MB to prevent DoS)
    local size=$(stat -c%s "$file" 2>/dev/null || echo "0")
    if [ "$size" -gt 1048576 ]; then
        echo '{"error": "file too large", "filename": "'$filename'"}'
        return
    fi

    # Extract YAML frontmatter (between --- markers)
    # Performance: Single awk pass instead of multiple grep calls
    local frontmatter=$(awk '
        BEGIN { in_yaml=0; yaml="" }
        /^---$/ {
            if (in_yaml == 0) {
                in_yaml = 1
                next
            } else {
                print yaml
                exit
            }
        }
        in_yaml == 1 { yaml = yaml $0 "\n" }
    ' "$file" 2>/dev/null)

    # Extract first non-empty heading as preview (after frontmatter)
    local preview=$(awk '
        BEGIN { in_yaml=0; yaml_done=0 }
        /^---$/ {
            if (in_yaml == 0) {
                in_yaml = 1
                next
            } else {
                in_yaml = 0
                yaml_done = 1
                next
            }
        }
        yaml_done == 1 && /^# / {
            # Remove leading # and whitespace
            gsub(/^#+ */, "")
            print
            exit
        }
    ' "$file" 2>/dev/null)

    # Parse YAML fields using grep (simpler than awk for key:value)
    # Escape double quotes for JSON
    local from=$(echo "$frontmatter" | grep "^from:" | sed 's/^from: *//' | sed 's/"/\\"/g' | head -1)
    local to=$(echo "$frontmatter" | grep "^to:" | sed 's/^to: *//' | sed 's/"/\\"/g' | head -1)
    local timestamp=$(echo "$frontmatter" | grep "^timestamp:" | sed 's/^timestamp: *//' | head -1)
    local priority=$(echo "$frontmatter" | grep "^priority:" | sed 's/^priority: *//' | head -1)
    local message_id=$(echo "$frontmatter" | grep "^message_id:" | sed 's/^message_id: *//' | head -1)

    # Parse tags array (YAML list format: [tag1, tag2, tag3])
    local tags=$(echo "$frontmatter" | grep "^tags:" | sed 's/^tags: *//' | sed 's/\[//' | sed 's/\]//' | sed 's/"/\\"/g' | head -1)

    # Default values if not found
    from="${from:-unknown}"
    to="${to:-human}"
    timestamp="${timestamp:-}"
    priority="${priority:-normal}"
    message_id="${message_id:-$filename}"
    tags="${tags:-}"
    preview="${preview:-No preview available}"

    # Escape preview for JSON (replace newlines, quotes)
    preview=$(echo "$preview" | sed 's/"/\\"/g' | tr -d '\n' | head -c 200)

    # Calculate file modification time as fallback timestamp
    local mtime=$(stat -c%Y "$file" 2>/dev/null || echo "0")
    if [ -z "$timestamp" ]; then
        timestamp=$(date -d "@$mtime" --iso-8601=seconds 2>/dev/null || echo "")
    fi

    # Output JSON object (single line)
    cat <<EOF
{"filename":"$filename","from":"$from","to":"$to","timestamp":"$timestamp","priority":"$priority","message_id":"$message_id","tags":"$tags","preview":"$preview","size":$size}
EOF
}

# Export function for xargs parallel execution
export -f parse_message_metadata
export DAEMON_ROOT

# Get all .md files in inbox, sorted by modification time (newest first)
# Performance: Bounded to max 100 messages
mapfile -t message_files < <(
    find "$INBOX_DIR" -maxdepth 1 -name "*.md" -type f 2>/dev/null | \
    xargs -r ls -t 2>/dev/null | \
    head -100
)

# Count total messages
total=${#message_files[@]}

# Parse messages in parallel (up to 4 at a time for performance)
# Performance: xargs -P4 for parallel processing
echo "{"
echo "  \"folder\": \"$FOLDER\","
echo "  \"total\": $total,"
echo "  \"messages\": ["

if [ "$total" -gt 0 ]; then
    # Use printf to avoid issues with filenames containing special chars
    # Process in parallel, then join with commas
    printf '%s\n' "${message_files[@]}" | \
        xargs -I{} -P4 bash -c 'parse_message_metadata "{}"' | \
        awk 'NR > 1 { print "," } { printf "    %s", $0 }'
    echo ""
fi

echo "  ]"
echo "}"
