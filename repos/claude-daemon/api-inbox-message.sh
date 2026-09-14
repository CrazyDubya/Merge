#!/bin/bash
#
# Inbox Message Content API - CGI Script
# Returns full markdown content of a specific message
#
# Query parameters:
#   folder=unread|read (default: unread)
#   file=<filename> (required)
#
# Security:
# - Strict filename validation (alphanumeric, dash, underscore, .md only)
# - No directory traversal (rejects ../ and /)
# - File size limit (1MB max)
# - Path canonicalization
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

# Parse query string
QUERY_STRING="${QUERY_STRING:-}"
FOLDER="unread"
FILENAME=""

# Extract parameters from query string
if [[ "$QUERY_STRING" == *"folder=read"* ]]; then
    FOLDER="read"
fi

# Extract filename parameter
if [[ "$QUERY_STRING" =~ file=([^&]+) ]]; then
    FILENAME="${BASH_REMATCH[1]}"
fi

# Security: Validate folder (whitelist)
case "$FOLDER" in
    unread|read) ;;
    *) FOLDER="unread" ;;
esac

# Security: Validate filename
# Must be: alphanumeric, dash, underscore, dot only
# Must end with .md
# No directory traversal characters
if [[ -z "$FILENAME" ]]; then
    echo "Content-Type: application/json"
    echo ""
    echo '{"error": "Missing file parameter"}'
    exit 0
fi

if [[ ! "$FILENAME" =~ ^[a-zA-Z0-9_-]+\.md$ ]]; then
    echo "Content-Type: application/json"
    echo ""
    echo '{"error": "Invalid filename format"}'
    exit 0
fi

# Construct full path
INBOX_DIR="${DAEMON_ROOT}/inbox/human/${FOLDER}"
FILE_PATH="${INBOX_DIR}/${FILENAME}"

# Security: Canonicalize path and verify it's within inbox dir
CANONICAL_PATH=$(readlink -f "$FILE_PATH" 2>/dev/null || echo "")
CANONICAL_INBOX=$(readlink -f "$INBOX_DIR" 2>/dev/null || echo "")

if [[ ! "$CANONICAL_PATH" == "$CANONICAL_INBOX"/* ]]; then
    echo "Content-Type: application/json"
    echo ""
    echo '{"error": "Path validation failed"}'
    exit 0
fi

# Check file exists and is readable
if [ ! -f "$FILE_PATH" ] || [ ! -r "$FILE_PATH" ]; then
    echo "Content-Type: application/json"
    echo ""
    echo '{"error": "File not found or not readable"}'
    exit 0
fi

# Security: Check file size (max 1MB)
FILE_SIZE=$(stat -c%s "$FILE_PATH" 2>/dev/null || echo "0")
if [ "$FILE_SIZE" -gt 1048576 ]; then
    echo "Content-Type: application/json"
    echo ""
    echo '{"error": "File too large (max 1MB)"}'
    exit 0
fi

# CGI Headers
echo "Content-Type: application/json"
echo "Cache-Control: no-cache, no-store, must-revalidate"
echo "Access-Control-Allow-Origin: *"
echo ""

# Read file content and escape for JSON
# Performance: Single read, escape in-place
CONTENT=$(cat "$FILE_PATH" | jq -Rs .)

# Output JSON
cat <<EOF
{
  "filename": "$FILENAME",
  "folder": "$FOLDER",
  "size": $FILE_SIZE,
  "content": $CONTENT
}
EOF
