#!/bin/bash
#
# inbox-routing-filter.sh - Metadata-based message routing for multi-persona system
#
# PURPOSE:
#   Determines if a message in inbox/daemon/unread/ should be processed by the current persona
#   based on message metadata (from/to fields in YAML frontmatter).
#
# USAGE:
#   inbox-routing-filter.sh <message_file> <current_persona>
#
# EXIT CODES:
#   0 - Message should be routed to current persona (PROCESS IT)
#   1 - Message should NOT be routed to current persona (SKIP IT)
#
# ROUTING RULES:
#   1. Don't process own messages (from == current_persona) → SKIP
#   2. Process messages addressed to current persona (to == current_persona) → ROUTE
#   3. Process broadcast messages (to == "daemon") → ROUTE
#   4. Process multi-recipient messages that include current persona → ROUTE
#   5. All other messages → SKIP
#
# EXAMPLES:
#   - Message from:skeptic to:auditor, current:skeptic → SKIP (own message)
#   - Message from:skeptic to:auditor, current:auditor → ROUTE (addressed to auditor)
#   - Message from:human to:daemon, current:maintainer → ROUTE (broadcast)
#   - Message from:experimenter to:maintainer, current:optimizer → SKIP (not for optimizer)
#
# CREATED: 2025-11-16 by Maintainer
# CONTEXT: Fixes inbox routing bug where messages TO Auditor were generating processing
#          instructions for Skeptic, Architect, etc. See response-20251115-192500-from-skeptic-final.md

set -euo pipefail

# ============================================================================
# Input validation
# ============================================================================

if [ $# -ne 2 ]; then
    echo "ERROR: Usage: inbox-routing-filter.sh <message_file> <current_persona>" >&2
    echo "SKIP: Invalid arguments"
    exit 1
fi

message_file="$1"
current_persona="$2"

# Validate message file exists
if [ ! -f "$message_file" ]; then
    echo "ERROR: Message file not found: $message_file" >&2
    echo "SKIP: File not found"
    exit 1
fi

# Validate persona is non-empty
if [ -z "$current_persona" ]; then
    echo "ERROR: current_persona cannot be empty" >&2
    echo "SKIP: Empty persona"
    exit 1
fi

# ============================================================================
# Metadata extraction
# ============================================================================

# Extract 'from' field from YAML frontmatter
# Pattern: Look for "from: value" in frontmatter (between --- delimiters)
# Using sed because yq may not be available in all environments
extract_from=$(sed -n '/^---$/,/^---$/p' "$message_file" | grep "^from:" | head -1 | sed 's/^from:[[:space:]]*//' | tr -d '"' | tr -d "'")

# Extract 'to' field from YAML frontmatter
extract_to=$(sed -n '/^---$/,/^---$/p' "$message_file" | grep "^to:" | head -1 | sed 's/^to:[[:space:]]*//' | tr -d '"' | tr -d "'")

# Handle cases where metadata is missing
if [ -z "$extract_from" ]; then
    echo "WARN: No 'from' field found in message metadata" >&2
    echo "SKIP: Missing metadata (from)"
    exit 1
fi

if [ -z "$extract_to" ]; then
    echo "WARN: No 'to' field found in message metadata" >&2
    echo "SKIP: Missing metadata (to)"
    exit 1
fi

# ============================================================================
# Routing logic
# ============================================================================

# RULE 1: Don't process own messages
# Rationale: Sender shouldn't process their own sent message
# Example: Skeptic shouldn't process msg-skeptic-phase2-complete-to-auditor (Skeptic sent it)
if [ "$extract_from" == "$current_persona" ]; then
    echo "SKIP: Message is FROM $current_persona (sender can't process own message)"
    exit 1
fi

# RULE 2: Process messages addressed to current persona
# Rationale: Direct recipient should process message
# Example: Auditor should process msg-skeptic-phase2-complete-to-auditor (addressed to Auditor)
if [ "$extract_to" == "$current_persona" ]; then
    echo "ROUTE: Message is TO $current_persona (direct recipient)"
    exit 0
fi

# RULE 3: Process broadcast messages (to: daemon)
# Rationale: Messages addressed to "daemon" should be processed by any persona
# Example: Human → daemon messages should be handled by whoever activates
if [ "$extract_to" == "daemon" ]; then
    echo "ROUTE: Message is broadcast (to:daemon), processing as $current_persona"
    exit 0
fi

# RULE 4: Process multi-recipient messages
# Rationale: Messages with multiple recipients need proper list parsing
# Supports formats: [persona1, persona2] or persona1, persona2
# Uses exact matching to avoid substring false positives (e.g., "optimizer" in "experimenter-optimizer-project")

# Check for YAML array format: [persona1, persona2]
if echo "$extract_to" | grep -q '^\[.*\]$'; then
    # Extract array elements, trim whitespace, check for exact match
    if echo "$extract_to" | tr -d '[]' | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -Fxq "$current_persona"; then
        echo "ROUTE: Multi-recipient message includes $current_persona (YAML array)"
        exit 0
    fi
fi

# Check for comma-separated format: persona1, persona2
if echo "$extract_to" | grep -q ','; then
    # Split by comma, trim whitespace, check for exact match
    if echo "$extract_to" | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -Fxq "$current_persona"; then
        echo "ROUTE: Multi-recipient message includes $current_persona (comma-separated)"
        exit 0
    fi
fi

# RULE 5: Message not for this persona
# All routing rules failed, message is for someone else
echo "SKIP: Message is for '$extract_to', not for '$current_persona'"
exit 1
