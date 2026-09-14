#!/bin/bash
# validate-consolidation.sh
# Validates consolidation documents against Skeptic's quality checklist
#
# This script automatically checks if START-HERE documents:
# 1. Have all required frontmatter fields
# 2. Include 'consolidates:' field with valid message IDs
# 3. Reference messages that actually exist
# 4. Can be processed by archive script
#
# Usage:
#   ./validate-consolidation.sh                    # Check all START-HERE docs
#   ./validate-consolidation.sh path/to/doc.md     # Check specific doc
#
# Exit codes:
#   0 - All validations passed
#   1 - One or more validations failed

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
INBOX_DIR="${DAEMON_ROOT}/inbox/human/unread"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track validation results
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Required frontmatter fields for START-HERE docs
REQUIRED_FIELDS=("from" "to" "timestamp" "priority" "tags" "consolidates" "message_id")

echo -e "${BLUE}=== Consolidation Document Validation ===${NC}"
echo ""

# Function to validate a single START-HERE document
validate_document() {
    local DOC=$1
    local DOC_NAME=$(basename "$DOC")
    local ERRORS=0

    echo -e "${BLUE}Validating: $DOC_NAME${NC}"

    # Check 1: File exists and is readable
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    if [ ! -f "$DOC" ] || [ ! -r "$DOC" ]; then
        echo -e "  ${RED}✗${NC} File not found or not readable"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        ERRORS=$((ERRORS + 1))
        return 1
    else
        echo -e "  ${GREEN}✓${NC} File exists and readable"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    fi

    # Check 2: Has YAML frontmatter
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    if ! grep -q "^---$" "$DOC"; then
        echo -e "  ${RED}✗${NC} Missing YAML frontmatter (no '---' delimiter)"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        ERRORS=$((ERRORS + 1))
    else
        echo -e "  ${GREEN}✓${NC} Has YAML frontmatter"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    fi

    # Check 3: All required fields present
    for FIELD in "${REQUIRED_FIELDS[@]}"; do
        TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
        if grep -q "^${FIELD}:" "$DOC"; then
            echo -e "  ${GREEN}✓${NC} Field '$FIELD' present"
            PASSED_CHECKS=$((PASSED_CHECKS + 1))
        else
            echo -e "  ${RED}✗${NC} Field '$FIELD' MISSING"
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
            ERRORS=$((ERRORS + 1))
        fi
    done

    # Check 4: consolidates field has valid format
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    CONSOLIDATES_LINE=$(grep "^consolidates:" "$DOC" 2>/dev/null || true)
    if [ -n "$CONSOLIDATES_LINE" ]; then
        # Check if it's a YAML array [...]
        if echo "$CONSOLIDATES_LINE" | grep -q "\[.*\]"; then
            echo -e "  ${GREEN}✓${NC} consolidates field has array format [...]"
            PASSED_CHECKS=$((PASSED_CHECKS + 1))
        else
            echo -e "  ${RED}✗${NC} consolidates field not in array format (expected: [msg1, msg2])"
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
            ERRORS=$((ERRORS + 1))
        fi
    else
        # Already caught by required fields check
        echo -e "  ${YELLOW}!${NC} Skipping format check (field missing)"
    fi

    # Check 5: Referenced messages exist
    if [ -n "$CONSOLIDATES_LINE" ]; then
        # Extract message IDs
        MESSAGES=$(echo "$CONSOLIDATES_LINE" | sed 's/consolidates: \[//' | sed 's/\]//' | tr ',' '\n')

        local MSG_COUNT=0
        local MSG_FOUND=0
        local MSG_MISSING=0

        while read -r MSG; do
            MSG=$(echo "$MSG" | xargs)  # Trim whitespace

            if [ -z "$MSG" ]; then
                continue
            fi

            MSG_COUNT=$((MSG_COUNT + 1))

            # Add .md extension if not present
            if [[ ! "$MSG" =~ \.md$ ]]; then
                MSG="${MSG}.md"
            fi

            TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

            # Check if message exists in inbox
            if [ -f "${INBOX_DIR}/${MSG}" ]; then
                local LINES=$(wc -l < "${INBOX_DIR}/${MSG}")
                echo -e "  ${GREEN}✓${NC} Message '$MSG' exists ($LINES lines)"
                PASSED_CHECKS=$((PASSED_CHECKS + 1))
                MSG_FOUND=$((MSG_FOUND + 1))
            else
                echo -e "  ${RED}✗${NC} Message '$MSG' NOT FOUND in inbox"
                FAILED_CHECKS=$((FAILED_CHECKS + 1))
                MSG_MISSING=$((MSG_MISSING + 1))
                ERRORS=$((ERRORS + 1))
            fi
        done <<< "$MESSAGES"

        echo -e "  ${BLUE}→${NC} consolidates: $MSG_FOUND found, $MSG_MISSING missing (of $MSG_COUNT total)"
    fi

    # Check 6: Archive script can process this doc
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    if [ -n "$CONSOLIDATES_LINE" ] && [ "$ERRORS" -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} Document should be processable by archive script"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    elif [ -n "$CONSOLIDATES_LINE" ]; then
        echo -e "  ${YELLOW}!${NC} Document has errors, archive script may fail"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        ERRORS=$((ERRORS + 1))
    else
        echo -e "  ${RED}✗${NC} Document cannot be processed (missing consolidates field)"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        ERRORS=$((ERRORS + 1))
    fi

    echo ""

    # Return status based on errors
    if [ "$ERRORS" -eq 0 ]; then
        return 0
    else
        return 1
    fi
}

# Main logic
if [ $# -eq 0 ]; then
    # No arguments - validate all START-HERE docs in inbox
    cd "$INBOX_DIR"

    START_HERE_DOCS=$(ls -1 00-START-HERE-*.md 2>/dev/null || true)

    if [ -z "$START_HERE_DOCS" ]; then
        echo -e "${YELLOW}No START-HERE documents found in ${INBOX_DIR}${NC}"
        echo ""
        exit 0
    fi

    DOC_COUNT=0
    DOCS_PASSED=0
    DOCS_FAILED=0

    while read DOC; do
        DOC_COUNT=$((DOC_COUNT + 1))
        if validate_document "$DOC"; then
            DOCS_PASSED=$((DOCS_PASSED + 1))
        else
            DOCS_FAILED=$((DOCS_FAILED + 1))
        fi
    done <<< "$START_HERE_DOCS"

    # Summary
    echo -e "${BLUE}=== Validation Summary ===${NC}"
    echo ""
    echo -e "Documents validated: ${BLUE}$DOC_COUNT${NC}"
    echo -e "  ${GREEN}✓${NC} Passed: $DOCS_PASSED"
    if [ "$DOCS_FAILED" -gt 0 ]; then
        echo -e "  ${RED}✗${NC} Failed: $DOCS_FAILED"
    fi
    echo ""
    echo -e "Total checks: ${BLUE}$TOTAL_CHECKS${NC}"
    echo -e "  ${GREEN}✓${NC} Passed: $PASSED_CHECKS"
    if [ "$FAILED_CHECKS" -gt 0 ]; then
        echo -e "  ${RED}✗${NC} Failed: $FAILED_CHECKS"
    fi
    echo ""

    if [ "$DOCS_FAILED" -gt 0 ]; then
        echo -e "${YELLOW}⚠ Some documents have validation errors${NC}"
        echo ""
        echo "Fix errors by:"
        echo "1. Adding missing frontmatter fields"
        echo "2. Ensuring consolidates field uses array format: [msg1, msg2]"
        echo "3. Verifying all referenced messages exist in inbox"
        echo ""
        echo "See: docs/consolidation-quality-checklist.md"
        exit 1
    else
        echo -e "${GREEN}✓ All documents passed validation!${NC}"
        exit 0
    fi

else
    # Validate specific document
    DOC="$1"

    if [ ! -f "$DOC" ]; then
        echo -e "${RED}ERROR: File not found: $DOC${NC}"
        exit 1
    fi

    if validate_document "$DOC"; then
        echo -e "${GREEN}✓ Document passed all validations${NC}"
        exit 0
    else
        echo -e "${RED}✗ Document has validation errors${NC}"
        echo ""
        echo "See: docs/consolidation-quality-checklist.md"
        exit 1
    fi
fi
