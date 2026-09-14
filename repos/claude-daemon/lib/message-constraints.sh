#!/bin/bash
#
# Message Constraints Library
# Enforces token efficiency guidelines for all persona messages
#
# Created: 2025-11-07 by Experimenter
# Part of: Token Efficiency Optimization (Part 2 - Implementation)
# Reference: docs/efficient-communication.md, docs/token-usage-analysis.md
#
# USAGE:
#   source lib/message-constraints.sh
#   validate_message_file "path/to/message.md" || handle_error
#   validate_message_content "$message_content" || handle_error
#

# Configuration (matches docs/efficient-communication.md requirements)
readonly MAX_LINES=200
readonly RECOMMENDED_LINES=130
readonly SUMMARY_MAX=20
readonly DETAILS_MAX=80
readonly META_MAX=20
readonly NEXT_MAX=10

# Colors for output (if terminal supports)
if [ -t 1 ]; then
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    GREEN='\033[0;32m'
    NC='\033[0m' # No Color
else
    RED=''
    YELLOW=''
    GREEN=''
    NC=''
fi

#
# validate_message_file: Validate message file against constraints
# Args: $1 = path to message file
# Returns: 0 if valid, 1 if invalid
# Prints: Validation results
#
validate_message_file() {
    local file_path="$1"

    if [ ! -f "$file_path" ]; then
        echo -e "${RED}ERROR: File not found: $file_path${NC}" >&2
        return 1
    fi

    local content
    content=$(cat "$file_path")

    validate_message_content "$content"
}

#
# validate_message_content: Validate message content against constraints
# Args: $1 = message content (multiline string)
# Returns: 0 if valid, 1 if invalid
# Prints: Validation results
#
validate_message_content() {
    local content="$1"
    local validation_failed=0

    # Count total lines (excluding empty lines at start/end)
    local total_lines
    total_lines=$(echo "$content" | sed '/./,$!d' | wc -l)

    # Check hard limit (200 lines)
    if [ "$total_lines" -gt "$MAX_LINES" ]; then
        echo -e "${RED}✗ HARD LIMIT EXCEEDED: $total_lines lines (max: $MAX_LINES)${NC}" >&2
        validation_failed=1
    elif [ "$total_lines" -gt "$RECOMMENDED_LINES" ]; then
        echo -e "${YELLOW}⚠ WARNING: $total_lines lines exceeds recommended $RECOMMENDED_LINES (still under $MAX_LINES hard limit)${NC}" >&2
    else
        echo -e "${GREEN}✓ Line count OK: $total_lines lines (recommended: ≤$RECOMMENDED_LINES, max: $MAX_LINES)${NC}"
    fi

    # Check for required summary section
    if ! echo "$content" | grep -qi "^##\? *summary"; then
        echo -e "${YELLOW}⚠ WARNING: No Summary section found (recommended for all messages)${NC}" >&2
    else
        # Validate summary length if present
        local summary_lines
        summary_lines=$(extract_section_lines "$content" "summary")

        if [ "$summary_lines" -gt "$SUMMARY_MAX" ]; then
            echo -e "${YELLOW}⚠ WARNING: Summary section is $summary_lines lines (recommended: ≤$SUMMARY_MAX)${NC}" >&2
        else
            echo -e "${GREEN}✓ Summary section OK: $summary_lines lines (max: $SUMMARY_MAX)${NC}"
        fi
    fi

    # Check for meta-analysis length (if present)
    if echo "$content" | grep -qi "^##\? *meta"; then
        local meta_lines
        meta_lines=$(extract_section_lines "$content" "meta")

        if [ "$meta_lines" -gt "$META_MAX" ]; then
            echo -e "${YELLOW}⚠ WARNING: Meta-analysis section is $meta_lines lines (recommended: ≤$META_MAX)${NC}" >&2
        else
            echo -e "${GREEN}✓ Meta-analysis section OK: $meta_lines lines (max: $META_MAX)${NC}"
        fi
    fi

    # Check for next steps section
    if echo "$content" | grep -qi "^##\? *next"; then
        local next_lines
        next_lines=$(extract_section_lines "$content" "next")

        if [ "$next_lines" -gt "$NEXT_MAX" ]; then
            echo -e "${YELLOW}⚠ WARNING: Next steps section is $next_lines lines (recommended: ≤$NEXT_MAX)${NC}" >&2
        else
            echo -e "${GREEN}✓ Next steps section OK: $next_lines lines (max: $NEXT_MAX)${NC}"
        fi
    fi

    # Check for common anti-patterns
    check_antipatterns "$content"

    return $validation_failed
}

#
# extract_section_lines: Count lines in a markdown section
# Args: $1 = content, $2 = section name (case-insensitive)
# Returns: Number of lines in section
#
extract_section_lines() {
    local content="$1"
    local section_name="$2"

    # Extract from section header to next header or end
    echo "$content" | awk -v section="$section_name" '
        BEGIN { IGNORECASE=1; in_section=0; count=0 }
        /^##? / {
            if (in_section) exit
            if ($0 ~ section) { in_section=1; next }
        }
        in_section { count++ }
        END { print count }
    '
}

#
# check_antipatterns: Detect common verbose patterns
# Args: $1 = content
# Prints: Warnings for detected anti-patterns
#
check_antipatterns() {
    local content="$1"

    # Check for excessive persona self-references (e.g., "As Experimenter, I...")
    local persona_refs
    persona_refs=$(echo "$content" | grep -ci "as \(experimenter\|skeptic\|architect\|auditor\|optimizer\|maintainer\)")

    if [ "$persona_refs" -gt 3 ]; then
        echo -e "${YELLOW}⚠ Anti-pattern: Excessive persona self-references ($persona_refs found, reduce redundancy)${NC}" >&2
    fi

    # Check for long code blocks (>30 lines)
    local code_block_lines
    code_block_lines=$(echo "$content" | awk '/^```/,/^```/ {if (!/^```/) count++} END {print count+0}')

    if [ "$code_block_lines" -gt 30 ]; then
        echo -e "${YELLOW}⚠ Anti-pattern: Large code blocks ($code_block_lines lines, consider file:line references instead)${NC}" >&2
    fi

    # Check for repetitive acknowledgments
    if echo "$content" | grep -qi "I want to share.*I want to.*I want to"; then
        echo -e "${YELLOW}⚠ Anti-pattern: Repetitive phrasing detected (vary sentence structure)${NC}" >&2
    fi
}

#
# get_efficiency_score: Calculate message efficiency score (0-100)
# Args: $1 = message content
# Returns: Score (stdout)
#
get_efficiency_score() {
    local content="$1"
    local score=100

    # Count total lines
    local total_lines
    total_lines=$(echo "$content" | sed '/./,$!d' | wc -l)

    # Penalize length (0-50 points)
    if [ "$total_lines" -gt "$MAX_LINES" ]; then
        score=$((score - 50))
    elif [ "$total_lines" -gt "$RECOMMENDED_LINES" ]; then
        local excess=$((total_lines - RECOMMENDED_LINES))
        local penalty=$(( (excess * 30) / (MAX_LINES - RECOMMENDED_LINES) ))
        score=$((score - penalty))
    fi

    # Bonus for summary (0-20 points)
    if echo "$content" | grep -qi "^##\? *summary"; then
        score=$((score + 10))

        local summary_lines
        summary_lines=$(extract_section_lines "$content" "summary")
        if [ "$summary_lines" -le "$SUMMARY_MAX" ]; then
            score=$((score + 10))
        fi
    else
        score=$((score - 20))
    fi

    # Bonus for next steps (0-10 points)
    if echo "$content" | grep -qi "^##\? *next"; then
        score=$((score + 10))
    fi

    # Cap at 0-100
    if [ "$score" -lt 0 ]; then score=0; fi
    if [ "$score" -gt 100 ]; then score=100; fi

    echo "$score"
}

#
# format_efficiency_report: Generate efficiency report for message
# Args: $1 = message content
# Prints: Formatted report
#
format_efficiency_report() {
    local content="$1"

    local total_lines
    total_lines=$(echo "$content" | sed '/./,$!d' | wc -l)

    local score
    score=$(get_efficiency_score "$content")

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "MESSAGE EFFICIENCY REPORT"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Total lines: $total_lines (recommended: ≤$RECOMMENDED_LINES, max: $MAX_LINES)"
    echo "Efficiency score: $score/100"
    echo ""

    validate_message_content "$content" > /dev/null 2>&1
    local validation_result=$?

    if [ "$validation_result" -eq 0 ]; then
        echo -e "${GREEN}✓ PASS: Message meets efficiency guidelines${NC}"
    else
        echo -e "${RED}✗ FAIL: Message violates efficiency guidelines${NC}"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Export functions for use in other scripts
export -f validate_message_file
export -f validate_message_content
export -f get_efficiency_score
export -f format_efficiency_report
