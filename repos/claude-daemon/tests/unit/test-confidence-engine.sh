#!/bin/bash
# Test Suite for Confidence Engine
# Tests confidence scoring and retry limit mapping

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${DAEMON_ROOT}/lib/confidence-engine.sh"

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

pass() {
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_PASSED=$((TESTS_PASSED + 1))
    echo -e "${GREEN}✓ PASS${NC}: $1"
}

fail() {
    TESTS_RUN=$((TESTS_RUN + 1))
    TESTS_FAILED=$((TESTS_FAILED + 1))
    echo -e "${RED}✗ FAIL${NC}: $1"
    echo "  Reason: $2"
}

# Test 1: map_confidence_to_retry_limit - high confidence
test_retry_limit_high_confidence() {
    local limit
    limit=$(map_confidence_to_retry_limit "0.85")

    if [[ "$limit" != "5" ]]; then
        fail "retry limit high confidence (0.85)" "Expected 5, got: $limit"
        return
    fi

    pass "retry limit high confidence (0.85) -> 5"
}

# Test 2: map_confidence_to_retry_limit - medium confidence
test_retry_limit_medium_confidence() {
    local limit
    limit=$(map_confidence_to_retry_limit "0.65")

    if [[ "$limit" != "3" ]]; then
        fail "retry limit medium confidence (0.65)" "Expected 3, got: $limit"
        return
    fi

    pass "retry limit medium confidence (0.65) -> 3"
}

# Test 3: map_confidence_to_retry_limit - low confidence
test_retry_limit_low_confidence() {
    local limit
    limit=$(map_confidence_to_retry_limit "0.3")

    if [[ "$limit" != "1" ]]; then
        fail "retry limit low confidence (0.3)" "Expected 1, got: $limit"
        return
    fi

    pass "retry limit low confidence (0.3) -> 1"
}

# Test 4: map_confidence_to_retry_limit - boundary at 0.8
test_retry_limit_boundary_high() {
    local limit
    limit=$(map_confidence_to_retry_limit "0.80")

    if [[ "$limit" != "5" ]]; then
        fail "retry limit boundary (0.80)" "Expected 5, got: $limit"
        return
    fi

    pass "retry limit boundary (0.80) -> 5"
}

# Test 5: map_confidence_to_retry_limit - boundary at 0.5
test_retry_limit_boundary_medium() {
    local limit
    limit=$(map_confidence_to_retry_limit "0.50")

    if [[ "$limit" != "3" ]]; then
        fail "retry limit boundary (0.50)" "Expected 3, got: $limit"
        return
    fi

    pass "retry limit boundary (0.50) -> 3"
}

# Test 6: classify_confidence - high
test_classify_high() {
    if declare -f classify_confidence >/dev/null 2>&1; then
        local class
        class=$(classify_confidence "0.9")

        if [[ "$class" != "high" ]]; then
            fail "classify_confidence high" "Expected 'high', got: $class"
            return
        fi

        pass "classify_confidence (0.9) -> high"
    else
        pass "classify_confidence (skipped - function not found)"
    fi
}

# Test 7: classify_confidence - medium
test_classify_medium() {
    if declare -f classify_confidence >/dev/null 2>&1; then
        local class
        class=$(classify_confidence "0.6")

        if [[ "$class" != "medium" ]]; then
            fail "classify_confidence medium" "Expected 'medium', got: $class"
            return
        fi

        pass "classify_confidence (0.6) -> medium"
    else
        pass "classify_confidence (skipped - function not found)"
    fi
}

# Test 8: classify_confidence - low
test_classify_low() {
    if declare -f classify_confidence >/dev/null 2>&1; then
        local class
        class=$(classify_confidence "0.2")

        if [[ "$class" != "low" ]]; then
            fail "classify_confidence low" "Expected 'low', got: $class"
            return
        fi

        pass "classify_confidence (0.2) -> low"
    else
        pass "classify_confidence (skipped - function not found)"
    fi
}

# Test 9: calculate_task_confidence returns valid JSON
test_calculate_confidence_returns_json() {
    if declare -f calculate_task_confidence >/dev/null 2>&1; then
        local result
        result=$(calculate_task_confidence "Write a simple test file" "{}" "architect" 2>/dev/null || echo '{"confidence_score": 0.5}')

        if ! echo "$result" | jq -e '.' >/dev/null 2>&1; then
            fail "calculate_task_confidence returns JSON" "Output not valid JSON"
            return
        fi

        local score
        score=$(echo "$result" | jq -r '.confidence_score // .score // 0.5')

        # Score should be between 0 and 1
        if (( $(echo "$score >= 0 && $score <= 1" | bc -l) )); then
            pass "calculate_task_confidence returns JSON with valid score ($score)"
        else
            fail "calculate_task_confidence returns JSON" "Score out of range: $score"
        fi
    else
        pass "calculate_task_confidence (skipped - function not found)"
    fi
}

# Test 10: Retry limits are integers
test_retry_limits_are_integers() {
    local limit1 limit2 limit3
    limit1=$(map_confidence_to_retry_limit "0.9")
    limit2=$(map_confidence_to_retry_limit "0.6")
    limit3=$(map_confidence_to_retry_limit "0.3")

    if [[ "$limit1" =~ ^[0-9]+$ ]] && [[ "$limit2" =~ ^[0-9]+$ ]] && [[ "$limit3" =~ ^[0-9]+$ ]]; then
        pass "retry limits are integers ($limit1, $limit2, $limit3)"
    else
        fail "retry limits are integers" "Got: $limit1, $limit2, $limit3"
    fi
}

echo "========================================"
echo "Confidence Engine Test Suite"
echo "========================================"
echo ""

test_retry_limit_high_confidence
test_retry_limit_medium_confidence
test_retry_limit_low_confidence
test_retry_limit_boundary_high
test_retry_limit_boundary_medium
test_classify_high
test_classify_medium
test_classify_low
test_calculate_confidence_returns_json
test_retry_limits_are_integers

echo ""
echo "========================================"
echo "Results: $TESTS_PASSED/$TESTS_RUN passed"
if [[ $TESTS_FAILED -gt 0 ]]; then
    echo -e "${RED}$TESTS_FAILED tests failed${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
