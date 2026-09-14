#!/bin/bash
# Test Suite for Retry Orchestrator
# Tests the retry-until-verified loop with episode tracking

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Source dependencies first
source "${DAEMON_ROOT}/lib/confidence-engine.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/episodic-memory.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/checkpoint-manager.sh" 2>/dev/null || true
source "${DAEMON_ROOT}/lib/retry-orchestrator.sh"

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

# Test 1: execute_with_retry function exists
test_function_exists() {
    if declare -f execute_with_retry >/dev/null 2>&1; then
        pass "execute_with_retry function exists"
    else
        fail "execute_with_retry function exists" "Function not found"
    fi
}

# Test 2: Episode variable is properly initialized
test_episode_variable_pattern() {
    # Check the source code for proper episode handling
    if grep -q 'local episode=""' "${DAEMON_ROOT}/lib/retry-orchestrator.sh"; then
        pass "episode variable properly initialized"
    else
        fail "episode variable properly initialized" "Pattern not found"
    fi
}

# Test 3: Episode accumulation pattern (capture return value)
test_episode_accumulation_pattern() {
    # Check that add_action_to_episode captures return value
    if grep -q 'episode=\$(add_action_to_episode' "${DAEMON_ROOT}/lib/retry-orchestrator.sh"; then
        pass "episode accumulation captures return value"
    else
        fail "episode accumulation captures return value" "Pattern 'episode=\$(add_action_to_episode' not found"
    fi
}

# Test 4: save_episode is called
test_save_episode_called() {
    if grep -q 'save_episode "\$episode"' "${DAEMON_ROOT}/lib/retry-orchestrator.sh"; then
        pass "save_episode is called"
    else
        fail "save_episode is called" "save_episode call not found"
    fi
}

# Test 5: close_episode captures return value
test_close_episode_pattern() {
    if grep -q 'episode=\$(close_episode' "${DAEMON_ROOT}/lib/retry-orchestrator.sh"; then
        pass "close_episode captures return value"
    else
        fail "close_episode captures return value" "Pattern not found"
    fi
}

# Test 6: Task is passed to execute_task_action
test_task_passed_to_execute() {
    if grep -q 'execute_task_action "\$persona" "\$task_title"' "${DAEMON_ROOT}/lib/retry-orchestrator.sh"; then
        pass "task passed to execute_task_action"
    else
        fail "task passed to execute_task_action" "Pattern not found"
    fi
}

# Test 7: Retry orchestrator logs are present
test_logging_present() {
    if grep -q 'RETRY_ORCHESTRATOR:' "${DAEMON_ROOT}/lib/retry-orchestrator.sh"; then
        pass "RETRY_ORCHESTRATOR logging present"
    else
        fail "RETRY_ORCHESTRATOR logging present" "No logging found"
    fi
}

# Test 8: Confidence scoring integration
test_confidence_integration() {
    if grep -q 'calculate_task_confidence\|confidence_score' "${DAEMON_ROOT}/lib/retry-orchestrator.sh"; then
        pass "confidence scoring integrated"
    else
        fail "confidence scoring integrated" "No confidence integration found"
    fi
}

# Test 9: Episodic memory library sourced
test_episodic_memory_sourced() {
    if grep -q 'source.*episodic-memory.sh' "${DAEMON_ROOT}/lib/retry-orchestrator.sh"; then
        pass "episodic-memory.sh is sourced"
    else
        fail "episodic-memory.sh is sourced" "Source statement not found"
    fi
}

# Test 10: Retry loop has max attempts limit
test_max_attempts_limit() {
    if grep -q 'max_attempts\|retry_limit' "${DAEMON_ROOT}/lib/retry-orchestrator.sh"; then
        pass "max attempts limit exists"
    else
        fail "max attempts limit exists" "No limit found"
    fi
}

# Test 11: Episode tracking uses full JSON, not just ID
test_episode_json_not_id() {
    # Check we're NOT just using episode_id for function calls (old broken pattern)
    if grep -q 'add_action_to_episode "\$episode_id"' "${DAEMON_ROOT}/lib/retry-orchestrator.sh" 2>/dev/null; then
        fail "episode tracking uses full JSON" "Found old pattern: episode_id being passed instead of episode"
    else
        pass "episode tracking uses full JSON (not just ID)"
    fi
}

# Test 12: Check both success and failure paths record actions
test_both_paths_record_actions() {
    local success_count failure_count
    success_count=$(grep -c 'execution_success' "${DAEMON_ROOT}/lib/retry-orchestrator.sh" 2>/dev/null || echo "0")
    failure_count=$(grep -c 'execution_failure' "${DAEMON_ROOT}/lib/retry-orchestrator.sh" 2>/dev/null || echo "0")

    if [[ "$success_count" -ge 1 ]] && [[ "$failure_count" -ge 1 ]]; then
        pass "both success and failure paths record actions"
    else
        fail "both paths record actions" "success=$success_count failure=$failure_count"
    fi
}

echo "========================================"
echo "Retry Orchestrator Test Suite"
echo "========================================"
echo ""

test_function_exists
test_episode_variable_pattern
test_episode_accumulation_pattern
test_save_episode_called
test_close_episode_pattern
test_task_passed_to_execute
test_logging_present
test_confidence_integration
test_episodic_memory_sourced
test_max_attempts_limit
test_episode_json_not_id
test_both_paths_record_actions

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
