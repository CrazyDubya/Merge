#!/bin/bash
# Test Suite for Episodic Memory
# Tests the functional chaining API for episode tracking

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${DAEMON_ROOT}/lib/episodic-memory.sh"

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

# Test 1: create_episode returns valid JSON
test_create_episode_returns_json() {
    local episode
    episode=$(create_episode "test-goal-123" "test-context")

    if ! echo "$episode" | jq -e '.' >/dev/null 2>&1; then
        fail "create_episode returns valid JSON" "Output is not valid JSON"
        return
    fi

    local episode_id
    episode_id=$(echo "$episode" | jq -r '.episode_id')

    if [[ ! "$episode_id" =~ ^ep_ ]]; then
        fail "create_episode returns valid JSON" "episode_id should start with 'ep_', got: $episode_id"
        return
    fi

    pass "create_episode returns valid JSON"
}

# Test 2: create_episode has correct structure
test_create_episode_structure() {
    local episode
    episode=$(create_episode "goal-abc" "workflow")

    local status goal_id actions_count
    status=$(echo "$episode" | jq -r '.status')
    goal_id=$(echo "$episode" | jq -r '.goal_id')
    actions_count=$(echo "$episode" | jq '.actions | length')

    if [[ "$status" != "active" ]]; then
        fail "create_episode structure" "status should be 'active', got: $status"
        return
    fi

    if [[ "$goal_id" != "goal-abc" ]]; then
        fail "create_episode structure" "goal_id mismatch"
        return
    fi

    if [[ "$actions_count" != "0" ]]; then
        fail "create_episode structure" "actions should be empty array"
        return
    fi

    pass "create_episode structure"
}

# Test 3: add_action_to_episode accumulates actions
test_add_action_accumulates() {
    local episode
    episode=$(create_episode "test-goal" "ctx")

    local action1='{"type":"task_start","task":"test1"}'
    local action2='{"type":"task_end","task":"test1"}'

    episode=$(add_action_to_episode "$episode" "$action1")
    episode=$(add_action_to_episode "$episode" "$action2")

    local count
    count=$(echo "$episode" | jq '.actions | length')

    if [[ "$count" -ne 2 ]]; then
        fail "add_action accumulates" "Expected 2 actions, got $count"
        return
    fi

    # Verify first action type
    local first_type
    first_type=$(echo "$episode" | jq -r '.actions[0].type')
    if [[ "$first_type" != "task_start" ]]; then
        fail "add_action accumulates" "First action type wrong: $first_type"
        return
    fi

    pass "add_action accumulates"
}

# Test 4: add_action adds timestamp
test_add_action_adds_timestamp() {
    local episode
    episode=$(create_episode "test" "ctx")

    local action='{"type":"test_action"}'
    episode=$(add_action_to_episode "$episode" "$action")

    local timestamp
    timestamp=$(echo "$episode" | jq -r '.actions[0].timestamp')

    if [[ "$timestamp" == "null" ]] || [[ -z "$timestamp" ]]; then
        fail "add_action adds timestamp" "No timestamp found"
        return
    fi

    pass "add_action adds timestamp"
}

# Test 5: close_episode sets status
test_close_episode_sets_status() {
    local episode
    episode=$(create_episode "test" "ctx")
    episode=$(close_episode "$episode" "success")

    local status
    status=$(echo "$episode" | jq -r '.status')

    if [[ "$status" != "closed" ]]; then
        fail "close_episode sets status" "Expected 'closed', got: $status"
        return
    fi

    pass "close_episode sets status"
}

# Test 6: close_episode sets outcome
test_close_episode_sets_outcome() {
    local episode
    episode=$(create_episode "test" "ctx")
    episode=$(close_episode "$episode" "failure")

    local outcome
    outcome=$(echo "$episode" | jq -r '.outcome')

    if [[ "$outcome" != "failure" ]]; then
        fail "close_episode sets outcome" "Expected 'failure', got: $outcome"
        return
    fi

    pass "close_episode sets outcome"
}

# Test 7: Full lifecycle
test_full_lifecycle() {
    local episode
    episode=$(create_episode "lifecycle-test" "integration")

    # Add actions
    episode=$(add_action_to_episode "$episode" '{"type":"start"}')
    episode=$(add_action_to_episode "$episode" '{"type":"work"}')
    episode=$(add_action_to_episode "$episode" '{"type":"complete"}')

    # Close
    episode=$(close_episode "$episode" "success")

    # Verify
    local status actions_count outcome
    status=$(echo "$episode" | jq -r '.status')
    actions_count=$(echo "$episode" | jq '.actions | length')
    outcome=$(echo "$episode" | jq -r '.outcome')

    if [[ "$status" != "closed" ]] || [[ "$actions_count" -ne 3 ]] || [[ "$outcome" != "success" ]]; then
        fail "full lifecycle" "status=$status actions=$actions_count outcome=$outcome"
        return
    fi

    pass "full lifecycle"
}

# Test 8: save_episode creates file
test_save_episode() {
    local test_episodes_file="/tmp/test-episodes-$$.jsonl"

    # Temporarily override DAEMON_ROOT
    local orig_daemon_root="$DAEMON_ROOT"
    mkdir -p /tmp/test-daemon-$$/memory
    DAEMON_ROOT="/tmp/test-daemon-$$"

    local episode
    episode=$(create_episode "save-test" "ctx")
    episode=$(close_episode "$episode" "success")

    save_episode "$episode"

    if [[ -f "${DAEMON_ROOT}/memory/episodes.jsonl" ]]; then
        local line_count
        line_count=$(wc -l < "${DAEMON_ROOT}/memory/episodes.jsonl")
        if [[ "$line_count" -ge 1 ]]; then
            pass "save_episode creates file"
        else
            fail "save_episode creates file" "File empty"
        fi
    else
        fail "save_episode creates file" "File not created"
    fi

    # Cleanup
    rm -rf "/tmp/test-daemon-$$"
    DAEMON_ROOT="$orig_daemon_root"
}

echo "========================================"
echo "Episodic Memory Test Suite"
echo "========================================"
echo ""

test_create_episode_returns_json
test_create_episode_structure
test_add_action_accumulates
test_add_action_adds_timestamp
test_close_episode_sets_status
test_close_episode_sets_outcome
test_full_lifecycle
test_save_episode

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
