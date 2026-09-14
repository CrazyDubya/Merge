#!/bin/bash
# Test Suite for Task State Management Library
#
# Validates task state transitions, persona matching, and error handling

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get daemon root
DAEMON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASKS_DIR="${DAEMON_ROOT}/tasks"

# Source the library we're testing
source "${DAEMON_ROOT}/lib/task-state-management.sh"

# Mock log function for testing
log() {
    local level="$1"
    shift
    echo "[TEST-LOG][$level] $*" >&2
}

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Test result tracking
print_test_result() {
    local test_name="$1"
    local result="$2"
    local message="$3"

    TESTS_RUN=$((TESTS_RUN + 1))

    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✓ PASS${NC}: $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗ FAIL${NC}: $test_name"
        echo "  Reason: $message"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Create temporary test directory and queue file
create_test_queue() {
    local test_dir
    test_dir=$(mktemp -d)
    local test_queue="${test_dir}/queue.md"

    cat > "$test_queue" << 'EOF'
# Task Queue

## Pending Tasks

- [ ] [OPTIMIZER] Test optimizer task
- [ ] [ARCHITECT] Test architect task
- [ ] Untagged test task
- [ ] [MAINTAINER] Test maintainer task

## Completed Tasks

- [x] Previous completed task

EOF

    echo "$test_dir"
}

# Test 1: extract_persona_tag function
test_extract_persona_tag() {
    local test_name="extract_persona_tag: various formats"

    local result1=$(extract_persona_tag "[OPTIMIZER] Test task")
    local result2=$(extract_persona_tag "[ARCHITECT] Another task")
    local result3=$(extract_persona_tag "Untagged task")
    local result4=$(extract_persona_tag "[optimizer] Lowercase tag")

    if [ "$result1" = "optimizer" ] && [ "$result2" = "architect" ] && [ -z "$result3" ]; then
        print_test_result "$test_name" "PASS"
    else
        print_test_result "$test_name" "FAIL" "Got: '$result1', '$result2', '$result3'"
    fi
}

# Test 2: get_next_task_for_persona - matching task
test_get_next_task_matching() {
    local test_name="get_next_task_for_persona: finds matching task"

    local test_dir=$(create_test_queue)
    local saved_tasks_dir="$TASKS_DIR"
    export TASKS_DIR="$test_dir"

    local result=$(get_next_task_for_persona "optimizer")

    export TASKS_DIR="$saved_tasks_dir"
    rm -rf "$test_dir"

    if echo "$result" | grep -q "\[OPTIMIZER\]"; then
        print_test_result "$test_name" "PASS"
    else
        print_test_result "$test_name" "FAIL" "Expected OPTIMIZER task, got: $result"
    fi
}

# Test 3: get_next_task_for_persona - untagged fallback
test_get_next_task_untagged() {
    local test_name="get_next_task_for_persona: falls back to untagged"

    local test_dir
    test_dir=$(mktemp -d)
    local test_queue="${test_dir}/queue.md"

    cat > "$test_queue" << 'EOF'
# Task Queue

## Pending Tasks

- [ ] [OPTIMIZER] Test optimizer task
- [ ] Untagged test task

## Completed Tasks

EOF

    local saved_tasks_dir="$TASKS_DIR"
    export TASKS_DIR="$test_dir"

    # Architect should get untagged task (no architect tasks available)
    local result=$(get_next_task_for_persona "architect")

    export TASKS_DIR="$saved_tasks_dir"
    rm -rf "$test_dir"

    if echo "$result" | grep -q "Untagged test task"; then
        print_test_result "$test_name" "PASS"
    else
        print_test_result "$test_name" "FAIL" "Expected untagged task, got: $result"
    fi
}

# Test 4: mark_task_in_progress
test_mark_task_in_progress() {
    local test_name="mark_task_in_progress: marks task correctly"

    local test_dir=$(create_test_queue)
    local saved_tasks_dir="$TASKS_DIR"
    export TASKS_DIR="$test_dir"

    mark_task_in_progress "[OPTIMIZER] Test optimizer task" "optimizer" 2>/dev/null

    local result=$(grep "\[~\]" "$test_dir/queue.md")

    export TASKS_DIR="$saved_tasks_dir"
    rm -rf "$test_dir"

    if echo "$result" | grep -q "in-progress: optimizer"; then
        print_test_result "$test_name" "PASS"
    else
        print_test_result "$test_name" "FAIL" "Task not marked in-progress correctly: $result"
    fi
}

# Test 5: has_in_progress_work
test_has_in_progress_work() {
    local test_name="has_in_progress_work: detects in-progress tasks"

    local test_dir
    test_dir=$(mktemp -d)
    local test_queue="${test_dir}/queue.md"

    cat > "$test_queue" << 'EOF'
# Task Queue

## Pending Tasks

- [~] [OPTIMIZER] Test task (in-progress: optimizer, started: 2025-10-30T17:00:00Z)
- [ ] [ARCHITECT] Another task

## Completed Tasks

EOF

    local saved_tasks_dir="$TASKS_DIR"
    export TASKS_DIR="$test_dir"

    if has_in_progress_work "optimizer"; then
        local has_work="yes"
    else
        local has_work="no"
    fi

    if has_in_progress_work "architect"; then
        local has_other_work="yes"
    else
        local has_other_work="no"
    fi

    export TASKS_DIR="$saved_tasks_dir"
    rm -rf "$test_dir"

    if [ "$has_work" = "yes" ] && [ "$has_other_work" = "no" ]; then
        print_test_result "$test_name" "PASS"
    else
        print_test_result "$test_name" "FAIL" "Expected optimizer=yes, architect=no, got optimizer=$has_work, architect=$has_other_work"
    fi
}

# Test 6: mark_task_completed_enhanced
test_mark_task_completed_enhanced() {
    local test_name="mark_task_completed_enhanced: completes in-progress task"

    local test_dir
    test_dir=$(mktemp -d)
    local test_queue="${test_dir}/queue.md"

    cat > "$test_queue" << 'EOF'
# Task Queue

## Pending Tasks

- [~] Test task (in-progress: optimizer, started: 2025-10-30T17:00:00Z)

## Completed Tasks

EOF

    local saved_tasks_dir="$TASKS_DIR"
    export TASKS_DIR="$test_dir"

    mark_task_completed_enhanced "Test task" "optimizer" 2>/dev/null

    local result=$(grep "\[x\]" "$test_queue")

    export TASKS_DIR="$saved_tasks_dir"
    rm -rf "$test_dir"

    if echo "$result" | grep -q "completed.*by: optimizer"; then
        print_test_result "$test_name" "PASS"
    else
        print_test_result "$test_name" "FAIL" "Task not completed correctly"
    fi
}

# Test 7: State transition flow
test_state_transition_flow() {
    local test_name="State transitions: [ ] → [~] → [x]"

    local test_dir=$(create_test_queue)
    local saved_tasks_dir="$TASKS_DIR"
    export TASKS_DIR="$test_dir"

    # Start: [ ]
    local initial=$(grep "Test optimizer task" "$test_dir/queue.md" | grep -c "\[ \]")

    # Transition to [~]
    mark_task_in_progress "[OPTIMIZER] Test optimizer task" "optimizer" 2>/dev/null
    local in_progress=$(grep "Test optimizer task" "$test_dir/queue.md" | grep -c "\[~\]")

    # Transition to [x]
    mark_task_completed_enhanced "[OPTIMIZER] Test optimizer task" "optimizer" 2>/dev/null
    local completed=$(grep "Test optimizer task" "$test_dir/queue.md" | grep -c "\[x\]")

    export TASKS_DIR="$saved_tasks_dir"
    rm -rf "$test_dir"

    if [ "$initial" = "1" ] && [ "$in_progress" = "1" ] && [ "$completed" = "1" ]; then
        print_test_result "$test_name" "PASS"
    else
        print_test_result "$test_name" "FAIL" "Transitions: initial=$initial, progress=$in_progress, complete=$completed"
    fi
}

# Test 8: Error handling - missing file
test_error_handling_missing_file() {
    local test_name="Error handling: missing queue file"

    local saved_tasks_dir="$TASKS_DIR"
    export TASKS_DIR="/tmp/nonexistent_$$"

    if mark_task_in_progress "Test task" "optimizer" 2>/dev/null; then
        print_test_result "$test_name" "FAIL" "Should have failed with missing file"
    else
        print_test_result "$test_name" "PASS"
    fi

    export TASKS_DIR="$saved_tasks_dir"
}

# Test 9: get_in_progress_tasks
test_get_in_progress_tasks() {
    local test_name="get_in_progress_tasks: returns correct tasks"

    local test_dir
    test_dir=$(mktemp -d)
    local test_queue="${test_dir}/queue.md"

    cat > "$test_queue" << 'EOF'
# Task Queue

## Pending Tasks

- [~] Task 1 (in-progress: optimizer, started: 2025-10-30T17:00:00Z)
- [~] Task 2 (in-progress: optimizer, started: 2025-10-30T17:05:00Z)
- [~] Task 3 (in-progress: architect, started: 2025-10-30T17:10:00Z)

## Completed Tasks

EOF

    local saved_tasks_dir="$TASKS_DIR"
    export TASKS_DIR="$test_dir"

    local optimizer_tasks=$(get_in_progress_tasks "optimizer" | wc -l)
    local architect_tasks=$(get_in_progress_tasks "architect" | wc -l)

    export TASKS_DIR="$saved_tasks_dir"
    rm -rf "$test_dir"

    if [ "$optimizer_tasks" = "2" ] && [ "$architect_tasks" = "1" ]; then
        print_test_result "$test_name" "PASS"
    else
        print_test_result "$test_name" "FAIL" "Expected optimizer=2, architect=1, got optimizer=$optimizer_tasks, architect=$architect_tasks"
    fi
}

# Test 10: Case insensitivity of persona tags
test_case_insensitivity() {
    local test_name="Persona matching: case insensitive"

    local result1=$(extract_persona_tag "[OPTIMIZER] Task")
    local result2=$(extract_persona_tag "[Optimizer] Task")
    local result3=$(extract_persona_tag "[optimizer] Task")

    if [ "$result1" = "optimizer" ] && [ "$result2" = "optimizer" ] && [ "$result3" = "optimizer" ]; then
        print_test_result "$test_name" "PASS"
    else
        print_test_result "$test_name" "FAIL" "Case sensitivity issues"
    fi
}

# Run all tests
echo "=== Task State Management Test Suite ==="
echo "Testing: ${DAEMON_ROOT}/lib/task-state-management.sh"
echo ""
echo "Running tests..."
echo ""

test_extract_persona_tag
test_get_next_task_matching
test_get_next_task_untagged
test_mark_task_in_progress
test_has_in_progress_work
test_mark_task_completed_enhanced
test_state_transition_flow
test_error_handling_missing_file
test_get_in_progress_tasks
test_case_insensitivity

echo ""
echo "=== Test Results ==="
echo "Tests run: $TESTS_RUN"
echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo "Safe to integrate task state management library."
    exit 0
else
    echo ""
    echo -e "${RED}✗ Some tests failed${NC}"
    echo "Do NOT integrate until failures are resolved."
    exit 1
fi
