#!/bin/bash
# Test Suite for Batch Read Helpers
#
# This test suite validates error handling and safe defaults in the batch read
# optimization. Run this before deploying changes to production.
#
# Usage: ./scripts/test-batch-read-helpers.sh
# Exit codes: 0 = all tests pass, 1 = one or more tests failed

set -e  # Exit on error (will be overridden in test functions)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get daemon root
DAEMON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Source the helpers we're testing
source "${DAEMON_ROOT}/lib/batch-read-helpers.sh"

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

# Mock log function for testing (prevent spamming real logs)
log() {
    local level="$1"
    local message="$2"
    echo "[TEST-LOG][$level] $message" >&2
}

echo "=== Batch Read Helpers Test Suite ==="
echo "Testing: ${DAEMON_ROOT}/lib/batch-read-helpers.sh"
echo ""

# Test 1: read_emotional_state_batch with valid file
test_valid_emotional_state() {
    local test_name="read_emotional_state_batch: valid file"

    # Use real emotional file
    export EMOTIONAL_FILE="${DAEMON_ROOT}/triggers/emotional.json"

    if [ ! -f "$EMOTIONAL_FILE" ]; then
        print_test_result "$test_name" "FAIL" "Emotional file not found: $EMOTIONAL_FILE"
        return
    fi

    local result
    result=$(read_emotional_state_batch)
    local exit_code=$?

    if [ $exit_code -eq 0 ] && [ -n "$result" ] && [ "$result" != "null" ]; then
        # Verify JSON structure
        local frustration
        frustration=$(echo "$result" | jq -r '.frustration' 2>/dev/null)

        if [ -n "$frustration" ] && [ "$frustration" != "null" ]; then
            print_test_result "$test_name" "PASS"
        else
            print_test_result "$test_name" "FAIL" "Missing frustration field in result"
        fi
    else
        print_test_result "$test_name" "FAIL" "Exit code: $exit_code, Result empty or null"
    fi
}

# Test 2: read_emotional_state_batch with missing file
test_missing_emotional_file() {
    local test_name="read_emotional_state_batch: missing file"

    # Point to non-existent file
    export EMOTIONAL_FILE="/tmp/nonexistent_emotional_$$.json"

    local result
    result=$(read_emotional_state_batch 2>/dev/null)
    local exit_code=$?

    # Should return safe defaults (exit code 1) but not crash
    if [ $exit_code -eq 1 ] && [ -n "$result" ]; then
        # Verify safe defaults have high thresholds
        local frustration_thresh
        frustration_thresh=$(echo "$result" | jq -r '.frustration_thresh' 2>/dev/null)

        if [ "$frustration_thresh" = "999" ]; then
            print_test_result "$test_name" "PASS"
        else
            print_test_result "$test_name" "FAIL" "Safe defaults not applied (thresh=$frustration_thresh)"
        fi
    else
        print_test_result "$test_name" "FAIL" "Exit code: $exit_code (expected 1)"
    fi
}

# Test 3: read_emotional_state_batch with corrupted file
test_corrupted_emotional_file() {
    local test_name="read_emotional_state_batch: corrupted JSON"

    # Create temporary corrupted file
    local temp_file="/tmp/corrupted_emotional_$$.json"
    echo "{this is not valid json" > "$temp_file"
    export EMOTIONAL_FILE="$temp_file"

    local result
    result=$(read_emotional_state_batch 2>/dev/null)
    local exit_code=$?

    # Cleanup
    rm -f "$temp_file"

    # Should return safe defaults
    if [ $exit_code -eq 1 ] && [ -n "$result" ]; then
        local frustration
        frustration=$(echo "$result" | jq -r '.frustration' 2>/dev/null)

        if [ "$frustration" = "0" ]; then
            print_test_result "$test_name" "PASS"
        else
            print_test_result "$test_name" "FAIL" "Safe defaults not applied correctly"
        fi
    else
        print_test_result "$test_name" "FAIL" "Exit code: $exit_code or empty result"
    fi
}

# Test 4: read_chaos_config_batch with valid file
test_valid_chaos_config() {
    local test_name="read_chaos_config_batch: valid file"

    export CHAOS_FILE="${DAEMON_ROOT}/triggers/chaos.json"

    if [ ! -f "$CHAOS_FILE" ]; then
        print_test_result "$test_name" "FAIL" "Chaos file not found: $CHAOS_FILE"
        return
    fi

    local result
    result=$(read_chaos_config_batch)
    local exit_code=$?

    if [ $exit_code -eq 0 ] && [ -n "$result" ]; then
        local enabled
        enabled=$(echo "$result" | jq -r '.enabled' 2>/dev/null)

        if [ "$enabled" = "true" ] || [ "$enabled" = "false" ]; then
            print_test_result "$test_name" "PASS"
        else
            print_test_result "$test_name" "FAIL" "Invalid enabled value: $enabled"
        fi
    else
        print_test_result "$test_name" "FAIL" "Exit code: $exit_code"
    fi
}

# Test 5: read_chaos_config_batch with missing file
test_missing_chaos_file() {
    local test_name="read_chaos_config_batch: missing file"

    export CHAOS_FILE="/tmp/nonexistent_chaos_$$.json"

    local result
    result=$(read_chaos_config_batch 2>/dev/null)
    local exit_code=$?

    # Should return safe defaults (chaos disabled)
    if [ $exit_code -eq 1 ] && [ -n "$result" ]; then
        local enabled
        enabled=$(echo "$result" | jq -r '.enabled' 2>/dev/null)

        if [ "$enabled" = "false" ]; then
            print_test_result "$test_name" "PASS"
        else
            print_test_result "$test_name" "FAIL" "Chaos not disabled in safe defaults"
        fi
    else
        print_test_result "$test_name" "FAIL" "Exit code: $exit_code"
    fi
}

# Test 6: validate_emotional_state with valid data
test_validate_emotional_state_valid() {
    local test_name="validate_emotional_state: valid data"

    local valid_json='{
        "frustration": 5,
        "frustration_thresh": 10,
        "success_streak": 3,
        "success_thresh": 5,
        "failure_streak": 0,
        "failure_thresh": 3,
        "stuck_minutes": 0,
        "stuck_thresh": 60
    }'

    if validate_emotional_state "$valid_json" 2>/dev/null; then
        print_test_result "$test_name" "PASS"
    else
        print_test_result "$test_name" "FAIL" "Valid data rejected"
    fi
}

# Test 7: validate_emotional_state with missing fields
test_validate_emotional_state_missing_fields() {
    local test_name="validate_emotional_state: missing fields"

    local invalid_json='{
        "frustration": 5,
        "frustration_thresh": 10
    }'

    if validate_emotional_state "$invalid_json" 2>/dev/null; then
        print_test_result "$test_name" "FAIL" "Invalid data accepted"
    else
        print_test_result "$test_name" "PASS"
    fi
}

# Test 8: extract_json_field helper
test_extract_json_field() {
    local test_name="extract_json_field: field extraction"

    local test_json='{"foo": "bar", "nested": {"value": 42}}'

    local result1
    result1=$(extract_json_field "$test_json" ".foo")

    local result2
    result2=$(extract_json_field "$test_json" ".nested.value")

    if [ "$result1" = "bar" ] && [ "$result2" = "42" ]; then
        print_test_result "$test_name" "PASS"
    else
        print_test_result "$test_name" "FAIL" "Got: '$result1', '$result2'"
    fi
}

# Test 9: Performance test (batch vs sequential)
test_performance_improvement() {
    local test_name="Performance: batch vs sequential"

    export EMOTIONAL_FILE="${DAEMON_ROOT}/triggers/emotional.json"

    if [ ! -f "$EMOTIONAL_FILE" ]; then
        print_test_result "$test_name" "FAIL" "Emotional file not found"
        return
    fi

    # Time sequential reads
    local start_seq=$(date +%s%N)
    for i in {1..10}; do
        jq -r '.current_state.frustration_level' "$EMOTIONAL_FILE" >/dev/null 2>&1
        jq -r '.thresholds.high_frustration.value' "$EMOTIONAL_FILE" >/dev/null 2>&1
        jq -r '.current_state.success_streak' "$EMOTIONAL_FILE" >/dev/null 2>&1
    done
    local end_seq=$(date +%s%N)
    local time_seq=$(( (end_seq - start_seq) / 1000000 ))  # Convert to ms

    # Time batch reads
    local start_batch=$(date +%s%N)
    for i in {1..10}; do
        read_emotional_state_batch >/dev/null 2>&1
    done
    local end_batch=$(date +%s%N)
    local time_batch=$(( (end_batch - start_batch) / 1000000 ))  # Convert to ms

    # Batch should be faster
    if [ $time_batch -lt $time_seq ]; then
        local improvement=$(( (time_seq - time_batch) * 100 / time_seq ))
        echo "  Sequential: ${time_seq}ms, Batch: ${time_batch}ms (${improvement}% improvement)"
        print_test_result "$test_name" "PASS"
    else
        print_test_result "$test_name" "FAIL" "Batch slower: ${time_batch}ms vs ${time_seq}ms"
    fi
}

# Test 10: read_activation_floor_batch
test_activation_floor_batch() {
    local test_name="read_activation_floor_batch: valid file"

    export EMOTIONAL_FILE="${DAEMON_ROOT}/triggers/emotional.json"

    if [ ! -f "$EMOTIONAL_FILE" ]; then
        print_test_result "$test_name" "FAIL" "Emotional file not found"
        return
    fi

    local result
    result=$(read_activation_floor_batch 2>/dev/null)
    local exit_code=$?

    if [ $exit_code -eq 0 ] && [ -n "$result" ]; then
        local floor_hours
        floor_hours=$(echo "$result" | jq -r '.floor_hours' 2>/dev/null)

        if [ -n "$floor_hours" ] && [ "$floor_hours" != "null" ]; then
            print_test_result "$test_name" "PASS"
        else
            print_test_result "$test_name" "FAIL" "Missing floor_hours field"
        fi
    else
        print_test_result "$test_name" "FAIL" "Exit code: $exit_code"
    fi
}

# Run all tests
echo "Running tests..."
echo ""

test_valid_emotional_state
test_missing_emotional_file
test_corrupted_emotional_file
test_valid_chaos_config
test_missing_chaos_file
test_validate_emotional_state_valid
test_validate_emotional_state_missing_fields
test_extract_json_field
test_performance_improvement
test_activation_floor_batch

echo ""
echo "=== Test Results ==="
echo "Tests run: $TESTS_RUN"
echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo "Safe to deploy batch read optimization."
    exit 0
else
    echo ""
    echo -e "${RED}✗ Some tests failed${NC}"
    echo "Do NOT deploy until failures are resolved."
    exit 1
fi
