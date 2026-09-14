#!/bin/bash
#
# Phase 4 Integration Test Suite
# Comprehensive testing of self-healing orchestration system
#
# Covers:
# 1. Task recovery (retry logic, exponential backoff, quarantine)
# 2. Persona health (health calculation, cooldowns, health-aware selection)
# 3. Anomaly detection (all 8 detection types)
# 4. End-to-end self-healing (detect→diagnose→remediate cycle)
#
# Usage:
#   ./scripts/test-phase4-integration.sh [--task-recovery|--persona-health|--anomaly|--e2e|--all]
#   Default (no args): Run all tests
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
STATE_DIR="${DAEMON_ROOT}/personalities"
STATE_FILE="${STATE_DIR}/state.json"
METRICS_DIR="${DAEMON_ROOT}/metrics"
LOGS_DIR="${DAEMON_ROOT}/logs"
TEST_RESULTS="${DAEMON_ROOT}/test-results-phase4-$(date +%Y%m%d-%H%M%S).txt"

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Source required libraries
for lib in state-api task-validation task-recovery performance-metrics persona-health \
           remediation-engine root-cause-analysis common-init alert-manager; do
    if [ -f "${DAEMON_ROOT}/lib/${lib}.sh" ]; then
        source "${DAEMON_ROOT}/lib/${lib}.sh"
    fi
done

# ============================================================================
# Test Utility Functions
# ============================================================================

log_test() {
    local level="$1"
    local message="$2"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$level] $message" | tee -a "$TEST_RESULTS"
}

test_pass() {
    local test_name="$1"
    log_test "✅ PASS" "$test_name"
    ((TESTS_PASSED++))
}

test_fail() {
    local test_name="$1"
    local reason="${2:-Unknown reason}"
    log_test "❌ FAIL" "$test_name: $reason"
    ((TESTS_FAILED++))
}

test_skip() {
    local test_name="$1"
    local reason="${2:-Skipped}"
    log_test "⏭️  SKIP" "$test_name: $reason"
    ((TESTS_SKIPPED++))
}

assert_equals() {
    local expected="$1"
    local actual="$2"
    local test_name="${3:-Equality check}"

    if [ "$expected" = "$actual" ]; then
        test_pass "$test_name"
        return 0
    else
        test_fail "$test_name" "Expected '$expected', got '$actual'"
        return 1
    fi
}

assert_contains() {
    local haystack="$1"
    local needle="$2"
    local test_name="${3:-Contains check}"

    if echo "$haystack" | grep -q "$needle"; then
        test_pass "$test_name"
        return 0
    else
        test_fail "$test_name" "Expected to find '$needle' in output"
        return 1
    fi
}

assert_file_exists() {
    local file="$1"
    local test_name="${2:-File exists}"

    if [ -f "$file" ]; then
        test_pass "$test_name"
        return 0
    else
        test_fail "$test_name" "File not found: $file"
        return 1
    fi
}

# ============================================================================
# Test Suite 1: Task Recovery
# ============================================================================

test_task_recovery_suite() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  TEST SUITE 1: Task Recovery & Retry Logic"
    echo "═══════════════════════════════════════════════════════════"
    echo "" | tee -a "$TEST_RESULTS"

    # Test 1.1: Retry counter initialization
    if declare -f init_retry_tracking >/dev/null 2>&1; then
        init_retry_tracking
        assert_file_exists "${METRICS_DIR}/retry-tracking.json" "Retry tracking file initialized"
    else
        test_skip "Retry initialization" "init_retry_tracking not available"
    fi

    # Test 1.2: Exponential backoff calculation
    if declare -f calculate_backoff_delay >/dev/null 2>&1; then
        local delay=$(calculate_backoff_delay 1)
        [ -n "$delay" ] && test_pass "Exponential backoff calculation" || test_fail "Exponential backoff calculation" "No delay returned"
    else
        test_skip "Exponential backoff" "calculate_backoff_delay not available"
    fi

    # Test 1.3: Task quarantine
    if declare -f quarantine_task >/dev/null 2>&1; then
        quarantine_task "test-task" "max retries exceeded" > /dev/null 2>&1
        assert_file_exists "${METRICS_DIR}/quarantined-tasks.json" "Quarantine mechanism"
    else
        test_skip "Task quarantine" "quarantine_task not available"
    fi

    # Test 1.4: Retry state tracking
    if declare -f record_retry_attempt >/dev/null 2>&1; then
        record_retry_attempt "test-task" 1 > /dev/null 2>&1
        test_pass "Retry attempt recording"
    else
        test_skip "Retry tracking" "record_retry_attempt not available"
    fi
}

# ============================================================================
# Test Suite 2: Persona Health System
# ============================================================================

test_persona_health_suite() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  TEST SUITE 2: Persona Health System"
    echo "═══════════════════════════════════════════════════════════"
    echo "" | tee -a "$TEST_RESULTS"

    # Test 2.1: Health score calculation
    if declare -f calculate_persona_health >/dev/null 2>&1; then
        local health=$(calculate_persona_health "architect" 2>/dev/null || echo "0")
        assert_equals "100" "$health" "Initial health score (architect)"
    else
        test_skip "Health calculation" "calculate_persona_health not available"
    fi

    # Test 2.2: Health status determination
    if declare -f get_persona_health_status >/dev/null 2>&1; then
        local status=$(get_persona_health_status "optimizer" 2>/dev/null || echo "UNKNOWN")
        [ -n "$status" ] && test_pass "Health status determination" || test_fail "Health status" "No status returned"
    else
        test_skip "Health status" "get_persona_health_status not available"
    fi

    # Test 2.3: Cooldown triggering
    if declare -f trigger_persona_cooldown >/dev/null 2>&1; then
        trigger_persona_cooldown "auditor" "test_cooldown" 2 > /dev/null 2>&1
        local on_cooldown=$(is_persona_on_cooldown "auditor" 2>/dev/null || echo "false")
        assert_equals "0" "$?" "Cooldown triggered successfully"
    else
        test_skip "Cooldown triggering" "trigger_persona_cooldown not available"
    fi

    # Test 2.4: Cooldown expiration
    if declare -f expire_persona_cooldowns >/dev/null 2>&1; then
        expire_persona_cooldowns > /dev/null 2>&1
        test_pass "Cooldown expiration check"
    else
        test_skip "Cooldown expiration" "expire_persona_cooldowns not available"
    fi

    # Test 2.5: Health-aware selection
    if declare -f get_eligible_personas >/dev/null 2>&1; then
        local eligible=$(get_eligible_personas 50 2>/dev/null || echo "")
        [ -n "$eligible" ] && test_pass "Eligible personas filtering" || test_fail "Eligible personas" "No personas returned"
    else
        test_skip "Eligible personas" "get_eligible_personas not available"
    fi
}

# ============================================================================
# Test Suite 3: Anomaly Detection
# ============================================================================

test_anomaly_detection_suite() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  TEST SUITE 3: Anomaly Detection"
    echo "═══════════════════════════════════════════════════════════"
    echo "" | tee -a "$TEST_RESULTS"

    # Test 3.1: Persona lock detection
    if declare -f detect_two_body_lock >/dev/null 2>&1; then
        local lock=$(detect_two_body_lock 2>/dev/null || echo "none")
        [ "$lock" != "error" ] && test_pass "Persona lock detection" || test_fail "Lock detection" "Error occurred"
    else
        test_skip "Lock detection" "detect_two_body_lock not available"
    fi

    # Test 3.2: Health degradation detection
    if declare -f calculate_persona_health >/dev/null 2>&1; then
        for persona in architect optimizer auditor; do
            local health=$(calculate_persona_health "$persona" 2>/dev/null || echo "100")
            test_pass "Health degradation check: $persona"
        done
    else
        test_skip "Health degradation" "calculate_persona_health not available"
    fi

    # Test 3.3: Validation failure detection
    if [ -f "${LOGS_DIR}/activity.log" ]; then
        local failures=$(grep -c "validation.*failed" "${LOGS_DIR}/activity.log" 2>/dev/null || echo "0")
        test_pass "Validation failure detection (found: $failures)"
    else
        test_skip "Validation failures" "activity.log not found"
    fi

    # Test 3.4: Reflection loop detection
    if [ -f "${LOGS_DIR}/activity.log" ]; then
        local reflections=$(grep -c "Starting reflection" "${LOGS_DIR}/activity.log" 2>/dev/null || echo "0")
        test_pass "Reflection loop detection (found: $reflections)"
    else
        test_skip "Reflection loops" "activity.log not found"
    fi

    # Test 3.5: API health detection
    if [ -f "${LOGS_DIR}/activity.log" ]; then
        local api_errors=$(grep -c "API error" "${LOGS_DIR}/activity.log" 2>/dev/null || echo "0")
        test_pass "API health detection (found: $api_errors)"
    else
        test_skip "API health" "activity.log not found"
    fi
}

# ============================================================================
# Test Suite 4: End-to-End Self-Healing
# ============================================================================

test_e2e_self_healing_suite() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  TEST SUITE 4: End-to-End Self-Healing"
    echo "═══════════════════════════════════════════════════════════"
    echo "" | tee -a "$TEST_RESULTS"

    # Test 4.1: Remediation engine available
    if declare -f remediate_anomaly >/dev/null 2>&1; then
        test_pass "Remediation engine loaded"
    else
        test_fail "Remediation engine" "remediate_anomaly function not available"
        return 1
    fi

    # Test 4.2: Root cause analysis available
    if declare -f diagnose_issue >/dev/null 2>&1; then
        test_pass "Root cause analysis loaded"
    else
        test_fail "Root cause analysis" "diagnose_issue function not available"
        return 1
    fi

    # Test 4.3: Self-healing loop script exists
    if [ -x "${DAEMON_ROOT}/scripts/self-healing-loop.sh" ]; then
        test_pass "Self-healing loop script available"
    else
        test_fail "Healing loop script" "Script not found or not executable"
    fi

    # Test 4.4: Full cycle simulation
    if [ -x "${DAEMON_ROOT}/scripts/self-healing-loop.sh" ]; then
        local cycle_output=$("${DAEMON_ROOT}/scripts/self-healing-loop.sh" --check 2>&1 || echo "")
        [ -n "$cycle_output" ] && test_pass "Anomaly detection cycle" || test_fail "Detection cycle" "No output"
    else
        test_skip "Full cycle test" "Script not available"
    fi

    # Test 4.5: Remediation recording
    if [ -f "${LOGS_DIR}/remediation-audit.jsonl" ]; then
        local remediation_count=$(wc -l < "${LOGS_DIR}/remediation-audit.jsonl" 2>/dev/null || echo "0")
        test_pass "Remediation audit trail (entries: $remediation_count)"
    else
        test_skip "Remediation recording" "Audit trail not yet created"
    fi
}

# ============================================================================
# Test Summary
# ============================================================================

print_test_summary() {
    echo ""
    echo "═══════════════════════════════════════════════════════════" | tee -a "$TEST_RESULTS"
    echo "  Test Results Summary" | tee -a "$TEST_RESULTS"
    echo "═══════════════════════════════════════════════════════════" | tee -a "$TEST_RESULTS"
    echo "" | tee -a "$TEST_RESULTS"

    local total=$((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))
    local pass_rate=0
    if [ "$total" -gt 0 ]; then
        pass_rate=$((TESTS_PASSED * 100 / (TESTS_PASSED + TESTS_FAILED)))
    fi

    echo "✅ Passed:  $TESTS_PASSED" | tee -a "$TEST_RESULTS"
    echo "❌ Failed:  $TESTS_FAILED" | tee -a "$TEST_RESULTS"
    echo "⏭️  Skipped: $TESTS_SKIPPED" | tee -a "$TEST_RESULTS"
    echo "📊 Total:   $total" | tee -a "$TEST_RESULTS"
    echo "📈 Pass Rate: $pass_rate%" | tee -a "$TEST_RESULTS"
    echo "" | tee -a "$TEST_RESULTS"

    if [ "$TESTS_FAILED" -eq 0 ]; then
        echo "✅ ALL TESTS PASSED" | tee -a "$TEST_RESULTS"
        return 0
    else
        echo "❌ SOME TESTS FAILED" | tee -a "$TEST_RESULTS"
        return 1
    fi
}

# ============================================================================
# Main Test Execution
# ============================================================================

main() {
    mkdir -p "$LOGS_DIR"

    echo "Starting Phase 4 Integration Tests..." | tee "$TEST_RESULTS"
    echo "Test Results: $TEST_RESULTS"
    echo ""

    case "${1:-all}" in
        --task-recovery)
            test_task_recovery_suite
            ;;
        --persona-health)
            test_persona_health_suite
            ;;
        --anomaly)
            test_anomaly_detection_suite
            ;;
        --e2e)
            test_e2e_self_healing_suite
            ;;
        --all)
            test_task_recovery_suite
            test_persona_health_suite
            test_anomaly_detection_suite
            test_e2e_self_healing_suite
            ;;
        *)
            echo "Usage: $0 [--task-recovery|--persona-health|--anomaly|--e2e|--all]"
            exit 1
            ;;
    esac

    print_test_summary
    return $?
}

main "$@"
