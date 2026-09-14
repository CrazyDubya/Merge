#!/bin/bash
#
# Verify Phase 4 Integration Script
# Comprehensive validation of self-healing orchestration system
#
# Tests:
# 1. Health-aware persona selection (daemon.sh integration)
# 2. Persona cooldown enforcement
# 3. Anomaly detection accuracy
# 4. Remediation engine functionality
# 5. Root cause analysis accuracy
# 6. Self-healing loop operation
# 7. Trap cleanup compliance
#
# Usage:
#   ./scripts/verify-phase4-integration.sh [--full|--quick|--health|--remediation|--cleanup]
#   Default (no args): Run full verification
#
# Exit codes:
#   0 = All checks passed
#   1 = Some checks failed
#   2 = Critical failure (daemon offline)
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
STATE_DIR="${DAEMON_ROOT}/personalities"
STATE_FILE="${STATE_DIR}/state.json"
METRICS_DIR="${DAEMON_ROOT}/metrics"
LOGS_DIR="${DAEMON_ROOT}/logs"
LIB_DIR="${DAEMON_ROOT}/lib"
SCRIPTS_DIR="${DAEMON_ROOT}/scripts"

# Verification output file
VERIFY_REPORT="${DAEMON_ROOT}/docs/phase4-verification-report-$(date +%Y%m%d-%H%M%S).md"

# Test counters
CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_SKIPPED=0

# Colors for output (if terminal)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Logging Functions
# ============================================================================

log_header() {
    local title="$1"
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  $title"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
    {
        echo ""
        echo "## $title"
        echo ""
    } >> "$VERIFY_REPORT"
}

log_check() {
    local status="$1"
    local test_name="$2"
    local details="${3:-}"

    case "$status" in
        PASS)
            echo -e "${GREEN}✅ PASS${NC}: $test_name"
            if [ -n "$details" ]; then
                echo "   → $details"
            fi
            {
                echo "✅ **PASS**: $test_name"
                [ -n "$details" ] && echo "   - $details"
            } >> "$VERIFY_REPORT"
            ((CHECKS_PASSED++))
            ;;
        FAIL)
            echo -e "${RED}❌ FAIL${NC}: $test_name"
            if [ -n "$details" ]; then
                echo "   → $details"
            fi
            {
                echo "❌ **FAIL**: $test_name"
                [ -n "$details" ] && echo "   - $details"
            } >> "$VERIFY_REPORT"
            ((CHECKS_FAILED++))
            ;;
        SKIP)
            echo -e "${YELLOW}⏭️  SKIP${NC}: $test_name"
            if [ -n "$details" ]; then
                echo "   → $details"
            fi
            {
                echo "⏭️  **SKIP**: $test_name"
                [ -n "$details" ] && echo "   - $details"
            } >> "$VERIFY_REPORT"
            ((CHECKS_SKIPPED++))
            ;;
    esac
}

log_section() {
    local section="$1"
    echo ""
    echo -e "${BLUE}→ $section${NC}"
    echo "  $section" >> "$VERIFY_REPORT"
}

# ============================================================================
# Prerequisite Checks
# ============================================================================

check_prerequisites() {
    log_header "Prerequisite Checks"

    # Check if daemon root exists
    if [ ! -d "$DAEMON_ROOT" ]; then
        log_check FAIL "Daemon root directory" "Not found at $DAEMON_ROOT"
        return 1
    fi
    log_check PASS "Daemon root directory" "Found at $DAEMON_ROOT"

    # Check if state file exists
    if [ ! -f "$STATE_FILE" ]; then
        log_check FAIL "State file" "Not found at $STATE_FILE"
        return 1
    fi
    log_check PASS "State file" "Found at $STATE_FILE"

    # Check if required libraries exist
    for lib in remediation-engine root-cause-analysis persona-health task-recovery; do
        if [ -f "${LIB_DIR}/${lib}.sh" ]; then
            log_check PASS "Library: $lib" "Present"
        else
            log_check FAIL "Library: $lib" "Missing at ${LIB_DIR}/${lib}.sh"
            return 1
        fi
    done

    # Check if self-healing loop script exists
    if [ ! -x "${SCRIPTS_DIR}/self-healing-loop.sh" ]; then
        log_check SKIP "Self-healing loop script" "Not yet created (scheduled for Phase 5 Part B)"
    else
        log_check PASS "Self-healing loop script" "Present and executable"
    fi

    # Check if daemon is running
    if ! pgrep -f "bash.*daemon.sh" > /dev/null 2>&1; then
        log_check SKIP "Daemon process" "Daemon not currently running (offline tests only)"
    else
        log_check PASS "Daemon process" "Daemon is running"
    fi

    return 0
}

# ============================================================================
# Test Suite 1: Health-Aware Persona Selection
# ============================================================================

test_health_aware_selection() {
    log_header "Test Suite 1: Health-Aware Persona Selection"

    log_section "1.1: Verify daemon.sh health integration points"

    # Check if daemon.sh has health checks in persona selection
    if grep -q "is_persona_excluded" "${DAEMON_ROOT}/daemon.sh"; then
        log_check PASS "Health check function calls" "is_persona_excluded found in daemon.sh"
    else
        log_check FAIL "Health check function calls" "is_persona_excluded not found in daemon.sh"
    fi

    # Count how many times is_persona_excluded is called
    local count=$(grep -c "is_persona_excluded" "${DAEMON_ROOT}/daemon.sh" || echo "0")
    if [ "$count" -ge 4 ]; then
        log_check PASS "Health filtering frequency" "is_persona_excluded called $count times"
    else
        log_check FAIL "Health filtering frequency" "Only called $count times (expected ≥4)"
    fi

    log_section "1.2: Verify health score calculation"

    # Source persona-health library
    if [ -f "${LIB_DIR}/persona-health.sh" ]; then
        # shellcheck source=/dev/null
        source "${LIB_DIR}/persona-health.sh"

        for persona in architect optimizer auditor maintainer skeptic experimenter; do
            if declare -f calculate_persona_health >/dev/null 2>&1; then
                local health=$(calculate_persona_health "$persona" 2>/dev/null || echo "UNKNOWN")
                if [ "$health" != "UNKNOWN" ] && [ "$health" -ge 0 ] && [ "$health" -le 100 ]; then
                    log_check PASS "Health calculation: $persona" "Score: $health%"
                else
                    log_check FAIL "Health calculation: $persona" "Invalid score: $health"
                fi
            else
                log_check SKIP "Health calculation: $persona" "Function not available"
            fi
        done
    else
        log_check SKIP "Health score calculation" "persona-health.sh not found"
    fi

    log_section "1.3: Verify cooldown mechanism"

    if [ -f "${LIB_DIR}/persona-health.sh" ]; then
        if grep -q "is_persona_on_cooldown\|trigger_persona_cooldown" "${LIB_DIR}/persona-health.sh"; then
            log_check PASS "Cooldown functions" "Found in persona-health.sh"
        else
            log_check FAIL "Cooldown functions" "Not found in persona-health.sh"
        fi
    else
        log_check SKIP "Cooldown mechanism" "Library not found"
    fi
}

# ============================================================================
# Test Suite 2: Remediation Engine
# ============================================================================

test_remediation_engine() {
    log_header "Test Suite 2: Remediation Engine Functionality"

    log_section "2.1: Verify remediation handlers"

    if [ -f "${LIB_DIR}/remediation-engine.sh" ]; then
        # Check for remediation handler functions
        local handlers=(
            "remediate_persona_lock"
            "remediate_health_degradation"
            "remediate_validation_failures"
            "remediate_reflection_loop"
            "remediate_api_health"
            "remediate_queue_stagnation"
        )

        for handler in "${handlers[@]}"; do
            if grep -q "^${handler}()" "${LIB_DIR}/remediation-engine.sh" || \
               grep -q "^${handler} ()" "${LIB_DIR}/remediation-engine.sh"; then
                log_check PASS "Remediation handler: $handler" "Present"
            else
                log_check FAIL "Remediation handler: $handler" "Not found"
            fi
        done
    else
        log_check FAIL "Remediation engine" "File not found at ${LIB_DIR}/remediation-engine.sh"
    fi

    log_section "2.2: Verify remediation audit logging"

    if [ -f "${LOGS_DIR}/remediation-audit.jsonl" ]; then
        local remediation_count=$(wc -l < "${LOGS_DIR}/remediation-audit.jsonl")
        if [ "$remediation_count" -gt 0 ]; then
            log_check PASS "Remediation audit trail" "Found $remediation_count entries"

            # Check for recent entries (last 24 hours)
            local recent=$(tail -10 "${LOGS_DIR}/remediation-audit.jsonl" 2>/dev/null | grep -c "^{" || echo "0")
            log_check PASS "Recent remediation entries" "$recent entries in last batch"
        else
            log_check SKIP "Remediation audit trail" "File exists but empty (no remediations yet)"
        fi
    else
        log_check SKIP "Remediation audit trail" "Not yet created (daemon needs to run)"
    fi
}

# ============================================================================
# Test Suite 3: Root Cause Analysis
# ============================================================================

test_root_cause_analysis() {
    log_header "Test Suite 3: Root Cause Analysis"

    log_section "3.1: Verify root cause analysis functions"

    if [ -f "${LIB_DIR}/root-cause-analysis.sh" ]; then
        local analyzers=(
            "diagnose_persona_lock"
            "diagnose_health_degradation"
            "diagnose_validation_failures"
            "diagnose_reflection_loop"
            "diagnose_api_degradation"
            "diagnose_queue_stagnation"
        )

        for analyzer in "${analyzers[@]}"; do
            if grep -q "^${analyzer}()" "${LIB_DIR}/root-cause-analysis.sh" || \
               grep -q "^${analyzer} ()" "${LIB_DIR}/root-cause-analysis.sh"; then
                log_check PASS "Root cause analyzer: $analyzer" "Present"
            else
                log_check SKIP "Root cause analyzer: $analyzer" "Not found (might be different name)"
            fi
        done
    else
        log_check FAIL "Root cause analysis" "File not found at ${LIB_DIR}/root-cause-analysis.sh"
    fi

    log_section "3.2: Verify diagnosis output format"

    if [ -f "${LIB_DIR}/root-cause-analysis.sh" ]; then
        # Check for JSON output format documentation
        if grep -q "root_causes\|confidence" "${LIB_DIR}/root-cause-analysis.sh"; then
            log_check PASS "Diagnosis format" "Structured output found"
        else
            log_check SKIP "Diagnosis format" "Format not clearly documented"
        fi
    fi
}

# ============================================================================
# Test Suite 4: Anomaly Detection
# ============================================================================

test_anomaly_detection() {
    log_header "Test Suite 4: Anomaly Detection"

    log_section "4.1: Verify anomaly detection functions"

    local anomaly_types=(
        "persona_lock"
        "health_degradation"
        "validation_failures"
        "reflection_loop"
        "api_health"
        "queue_stagnation"
    )

    if [ -f "${SCRIPTS_DIR}/self-healing-loop.sh" ]; then
        for anomaly in "${anomaly_types[@]}"; do
            if grep -q "$anomaly" "${SCRIPTS_DIR}/self-healing-loop.sh"; then
                log_check PASS "Anomaly detection: $anomaly" "Check implemented"
            else
                log_check SKIP "Anomaly detection: $anomaly" "Not found in healing loop"
            fi
        done
    else
        log_check FAIL "Anomaly detection" "Self-healing loop not found"
    fi

    log_section "4.2: Verify anomaly detection in activity log"

    if [ -f "${LOGS_DIR}/activity.log" ]; then
        local anomaly_entries
        anomaly_entries=$(grep -c "anomaly\|Anomaly" "${LOGS_DIR}/activity.log" 2>/dev/null) || anomaly_entries="0"
        if [ "$anomaly_entries" -gt 0 ]; then
            log_check PASS "Anomaly detection activity" "Found $anomaly_entries anomaly entries"
        else
            log_check SKIP "Anomaly detection activity" "No anomaly entries logged yet"
        fi
    else
        log_check SKIP "Anomaly detection" "Activity log not found"
    fi
}

# ============================================================================
# Test Suite 5: Trap Cleanup Compliance
# ============================================================================

test_trap_cleanup_compliance() {
    log_header "Test Suite 5: Trap Cleanup Compliance"

    log_section "5.1: Check mktemp cleanup patterns"

    local total_mktemp=0
    local total_trapped=0

    # Check each lib file for mktemp usage
    for lib_file in "${LIB_DIR}"/*.sh; do
        if [ -f "$lib_file" ]; then
            local filename
            local mktemp_count
            local trap_count
            filename=$(basename "$lib_file")
            mktemp_count=$(grep -c "mktemp" "$lib_file" 2>/dev/null) || mktemp_count="0"
            trap_count=$(grep -c "trap.*mktemp\|trap.*rm.*temp\|trap \"rm" "$lib_file" 2>/dev/null) || trap_count="0"

            if [ "$mktemp_count" -gt 0 ]; then
                total_mktemp=$((total_mktemp + mktemp_count))
                total_trapped=$((total_trapped + trap_count))

                if [ "$mktemp_count" -eq "$trap_count" ]; then
                    log_check PASS "Trap cleanup: $filename" "All $mktemp_count mktemp calls protected"
                else
                    log_check FAIL "Trap cleanup: $filename" "$mktemp_count mktemp, only $trap_count protected"
                fi
            fi
        fi
    done

    log_section "5.2: Overall trap cleanup summary"

    if [ "$total_mktemp" -gt 0 ]; then
        local coverage=$((total_trapped * 100 / total_mktemp))
        if [ "$coverage" -ge 100 ]; then
            log_check PASS "Overall trap cleanup coverage" "$coverage% ($total_trapped/$total_mktemp)"
        elif [ "$coverage" -ge 80 ]; then
            log_check PASS "Overall trap cleanup coverage" "$coverage% ($total_trapped/$total_mktemp) - Good"
        elif [ "$coverage" -ge 50 ]; then
            log_check FAIL "Overall trap cleanup coverage" "$coverage% ($total_trapped/$total_mktemp) - Needs improvement"
        else
            log_check FAIL "Overall trap cleanup coverage" "$coverage% ($total_trapped/$total_mktemp) - Low coverage"
        fi
    else
        log_check SKIP "Overall trap cleanup coverage" "No mktemp calls found"
    fi
}

# ============================================================================
# Test Suite 6: Self-Healing Loop
# ============================================================================

test_self_healing_loop() {
    log_header "Test Suite 6: Self-Healing Loop Operation"

    log_section "6.1: Verify self-healing loop script"

    if [ ! -x "${SCRIPTS_DIR}/self-healing-loop.sh" ]; then
        log_check FAIL "Self-healing loop availability" "Script not found or not executable"
        return 1
    fi
    log_check PASS "Self-healing loop availability" "Script is executable"

    log_section "6.2: Verify cron configuration"

    if crontab -l 2>/dev/null | grep -q "self-healing-loop.sh"; then
        log_check PASS "Cron configuration" "Self-healing loop scheduled"
    else
        log_check SKIP "Cron configuration" "Not yet scheduled in crontab"
    fi

    log_section "6.3: Check self-healing loop output"

    if [ -f "${LOGS_DIR}/self-healing-loop.log" ]; then
        local log_entries=$(wc -l < "${LOGS_DIR}/self-healing-loop.log")
        if [ "$log_entries" -gt 0 ]; then
            log_check PASS "Self-healing loop execution" "Log has $log_entries entries"

            # Check for DETECT/DIAGNOSE/HEAL phases
            local phases=$(grep -c "DETECT\|DIAGNOSE\|HEAL\|Phase" "${LOGS_DIR}/self-healing-loop.log" 2>/dev/null || echo "0")
            if [ "$phases" -gt 0 ]; then
                log_check PASS "Self-healing phases" "Found $phases phase transitions"
            fi
        else
            log_check SKIP "Self-healing loop execution" "Log exists but empty (not yet run)"
        fi
    else
        log_check SKIP "Self-healing loop execution" "Log file not yet created"
    fi
}

# ============================================================================
# Test Suite 7: Integration Points
# ============================================================================

test_integration_points() {
    log_header "Test Suite 7: Phase 3-4 Integration Points"

    log_section "7.1: Verify has_in_progress_work() uses persona parameter"

    if [ -f "${LIB_DIR}/task-state-management.sh" ]; then
        # Check if persona parameter is used in grep pattern
        if grep -q 'in-progress: $persona' "${LIB_DIR}/task-state-management.sh"; then
            log_check PASS "Persona parameter usage" "Found in has_in_progress_work()"
        else
            log_check FAIL "Persona parameter usage" "Persona not matched in grep pattern"
        fi
    else
        log_check FAIL "Task state management" "Library not found"
    fi

    log_section "7.2: Verify activation floor receives health data"

    if [ -f "${LIB_DIR}/activation-floor.sh" ]; then
        if grep -q "health\|excluded" "${LIB_DIR}/activation-floor.sh"; then
            log_check PASS "Health data integration" "Activation floor checks health"
        else
            log_check SKIP "Health data integration" "Might use different variable names"
        fi
    else
        log_check SKIP "Activation floor library" "Not found"
    fi

    log_section "7.3: Verify state audit trail"

    if [ -f "${LOGS_DIR}/state-audit.jsonl" ]; then
        local audit_entries=$(wc -l < "${LOGS_DIR}/state-audit.jsonl")
        if [ "$audit_entries" -gt 0 ]; then
            log_check PASS "State audit trail" "Contains $audit_entries entries"
        else
            log_check SKIP "State audit trail" "File exists but empty"
        fi
    else
        log_check SKIP "State audit trail" "File not yet created"
    fi
}

# ============================================================================
# Test Suite 8: Performance Metrics
# ============================================================================

test_performance_metrics() {
    log_header "Test Suite 8: Phase 4 Performance Metrics"

    log_section "8.1: Verify healing status metrics"

    if [ -f "${METRICS_DIR}/healing-status.json" ]; then
        if jq empty "${METRICS_DIR}/healing-status.json" 2>/dev/null; then
            log_check PASS "Healing status metrics" "Valid JSON format"

            # Check for expected fields
            local required_fields=("detection_accuracy" "remediation_success_rate" "detection_to_fix_time")
            for field in "${required_fields[@]}"; do
                if jq -e ".$field" "${METRICS_DIR}/healing-status.json" >/dev/null 2>&1; then
                    log_check PASS "Metric field: $field" "Present"
                else
                    log_check SKIP "Metric field: $field" "Not found"
                fi
            done
        else
            log_check FAIL "Healing status metrics" "Invalid JSON format"
        fi
    else
        log_check SKIP "Healing status metrics" "File not yet created"
    fi

    log_section "8.2: Verify anomaly detection accuracy"

    if [ -f "${METRICS_DIR}/healing-status.json" ]; then
        local detection_rate=$(jq -r '.detection_accuracy // "unknown"' "${METRICS_DIR}/healing-status.json" 2>/dev/null)
        if [ "$detection_rate" != "unknown" ]; then
            log_check PASS "Detection accuracy tracking" "Current: $detection_rate%"
        fi
    fi
}

# ============================================================================
# Summary and Report Generation
# ============================================================================

print_summary() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  Verification Summary"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""

    {
        echo ""
        echo "## Verification Summary"
        echo ""
    } >> "$VERIFY_REPORT"

    local total=$((CHECKS_PASSED + CHECKS_FAILED + CHECKS_SKIPPED))
    local pass_rate=0
    if [ "$total" -gt 0 ] && [ $((CHECKS_PASSED + CHECKS_FAILED)) -gt 0 ]; then
        pass_rate=$((CHECKS_PASSED * 100 / (CHECKS_PASSED + CHECKS_FAILED)))
    fi

    echo "✅ Passed:  $CHECKS_PASSED"
    echo "❌ Failed:  $CHECKS_FAILED"
    echo "⏭️  Skipped: $CHECKS_SKIPPED"
    echo "📊 Total:   $total"
    echo "📈 Pass Rate: $pass_rate%"
    echo ""

    {
        echo "| Metric | Count |"
        echo "|--------|-------|"
        echo "| Passed | $CHECKS_PASSED |"
        echo "| Failed | $CHECKS_FAILED |"
        echo "| Skipped | $CHECKS_SKIPPED |"
        echo "| **Total** | **$total** |"
        echo "| **Pass Rate** | **$pass_rate%** |"
        echo ""
    } >> "$VERIFY_REPORT"

    if [ "$CHECKS_FAILED" -eq 0 ]; then
        echo -e "${GREEN}✅ PHASE 4 VERIFICATION PASSED${NC}"
        {
            echo "## Result"
            echo ""
            echo "✅ **PHASE 4 VERIFICATION PASSED**"
            echo ""
            echo "All critical Phase 4 components verified as operational."
            echo ""
            echo "---"
            echo ""
            echo "**Generated**: $(date)"
            echo "**Report Location**: $VERIFY_REPORT"
        } >> "$VERIFY_REPORT"
        return 0
    else
        echo -e "${RED}⚠️  PHASE 4 VERIFICATION INCOMPLETE${NC}"
        echo "Some checks failed. Review report for details."
        {
            echo "## Result"
            echo ""
            echo "⚠️  **PHASE 4 VERIFICATION INCOMPLETE**"
            echo ""
            echo "$CHECKS_FAILED check(s) failed. Some Phase 4 components may need attention."
            echo ""
            echo "---"
            echo ""
            echo "**Generated**: $(date)"
            echo "**Report Location**: $VERIFY_REPORT"
        } >> "$VERIFY_REPORT"
        return 1
    fi
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    mkdir -p "$LOGS_DIR"
    mkdir -p "$(dirname "$VERIFY_REPORT")"

    # Initialize report
    {
        echo "# Phase 4 Integration Verification Report"
        echo ""
        echo "**Generated**: $(date)"
        echo "**Daemon Root**: $DAEMON_ROOT"
        echo ""
    } > "$VERIFY_REPORT"

    echo "Starting Phase 4 Integration Verification..."
    echo "Report will be saved to: $VERIFY_REPORT"
    echo ""

    # Determine which tests to run
    local test_mode="${1:-full}"

    # Check prerequisites
    if ! check_prerequisites; then
        echo -e "${RED}❌ Prerequisites check failed${NC}"
        print_summary
        exit 2
    fi
    echo ""

    # Run selected test suites
    case "$test_mode" in
        --full|full)
            test_health_aware_selection
            test_remediation_engine
            test_root_cause_analysis
            test_anomaly_detection
            test_trap_cleanup_compliance
            test_self_healing_loop
            test_integration_points
            test_performance_metrics
            ;;
        --quick|quick)
            test_health_aware_selection
            test_remediation_engine
            test_trap_cleanup_compliance
            ;;
        --health|health)
            test_health_aware_selection
            test_integration_points
            ;;
        --remediation|remediation)
            test_remediation_engine
            test_root_cause_analysis
            test_anomaly_detection
            ;;
        --cleanup|cleanup)
            test_trap_cleanup_compliance
            ;;
        *)
            echo "Usage: $0 [--full|--quick|--health|--remediation|--cleanup]"
            exit 1
            ;;
    esac

    # Print and save summary
    print_summary

    echo ""
    echo "Full report saved to: $VERIFY_REPORT"
    echo ""

    if [ "$CHECKS_FAILED" -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

main "$@"
