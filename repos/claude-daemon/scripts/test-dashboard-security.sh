#!/bin/bash
# Dashboard security regression tests
# Maintainer: Verifies security fixes remain in place
#
# Purpose:
# - Test XSS protection (input validation + output escaping)
# - Test temp file security (unpredictable names)
# - Test file permissions (600)
# - Catch security regressions before production
#
# Usage:
#   ./test-dashboard-security.sh           # Run all tests
#   ./test-dashboard-security.sh --verbose # Show detailed output

set -euo pipefail

DAEMON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${DAEMON_ROOT}/lib/dashboard-updates.sh"

VERBOSE="${1:-}"
TESTS_PASSED=0
TESTS_FAILED=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

test_pass() {
    TESTS_PASSED=$((TESTS_PASSED + 1))
    echo -e "${GREEN}✓${NC} $1"
}

test_fail() {
    TESTS_FAILED=$((TESTS_FAILED + 1))
    echo -e "${RED}✗${NC} $1"
    [[ -n "$VERBOSE" ]] && echo "  Details: $2"
}

test_info() {
    if [[ -n "$VERBOSE" ]]; then
        echo "  $1"
    fi
    return 0
}

# Test 1: XSS via script tags should be rejected
test_xss_script_tag() {
    local result
    if result=$(update_current_activity "test" "<script>alert(1)</script>" "test" "test" 2>&1); then
        test_fail "XSS: <script> tag not rejected" "$result"
    else
        if [[ "$result" =~ "contains HTML tags" ]]; then
            test_pass "XSS: <script> tag rejected"
        else
            test_fail "XSS: Wrong error message for <script>" "$result"
        fi
    fi
}

# Test 2: XSS via img onerror should be rejected
test_xss_onerror() {
    local result
    if result=$(update_current_activity "test" "<img src=x onerror=alert(1)>" "test" "test" 2>&1); then
        test_fail "XSS: onerror attribute not rejected" "$result"
    else
        if [[ "$result" =~ "contains JavaScript" ]]; then
            test_pass "XSS: onerror attribute rejected"
        else
            test_fail "XSS: Wrong error message for onerror" "$result"
        fi
    fi
}

# Test 3: XSS via javascript: protocol should be rejected
test_xss_javascript_protocol() {
    local result
    if result=$(update_mood "javascript:alert(1)" "😈" "test" 0 2>&1); then
        test_fail "XSS: javascript: protocol not rejected" "$result"
    else
        if [[ "$result" =~ "contains JavaScript" ]]; then
            test_pass "XSS: javascript: protocol rejected"
        else
            test_fail "XSS: Wrong error message for javascript:" "$result"
        fi
    fi
}

# Test 4: XSS via onclick should be rejected
test_xss_onclick() {
    local result
    if result=$(add_decision "test" "test onclick=alert(1)" "test" "test" 2>&1); then
        test_fail "XSS: onclick attribute not rejected" "$result"
    else
        if [[ "$result" =~ "contains JavaScript" ]]; then
            test_pass "XSS: onclick attribute rejected"
        else
            test_fail "XSS: Wrong error message for onclick" "$result"
        fi
    fi
}

# Test 5: XSS via data URI should be rejected
test_xss_data_uri() {
    local result
    if result=$(add_insight "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==" "test" "test" 2>&1); then
        test_fail "XSS: data URI not rejected" "$result"
    else
        if [[ "$result" =~ "contains data URI" ]]; then
            test_pass "XSS: data URI rejected"
        else
            test_fail "XSS: Wrong error message for data URI" "$result"
        fi
    fi
}

# Test 6: Normal HTML-like text (not actual tags) should be rejected
test_xss_html_entities() {
    local result
    if result=$(update_curiosity "test" "What about <tag> structure?" "test" 2>&1); then
        test_fail "XSS: HTML-like text not rejected" "$result"
    else
        if [[ "$result" =~ "contains HTML tags" ]]; then
            test_pass "XSS: HTML-like text rejected (conservative)"
        else
            test_fail "XSS: Wrong error message for HTML-like text" "$result"
        fi
    fi
}

# Test 7: Legitimate content should pass
test_legitimate_content() {
    local result
    if result=$(update_current_activity "test" "Fixing XSS vulnerabilities" "Security is important" "Focused" 2>&1); then
        test_pass "Legitimate content accepted"
    else
        test_fail "Legitimate content rejected" "$result"
    fi
}

# Test 8: File permissions are correct
test_file_permissions() {
    local perms=$(stat -c "%a" "$DASHBOARD_STATE" 2>/dev/null || stat -f "%Lp" "$DASHBOARD_STATE" 2>/dev/null)

    if [[ "$perms" == "600" ]]; then
        test_pass "File permissions: dashboard-state.json is 600"
    else
        test_fail "File permissions: dashboard-state.json is $perms (should be 600)" ""
    fi

    if [[ -f "$DASHBOARD_LOCK" ]]; then
        local lock_perms=$(stat -c "%a" "$DASHBOARD_LOCK" 2>/dev/null || stat -f "%Lp" "$DASHBOARD_LOCK" 2>/dev/null)
        if [[ "$lock_perms" == "600" ]]; then
            test_pass "File permissions: .dashboard-state.lock is 600"
        else
            test_fail "File permissions: .dashboard-state.lock is $lock_perms (should be 600)" ""
        fi
    fi
}

# Test 9: escapeHtml function exists in dashboard.html
test_escape_function_exists() {
    if grep -q "function escapeHtml" "${DAEMON_ROOT}/dashboard.html"; then
        test_pass "Output escaping: escapeHtml() function exists"
    else
        test_fail "Output escaping: escapeHtml() function missing" ""
    fi
}

# Test 10: escapeHtml is actually used
test_escape_function_used() {
    local usage_count=$(grep -c "escapeHtml(" "${DAEMON_ROOT}/dashboard.html" || echo "0")

    if [[ "$usage_count" -ge 20 ]]; then
        test_pass "Output escaping: escapeHtml() used $usage_count times (good coverage)"
    elif [[ "$usage_count" -ge 10 ]]; then
        test_fail "Output escaping: escapeHtml() only used $usage_count times (may be incomplete)" ""
    else
        test_fail "Output escaping: escapeHtml() only used $usage_count times (insufficient)" ""
    fi
}

# Test 11: Temp files use mktemp
test_temp_file_security() {
    if grep -q "mktemp" "${DAEMON_ROOT}/lib/dashboard-updates.sh"; then
        test_pass "Temp file security: mktemp is used"
    else
        test_fail "Temp file security: mktemp not found (may use predictable names)" ""
    fi

    # Check for vulnerable pattern
    if grep -q '\.tmp' "${DAEMON_ROOT}/lib/dashboard-updates.sh"; then
        test_fail "Temp file security: .tmp pattern found (potentially vulnerable)" ""
    else
        test_pass "Temp file security: No .tmp pattern found"
    fi
}

# Test 12: Cleanup traps exist
test_cleanup_traps() {
    local trap_count=$(grep -c "trap.*rm -f" "${DAEMON_ROOT}/lib/dashboard-updates.sh" || echo "0")

    if [[ "$trap_count" -ge 6 ]]; then
        test_pass "Temp file cleanup: $trap_count cleanup traps found (all functions protected)"
    elif [[ "$trap_count" -ge 3 ]]; then
        test_fail "Temp file cleanup: Only $trap_count cleanup traps (should be 6)" ""
    else
        test_fail "Temp file cleanup: Insufficient cleanup traps ($trap_count)" ""
    fi
}

# Main execution
main() {
    echo "Dashboard Security Regression Tests"
    echo "===================================="
    echo ""

    # Backup current dashboard state
    BACKUP="${DASHBOARD_STATE}.test-backup.$(date +%s)"
    cp "$DASHBOARD_STATE" "$BACKUP"
    trap "test -f '$BACKUP' && mv '$BACKUP' '$DASHBOARD_STATE' || true" EXIT ERR INT TERM
    test_info "Created backup: $BACKUP"

    echo "Running XSS Protection Tests..."
    test_xss_script_tag
    test_xss_onerror
    test_xss_javascript_protocol
    test_xss_onclick
    test_xss_data_uri
    test_xss_html_entities
    test_legitimate_content

    echo ""
    echo "Running File Security Tests..."
    test_file_permissions
    test_temp_file_security
    test_cleanup_traps

    echo ""
    echo "Running Output Escaping Tests..."
    test_escape_function_exists
    test_escape_function_used

    # Restore dashboard state
    mv "$BACKUP" "$DASHBOARD_STATE"

    echo ""
    echo "===================================="
    echo "Results: $TESTS_PASSED passed, $TESTS_FAILED failed"

    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}All security tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}Some security tests failed. Review fixes before deployment.${NC}"
        exit 1
    fi
}

main
