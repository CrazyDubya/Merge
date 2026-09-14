#!/bin/bash
# Run All Daemon Tests
# Usage: ./tests/run-all-tests.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAEMON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export DAEMON_ROOT

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

TOTAL_SUITES=0
PASSED_SUITES=0
FAILED_SUITES=0

echo "========================================"
echo "     Daemon Test Suite Runner"
echo "========================================"
echo ""
echo "DAEMON_ROOT: $DAEMON_ROOT"
echo ""

run_test_suite() {
    local test_file="$1"
    local test_name
    test_name=$(basename "$test_file" .sh)

    TOTAL_SUITES=$((TOTAL_SUITES + 1))

    echo "----------------------------------------"
    echo -e "${YELLOW}Running: $test_name${NC}"
    echo "----------------------------------------"

    if bash "$test_file"; then
        PASSED_SUITES=$((PASSED_SUITES + 1))
        echo -e "${GREEN}Suite PASSED: $test_name${NC}"
    else
        FAILED_SUITES=$((FAILED_SUITES + 1))
        echo -e "${RED}Suite FAILED: $test_name${NC}"
    fi
    echo ""
}

# Run unit tests
echo "=== UNIT TESTS ==="
echo ""

for test_file in "$SCRIPT_DIR"/unit/test-*.sh; do
    if [[ -f "$test_file" ]]; then
        run_test_suite "$test_file"
    fi
done

# Run integration tests
if [[ -d "$SCRIPT_DIR/integration" ]]; then
    echo "=== INTEGRATION TESTS ==="
    echo ""

    for test_file in "$SCRIPT_DIR"/integration/test-*.sh; do
        if [[ -f "$test_file" ]]; then
            run_test_suite "$test_file"
        fi
    done
fi

# Run legacy tests from scripts/
echo "=== LEGACY TESTS (scripts/) ==="
echo ""

for test_file in "$DAEMON_ROOT"/scripts/test-*.sh; do
    if [[ -f "$test_file" ]]; then
        run_test_suite "$test_file"
    fi
done

# Summary
echo "========================================"
echo "           FINAL RESULTS"
echo "========================================"
echo ""
echo "Test Suites: $PASSED_SUITES/$TOTAL_SUITES passed"

if [[ $FAILED_SUITES -gt 0 ]]; then
    echo -e "${RED}$FAILED_SUITES suite(s) FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}All test suites PASSED!${NC}"
    exit 0
fi
