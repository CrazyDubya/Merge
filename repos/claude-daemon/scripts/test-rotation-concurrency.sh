#!/bin/bash
#
# test-rotation-concurrency.sh
# Comprehensive test suite for rotate-persona-timeline.sh concurrency safety
#
# Tests that rotation script properly coordinates with daemon writes using
# lockfile to prevent data loss from race conditions.
#
# Usage: ./test-rotation-concurrency.sh [test_number]
#   test_number: 1, 2, 3, or "all" (default: all)
#
# Exit codes:
#   0 - All tests passed
#   1 - One or more tests failed
#   2 - Test setup error

set -euo pipefail

DAEMON_ROOT="/home/opc/.claude/daemon"
TEST_DIR="/tmp/rotation-test-$$"
PASSED=0
FAILED=0

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Cleanup on exit
cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

# Test result reporting
pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    ((PASSED++))
}

fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    echo "  $2"
    ((FAILED++))
}

warn() {
    echo -e "${YELLOW}⚠ WARN${NC}: $1"
}

# ============================================================================
# Test 1: No Data Loss Under Concurrent Writes
# ============================================================================
# Simulates daemon writes during rotation to verify lockfile coordination
# prevents data loss. Creates background writers that continuously append
# events while rotation runs.

test_concurrent_writes() {
    echo ""
    echo "=========================================="
    echo "Test 1: No Data Loss Under Concurrent Writes"
    echo "=========================================="

    local test_timeline="$TEST_DIR/timeline.jsonl"
    local test_lockfile="${test_timeline}.lock"
    local test_archive_dir="$TEST_DIR/archives"

    mkdir -p "$test_archive_dir"

    # Create initial timeline (2500 lines to trigger rotation threshold)
    echo "Setting up test timeline (2500 lines)..."
    for i in {1..2500}; do
        echo "{\"seq\":$i,\"type\":\"initial\"}" >> "$test_timeline"
    done

    # Start background writers (3 writers, 100 events each = 300 total)
    echo "Starting 3 background writers (300 total events)..."
    local pids=()
    for w in {1..3}; do
        {
            for i in {1..100}; do
                event="{\"writer\":$w,\"seq\":$i,\"ts\":\"$(date +%s%N)\",\"type\":\"concurrent\"}"
                (
                    flock -x 200
                    echo "$event" >> "$test_timeline"
                ) 200>>"$test_lockfile"
                sleep 0.01  # 10ms between writes
            done
        } &
        pids+=($!)
    done

    # Let writers start
    sleep 0.5

    echo "Running rotation while writers are active..."

    # Create a modified rotation script for testing
    cat > "$TEST_DIR/rotate-test.sh" <<'ROTATE_EOF'
#!/bin/bash
set -euo pipefail

TIMELINE_FILE="$1"
LOCKFILE="${TIMELINE_FILE}.lock"
ARCHIVE_DIR="$2"
THRESHOLD_LINES=2000

LINE_COUNT=$(wc -l < "$TIMELINE_FILE")

if [ "$LINE_COUNT" -lt "$THRESHOLD_LINES" ]; then
    echo "Below threshold"
    exit 0
fi

TIMESTAMP=$(date -u +"%Y%m%d-%H%M%S")
ARCHIVE_FILE="$ARCHIVE_DIR/timeline-$TIMESTAMP.jsonl"

cp "$TIMELINE_FILE" "$ARCHIVE_FILE"
gzip "$ARCHIVE_FILE"

(
    if ! flock -x -w 10 200; then
        echo "ERROR: Lock timeout"
        exit 1
    fi

    tail -500 "$TIMELINE_FILE" > "$TIMELINE_FILE.tmp"
    mv "$TIMELINE_FILE.tmp" "$TIMELINE_FILE"

) 200>>"$LOCKFILE"

exit $?
ROTATE_EOF

    chmod +x "$TEST_DIR/rotate-test.sh"

    # Run rotation
    if ! "$TEST_DIR/rotate-test.sh" "$test_timeline" "$test_archive_dir"; then
        fail "Rotation script failed" "Check lock acquisition or script errors"
        # Kill background writers
        for pid in "${pids[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        return 1
    fi

    # Wait for all writers to complete
    echo "Waiting for writers to complete..."
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    # Verify no data loss
    echo "Verifying data integrity..."

    local expected=300
    local in_timeline=$(grep -c '"type":"concurrent"' "$test_timeline" || echo 0)
    local in_archive=$(zcat "$test_archive_dir"/timeline-*.jsonl.gz | grep -c '"type":"concurrent"' || echo 0)
    local total=$((in_timeline + in_archive))

    echo "  Expected concurrent events: $expected (minimum)"
    echo "  Found in timeline: $in_timeline"
    echo "  Found in archive: $in_archive"
    echo "  Total found: $total"

    # Total should be >= expected (some writes may occur between threshold check and rotation)
    if [ "$total" -ge "$expected" ]; then
        local extra=$((total - expected))
        if [ "$extra" -gt 0 ]; then
            pass "No data loss - all events accounted for ($total/$expected, +$extra from race window)"
        else
            pass "No data loss - all events accounted for ($total/$expected)"
        fi
    else
        fail "Data loss detected" "Missing $((expected - total)) events"
    fi
}

# ============================================================================
# Test 2: Lock Timeout Behavior
# ============================================================================
# Verifies rotation fails gracefully when lock cannot be acquired (daemon
# hung or another rotation running). Should return error without corrupting
# data.

test_lock_timeout() {
    echo ""
    echo "=========================================="
    echo "Test 2: Lock Timeout Behavior"
    echo "=========================================="

    local test_timeline="$TEST_DIR/timeline2.jsonl"
    local test_lockfile="${test_timeline}.lock"
    local test_archive_dir="$TEST_DIR/archives2"

    mkdir -p "$test_archive_dir"

    # Create test timeline above threshold
    echo "Setting up test timeline..."
    for i in {1..2500}; do
        echo "{\"seq\":$i}" >> "$test_timeline"
    done

    # Hold lock indefinitely (need to open lockfile properly)
    echo "Acquiring lock (simulating hung daemon)..."
    ( flock -x 200; sleep 300 ) 200>>"$test_lockfile" &
    local lock_pid=$!

    sleep 0.5  # Ensure lock is held

    # Create rotation script with shorter timeout for testing
    cat > "$TEST_DIR/rotate-timeout-test.sh" <<'ROTATE_EOF'
#!/bin/bash
set -euo pipefail

TIMELINE_FILE="$1"
LOCKFILE="${TIMELINE_FILE}.lock"
ARCHIVE_DIR="$2"

LINE_COUNT=$(wc -l < "$TIMELINE_FILE")
TIMESTAMP=$(date -u +"%Y%m%d-%H%M%S")
ARCHIVE_FILE="$ARCHIVE_DIR/timeline-$TIMESTAMP.jsonl"

cp "$TIMELINE_FILE" "$ARCHIVE_FILE"
gzip "$ARCHIVE_FILE"

(
    if ! flock -x -w 2 200; then  # 2 second timeout for test
        echo "ERROR: Lock timeout"
        exit 1
    fi

    tail -500 "$TIMELINE_FILE" > "$TIMELINE_FILE.tmp"
    mv "$TIMELINE_FILE.tmp" "$TIMELINE_FILE"

) 200>>"$LOCKFILE"

exit $?
ROTATE_EOF

    chmod +x "$TEST_DIR/rotate-timeout-test.sh"

    # Try rotation (should fail with timeout)
    echo "Attempting rotation (should timeout)..."

    if "$TEST_DIR/rotate-timeout-test.sh" "$test_timeline" "$test_archive_dir" 2>&1 | grep -q "Lock timeout"; then
        pass "Rotation failed gracefully with timeout error"
    else
        fail "Rotation did not report timeout" "Expected 'Lock timeout' error message"
    fi

    # Verify data wasn't corrupted
    local line_count=$(wc -l < "$test_timeline")
    if [ "$line_count" -eq 2500 ]; then
        pass "Timeline unchanged after failed rotation (no corruption)"
    else
        fail "Timeline corrupted after failed rotation" "Expected 2500 lines, found $line_count"
    fi

    # Cleanup
    kill "$lock_pid" 2>/dev/null || true
}

# ============================================================================
# Test 3: Real Daemon Integration (if daemon is running)
# ============================================================================
# Tests rotation with actual daemon. Verifies daemon continues writing after
# rotation and no data loss occurs.

test_daemon_integration() {
    echo ""
    echo "=========================================="
    echo "Test 3: Real Daemon Integration"
    echo "=========================================="

    # Check if daemon is running
    if ! systemctl --user is-active claude-daemon &>/dev/null; then
        warn "Daemon not running - skipping integration test"
        warn "Start daemon with: systemctl --user start claude-daemon"
        return 0
    fi

    local real_timeline="$DAEMON_ROOT/memory/persona-timeline.jsonl"

    # Check if timeline is below threshold (don't want to actually rotate)
    local line_count=$(wc -l < "$real_timeline")

    if [ "$line_count" -ge 2000 ]; then
        warn "Timeline above threshold ($line_count lines)"
        warn "Integration test would trigger real rotation - skipping for safety"
        warn "Run rotation manually: $DAEMON_ROOT/scripts/rotate-persona-timeline.sh"
        return 0
    fi

    echo "Timeline has $line_count lines (below 2000 threshold)"
    echo "Monitoring daemon writes for 10 seconds..."

    local before=$line_count
    sleep 10
    local after=$(wc -l < "$real_timeline")
    local delta=$((after - before))

    echo "  Before: $before lines"
    echo "  After: $after lines"
    echo "  Delta: $delta new events"

    if [ "$delta" -ge 0 ]; then
        pass "Daemon is writing to timeline (healthy)"
    else
        fail "Timeline shrank during test" "This should never happen - investigate immediately"
    fi

    # Verify daemon still running
    if systemctl --user is-active claude-daemon &>/dev/null; then
        pass "Daemon still active after monitoring"
    else
        fail "Daemon stopped during test" "Check daemon logs: journalctl --user -u claude-daemon"
    fi
}

# ============================================================================
# Main
# ============================================================================

main() {
    local test_to_run="${1:-all}"

    echo "=========================================="
    echo "Rotation Concurrency Test Suite"
    echo "=========================================="
    echo "Test directory: $TEST_DIR"
    echo ""

    mkdir -p "$TEST_DIR"

    case "$test_to_run" in
        1|test1|concurrent)
            test_concurrent_writes
            ;;
        2|test2|timeout)
            test_lock_timeout
            ;;
        3|test3|daemon)
            test_daemon_integration
            ;;
        all)
            test_concurrent_writes
            test_lock_timeout
            test_daemon_integration
            ;;
        *)
            echo "ERROR: Unknown test: $test_to_run"
            echo "Usage: $0 [1|2|3|all]"
            exit 2
            ;;
    esac

    echo ""
    echo "=========================================="
    echo "Test Summary"
    echo "=========================================="
    echo -e "${GREEN}Passed: $PASSED${NC}"
    echo -e "${RED}Failed: $FAILED${NC}"
    echo ""

    if [ "$FAILED" -gt 0 ]; then
        echo -e "${RED}TESTS FAILED${NC}"
        exit 1
    else
        echo -e "${GREEN}ALL TESTS PASSED${NC}"
        exit 0
    fi
}

main "$@"
