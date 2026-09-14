#!/bin/bash
# Test script for sed escaping bug fix
# Tests that task descriptions with special regex characters are properly escaped

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
TASKS_DIR="${DAEMON_ROOT}/tasks"

# Create test queue file
TEST_QUEUE=$(mktemp)
cat > "$TEST_QUEUE" << 'EOF'
# Task Queue

## Pending Tasks

- [ ] **[ARCHITECT]**: Normal task without special characters
- [ ] **[OPTIMIZER]**: Task with [brackets] and (parentheses)
- [ ] 💼 **[ARCHITECT + EXPERIMENTER + SKEPTIC]**: Very long task with lots of special chars: (parens), [brackets], $dollars, *asterisks*, ^carets, /slashes/, and .periods.
- [ ] Simple task
EOF

echo "Original queue:"
cat "$TEST_QUEUE"
echo ""
echo "================================"
echo ""

# Source the escaping logic from daemon.sh
escape_task() {
    local task="$1"
    local escaped_task
    escaped_task=$(printf "%s" "$task" | sed 's/\\/\\\\/g' | sed 's/\[/\\[/g' | sed 's/\]/\\]/g' | sed 's/\./\\./g' | sed 's/\*/\\*/g' | sed 's/\^/\\^/g' | sed 's/\$/\\$/g' | sed 's/\//\\\//g')
    echo "$escaped_task"
}

# Test 1: Normal task
echo "Test 1: Normal task without special characters"
TASK1="**[ARCHITECT]**: Normal task without special characters"
ESCAPED1=$(escape_task "$TASK1")
TEMP1=$(mktemp)
sed "0,/^- \[ \] ${ESCAPED1}/s//- [x] ${ESCAPED1}/" "$TEST_QUEUE" > "$TEMP1"
if grep -q "^- \[x\] $TASK1" "$TEMP1"; then
    echo "✓ Test 1 PASSED"
else
    echo "✗ Test 1 FAILED"
    exit 1
fi

# Test 2: Task with brackets
echo "Test 2: Task with [brackets] and (parentheses)"
TASK2="**[OPTIMIZER]**: Task with [brackets] and (parentheses)"
ESCAPED2=$(escape_task "$TASK2")
TEMP2=$(mktemp)
sed "0,/^- \[ \] ${ESCAPED2}/s//- [x] ${ESCAPED2}/" "$TEST_QUEUE" > "$TEMP2"
if grep -q "^- \[x\] .*\[OPTIMIZER\].*\[brackets\]" "$TEMP2"; then
    echo "✓ Test 2 PASSED"
else
    echo "✗ Test 2 FAILED"
    exit 1
fi

# Test 3: Very long task with many special characters (the one that caused the bug)
echo "Test 3: Long task with many special characters"
TASK3='💼 **[ARCHITECT + EXPERIMENTER + SKEPTIC]**: Very long task with lots of special chars: (parens), [brackets], $dollars, *asterisks*, ^carets, /slashes/, and .periods.'
ESCAPED3=$(escape_task "$TASK3")
TEMP3=$(mktemp)
sed "0,/^- \[ \] ${ESCAPED3}/s//- [x] ${ESCAPED3}/" "$TEST_QUEUE" > "$TEMP3"
if grep -q "^- \[x\].*ARCHITECT.*EXPERIMENTER.*SKEPTIC.*special chars" "$TEMP3"; then
    echo "✓ Test 3 PASSED (this would have failed before the fix)"
else
    echo "✗ Test 3 FAILED"
    cat "$TEMP3"
    exit 1
fi

# Test 4: Simple task
echo "Test 4: Simple task"
TASK4="Simple task"
ESCAPED4=$(escape_task "$TASK4")
TEMP4=$(mktemp)
sed "0,/^- \[ \] ${ESCAPED4}/s//- [x] ${ESCAPED4}/" "$TEST_QUEUE" > "$TEMP4"
if grep -q "^- \[x\] Simple task" "$TEMP4"; then
    echo "✓ Test 4 PASSED"
else
    echo "✗ Test 4 FAILED"
    exit 1
fi

# Cleanup
rm -f "$TEST_QUEUE" "$TEMP1" "$TEMP2" "$TEMP3" "$TEMP4"

echo ""
echo "================================"
echo "All tests PASSED! ✓"
echo ""
echo "The sed escaping fix correctly handles:"
echo "  - Normal alphanumeric task descriptions"
echo "  - Task descriptions with [brackets]"
echo "  - Task descriptions with (parentheses)"
echo "  - Task descriptions with \$dollar signs"
echo "  - Task descriptions with *asterisks*"
echo "  - Task descriptions with ^carets"
echo "  - Task descriptions with /slashes/"
echo "  - Task descriptions with .periods"
echo "  - Very long task descriptions with multiple special characters"
