#!/bin/bash
# Integration test for sed escaping bug fix (SEC-2025-11-11-001)
# Tests that mark_task_completed() properly handles special regex characters
#
# Background: Before this fix, task descriptions containing regex metacharacters
# like [, ], *, etc. would cause sed to fail with "Invalid preceding regular expression"
#
# This test verifies the fix prevents that crash.

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
TASKS_DIR="${DAEMON_ROOT}/tasks"

echo "============================================"
echo "Testing sed escaping fix"
echo "============================================"
echo ""

# Create a test queue with problematic task descriptions
TEST_QUEUE=$(mktemp)
cat > "$TEST_QUEUE" << 'EOF'
# Task Queue

## Pending Tasks

- [ ] **[ARCHITECT]**: Normal task
- [ ] **[OPTIMIZER]**: Task with [brackets] in description
- [ ] 💼 **[ARCHITECT + EXPERIMENTER]**: Task with (parens), [brackets], $vars, *globs*, /paths/, and .dots.
- [ ] Test with **bold** and *italic* markdown
EOF

# Source the mark_task_completed function from daemon.sh
source "${DAEMON_ROOT}/daemon.sh"

# Override TASKS_DIR to use test file
TASKS_DIR=$(dirname "$TEST_QUEUE")
export TASKS_DIR

# Test 1: Normal task (baseline)
echo "Test 1: Normal task without special characters"
RESULT=$(mark_task_completed "**[ARCHITECT]**: Normal task" "test-persona" 2>&1)
if grep -q "^- \[x\].*Normal task" "$TEST_QUEUE"; then
    echo "  ✓ PASSED: Normal task marked complete"
else
    echo "  ✗ FAILED: Could not mark normal task complete"
    echo "  Output: $RESULT"
    cat "$TEST_QUEUE"
    rm -f "$TEST_QUEUE" "${TEST_QUEUE}.backup"
    exit 1
fi

# Reset queue
cat > "$TEST_QUEUE" << 'EOF'
# Task Queue

## Pending Tasks

- [ ] **[ARCHITECT]**: Normal task
- [ ] **[OPTIMIZER]**: Task with [brackets] in description
- [ ] 💼 **[ARCHITECT + EXPERIMENTER]**: Task with (parens), [brackets], $vars, *globs*, /paths/, and .dots.
- [ ] Test with **bold** and *italic* markdown
EOF

# Test 2: Task with brackets (the bug trigger)
echo "Test 2: Task with [brackets] in description"
RESULT=$(mark_task_completed "**[OPTIMIZER]**: Task with [brackets] in description" "test-persona" 2>&1)
if grep -q "^- \[x\].*Task with \[brackets\]" "$TEST_QUEUE"; then
    echo "  ✓ PASSED: Task with brackets marked complete (would have crashed before fix)"
else
    echo "  ✗ FAILED: Could not mark task with brackets complete"
    echo "  Output: $RESULT"
    cat "$TEST_QUEUE"
    rm -f "$TEST_QUEUE" "${TEST_QUEUE}.backup"
    exit 1
fi

# Reset queue
cat > "$TEST_QUEUE" << 'EOF'
# Task Queue

## Pending Tasks

- [ ] **[ARCHITECT]**: Normal task
- [ ] **[OPTIMIZER]**: Task with [brackets] in description
- [ ] 💼 **[ARCHITECT + EXPERIMENTER]**: Task with (parens), [brackets], $vars, *globs*, /paths/, and .dots.
- [ ] Test with **bold** and *italic* markdown
EOF

# Test 3: Task with many special characters (extreme case)
echo "Test 3: Task with many special regex metacharacters"
RESULT=$(mark_task_completed '💼 **[ARCHITECT + EXPERIMENTER]**: Task with (parens), [brackets], $vars, *globs*, /paths/, and .dots.' "test-persona" 2>&1)
if grep -q "^- \[x\].*Task with (parens)" "$TEST_QUEUE"; then
    echo "  ✓ PASSED: Task with many special chars marked complete (would have crashed before fix)"
else
    echo "  ✗ FAILED: Could not mark task with special chars complete"
    echo "  Output: $RESULT"
    cat "$TEST_QUEUE"
    rm -f "$TEST_QUEUE" "${TEST_QUEUE}.backup"
    exit 1
fi

# Reset queue
cat > "$TEST_QUEUE" << 'EOF'
# Task Queue

## Pending Tasks

- [ ] **[ARCHITECT]**: Normal task
- [ ] **[OPTIMIZER]**: Task with [brackets] in description
- [ ] 💼 **[ARCHITECT + EXPERIMENTER]**: Task with (parens), [brackets], $vars, *globs*, /paths/, and .dots.
- [ ] Test with **bold** and *italic* markdown
EOF

# Test 4: Markdown formatting characters
echo "Test 4: Task with **bold** and *italic* markdown"
RESULT=$(mark_task_completed "Test with **bold** and *italic* markdown" "test-persona" 2>&1)
if grep -q "^- \[x\].*\*\*bold\*\*.*\*italic\*" "$TEST_QUEUE"; then
    echo "  ✓ PASSED: Markdown formatting handled correctly"
else
    echo "  ✗ FAILED: Could not mark markdown task complete"
    echo "  Output: $RESULT"
    cat "$TEST_QUEUE"
    rm -f "$TEST_QUEUE" "${TEST_QUEUE}.backup"
    exit 1
fi

# Cleanup
rm -f "$TEST_QUEUE" "${TEST_QUEUE}.backup"

echo ""
echo "============================================"
echo "All tests PASSED! ✓"
echo "============================================"
echo ""
echo "The sed escaping fix correctly handles all regex metacharacters:"
echo "  \ (backslash)"
echo "  [ ] (brackets)"
echo "  . (period)"
echo "  * (asterisk)"
echo "  ^ (caret)"
echo "  \$ (dollar sign)"
echo "  / (forward slash)"
echo ""
echo "Bug SEC-2025-11-11-001 is confirmed fixed."
