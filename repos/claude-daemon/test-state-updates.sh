#!/bin/bash
# Test script for state.json persona metrics updates
# Created by: The Experimenter
# Purpose: Verify state.json gets updated correctly

set -euo pipefail

STATE_FILE="/home/opc/.claude/daemon/personalities/state.json"

echo "=== State.json Update Test ==="
echo

# Test 1: Verify file exists
echo "Test 1: File exists"
if [[ -f "$STATE_FILE" ]]; then
    echo "✅ state.json exists"
else
    echo "❌ state.json missing"
    exit 1
fi

# Test 2: Check current metrics
echo
echo "Test 2: Current persona metrics"
echo "Experimenter stats:"
jq '.personas.experimenter | {activations: .total_activations, completed: .tasks_completed, failed: .tasks_failed, last_active}' "$STATE_FILE"

echo
echo "Skeptic stats:"
jq '.personas.skeptic | {activations: .total_activations, completed: .tasks_completed, failed: .tasks_failed, last_active}' "$STATE_FILE"

# Test 3: Simulate task completion update
echo
echo "Test 3: Simulating task completion for experimenter..."
TEMP_FILE=$(mktemp)

jq '.personas.experimenter.tasks_completed += 1 |
    .personas.experimenter.total_activations += 1 |
    .personas.experimenter.last_active = (now | todate)' \
    "$STATE_FILE" > "$TEMP_FILE"

echo "New experimenter stats (DRY RUN):"
jq '.personas.experimenter | {activations: .total_activations, completed: .tasks_completed, failed: .tasks_failed, last_active}' "$TEMP_FILE"

rm "$TEMP_FILE"

# Test 4: Check for staleness
echo
echo "Test 4: Checking for stale data..."
LAST_SWITCH=$(jq -r '.last_switch_time' "$STATE_FILE")
echo "Last switch: $LAST_SWITCH"

CURRENT_PERSONA=$(jq -r '.current_persona' "$STATE_FILE")
echo "Current persona: $CURRENT_PERSONA"

# Compare with success-rates.json
echo
echo "Test 5: Data consistency check"
echo "Comparing state.json vs success-rates.json..."

STATE_TASKS=$(jq '.personas.experimenter.tasks_completed' "$STATE_FILE")
RATES_TASKS=$(jq '.personas.experimenter.completed' /home/opc/.claude/daemon/metrics/success-rates.json)

echo "state.json experimenter tasks: $STATE_TASKS"
echo "success-rates.json experimenter tasks: $RATES_TASKS"

if [[ "$STATE_TASKS" == "$RATES_TASKS" ]]; then
    echo "✅ Data is consistent"
else
    echo "❌ INCONSISTENCY DETECTED!"
    echo "   Difference: $((RATES_TASKS - STATE_TASKS)) tasks not tracked in state.json"
fi

echo
echo "=== Test Complete ==="
