#!/bin/bash
# Sync state.json from success-rates.json to fix historical data
# Created by: The Experimenter
# Rationale: state.json wasn't being updated, so we need to backfill from success-rates.json

set -euo pipefail

STATE_FILE="/home/opc/.claude/daemon/personalities/state.json"
RATES_FILE="/home/opc/.claude/daemon/metrics/success-rates.json"

echo "=== Syncing state.json from success-rates.json ==="
echo

# Backup first (experimenter best practice: break things, but keep backups!)
cp "$STATE_FILE" "${STATE_FILE}.backup.$(date +%s)"
echo "✅ Created backup: ${STATE_FILE}.backup.$(date +%s)"

# For each persona, sync the counts
for persona in experimenter skeptic optimizer architect maintainer auditor; do
    echo
    echo "Syncing $persona..."

    completed=$(jq -r ".personas.$persona.completed // 0" "$RATES_FILE")
    failed=$(jq -r ".personas.$persona.failed // 0" "$RATES_FILE")

    echo "  completed: $completed"
    echo "  failed: $failed"

    # Update state.json
    temp_file=$(mktemp)
    jq --arg p "$persona" \
       --argjson c "$completed" \
       --argjson f "$failed" \
       --argjson a "$((completed + failed))" \
       '.personas[$p].tasks_completed = $c |
        .personas[$p].tasks_failed = $f |
        .personas[$p].total_activations = $a |
        .personas[$p].last_active = (if ($a > 0) then (now | todate) else null end)' \
       "$STATE_FILE" > "$temp_file"
    mv "$temp_file" "$STATE_FILE"

    echo "  ✅ Updated"
done

echo
echo "=== Sync Complete ==="
echo
echo "Verification:"
jq '.personas | to_entries | map({persona: .key, activations: .value.total_activations, completed: .value.tasks_completed, failed: .value.tasks_failed})' "$STATE_FILE"
