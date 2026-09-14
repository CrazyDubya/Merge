#!/bin/bash
# Validate state.json integrity - check for phantom personas

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAEMON_ROOT="$(dirname "$SCRIPT_DIR")"
STATE_FILE="$DAEMON_ROOT/personalities/state.json"

# Valid persona names (lowercase only, no special chars)
VALID_PERSONAS=("auditor" "optimizer" "architect" "experimenter" "maintainer" "skeptic")

# Check if state.json exists
if [ ! -f "$STATE_FILE" ]; then
    echo "ERROR: state.json not found at $STATE_FILE"
    exit 1
fi

# Get all persona keys from state.json
mapfile -t all_personas < <(jq -r '.personas | keys[]' "$STATE_FILE")

# Check each persona
phantom_count=0
for persona in "${all_personas[@]}"; do
    # Check if persona is in valid list
    is_valid=0
    for valid in "${VALID_PERSONAS[@]}"; do
        if [ "$persona" = "$valid" ]; then
            is_valid=1
            break
        fi
    done
    
    if [ $is_valid -eq 0 ]; then
        echo "PHANTOM DETECTED: '$persona'"
        phantom_count=$((phantom_count + 1))
    fi
    
    # Check for invalid characters (should only be lowercase letters)
    if ! [[ "$persona" =~ ^[a-z]+$ ]]; then
        echo "INVALID FORMAT: '$persona' (should only contain lowercase letters)"
        phantom_count=$((phantom_count + 1))
    fi
done

if [ $phantom_count -eq 0 ]; then
    echo "✓ state.json validation PASSED - no phantom personas detected"
    exit 0
else
    echo "✗ state.json validation FAILED - found $phantom_count phantom/invalid personas"
    exit 1
fi
