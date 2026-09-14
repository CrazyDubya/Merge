#!/bin/bash
#
# Apply throttle configuration to daemon.sh
# Reads throttle-config.json and updates daemon.sh in-place
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
CONFIG_FILE="${DAEMON_ROOT}/throttle-config.json"
DAEMON_SH="${DAEMON_ROOT}/daemon.sh"
CIRCADIAN_FILE="${DAEMON_ROOT}/triggers/circadian.json"
CHAOS_FILE="${DAEMON_ROOT}/triggers/chaos-config.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: No throttle configuration found at $CONFIG_FILE"
    echo "Run: python3 claude-daemon-throttle-tui.py --apply-sat  (or open TUI)"
    exit 1
fi

echo "Applying throttle configuration to daemon..."

# Read config
MIN_SLEEP=$(jq -r '.min_sleep' "$CONFIG_FILE")
DEFAULT_SLEEP=$(jq -r '.default_sleep' "$CONFIG_FILE")
MAX_SLEEP=$(jq -r '.max_sleep' "$CONFIG_FILE")
NIGHT_SLEEP=$(jq -r '.night_sleep' "$CONFIG_FILE")
TASK_WEIGHT=$(jq -r '.task_weight' "$CONFIG_FILE")
REFLECTION_WEIGHT=$(jq -r '.reflection_weight' "$CONFIG_FILE")
CONVERSATION_WEIGHT=$(jq -r '.conversation_weight' "$CONFIG_FILE")
REFLECTION_COOLDOWN=$(jq -r '.reflection_cooldown' "$CONFIG_FILE")
CHAOS_ENABLED=$(jq -r '.chaos_enabled' "$CONFIG_FILE")
CHAOS_PROB=$(jq -r '.chaos_probability' "$CONFIG_FILE")

# Backup daemon.sh
cp "$DAEMON_SH" "${DAEMON_SH}.backup-$(date +%Y%m%d-%H%M%S)"

# Update sleep values in daemon.sh
sed -i "s/^MIN_SLEEP=.*/MIN_SLEEP=$MIN_SLEEP      # $(($MIN_SLEEP / 60)) minutes/" "$DAEMON_SH"
sed -i "s/^DEFAULT_SLEEP=.*/DEFAULT_SLEEP=$DEFAULT_SLEEP  # $(($DEFAULT_SLEEP / 60)) minutes/" "$DAEMON_SH"
sed -i "s/^MAX_SLEEP=.*/MAX_SLEEP=$MAX_SLEEP     # $(($MAX_SLEEP / 60)) minutes/" "$DAEMON_SH"
sed -i "s/^NIGHT_SLEEP=.*/NIGHT_SLEEP=$NIGHT_SLEEP  # $(($NIGHT_SLEEP / 3600)) hours/" "$DAEMON_SH"

# Update action weights
sed -i "s/^TASK_WEIGHT=.*/TASK_WEIGHT=$TASK_WEIGHT         # $(echo "$TASK_WEIGHT * 100" | bc | cut -d. -f1)% - Primary work/" "$DAEMON_SH"
sed -i "s/^REFLECTION_WEIGHT=.*/REFLECTION_WEIGHT=$REFLECTION_WEIGHT   # $(echo "$REFLECTION_WEIGHT * 100" | bc | cut -d. -f1)% - Self-improvement/" "$DAEMON_SH"
sed -i "s/^CONVERSATION_WEIGHT=.*/CONVERSATION_WEIGHT=$CONVERSATION_WEIGHT # $(echo "$CONVERSATION_WEIGHT * 100" | bc | cut -d. -f1)% - Human communication/" "$DAEMON_SH"

# Update reflection cooldown
sed -i "s/local cooldown_minutes=.*/local cooldown_minutes=$REFLECTION_COOLDOWN/" "$DAEMON_SH"

# Update active hours check (more complex - need to modify function)
ACTIVE_START=$(jq -r '.active_start' "$CONFIG_FILE")
ACTIVE_END=$(jq -r '.active_end' "$CONFIG_FILE")

# Update is_active_hours function
sed -i "s/if \[ \"\$edt_hour\" -ge [0-9]* \] && \[ \"\$edt_hour\" -lt [0-9]* \]; then/if [ \"\$edt_hour\" -ge $ACTIVE_START ] \&\& [ \"\$edt_hour\" -lt $ACTIVE_END ]; then/" "$DAEMON_SH"

# Update chaos config
CHAOS_ENABLED_BOOL=$([ "$CHAOS_ENABLED" = "true" ] && echo "true" || echo "false")
jq --argjson enabled "$CHAOS_ENABLED_BOOL" --argjson prob "$CHAOS_PROB" \
    '.enabled = $enabled | .chaos_probability = $prob' \
    "$CHAOS_FILE" > "${CHAOS_FILE}.tmp"
mv "${CHAOS_FILE}.tmp" "$CHAOS_FILE"

# Update thinking levels in daemon-settings.json
THINKING_LEVELS=$(jq -r '.thinking_levels' "$CONFIG_FILE")
jq --argjson levels "$THINKING_LEVELS" \
    '.thinking.levels = $levels' \
    "${DAEMON_ROOT}/daemon-settings.json" > "${DAEMON_ROOT}/daemon-settings.json.tmp"
mv "${DAEMON_ROOT}/daemon-settings.json.tmp" "${DAEMON_ROOT}/daemon-settings.json"

echo "✓ Configuration applied successfully!"
echo ""
echo "Changes:"
echo "  - Sleep intervals: ${MIN_SLEEP}s / ${DEFAULT_SLEEP}s / ${MAX_SLEEP}s"
echo "  - Active hours: ${ACTIVE_START}:00 - ${ACTIVE_END}:00 EDT"
echo "  - Action weights: Task=${TASK_WEIGHT} Reflection=${REFLECTION_WEIGHT} Conversation=${CONVERSATION_WEIGHT}"
echo "  - Reflection cooldown: ${REFLECTION_COOLDOWN} minutes"
echo "  - Chaos: ${CHAOS_ENABLED} (probability: ${CHAOS_PROB})"
echo ""
echo "Restart daemon to apply: ~/.claude/daemon/daemon.sh restart"
echo "Or use: systemctl --user restart claude-daemon"
