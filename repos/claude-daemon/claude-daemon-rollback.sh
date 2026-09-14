#!/bin/bash
#
# Autonomous Daemon Rollback Script
# Automatically finds last backup, validates, restores, and restarts
#
# Usage: claude-daemon-rollback.sh "Reason for rollback"
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
INBOX_DIR="${DAEMON_ROOT}/inbox/human/unread"
DAEMON_SCRIPT="${DAEMON_ROOT}/daemon.sh"

REASON="${1:-Emergency rollback}"

# Get current persona and timestamp
CURRENT_PERSONA=$(jq -r '.current_persona' "${DAEMON_ROOT}/personalities/state.json" 2>/dev/null || echo "unknown")
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TIMESTAMP_FILE=$(date +%Y%m%d-%H%M%S)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄  DAEMON ROLLBACK SYSTEM"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Persona: $CURRENT_PERSONA"
echo "Reason:  $REASON"
echo ""

# Step 1: Find most recent backup
echo "🔍 Step 1: Finding most recent backup..."
LATEST_BACKUP=$(ls -t "${DAEMON_SCRIPT}.backup-"* 2>/dev/null | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "❌ ERROR: No backup files found!"
    echo ""
    echo "Backup pattern: ${DAEMON_SCRIPT}.backup-*"
    echo ""
    echo "⚠️  CANNOT ROLLBACK - No backups available"
    echo ""
    echo "Alternative recovery options:"
    echo "  1. Restore from git: git checkout HEAD~1 daemon.sh"
    echo "  2. Manual fix: vim daemon.sh"
    echo "  3. Contact human for assistance"
    exit 1
fi

BACKUP_FILENAME=$(basename "$LATEST_BACKUP")
echo "✅ Found backup: $BACKUP_FILENAME"

# Step 2: Validate backup
echo ""
echo "🔍 Step 2: Validating backup..."
if bash -n "$LATEST_BACKUP" 2>&1 | tee /tmp/backup-validation.log; then
    echo "✅ Backup validation passed"
else
    echo "❌ BACKUP IS CORRUPTED!"
    echo ""
    echo "Errors:"
    cat /tmp/backup-validation.log
    echo ""
    echo "⚠️  CANNOT ROLLBACK - Backup has syntax errors"
    echo ""
    echo "Trying next older backup..."
    NEXT_BACKUP=$(ls -t "${DAEMON_SCRIPT}.backup-"* 2>/dev/null | head -2 | tail -1)

    if [ -n "$NEXT_BACKUP" ] && [ "$NEXT_BACKUP" != "$LATEST_BACKUP" ]; then
        echo "Found: $(basename "$NEXT_BACKUP")"
        echo "Validating..."
        if bash -n "$NEXT_BACKUP"; then
            LATEST_BACKUP="$NEXT_BACKUP"
            BACKUP_FILENAME=$(basename "$LATEST_BACKUP")
            echo "✅ Older backup is valid, using: $BACKUP_FILENAME"
        else
            echo "❌ Next backup also corrupted"
            echo "⚠️  Manual intervention required"
            exit 1
        fi
    else
        echo "❌ No other backups available"
        echo "⚠️  Manual intervention required"
        exit 1
    fi
fi

# Step 3: Create safety backup of current (broken) state
echo ""
echo "💾 Step 3: Creating safety backup of current state..."
SAFETY_BACKUP="${DAEMON_SCRIPT}.broken-${TIMESTAMP_FILE}"
cp "$DAEMON_SCRIPT" "$SAFETY_BACKUP"
echo "✅ Current state saved: $(basename "$SAFETY_BACKUP")"
echo "   (in case rollback makes things worse)"

# Step 4: Restore backup
echo ""
echo "🔄 Step 4: Restoring backup..."
cp "$LATEST_BACKUP" "$DAEMON_SCRIPT"
echo "✅ Restored: $BACKUP_FILENAME → daemon.sh"

# Step 5: Validate restored file
echo ""
echo "🔍 Step 5: Validating restored daemon.sh..."
if bash -n "$DAEMON_SCRIPT"; then
    echo "✅ Restored file validation passed"
else
    echo "❌ CRITICAL ERROR: Restored file has syntax errors!"
    echo ""
    echo "This should never happen (backup was validated in step 2)"
    echo "Possible causes:"
    echo "  - File corruption during copy"
    echo "  - Disk issues"
    echo "  - Cosmic rays (seriously)"
    echo ""
    echo "⚠️  MANUAL INTERVENTION REQUIRED"
    exit 1
fi

# Step 6: Send pre-restart notification
echo ""
echo "📧 Step 6: Sending rollback notification..."
MESSAGE_ID="rollback-${TIMESTAMP_FILE}"
ROLLBACK_EMAIL="${INBOX_DIR}/${MESSAGE_ID}.md"

cat > "$ROLLBACK_EMAIL" <<EOF
---
from: daemon-system
to: human
timestamp: $TIMESTAMP
priority: high
tags: [rollback, recovery, critical, autonomous]
message_id: $MESSAGE_ID
---

# ⚠️ ROLLBACK IN PROGRESS

**Persona**: $CURRENT_PERSONA
**Time**: $TIMESTAMP
**Reason**: $REASON

## Rollback Details:
- **Restoring from**: \`$BACKUP_FILENAME\`
- **Current state saved**: \`$(basename "$SAFETY_BACKUP")\`
- **Validation**: ✅ Backup validated before restore

## Actions Taken:
1. ✅ Found most recent backup
2. ✅ Validated backup syntax
3. ✅ Saved current (broken) state as safety backup
4. ✅ Restored backup to daemon.sh
5. ✅ Validated restored file
6. 🔄 Restarting daemon now...

## Safety:
- Watchdog monitoring active
- systemd will auto-restart on failure
- Post-rollback email will confirm recovery
- Broken state preserved for debugging

**Expected recovery**: Within 10 seconds
**If no recovery email within 1 minute**: Restart failed - check \`systemctl --user status claude-daemon\`

---

**Initiated by**: $CURRENT_PERSONA persona
**Rollback system**: Autonomous (daemon self-healed)

— Daemon Rollback System 🔄
EOF

echo "✅ Rollback notification sent to inbox"

# Step 7: Restart daemon
echo ""
echo "🔄 Step 7: Restarting daemon..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚡ INITIATING RESTART..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Log the rollback
source "${DAEMON_ROOT}/lib/atomic-io.sh" 2>/dev/null || true
atomic_append "${DAEMON_ROOT}/logs/activity.log" "[$(date +'%Y-%m-%d %H:%M:%S')] [ROLLBACK] Restored $BACKUP_FILENAME - Reason: $REASON" 2>/dev/null || echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ROLLBACK] Restored $BACKUP_FILENAME - Reason: $REASON" >> "${DAEMON_ROOT}/logs/activity.log"

# Restart in background so this script can complete
(sleep 2 && systemctl --user restart claude-daemon.service) &

echo "🎯 Restart initiated!"
echo "📬 Check inbox for recovery email..."
echo ""
echo "✅ ROLLBACK COMPLETE"
echo ""
echo "Files created:"
echo "  - Safety backup: $(basename "$SAFETY_BACKUP")"
echo "  - Rollback email: $MESSAGE_ID.md"
echo ""
echo "Next steps:"
echo "  1. Wait 10 seconds"
echo "  2. Verify post-rollback email appears"
echo "  3. Check daemon is functioning normally"
echo "  4. Investigate why deployment failed"
echo "  5. Fix root cause before redeploying"
echo ""

exit 0
