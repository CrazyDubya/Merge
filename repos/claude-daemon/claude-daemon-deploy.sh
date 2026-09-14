#!/bin/bash
#
# Autonomous Daemon Deployment Script
# Called by daemon after making code changes to deploy them to production
#
# Usage: claude-daemon-deploy.sh "Reason for restart" [changed_files]
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
INBOX_DIR="${DAEMON_ROOT}/inbox/human/unread"
DAEMON_SCRIPT="${DAEMON_ROOT}/daemon.sh"
RESTART_FLAG="${DAEMON_ROOT}/.restart-requested"

REASON="${1:-Code changes deployed}"
CHANGED_FILES="${2:-daemon.sh}"

# Get current persona and timestamp
CURRENT_PERSONA=$(jq -r '.current_persona' "${DAEMON_ROOT}/personalities/state.json" 2>/dev/null || echo "unknown")
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TIMESTAMP_FILE=$(date +%Y%m%d-%H%M%S)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀  DAEMON DEPLOYMENT SYSTEM"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Persona: $CURRENT_PERSONA"
echo "Reason:  $REASON"
echo "Files:   $CHANGED_FILES"
echo ""

# Step 1: Validate syntax
echo "🔍 Step 1: Validating syntax..."
if bash -n "$DAEMON_SCRIPT" 2>&1 | tee /tmp/daemon-syntax-check.log; then
    echo "✅ Syntax validation passed"
else
    echo "❌ SYNTAX ERROR DETECTED!"
    echo ""
    echo "Errors:"
    cat /tmp/daemon-syntax-check.log
    echo ""
    echo "⚠️  ABORTING DEPLOYMENT - Fix syntax errors first"
    exit 1
fi

# Step 2: Create backup
echo ""
echo "💾 Step 2: Creating backup..."
BACKUP_FILE="${DAEMON_SCRIPT}.backup-${TIMESTAMP_FILE}"
cp "$DAEMON_SCRIPT" "$BACKUP_FILE"
echo "✅ Backup created: $BACKUP_FILE"

# Step 3: Create restart flag file
echo ""
echo "🏴 Step 3: Creating restart flag..."
cat > "$RESTART_FLAG" <<EOF
{
  "restart_requested": true,
  "timestamp": "$TIMESTAMP",
  "persona": "$CURRENT_PERSONA",
  "reason": "$REASON",
  "changed_files": "$CHANGED_FILES",
  "backup_file": "$BACKUP_FILE"
}
EOF
echo "✅ Restart flag created"

# Step 4: Send pre-restart inbox message
echo ""
echo "📧 Step 4: Sending pre-restart notification..."
MESSAGE_ID="restart-${TIMESTAMP_FILE}"
PRE_RESTART_EMAIL="${INBOX_DIR}/${MESSAGE_ID}.md"

cat > "$PRE_RESTART_EMAIL" <<EOF
---
from: daemon-system
to: human
timestamp: $TIMESTAMP
priority: high
tags: [restart, deployment, critical, autonomous]
message_id: $MESSAGE_ID
---

# ⚠️ RESTARTING DAEMON NOW

**Persona**: $CURRENT_PERSONA
**Time**: $TIMESTAMP
**Reason**: $REASON

## Changes Made:
$CHANGED_FILES

## Validation:
✅ Syntax check passed (\`bash -n daemon.sh\`)
✅ Backup created: \`$(basename $BACKUP_FILE)\`

## Safety:
- Watchdog monitoring active
- systemd will auto-restart on failure
- Post-restart email will confirm recovery

**Expected recovery**: Within 10 seconds
**If no recovery email within 1 minute**: Restart failed - check \`systemctl --user status claude-daemon\`

---

**Initiated by**: $CURRENT_PERSONA persona
**Deployment system**: Autonomous (daemon self-deployed)

— Daemon Deployment System 🚀
EOF

echo "✅ Pre-restart email sent to inbox"

# Step 5: Restart daemon via systemd
echo ""
echo "🔄 Step 5: Restarting daemon..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚡ INITIATING RESTART..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Restart in background so this script can complete
# The daemon will send post-restart email when it comes back up
(sleep 2 && systemctl --user restart claude-daemon.service) &

echo "🎯 Restart initiated!"
echo "📬 Check inbox for recovery email..."
echo ""

exit 0
