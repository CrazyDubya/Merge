#!/bin/bash
#
# Daemon Backup Script
# Backs up critical daemon files to Cloudflare R2 with GPG encryption
#
# Usage: backup-daemon.sh [daily|weekly|monthly|yearly]
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
BACKUP_TYPE="${1:-manual}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
DATE_STAMP=$(date +%Y-%m-%d)
BACKUP_NAME="daemon-backup-${BACKUP_TYPE}-${TIMESTAMP}"
TEMP_DIR="/tmp/${BACKUP_NAME}"
GPG_KEY="319A72D7899CC40E"
RCLONE_REMOTE="r2-daemon:claude-orc"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦  DAEMON BACKUP SYSTEM"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Type:      $BACKUP_TYPE"
echo "Date:      $DATE_STAMP"
echo "Backup:    $BACKUP_NAME"
echo ""

# Create temporary backup directory
echo "🗂️  Step 1: Creating backup structure..."
mkdir -p "$TEMP_DIR"

# Copy critical files
echo "📋 Step 2: Copying critical files..."

# Core identity
mkdir -p "$TEMP_DIR"
cp "$DAEMON_ROOT/daemon.sh" "$TEMP_DIR/"
cp "$DAEMON_ROOT/daemon-settings.json" "$TEMP_DIR/"

# Personalities
mkdir -p "$TEMP_DIR/personalities"
cp -r "$DAEMON_ROOT/personalities"/* "$TEMP_DIR/personalities/"

# Memory
mkdir -p "$TEMP_DIR/memory"
cp "$DAEMON_ROOT/memory/conversation-id.txt" "$TEMP_DIR/memory/" 2>/dev/null || true
cp "$DAEMON_ROOT/memory/persona-timeline.jsonl" "$TEMP_DIR/memory/"
cp "$DAEMON_ROOT/memory/emergence-log.md" "$TEMP_DIR/memory/"
cp -r "$DAEMON_ROOT/memory/archived" "$TEMP_DIR/memory/" 2>/dev/null || true
cp "$DAEMON_ROOT/memory/inter-persona-dialogue.md" "$TEMP_DIR/memory/" 2>/dev/null || true

# Tasks
mkdir -p "$TEMP_DIR/tasks"
cp "$DAEMON_ROOT/tasks/queue.md" "$TEMP_DIR/tasks/" 2>/dev/null || true
cp -r "$DAEMON_ROOT/tasks/completed" "$TEMP_DIR/tasks/" 2>/dev/null || true

# Metrics
mkdir -p "$TEMP_DIR/metrics"
cp "$DAEMON_ROOT/metrics/switch-history.jsonl" "$TEMP_DIR/metrics/" 2>/dev/null || true
cp "$DAEMON_ROOT/metrics/personas.json" "$TEMP_DIR/metrics/" 2>/dev/null || true

# Triggers
mkdir -p "$TEMP_DIR/triggers"
cp -r "$DAEMON_ROOT/triggers"/*.json "$TEMP_DIR/triggers/" 2>/dev/null || true

# Scripts and libraries
mkdir -p "$TEMP_DIR/scripts" "$TEMP_DIR/lib"
cp -r "$DAEMON_ROOT/scripts"/* "$TEMP_DIR/scripts/" 2>/dev/null || true
cp -r "$DAEMON_ROOT/lib"/* "$TEMP_DIR/lib/" 2>/dev/null || true
cp "$DAEMON_ROOT"/*.sh "$TEMP_DIR/" 2>/dev/null || true

# Inbox (recent messages only - last 50 from each)
mkdir -p "$TEMP_DIR/inbox/daemon/unread" "$TEMP_DIR/inbox/human/unread"
ls -t "$DAEMON_ROOT/inbox/daemon/unread"/*.md 2>/dev/null | head -50 | xargs -I{} cp {} "$TEMP_DIR/inbox/daemon/unread/" || true
ls -t "$DAEMON_ROOT/inbox/human/unread"/*.md 2>/dev/null | head -50 | xargs -I{} cp {} "$TEMP_DIR/inbox/human/unread/" || true

echo "✅ Files copied"

# Create backup metadata
echo "📝 Step 3: Creating backup metadata..."
cat > "$TEMP_DIR/BACKUP_INFO.txt" <<EOF
Daemon Backup
=============

Backup Type: $BACKUP_TYPE
Date: $DATE_STAMP
Timestamp: $TIMESTAMP
Hostname: $(hostname)
Daemon Version: $(head -1 "$DAEMON_ROOT/daemon.sh" | grep -oP 'v\d+\.\d+' || echo "unknown")

Contents:
- daemon.sh and core scripts
- daemon-settings.json
- All personality definitions and state
- Complete memory timeline ($(wc -l < "$DAEMON_ROOT/memory/persona-timeline.jsonl" 2>/dev/null || echo "0") events)
- Emergence logs and archives
- Task queue and completed tasks
- Metrics and switch history
- Triggers configuration
- Recent inbox messages (last 50 per folder)

GPG Key: $GPG_KEY
Compression: tar + gzip
Encryption: GPG AES256

Restore command:
  ~/.claude/daemon/scripts/restore-daemon.sh $BACKUP_NAME

Created: $(date)
EOF

echo "✅ Metadata created"

# Create tarball
echo "📦 Step 4: Creating compressed archive..."
TARBALL="/tmp/${BACKUP_NAME}.tar.gz"
tar -czf "$TARBALL" -C /tmp "$BACKUP_NAME"
rm -rf "$TEMP_DIR"

TARBALL_SIZE=$(du -h "$TARBALL" | cut -f1)
echo "✅ Archive created: $TARBALL_SIZE"

# Encrypt with GPG
echo "🔐 Step 5: Encrypting backup..."
gpg --encrypt --recipient "$GPG_KEY" --trust-model always --output "${TARBALL}.gpg" "$TARBALL"
rm -f "$TARBALL"

ENCRYPTED_SIZE=$(du -h "${TARBALL}.gpg" | cut -f1)
echo "✅ Backup encrypted: $ENCRYPTED_SIZE"

# Upload to R2
echo "☁️  Step 6: Uploading to Cloudflare R2..."
rclone copy "${TARBALL}.gpg" "${RCLONE_REMOTE}/${BACKUP_TYPE}/" --progress

echo "✅ Upload complete"

# Verify upload
echo "🔍 Step 7: Verifying upload..."
if rclone ls "${RCLONE_REMOTE}/${BACKUP_TYPE}/${BACKUP_NAME}.tar.gz.gpg" >/dev/null 2>&1; then
    echo "✅ Backup verified in R2"

    # Clean up local encrypted file
    rm -f "${TARBALL}.gpg"
    echo "🧹 Local encrypted file removed"
else
    echo "❌ VERIFICATION FAILED - keeping local backup"
    echo "   Local backup: ${TARBALL}.gpg"
    exit 1
fi

# Apply retention policy
echo "🗑️  Step 8: Applying retention policy..."

case $BACKUP_TYPE in
    daily)
        RETENTION_DAYS=30
        ;;
    weekly)
        RETENTION_DAYS=56  # 8 weeks
        ;;
    monthly)
        RETENTION_DAYS=365  # 12 months
        ;;
    yearly)
        RETENTION_DAYS=36500  # Keep forever (100 years)
        ;;
    *)
        RETENTION_DAYS=90  # Manual backups: 90 days
        ;;
esac

if [ "$RETENTION_DAYS" -lt 36500 ]; then
    CUTOFF_DATE=$(date -d "$RETENTION_DAYS days ago" +%Y%m%d)

    # List old backups
    OLD_BACKUPS=$(rclone ls "${RCLONE_REMOTE}/${BACKUP_TYPE}/" | awk '{print $2}' | grep "daemon-backup-${BACKUP_TYPE}-" | while read backup; do
        BACKUP_DATE=$(echo "$backup" | grep -oP '\d{8}' | head -1)
        if [ "$BACKUP_DATE" -lt "$CUTOFF_DATE" ]; then
            echo "$backup"
        fi
    done)

    if [ -n "$OLD_BACKUPS" ]; then
        echo "Found old backups to remove:"
        echo "$OLD_BACKUPS"
        echo "$OLD_BACKUPS" | while read backup; do
            echo "  Removing: $backup"
            rclone delete "${RCLONE_REMOTE}/${BACKUP_TYPE}/$backup"
        done
        echo "✅ Old backups cleaned up"
    else
        echo "✅ No old backups to remove"
    fi
else
    echo "✅ Yearly backups - no retention cleanup"
fi

# Log to daemon
echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] Backup completed: ${BACKUP_TYPE} (${ENCRYPTED_SIZE})" >> "${DAEMON_ROOT}/logs/activity.log"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅  BACKUP COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Type:       $BACKUP_TYPE"
echo "Size:       $ENCRYPTED_SIZE"
echo "Location:   ${RCLONE_REMOTE}/${BACKUP_TYPE}/${BACKUP_NAME}.tar.gz.gpg"
echo "Retention:  $RETENTION_DAYS days"
echo ""
echo "Restore command:"
echo "  ~/.claude/daemon/scripts/restore-daemon.sh ${BACKUP_TYPE}/${BACKUP_NAME}"
echo ""

exit 0
