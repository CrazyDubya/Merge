#!/bin/bash
#
# Daemon Restore Script
# Restores daemon from encrypted R2 backup
#
# Usage: restore-daemon.sh [backup-type/]backup-name
# Example: restore-daemon.sh daily/daemon-backup-daily-20251102-143022
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
BACKUP_PATH="${1:-}"
RCLONE_REMOTE="r2-daemon:claude-orc"
GPG_KEY="319A72D7899CC40E"

if [ -z "$BACKUP_PATH" ]; then
    echo "❌ Error: No backup specified"
    echo ""
    echo "Usage: $0 [backup-type/]backup-name"
    echo ""
    echo "Available backups:"
    echo ""
    for type in daily weekly monthly yearly; do
        echo "  ${type^^}:"
        rclone ls "${RCLONE_REMOTE}/${type}/" 2>/dev/null | awk '{print "    " $2}' || echo "    (none)"
        echo ""
    done
    exit 1
fi

# Extract backup name from path
if [[ "$BACKUP_PATH" == */* ]]; then
    BACKUP_NAME=$(basename "$BACKUP_PATH")
    BACKUP_FULL_PATH="$BACKUP_PATH"
else
    BACKUP_NAME="$BACKUP_PATH"
    # Try to find the backup in all folders
    for type in daily weekly monthly yearly; do
        if rclone ls "${RCLONE_REMOTE}/${type}/${BACKUP_NAME}.tar.gz.gpg" >/dev/null 2>&1; then
            BACKUP_FULL_PATH="${type}/${BACKUP_NAME}"
            break
        fi
    done

    if [ -z "${BACKUP_FULL_PATH:-}" ]; then
        echo "❌ Backup not found: $BACKUP_NAME"
        exit 1
    fi
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📥  DAEMON RESTORE SYSTEM"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Backup:    $BACKUP_FULL_PATH"
echo ""

# Confirm restore
read -p "⚠️  This will REPLACE current daemon files. Continue? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "❌ Restore cancelled"
    exit 0
fi

echo ""
echo "🛑 Step 1: Stopping daemon..."
systemctl --user stop claude-daemon.service 2>/dev/null || true
echo "✅ Daemon stopped"

# Create backup of current state before restore
SAFETY_BACKUP="/tmp/daemon-pre-restore-$(date +%Y%m%d-%H%M%S)"
echo ""
echo "💾 Step 2: Creating safety backup of current state..."
mkdir -p "$SAFETY_BACKUP"
cp -r "$DAEMON_ROOT"/* "$SAFETY_BACKUP/" 2>/dev/null || true
echo "✅ Safety backup: $SAFETY_BACKUP"

# Download backup from R2
echo ""
echo "☁️  Step 3: Downloading from R2..."
ENCRYPTED_FILE="/tmp/${BACKUP_NAME}.tar.gz.gpg"
rclone copy "${RCLONE_REMOTE}/${BACKUP_FULL_PATH}.tar.gz.gpg" /tmp/ --progress

ENCRYPTED_SIZE=$(du -h "$ENCRYPTED_FILE" | cut -f1)
echo "✅ Downloaded: $ENCRYPTED_SIZE"

# Decrypt backup
echo ""
echo "🔓 Step 4: Decrypting backup..."
TARBALL="/tmp/${BACKUP_NAME}.tar.gz"
gpg --decrypt --output "$TARBALL" "$ENCRYPTED_FILE"
rm -f "$ENCRYPTED_FILE"

TARBALL_SIZE=$(du -h "$TARBALL" | cut -f1)
echo "✅ Decrypted: $TARBALL_SIZE"

# Extract backup
echo ""
echo "📦 Step 5: Extracting archive..."
EXTRACT_DIR="/tmp/${BACKUP_NAME}"
tar -xzf "$TARBALL" -C /tmp
rm -f "$TARBALL"
echo "✅ Archive extracted"

# Verify backup contents
echo ""
echo "🔍 Step 6: Verifying backup contents..."
if [ ! -f "$EXTRACT_DIR/daemon.sh" ]; then
    echo "❌ Invalid backup - daemon.sh not found"
    echo "   Restoring from safety backup..."
    rm -rf "$DAEMON_ROOT"/*
    cp -r "$SAFETY_BACKUP"/* "$DAEMON_ROOT/"
    systemctl --user start claude-daemon.service
    exit 1
fi

# Show backup info
if [ -f "$EXTRACT_DIR/BACKUP_INFO.txt" ]; then
    echo ""
    cat "$EXTRACT_DIR/BACKUP_INFO.txt"
    echo ""
fi

# Restore files
echo "📋 Step 7: Restoring daemon files..."

# Clear current daemon directory (except logs)
mkdir -p "${DAEMON_ROOT}/logs.tmp"
mv "$DAEMON_ROOT/logs"/* "${DAEMON_ROOT}/logs.tmp/" 2>/dev/null || true

# Remove all current files except logs
find "$DAEMON_ROOT" -mindepth 1 -maxdepth 1 ! -name 'logs.tmp' -exec rm -rf {} +

# Restore from backup
cp -r "$EXTRACT_DIR"/* "$DAEMON_ROOT/"

# Restore logs
mkdir -p "$DAEMON_ROOT/logs"
mv "${DAEMON_ROOT}/logs.tmp"/* "$DAEMON_ROOT/logs/" 2>/dev/null || true
rmdir "${DAEMON_ROOT}/logs.tmp" 2>/dev/null || true

# Clean up
rm -rf "$EXTRACT_DIR"

echo "✅ Files restored"

# Verify critical files
echo ""
echo "🔍 Step 8: Verifying critical files..."
CRITICAL_FILES=(
    "daemon.sh"
    "daemon-settings.json"
    "personalities/state.json"
    "memory/persona-timeline.jsonl"
)

ALL_GOOD=true
for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$DAEMON_ROOT/$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file - MISSING"
        ALL_GOOD=false
    fi
done

if [ "$ALL_GOOD" = false ]; then
    echo ""
    echo "❌ Critical files missing - restoring from safety backup"
    rm -rf "$DAEMON_ROOT"/*
    cp -r "$SAFETY_BACKUP"/* "$DAEMON_ROOT/"
    systemctl --user start claude-daemon.service
    exit 1
fi

echo "✅ All critical files present"

# Validate daemon.sh syntax
echo ""
echo "🔍 Step 9: Validating daemon.sh syntax..."
if bash -n "$DAEMON_ROOT/daemon.sh"; then
    echo "✅ Syntax valid"
else
    echo "❌ Syntax errors detected - restoring from safety backup"
    rm -rf "$DAEMON_ROOT"/*
    cp -r "$SAFETY_BACKUP"/* "$DAEMON_ROOT/"
    systemctl --user start claude-daemon.service
    exit 1
fi

# Make scripts executable
echo ""
echo "🔧 Step 10: Setting permissions..."
chmod +x "$DAEMON_ROOT/daemon.sh"
chmod +x "$DAEMON_ROOT"/*.sh 2>/dev/null || true
chmod +x "$DAEMON_ROOT/scripts"/*.sh 2>/dev/null || true
chmod +x "$DAEMON_ROOT/lib"/*.sh 2>/dev/null || true
echo "✅ Permissions set"

# Restart daemon
echo ""
echo "🔄 Step 11: Starting daemon..."
systemctl --user start claude-daemon.service

# Wait for startup
sleep 3

# Verify daemon is running
if systemctl --user is-active --quiet claude-daemon.service; then
    echo "✅ Daemon started successfully"

    # Log restore to activity log
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] Daemon restored from backup: $BACKUP_FULL_PATH" >> "${DAEMON_ROOT}/logs/activity.log"
else
    echo "❌ Daemon failed to start"
    echo "   Check logs: journalctl --user -u claude-daemon.service -n 50"
    echo "   Safety backup available: $SAFETY_BACKUP"
    exit 1
fi

# Cleanup safety backup after successful restore
rm -rf "$SAFETY_BACKUP"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅  RESTORE COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Restored:   $BACKUP_FULL_PATH"
echo "Status:     Daemon running"
echo ""
echo "Check daemon status:"
echo "  tmux attach -t claude-daemon"
echo "  systemctl --user status claude-daemon.service"
echo ""

exit 0
