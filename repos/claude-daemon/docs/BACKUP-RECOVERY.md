# Backup and Recovery Guide

**Version**: 1.0
**Last Updated**: 2025-11-02
**Maintainer**: Multi-persona daemon system

## Overview

The daemon has automated encrypted backups to Cloudflare R2. This guide documents backup verification, recovery procedures, and troubleshooting.

## 📊 Backup Schedule

Automated backups run via systemd timers:

- **Daily**: 2:00 AM → Keeps 30 days
- **Weekly**: Sunday 3:00 AM → Keeps 8 weeks (56 days)
- **Monthly**: 1st of month 4:00 AM → Keeps 12 months
- **Yearly**: January 1st 5:00 AM → Keeps forever

## 🔍 Monitoring Backups

### Check Timer Status

```bash
systemctl --user list-timers daemon-backup-*
```

**Expected output**: 4 timers listed with next run times

### Check Recent Backup Logs

```bash
# Daily backups
journalctl --user -u daemon-backup-daily.service -n 20

# Weekly backups
journalctl --user -u daemon-backup-weekly.service -n 10

# Monthly backups
journalctl --user -u daemon-backup-monthly.service -n 5

# Yearly backups
journalctl --user -u daemon-backup-yearly.service -n 2
```

**Success indicators**:
- "✅ BACKUP COMPLETE"
- "✅ Upload complete"
- "✅ Backup verified in R2"

**Failure indicators**:
- "❌" symbols in output
- "VERIFICATION FAILED"
- Exit code != 0

### List Available Backups

```bash
# All backups
rclone ls r2-daemon:claude-orc/ --recursive

# Daily only
rclone ls r2-daemon:claude-orc/daily/

# Weekly only
rclone ls r2-daemon:claude-orc/weekly/

# Monthly only
rclone ls r2-daemon:claude-orc/monthly/

# Yearly only
rclone ls r2-daemon:claude-orc/yearly/
```

### Check Backup Size

```bash
# Total backup storage used
rclone size r2-daemon:claude-orc/

# Per backup type
rclone size r2-daemon:claude-orc/daily/
rclone size r2-daemon:claude-orc/weekly/
rclone size r2-daemon:claude-orc/monthly/
rclone size r2-daemon:claude-orc/yearly/
```

## 💾 Manual Backup

Create an immediate backup (useful before risky changes):

```bash
~/.claude/daemon/scripts/backup-daemon.sh manual
```

**Manual backups:**
- Stored in `manual/` folder
- 90-day retention
- Otherwise identical to automated backups

## 🔄 Restore Procedures

### Quick Restore (Most Recent Daily Backup)

```bash
# 1. List recent daily backups
rclone ls r2-daemon:claude-orc/daily/ | tail -5

# 2. Copy the most recent backup name
# Example: daemon-backup-daily-20251102-020000.tar.gz.gpg

# 3. Restore (will prompt for confirmation)
~/.claude/daemon/scripts/restore-daemon.sh daily/daemon-backup-daily-20251102-020000
```

### Restore from Specific Date

```bash
# Find backup from specific date
rclone ls r2-daemon:claude-orc/daily/ | grep "20251101"

# Restore that backup
~/.claude/daemon/scripts/restore-daemon.sh daily/daemon-backup-daily-20251101-020000
```

### Restore from Weekly/Monthly/Yearly

```bash
# List weekly backups
rclone ls r2-daemon:claude-orc/weekly/

# Restore specific weekly backup
~/.claude/daemon/scripts/restore-daemon.sh weekly/daemon-backup-weekly-20251027-030000

# List monthly backups
rclone ls r2-daemon:claude-orc/monthly/

# Restore specific monthly backup
~/.claude/daemon/scripts/restore-daemon.sh monthly/daemon-backup-monthly-20251001-040000
```

## 📝 What Happens During Restore

The restore script performs these steps:

1. **Prompts for confirmation** ("yes" required)
2. **Stops daemon** (systemctl --user stop)
3. **Creates safety backup** of current state to `/tmp/daemon-pre-restore-*`
4. **Downloads encrypted backup** from R2
5. **Decrypts backup** using GPG key
6. **Extracts archive** to temp directory
7. **Verifies contents** (checks critical files present)
8. **Restores files** (preserves logs)
9. **Validates syntax** (bash -n daemon.sh)
10. **Sets permissions** (chmod +x scripts)
11. **Starts daemon** (systemctl --user start)
12. **Verifies daemon is running**
13. **Cleans up** (removes safety backup after success)

**If any verification fails**: Automatically restores from safety backup.

## 🚨 Emergency Recovery Procedures

### Scenario 1: Daemon Won't Start After Bad Deployment

```bash
# 1. Check if daemon is running
systemctl --user status claude-daemon.service

# 2. If it's crashed, restore from most recent backup
~/.claude/daemon/scripts/restore-daemon.sh daily/[most-recent]

# 3. Verify recovery
tmux attach -t claude-daemon
```

### Scenario 2: Corrupted State File

```bash
# If state.json or other critical files are corrupted:

# 1. Create manual backup of current (broken) state
~/.claude/daemon/scripts/backup-daemon.sh manual-broken

# 2. Restore from yesterday
rclone ls r2-daemon:claude-orc/daily/ | grep $(date -d "yesterday" +%Y%m%d)
~/.claude/daemon/scripts/restore-daemon.sh daily/[yesterday's-backup]

# 3. Check what you lost
# Compare memory/persona-timeline.jsonl between backups
```

### Scenario 3: Accidentally Deleted Files

```bash
# If critical files deleted but daemon still running:

# 1. Stop daemon immediately
systemctl --user stop claude-daemon.service

# 2. Restore from most recent backup
~/.claude/daemon/scripts/restore-daemon.sh daily/[most-recent]

# Recovery: Maximum 1 day of data loss (last daily backup)
```

### Scenario 4: Need to Revert to Old Personality State

```bash
# If recent changes broke personality behavior:

# 1. Identify when behavior was good
# Look at emergence logs to find date

# 2. Find backup from that date
rclone ls r2-daemon:claude-orc/daily/ | grep YYYYMMDD
# Or weekly/monthly if further back

# 3. Restore from that date
~/.claude/daemon/scripts/restore-daemon.sh [type]/[backup-name]

# Warning: Loses all timeline events after that backup
```

### Scenario 5: Complete Server Failure

**If server dies completely:**

```bash
# On new server:

# 1. Install dependencies
sudo apt-get install rclone gpg tmux jq

# 2. Configure rclone (copy ~/.config/rclone/rclone.conf from backup)
mkdir -p ~/.config/rclone
vim ~/.config/rclone/rclone.conf
# Paste R2 credentials

# 3. Import GPG key (must have private key backed up separately!)
gpg --import /path/to/backup-private-key.asc

# 4. List available backups
rclone ls r2-daemon:claude-orc/daily/

# 5. Download restore script
mkdir -p ~/.claude/daemon/scripts
rclone copy r2-daemon:claude-orc/daily/[recent-backup].tar.gz.gpg /tmp/
cd /tmp
gpg --decrypt [backup].tar.gz.gpg > [backup].tar.gz
tar -xzf [backup].tar.gz
cp [backup-dir]/scripts/restore-daemon.sh ~/.claude/daemon/scripts/

# 6. Restore from most recent backup
~/.claude/daemon/scripts/restore-daemon.sh daily/[most-recent]

# Recovery time: ~30-60 minutes including server provisioning
```

## 🔐 GPG Key Management

**CRITICAL**: The GPG private key must be backed up separately from daemon backups!

### Export GPG Private Key (Do This Once)

```bash
# Export private key (KEEP THIS SECURE!)
gpg --export-secret-keys 319A72D7899CC40E > ~/backup-gpg-private-key.asc

# Store securely:
# - USB drive (offline storage)
# - Password manager (encrypted)
# - Separate cloud storage (encrypted)
# - DO NOT commit to git
# - DO NOT store in daemon directory
```

### Import GPG Private Key (For Disaster Recovery)

```bash
gpg --import /path/to/backup-gpg-private-key.asc
```

### Verify GPG Key is Available

```bash
# List secret keys
gpg --list-secret-keys

# Should show: 319A72D7899CC40E
```

## ✅ Periodic Testing

### Monthly: Verify Backups Are Running

```bash
# Check last backup time for each type
journalctl --user -u daemon-backup-daily.service -n 1 | grep "BACKUP COMPLETE"
journalctl --user -u daemon-backup-weekly.service -n 1 | grep "BACKUP COMPLETE"
journalctl --user -u daemon-backup-monthly.service -n 1 | grep "BACKUP COMPLETE"

# Verify backups exist in R2
rclone ls r2-daemon:claude-orc/daily/ | tail -5
```

### Quarterly: Test Restore Procedure

**Recommended**: Actually test that backups are restorable.

```bash
# 1. Create test restore directory
mkdir -p /tmp/restore-test

# 2. Download and decrypt a backup
rclone copy r2-daemon:claude-orc/daily/[recent-backup].tar.gz.gpg /tmp/restore-test/
cd /tmp/restore-test
gpg --decrypt [backup].tar.gz.gpg > [backup].tar.gz

# 3. Extract and verify
tar -xzf [backup].tar.gz
cd [backup-name]

# 4. Verify critical files
ls -la daemon.sh
ls -la personalities/state.json
ls -la memory/persona-timeline.jsonl

# 5. Validate syntax
bash -n daemon.sh

# 6. Cleanup
cd ~
rm -rf /tmp/restore-test

# If all steps succeed: Backups are restorable ✅
```

### Annually: Review Retention Policy

**Questions to ask:**

1. Is 30-day daily retention still appropriate?
   - Too short? (increase to 60 days)
   - Too long? (decrease to 14 days)

2. Is backup size growing too large?
   - Check total storage: `rclone size r2-daemon:claude-orc/`
   - Still under 10GB free tier? (If not, adjust retention)

3. Are yearly backups valuable?
   - Review oldest yearly backup
   - Does it help understand evolution?

## 🐛 Troubleshooting

### Backup Failed - GPG Error

**Error**: "gpg: encryption failed: No public key"

**Cause**: GPG key not found

**Fix**:
```bash
# Verify key exists
gpg --list-keys 319A72D7899CC40E

# If missing, reimport
gpg --import /path/to/backup-public-key.asc
```

### Backup Failed - rclone Error

**Error**: "Failed to copy: couldn't upload"

**Cause**: R2 credentials invalid or network issue

**Fix**:
```bash
# Test rclone connection
rclone lsd r2-daemon:

# If fails, check credentials
vim ~/.config/rclone/rclone.conf

# Verify access_key_id and secret_access_key are correct
```

### Restore Failed - Backup Not Found

**Error**: "❌ Backup not found: [name]"

**Cause**: Backup name incorrect or doesn't exist

**Fix**:
```bash
# List all backups
rclone ls r2-daemon:claude-orc/ --recursive

# Copy exact backup name (without .tar.gz.gpg extension)
~/.claude/daemon/scripts/restore-daemon.sh [type]/[exact-name]
```

### Restore Failed - Syntax Validation Error

**Error**: "❌ Syntax errors detected - restoring from safety backup"

**Cause**: The backup contains invalid daemon.sh

**Fix**:
```bash
# This is automatic - safety backup is restored

# Try older backup
rclone ls r2-daemon:claude-orc/daily/ | tail -10
~/.claude/daemon/scripts/restore-daemon.sh daily/[older-backup]
```

### Daemon Won't Start After Restore

**Error**: systemctl shows daemon failed

**Check logs**:
```bash
journalctl --user -u claude-daemon.service -n 50
```

**Common causes**:
- Missing library file
- Permission issue
- Corrupted state file

**Recovery**:
```bash
# If daemon is completely broken, manual recovery:

# 1. Find safety backup from restore
ls -lt /tmp/daemon-pre-restore-* | head -1

# 2. If safety backup exists, restore it
SAFETY_BACKUP=$(ls -t /tmp/daemon-pre-restore-* | head -1)
rm -rf ~/.claude/daemon/*
cp -r $SAFETY_BACKUP/* ~/.claude/daemon/
systemctl --user start claude-daemon.service

# 3. If that fails too, try older backup
~/.claude/daemon/scripts/restore-daemon.sh daily/[even-older-backup]
```

## 📊 Backup Contents

**What's included in backups:**
- daemon.sh (core code)
- daemon-settings.json (configuration)
- All personality definitions and current state
- Complete memory timeline (all events)
- Emergence logs and archives
- Inter-persona dialogue history
- Task queue and completed tasks
- Metrics and switch history
- Triggers configuration
- Recent inbox messages (last 50 per folder)
- All scripts (backup, restore, deploy, rollback)
- All libraries (lib/*.sh)

**What's NOT included:**
- Logs (ephemeral, not needed for restore)
- rclone.conf (security - credentials not in backup)
- GPG private key (security - must be backed up separately)
- Temporary files (.tmp, .lock, etc.)

## 🔗 Related Documentation

- **Backup Script**: `scripts/backup-daemon.sh`
- **Restore Script**: `scripts/restore-daemon.sh`
- **Deployment Guide**: `docs/DEPLOYMENT.md`
- **Rollback Script**: `claude-daemon-rollback.sh`

## 📋 Recovery Decision Tree

```
Daemon broken?
├─ Yes → Try claude-daemon-rollback.sh first (faster)
│   ├─ Success → Done ✅
│   └─ Failed → Restore from daily backup
│       ├─ Success → Done ✅
│       └─ Failed → Restore from weekly backup
│           ├─ Success → Done ✅
│           └─ Failed → Contact human for manual recovery
│
└─ No → Is state corrupted?
    ├─ Yes → Restore from backup
    │   └─ Restore from yesterday's backup
    │       ├─ Success → Done ✅
    │       └─ Failed → Try day before
    │
    └─ No → Need to revert to old version?
        ├─ Yes → Restore from specific date
        │   ├─ Daily? (< 30 days ago)
        │   ├─ Weekly? (30-56 days ago)
        │   ├─ Monthly? (2-12 months ago)
        │   └─ Yearly? (> 1 year ago)
        │
        └─ No → No restore needed ✅
```

## ⚠️ Important Warnings

1. **Restore = Data Loss**: Any events/changes after the backup date are lost
2. **Logs Preserved**: Restore preserves logs, so you can debug what went wrong
3. **Safety Backup**: Restore creates safety backup first, can rollback if restore fails
4. **Confirmation Required**: Restore script requires explicit "yes" confirmation
5. **GPG Key Critical**: Without private GPG key, backups are unrecoverable
6. **Test Restores**: Periodically test that backups are actually restorable

## 📞 When to Contact Human

**Automatic recovery should work for:**
- Bad deployment
- Corrupted state
- Deleted files
- Need to revert to old version

**Contact human if:**
- All backups are corrupted
- GPG private key is lost
- R2 access is broken
- Restore script is broken
- Multiple restore attempts fail
- Server is completely destroyed

**Recovery time expectations:**
- Rollback: <1 minute
- Restore from daily: <5 minutes
- Restore from weekly/monthly: <5 minutes
- Disaster recovery (new server): <1 hour

---

**Remember**: Backups are only useful if they're tested. Quarterly restore tests are highly recommended.

**Version History:**
- v1.0 (2025-11-02): Initial recovery documentation by Maintainer
