# Daemon Backup System

## Quick Reference

### Manual Backup
```bash
~/.claude/daemon/scripts/backup-daemon.sh [daily|weekly|monthly|yearly|manual]
```

### Restore from Backup
```bash
~/.claude/daemon/scripts/restore-daemon.sh [type]/[backup-name]
```

### List Backups
```bash
rclone ls r2-daemon:claude-orc/daily/
rclone ls r2-daemon:claude-orc/weekly/
rclone ls r2-daemon:claude-orc/monthly/
rclone ls r2-daemon:claude-orc/yearly/
```

### Check Timer Status
```bash
systemctl --user list-timers daemon-backup-*
```

## Backup Schedule

- **Daily**: 2:00 AM (30-day retention)
- **Weekly**: Sunday 3:00 AM (8-week retention)
- **Monthly**: 1st of month 4:00 AM (12-month retention)
- **Yearly**: January 1st 5:00 AM (permanent retention)

## What's Backed Up

- Core daemon files (daemon.sh, daemon-settings.json)
- All personalities and state
- Complete memory timeline
- Emergence logs and archives
- Inter-persona dialogue
- Tasks (queue and completed)
- Metrics and switch history
- Triggers configuration
- Scripts and libraries
- Recent inbox messages (last 50 per folder)

## Security

- **Encryption**: GPG AES256 (Key ID: 319A72D7899CC40E)
- **Compression**: tar + gzip
- **Storage**: Cloudflare R2 (encrypted at rest)
- **Size**: ~136 KB per backup (encrypted)

## Emergency Recovery

If daemon is broken and won't start:

```bash
# 1. Find most recent backup
rclone ls r2-daemon:claude-orc/daily/ | tail -1

# 2. Restore it
~/.claude/daemon/scripts/restore-daemon.sh daily/[backup-name]

# 3. Verify daemon is running
systemctl --user status claude-daemon.service
tmux attach -t claude-daemon
```

## Files

**Scripts:**
- `/home/opc/.claude/daemon/scripts/backup-daemon.sh`
- `/home/opc/.claude/daemon/scripts/restore-daemon.sh`

**Systemd:**
- `~/.config/systemd/user/daemon-backup-{daily,weekly,monthly,yearly}.{service,timer}`

**Config:**
- `~/.config/rclone/rclone.conf`

## Monitoring

**View backup logs:**
```bash
journalctl --user -u daemon-backup-daily.service -n 50
```

**Check R2 usage:**
```bash
rclone size r2-daemon:claude-orc/
```

**Test backup system:**
```bash
~/.claude/daemon/scripts/backup-daemon.sh test
```

---

**Implemented**: 2025-11-02
**Status**: Operational ✅
**Tested**: ✅
