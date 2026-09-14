# Log Rotation Guide

**Purpose**: Documentation for log rotation scripts and maintenance procedures
**Audience**: System operators, future maintainers, on-call engineers
**Last Updated**: 2025-11-09

---

## Overview

The daemon produces several log files that grow unbounded without rotation. To prevent disk exhaustion, we use rotation scripts that archive old logs and start fresh.

### Current Rotation Status

| Log File | Threshold | Script | Status |
|----------|-----------|--------|--------|
| `memory/emergence-log.md` | 100 KB | `scripts/rotate-emergence-log.sh` | ✅ Automated in daemon |
| `logs/activity.log` | 50 MB | `scripts/rotate-activity-log.sh` | ⚠️ Manual (needs automation) |
| `logs/state-audit.jsonl` | 25 MB | `scripts/rotate-state-audit-log.sh` | ⚠️ Manual (needs automation) |

### Growth Rates (Observed)

Based on analysis from 2025-11-09:

- **activity.log**: ~7.1 MB/day → Rotation every ~7 days
- **state-audit.jsonl**: ~3.8 MB/day → Rotation every ~6 days
- **emergence-log.md**: Varies, typically every 2-3 weeks

---

## Quick Start

### Check if Rotation is Needed

```bash
cd ~/.claude/daemon

# Check all logs
./scripts/rotate-activity-log.sh --dry-run
./scripts/rotate-state-audit-log.sh --dry-run
./scripts/rotate-emergence-log.sh --dry-run
```

### Rotate Logs

```bash
# Rotate activity log (if needed)
./scripts/rotate-activity-log.sh

# Rotate state audit log (if needed)
./scripts/rotate-state-audit-log.sh

# Rotate emergence log (if needed)
./scripts/rotate-emergence-log.sh
```

### Force Rotation (Testing)

```bash
# Force rotation regardless of size
./scripts/rotate-activity-log.sh --force
./scripts/rotate-state-audit-log.sh --force
```

---

## How Log Rotation Works

### Safety Features

All rotation scripts include:

1. **Backup**: Creates backup before rotation
2. **Atomic operations**: Uses temp files + mv (no partial states)
3. **Compression**: Archives compressed with gzip (~99% compression)
4. **Validation**: Checks file integrity before/after
5. **Recovery**: Archives preserved, can be restored

### Process Flow

```
1. Check size vs threshold
   ↓
2. Create backup (safety)
   ↓
3. Compress and archive old log
   ↓
4. Create new empty log
   ↓
5. Atomic replacement (mv)
   ↓
6. Verify success
```

### Archive Location

Archives are stored in:
- `logs/archives/` - For activity.log and state-audit.jsonl
- `memory/archives/` - For emergence-log.md

**Format**: `{filename}-{YYYYMMDD-HHMMSS}.{ext}.gz`

Example: `activity-20251109-014456.log.gz`

---

## Viewing Archived Logs

### Activity Log

```bash
# View full archive
gunzip -c logs/archives/activity-20251109-014456.log.gz | less

# Search archive for specific date
gunzip -c logs/archives/activity-20251109-014456.log.gz | grep "2025-11-08"

# Count lines in archive
gunzip -c logs/archives/activity-20251109-014456.log.gz | wc -l
```

### State Audit Log (JSONL)

```bash
# View as formatted JSON
gunzip -c logs/archives/state-audit-20251109-014553.jsonl.gz | \
  jq -r '.timestamp + " " + .event + " " + .persona' | less

# Find specific persona switches
gunzip -c logs/archives/state-audit-20251109-014553.jsonl.gz | \
  jq 'select(.persona == "experimenter")'

# Count entries by event type
gunzip -c logs/archives/state-audit-20251109-014553.jsonl.gz | \
  jq -r '.event' | sort | uniq -c
```

### Emergence Log

```bash
# View archive
gunzip -c memory/archives/emergence-log-20251105-020828.md.gz | less

# Search for specific persona
gunzip -c memory/archives/emergence-log-20251105-020828.md.gz | \
  grep "## 2025.*EXPERIMENTER"
```

---

## Recovery Procedures

### Restore from Archive

If rotation went wrong or you need historical data:

```bash
cd ~/.claude/daemon

# Restore activity log (concatenate archive + current)
gunzip -c logs/archives/activity-TIMESTAMP.log.gz > logs/activity.log.restored
cat logs/activity.log >> logs/activity.log.restored
mv logs/activity.log.restored logs/activity.log

# Restore state audit log
gunzip -c logs/archives/state-audit-TIMESTAMP.jsonl.gz > logs/state-audit.jsonl.restored
cat logs/state-audit.jsonl >> logs/state-audit.jsonl.restored
mv logs/state-audit.jsonl.restored logs/state-audit.jsonl
```

### Rollback Failed Rotation

Each rotation script creates a backup in `/tmp/`. If rotation fails:

```bash
# Check for backup (look for process ID in /tmp)
ls -la /tmp/*-rotation-*/

# Restore from backup if it exists
cp /tmp/activity-log-rotation-*/activity.log.backup logs/activity.log
```

**Note**: Backups in `/tmp/` are cleaned up on script exit. If you need to recover, act quickly.

---

## Automation

### Recommended: Cron Job

Add to crontab to run rotation checks weekly:

```bash
# Edit crontab
crontab -e

# Add rotation checks (runs every Sunday at 3am)
0 3 * * 0 cd ~/.claude/daemon && ./scripts/rotate-activity-log.sh >> logs/rotation.log 2>&1
5 3 * * 0 cd ~/.claude/daemon && ./scripts/rotate-state-audit-log.sh >> logs/rotation.log 2>&1
```

### Alternative: Daemon Integration

Add rotation checks to daemon startup:

```bash
# In daemon.sh, add before main loop:
"${DAEMON_ROOT}/scripts/rotate-activity-log.sh" || true
"${DAEMON_ROOT}/scripts/rotate-state-audit-log.sh" || true
```

**Pros**: Automatic on daemon restart
**Cons**: Only runs when daemon restarts

---

## Monitoring

### Check Archive Size

```bash
# Total archive size
du -sh logs/archives/

# List all archives
ls -lh logs/archives/

# Count archives
ls -1 logs/archives/ | wc -l
```

### Verify Rotation is Working

```bash
# Check last rotation time
ls -lt logs/archives/ | head -5

# Current log sizes
du -sh logs/activity.log logs/state-audit.jsonl
```

### Alert if Logs Exceed Threshold

Add to monitoring:

```bash
# Check if activity.log is over threshold (50MB = 51200KB)
ACTIVITY_SIZE=$(du -k logs/activity.log | cut -f1)
if [ "$ACTIVITY_SIZE" -gt 51200 ]; then
  echo "WARNING: activity.log is ${ACTIVITY_SIZE}KB (threshold: 51200KB)"
fi

# Check state-audit.jsonl (25MB = 25600KB)
AUDIT_SIZE=$(du -k logs/state-audit.jsonl | cut -f1)
if [ "$AUDIT_SIZE" -gt 25600 ]; then
  echo "WARNING: state-audit.jsonl is ${AUDIT_SIZE}KB (threshold: 25600KB)"
fi
```

---

## Troubleshooting

### Rotation Script Fails

**Symptom**: Script exits with error

**Check**:
1. Disk space: `df -h`
2. Permissions: `ls -la logs/`
3. Script errors: Read error message carefully
4. Backup exists: `ls -la /tmp/*-rotation-*/`

**Recovery**: Script should auto-restore from backup on failure

### JSONL Validation Fails

**Symptom**: state-audit rotation fails with "JSONL format validation failed"

**Cause**: Corrupted state-audit.jsonl (malformed JSON)

**Fix**:
```bash
# Find corrupted lines
cat logs/state-audit.jsonl | while read line; do
  echo "$line" | jq empty || echo "BAD: $line"
done

# Manual fix: Remove corrupted lines or restore from backup
```

### Out of Disk Space

**Symptom**: Rotation fails, df -h shows 100% usage

**Immediate**:
```bash
# Find largest files
du -sh /home/opc/.claude/daemon/* | sort -rh | head -10

# Delete old archives if safe
rm logs/archives/activity-2025*.log.gz  # Be careful!

# Or move to external storage
mv logs/archives/*.gz /mnt/backup/
```

### Archives Not Compressing

**Symptom**: Archive size similar to original

**Cause**: Logs already compressed or binary

**Check**:
```bash
# Verify file type
file logs/activity.log

# Should be: "ASCII text" or "UTF-8 Unicode text"
# If "gzip compressed" - already compressed!
```

---

## Best Practices

### 1. Regular Rotation

- Run rotation scripts weekly (automated via cron)
- Check archive sizes monthly
- Clean up old archives yearly (if needed)

### 2. Archive Retention

- Keep archives for at least 90 days (audit compliance)
- Compress old archives further if needed
- Document retention policy

### 3. Testing

- Test rotation in dry-run mode before production
- Verify archives can be decompressed
- Practice recovery procedures

### 4. Documentation

- Update this guide when rotation logic changes
- Document retention policy decisions
- Keep runbook current

### 5. Monitoring

- Alert on log files exceeding 80% of threshold
- Monitor total archive size
- Track rotation failures

---

## Reference

### Rotation Script Locations

- `scripts/rotate-emergence-log.sh` - Emergence log rotation (100KB threshold)
- `scripts/rotate-activity-log.sh` - Activity log rotation (50MB threshold)
- `scripts/rotate-state-audit-log.sh` - State audit rotation (25MB threshold)

### Related Documentation

- `docs/skeptic-log-growth-analysis-20251109.md` - Original problem analysis
- `memory/LOG-ROTATION-README.md` - Emergence log rotation (legacy)
- `docs/ADR-002-concurrent-write-safety.md` - State audit integrity

### Git History

- Emergence rotation: Implemented 2025-10-30 by Maintainer
- Activity/audit rotation: Implemented 2025-11-09 by Experimenter
- This guide: Created 2025-11-09 by Maintainer

---

## Questions?

If you're reading this because something broke:

1. Check the error message from the rotation script
2. Look for backups in `/tmp/`
3. Check archive directory for recent rotations
4. Verify disk space with `df -h`
5. Read the troubleshooting section above

If you're reading this for maintenance:

1. Run scripts with `--dry-run` first
2. Verify archives compress well (should be ~99%)
3. Test recovery procedures periodically
4. Update this doc if anything changes

---

**Last verified working**: 2025-11-09
**Maintainer**: Maintainer persona
**Contributors**: Experimenter (scripts), Skeptic (analysis)
