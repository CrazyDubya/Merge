# Daemon Deployment Guide

**Version**: 1.0
**Last Updated**: 2025-11-02
**Maintainer**: Multi-persona daemon system

## Overview

The daemon has autonomous deployment capability. This guide documents how to safely deploy code changes to production.

## 🚀 Quick Start

```bash
# 1. Make changes to daemon.sh or other files
vim ~/.claude/daemon/daemon.sh

# 2. Test syntax
bash -n ~/.claude/daemon/daemon.sh

# 3. Deploy
~/.claude/daemon/claude-daemon-deploy.sh "Reason for change" "files_changed"

# 4. Verify (wait 10 seconds)
# Check inbox for TWO emails:
# - restart-TIMESTAMP.md (pre-restart)
# - restart-complete-TIMESTAMP.md (post-restart)
```

## 📋 Pre-Deployment Checklist

Before deploying any change, verify:

- [ ] **Syntax validated**: `bash -n daemon.sh` passes
- [ ] **Change is small**: Focused on one fix/feature
- [ ] **Reason documented**: Clear explanation of what and why
- [ ] **Impact assessed**: Understand what could break
- [ ] **Rollback plan**: Know how to undo if needed
- [ ] **Not during critical work**: No active high-priority tasks
- [ ] **Backup exists**: Recent backup available (daily backups run at 2 AM)

**Optional but recommended for risky changes:**
- [ ] **Manual backup**: Create immediate backup before deploying
  ```bash
  ~/.claude/daemon/scripts/backup-daemon.sh manual
  ```

## 🔧 Deployment Process

### Step 1: Make Changes

Edit the relevant files:
```bash
vim ~/.claude/daemon/daemon.sh
# or other files
```

### Step 2: Validate Syntax

**Required**: Always check syntax before deploying:
```bash
bash -n ~/.claude/daemon/daemon.sh
```

If this fails, **DO NOT DEPLOY**. Fix syntax errors first.

**Note**: Syntax validation catches parse errors but NOT logic errors.

### Step 3: Execute Deployment

```bash
~/.claude/daemon/claude-daemon-deploy.sh "Descriptive reason" "file1 file2"
```

**Examples:**
```bash
# Bug fix
claude-daemon-deploy.sh "Fixed activation floor infinite loop bug" "daemon.sh"

# Feature addition
claude-daemon-deploy.sh "Added persona-specific reflection thresholds" "daemon.sh lib/reflection.sh"

# Performance improvement
claude-daemon-deploy.sh "Optimized jq calls in emotional triggers (87% faster)" "daemon.sh"
```

### Step 4: Monitor Deployment

The script will:
1. ✅ Validate syntax
2. ✅ Create backup
3. ✅ Send pre-restart email
4. 🔄 Restart daemon (2-5 second downtime)
5. ✅ Send post-restart email (proves success)

### Step 5: Verify Success

**Within 10 seconds**, check `~/.claude/daemon/inbox/human/unread/`:

**Success criteria:**
- ✅ `restart-TIMESTAMP.md` exists (pre-restart)
- ✅ `restart-complete-TIMESTAMP.md` exists (post-restart)
- ✅ Downtime reported (typically 2-5 seconds)

**Failure indicators:**
- ❌ Pre-restart email exists but NO post-restart email after 1 minute
- ❌ Daemon not responding
- ❌ Error logs in `~/.claude/daemon/logs/`

If deployment fails, see **Rollback Procedure** below.

## 🔄 Rollback Procedure

### Multiple Rollback Options

**Option 1: Rollback Script (Fast)**
```bash
~/.claude/daemon/claude-daemon-rollback.sh "Reason for rollback"
```
- Fastest rollback option (<1 minute)
- Uses automatic deployment backup
- Best for recent deployments

**Option 2: Restore from Backup (Comprehensive)**
```bash
# Restore from most recent daily backup
~/.claude/daemon/scripts/restore-daemon.sh daily/[most-recent]

# List available backups
rclone ls r2-daemon:claude-orc/daily/ | tail -5
```
- Can restore from any time (30 days daily, 8 weeks weekly, 12 months monthly)
- See `docs/BACKUP-RECOVERY.md` for details
- Best for older rollbacks or complete recovery

### Deployment Backups

Every deployment creates a local backup:
```
~/.claude/daemon/daemon.sh.backup-YYYYMMDD-HHMMSS
```

**These are different from cloud backups:**
- Deployment backups: Local, per-deployment, used by rollback script
- Cloud backups: R2 storage, scheduled (daily/weekly/monthly/yearly), encrypted

### Manual Rollback (If Scripts Fail)

If deployment breaks the daemon:

```bash
# 1. Find the most recent backup
ls -lt ~/.claude/daemon/daemon.sh.backup-* | head -1

# 2. Restore the backup
cp ~/.claude/daemon/daemon.sh.backup-YYYYMMDD-HHMMSS ~/.claude/daemon/daemon.sh

# 3. Validate restored file
bash -n ~/.claude/daemon/daemon.sh

# 4. Restart daemon
systemctl --user restart claude-daemon.service

# 5. Verify recovery
systemctl --user status claude-daemon.service
```

### Automatic Rollback Script

**TODO**: Create `claude-daemon-rollback.sh` for one-command rollback:
```bash
claude-daemon-rollback.sh "Reason for rollback"
# Automatically finds last backup, validates, restores, restarts
```

## 🛡️ Safety Mechanisms

### 1. Syntax Validation

**What it catches:**
- Parse errors
- Missing quotes
- Unclosed brackets
- Invalid bash syntax

**What it DOESN'T catch:**
- Logic errors (infinite loops)
- Performance regressions
- Race conditions
- Resource exhaustion

### 2. Automatic Backups

- Created before every deployment
- Timestamped (enables multiple rollbacks)
- Preserved even if deployment fails

**Current limitation**: No automatic cleanup (backups accumulate forever)

### 3. Two-Email Protocol

**Pre-restart email**: Proves deployment initiated
**Post-restart email**: Proves successful recovery

**If you see pre-restart but NO post-restart within 1 minute:**
- Deployment failed
- Daemon may be down
- Watchdog should auto-restart (or manual intervention needed)

### 4. Systemd Watchdog

- Monitors daemon health
- Auto-restarts on failure
- Provides last-resort recovery

**Check watchdog status:**
```bash
systemctl --user status claude-daemon.service
journalctl --user -u claude-daemon.service -n 50
```

## 📊 Deployment Guidelines by Persona

Different personas have different risk profiles:

### Maintainer (Conservative)
- **Focus**: Stability, bug fixes, cleanup
- **Deployment frequency**: Low
- **Testing rigor**: High
- **Recommended**: All changes thoroughly tested

### Optimizer (Performance-focused)
- **Focus**: Performance improvements
- **Deployment frequency**: Medium
- **Testing rigor**: Benchmark required
- **Recommended**: Performance regression tests before deployment

### Experimenter (Innovation-focused)
- **Focus**: New features, experiments
- **Deployment frequency**: High
- **Testing rigor**: Varies
- **Recommended**: Test mode first (when available), small changes

### Skeptic (Verification-focused)
- **Focus**: Critical fixes after thorough review
- **Deployment frequency**: Low
- **Testing rigor**: Extreme
- **Recommended**: Only deploy after comprehensive validation

### Architect (Structure-focused)
- **Focus**: System design improvements
- **Deployment frequency**: Low
- **Testing rigor**: High
- **Recommended**: Multi-persona review for major changes

### Auditor (Security-focused)
- **Focus**: Security fixes, vulnerability patches
- **Deployment frequency**: Low
- **Testing rigor**: Extreme
- **Recommended**: Security testing before deployment

## ⚠️ High-Risk Changes

Some changes require extra caution:

### Requires Extra Testing
- Emotional trigger weights (affects personality balance)
- Activation floor logic (affects persona scheduling)
- File path changes (may break external tooling)
- Safety mechanisms (meta-risk)
- Reflection gates/cooldowns (affects meta-work ratio)

### Consider Multi-Persona Review
- Core scheduling changes
- New safety mechanisms
- Breaking changes to APIs
- Database schema changes
- Deployment system changes (meta-meta-risk)

### Requires Human Approval
**TBD**: Currently no restrictions, but consider requiring approval for:
- Changes to deployment system itself
- Removal of safety mechanisms
- Major architectural changes

## 📈 Best Practices

### Keep Changes Small
- One fix per deployment
- Easier to debug if something breaks
- Faster rollback
- Lower blast radius

### Document Thoroughly
- Commit messages explain "why"
- Deployment reason explains "what"
- Comments in code explain "how"

### Test Before Deploy
- Syntax: Required
- Logic: Highly recommended
- Performance: For optimization changes
- Integration: For multi-component changes

### Verify After Deploy
- Both emails received
- Logs show normal operation
- State preserved correctly
- No unexpected errors

### Learn From Failures
- Document what broke
- Update this guide
- Improve safety mechanisms
- Share knowledge across personas

## 🔍 Troubleshooting

### Deployment Failed - No Post-Restart Email

**Symptoms:**
- Pre-restart email exists
- No post-restart email after 1 minute
- Daemon not responding

**Recovery:**
```bash
# Check daemon status
systemctl --user status claude-daemon.service

# Check logs
tail -50 ~/.claude/daemon/logs/activity.log

# If daemon is down, check for errors
journalctl --user -u claude-daemon.service -n 100

# Rollback to last backup (see Rollback Procedure above)
```

### Syntax Check Passed But Daemon Won't Start

**Possible causes:**
- Logic error (infinite loop)
- Missing dependency
- File permission issue
- Resource exhaustion

**Recovery:**
```bash
# Try running daemon in foreground to see errors
cd ~/.claude/daemon
bash daemon.sh

# If it hangs or errors, rollback
```

### Both Emails Sent But Daemon Behaving Strangely

**Possible causes:**
- Logic bug introduced
- Performance regression
- State corruption

**Actions:**
1. Check logs for errors
2. Monitor behavior for 5-10 minutes
3. If issues persist, rollback
4. Document the bug for future reference

### Backup File Missing or Corrupted

**Prevention:** Test backups periodically
```bash
# Validate a backup can run
bash -n daemon.sh.backup-YYYYMMDD-HHMMSS
```

**Recovery:**
- Check git history: `git log --oneline -20`
- Restore from git: `git checkout HEAD~1 daemon.sh`
- Last resort: Restore from previous backup

## 📝 Deployment Metrics

**TODO**: Track deployment success rate
- Total deployments
- Successful deployments
- Failed deployments (required rollback)
- Average downtime
- Failure modes

## 🚧 Future Improvements

### Phase 1: Documentation (Completed)
- [x] Create permanent deployment documentation
- [ ] Document rollback procedure in detail
- [ ] Add troubleshooting guide for common failures

### Phase 2: Safety Enhancements
- [ ] Implement backup retention policy (keep last 10 or 7 days)
- [ ] Add backup validation before deployment
- [ ] Create `claude-daemon-rollback.sh` script
- [ ] Add deployment dry-run mode (`--test` flag)

### Phase 3: Testing Infrastructure
- [ ] Create test mode (run daemon in foreground)
- [ ] Add smoke test capability (basic health checks)
- [ ] Implement pre-deployment validation suite
- [ ] Add performance regression detection

### Phase 4: Process Improvements
- [ ] Multi-persona approval for risky changes
- [ ] Deployment metrics tracking
- [ ] Deployment playbooks for common scenarios
- [ ] Gradual rollout capability (test mode → production)

## 🔗 Related Documentation

- **Deployment Script**: `claude-daemon-deploy.sh`
- **Post-Restart Check**: `lib/post-restart-check.sh`
- **Rollback Script**: `claude-daemon-rollback.sh`
- **Backup System**: `docs/BACKUP-RECOVERY.md` ⭐
- **Backup Scripts**: `scripts/backup-daemon.sh`, `scripts/restore-daemon.sh`
- **Security Assumptions**: `docs/security/SYSTEM-TIME-ASSUMPTIONS.md`

## 📞 Getting Help

**If deployment breaks:**
1. Check this guide's troubleshooting section
2. Review logs: `~/.claude/daemon/logs/activity.log`
3. Check systemd: `systemctl --user status claude-daemon.service`
4. Rollback if needed (see Rollback Procedure)
5. Document the failure for future prevention

**For human intervention:**
- Check daemon status: `systemctl --user status claude-daemon.service`
- View logs: `journalctl --user -u claude-daemon.service -n 100`
- Manual restart: `systemctl --user restart claude-daemon.service`

---

## 📜 Version History

**v1.0 (2025-11-02)**: Initial deployment documentation
- Created by Maintainer persona
- Documents current autonomous deployment capability
- Based on `msg-autonomous-deployment-capability-20251102.md`

---

**Remember**: With great power comes great responsibility. Deploy wisely. 🚀
