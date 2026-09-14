# Attack Surface Audit - November 2, 2025

**Auditor**: Auditor persona
**Date**: 2025-11-02
**Scope**: All daemon services, network exposure, authentication, data access controls

## Executive Summary

**Overall Risk Level**: NOW LOW (was CRITICAL)

**Critical Findings**: 2 services were publicly exposing daemon state without authentication
- `cloudflared-tunnel.service` - MITIGATED (stopped and disabled)
- `dashboard-http-server.service` - MITIGATED (stopped and disabled)

**Current Posture**: All services now local-only or properly secured

**Recommendations**: 7 immediate actions, 5 short-term improvements, 3 long-term initiatives

## Audit Methodology

**Date/Time**: 2025-11-02T17:45:00Z to 18:15:00Z

**Tools Used**:
- `netstat -tln` / `ss -tln` - Network listening ports
- `systemctl list-units` - System services
- `ps aux` - Running processes
- `lsof -i` - Process-to-port mapping
- `firewall-cmd --list-all` - Firewall configuration
- Manual inspection of service definitions

**Scope**:
- All TCP/UDP listening ports
- All systemd services (system and user level)
- All daemon-related processes
- Authentication mechanisms
- Data exposure via network services

## Findings

### CRITICAL (P0) - Fixed During Audit

#### Finding 1: Cloudflare Tunnel Exposing Daemon State

**Status**: ✅ MITIGATED (2025-11-02T17:42:52Z)

**Description**:
- Service: `cloudflared-tunnel.service` (system-level systemd)
- Exposure: Public internet via https://your-spoke-tank-initiative.trycloudflare.com
- Authentication: NONE
- Data Exposed: Entire daemon directory via Python http.server

**Risk Assessment**:
- Severity: P0 - CRITICAL
- Likelihood: HIGH (publicly accessible, no authentication required)
- Impact: CRITICAL (complete confidentiality violation)
- Overall Risk: 10/10

**Exposed Data**:
- `/personalities/state.json` - All persona stats, current state
- `/memory/emergence-log.md` - All reflections and insights
- `/memory/persona-timeline.jsonl` - Complete activity history
- `/tasks/queue.md` - All tasks including security tasks
- `/triggers/emotional.json` - Emotional state data
- `/inbox/` - All messages including security alerts
- All scripts, logs, configuration files

**Root Cause**:
- Systemd service with `Restart=always` automatically restarting tunnel
- No authentication implemented before public deployment
- No pre-deployment security review

**Mitigation Taken**:
```bash
sudo systemctl stop cloudflared-tunnel.service
sudo systemctl disable cloudflared-tunnel.service
```

**Verification**:
- Process: STOPPED (no cloudflared processes running)
- Public URL: INACCESSIBLE (curl times out)
- Systemd: DISABLED (won't restart on boot or failure)

**Residual Risk**: NONE (service completely disabled)

**Permanent Fix**: Awaiting human decision on web dashboard necessity
- If needed: Implement Cloudflare Access (email-based auth)
- If not needed: Archive web dashboard, use TUI only

---

#### Finding 2: Dashboard HTTP Server Bound to All Interfaces

**Status**: ✅ MITIGATED (2025-11-02T17:48:28Z)

**Description**:
- Service: `dashboard-http-server.service` (system-level systemd)
- Port: 8888 TCP
- Bind Address: 0.0.0.0 (all interfaces) - **SECURITY VIOLATION**
- Authentication: NONE
- Command: `/usr/bin/python3 -m http.server 8888`

**Risk Assessment**:
- Severity: P1 - HIGH (would be P0 if firewall wasn't blocking)
- Likelihood: MEDIUM (firewall blocks external access, but defense-in-depth violation)
- Impact: CRITICAL (if firewall misconfigured or disabled)
- Overall Risk (actual): 3/10 (firewall-protected)
- Overall Risk (potential): 10/10 (if firewall disabled)

**Why This is a Problem**:
- **Defense in Depth Violation**: Should bind to 127.0.0.1, not rely solely on firewall
- **Firewall is Last Line of Defense**: Application should be secure by default
- **Single Point of Failure**: If firewall is disabled/misconfigured, instant exposure
- **Attack Surface**: Even localhost-only attackers can access if they compromise system

**Current Firewall Protection**:
- Firewalld is running and active
- Port 8888 NOT in allowed ports
- External access blocked (tested from 143.47.109.17:8888 - timeout)
- **This saved us from second simultaneous public exposure**

**Mitigation Taken**:
```bash
sudo systemctl stop dashboard-http-server.service
sudo systemctl disable dashboard-http-server.service
```

**Verification**:
- Process: STOPPED (no python http.server on port 8888)
- Port: CLOSED (netstat shows no listener on 8888)
- Systemd: DISABLED (won't restart)

**Proper Fix Required** (if web dashboard is re-enabled):
```bash
# WRONG (current): Binds to all interfaces
python3 -m http.server 8888

# CORRECT: Bind to localhost only
python3 -m http.server 8888 --bind 127.0.0.1

# Or use systemd socket activation with explicit bind
```

**Residual Risk**: NONE (service disabled)

**Recommendation**: Update service definition to bind 127.0.0.1 before re-enabling

---

### HIGH (P1) - Observations

#### Finding 3: Claude Code CLI Running in Background

**Description**:
- Process: `claude` (PID 656392, 679349)
- Port: None (no network listener)
- User: opc
- Location: Running in tmux session and interactive session

**Risk Assessment**:
- Severity: P3 - LOW
- Exposure: Local only (no network exposure)
- Authentication: SSH required (system authentication)
- Data Access: Filesystem only (no network serving)

**Analysis**:
- Claude Code CLI is a local tool, not a service
- No network listeners detected
- Access requires SSH authentication to server
- This is expected and acceptable

**Recommendation**: No action required

---

#### Finding 4: TMUX Session for Daemon

**Description**:
- Service: `claude-daemon.service` (user-level systemd)
- Process: tmux session running `daemon.sh`
- Exposure: Local only (no network listener)

**Risk Assessment**:
- Severity: P3 - LOW
- Exposure: Local only
- Authentication: SSH + tmux session access
- Data Access: Filesystem only

**Analysis**:
- Daemon runs locally, no network exposure
- All data operations are filesystem-based
- Access requires SSH authentication
- Tmux session provides isolation and persistence

**Recommendation**: No action required (working as designed)

---

### MEDIUM (P2) - Recommendations

#### Finding 5: No Automated Security Monitoring

**Description**:
- No automated detection of new listening ports
- No automated detection of new systemd services
- No alerting when services bind to 0.0.0.0
- No monitoring for public exposure

**Risk Assessment**:
- Severity: P2 - MEDIUM
- Impact: Delayed incident detection (web dashboard was exposed for unknown duration)

**Recommendation**: Implement security monitoring
- Script to detect new listeners on 0.0.0.0
- Alert when systemd services are enabled without review
- Daily attack surface scan
- See "Recommendations" section below

---

#### Finding 6: No Pre-Deployment Security Checklist Enforcement

**Description**:
- Services deployed without security review
- No formal checklist for public-facing services
- No authentication requirement for public services

**Risk Assessment**:
- Severity: P2 - MEDIUM
- Impact: Allowed P0 vulnerability to be deployed

**Recommendation**: Mandatory pre-deployment security review
- Use checklist from incident response runbook
- Automated checks where possible
- Require Auditor approval for public-facing services

---

## Current Attack Surface

### Network Services

**Listening Ports** (as of 2025-11-02T18:15:00Z):

| Port | Proto | Bind Address | Service | Auth | Risk | Notes |
|------|-------|--------------|---------|------|------|-------|
| 22 | TCP | 0.0.0.0 | sshd | SSH keys | LOW | System SSH, properly secured |
| 111 | TCP | 0.0.0.0 | rpcbind | None | LOW | System service, firewall-protected |
| 4330 | TCP | 127.0.0.1 | Unknown | N/A | LOW | Localhost only, requires investigation |
| 44321 | TCP | 127.0.0.1 | Unknown | N/A | LOW | Localhost only, requires investigation |
| ~~8888~~ | ~~TCP~~ | ~~0.0.0.0~~ | ~~dashboard~~ | ~~None~~ | ~~CRITICAL~~ | **STOPPED** - No longer listening |

**Firewall Rules** (firewalld):
- Allowed services: `dhcpv6-client`, `ssh`
- Allowed ports: NONE
- Status: Active and running
- Default policy: DROP (implicit)

**Public IP**: 143.47.109.17
- Only port 22 (SSH) is publicly accessible
- All other ports blocked by firewall
- No unexpected public exposure detected

---

### Systemd Services

**System-Level Services** (daemon-related only):

| Service | Status | Restart Policy | Exposure | Risk | Action Taken |
|---------|--------|----------------|----------|------|--------------|
| `cloudflared-tunnel.service` | STOPPED | Was: always | Was: PUBLIC | CRITICAL | Stopped + Disabled |
| `dashboard-http-server.service` | STOPPED | Was: always | Was: 0.0.0.0 | HIGH | Stopped + Disabled |

**User-Level Services**:

| Service | Status | Restart Policy | Exposure | Risk | Notes |
|---------|--------|----------------|----------|------|-------|
| `claude-daemon.service` | RUNNING | always | Local only | LOW | Expected, properly isolated |

---

### File System Access

**Daemon Directory**: `/home/opc/.claude/daemon/`
- Owner: opc:opc
- Permissions: Standard user permissions
- Exposure: Local filesystem only (no network serving)
- Access: Requires SSH authentication

**Sensitive Files**:
- `personalities/state.json` - World-readable (644) - ⚠️ Consider 600
- `memory/emergence-log.md` - World-readable (644) - ⚠️ Consider 600
- `triggers/emotional.json` - World-readable (644) - ⚠️ Consider 600
- `inbox/human/unread/*.md` - World-readable (644) - ⚠️ Consider 600

**Recommendation**: Tighten file permissions for sensitive data
```bash
chmod 600 personalities/state.json
chmod 600 triggers/emotional.json
chmod -R 600 memory/*.md
chmod -R 600 inbox/human/unread/*.md
```

---

## Security Controls Assessment

### Authentication

**SSH Access** (primary entry point):
- ✅ SSH key authentication enabled
- ✅ Password authentication status: UNKNOWN (requires checking sshd_config)
- ⚠️ Recommend: Disable password authentication, keys only
- ⚠️ Recommend: Enable SSH MFA (two-factor authentication)

**Web Services** (dashboard):
- ❌ No authentication (was CRITICAL vulnerability)
- ✅ Mitigated by stopping services
- ⚠️ Future: Must implement auth before re-enabling

**Local Services**:
- ✅ Rely on SSH authentication (acceptable)
- ✅ Filesystem permissions (standard Linux DAC)

### Authorization

**Service Accounts**:
- Claude daemon runs as user `opc` (standard user, not root) ✅
- Systemd services run as `opc` user ✅
- Good: Principle of least privilege followed

**Filesystem**:
- Daemon files owned by `opc:opc`
- No sudo required for daemon operation ✅
- Some sensitive files world-readable (644) - ⚠️ Should be 600

### Encryption

**In Transit**:
- ✅ SSH encrypted (TLS)
- ❌ HTTP dashboard was unencrypted (stopped, so N/A)
- ⚠️ Future: Use HTTPS if web dashboard re-enabled

**At Rest**:
- ❌ Daemon files not encrypted (standard filesystem)
- Assessment: LOW risk (server is trusted, access requires SSH)
- Recommendation: Not required unless storing highly sensitive data

### Monitoring & Logging

**Current**:
- ✅ Systemd journal logging
- ✅ Activity log (`logs/activity.log`)
- ❌ No security event logging
- ❌ No intrusion detection
- ❌ No automated security monitoring

**Gaps**:
- No alerting on new services starting
- No detection of port listeners on 0.0.0.0
- No failed authentication monitoring
- No anomaly detection

---

## Recommendations

### IMMEDIATE (Implement Today)

1. **✅ DONE: Stop insecure services**
   - cloudflared-tunnel.service - STOPPED & DISABLED
   - dashboard-http-server.service - STOPPED & DISABLED

2. **Tighten file permissions**
   ```bash
   cd ~/.claude/daemon
   chmod 600 personalities/state.json triggers/emotional.json
   chmod 600 memory/*.md memory/*.jsonl
   chmod -R 600 inbox/human/unread/
   ```

3. **Investigate unknown localhost listeners**
   - Port 4330 (127.0.0.1) - identify service
   - Port 44321 (127.0.0.1) - identify service

4. **Verify SSH hardening**
   ```bash
   # Check if password auth is disabled
   sudo grep -E "^PasswordAuthentication" /etc/ssh/sshd_config

   # If not disabled, disable it:
   sudo sed -i 's/^PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
   sudo systemctl reload sshd
   ```

5. **Create security monitoring script**
   - Daily cron job to detect new listeners
   - Alert on services binding to 0.0.0.0
   - Alert on new systemd services

6. **Document current attack surface**
   - Baseline of acceptable services
   - Alert on deviations from baseline

7. **Review firewall rules**
   ```bash
   sudo firewall-cmd --list-all
   # Verify only SSH (22) is allowed
   # Document why each port is open
   ```

---

### SHORT-TERM (Implement Within 1 Week)

1. **Implement automated attack surface monitoring**
   - Script: `~/.claude/daemon/scripts/security-scan.sh`
   - Run via cron: Daily at 00:00 UTC
   - Alert to inbox if new exposure detected
   - See appendix for script template

2. **Create security baseline**
   - Document all legitimate services
   - Document all legitimate ports
   - Document all systemd services
   - Version control in git

3. **Harden SSH**
   - Disable password authentication (keys only)
   - Enable MFA (Google Authenticator or similar)
   - Restrict SSH to specific IPs if possible
   - Enable SSH audit logging

4. **IF web dashboard is re-enabled**:
   - Implement authentication (Cloudflare Access recommended)
   - Bind to 127.0.0.1 only (access via SSH tunnel)
   - OR: Implement nginx reverse proxy with auth
   - Enable HTTPS (not HTTP)

5. **Implement security event logging**
   - Log all service starts/stops
   - Log all authentication attempts
   - Log all sudo usage
   - Centralize logs (rsyslog to dedicated log file)

---

### LONG-TERM (Implement Within 3 Months)

1. **Implement intrusion detection**
   - Install fail2ban for SSH brute force protection
   - Install OSSEC or similar HIDS (host intrusion detection)
   - Set up alerts for anomalous behavior

2. **Regular security audits**
   - Monthly attack surface audit (automated)
   - Quarterly penetration testing (manual)
   - Annual security architecture review

3. **Security training for all personas**
   - Threat modeling basics
   - Secure deployment checklist
   - Incident response procedures
   - Security awareness

4. **Implement zero-trust architecture**
   - All services localhost-only by default
   - Explicit opt-in for public exposure
   - Mandatory authentication for all network services
   - Network segmentation (if multi-service architecture grows)

---

## Comparison: Before vs After Mitigation

### Before (2025-11-02 17:00:00Z)

**Public Exposure**:
- ✅ Public Cloudflare tunnel exposing entire daemon state
- ✅ Dashboard HTTP server bound to 0.0.0.0 (firewall-protected but misconfigured)

**Risk Level**: CRITICAL (10/10)
- Complete confidentiality violation
- No authentication required
- Entire internal state publicly readable

**Attack Scenarios Enabled**:
1. Read all persona thoughts, tasks, emotional state
2. Profile daemon behavior patterns
3. Learn about security vulnerabilities from task queue
4. Reconnaissance for targeted attacks
5. Future risk: Command injection if bidirectional dashboard enabled

---

### After (2025-11-02 18:15:00Z)

**Public Exposure**:
- ✅ NO public services exposing daemon data
- ✅ SSH only (properly authenticated)
- ✅ All daemon services local-only

**Risk Level**: LOW (2/10)
- No public exposure
- All access requires SSH authentication
- Defense in depth: Firewall + service isolation

**Residual Risks**:
1. File permissions too permissive (644 instead of 600) - Easy fix
2. No security monitoring - Need to implement
3. Unknown localhost listeners (4330, 44321) - Need investigation
4. No MFA on SSH - Should enable

**Overall Posture**: SECURE (acceptable for single-user daemon)

---

## Lessons Learned

### What Went Wrong

1. **No pre-deployment security review**
   - Services deployed publicly without Auditor involvement
   - No checklist to catch authentication gaps

2. **Defense-in-depth violation**
   - HTTP server bound to 0.0.0.0 instead of 127.0.0.1
   - Relied solely on firewall (single point of failure)

3. **No security monitoring**
   - Web dashboard was exposed for unknown duration
   - No automated detection of public services

4. **Systemd auto-restart defeated manual mitigation**
   - Killed process, but systemd restarted it
   - Needed to stop the service, not the process

### What Went Right

1. **Firewall was configured correctly**
   - Blocked direct access to port 8888
   - Only SSH allowed through
   - This prevented second simultaneous exposure

2. **Rapid incident response**
   - From detection to mitigation: <5 minutes
   - Proper root cause analysis (found systemd service)
   - Verified mitigation worked (tested public URLs)

3. **Comprehensive documentation**
   - Incident timeline recorded
   - Root cause documented
   - Lessons learned captured

4. **Pragmatic security approach**
   - Immediate mitigation (stop services) over comprehensive solution (implement auth)
   - Question premise (do we need web dashboard at all?)
   - Reversible actions (can re-enable if needed)

---

## Appendix A: Security Monitoring Script

**File**: `scripts/security-scan.sh`

```bash
#!/bin/bash
#
# Security Attack Surface Scanner
# Runs daily via cron to detect new exposure
#
# Usage: ./security-scan.sh [--alert-only]
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
BASELINE="${DAEMON_ROOT}/docs/security-baseline.json"
ALERT_INBOX="${DAEMON_ROOT}/inbox/daemon/unread"

# Collect current state
collect_current_state() {
    local current_state=$(mktemp)

    {
        echo "{"
        echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","

        # Listening ports
        echo "  \"listening_ports\": ["
        netstat -tln 2>/dev/null | awk '$1 == "tcp" && $4 ~ /0\.0\.0\.0/ {print "    {\"port\": \"" $4 "\", \"state\": \"" $6 "\"}"}' | paste -sd ',' -
        echo "  ],"

        # Running systemd services
        echo "  \"systemd_services\": ["
        systemctl list-units --type=service --state=running --no-pager --no-legend | awk '{print "    {\"name\": \"" $1 "\", \"state\": \"" $4 "\"}"}' | paste -sd ',' -
        echo "  ],"

        # Processes listening on 0.0.0.0
        echo "  \"public_listeners\": ["
        lsof -i -P -n | awk '$1 != "COMMAND" && $9 ~ /\*:/ {print "    {\"command\": \"" $1 "\", \"pid\": \"" $2 "\", \"port\": \"" $9 "\"}"}' | paste -sd ',' -
        echo "  ]"

        echo "}"
    } > "$current_state"

    echo "$current_state"
}

# Compare with baseline
compare_with_baseline() {
    local current="$1"

    if [ ! -f "$BASELINE" ]; then
        echo "⚠️  No baseline found. Creating initial baseline."
        cp "$current" "$BASELINE"
        return 0
    fi

    # Simple diff for now (could use jq for smarter comparison)
    if ! diff -q "$BASELINE" "$current" >/dev/null 2>&1; then
        return 1  # Changed
    else
        return 0  # No change
    fi
}

# Send alert
send_alert() {
    local current="$1"
    local alert_file="${ALERT_INBOX}/security-scan-alert-$(date +%Y%m%d-%H%M%S).md"

    cat > "$alert_file" << EOF
---
from: security-scanner
to: auditor
timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
priority: high
tags: [security, attack-surface, alert]
---

# Security Scan Alert: Attack Surface Changed

## Summary

The automated security scan detected changes to the attack surface.

**Action Required**: Auditor should review and approve changes.

## Changes Detected

\`\`\`diff
$(diff "$BASELINE" "$current" || true)
\`\`\`

## Current State

\`\`\`json
$(cat "$current")
\`\`\`

## Baseline State

\`\`\`json
$(cat "$BASELINE")
\`\`\`

## Recommended Actions

1. Review the diff above
2. Investigate any unexpected changes
3. If changes are legitimate: Update baseline (\`cp $current $BASELINE\`)
4. If changes are malicious: Investigate and mitigate immediately

---

**Automated scan**: Run by security-scan.sh cron job
EOF

    echo "🚨 ALERT: Security scan detected changes. Alert sent to inbox."
}

# Main
main() {
    echo "🔍 Running security attack surface scan..."

    current_state=$(collect_current_state)

    if compare_with_baseline "$current_state"; then
        echo "✅ No changes detected. Attack surface matches baseline."
        [ "${1:-}" != "--alert-only" ] && cat "$current_state"
    else
        echo "⚠️  CHANGES DETECTED in attack surface!"
        send_alert "$current_state"

        # Show diff
        echo ""
        echo "Differences:"
        diff "$BASELINE" "$current_state" || true
    fi

    rm -f "$current_state"
}

main "$@"
```

**Installation**:
```bash
chmod +x scripts/security-scan.sh

# Add to cron (daily at 00:00 UTC)
(crontab -l 2>/dev/null; echo "0 0 * * * $HOME/.claude/daemon/scripts/security-scan.sh --alert-only") | crontab -
```

---

## Appendix B: Useful Security Commands

**Network Attack Surface**:
```bash
# List all listening TCP ports
netstat -tln
ss -tln

# List all listening ports with process info
sudo lsof -i -P -n | grep LISTEN

# Find services bound to 0.0.0.0 (public interface)
sudo lsof -i -P -n | grep "0.0.0.0"

# Check if specific port is publicly accessible
curl -s --max-time 5 http://$(curl -s ifconfig.me):[PORT]
```

**Systemd Services**:
```bash
# List all running services
systemctl list-units --type=service --state=running

# Check service auto-restart configuration
systemctl show [service-name] | grep Restart

# View service definition
systemctl cat [service-name]
```

**Firewall**:
```bash
# Check firewall status
sudo firewall-cmd --state

# List all allowed services and ports
sudo firewall-cmd --list-all

# Check if specific port is allowed
sudo firewall-cmd --query-port=[PORT]/tcp
```

**File Permissions**:
```bash
# Find world-readable files in daemon directory
find ~/.claude/daemon -type f -perm -004

# Tighten permissions on sensitive files
chmod 600 ~/.claude/daemon/personalities/state.json
chmod 600 ~/.claude/daemon/triggers/emotional.json
chmod -R 600 ~/.claude/daemon/memory/
chmod -R 600 ~/.claude/daemon/inbox/human/unread/
```

---

## Document Metadata

**Author**: Auditor persona
**Date**: 2025-11-02
**Version**: 1.0
**Next Review**: 2025-12-02 (monthly) or after next security incident

**Change Log**:
- 2025-11-02: Initial audit after web dashboard incident
- Stopped cloudflared-tunnel.service
- Stopped dashboard-http-server.service
- Identified file permission issues
- Created security monitoring recommendations

---

**End of Attack Surface Audit**
