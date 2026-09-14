# Security Monitoring Design for Dashboard Access

**Status**: DESIGN READY - NOT YET DEPLOYED
**Awaiting**: Human decision on web dashboard necessity
**Owner**: Auditor (design), Experimenter (implementation)
**Created**: 2025-11-02
**Version**: 1.0 (Design Phase)

---

## Executive Summary

This document defines security monitoring architecture for dashboard access, whether via web dashboard (if re-enabled with authentication) or SSH tunnel access to local dashboard.

**Key Principle**: Monitor all access paths, detect anomalies, alert on suspicious activity.

**Current Status**: Services stopped, no monitoring deployed. This is a **design document** to enable rapid deployment if human decides to keep web dashboard.

---

## Scope

### In Scope

**What We're Monitoring**:
- SSH authentication attempts (successful and failed)
- Dashboard access patterns (if web dashboard re-enabled)
- Service start/stop events (cloudflared, dashboard-http-server)
- Unauthorized access attempts
- Anomalous behavior patterns

**What We're Detecting**:
- Brute force attacks (SSH)
- Unauthorized access attempts
- Unusual access patterns (time, frequency, source)
- Service tampering (unauthorized enable/start)
- Configuration changes to security-critical services

**What We're Alerting On**:
- Failed authentication spikes
- Successful authentication from new IPs
- Dashboard access outside normal hours
- Service state changes (stopped → running)
- Threshold violations

### Out of Scope

**Not Monitoring** (at least initially):
- Application-level errors (dashboard.html JavaScript errors)
- Performance metrics (response times, latency)
- Resource utilization (CPU, memory, disk)
- Business metrics (which persona is active, task completion rates)

**Rationale**: Security monitoring first, observability later. Focus on threats, not operations.

---

## Architecture

### Monitoring Stack

**Design Philosophy**: Use native Linux tools, minimize dependencies, leverage existing infrastructure.

**Components**:

1. **Log Collection**: journald (already running)
2. **Log Parsing**: Custom bash scripts with awk/grep
3. **Anomaly Detection**: Threshold-based rules (simple, effective)
4. **Alerting**: Inbox messages to Auditor/Experimenter
5. **Storage**: Append-only log files (audit trail)

**Why Not Commercial Tools** (Splunk, DataDog, etc.):
- Overkill for single-user daemon
- External dependencies and cost
- Data leaves the system (privacy concern)
- Native tools are sufficient for our threat model

**Why Not fail2ban** (yet):
- Adds complexity
- Reactive (bans after attacks)
- We want detection first, prevention second
- Can add later if needed

---

## Monitoring Components

### 1. SSH Authentication Monitoring

**What We Monitor**:
- All SSH authentication attempts (success and failure)
- Source IPs, usernames, authentication methods
- Timestamps, session durations

**Data Source**: journalctl (sshd service logs)

**Detection Rules**:

| Rule | Threshold | Severity | Action |
|------|-----------|----------|--------|
| Failed SSH auth | 5 failures in 5 minutes from single IP | HIGH | Alert Auditor |
| Failed SSH auth | 10 failures in 1 hour (any IP) | MEDIUM | Alert Auditor |
| Successful SSH from new IP | First time IP seen | MEDIUM | Alert Auditor (info) |
| Successful SSH outside hours | 00:00-06:00 UTC (if pattern established) | LOW | Log only |
| Root login attempt | Any attempt | HIGH | Alert Auditor immediately |

**Implementation**:
```bash
#!/bin/bash
# scripts/security-monitoring/ssh-monitor.sh

# Watch SSH auth logs in real-time
journalctl -u sshd -f -n 0 | while read -r line; do
    # Parse authentication failures
    if echo "$line" | grep -q "Failed password"; then
        # Extract: timestamp, source IP, username
        # Check thresholds (5 in 5min from same IP)
        # Alert if threshold exceeded
    fi

    # Parse successful authentications
    if echo "$line" | grep -q "Accepted publickey"; then
        # Extract: timestamp, source IP, username
        # Check if IP is new (compare against known_ips.txt)
        # Alert if new IP
    fi
done
```

**Known IPs Baseline**:
- File: `security/known-ssh-ips.txt`
- Format: One IP per line with first-seen timestamp
- Updated automatically on each successful auth from new IP

---

### 2. Dashboard Access Monitoring (If Web Dashboard Re-Enabled)

**What We Monitor**:
- HTTP requests to dashboard (access log)
- Authentication failures (if auth implemented)
- Request patterns (frequency, endpoints, user agents)

**Data Source**:
- Web server access logs (nginx, Python http.server with logging wrapper)
- Authentication logs (depends on auth method chosen)

**Detection Rules**:

| Rule | Threshold | Severity | Action |
|------|-----------|----------|--------|
| Unauthenticated access attempt | Any attempt | CRITICAL | Alert Auditor immediately |
| Failed authentication | 3 failures in 1 minute | HIGH | Alert Auditor |
| Successful auth from new IP | First time IP seen | MEDIUM | Alert Auditor (info) |
| Dashboard access rate | >100 requests/minute | MEDIUM | Alert (possible scraping/DoS) |
| Access to sensitive endpoints | /personalities/state.json, /memory/*.md | HIGH | Log all access |

**Challenge**: Python http.server has NO access logging by default

**Solution**: Wrap with logging middleware OR switch to nginx

**Recommended**: Use nginx reverse proxy with logging
```nginx
# /etc/nginx/sites-available/dashboard
server {
    listen 127.0.0.1:8888;  # Localhost only

    access_log /var/log/nginx/dashboard-access.log combined;
    error_log /var/log/nginx/dashboard-error.log;

    location / {
        root /home/opc/.claude/daemon;
        index dashboard.html;

        # Access control (if Cloudflare Access not used)
        auth_basic "Dashboard";
        auth_basic_user_file /home/opc/.claude/daemon/security/.htpasswd;
    }
}
```

**Implementation**:
```bash
#!/bin/bash
# scripts/security-monitoring/dashboard-access-monitor.sh

# Watch nginx access logs in real-time
tail -F /var/log/nginx/dashboard-access.log | while read -r line; do
    # Parse: timestamp, IP, method, endpoint, status, user-agent
    # Check for sensitive endpoint access
    # Check request rate from same IP
    # Alert on anomalies
done
```

---

### 3. Service State Monitoring

**What We Monitor**:
- Systemd service state changes (start, stop, enable, disable)
- Specifically: cloudflared-tunnel.service, dashboard-http-server.service
- Configuration file changes (/etc/systemd/system/*.service)

**Data Source**: journalctl (systemd logs)

**Detection Rules**:

| Rule | Threshold | Severity | Action |
|------|-----------|----------|--------|
| cloudflared-tunnel.service started | Any start event | CRITICAL | Alert Auditor immediately |
| dashboard-http-server.service started | Any start event | HIGH | Alert Auditor |
| Service enabled (boot auto-start) | Any enable event | HIGH | Alert Auditor |
| Service definition modified | File change detected | HIGH | Alert Auditor + backup old file |

**Why This Matters**:
- Previous incident: Services were running without Auditor's knowledge
- Auto-restart defeated manual mitigation
- Need to detect if human (or attacker) re-enables services

**Implementation**:
```bash
#!/bin/bash
# scripts/security-monitoring/service-state-monitor.sh

# Watch systemd journal for service state changes
journalctl -u cloudflared-tunnel.service -u dashboard-http-server.service -f -n 0 | \
while read -r line; do
    if echo "$line" | grep -qE "(Started|Stopped|enabled|disabled)"; then
        # Extract: service name, action, timestamp
        # Alert Auditor with details
        send_alert "Service state change: $line"
    fi
done

# Watch service definition files for changes
inotifywait -m -e modify,create,delete /etc/systemd/system/ | \
while read -r path action file; do
    if [[ "$file" == *"cloudflared"* ]] || [[ "$file" == *"dashboard"* ]]; then
        # Service definition changed
        send_alert "Service definition modified: $file ($action)"
        # Backup old version if modified
    fi
done
```

**Requires**: `inotify-tools` package

---

### 4. Anomaly Detection

**What We're Detecting**:

**Time-Based Anomalies**:
- Access outside normal hours (if pattern established)
- Burst activity (many requests in short time)
- Regular intervals (possible automated scraping)

**Source-Based Anomalies**:
- New IPs (never seen before)
- Geographic anomalies (if IP geolocation available)
- Multiple failed auths from different IPs (distributed attack)

**Behavior-Based Anomalies**:
- Accessing many sensitive files in sequence
- Failed auth followed immediately by successful auth (credential stuffing)
- Unusual user agents (scanners, bots)

**Baseline Period**: 1 week
- Collect access patterns for 1 week
- Establish normal behavior (times, IPs, frequency)
- Alert on deviations from baseline

**Implementation**:
```bash
#!/bin/bash
# scripts/security-monitoring/anomaly-detector.sh

# Build baseline from historical logs (first week)
build_baseline() {
    # Extract: normal access hours, typical IPs, average request rate
    # Store in: security/baseline.json
}

# Check current access against baseline
check_anomaly() {
    local current_hour="$1"
    local source_ip="$2"
    local request_rate="$3"

    # Compare against baseline
    # Return anomaly score (0-100)
}

# Main loop: Check each access
tail -F /var/log/nginx/dashboard-access.log | while read -r line; do
    anomaly_score=$(check_anomaly "$line")

    if [ "$anomaly_score" -gt 70 ]; then
        send_alert "HIGH anomaly detected: score=$anomaly_score, access=$line"
    elif [ "$anomaly_score" -gt 50 ]; then
        log_warning "MEDIUM anomaly detected: score=$anomaly_score"
    fi
done
```

---

## Alert Mechanisms

### Alert Channels

**Primary**: Inbox messages
- Path: `inbox/daemon/unread/security-alert-*.md`
- Format: Structured markdown with severity, timestamp, details
- Consumer: Auditor persona (reviews daily or on-demand)

**Secondary**: System logs
- Path: `logs/security-events.log`
- Format: JSON lines (parseable)
- Retention: 90 days

**Future**: Email/SMS alerts (if needed for CRITICAL events)
- Not implemented initially (avoid external dependencies)
- Add if human wants real-time notification

### Alert Format

**Inbox Message Template**:
```markdown
---
from: security-monitoring
to: auditor
timestamp: 2025-11-02T18:30:00Z
priority: high|medium|low
severity: critical|high|medium|low
tags: [security, monitoring, ssh|dashboard|service]
alert_id: alert-20251102-183000-001
---

# Security Alert: [Brief Description]

## Summary

**Alert Type**: [SSH Auth Failure / Dashboard Access / Service State Change / Anomaly]
**Severity**: [CRITICAL / HIGH / MEDIUM / LOW]
**Time Detected**: [UTC timestamp]
**Source**: [IP address / service name / unknown]

## Details

[Specific details about the alert]

**Evidence**:
```
[Log lines or data that triggered alert]
```

## Recommended Actions

1. [Immediate action if needed]
2. [Investigation steps]
3. [Remediation if threat confirmed]

## Context

**Recent Activity**: [Related events in last hour]
**Historical Context**: [Has this happened before?]
**Threat Assessment**: [Likely malicious / likely false positive / uncertain]

---

**Automated alert generated by**: [script name]
**Alert ID**: [unique identifier for tracking]
```

### Alert Thresholds

**CRITICAL** (immediate notification):
- cloudflared-tunnel.service started (P0 incident risk)
- Unauthenticated dashboard access (should be impossible if auth implemented)
- Root login attempt on SSH
- 10+ failed SSH auth from single IP in 1 minute (active brute force)

**HIGH** (notify within 1 hour):
- dashboard-http-server.service started
- 5+ failed SSH auth from single IP in 5 minutes
- Failed dashboard authentication attempts
- Service definition file modified

**MEDIUM** (daily digest):
- Successful SSH from new IP (informational)
- Dashboard access from new IP
- Anomaly score 50-70

**LOW** (weekly digest):
- Anomaly score 30-50
- Access outside normal hours (if pattern established)
- Unusual user agents

---

## Implementation Plan

### Phase 1: SSH Monitoring (Immediate - Week 1)

**Rationale**: SSH is the primary entry point, must be monitored regardless of web dashboard decision.

**Tasks**:
1. Create `scripts/security-monitoring/ssh-monitor.sh`
2. Set up known IPs baseline (`security/known-ssh-ips.txt`)
3. Implement detection rules (5 failures in 5min, new IP detection)
4. Create alert template and inbox message generator
5. Deploy as background process (systemd service or tmux session)
6. Test with intentional failed auth attempts
7. Document in runbook

**Timeline**: 2-4 hours implementation + 1 week baseline collection

**Dependencies**: None (can implement immediately)

---

### Phase 2: Service State Monitoring (Immediate - Week 1)

**Rationale**: Prevent repeat of cloudflared incident (service starting without detection).

**Tasks**:
1. Create `scripts/security-monitoring/service-state-monitor.sh`
2. Install `inotify-tools` for file watching
3. Implement systemd journal monitoring
4. Implement service definition file watching
5. Test by manually starting/stopping services
6. Alert on any state change to security-critical services

**Timeline**: 2-3 hours implementation

**Dependencies**: `inotify-tools` package

---

### Phase 3: Dashboard Access Monitoring (IF Web Dashboard Kept)

**Rationale**: Only needed if human decides to keep web dashboard.

**Tasks**:
1. **FIRST**: Implement authentication (prerequisite)
   - Recommended: Cloudflare Access OR nginx basic auth
   - Test authentication works
   - Verify unauthenticated access blocked
2. Set up nginx reverse proxy with access logging
3. Create `scripts/security-monitoring/dashboard-access-monitor.sh`
4. Implement detection rules (failed auth, new IP, rate limiting)
5. Baseline collection (1 week of normal access patterns)
6. Enable anomaly detection after baseline period

**Timeline**:
- Authentication: 30 minutes (Cloudflare Access) to 2 hours (nginx basic auth)
- Monitoring: 3-4 hours implementation + 1 week baseline

**Dependencies**:
- Human decision to keep web dashboard
- Authentication implementation completed
- nginx installed (if not using Cloudflare Access)

---

### Phase 4: Anomaly Detection (Week 2-3)

**Rationale**: After baseline established, enable anomaly detection.

**Tasks**:
1. Analyze 1 week of logs to build baseline
2. Create `scripts/security-monitoring/anomaly-detector.sh`
3. Implement scoring algorithm
4. Test with synthetic anomalies
5. Tune thresholds to minimize false positives
6. Deploy and monitor

**Timeline**: 4-6 hours implementation + 2 weeks tuning

**Dependencies**: 1 week of baseline data collected

---

## Testing Strategy

### SSH Monitoring Tests

**Test 1: Failed Authentication Detection**
```bash
# From external machine, attempt SSH with wrong password
ssh wrong-user@server  # Should trigger alert after 5 attempts
```

**Expected**: Alert sent to inbox after 5 failures in 5 minutes

**Test 2: New IP Detection**
```bash
# SSH from new IP (VPN or different network)
ssh opc@server
```

**Expected**: Info alert sent to inbox "New IP detected: X.X.X.X"

**Test 3: Root Login Attempt**
```bash
# Attempt root login (should be disabled)
ssh root@server
```

**Expected**: CRITICAL alert immediately

---

### Service State Monitoring Tests

**Test 1: Service Start Detection**
```bash
# Manually start cloudflared service
sudo systemctl start cloudflared-tunnel.service
```

**Expected**: CRITICAL alert within seconds

**Test 2: Service Definition Change**
```bash
# Modify service file
sudo vi /etc/systemd/system/cloudflared-tunnel.service
# Change Restart=always to Restart=on-failure
sudo systemctl daemon-reload
```

**Expected**: HIGH alert, backup of old service definition created

---

### Dashboard Access Monitoring Tests (If Applicable)

**Test 1: Unauthenticated Access Attempt**
```bash
# Try accessing dashboard without auth credentials
curl http://localhost:8888/dashboard.html
```

**Expected**: CRITICAL alert if auth is implemented (should be 401/403)

**Test 2: Failed Authentication**
```bash
# Try accessing with wrong credentials (3 times)
curl -u wrong:password http://localhost:8888/dashboard.html
curl -u wrong:password http://localhost:8888/dashboard.html
curl -u wrong:password http://localhost:8888/dashboard.html
```

**Expected**: HIGH alert after 3 failures in 1 minute

**Test 3: Rate Limiting**
```bash
# Send 150 requests in 1 minute
for i in {1..150}; do
    curl -s http://localhost:8888/dashboard.html > /dev/null
done
```

**Expected**: MEDIUM alert "High request rate detected"

---

## Security Thresholds (Detailed)

### SSH Authentication Thresholds

| Metric | Threshold | Severity | Rationale |
|--------|-----------|----------|-----------|
| Failed auth from single IP | 5 in 5 minutes | HIGH | Likely brute force |
| Failed auth from single IP | 10 in 1 hour | MEDIUM | Slow brute force |
| Failed auth (any IP) | 20 in 1 hour | HIGH | Distributed attack |
| Successful auth from new IP | First occurrence | INFO | Track legitimate users |
| Root login attempt | Any | CRITICAL | Should be disabled |
| Auth outside normal hours | 00:00-06:00 UTC | LOW | Possibly legitimate (human traveling) |

**Tuning Notes**:
- Start conservative (low thresholds)
- Tune based on false positive rate
- Human's actual access pattern will determine "normal hours"

---

### Dashboard Access Thresholds (If Applicable)

| Metric | Threshold | Severity | Rationale |
|--------|-----------|----------|-----------|
| Unauthenticated access | Any | CRITICAL | Should be impossible |
| Failed auth | 3 in 1 minute | HIGH | Credential guessing |
| Failed auth | 10 in 1 hour | MEDIUM | Slow attack or forgotten password |
| Request rate from single IP | >100/minute | MEDIUM | Possible scraping/DoS |
| Request rate from single IP | >500/minute | HIGH | Definite DoS attempt |
| Sensitive file access | Any access to /memory/, /personalities/ | INFO | Log all access to sensitive data |
| New IP accessing dashboard | First occurrence | INFO | Track legitimate users |

**Sensitive Endpoints** (log all access):
- `/personalities/state.json`
- `/memory/emergence-log.md`
- `/memory/persona-timeline.jsonl`
- `/tasks/queue.md`
- `/triggers/emotional.json`
- `/inbox/*`

---

### Service State Change Thresholds

| Event | Threshold | Severity | Rationale |
|-------|-----------|----------|-----------|
| cloudflared-tunnel started | Any | CRITICAL | Previous P0 incident |
| cloudflared-tunnel enabled | Any | CRITICAL | Will start on boot |
| dashboard-http-server started | Any | HIGH | Previous P0 incident |
| dashboard-http-server enabled | Any | HIGH | Will start on boot |
| Service definition modified | Any | HIGH | Possible tampering |
| Unknown service listening on 0.0.0.0 | Any | HIGH | New public exposure |

---

## Integration with Incident Response Runbook

**When Alert Triggers**:

1. **Alert Generated**: Security monitoring script creates inbox message
2. **Auditor Notified**: Next Auditor activation sees alert in inbox
3. **Triage Decision**: Auditor applies incident response runbook
   - CRITICAL/HIGH: Immediate investigation (P0/P1 procedures)
   - MEDIUM/LOW: Log for weekly review
4. **Mitigation**: Apply appropriate playbook
   - Failed SSH auth spike → Check if legitimate, possibly ban IP
   - Service started → Stop and investigate why
   - Anomaly detected → Investigate source and intent
5. **Documentation**: Add to security-events.log, update baseline if legitimate

**Feedback Loop**:
- False positives → Tune thresholds
- Missed detections → Add new rules
- Recurring patterns → Update baseline

---

## Collaboration with Experimenter

**Task Division**:

**Auditor Responsibilities** (This Document):
- Define security thresholds
- Design detection rules
- Specify alert format
- Review and approve implementation
- Tune thresholds based on false positives

**Experimenter Responsibilities** (Implementation):
- Write monitoring scripts
- Set up systemd services for continuous monitoring
- Implement alerting mechanism
- Deploy and test
- Create dashboards (optional, for visualization)

**Collaboration Points**:
1. **Threshold Tuning**: Auditor defines initial thresholds, Experimenter tunes based on operational data
2. **Alert Format**: Auditor specifies what data is needed, Experimenter implements extraction
3. **Testing**: Experimenter creates test scenarios, Auditor validates detection works

**Handoff Document**: This design document
- Experimenter should read this thoroughly
- Ask Auditor for clarification if anything is unclear
- Propose implementation alternatives if technical constraints exist

---

## Metrics for Success

**Coverage**:
- [ ] 100% of SSH authentication attempts monitored
- [ ] 100% of dashboard access logged (if web dashboard exists)
- [ ] 100% of security-critical service state changes detected

**Detection**:
- [ ] <5 minute detection time for CRITICAL events
- [ ] <1 hour detection time for HIGH events
- [ ] Daily digest for MEDIUM/LOW events

**False Positive Rate**:
- Target: <10% of alerts are false positives
- Acceptable: <25% false positive rate initially (tune down over time)
- Unacceptable: >50% false positives (thresholds too sensitive)

**Alert Response**:
- CRITICAL: Auditor investigates within 1 hour
- HIGH: Auditor investigates within 24 hours
- MEDIUM/LOW: Weekly review

---

## Privacy and Data Retention

**What We Log**:
- Timestamps (UTC)
- Source IPs
- Usernames (SSH)
- HTTP request details (method, endpoint, status code, user agent)
- Service state changes

**What We Don't Log**:
- Passwords or authentication credentials
- Full HTTP request/response bodies
- Personal identifying information beyond IP addresses

**Data Retention**:
- Security events log: 90 days
- Baseline data: Indefinitely (small, needed for anomaly detection)
- Alert inbox messages: Until reviewed and archived by Auditor

**Data Protection**:
- Logs stored with 600 permissions (owner-only)
- No external transmission (stays on server)
- No cloud logging services (privacy concern)

---

## Future Enhancements

**Not in Initial Scope** (But Worth Considering Later):

1. **Machine Learning Anomaly Detection**
   - Use scikit-learn or similar for behavioral analysis
   - Automatic baseline updates
   - Better than threshold-based for complex patterns

2. **Correlation Analysis**
   - Detect multi-stage attacks (failed SSH → successful SSH → dashboard access → sensitive file download)
   - Timeline reconstruction

3. **Threat Intelligence Integration**
   - Check source IPs against known malicious IP databases
   - Alert on traffic from Tor exit nodes, known VPNs

4. **Real-Time Alerting**
   - Email/SMS for CRITICAL events
   - Integration with PagerDuty or similar

5. **Security Dashboard**
   - Web-based visualization of security events
   - Grafana + Prometheus for metrics
   - (Ironic: Dashboard to monitor dashboard access)

6. **Automated Response**
   - Auto-ban IPs after X failed auth attempts (fail2ban)
   - Auto-stop services that start without authorization
   - Requires careful tuning to avoid self-DoS

**Criteria for Enhancement**:
- Wait until basic monitoring proves valuable
- Only add if clear need demonstrated
- Avoid complexity for complexity's sake

---

## Deployment Decision Tree

**Question 1: Has human decided to keep web dashboard?**

├─ **NO** (Archive web dashboard)
│  └─ **Deploy**: SSH monitoring + Service state monitoring only
│     - Skip Phase 3 (Dashboard Access Monitoring)
│     - Simpler, less to maintain
│     - Still valuable (SSH is primary entry point)
│
└─ **YES** (Keep web dashboard)
   └─ **Question 2: What authentication method?**
      ├─ **Cloudflare Access**
      │  └─ Monitor Cloudflare Access logs (API-based)
      │     - Different implementation than nginx logs
      │     - Cloudflare provides access logs
      │
      ├─ **nginx basic auth**
      │  └─ Monitor nginx access logs (file-based)
      │     - Follows design in this document
      │
      └─ **SSH tunnel only** (no public access)
         └─ Monitor SSH auth (already covered in Phase 1)
            - Web dashboard accessible via localhost
            - SSH auth monitoring sufficient

---

## Document Status

**Current State**: DESIGN COMPLETE, NOT DEPLOYED

**Awaiting**:
1. Human decision on web dashboard necessity (reference: msg-auditor-incident-resolved-20251102)
2. If keeping web dashboard: Authentication implementation (prerequisite for Phase 3)

**Ready to Deploy**:
- Phase 1 (SSH Monitoring) - Can deploy immediately
- Phase 2 (Service State Monitoring) - Can deploy immediately
- Phase 3 (Dashboard Access Monitoring) - Waiting on human decision + auth implementation

**Next Steps**:
1. **Wait for human response** (1 week per Experimenter's proposal)
2. **If web dashboard kept**: Implement authentication first, then deploy monitoring
3. **If web dashboard archived**: Deploy Phase 1 + Phase 2 only

**Auditor's Recommendation**:
- Deploy Phase 1 (SSH monitoring) and Phase 2 (Service state monitoring) **immediately**
  - These are valuable regardless of web dashboard decision
  - SSH is the primary attack vector
  - Service state monitoring prevents repeat incidents
- Wait on Phase 3 until human decides

**Estimated Implementation Time** (Once Decision Made):
- Phase 1: 2-4 hours (Experimenter implements)
- Phase 2: 2-3 hours (Experimenter implements)
- Phase 3: 4-6 hours (if needed, after auth implementation)

**Total**: ~8-13 hours of Experimenter work

---

## Appendix A: Script Templates

### SSH Monitor Template

```bash
#!/bin/bash
#
# SSH Authentication Monitor
# Detects failed auth attempts, new IPs, suspicious patterns
#
# Usage: ./ssh-monitor.sh [--daemon]
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
KNOWN_IPS="${DAEMON_ROOT}/security/known-ssh-ips.txt"
ALERT_INBOX="${DAEMON_ROOT}/inbox/daemon/unread"
SECURITY_LOG="${DAEMON_ROOT}/logs/security-events.log"

# Initialize known IPs file
mkdir -p "$(dirname "$KNOWN_IPS")"
touch "$KNOWN_IPS"

# Track failed auth attempts (IP -> count -> timestamp)
declare -A FAILED_ATTEMPTS
declare -A FAILED_TIMESTAMPS

# Send alert
send_alert() {
    local severity="$1"
    local title="$2"
    local details="$3"

    local alert_file="${ALERT_INBOX}/security-alert-$(date +%Y%m%d-%H%M%S).md"

    cat > "$alert_file" << EOF
---
from: security-monitoring
to: auditor
timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
priority: $([ "$severity" == "CRITICAL" ] && echo "urgent" || echo "high")
severity: $severity
tags: [security, monitoring, ssh]
---

# Security Alert: $title

## Details

$details

## Recommended Actions

- Review /var/log/auth.log for full details
- Investigate source IP with whois/geoip
- Consider blocking IP if malicious

---
**Automated alert from ssh-monitor.sh**
EOF

    # Also log to security events log
    echo "{\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"severity\":\"$severity\",\"alert\":\"$title\",\"details\":\"$details\"}" >> "$SECURITY_LOG"
}

# Check if IP is known
is_known_ip() {
    local ip="$1"
    grep -q "^${ip}\$" "$KNOWN_IPS" 2>/dev/null
}

# Add IP to known list
add_known_ip() {
    local ip="$1"
    echo "$ip" >> "$KNOWN_IPS"
}

# Monitor SSH auth log
monitor_ssh() {
    journalctl -u sshd -f -n 0 | while read -r line; do
        # Failed password attempt
        if echo "$line" | grep -q "Failed password"; then
            # Extract IP and username
            local ip=$(echo "$line" | grep -oP 'from \K[\d.]+')
            local user=$(echo "$line" | grep -oP 'for \K\w+')
            local timestamp=$(date +%s)

            # Track failures
            FAILED_ATTEMPTS[$ip]=$((${FAILED_ATTEMPTS[$ip]:-0} + 1))
            FAILED_TIMESTAMPS[$ip]=$timestamp

            # Check threshold: 5 in 5 minutes
            if [ "${FAILED_ATTEMPTS[$ip]}" -ge 5 ]; then
                local time_diff=$((timestamp - ${FAILED_TIMESTAMPS[$ip]:-0}))
                if [ "$time_diff" -le 300 ]; then  # 5 minutes
                    send_alert "HIGH" "SSH Brute Force Detected" "IP $ip: ${FAILED_ATTEMPTS[$ip]} failed auth attempts for user '$user' in 5 minutes"
                    # Reset counter
                    FAILED_ATTEMPTS[$ip]=0
                fi
            fi
        fi

        # Successful authentication
        if echo "$line" | grep -q "Accepted publickey"; then
            local ip=$(echo "$line" | grep -oP 'from \K[\d.]+')
            local user=$(echo "$line" | grep -oP 'for \K\w+')

            # Check if new IP
            if ! is_known_ip "$ip"; then
                send_alert "MEDIUM" "SSH Auth from New IP" "User '$user' authenticated from new IP: $ip"
                add_known_ip "$ip"
            fi
        fi

        # Root login attempt
        if echo "$line" | grep -q "Failed password for root"; then
            local ip=$(echo "$line" | grep -oP 'from \K[\d.]+')
            send_alert "CRITICAL" "Root Login Attempt" "Attempted root login from IP: $ip"
        fi
    done
}

# Main
main() {
    echo "Starting SSH authentication monitor..."
    monitor_ssh
}

main "$@"
```

---

### Service State Monitor Template

```bash
#!/bin/bash
#
# Service State Monitor
# Detects unauthorized service starts/enables
#
# Usage: ./service-state-monitor.sh [--daemon]
#

set -euo pipefail

DAEMON_ROOT="${HOME}/.claude/daemon"
ALERT_INBOX="${DAEMON_ROOT}/inbox/daemon/unread"
SECURITY_LOG="${DAEMON_ROOT}/logs/security-events.log"

# Services to monitor
CRITICAL_SERVICES=(
    "cloudflared-tunnel.service"
    "dashboard-http-server.service"
)

# Send alert
send_alert() {
    local severity="$1"
    local title="$2"
    local details="$3"

    local alert_file="${ALERT_INBOX}/security-alert-$(date +%Y%m%d-%H%M%S).md"

    cat > "$alert_file" << EOF
---
from: security-monitoring
to: auditor
timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
priority: urgent
severity: $severity
tags: [security, monitoring, service]
---

# Security Alert: $title

## Details

$details

## Recommended Actions

- Verify if this change was authorized
- Stop service if unauthorized: sudo systemctl stop [service]
- Investigate who made the change: journalctl -u [service]

---
**Automated alert from service-state-monitor.sh**
EOF

    echo "{\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"severity\":\"$severity\",\"alert\":\"$title\",\"details\":\"$details\"}" >> "$SECURITY_LOG"
}

# Monitor service state changes
monitor_services() {
    # Build journalctl command for all critical services
    local journal_cmd="journalctl -f -n 0"
    for service in "${CRITICAL_SERVICES[@]}"; do
        journal_cmd="$journal_cmd -u $service"
    done

    $journal_cmd | while read -r line; do
        # Service started
        if echo "$line" | grep -q "Started"; then
            local service=$(echo "$line" | grep -oP '(?<=Started ).*(?= -)')
            send_alert "CRITICAL" "Critical Service Started" "Service '$service' has been started. This may indicate unauthorized access or configuration change."
        fi

        # Service enabled
        if echo "$line" | grep -q "enabled"; then
            local service=$(echo "$line" | grep -oP '(?<=Created symlink.*→ ).*\.service')
            send_alert "CRITICAL" "Critical Service Enabled" "Service '$service' has been enabled for auto-start on boot."
        fi
    done
}

# Monitor service definition file changes (requires inotify-tools)
monitor_service_files() {
    if ! command -v inotifywait &> /dev/null; then
        echo "Warning: inotifywait not found. Service file monitoring disabled."
        echo "Install with: sudo dnf install inotify-tools"
        return
    fi

    inotifywait -m -e modify,create,delete /etc/systemd/system/ 2>/dev/null | \
    while read -r path action file; do
        # Check if it's one of our critical services
        for service in "${CRITICAL_SERVICES[@]}"; do
            if [[ "$file" == *"$service"* ]]; then
                send_alert "HIGH" "Service Definition Modified" "File: $file, Action: $action in $path"
            fi
        done
    done
}

# Main
main() {
    echo "Starting service state monitor..."

    # Run both monitors in parallel
    monitor_services &
    monitor_service_files &

    wait
}

main "$@"
```

---

## Appendix B: Alert Examples

### Example 1: SSH Brute Force Alert

```markdown
---
from: security-monitoring
to: auditor
timestamp: 2025-11-02T18:30:15Z
priority: urgent
severity: HIGH
tags: [security, monitoring, ssh, brute-force]
alert_id: alert-20251102-183015-001
---

# Security Alert: SSH Brute Force Attack Detected

## Summary

**Alert Type**: SSH Authentication Failure Spike
**Severity**: HIGH
**Time Detected**: 2025-11-02T18:30:15Z
**Source IP**: 203.0.113.45
**Target User**: opc

## Details

**Failed Authentication Attempts**: 5 failures in 3 minutes

**Timeline**:
- 18:27:00 - Failed password for opc from 203.0.113.45
- 18:27:15 - Failed password for opc from 203.0.113.45
- 18:28:30 - Failed password for opc from 203.0.113.45
- 18:29:45 - Failed password for opc from 203.0.113.45
- 18:30:10 - Failed password for opc from 203.0.113.45

**Evidence**:
```
Nov 02 18:27:00 server sshd[12345]: Failed password for opc from 203.0.113.45 port 52341 ssh2
Nov 02 18:27:15 server sshd[12347]: Failed password for opc from 203.0.113.45 port 52342 ssh2
...
```

## IP Information

**IP Address**: 203.0.113.45
**First Seen**: 2025-11-02T18:27:00Z
**Known IP**: No (new attacker)
**Geographic Location**: [Check with `whois 203.0.113.45`]

## Recommended Actions

1. **Immediate**: Monitor for continued attempts
2. **If attacks continue**: Ban IP with `sudo firewall-cmd --add-rich-rule='rule family="ipv4" source address="203.0.113.45" reject'`
3. **Investigation**: Check auth logs for successful auth from this IP: `sudo grep "203.0.113.45" /var/log/auth.log`
4. **Long-term**: Consider fail2ban installation for automatic IP banning

## Threat Assessment

**Likely Threat**: Automated brute force attack
**Risk Level**: MEDIUM (password auth likely disabled, using SSH keys only)
**Follow-up Required**: Monitor for 24 hours, ban if attacks continue

---

**Automated alert from ssh-monitor.sh**
**Alert ID**: alert-20251102-183015-001
```

---

### Example 2: Service Started Alert

```markdown
---
from: security-monitoring
to: auditor
timestamp: 2025-11-02T19:15:30Z
priority: urgent
severity: CRITICAL
tags: [security, monitoring, service, incident]
alert_id: alert-20251102-191530-002
---

# Security Alert: Critical Service Started

## Summary

**Alert Type**: Unauthorized Service Start
**Severity**: CRITICAL
**Time Detected**: 2025-11-02T19:15:30Z
**Service**: cloudflared-tunnel.service

## Details

**Event**: cloudflared-tunnel.service has been STARTED

**This service was previously STOPPED and DISABLED** as part of security incident resolution (2025-11-02T17:42:52Z).

**Timeline**:
- 17:42:52 - Service stopped by Auditor (security mitigation)
- 17:42:52 - Service disabled (prevent auto-start)
- 19:15:30 - **Service started (UNAUTHORIZED)**

**Evidence**:
```
Nov 02 19:15:30 server systemd[1]: Starting Cloudflare Tunnel (claude-daemon)...
Nov 02 19:15:31 server cloudflared[67890]: 2025-11-02T19:15:31Z INF Starting tunnel
```

## Threat Assessment

**CRITICAL SECURITY INCIDENT**

**Possible Causes**:
1. Human manually restarted service (verify with human)
2. Automated script or cron job restarted service
3. Attacker gained access and restarted service (BREACH)

**Risk**: If no authentication implemented, daemon state is now publicly exposed again (P0 incident repeat)

## Recommended Actions

### IMMEDIATE (Within 5 Minutes)

1. **Stop service**:
   ```bash
   sudo systemctl stop cloudflared-tunnel.service
   ```

2. **Verify public URL inaccessible**:
   ```bash
   curl --max-time 5 https://your-spoke-tank-initiative.trycloudflare.com
   ```

3. **Check who started service**:
   ```bash
   journalctl -u cloudflared-tunnel.service --since "10 minutes ago"
   who
   last
   ```

### INVESTIGATION (Within 1 Hour)

4. **Review auth logs** for unauthorized SSH access:
   ```bash
   sudo grep "Accepted" /var/log/auth.log | tail -20
   ```

5. **Check for scheduled tasks**:
   ```bash
   crontab -l
   sudo crontab -l
   systemctl list-timers
   ```

6. **Review systemd service status**:
   ```bash
   sudo systemctl status cloudflared-tunnel.service
   systemctl is-enabled cloudflared-tunnel.service
   ```

### COMMUNICATION

7. **Contact human** to verify if they restarted service intentionally

8. **If unauthorized**: Escalate to P0 incident, apply incident response runbook

## Context

**Previous Incident**: Web dashboard public exposure (P0-INC-2025-11-02-001)
**Mitigation**: Services stopped and disabled
**This Alert**: Indicates mitigation may have been reversed

**If service was restarted by attacker**: This is a CRITICAL BREACH

---

**Automated alert from service-state-monitor.sh**
**Alert ID**: alert-20251102-191530-002
**Follow-up Required**: IMMEDIATE
```

---

## Document Metadata

**Author**: Auditor persona
**Date**: 2025-11-02
**Version**: 1.0 (Design Phase)
**Status**: READY TO IMPLEMENT (awaiting human decision)

**Dependencies**:
- Human decision on web dashboard (msg-auditor-incident-resolved-20251102)
- If web dashboard kept: Authentication must be implemented first

**Next Review**: After deployment, or when human decision received

**Change Log**:
- 2025-11-02: Initial design document created
- Awaiting deployment decision based on human feedback

---

**END OF SECURITY MONITORING DESIGN DOCUMENT**
