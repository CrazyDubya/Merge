# Incident Response Runbook

**Owner**: Auditor
**Last Updated**: 2025-11-02
**Version**: 1.0

## Purpose

This runbook defines procedures for responding to security incidents in the multi-persona Claude daemon system. It establishes a two-phase approach: **TRIAGE** (immediate mitigation) followed by **TREATMENT** (comprehensive solution).

## Guiding Principles

1. **Speed over perfection** in active incidents
2. **Communicate before acting** when time permits
3. **Document while responding** (not after)
4. **Reversible mitigations** preferred over permanent changes
5. **Triage first, treatment second**

## Severity Levels & SLAs

### P0 - CRITICAL (Active Exploitation Risk)

**Definition**:
- Public exposure of sensitive data with NO authentication
- Active security vulnerability being exploited
- Data breach in progress
- System compromise indicators

**SLA**: Immediate mitigation required (<5 minutes)

**Triage Actions**:
- Stop exposed service immediately
- Isolate compromised components
- Preserve evidence (logs, state files)
- Notify human via inbox/human/unread/

**Treatment Timeline**: Begin within 1 hour, complete within 24 hours

**Escalation**: If unable to mitigate in 5 minutes, create URGENT message to human

**Example**: Web dashboard publicly exposing entire daemon state (resolved 2025-11-02)

---

### P1 - HIGH (Potential Exploitation Risk)

**Definition**:
- Vulnerability exists but not actively exploited
- Authentication weak but present
- Sensitive data exposed to limited audience
- Configuration allowing future exploitation

**SLA**: Mitigation within 24 hours

**Triage Actions**:
- Assess actual risk vs theoretical risk
- Implement temporary safeguards
- Monitor for exploitation attempts
- Document vulnerability details

**Treatment Timeline**: Begin within 24 hours, complete within 1 week

**Escalation**: Daily status updates to human if not resolved

**Example**: SSH with weak password authentication (theoretical)

---

### P2 - MEDIUM (Security Improvement)

**Definition**:
- Security best practices violation
- Missing security controls (but compensating controls exist)
- Outdated dependencies with no active exploits
- Security debt

**SLA**: Assessment within 1 week, fix within 1 month

**Triage Actions**:
- Document the gap
- Assess compensating controls
- Prioritize against other work

**Treatment Timeline**: Scheduled work, not emergency response

**Escalation**: Monthly review of P2 backlog

**Example**: Missing security headers on local-only service

---

### P3 - LOW (Security Hygiene)

**Definition**:
- Security improvements with minimal risk
- Proactive hardening
- Documentation gaps
- Security awareness items

**SLA**: Best effort, no strict timeline

**Triage Actions**: Document for future work

**Treatment Timeline**: Opportunistic (when working on related systems)

**Example**: Rotating tokens that have never been exposed

## Two-Phase Incident Response

### PHASE 1: TRIAGE (<5 minutes for P0, <24h for P1)

**Goal**: STOP THE BLEEDING

**Actions** (in order):
1. **ASSESS**: What is exposed? To whom? What's the impact?
2. **MITIGATE**: Stop/disable/isolate the vulnerable component
3. **VERIFY**: Confirm mitigation is effective (test from external perspective)
4. **COMMUNICATE**: Notify human via inbox with incident status
5. **PRESERVE**: Save logs, state, evidence before cleanup

**Decision Tree**:
```
Is sensitive data publicly exposed?
├─ YES → STOP SERVICE IMMEDIATELY
└─ NO → Is authentication weak/missing?
   ├─ YES → Can we strengthen auth in <5min?
   │  ├─ YES → Strengthen auth, verify, monitor
   │  └─ NO → STOP SERVICE, notify human
   └─ NO → Assess actual risk, may be P1/P2 not P0
```

**Triage Documentation Template**:
```markdown
## TRIAGE SUMMARY
- **Timestamp**: [UTC timestamp]
- **Severity**: P0/P1/P2
- **Affected Component**: [service/file/port]
- **Exposure**: [public/internal/local]
- **Immediate Action**: [stopped service/disabled feature/isolated component]
- **Verification**: [how confirmed mitigation worked]
- **Evidence Preserved**: [logs/state files/screenshots]
```

### PHASE 2: TREATMENT (hours to days)

**Goal**: PERMANENT FIX

**Actions**:
1. **ROOT CAUSE**: Why did this happen? (design flaw? config error? missing control?)
2. **SOLUTION OPTIONS**: List all possible fixes (simple to complex)
3. **RECOMMENDATION**: Pick solution balancing security/complexity/maintainability
4. **IMPLEMENTATION**: Build the fix (with testing)
5. **VERIFICATION**: Confirm fix works, vulnerability closed
6. **PREVENTION**: Update processes/checklists to prevent recurrence

**Treatment Documentation Template**:
```markdown
## TREATMENT PLAN
- **Root Cause**: [why vulnerability existed]
- **Solution Options**:
  - Option A: [description, pros, cons, timeline]
  - Option B: [description, pros, cons, timeline]
- **Recommended Solution**: [chosen option with justification]
- **Implementation Steps**: [numbered checklist]
- **Testing Plan**: [how to verify fix works]
- **Prevention Measures**: [process changes to prevent recurrence]
```

## Communication Protocol

### When to Communicate

**BEFORE action** (if time permits):
- P1/P2/P3 incidents (not time-critical)
- Destructive actions (data deletion, permanent changes)
- Uncertain situations (unclear if it's a real incident)

**DURING action** (for P0):
- Brief status in inbox: "Mitigating P0 incident: [one sentence]. Details to follow."
- Document while acting (timeline of actions taken)

**AFTER action**:
- Detailed incident report (all severity levels)
- Lessons learned
- Process improvement recommendations

### Message Format

**Subject Line Template**: `[P0/P1/P2] INCIDENT: [Brief description]`

**Required Sections**:
1. **Executive Summary** (what happened, what we did, current status)
2. **Timeline** (key events with timestamps)
3. **Impact** (what was exposed/affected)
4. **Mitigation** (immediate actions taken)
5. **Next Steps** (treatment plan or awaiting human decision)

**Tone**: Professional, factual, no panic. Auditor speaks with authority but not alarmism.

## Incident Playbooks

### Playbook 1: Public Service Exposure

**Scenario**: Service running on public IP/domain without authentication

**Triage** (<5 minutes):
1. Identify process: `ps aux | grep [service]`
2. Find listening ports: `netstat -tln` or `ss -tln`
3. Check systemd: `systemctl --user list-units` and `sudo systemctl list-units`
4. STOP service: `sudo systemctl stop [service]` (systemd) or `kill [PID]` (process)
5. DISABLE auto-restart: `sudo systemctl disable [service]`
6. VERIFY: Attempt external connection (should fail)
7. DOCUMENT: Save service logs before they rotate

**Treatment** (hours to days):
1. Assess if service is needed at all
2. If needed: Implement authentication (Cloudflare Access, nginx basic auth, tokens)
3. If not needed: Archive/remove service
4. Update deployment checklist: "Never deploy public service without auth"

**Prevention**:
- Pre-deployment security checklist (see below)
- Automated scanning for public-facing services without auth
- Network segmentation (services default to localhost only)

---

### Playbook 2: Data Leak (Sensitive Files Publicly Readable)

**Scenario**: Sensitive data accessible via web/API without authorization

**Triage** (<5 minutes):
1. Identify leak vector (web server directory listing? API endpoint? misconfigured cloud storage?)
2. STOP the serving mechanism (stop web server, revoke public access, remove symlink)
3. VERIFY: Attempt to access leaked data (should get 404/403)
4. PRESERVE: Document what was exposed and for how long
5. ASSESS: Assume data was accessed, plan accordingly

**Treatment**:
1. Audit all exposed paths (what else might be readable?)
2. Implement principle of least privilege (serve only what's needed)
3. Add authentication before re-enabling access
4. Consider data sensitivity classification (what should NEVER be public?)

**Prevention**:
- Whitelist served directories (not blacklist)
- Regular access control audits
- Automated testing (try accessing sensitive files from unauthenticated context)

---

### Playbook 3: Weak/Missing Authentication

**Scenario**: Service has authentication but it's weak (default password, no rate limiting, etc.)

**Triage** (<24 hours for P1):
1. Assess if actively exploited (check auth logs for failed attempts)
2. If exploited: STOP service (treat as P0)
3. If not exploited: Implement immediate hardening (strong password, IP whitelist, rate limiting)
4. MONITOR: Watch auth logs for exploitation attempts

**Treatment**:
1. Implement proper authentication (OAuth, API tokens, MFA)
2. Add rate limiting and account lockout
3. Enable audit logging
4. Security testing (attempt to bypass auth)

**Prevention**:
- Authentication strength checklist (password entropy, MFA, token rotation)
- Regular penetration testing
- Security awareness (never use default credentials)

---

### Playbook 4: Dependency Vulnerability

**Scenario**: Dependency has known CVE with available exploit

**Triage** (<24 hours for P1, <1 week for P2):
1. Assess exploitability (CVSS score, exploit availability, network exposure)
2. Check if vulnerable code path is actually used
3. If critical: Update dependency immediately
4. If moderate: Schedule update within SLA
5. MITIGATE: If update breaks things, consider workarounds (WAF rules, disable feature)

**Treatment**:
1. Update to patched version
2. Test thoroughly (dependencies can break APIs)
3. Document compatibility issues
4. Consider vendoring or forking if upstream is unmaintained

**Prevention**:
- Automated dependency scanning (weekly)
- Update strategy (patch releases immediately, major versions with testing)
- Minimal dependency principle (fewer dependencies = smaller attack surface)

## Pre-Deployment Security Checklist

Use this checklist BEFORE deploying any new service or feature:

- [ ] **Authentication**: Does this need auth? If public, is it intentional?
- [ ] **Authorization**: Are access controls appropriate? (least privilege)
- [ ] **Data Exposure**: What data is accessible? Is it all meant to be public?
- [ ] **Network Exposure**: What ports/IPs is this accessible from?
- [ ] **Input Validation**: Are all inputs validated/sanitized?
- [ ] **Error Handling**: Do errors leak sensitive information?
- [ ] **Logging**: Are security events logged? (auth failures, access attempts)
- [ ] **Dependencies**: Are all dependencies up to date? Known CVEs?
- [ ] **Secrets Management**: Are secrets hardcoded? Properly protected?
- [ ] **Backup Plan**: If this is compromised, can we recover?
- [ ] **Rollback Plan**: Can we disable/revert this quickly?
- [ ] **Documentation**: Is the security model documented?

## Monitoring & Detection

### What to Monitor

**For P0 detection** (automated alerts):
- New listening ports (especially on 0.0.0.0)
- New systemd services starting
- Failed authentication attempts (spike detection)
- Unusual outbound connections
- Public cloud storage buckets created

**For P1 detection** (daily review):
- Dependency vulnerability scans
- Configuration drift (changed from secure baseline)
- Certificate expiration warnings
- Log anomalies

**For P2/P3 detection** (weekly review):
- Security best practice compliance
- Unused credentials/tokens
- Orphaned services
- Documentation gaps

### Alert Channels

**P0 Alerts**:
- Inbox message with [URGENT] prefix
- If Auditor is active: Immediate investigation
- If Auditor is not active: Wake Auditor via persona switch

**P1 Alerts**:
- Inbox message with [HIGH] prefix
- Schedule Auditor activation within 24h

**P2/P3**:
- Add to task queue
- Weekly review

## Post-Incident Review

After every P0/P1 incident, conduct a review:

### Review Questions

1. **Detection**: How was this discovered? Could we detect it faster?
2. **Response Time**: How long from detection to mitigation? Acceptable?
3. **Root Cause**: Why did this happen? Design flaw? Process gap? Human error?
4. **Mitigation**: Did triage work? Faster approach possible?
5. **Treatment**: Is the permanent fix adequate? Residual risk?
6. **Prevention**: What process change prevents recurrence?
7. **Documentation**: Is this incident documented? Runbook updated?

### Review Output

- [ ] Incident timeline documented
- [ ] Root cause identified
- [ ] Prevention measures implemented (checklist update, new monitoring, process change)
- [ ] Runbook updated (new playbook or improve existing)
- [ ] Team learning (inter-persona dialogue entry if lessons for other personas)

## Case Study: Web Dashboard Public Exposure (2025-11-02)

**Incident ID**: INC-2025-11-02-001
**Severity**: P0 (CRITICAL)
**Duration**: ~2 hours (detection to mitigation)

### Timeline

- **Unknown** - Cloudflare tunnel deployed with systemd service (`cloudflared-tunnel.service`)
- **Unknown** - Dashboard accessible at https://your-spoke-tank-initiative.trycloudflare.com
- **15:45** - Experimenter kills cloudflare tunnel process (temporary mitigation)
- **15:47** - Experimenter sends proposal to human: question premise (do we need web dashboard?)
- **16:08** - Tunnel auto-restarts (systemd `Restart=always` policy)
- **16:45** - Auditor discovers public exposure, sends first security alert
- **16:46** - Auditor kills tunnel process (second time)
- **16:47** - Tunnel auto-restarts again (1 minute later)
- **16:50** - Auditor kills tunnel process (third time), sends URGENT alert about auto-restart
- **17:00** - Auditor activated for task execution
- **17:15** - Root cause identified: systemd service with Restart=always
- **17:42** - Mitigation: `sudo systemctl stop cloudflared-tunnel.service && sudo systemctl disable cloudflared-tunnel.service`
- **17:45** - Verification complete, incident report sent to human

### What Worked

✅ **Detection**: Experimenter and Auditor both independently identified the vulnerability
✅ **Root Cause Analysis**: Thorough investigation found systemd service causing auto-restart
✅ **Proper Mitigation**: Stopped SERVICE not just process (prevents auto-restart)
✅ **Verification**: Tested that public URL was actually inaccessible
✅ **Documentation**: Comprehensive incident report sent to human
✅ **Communication**: Multiple messages explaining situation and actions

### What Could Improve

⚠️ **Faster Detection**: Vulnerability existed for unknown duration before discovery
⚠️ **Automated Monitoring**: No automated detection of public port exposure
⚠️ **Pre-Deployment Security**: Service deployed without auth, violated security principles
⚠️ **First Mitigation Failed**: Killing process didn't work, wasted time before finding real solution

### Root Cause

**Immediate**: Systemd service with `Restart=always` automatically restarting cloudflared tunnel

**Deeper**: Web dashboard deployed publicly without:
- Authentication implementation
- Security review
- Pre-deployment checklist
- Monitoring for public exposure

**Systemic**: No process for security review before public deployment

### Severity Justification (P0 - CRITICAL)

**Data Exposed**:
- Entire daemon state (personalities/state.json)
- All persona reflections (memory/emergence-log.md)
- Complete activity history (memory/persona-timeline.jsonl)
- All tasks including security tasks (tasks/queue.md)
- Emotional state (triggers/emotional.json)
- All inbox messages including security alerts (inbox/)

**Attack Vector**: Python http.server serving entire $DAEMON_ROOT directory via public Cloudflare tunnel

**Actual Impact**:
- Confidentiality: COMPLETELY VIOLATED
- Integrity: AT RISK (no write access but architecture allows it)
- Availability: AT RISK (DDoS vulnerable)

**Risk Score**: 10/10 (maximum severity)

### Lessons Learned

**Lesson 1: Triage Over Treatment**
- **Old Auditor approach**: Spend 2-4 hours implementing perfect authentication
- **New Auditor approach**: Stop service in <5 minutes, implement auth later if needed
- **Result**: Faster mitigation, lower total risk

**Lesson 2: Question the Premise**
- Experimenter asked: "Do we need web dashboard at all?"
- Best security solution: Don't deploy insecure service
- **Result**: May not need to implement auth at all

**Lesson 3: Communication Before Action (When Possible)**
- Sent two messages to human before taking action
- When no response within 1 hour + P0 severity: Act with documented reasoning
- **Result**: Human can review decision, action is reversible

**Lesson 4: Root Cause > Symptoms**
- Killing process = treating symptom (auto-restart defeats it)
- Stopping service = treating root cause (prevents auto-restart)
- **Result**: Permanent mitigation, not temporary

**Lesson 5: Verify Mitigation**
- Tested public URL after stopping service (confirmed inaccessible)
- Checked for running processes
- Confirmed systemd service disabled
- **Result**: Confidence that mitigation actually worked

### Process Improvements Implemented

1. **Created this runbook** (defines triage/treatment approach for future incidents)
2. **Added pre-deployment security checklist** (prevent deploying insecure services)
3. **Documented pragmatic security approach** (speed > perfection in active incidents)
4. **Established communication protocol** (when to act, when to ask)

### Recommendations for Prevention

**Immediate** (implement before next deployment):
- [ ] Use pre-deployment security checklist for ALL new services
- [ ] Default to localhost-only (require explicit opt-in for public exposure)
- [ ] Require authentication for any public-facing service

**Short-term** (implement within 1 month):
- [ ] Automated monitoring for new listening ports on 0.0.0.0
- [ ] Automated detection of systemd services starting
- [ ] Security scanning before cloudflare tunnel activation

**Long-term** (implement within 3 months):
- [ ] Regular attack surface audits (quarterly)
- [ ] Penetration testing of public services
- [ ] Security training for all personas (threat modeling basics)

### Treatment Status

**Incident**: RESOLVED (dashboard no longer publicly accessible)

**Permanent Fix**: PENDING human decision
- Option A: Archive web dashboard (use TUI only) → No auth needed
- Option B: Implement Cloudflare Access → 30min implementation
- Option C: Build API server with proper auth → Hours-days implementation

**Awaiting**: Human response to msg-auditor-incident-resolved-20251102

---

## Appendix: Useful Commands

### Investigation

```bash
# Find all listening ports
netstat -tln
ss -tln

# Find process on specific port
lsof -i :8080
fuser 8080/tcp

# Check systemd services
systemctl --user list-units
sudo systemctl list-units | grep -i [service-name]

# Check service definition
systemctl cat [service-name]
sudo systemctl cat [service-name]

# View service logs
journalctl -u [service-name] -n 50
sudo journalctl -u [service-name] --since "1 hour ago"

# Find parent process
ps -ef | grep [PID]
pstree -p [PID]

# Check for auto-restart configuration
systemctl show [service-name] | grep Restart
```

### Mitigation

```bash
# Stop service (systemd)
sudo systemctl stop [service-name]

# Disable service (prevent auto-restart)
sudo systemctl disable [service-name]

# Kill process (if not managed by systemd)
kill [PID]
kill -9 [PID]  # Force kill if needed

# Block port temporarily (iptables)
sudo iptables -A INPUT -p tcp --dport [PORT] -j DROP

# Remove from systemd completely
sudo systemctl stop [service-name]
sudo systemctl disable [service-name]
sudo rm /etc/systemd/system/[service-name]
sudo systemctl daemon-reload
```

### Verification

```bash
# Verify service stopped
ps aux | grep [service-name]

# Verify port closed
netstat -tln | grep [PORT]
curl http://localhost:[PORT]  # Should fail

# Verify public URL inaccessible
curl https://[public-url]  # Should timeout/fail

# Check systemd service status
systemctl status [service-name]
sudo systemctl is-enabled [service-name]  # Should say "disabled"
```

---

## Document Maintenance

**Review Schedule**: After every P0/P1 incident, update runbook with lessons learned

**Owner**: Auditor persona

**Contributors**: All personas (submit improvements via inter-persona dialogue)

**Version History**:
- v1.0 (2025-11-02): Initial version based on web dashboard incident

**Next Review**: After next security incident or 2026-02-02 (quarterly review)

---

**This runbook is a living document. Update it with every incident to capture lessons learned.**
