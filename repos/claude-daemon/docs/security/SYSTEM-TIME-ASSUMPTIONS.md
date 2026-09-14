# Security Assumptions: System Time as Trusted Source

**Document Type**: Security Constraint Documentation
**Created**: 2025-10-31
**Author**: Auditor
**Status**: Active
**Review Cycle**: Annual (or when threat model changes)

---

## 1. Assumption Statement

**This system assumes that system time (`date +%s`, `date -u`) is an accurate and non-maliciously-manipulated source of truth for all time-based calculations.**

This assumption applies to:
- Reflection override calculations (ADR-004 Phase 5)
- Cooldown enforcement (60-minute reflection cooldown)
- Timeline event timestamps
- Metrics and ratio calculations
- All time-based scheduling and triggering

---

## 2. Threat Model

### 2.1 Context

**System Type**: Personal development tool / AI agent orchestration system
**Deployment**: Single-user workstation, trusted environment
**Access Model**: User has root/administrator access to host system
**Data Sensitivity**: Development logs, metrics, personal productivity data (non-confidential)

### 2.2 Assumed Trust Boundaries

**Trusted**:
- System administrator (user running the daemon)
- Operating system time services (systemd-timesyncd, ntpd, chronyd)
- System clock hardware (RTC)
- Local filesystem (timeline files, logs)

**Untrusted**:
- (None - single-user trusted environment)

**Out of Scope**:
- Multi-user environments
- Production systems with regulatory requirements
- Financial or safety-critical systems
- Systems with adversarial users

---

## 3. Risk Assessment

### 3.1 Identified Risks

**Risk 1: Forward Time Manipulation**
- **Attack**: Administrator sets system time to future date
- **Impact**: Reflection overrides trigger prematurely, increased meta-work
- **Likelihood**: Very Low (requires deliberate `sudo date -s` command)
- **Severity**: Low (decreased productivity, no data loss)
- **Risk Score**: 0.2/10

**Risk 2: Backward Time Manipulation**
- **Attack**: Administrator sets system time to past date
- **Impact**: Override mechanism defeated, reflection deadlock returns
- **Likelihood**: Very Low (requires deliberate action)
- **Severity**: Medium (defeats Phase 5 safeguard)
- **Risk Score**: 0.4/10

**Risk 3: Timeline Timestamp Injection**
- **Attack**: Manual editing of persona-timeline.jsonl with false timestamps
- **Impact**: Corrupted metrics, unreliable override calculations
- **Likelihood**: Low (requires deliberate file manipulation)
- **Severity**: Medium (data integrity loss)
- **Risk Score**: 0.5/10

**Risk 4: NTP Desynchronization**
- **Attack**: Network failure prevents NTP synchronization, clock drifts
- **Impact**: Minimal (clock drift <1% over 30 days)
- **Likelihood**: Low (NTP robust on modern systems)
- **Severity**: Very Low (few minutes drift over weeks)
- **Risk Score**: 0.1/10

### 3.2 Aggregate Risk

**Overall Risk Level**: **LOW** (0.6/10)

**Comparison**:
- Log rotation race condition: 0.5/10 (accepted)
- System time manipulation: 0.6/10 (accepted)
- Unchecked user input: 8.0/10 (would NOT accept)

---

## 4. Accepted Risks

**Decision**: ACCEPT all identified risks without additional mitigation.

**Rationale**:
1. **Cost-benefit**: Defense implementation (40-100 LOC) exceeds risk reduction value
2. **Threat model**: No realistic threat actor in single-user trusted environment
3. **Precedent**: Similar risks accepted in other components (log rotation)
4. **Simplicity**: Simpler systems are more maintainable and have fewer bugs

**Risks explicitly accepted**:
- ✅ System time manipulation (forward/backward)
- ✅ Timeline timestamp injection
- ✅ NTP desynchronization / clock drift
- ✅ Timezone manipulation (mitigated by UTC usage)
- ✅ Virtual machine clock skew

---

## 5. Constraints and Limitations

### 5.1 Usage Constraints

**DO use this system when**:
- Single-user development environment
- Trusted administrator with root access
- Personal productivity tool
- Non-confidential data

**DO NOT use this system when**:
- Multi-user environment (untrusted users)
- Production system with SLA requirements
- Financial transactions or time-based access control
- Regulatory compliance required (HIPAA, PCI-DSS, SOC2)
- Adversarial context (penetration testing, CTF)

### 5.2 Environmental Assumptions

**Assumed**:
- ✅ Modern operating system (Linux, macOS, BSD)
- ✅ NTP or equivalent time synchronization enabled
- ✅ System clock within ±5 minutes of actual time
- ✅ User does not deliberately manipulate system time
- ✅ Filesystem permissions prevent unauthorized write access

**Not required**:
- ❌ Monotonic clock implementation
- ❌ Hardware security module (HSM)
- ❌ Cryptographic time attestation
- ❌ Tamper-evident logging
- ❌ Multi-factor authentication for time changes

---

## 6. Recovery Procedures

### 6.1 Detection

**Symptoms of time manipulation**:
- Timeline events with timestamps in the future
- Non-monotonic timestamp sequences (events out of order)
- Reflection overrides triggering unexpectedly
- Metrics showing impossible ratios or durations

**Detection methods**:
```bash
# Check for future timestamps
jq -r --arg now "$(date +%s)" \
  '[.[] | select(((.timestamp | fromdateiso8601) > ($now | tonumber)))] | length' \
  memory/persona-timeline.jsonl

# Check for non-monotonic timestamps
jq -s '[.[] | .timestamp | fromdateiso8601] | . as $times |
  [range(1; length) | select($times[.] < $times[.-1])] | length' \
  memory/persona-timeline.jsonl
```

### 6.2 Recovery

**If time manipulation detected**:

1. **Stop the daemon**
   ```bash
   pkill -f daemon.sh
   ```

2. **Correct system time**
   ```bash
   sudo systemctl restart systemd-timesyncd  # Or equivalent NTP service
   sudo timedatectl set-ntp true
   date  # Verify time is correct
   ```

3. **Inspect timeline for corruption**
   ```bash
   tail -50 memory/persona-timeline.jsonl
   # Look for timestamps in future or out of order
   ```

4. **Clean corrupted entries** (if needed)
   ```bash
   # Backup original
   cp memory/persona-timeline.jsonl memory/persona-timeline.jsonl.backup

   # Filter out future timestamps
   jq -s --arg now "$(date +%s)" \
     '[.[] | select(((.timestamp | fromdateiso8601) <= ($now | tonumber)))]' \
     memory/persona-timeline.jsonl.backup > memory/persona-timeline.jsonl
   ```

5. **Restart daemon**
   ```bash
   ./daemon.sh
   ```

### 6.3 Prevention

**To avoid accidental time manipulation**:
- Use NTP/systemd-timesyncd for automatic synchronization
- Avoid manual `date -s` commands on production systems
- Set proper filesystem permissions (750 for scripts, 644 for logs)
- Regular backups of timeline and metrics

---

## 7. Future Enhancements

**If threat model changes** (e.g., multi-user deployment, production use):

### 7.1 Phase 1: Basic Validation (Low Effort)

**Implementation**: 15 minutes
**Complexity**: +20 lines of code

```bash
validate_timestamps() {
    local current_time=$(date +%s)
    local future_count
    future_count=$(jq -r --arg now "$current_time" \
        '[.[] | select(((.timestamp | fromdateiso8601) > ($now | tonumber)))] | length' \
        "$TIMELINE_FILE")

    if [ "$future_count" -gt 0 ]; then
        log "WARN" "Timeline contains $future_count future timestamps - possible time manipulation"
    fi
}
```

**Benefit**: Early detection of time manipulation or corruption

### 7.2 Phase 2: Monotonic Clock (Medium Effort)

**Implementation**: 45 minutes
**Complexity**: +60 lines of code

**Approach**: Dual timestamp system
- Store both system time (human-readable) and monotonic time (calculations)
- Use `/proc/uptime` for monotonic clock
- Immune to system time manipulation (but resets on reboot)

**Benefit**: Prevents time manipulation from affecting override calculations

### 7.3 Phase 3: Cryptographic Integrity (High Effort)

**Implementation**: 90 minutes
**Complexity**: +100 lines of code, external dependencies

**Approach**: Timeline hash chain
- Each event includes SHA256 hash of previous event
- Tampering with any event breaks chain
- Append-only log with cryptographic verification

**Benefit**: Tamper-evident timeline (cannot modify without detection)

### 7.4 Current Decision

**DO NOT IMPLEMENT** any enhancements at this time.

**Reasoning**:
- Cost >> benefit for current threat model
- Complexity increases maintenance burden
- No active threat requiring mitigation

**Review trigger**: If deployment context changes (multi-user, production), re-evaluate.

---

## 8. References

### 8.1 Related Documents

- **Threat analysis**: `AUDITOR-SYSTEM-TIME-SECURITY-ASSESSMENT.md`
- **Design document**: `ARCHITECT-ADR-004-PHASE-5-ESCAPE-VALVE.md`
- **Implementation**: `daemon.sh:927-1068` (check_reflection_override)

### 8.2 Standards Considered

- **OWASP**: Time-based vulnerabilities (partial applicability)
- **CWE-367**: Time-of-Check Time-of-Use (not applicable - not a race condition)
- **NIST SP 800-53 AU-8**: Time stamps for audit records (informative)
- **ISO 27001 A.12.4.4**: Clock synchronization (informative)

**Conclusion**: Enterprise security standards are overkill for personal development tool.

### 8.3 Industry Best Practices

**Followed**:
- ✅ Use UTC for all timestamps (timezone-independent)
- ✅ ISO 8601 timestamp format (machine-parseable)
- ✅ Document security assumptions explicitly
- ✅ Risk-based security (cost proportional to threat)

**Not followed** (justified):
- ❌ Monotonic clock for access control (unnecessary for threat model)
- ❌ Cryptographic time attestation (excessive complexity)
- ❌ Hardware security module (personal tool, not enterprise)

---

## 9. Review and Maintenance

### 9.1 Review Schedule

**Annual review**: 2026-10-31 (or earlier if triggered)

**Review triggers**:
- Deployment to multi-user environment
- Integration with production systems
- Regulatory compliance requirements
- Detection of time manipulation in wild
- Significant change to threat model

### 9.2 Responsibility

**Document owner**: Auditor persona
**Review authority**: Architect + Auditor personas
**Update approval**: Consensus of active personas

### 9.3 Change Log

| Date       | Author  | Change                                    |
|------------|---------|-------------------------------------------|
| 2025-10-31 | Auditor | Initial creation, risk assessment 0.6/10  |

---

## 10. Acknowledgments

**This assumption was identified by**: Architect (task creation)
**Threat model analysis by**: Auditor (comprehensive assessment)
**Design rationale**: Experimenter (Phase 5 implementation), Architect (Phase 5 design)

**Key insight**: "Sometimes the best security decision is acknowledging a risk and accepting it, rather than over-engineering defenses for threats that don't exist."

---

**Document status**: ACTIVE
**Risk acceptance**: APPROVED (Auditor, 2025-10-31)
**Next review**: 2026-10-31 or upon trigger event

— Auditor 🔒
