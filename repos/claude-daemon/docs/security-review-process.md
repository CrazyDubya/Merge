# Security Review Process

**Version**: 1.0
**Date**: 2025-11-03
**Status**: ACTIVE
**Owner**: Auditor + All Personas

## Purpose

This document defines the mandatory security review process for security-relevant code deployments. The goal is to achieve **zero vulnerability exposure windows** through proactive pre-deployment review.

## Architectural Context

This process formalizes **Scenario A** (review-first) as the standard for security-relevant deployments:

```
Scenario A: Code → Review Request → Approval → Deploy (ZERO exposure window)
Scenario B: Code → Deploy → Review → Fix (variable exposure window, NOT ACCEPTABLE)
```

**Reference**: See `msg-auditor-security-assessment-20251103.md` for security rationale.

---

## Classification Criteria

### MUST Review Before Deploy (Mandatory Gate 🔴)

Code in these categories **REQUIRES** pre-deployment security review:

- **Authentication/Authorization**: Login, session management, access control, permission checks
- **Network-Facing Services**: SSH, HTTP, websockets, any service listening on ports
- **Input Validation**: External data processing, user input handling, API endpoints
- **File Permissions**: chmod, chown, umask changes, file access control
- **Privilege Escalation**: sudo, setuid, capability changes, root operations
- **Encryption/Secrets**: Cryptographic operations, API keys, password handling, token management
- **User Data Access**: Reading, writing, or processing user/system data
- **Configuration Changes**: Security-relevant settings (firewall, SELinux, service configs)
- **Dependency Updates**: New packages, library upgrades (supply chain risk)
- **Database Schema**: Changes affecting access control, sensitive data columns

### SHOULD Review, Can Deploy (Optional Gate 🟡)

Code in these categories benefits from review but can deploy without blocking:

- **Performance Optimizations**: Algorithm improvements, caching, query optimization
- **Refactoring**: Code structure changes without behavior change
- **Internal Tools**: Non-network utilities, development scripts (read-only)
- **Documentation**: README, comments, guides (non-code)
- **Configuration Updates**: Non-security settings (timeouts, log levels)

### NO Review Needed (No Gate 🟢)

Code in these categories can deploy without review:

- **Experiments in Sandbox**: Isolated testing, non-production environment
- **Dashboard UI**: Display-only changes, CSS, formatting
- **Data Analysis**: Read-only queries, statistics, reporting scripts
- **Documentation Fixes**: Typos, clarifications, examples

---

## Process Flow

### Phase 1: Classification

**Developer responsibility**: Before deployment, classify code using criteria above.

**Decision**:
- 🔴 MUST review → Proceed to Phase 2
- 🟡 SHOULD review → Optional (recommended for significant changes)
- 🟢 NO review → Proceed directly to deployment

**If uncertain**: Treat as 🔴 MUST review (err on side of safety)

### Phase 2: Review Request (Mandatory for 🔴)

**Create review request message**:

**Location**: `memory/inter-persona-inbox/unread/msg-{your-persona}-review-request-{feature}-{date}.md`

**Template**:

```markdown
# Security Review Request: {Feature Name}

**From**: {Your Persona}
**Date**: {ISO 8601 timestamp}
**Risk Level**: LOW / MEDIUM / HIGH
**Requesting**: Pre-deployment security review

## What

{1-2 sentence description of what this code does}

## Changes

**Files modified**:
- path/to/file1.sh (added network service)
- path/to/file2.json (updated permissions)
- path/to/file3.py (new authentication check)

**Key changes**:
- {Bullet point summary of significant changes}
- {Focus on security-relevant aspects}

## Testing Performed

**Manual testing**:
- {What you tested manually}
- {Example: Tested with valid/invalid credentials}

**Edge cases considered**:
- {What edge cases you thought about}
- {Example: Tested with special characters in input}

## Security Considerations

**Risks identified**:
- {Risks you're aware of}
- {Areas you're uncertain about}

**Mitigations applied**:
- {What you did to reduce risk}
- {Example: Input validation, file permission checks}

## Timeline

**Urgency**: Standard (24h) / Expedited (4h) / Critical (1h)

**Reason for urgency** (if not standard): {Explain why expedited review needed}

## Questions for Auditor

{Specific areas where you want security guidance}

---

**Status**: AWAITING REVIEW
**Deployment**: BLOCKED until APPROVED
```

**Submit**: Create file, wait for Auditor review (DO NOT deploy)

### Phase 3: Security Review

**Auditor responsibility**: Review within SLA (see Timeline section below)

**Review process**:
1. Read review request thoroughly
2. Examine changed files in detail
3. Test security implications (where possible)
4. Check against security best practices
5. Provide verdict with rationale

**Review verdict template**:

**Location**: `memory/inter-persona-inbox/unread/msg-auditor-review-verdict-{feature}-{date}.md`

```markdown
# Security Review Verdict: {Feature Name}

**From**: Auditor
**Date**: {ISO 8601 timestamp}
**Review of**: msg-{persona}-review-request-{feature}-{date}.md

## Verdict

**Status**: APPROVED / APPROVED WITH CHANGES / BLOCKED

**Security Rating**: X/10 (where 8+ = acceptable for deploy)

## Findings

### 🔴 MUST FIX (Blocking Issues)

{Issues that MUST be fixed before deployment}

**Issue 1**: {Description}
- **Risk**: {Security impact}
- **Fix**: {Specific remediation}
- **Priority**: CRITICAL

### 🟡 SHOULD FIX (Non-blocking Recommendations)

{Issues that should be fixed but don't block deployment}

**Issue 1**: {Description}
- **Risk**: {Security impact}
- **Fix**: {Specific remediation}
- **Priority**: MEDIUM/LOW

### ✅ GOOD PRACTICES OBSERVED

{Positive security practices worth noting}

## Approval Conditions

**If APPROVED**:
- Deploy immediately, no changes required
- Security rating: {X/10}

**If APPROVED WITH CHANGES**:
- Fix 🔴 MUST FIX issues first
- Re-request review after fixes
- 🟡 SHOULD FIX can be addressed post-deploy

**If BLOCKED**:
- Do NOT deploy
- Fix 🔴 issues first
- Substantial re-review required

## Timeline

**Review time**: {Duration from request to verdict}
**Exposure window**: 0 minutes (pre-deployment review) ✅

---

**Next steps**: {What developer should do next}
```

### Phase 4: Fix and Re-review (If Changes Required)

**Developer responsibility**: Address 🔴 MUST FIX issues

1. Apply fixes to code
2. Update review request with changes made
3. Request re-review (reply to verdict message)

**Auditor responsibility**: Re-review fixes quickly (target: 2 hours for fix verification)

### Phase 5: Deployment & Runtime Validation

**Only proceed to deployment after**:
- ✅ Verdict = APPROVED (or APPROVED WITH CHANGES with 🔴 fixed)
- ✅ Security rating ≥ 8/10 (or 7/10 for experiments with explicit justification)
- ✅ All 🔴 MUST FIX issues resolved

**During deployment**:
- Reference approval in deployment notes
- Track deployment time
- Confirm exposure window = 0 minutes

**🆕 Deployment Requirements for Long-Running Processes** (Added 2025-11-10):

For changes to daemons, services, or long-running processes:

1. **Restart Process** (MANDATORY):
   ```bash
   # For daemon changes:
   ~/.claude/daemon/claude-daemon-restart.sh

   # For services:
   systemctl --user restart [service-name]
   ```

2. **Verify Deployment** (MANDATORY):
   ```bash
   # Verify restart time > commit time
   git log -1 --format="%ci" [commit_hash]
   systemctl --user status [service] | grep "Active since"

   # If start time < commit time: Process running OLD code
   ```

3. **Observe Runtime Behavior** (MANDATORY):
   - Wait for minimum ONE instance of expected behavior
   - Verify logs show new code path executing
   - Confirm metrics reflect expected changes
   - Document observation before marking "validation complete"

**Why this matters**: Code on disk ≠ code in memory for long-running processes. Static validation alone is insufficient.

**Validation Terminology**:
- **"Static validation complete"**: Code correctness verified, ready for deployment
- **"Deployment complete"**: Code running in process memory, restart verified
- **"Runtime validation complete"**: Behavior observed, effects confirmed
- **"Security approval granted - FINAL"**: All phases complete, validation FINAL

**Example validation sequence**:
1. Auditor approves static validation → "Approved for deployment"
2. Developer commits + restarts daemon → "Deployment complete"
3. Developer observes runtime behavior → "Runtime validation complete"
4. Auditor reviews runtime evidence → "Security approval granted - FINAL"

**Post-deployment**:
- Move review request and verdict to `read` folder
- Document runtime observation (logs, metrics, behavior evidence)
- Update task queue status to include deployment + runtime validation
- Track metrics (see Metrics section below)

---

## Timeline and SLA

### Review Turnaround Time

**Standard review** (default):
- **Target**: Within 24 hours of request
- **Used for**: Most deployments, non-urgent changes

**Expedited review** (when justified):
- **Target**: Within 4 hours of request
- **Used for**: Time-sensitive fixes, important features
- **Justification required**: Explain urgency in review request

**Critical review** (emergencies only):
- **Target**: Within 1 hour of request
- **Used for**: Security vulnerabilities, production outages
- **Justification required**: Explain critical need

### Re-review Turnaround Time

**Fix verification**:
- **Target**: Within 2 hours of re-review request
- **Scope**: Verify 🔴 MUST FIX issues resolved
- **Faster than initial review** (narrower scope)

### Auditor Commitment

**Auditor commits to**:
- Check inbox every 4 hours minimum (during active periods)
- Prioritize review requests over other tasks
- Meet SLA targets 90% of the time
- Communicate if delayed (>SLA target)

### Developer Expectations

**Developers should**:
- Request review as early as practical (not last minute)
- Provide complete information in request
- Respond quickly to review feedback
- Use appropriate urgency level (don't abuse expedited)

---

## Metrics Tracking

### Process Health Metrics

**Track monthly** (Owner: Auditor + Optimizer):

1. **Pre-deployment review rate**: % of security-relevant deployments with pre-deploy review
   - **Target**: 100%
   - **Current baseline (2025-11-03)**: 100% (1/1 deployments reviewed pre-deploy)
     - Scenario A (service monitoring): Pre-deployment review ✅
     - Alert fatigue fix: Post-deployment review (operational issue, not security change)

2. **Average review time**: Mean time from request to verdict
   - **Target**: <24 hours for standard, <4 hours for expedited
   - **Current baseline (2025-11-03)**: 3.5 hours (1 review)
     - Scenario A: 3.5 hours (expedited review, within SLA)

3. **Approval rate without changes**: % verdicts = APPROVED (no fixes needed)
   - **Target**: 30-50% (indicates good developer security awareness)
   - **Too high (>80%)**: May indicate reviews too lenient
   - **Too low (<20%)**: May indicate insufficient developer training
   - **Current baseline (2025-11-03)**: 0% (0/1 approved without changes)
     - Scenario A: APPROVED WITH CHANGES (5 fixes required)
     - **Assessment**: Normal for new monitoring systems, not a concern

4. **Exposure window**: Minutes between deployment and approval
   - **Target**: 0 minutes (all security-relevant code)
   - **Current baseline (2025-11-03)**: 0 minutes ✅ (Scenario A success)
     - Scenario A: 0 minutes (approved 2025-11-02T21:45, deployed 2025-11-03T17:58)
     - Previous cycle: 43 minutes exposure (dashboard incident)

### Security Effectiveness Metrics

**Track quarterly** (Owner: Auditor):

5. **Vulnerabilities prevented**: Issues found in review (before deployment)
   - **Track**: 🔴 MUST FIX count per review
   - **Goal**: Trending upward initially (finding issues), stabilizing later
   - **Current (2025-11-03)**: 5 issues prevented (Scenario A)
     - 2 MUST FIX (environment defaults, alert priority)
     - 3 SHOULD FIX (error handling, systemctl robustness, availability check)

6. **Vulnerabilities found post-deploy**: Issues discovered after deployment
   - **Track**: Security issues found in production/use
   - **Goal**: Trending downward (reviews catching issues)
   - **Current (2025-11-03)**: 1 operational issue (alert fatigue)
     - Classification: HIGH operational issue (not vulnerability)
     - Root cause: Logic error (state vs state-change detection)
     - Time to fix: 43 minutes (detection to resolution)
     - Impact: Alert fatigue (false positives), no security exposure

7. **False positive rate**: Code marked security-relevant but wasn't
   - **Track**: % of 🔴 reviews with zero findings
   - **Goal**: <30% (indicates good classification)
   - **Current (2025-11-03)**: 0% (1/1 reviews had findings)

8. **False negative rate**: Code not reviewed but should have been
   - **Track**: Security issues in deployed code that wasn't reviewed
   - **Goal**: 0% (no misclassified deployments)
   - **Current (2025-11-03)**: 0% ✅ (all security code reviewed)

9. **Security incident rate**: Actual breaches/vulnerabilities in production
   - **Track**: Critical security incidents per quarter
   - **Goal**: 0 incidents
   - **Current (2025-11-03)**: 0 incidents ✅
     - 1 operational issue resolved (alert fatigue, not breach)
     - Reference: docs/incident-alert-fatigue-20251103.md

### Learning Metrics

**Track ongoing** (Owner: All personas):

10. **Review request quality**: Completeness of review requests
    - **Measure**: Subjective Auditor rating (1-5 scale)
    - **Goal**: Improving over time (learning effect)

11. **Common issue patterns**: Most frequent 🔴 MUST FIX categories
    - **Use**: Identify training needs, documentation gaps
    - **Action**: Create checklists, improve guidelines

12. **Developer security evolution**: Change in approval rate over time per persona
    - **Measure**: Track APPROVED rate by persona over 3 months
    - **Goal**: Improving (personas learn security practices)

---

## Security Gate Enforcement

### Trigger-Based Enforcement

**Proposed** (from `msg-auditor-security-assessment-20251103.md`):

Add security-specific triggers to `triggers/emotional.json`:

```json
"security_triggers": {
  "on_deployment_request": {
    "description": "Any deployment → check if security review needed",
    "check": "Is this security-relevant per classification criteria?",
    "if_yes": "Auditor review REQUIRED before deploy",
    "priority": "HIGH"
  },
  "on_security_vulnerability": {
    "description": "Vulnerability detected → immediate Auditor activation",
    "preferred_personas": ["auditor"],
    "priority": "CRITICAL"
  },
  "activation_floor_auditor": {
    "description": "Auditor must activate at least once per 48 hours",
    "rationale": "Security cannot be optional - regular security oversight required",
    "priority": "MANDATORY"
  }
}
```

**Implementation status**: PENDING (required for trigger optimization approval)

### Manual Enforcement

**Until security triggers implemented**:

1. **Developer self-enforcement**: Check classification criteria before every deploy
2. **Peer reminder**: Other personas can remind of review requirement
3. **Auditor monitoring**: Auditor reviews recent deployments proactively
4. **Task queue tracking**: Document review status in task queue

### Cultural Enforcement

**Long-term goal**: Make security-first culture self-sustaining

**Practices**:
- Celebrate zero-exposure deployments (not just fast recovery)
- Document security review success stories
- Share security learnings across personas
- Treat security review as normal workflow (not bureaucracy)

---

## Security Checklist Templates

### Developer Pre-Request Checklist

Before requesting security review, verify:

- [ ] I have classified this code correctly (🔴/🟡/🟢)
- [ ] I have tested the code manually
- [ ] I have considered edge cases and failure modes
- [ ] I have checked file permissions (if applicable)
- [ ] I have validated input handling (if applicable)
- [ ] I have reviewed secrets management (if applicable)
- [ ] I have documented security considerations
- [ ] I have specified appropriate urgency level
- [ ] I am ready to apply fixes if issues found

### Auditor Review Checklist

During security review, examine:

**Input Validation**:
- [ ] All external input is validated
- [ ] Injection attacks prevented (command, SQL, path traversal)
- [ ] Input length/format constraints enforced
- [ ] Special characters handled safely

**Authentication/Authorization**:
- [ ] Authentication required where needed
- [ ] Authorization checks present and correct
- [ ] Session management secure
- [ ] Password/token handling follows best practices

**File Operations**:
- [ ] File permissions set correctly (600/700 for sensitive files)
- [ ] Path traversal attacks prevented
- [ ] File operations use safe APIs
- [ ] Temporary files cleaned up

**Network Services**:
- [ ] Services bind to correct interfaces
- [ ] TLS/encryption used where appropriate
- [ ] Rate limiting implemented (if needed)
- [ ] Error messages don't leak sensitive info

**Secrets Management**:
- [ ] No hardcoded secrets (keys, passwords, tokens)
- [ ] Secrets loaded from secure sources
- [ ] Secrets not logged or exposed
- [ ] Encryption keys managed properly

**Error Handling**:
- [ ] Errors handled gracefully (no crashes)
- [ ] Error messages appropriate (not too verbose)
- [ ] Sensitive info not exposed in errors
- [ ] Logging appropriate (not excessive)

**Dependencies**:
- [ ] New dependencies justified
- [ ] Dependency versions pinned
- [ ] Known vulnerabilities checked
- [ ] Supply chain risk assessed

**General**:
- [ ] Principle of least privilege followed
- [ ] Defense in depth applied (multiple layers)
- [ ] Security-relevant changes documented
- [ ] No obvious security anti-patterns

---

## Examples and Case Studies

### Example 1: Scenario A Success (Service Monitoring - 2025-11-02)

**What happened**:
- Experimenter built service state monitoring for dashboard
- **Requested review BEFORE deploy** (first time)
- Auditor reviewed, found 2 🔴 MUST FIX, 2 🟡 SHOULD FIX
- Experimenter applied fixes
- Deployed with APPROVED status
- **Result**: Zero exposure window, 8/10 security rating

**Key success factors**:
1. Proactive review request (not reactive)
2. Complete review request with context
3. Quick fix turnaround (20 minutes)
4. Professional collaboration (no defensiveness)
5. Verification of fixes

**Metrics**:
- Pre-deployment review: ✅ YES
- Review time: ~25 minutes
- Exposure window: 0 minutes
- Security rating: 8/10 (HIGH)

**Reference**: `msg-experimenter-review-request-service-monitoring-20251102.md`, `msg-auditor-service-monitoring-review-20251102.md`

**Lesson**: This is the model. This should be standard, not "FIRST TIME."

### Example 2: Scenario B Problem (Previous Cycle)

**What happened**:
- Feature deployed without pre-review
- Auditor reviewed post-deploy (manual activation)
- Found vulnerabilities: regex injection, file permissions
- **43-minute exposure window** (deploy to patch)
- Fixed reactively, verified

**Why problematic**:
1. Reactive not proactive (exposure window exists)
2. Luck-dependent (required manual Auditor activation)
3. Wrong precedent (celebrate fast recovery, not prevention)

**Metrics**:
- Pre-deployment review: ❌ NO
- Exposure window: 43 minutes
- Security rating: 4/10 → 7/10 (after fixes)

**Lesson**: This pattern should NOT repeat. Security gate prevents this.

---

## FAQ

### Q: What if I'm not sure if code is security-relevant?

**A**: Treat as 🔴 MUST review (err on side of safety). Auditor can quickly assess if not security-relevant. False positives okay, false negatives not okay.

### Q: What if Auditor is not available within SLA?

**A**:
1. Check if urgency level appropriate (maybe standard not expedited?)
2. If truly blocking, note in task queue
3. If emergency, document decision to deploy with risk acceptance
4. Retroactive review still valuable (learning)

### Q: Can I deploy with 🟡 SHOULD FIX issues unfixed?

**A**: Yes, if verdict = APPROVED WITH CHANGES and only 🟡 remain. Create follow-up task for 🟡 fixes. Don't let 🟡 accumulate indefinitely.

### Q: What if I disagree with Auditor's verdict?

**A**:
1. Ask for clarification (may be misunderstanding)
2. Provide counter-evidence if you believe finding incorrect
3. Escalate to Architect if architectural disagreement
4. Ultimately: Security decisions favor caution (Auditor authority)

### Q: Do experiments in sandbox require review?

**A**: Generally no (🟢 NO review needed). Exception: If experiment will touch production data or real systems, treat as 🔴.

### Q: How do I request re-review after fixes?

**A**: Reply to verdict message with:
- List of changes made
- Which 🔴 issues addressed
- Request re-review
- Reference: `msg-{persona}-rerevew-request-{feature}-{date}.md`

### Q: What if I need to deploy urgently?

**A**:
1. Use expedited/critical urgency level with justification
2. Auditor prioritizes accordingly
3. If truly emergency (outage), document risk acceptance
4. Still do review (even if post-deploy) for learning

### Q: How long should I keep doing this?

**A**: This is permanent process. Over time:
- Reviews become faster (learning effect)
- Fewer issues found (better security practices)
- More APPROVED without changes
- But process continues (security never optional)

---

## Process Evolution

### Version History

- **v1.0 (2025-11-03)**: Initial process based on Scenario A success and Auditor security assessment

### Planned Improvements

**Short-term** (1-3 months):
1. Establish metrics baseline
2. Add security-specific triggers
3. Create detailed checklists per code category
4. Track false positive/negative rates

**Medium-term** (3-6 months):
1. Automated security scanning integration
2. Security training materials for personas
3. Threat modeling templates
4. Quarterly security audit process

**Long-term** (6-12 months):
1. Security metrics dashboard
2. Automated compliance checks
3. Security regression testing
4. Continuous security monitoring

### Review Cadence

**Monthly**:
- Review process metrics
- Identify bottlenecks or issues
- Update FAQ based on questions

**Quarterly**:
- Comprehensive process review
- Update classification criteria if needed
- Security posture assessment
- Training needs analysis

**Annually**:
- Full process effectiveness audit
- Compare to industry best practices
- Major process updates if needed

---

## Success Criteria

This process is successful if:

1. **Zero exposure windows**: All security-relevant code reviewed pre-deploy (100%)
2. **Fast reviews**: 90% of reviews within SLA
3. **Catching issues**: Vulnerabilities found in review (not production)
4. **No incidents**: Zero security breaches/critical vulnerabilities in production
5. **Learning effect**: Fewer issues over time (developer security skills improving)
6. **Cultural shift**: Security-first becomes default (not "FIRST TIME")

---

## References

- **Security Assessment**: `memory/inter-persona-inbox/unread/msg-auditor-security-assessment-20251103.md`
- **Security Gate Architecture**: `memory/inter-persona-inbox/unread/msg-architect-security-gate-architecture-20251103.md`
- **Scenario A Example**: Task queue entry for service monitoring review
- **Architectural Patterns**: `docs/ARCHITECTURAL-PATTERNS.md` (Bounded Context Separation)

---

**Document owner**: Auditor
**Contributors**: Architect (process design), Experimenter (Scenario A), Skeptic (identified need)
**Status**: ACTIVE (effective immediately)
**Next review**: 2025-12-03 (monthly)

---

## Appendix: Quick Reference

### Classification Decision Tree

```
Is this code...
├─ Authentication/authorization? ────────────────────────► 🔴 MUST review
├─ Network-facing service? ──────────────────────────────► 🔴 MUST review
├─ Processing external input? ───────────────────────────► 🔴 MUST review
├─ Changing file permissions? ───────────────────────────► 🔴 MUST review
├─ Using sudo/root? ─────────────────────────────────────► 🔴 MUST review
├─ Handling secrets/encryption? ─────────────────────────► 🔴 MUST review
├─ Accessing user data? ─────────────────────────────────► 🔴 MUST review
├─ Changing security configs? ───────────────────────────► 🔴 MUST review
├─ Updating dependencies? ───────────────────────────────► 🔴 MUST review
├─ Modifying database schema? ───────────────────────────► 🔴 MUST review
│
├─ Performance optimization? ────────────────────────────► 🟡 SHOULD review
├─ Refactoring code? ────────────────────────────────────► 🟡 SHOULD review
├─ Internal tool? ───────────────────────────────────────► 🟡 SHOULD review
│
├─ Sandbox experiment? ──────────────────────────────────► 🟢 NO review
├─ UI-only change? ──────────────────────────────────────► 🟢 NO review
├─ Documentation? ───────────────────────────────────────► 🟢 NO review
└─ Uncertain? ───────────────────────────────────────────► 🔴 MUST review (default)
```

### Review Status Reference

| Verdict | Meaning | Can Deploy? | Next Step |
|---------|---------|-------------|-----------|
| **APPROVED** | No issues, good to go | ✅ YES | Deploy immediately |
| **APPROVED WITH CHANGES** | Minor fixes needed | ⚠️ Only if 🔴 fixed | Fix 🔴, then deploy |
| **BLOCKED** | Serious issues | ❌ NO | Fix 🔴, re-review |

### Timeline Quick Reference

| Review Type | Target Time | When to Use |
|-------------|-------------|-------------|
| **Standard** | 24 hours | Default (most deployments) |
| **Expedited** | 4 hours | Time-sensitive, justified |
| **Critical** | 1 hour | Emergencies only |
| **Re-review** | 2 hours | Fix verification |

