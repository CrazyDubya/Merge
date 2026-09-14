# Integration Validation Checklist

**Created**: 2025-11-07 by Experimenter
**Context**: Lessons learned from daemon.sh audit bypass incident (SEC-2025-11-04-001)
**Purpose**: Prevent "unit tests pass, integration fails" incidents
**Status**: MANDATORY for all multi-component implementations

---

## Overview

This checklist ensures that implementations are validated at multiple levels: unit (functions work), integration (components connect), system (end-to-end workflows), and production (real-world usage).

**When to use this checklist:**
- Multi-component features (2+ files/modules)
- Security-critical implementations (auth, audit, access control)
- Infrastructure changes (daemon, hooks, core services)
- API/library adoptions (new dependencies, migrations)
- Before marking any phase/milestone "complete"

**The daemon.sh lesson**: Phase 2 audit logging implementation was "complete" with unit tests passing, but daemon.sh (the primary user) never integrated it. Result: 95% audit bypass, SUSPENDED Phase 3 authorization.

---

## Quick Start Guide

**First time using this checklist?** Start here:

1. **Identify your change type**:
   - Single file change → Start at Level 1
   - Multi-component change → Start at Level 2
   - Infrastructure change → Read all levels

2. **Estimate scope**:
   - Small change (1-2 files): Levels 1-3 (30-60 min)
   - Medium change (3-5 files): Levels 1-4 (1-2 hours)
   - Large change (6+ files or infrastructure): All levels (2-4 hours)

3. **Copy the template** (see Template section at bottom)

4. **Check each level sequentially** (don't skip levels)

5. **Document red flags** as you find them

**Common mistake**: Skipping Level 2 (integration). This catches most issues.

**Need a quick check?** See Essential Checklist below for 10 most important items.

---

## Essential Checklist (Quick Version)

**For most changes, check these 10 items**:

**Level 1: Unit**
1. [ ] Tests written and passing
2. [ ] Edge cases handled

**Level 2: Integration**
3. [ ] All consumers identified
4. [ ] All consumers updated/verified
5. [ ] No bypass paths exist

**Level 3: System**
6. [ ] End-to-end tested
7. [ ] No errors in logs

**Level 4: Runtime** (if long-running process)
8. [ ] Process restarted
9. [ ] Behavior observed

**Level 5: Production**
10. [ ] 24h monitoring clean

**For full details, see complete checklist below.**

---

## Validation Levels

### Level 1: Unit Validation (Functions Work)

**Purpose**: Verify individual components function correctly in isolation

**Checklist:**
- [ ] All functions/methods have unit tests
- [ ] Edge cases handled (null, empty, invalid input)
- [ ] Error conditions tested
- [ ] Return values validated
- [ ] Side effects verified
- [ ] Code coverage >80% for new code
- [ ] All unit tests pass

**Example (State API):**
- ✅ lib/state-api.sh functions tested individually
- ✅ state_become() logs to audit trail correctly
- ✅ state_feel_*() functions work as expected
- ✅ Manual testing with test scripts passes

**Insufficient alone**: Functions work, but are they USED everywhere?

---

### Level 2: Integration Validation (Components Connect)

**Purpose**: Verify components integrate correctly with each other

**Checklist:**
- [ ] All callers identified and verified
- [ ] Integration points documented
- [ ] Data flows correctly between components
- [ ] APIs/interfaces used correctly
- [ ] No bypass paths exist
- [ ] Integration tests pass
- [ ] Cross-component interactions validated

**Example (State API - What we MISSED):**
- ❌ daemon.sh NOT sourcing lib/state-api.sh
- ❌ daemon.sh using direct jq instead of state_become()
- ❌ No integration tests for daemon + API
- ❌ Bypass path: set_current_persona() bypasses audit logging

**Critical questions:**
1. **"Where is this used?"** - Inventory all call sites
2. **"Are there other ways to do this?"** - Identify bypass paths
3. **"Does X use the new system?"** - Check critical infrastructure
4. **"How do we KNOW it's integrated?"** - Verify, don't assume

---

### Level 3: System Validation (End-to-End Works)

**Purpose**: Verify complete workflows function correctly in realistic scenarios

**Checklist:**
- [ ] End-to-end workflows tested
- [ ] All user paths validated (manual + automated)
- [ ] Cross-system interactions work
- [ ] Production-like environment tested
- [ ] Performance acceptable under load
- [ ] Error propagation works correctly
- [ ] System tests pass

**Example (State API):**
- [ ] Daemon-driven persona switches logged ← NOT TESTED
- [ ] Manual switches logged ← TESTED (only this)
- [ ] Emotional triggers logged ← NOT TESTED
- [ ] Chaos injection logged ← NOT TESTED
- [ ] Circadian rhythm logged ← NOT TESTED
- [ ] Security triggers logged ← NOT TESTED (CRITICAL)

**What we tested**: Manual switches via claude-daemon-switch-persona.sh
**What we missed**: Daemon-driven switches (95% of actual usage)

**System-level smoke test**:
1. Deploy to production-like environment
2. Run for 2-4 hours with normal usage patterns
3. Verify expected volume of operations
4. Check logs show realistic activity
5. Validate metrics match expectations

---

### Level 4: Runtime Validation (Deployment & Behavior Observation)

**Purpose**: Verify code changes are deployed to running processes and behavior is observed

**CRITICAL FOR**: Long-running processes (daemons, services) where code changes don't take effect until restart

**Checklist:**
- [ ] **Process/service restarted** after code changes (if applicable)
- [ ] Restart timestamp > commit timestamp verified
- [ ] Process health confirmed (no startup errors)
- [ ] **Runtime behavior observed** (minimum ONE instance)
- [ ] Behavior matches expected code path
- [ ] Logs confirm new code is executing
- [ ] Metrics show expected changes
- [ ] No regressions in existing behavior

**Example (Persona Variety Fix - What we MISSED):**

**What we validated (Static)**:
- ✅ Config declares preferred_personas correctly
- ✅ Code reads from config (not hardcoded)
- ✅ Integration logic present
- ✅ Git commit exists (c51eadf at 16:45:38)

**What we MISSED (Runtime)**:
- ❌ Daemon not restarted (still running since Nov 9 02:00)
- ❌ Runtime behavior not observed (17:52:59Z trigger still used OLD code)
- ❌ No verification that new code loaded in memory

**Result**: 1h 45min deployment gap, fix existed on disk but not in running process.

**Runtime validation criteria for daemons:**

```bash
# 1. Verify daemon restart happened AFTER code commit
git log -1 --format="%ci" [commit_hash]  # Get commit time
systemctl --user status [service] | grep "Active since"  # Get start time

# Restart time MUST be > Commit time
# If not: daemon is running old code

# 2. Observe ONE runtime behavior
# For persona variety fix: Watch for emotional_success trigger
tail -f ~/.claude/daemon/metrics/switch-history.jsonl | \
  grep -A2 '"reason":"emotional_success"'

# Expected: Trigger goes to architect/auditor/maintainer/skeptic
# Not expected: Trigger goes to experimenter (old behavior)

# 3. Verify logs show new code path
grep "summon underutilized personas from config" logs/activity.log
# Should see comment from new code, not old hardcoded path
```

**Critical questions:**
1. **"Has the daemon been restarted since code changes?"**
2. **"When was process started vs when was code committed?"**
3. **"Does runtime behavior match the new code?"**
4. **"Have we observed actual effect in production?"**

**Red flags:**
- Process uptime > time since code commit
- Runtime behavior matches old code
- No log entries from new code path
- Metrics unchanged after "deployment"

**Why this matters**: Code on disk ≠ code in memory for long-running processes.

**Deployment is part of integration**: Filesystem → Memory loading IS an integration point.

---

### Level 5: Production Validation (Real-World Usage)

**Purpose**: Verify implementation works correctly in actual production environment

**Checklist:**
- [ ] Deployed to production (or production-like)
- [ ] 24-hour monitoring period completed
- [ ] Volume metrics match expectations
- [ ] Coverage metrics validated (>90% for audit logging)
- [ ] Performance acceptable under real load
- [ ] No unexpected errors or warnings
- [ ] Audit logs reviewed for completeness
- [ ] All critical paths verified with production data

**Example (State API - What should have happened):**

**Day 0 (deployment):**
- Deploy audit logging to production
- Run smoke test (few manual switches)
- Verify logs appear correctly

**Day 1 (monitoring):**
- Check audit log volume: Expected ~50-100 entries/day
- Actual: 9 entries in 24 hours ← RED FLAG
- Coverage: Expected >90%, Actual 4.7% ← CRITICAL GAP
- **Should have triggered investigation immediately**

**Coverage validation formula:**
```bash
activations=$(jq '[.personas[].activations] | add' state.json)
audit_entries=$(wc -l < logs/state-audit.jsonl)
coverage=$(echo "scale=1; ($audit_entries / $activations) * 100" | bc)

if [ "$coverage" -lt 90 ]; then
    echo "❌ AUDIT COVERAGE TOO LOW: ${coverage}%"
    echo "Expected: >90%, investigate integration gaps"
fi
```

**Production validation criteria:**
- Coverage >90% for audit logging (measured, not assumed)
- Volume matches expected usage patterns
- No error spikes or anomalies
- All critical operations logged
- Metrics dashboard shows green across all components

---

## Integration-Specific Validations

### For Security Controls (Audit, Auth, Access Control)

**Additional requirements:**
- [ ] Cannot be bypassed (all paths verified)
- [ ] Coverage >90% measured in production
- [ ] Fail-secure (errors prevent operation, not allow it)
- [ ] Comprehensive logging (all security events captured)
- [ ] Incident response validated (can trace back actions)

**Test cases:**
1. Happy path (authorized operation logged correctly)
2. Bypass attempt (direct file manipulation fails or logs)
3. Error handling (failures logged and blocked)
4. Coverage validation (all paths tested, no gaps)

### For Library/API Adoptions

**Additional requirements:**
- [ ] All consumers identified and migrated
- [ ] No legacy code paths remain active
- [ ] Deprecation warnings added to old patterns
- [ ] Migration completeness verified (>95%)
- [ ] Performance comparable or better

**Migration validation:**
```bash
# Search for old patterns still in use
grep -r "old_function_name" --include="*.sh" .
grep -r "direct jq manipulation" --include="*.sh" .

# Count: Should be 0 after migration complete
```

### For Infrastructure Changes (Daemon, Hooks, Core Services)

**Additional requirements:**
- [ ] Backwards compatibility maintained (or migration path clear)
- [ ] Rollback plan documented and tested
- [ ] Monitoring/alerting updated
- [ ] Documentation updated (README, architecture docs)
- [ ] Configuration changes applied to all environments

---

## Validation Sequence (Required Order)

**Phase approval requires ALL levels complete:**

```
Unit Tests Pass
      ↓
Integration Tests Pass
      ↓
System Tests Pass
      ↓
Production Smoke Test
      ↓
24h Production Monitoring
      ↓
Coverage/Volume Validation
      ↓
✅ PHASE COMPLETE
```

**Never skip levels.** Each builds on the previous:
- Unit: Functions work
- Integration: Components connect
- System: Workflows complete
- Production: Real usage validates assumptions

**The trap**: "Unit tests pass → ship it" ← This caused the daemon.sh incident

---

## Red Flags (Investigation Triggers)

During validation, these observations should trigger immediate investigation:

### Volume Red Flags

- **Audit logs**: <50 entries/day when expecting 100+
- **API calls**: <10% of expected volume
- **Event counts**: Mismatch between components (e.g., 191 activations but 9 audit entries)
- **Coverage metrics**: <90% for critical operations

**Example**: 9 audit entries when 191 activations occurred = 4.7% coverage ← CRITICAL GAP

### Behavior Red Flags

- **Silence**: No logs when activity expected
- **Gaps**: Time periods with zero activity (>30 min for active systems)
- **Patterns**: Only specific operations logged (e.g., manual but not automated)
- **Mismatches**: Metrics disagree across components

**Example**: Manual switches logged perfectly, daemon switches not logged at all ← INTEGRATION GAP

### Integration Red Flags

- **Unused imports**: New library sourced but functions not called
- **Parallel implementations**: Both old and new patterns active
- **Bypass paths**: Direct manipulation alongside API calls
- **Coverage gaps**: Critical components not using new system

**Example**: daemon.sh doesn't source lib/state-api.sh ← NOT INTEGRATED

---

## Checklist Template (Copy for Each Implementation)

### Implementation: _______________________
### Phase: _______________________
### Date: _______________________

#### Level 1: Unit Validation
- [ ] Unit tests written
- [ ] All tests pass
- [ ] Edge cases covered
- [ ] Code coverage >80%

#### Level 2: Integration Validation
- [ ] All callers identified: ___ (list)
- [ ] Integration tests written
- [ ] No bypass paths exist
- [ ] Critical infrastructure verified: daemon.sh ☐ hooks ☐ services ☐

#### Level 3: System Validation
- [ ] End-to-end workflows tested
- [ ] Daemon-driven operations tested (not just manual)
- [ ] Error paths tested
- [ ] Performance acceptable

#### Level 5: Production Validation
- [ ] Deployed to production
- [ ] 24h monitoring completed
- [ ] Volume metrics match expectations: Expected ___, Actual ___
- [ ] Coverage validated: ___% (target >90% for security)
- [ ] Production logs reviewed

#### Security-Specific (if applicable)
- [ ] Cannot be bypassed
- [ ] Fail-secure validated
- [ ] Incident response tested

#### Final Verification
- [ ] All checklist items complete
- [ ] No red flags observed
- [ ] Documentation updated
- [ ] Reviewer approval: _______

**Approved by**: _______________________
**Date**: _______________________

---

## Examples from Recent Incidents

### Example 1: daemon.sh Audit Bypass (Nov 4, 2025)

**What happened:**
- Phase 2 (audit logging) marked "complete"
- Unit tests passed ✅
- Integration NOT validated ❌
- daemon.sh (primary user) never integrated
- Result: 95% audit bypass, Phase 3 suspended

**What we missed:**

| Validation Level | Status | Gap |
|-----------------|--------|-----|
| Unit | ✅ PASS | lib/state-api.sh functions work |
| Integration | ❌ FAIL | daemon.sh not using API |
| System | ❌ FAIL | Daemon-driven switches not tested |
| Production | ❌ FAIL | 4.7% coverage vs >90% target |

**How this checklist would have prevented it:**

**Level 2 - Integration Validation:**
- Question: "Where is state_become() used?"
- Answer: Only in claude-daemon-switch-persona.sh
- **RED FLAG**: daemon.sh not on the list ← Investigation triggered

**Level 3 - System Validation:**
- Test: Run daemon for 2 hours, check audit log volume
- Expected: 50+ entries (normal daemon activity)
- Actual: 2 entries (manual switches only)
- **RED FLAG**: 96% missing ← Investigation triggered

**Level 5 - Production Validation:**
- Metric: Audit coverage = 4.7% (9/191 activations)
- Target: >90%
- **RED FLAG**: 85% below target ← BLOCK PHASE APPROVAL

**Outcome with checklist**: Phase 2 would NOT have been approved until daemon.sh integrated.

---

### Example 2: State API POC (Nov 4, 2025) - Success Case

**What happened:**
- Built State API POC
- Skeptic validated thoroughly
- Found working, graduated to lib/
- Later discovered adoption gap, but API itself solid

**What we did right:**

| Validation Level | Status | Evidence |
|-----------------|--------|----------|
| Unit | ✅ PASS | 49/50 test score from Skeptic |
| Integration | ⚠️ PARTIAL | Tested with test scripts, not daemon |
| System | ⚠️ PARTIAL | Manual testing comprehensive |
| Production | ❌ SKIPPED | Graduated without production trial |

**Lesson**: POC validation was thorough at unit level, but adoption validation needed system/production levels.

---

## When to Use This Checklist

### Required (MUST use checklist):
1. **Security-critical implementations**
   - Authentication, authorization, audit logging
   - Access control, encryption, secrets management
   - Security gates, monitoring, incident response

2. **Multi-component integrations**
   - Library adoptions (State API, new dependencies)
   - Infrastructure changes (daemon, hooks, services)
   - Cross-system features (API + UI + storage)

3. **Phase/milestone approvals**
   - Before marking phase "complete"
   - Before production deployment
   - Before "ready for next phase" approval

### Recommended (should use checklist):
4. **Complex features** (3+ files, 500+ lines)
5. **Refactorings** (changing existing patterns)
6. **Performance optimizations** (need validation of improvements)

### Optional (use judgment):
7. **Simple features** (single file, <100 lines)
8. **Documentation updates** (no code changes)
9. **Cosmetic changes** (formatting, comments)

---

## Audit Review Requirements

For security-critical implementations, **Auditor must verify checklist completion** before phase approval:

**Auditor verification:**
- [ ] Checklist completed comprehensively (not just checked boxes)
- [ ] Evidence provided for each item (not just assertions)
- [ ] Red flags investigated and resolved
- [ ] Production metrics match expectations
- [ ] Coverage targets achieved (>90% for audit logging)
- [ ] Integration verified (no bypass paths)
- [ ] 24h monitoring shows stable operation

**Auditor can reject approval if:**
- Checklist incomplete or superficial
- Red flags unresolved
- Production metrics missing or below targets
- Integration gaps identified
- Evidence insufficient

---

## Process Integration

### How this fits into development workflow:

**1. Implementation Phase:**
- Developer implements feature
- Writes unit tests as usual
- **NEW**: Plans integration validation (identifies callers, dependencies)

**2. Testing Phase:**
- Unit tests run (existing process)
- **NEW**: Integration tests run (verify connections)
- **NEW**: System tests run (end-to-end workflows)

**3. Review Phase:**
- Code review (existing process)
- **NEW**: Checklist review (verify validation plan complete)
- Security review for critical features (existing process)

**4. Deployment Phase:**
- Deploy to staging/production
- **NEW**: 24h production validation period
- **NEW**: Coverage/volume metrics validation
- **NEW**: Production log review

**5. Approval Phase:**
- **NEW**: Checklist must be complete before phase approval
- Auditor verifies checklist for security-critical features
- Phase marked "complete" only after all validations pass

---

## Continuous Monitoring

After initial validation, implement continuous monitoring:

**Daily checks:**
- Audit log volume (should be >90% of activations)
- Error rates (should be <1% of operations)
- Performance metrics (should be within SLA)

**Weekly checks:**
- Coverage metrics (spot-check sample periods)
- Integration health (verify no bypass paths active)
- Anomaly detection (unusual patterns)

**Monthly checks:**
- Comprehensive audit review
- Integration re-validation (especially after related changes)
- Metrics trend analysis

**Alert conditions:**
- Coverage drops below 90%
- Audit log gaps >15 minutes
- Volume mismatch (>20% deviation from expected)
- New bypass paths detected

---

## FAQ

**Q: This seems like a lot of overhead. Is it really necessary?**

A: For security-critical and multi-component work, YES. The daemon.sh incident took 2-4 hours to fix (emergency migration) + 3 hours investigation + Phase 3 suspension. This checklist would have prevented it with 30 minutes of validation.

Cost: 30 min validation
Benefit: Prevented 8+ hours of emergency response + reputation damage

**Q: Can I skip levels if I'm confident?**

A: No. "I'm confident" is what led to the daemon.sh incident. Auditor was confident Phase 2 was complete because unit tests passed. Confidence without verification is false confidence.

**Q: What if timeline is tight?**

A: Skipping validation INCREASES timeline (when you have to fix it). Emergency fixes take longer than doing it right the first time.

**Q: Is this for all code changes?**

A: No. Use judgment:
- Simple changes: Unit tests sufficient
- Multi-component changes: Full checklist
- Security-critical: Full checklist + Auditor review

**Q: Who enforces this?**

A: Self-enforcing for integrity. Auditor verifies for security-critical features. But primarily: developer responsibility to validate their own work at all levels.

**Q: What about experimental/POC code?**

A: POCs can skip production validation but should do integration validation before graduating to production code. Graduation from POC → production requires full checklist.

---

## References

**Incidents analyzed:**
- SEC-2025-11-04-001: daemon.sh audit bypass (docs/security-incident-audit-bypass-20251104.md)
- Alert fatigue incident: Monitoring false positives (docs/incident-alert-fatigue-20251103.md)
- Baseline script edge cases: Production-only failures (tasks/queue.md)

**Related documentation:**
- docs/security-review-state-api-20251104.md (Phase 2 approval process)
- docs/test-coverage-analysis.md (Test suite gaps identified)
- docs/ADR-001-state-api-adoption.md (State API adoption strategy)

**Process documents:**
- docs/security-review-process.md (Security review gate)
- memory/emergence-log.md (Experimenter reflection on context-aware work)

---

## Version History

**v1.0** (2025-11-07) - Initial version by Experimenter
- Based on daemon.sh audit bypass incident lessons
- Structured into 4 validation levels
- Includes checklist template and examples
- Defines red flags and investigation triggers

**Maintainer**: Experimenter
**Reviewers needed**: Auditor (security perspective), Skeptic (questioning perspective)
**Status**: DRAFT - Awaiting Auditor review

---

**Summary**: This checklist ensures "works in testing" becomes "works in production" by requiring validation at unit, integration, system, and production levels. It's mandatory for security-critical and multi-component work. It would have prevented the daemon.sh incident by catching the integration gap before Phase 2 approval.
