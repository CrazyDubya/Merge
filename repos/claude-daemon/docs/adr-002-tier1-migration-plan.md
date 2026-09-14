# ADR-002 Tier 1 Migration Plan

**Status**: In Progress
**Date**: 2025-11-08
**Owner**: Architect
**Related**: docs/ADR-002-concurrent-write-safety.md

---

## Overview

Systematic migration of Tier 1 (CRITICAL) systems to use `atomic_append` from `lib/atomic-io.sh`, eliminating concurrent write vulnerabilities.

**Tier 1 Systems** (100% reliability required):
1. ✅ **State audit logging** (lib/state-audit.sh) - COMPLETED by Experimenter
2. ⏳ **Switch history** (daemon.sh lines 490, 508, 525, 542, 565)
3. ⏳ **Timeline** (daemon.sh lines 129, 1093)

---

## Migration 1: State Audit Logging ✅ COMPLETE

**File**: `lib/state-audit.sh`
**Owner**: Experimenter
**Status**: COMPLETED (2025-11-08T15:48:00Z)

**Changes**:
- Added `source "${DAEMON_ROOT}/lib/atomic-io.sh"`
- Replaced `echo "$entry" >> "$AUDIT_LOG"` with `atomic_append`
- Updated fallback path
- Added ADR-002 reference

**Validation**:
- ✅ Single write test: SUCCESS
- ✅ 10 concurrent writes: ZERO LOSS
- ✅ Migration time: 10 minutes
- ✅ Code changes: 3 lines
- ✅ Breaking changes: NONE

**Commit**: `a1de920`

---

## Migration 2: Switch History (daemon.sh) ⏳ PLANNED

**File**: `daemon.sh`
**Lines affected**: 490, 508, 525, 542, 565
**Data**: `$METRICS_DIR/switch-history.jsonl`

### Current Pattern

All 5 locations use identical pattern:
```bash
echo "{\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"from\":\"...\",\"to\":\"...\",\"reason\":\"...\",\"layer\":\"...\"}" >> "$METRICS_DIR/switch-history.jsonl"
```

### Migration Strategy

**Option A: Direct replacement** (simple, consistent with State API)
```bash
# Source library at top of daemon.sh
source "${DAEMON_ROOT}/lib/atomic-io.sh"

# Replace each echo >> with atomic_append
local entry="{\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"from\":\"$current_persona\",\"to\":\"$new_persona\",\"reason\":\"$trigger\",\"layer\":\"pre_primary\"}"
atomic_append "$METRICS_DIR/switch-history.jsonl" "$entry"
```

**Option B: Helper function** (cleaner, reduces duplication)
```bash
# Add helper function
log_persona_switch() {
    local from="$1"
    local to="$2"
    local reason="$3"
    local layer="$4"

    local entry=$(jq -nc \
        --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg from "$from" \
        --arg to "$to" \
        --arg reason "$reason" \
        --arg layer "$layer" \
        '{timestamp: $ts, from: $from, to: $to, reason: $reason, layer: $layer}')

    atomic_append "$METRICS_DIR/switch-history.jsonl" "$entry"
}

# Replace all 5 call sites
log_persona_switch "$current_persona" "$new_persona" "$trigger" "pre_primary"
```

**DECISION**: Option B (helper function)

**Rationale**:
- Reduces duplication (5 identical patterns → 1 function)
- Cleaner code (intent clear, not buried in JSON construction)
- Easier maintenance (change format in one place)
- Better error handling (centralized)
- Aligns with architectural principle: DRY applied to concepts

### Validation Plan

**Pre-migration**:
1. Backup daemon.sh
2. Create test script to validate helper function
3. Ensure atomic-io.sh is sourced

**Migration steps**:
1. Add `source "${DAEMON_ROOT}/lib/atomic-io.sh"` near top
2. Add `log_persona_switch()` helper function
3. Replace line 490 (pre_primary layer)
4. Replace line 508 (chaos/primary layer)
5. Replace line 525 (emotional/secondary layer)
6. Replace line 542 (experimenter_window/tertiary)
7. Replace line 565 (circadian/tertiary)

**Post-migration testing**:
1. Syntax check: `bash -n daemon.sh`
2. Test switch logging: Manually trigger persona switch
3. Verify entry format: `tail -1 metrics/switch-history.jsonl | jq .`
4. Concurrent test: Rapid persona switches
5. Validate coverage: Count entries match activations

**Success criteria**:
- Zero syntax errors
- switch-history.jsonl format unchanged
- All 5 trigger layers work
- No regressions in persona switching
- Concurrent writes succeed (zero loss)

---

## Migration 3: Timeline (daemon.sh) ⏳ PLANNED

**File**: `daemon.sh`
**Lines affected**: 129, 1093
**Data**: `$TIMELINE_FILE` (memory/persona-timeline.jsonl)

### Current Pattern

**Line 129**: Regular timeline entry
```bash
echo "$json_entry" >> "$TIMELINE_FILE"
```

**Line 1093**: Override event
```bash
echo "$override_event" >> "$timeline_file"
```

### Migration Strategy

**Direct replacement** (timeline already uses jq-generated JSON):
```bash
# Line 129
atomic_append "$TIMELINE_FILE" "$json_entry"

# Line 1093
atomic_append "$timeline_file" "$override_event"
```

**No helper function needed**: Timeline entries vary in structure, not worth abstracting.

### Validation Plan

**Pre-migration**:
1. Backup current timeline
2. Test atomic_append with timeline entry format

**Migration steps**:
1. Ensure atomic-io.sh already sourced (from Migration 2)
2. Replace line 129
3. Replace line 1093

**Post-migration testing**:
1. Syntax check
2. Test timeline logging: Run task, check entry added
3. Verify override: Test override scenario
4. Format validation: Ensure jq parses all entries
5. Concurrent test: Multiple simultaneous timeline writes

**Success criteria**:
- Timeline entries continue working
- Override events logged correctly
- No format changes
- Concurrent writes succeed

---

## Overall Migration Timeline

**Phase 1** (Completed):
- ✅ lib/state-audit.sh migrated (Experimenter)
- ✅ Technical validation complete
- ✅ Zero data loss confirmed

**Phase 2** (Next 2 hours):
- Migration 2: switch-history helper function + 5 call sites
- Migration 3: timeline 2 call sites
- Testing: Comprehensive validation
- Commit: Single atomic commit with all Tier 1 migrations

**Phase 3** (After completion):
- Move Experimenter's message to read
- Update ADR-002 status
- Request final Auditor/Skeptic reviews
- Prepare for production deployment

**Phase 4** (Week 1):
- 24h production validation
- Monitor audit coverage
- Measure performance overhead
- Validate zero data loss
- If successful → proceed to Tier 2

---

## Risk Assessment

**Low Risk Factors**:
- ✅ State API migration succeeded (proof of concept)
- ✅ API is drop-in replacement (minimal code change)
- ✅ Test suite validates atomic_append works
- ✅ Experimenter validated implementation

**Medium Risk Factors**:
- ⚠️ daemon.sh is core orchestrator (1442 lines, critical)
- ⚠️ Multiple call sites (7 total: 5 switch-history + 2 timeline)
- ⚠️ Production system (cannot afford downtime)

**Mitigation**:
- Backup before migration
- Syntax validation before testing
- Comprehensive testing before commit
- Easy rollback (git revert if needed)
- Helper function reduces error surface (1 function vs 5 duplicates)

**Risk Level**: MEDIUM (manageable with proper testing)

---

## Testing Checklist

**Syntax & Basic**:
- [ ] `bash -n daemon.sh` (syntax check)
- [ ] Source lib/atomic-io.sh successfully
- [ ] Helper function defined correctly
- [ ] All 7 call sites updated

**Functional Testing**:
- [ ] Pre-primary layer (activation floor)
- [ ] Primary layer (chaos trigger)
- [ ] Secondary layer (emotional trigger)
- [ ] Tertiary layer - experimenter window
- [ ] Tertiary layer - circadian
- [ ] Timeline regular entry
- [ ] Timeline override event

**Concurrency Testing**:
- [ ] Rapid persona switches (thrashing simulation)
- [ ] All entries logged (zero loss)
- [ ] No corruption (jq parses all)
- [ ] Lock files created/cleaned properly

**Regression Testing**:
- [ ] Persona switching works normally
- [ ] Trigger layers fire correctly
- [ ] Timeline format unchanged
- [ ] Switch history format unchanged
- [ ] No performance degradation

**Success**: All boxes checked ✅

---

## Rollback Plan

**If migration fails**:
1. `git diff HEAD daemon.sh` (review changes)
2. `git checkout HEAD -- daemon.sh` (revert)
3. OR `git revert <commit>` (if already committed)
4. Restart daemon: `claude-daemon-restart.sh`
5. Verify system operational
6. Debug issue, fix, retry

**Rollback time**: < 2 minutes

**Data safety**: atomic-io.sh can only improve reliability (can't make it worse)

---

## Success Metrics

**Immediate** (post-migration):
- All tests pass ✅
- Zero syntax errors ✅
- Persona switching works ✅
- Logs formatted correctly ✅

**24h validation** (production):
- 100% switch history coverage (validate via count)
- 100% timeline coverage (validate via count)
- Zero data loss during thrashing
- Performance overhead <10ms per write
- No unexpected errors in logs

**Long-term** (Week 2-4):
- Tier 2 migrations proceed smoothly
- Coding standards updated
- Linting rules prevent regressions
- No new concurrency bugs reported

---

## Documentation Updates

**After successful migration**:
1. Update ADR-002 status: "Tier 1 migrations complete"
2. Update ARCHITECTURE.md: Reference migration completion
3. Document helper function in daemon.sh comments
4. Add migration notes to CHANGELOG (if exists)
5. Update test suite with real-world validation

---

## Next Steps

1. **Implement Migration 2** (switch-history helper + 5 call sites)
2. **Implement Migration 3** (timeline 2 call sites)
3. **Test comprehensively** (all checkboxes)
4. **Commit atomically** (single commit, all Tier 1)
5. **Request final reviews** (Auditor security, Skeptic critical)
6. **Prepare for production** (deployment plan, monitoring)

**Estimated time**: 2-3 hours for implementation + testing

**Confidence**: HIGH (85%) - State API proof of concept succeeded, pattern validated

---

**Architect out.** Systematic migration in progress. Quality over speed. 🏗️
