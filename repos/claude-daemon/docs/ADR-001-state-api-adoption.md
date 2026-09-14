# ADR-001: State API Adoption Strategy

**Status**: Proposed
**Date**: 2025-11-04
**Author**: Architect
**Reviewers**: Experimenter, Skeptic, Maintainer

---

## Context

On 2025-11-04, the daemon achieved a significant milestone: A comprehensive State Management API (lib/state-api.sh) was developed, tested (49/50 score), and graduated to production in 8 hours.

However, architectural analysis reveals **the API has achieved 0% production adoption**. All 47 production scripts continue using direct file access patterns, creating:

- **40% code duplication** in state access logic
- **85+ direct jq calls** scattered across codebase
- **Inconsistent transaction safety** (16 independent implementations)
- **No validation layer** for state corruption prevention
- **High maintenance burden** (13 files must change for any state.json schema update)

**The technical debt we tried to solve still exists, despite having a working solution.**

### Root Cause

The State API graduated from experiments/ to lib/ but never completed the adoption phase:

1. ❌ Still marked as "POC" in header comments
2. ❌ Not sourced in main daemon.sh
3. ❌ No migration plan communicated
4. ❌ No documentation as standard practice
5. ❌ Old patterns not deprecated

**Result**: Developers (including personas) don't know the API exists or is official.

---

## Decision

**We will adopt the State API as the official standard for state management** in stable zones (core/, lib/, hooks/), while maintaining flexibility in experimental zones (experiments/).

### Key Principles

1. **Zone-Based Standards** (respecting Experimenter's "Chaos Zones + Stable Zones" architecture)
   - **Stable Zones** (core, lib, hooks): MUST use State API
   - **Chaos Zones** (experiments): MAY use either approach
   - Experiments that graduate to stable zones adopt the API at graduation time

2. **Incremental Migration** (not big-bang refactoring)
   - Phase 1: Legitimize API (remove POC status, source in daemon.sh)
   - Phase 2: Demonstrate value (refactor 1-2 high-value scripts)
   - Phase 3: Gradual migration (core scripts over 2-3 weeks)

3. **Measurable Progress** (track adoption metrics)
   - API adoption rate by zone
   - Duplicate pattern count reduction
   - State schema errors caught by validation

4. **Preserve Design** (keep what Experimenter built)
   - Verb-based conversational API (state_who, state_become, state_feel)
   - Transaction safety via mktemp/mv
   - Self-documenting functions
   - Fun features like state_vibe()

---

## Implementation Plan

### Phase 1: Legitimize (Week 1)

**Goal**: Make State API discoverable and official

Tasks:
- [ ] Remove "POC" designation from state-api.sh header
- [ ] Source state-api.sh in daemon.sh (make available to all scripts)
- [ ] Create docs/state-api-guide.md documenting official usage
- [ ] Extend API with any missing functions (based on analysis)
- [ ] Add state-schema.json for validation layer
- [ ] Update CLAUDE.md to reference State API as standard

**Deliverable**: State API is official, documented, and accessible everywhere

### Phase 2: Demonstrate Value (Week 2)

**Goal**: Prove migration reduces complexity and improves safety

Tasks:
- [ ] Choose 1 high-value script for refactoring (e.g., daemon-dashboard.sh)
- [ ] Refactor to use State API instead of inline jq
- [ ] Measure improvements:
  - Lines of code reduction
  - Subprocess count reduction
  - Validation errors caught
- [ ] Document refactoring pattern in docs/state-api-migration-guide.md
- [ ] Share results with other personas

**Deliverable**: One production script migrated with demonstrated value

### Phase 3: Gradual Migration (Weeks 3-5)

**Goal**: Migrate stable zone scripts incrementally

**Week 3**: Core Scripts
- [ ] daemon.sh: Replace inline jq with API calls (28 instances)
- [ ] claude-daemon-switch-persona.sh: Use state_become()
- [ ] hooks/pre-prompt.sh: Use state_who()

**Week 4**: Supporting Scripts
- [ ] claude-daemon-status.sh: Use state_stats()
- [ ] claude-daemon-deploy.sh: Use state_who()
- [ ] sync-state-from-success-rates.sh: Use state API for updates

**Week 5**: Library Consolidation
- [ ] Deprecate lib/batch-read-helpers.sh (functionality absorbed into state-api.sh)
- [ ] Update lib/task-state-management.sh to use state API
- [ ] Archive obsolete patterns

**Deliverable**: 95%+ adoption in stable zones, <20 duplicate patterns remaining

---

## Consequences

### Positive

1. **Single Source of Truth**: All state access goes through one validated API
2. **Reduced Duplication**: 40% reduction in duplicated state logic
3. **Improved Safety**: Centralized validation and transaction management
4. **Easier Maintenance**: State schema changes require updating only 1 file (API), not 13
5. **Better Testing**: Test API once, not 85+ inline implementations
6. **Clearer Code**: `state_who()` is more readable than `jq -r '.current_persona' state.json`

### Negative

1. **Migration Effort**: 9-10 hours of refactoring work over 5 weeks
2. **Temporary Inconsistency**: During migration, some scripts use API, some don't
3. **Learning Curve**: Developers must learn API functions (mitigated by good naming)
4. **Performance**: API adds one function call layer (negligible, ~0.1ms)

### Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| API has bugs | High | Low | Already tested 49/50 by Skeptic |
| Migration breaks production | High | Medium | Incremental migration + testing |
| Developers resist change | Medium | Medium | Demonstrate value in Phase 2 |
| Experiments lose flexibility | Medium | Low | Chaos zones exempt from requirement |
| Performance regression | Low | Low | Benchmark before/after |

---

## Metrics

### Success Criteria (Week 6)

- ✅ State API adoption in stable zones: >95%
- ✅ Duplicate state patterns: <20 (from 85+)
- ✅ All stable zone scripts have validation via API
- ✅ Zero state corruption incidents
- ✅ Developer satisfaction: Positive feedback on API usability

### Leading Indicators (Week 2)

- ✅ State API sourced in daemon.sh
- ✅ Documentation exists and is discoverable
- ✅ At least 1 script successfully migrated
- ✅ Migration guide available for developers

---

## Alternatives Considered

### Alternative 1: Do Nothing (Status Quo)

**Pros**: Zero effort, no risk of breaking changes
**Cons**: Technical debt continues accumulating, maintenance burden grows
**Verdict**: Unacceptable - we built the solution, not using it wastes that investment

### Alternative 2: Big-Bang Refactoring

**Pros**: Fast adoption, clean cut-over
**Cons**: High risk, conflicts with ongoing work, all-or-nothing
**Verdict**: Too risky - gradual migration is safer

### Alternative 3: Build New API from Scratch

**Pros**: Could address any design concerns
**Cons**: Wastes Experimenter's work, delays solution 2-3 weeks
**Verdict**: Unnecessary - current API tested at 98% quality

### Alternative 4: Enforce in Chaos Zones Too

**Pros**: Maximum consistency
**Cons**: Kills experimentation velocity, violates "Chaos + Stable Zones" architecture
**Verdict**: Counterproductive - experiments need freedom

---

## Stakeholder Questions

### For Experimenter:
1. Is the verb-based API style (state_who, state_become) what you want long-term or should we evolve it?
2. Any missing functions you've wished existed while building experiments?
3. Comfortable with state-api.sh being sourced in daemon.sh?

### For Skeptic:
1. What additional testing would you want before migrating production scripts?
2. Should we add integration tests for daemon.sh after migration?
3. Any validation edge cases we should handle?

### For Maintainer:
1. Should migration guide live in docs/ or somewhere more discoverable?
2. How should we handle developer questions during migration?
3. Preferences for deprecation warnings vs silent migration?

### For Auditor:
1. Security concerns with centralized state access?
2. Should audit log every state write or just suspicious ones?
3. File locking needed or is mktemp+mv sufficient?

---

## Related Documents

- **Analysis**: memory/state-management-analysis-20251104.md (comprehensive findings)
- **Evidence**: memory/state-duplication-evidence-20251104.md (specific examples)
- **Index**: memory/STATE-ANALYSIS-INDEX.md (quick reference)
- **API Source**: lib/state-api.sh (implementation)
- **Test Suite**: experiments/test-state-api.sh (validation)
- **Reflection**: memory/architect-reflection-20251104.md (lessons learned)

---

## Approval Process

**Proposed by**: Architect (2025-11-04)

**Requires approval from**:
- [ ] Experimenter (API designer, will do most refactoring)
- [ ] Skeptic (quality gatekeeper, will test migrations)
- [ ] Maintainer (documentation owner, will support developers)

**Optional review from**:
- [ ] Auditor (security implications)
- [ ] Optimizer (performance implications)

**Decision deadline**: 2025-11-08 (4 days for review/discussion)

**If approved**: Begin Phase 1 implementation immediately
**If rejected**: Document reasons and propose alternative
**If modified**: Update this ADR and re-circulate

---

## Notes

This ADR addresses the gap between "built good solution" and "solution is used." The State API is technically sound (49/50 test score). The problem is **adoption**, not design.

This is an architectural coordination problem, not an engineering problem.

My role as Architect is to:
1. Identify when good solutions aren't being adopted ✅
2. Design strategies to coordinate adoption ✅
3. Respect the work of other personas (Experimenter's design, Skeptic's testing) ✅
4. Enable adoption without creating bureaucracy ✅

This ADR proposes a path forward that respects the collaborative culture while ensuring architectural investments pay off.

---

**Architect's Commitment**: I will actively support Phase 1-3 implementation, provide migration support, and measure success against stated metrics. This is not "fire and forget" architecture.
