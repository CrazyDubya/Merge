# ADR-002: Concurrent Write Safety Strategy

**Status**: Proposed
**Date**: 2025-11-08
**Author**: Architect
**Related Issues**: State API audit drops (0.044%), watchdog false positives

---

## Context

### The Problem

Experimenter discovered 0.044% audit entry loss during thrashing (46/104,162 switches) - ALL losses occurred during high-concurrency periods (12+ switches/second). Investigation revealed **systemic architectural issue**: **no concurrency model** for shared file writes.

**Current Pattern**: 16+ scripts use `echo >> file` for append operations without coordination:
- `lib/state-audit.sh:51` - State audit logging (0.044% loss discovered)
- `daemon.sh:129,490,508,525,542,565` - Switch history, timeline writes
- `lib/post-restart-check.sh:89` - Activity log
- `daemon.sh:657,1093` - Task completion, timeline overrides
- Plus ~10 more across scripts/, experiments/, etc.

### Why This Matters

**Architectural Principle Violated**: **Data Integrity**

Without coordination, concurrent `echo >>` operations can:
1. **Lose data** (write failures under contention)
2. **Corrupt data** (interleaved lines, partial writes)
3. **Silent failures** (many use `|| true` for availability)

This is acceptable for **non-critical logs** (activity.log), but **unacceptable** for:
- **Audit trails** (state-audit.jsonl) - compliance/security requirement
- **State history** (switch-history.jsonl) - metrics/analysis foundation
- **Timeline** (persona-timeline.jsonl) - behavioral evolution tracking

### Current Architecture Gap

**ARCHITECTURE.md line 131** states:
> **Transaction Safety**: mktemp + atomic mv pattern for all writes.

But this only covers **state file updates** (read-modify-write). It does **not** cover **append operations** (concurrent writes to shared logs).

**Gap**: No documented concurrency model for append-only data structures.

---

## Decision

### Strategy: Risk-Based Write Coordination

Implement **tiered approach** based on data criticality:

**Tier 1 - CRITICAL** (flock required):
- Audit logs (compliance/security)
- State history (system integrity)
- Timeline (behavioral evolution)

**Tier 2 - IMPORTANT** (flock recommended):
- Metrics (accuracy valuable but not critical)
- Message inboxes (delivery matters)

**Tier 3 - OPTIONAL** (no coordination):
- Debug logs (best-effort acceptable)
- Temporary test outputs
- Experimental data collection

### Implementation: Atomic Append Library

Create `lib/atomic-io.sh` with reusable functions:

```bash
#!/bin/bash
# Atomic I/O operations for concurrent access

# Atomic append with flock
atomic_append() {
    local file="$1"
    local content="$2"
    local timeout="${3:-5}"  # seconds

    local lockfile="${file}.lock"

    # Ensure directory exists
    mkdir -p "$(dirname "$file")" 2>/dev/null || true

    # Acquire exclusive lock with timeout
    (
        flock -x -w "$timeout" 200 || {
            echo "ERROR: Failed to acquire lock on $file (timeout ${timeout}s)" >&2
            return 1
        }

        # Append while holding lock
        echo "$content" >> "$file" || {
            echo "ERROR: Failed to write to $file" >&2
            return 1
        }
    ) 200>>"$lockfile"

    return $?
}

# Atomic read (ensures no partial reads during writes)
atomic_read() {
    local file="$1"
    local lockfile="${file}.lock"

    [ -f "$file" ] || {
        echo "ERROR: File $file does not exist" >&2
        return 1
    }

    # Shared lock (multiple readers OK, blocks writers)
    (
        flock -s 200
        cat "$file"
    ) 200>>"$lockfile"
}

# Batch append (multiple lines atomically)
atomic_append_batch() {
    local file="$1"
    shift
    local lines=("$@")
    local lockfile="${file}.lock"

    mkdir -p "$(dirname "$file")" 2>/dev/null || true

    (
        flock -x 200 || return 1
        for line in "${lines[@]}"; do
            echo "$line" >> "$file" || return 1
        done
    ) 200>>"$lockfile"
}
```

### Migration Plan

**Phase 1: Library + Critical Systems** (Week 1)
- Create `lib/atomic-io.sh`
- Migrate Tier 1 (audit, history, timeline)
- Add integration tests (concurrent write verification)
- Validation: 24h production monitoring, zero data loss

**Phase 2: Important Systems** (Week 2-3)
- Migrate Tier 2 (metrics, inbox)
- Performance testing (ensure acceptable overhead)
- Document patterns in docs/coding-standards.md

**Phase 3: Standardization** (Week 4)
- Add linting rules (detect naked `echo >>`)
- Update ADR-001 State API adoption strategy
- Archive old atomic-write experiments

**Phase 4: Optional Systems** (Future)
- Evaluate Tier 3 on case-by-case basis
- Some may benefit (minimal cost), others won't

---

## Rationale

### Why flock?

**Alternatives Considered**:

1. **Advisory locking (flock)** ✅ CHOSEN
   - POSIX standard, widely available
   - Process-level coordination
   - Low overhead (<1ms typical)
   - Works across all UNIX systems
   - Automatic cleanup on process death

2. **Mandatory locking (lockfile command)**
   - Requires external dependency
   - More complex error handling
   - No significant benefits over flock

3. **Message queue (syslog, journald)**
   - Different architecture (centralized logging)
   - Loses flexibility (can't grep/jq raw files)
   - Requires service infrastructure
   - Overkill for this use case

4. **Database (SQLite)**
   - Heavyweight dependency
   - Changes data format (no longer JSONL)
   - Loses git-friendly diffs
   - Overkill for append-only logs

5. **Single-writer pattern (all writes through daemon)**
   - Requires architectural refactor
   - Breaks modularity (scripts can't write independently)
   - Complex coordination (message passing)
   - Doesn't match current distributed architecture

**Decision**: flock provides **sweet spot** of simplicity, safety, and minimal architectural change.

### Why Tiered Approach?

**Principle**: Apply constraints proportional to criticality.

- **Not all data is equal**: Debug logs != audit trails
- **Performance matters**: Unnecessary coordination wastes resources
- **Pragmatic architecture**: Perfect safety everywhere is expensive

**Counter-argument**: "Why not make everything Tier 1?"
- flock adds ~1ms per write (acceptable for critical, wasteful for verbose logs)
- Complexity costs (more code, more tests, more maintenance)
- Diminishing returns (debug logs SHOULD be best-effort)

### Why Not Fix in State API Only?

Experimenter's proposal (flock in `lib/state-audit.sh:51`) solves the **symptom**, not the **architecture**.

**Problems with point solutions**:
1. Next developer will write `echo >> newlog.jsonl` (repeats mistake)
2. switch-history.jsonl has same vulnerability (not yet discovered)
3. No reusable pattern (everyone reinvents flock)
4. No documentation (knowledge doesn't spread)

**Architectural solution**:
1. Library function (`atomic_append`) - reusable
2. ADR document (this file) - explains reasoning
3. Tiered guidelines - helps decisions
4. Migration plan - fixes existing code
5. Standards update - prevents future issues

**Principle**: "Fix the class of bugs, not just this bug."

---

## Consequences

### Positive

1. **Data Integrity**: Zero loss in critical logs (audit, history, timeline)
2. **Architectural Clarity**: Explicit concurrency model documented
3. **Reusability**: Pattern available for all future append operations
4. **Standards**: Clear guidelines for "when to coordinate writes"
5. **Simplicity**: Library abstracts complexity, call sites stay simple
6. **Performance**: Only coordinate where needed (tiered approach)

### Negative

1. **Migration Cost**: ~16 scripts need updates (estimated 8-12 hours)
2. **Performance Overhead**: ~1ms per critical write (acceptable)
3. **Lockfile Proliferation**: Each JSONL gets .lock sibling (minor clutter)
4. **Complexity**: New concept for contributors to learn
5. **Testing Burden**: Need concurrency tests (complex to write)

### Neutral

1. **Maintenance**: Long-term maintenance lower (fewer bugs) but initial higher (migration)
2. **Architecture Debt**: Pays down debt (fixes gap) but adds complexity (new library)

---

## Open Questions

### Q1: Should switch-history.jsonl use flock?

**Data**: 100% coverage observed (104,162/104,162 entries), but uses same `echo >>` pattern.

**Hypothesis**: Lucky timing (daemon writes most entries sequentially) OR separate code paths avoid contention.

**Decision**: YES, migrate to flock.
- **Rationale**: 100% coverage today doesn't guarantee 100% tomorrow. Thrashing demonstrated concurrent writes CAN happen. State history is Tier 1 (system integrity). Cost is low (library already exists).

### Q2: What's acceptable performance overhead?

**Benchmark needed**: Measure flock overhead in production conditions.

**Hypothesis**: <1ms per write typical, <10ms worst case (acceptable for Tier 1).

**Decision**: Proceed with migration, monitor performance, rollback if >10ms average observed.

### Q3: Should we extend State API with atomic operations?

**Proposal**: Add `state_audit_atomic()` that uses atomic_append internally.

**Pro**: Single source of truth, enforces consistency
**Con**: Couples State API to I/O strategy, reduces flexibility

**Decision**: YES, but as thin wrapper.
```bash
# lib/state-api.sh
state_audit() {
    # ... build entry ...
    atomic_append "$AUDIT_LOG" "$entry" || return 1
}
```

This keeps State API responsible for **what** to log, atomic-io responsible for **how** to write safely.

### Q4: How to prevent regressions?

**Linting**: Add check to catch naked `echo >> *.jsonl`

```bash
# In CI or pre-commit:
if grep -r "echo.*>>.*\.jsonl" lib/ hooks/ daemon.sh | grep -v atomic_append; then
    echo "ERROR: Found uncoordinated writes to JSONL files"
    echo "Use atomic_append from lib/atomic-io.sh for Tier 1/2 data"
    exit 1
fi
```

**Documentation**: Update docs/coding-standards.md with append guidelines.

**Code Review**: Auditor reviews all new append operations.

---

## Implementation Notes

### Testing Strategy

**Unit Tests** (`experiments/test-atomic-io.sh`):
- Single process writes
- Error handling (lock timeout, write failure)
- Batch append correctness

**Integration Tests** (`experiments/test-concurrent-append.sh`):
- 20 processes, 50 writes each (1000 total)
- Validate: exactly 1000 lines written, zero corruption
- Compare: with flock vs without flock (demonstrate improvement)

**Production Validation**:
- Deploy to production
- Monitor for 24 hours
- Measure: audit coverage, switch history coverage, timeline completeness
- Success criteria: 100% coverage maintained under thrashing

### Rollback Plan

If issues discovered:

1. **Performance degradation**: Increase lock timeout, optimize library
2. **Data corruption**: Debug race condition, fix library
3. **Deployment failure**: Rollback to previous version (git revert)
4. **Unacceptable overhead**: Demote some Tier 1 → Tier 2, accept best-effort

**Safety**: All migrations add coordination (can't make integrity worse).

### Documentation Updates

**Files to update**:
1. `ARCHITECTURE.md` - Add "Concurrency Model" section
2. `docs/coding-standards.md` - Add append operation guidelines
3. `ADR-001-state-api-adoption.md` - Reference ADR-002 for I/O safety
4. `lib/state-api.sh` - Add comments referencing atomic_append
5. `README.md` - Link to ADR-002 in architecture section

---

## Alternatives Considered

### Alternative 1: Accept Data Loss

**Argument**: 0.044% loss is acceptable for audit trail.

**Counter-argument**:
- Audit logs exist for compliance/security
- "Mostly accurate" audit = useless audit
- Loss rate could be higher (thrashing was extreme case, but not unique)
- Fix is simple (flock), why accept unnecessary data loss?

**Decision**: REJECTED. Audit integrity is binary (either reliable or useless).

### Alternative 2: Single-Writer Pattern

**Argument**: Route all writes through daemon (centralized coordination).

**Counter-argument**:
- Major architectural refactor (weeks of work)
- Breaks modularity (scripts can't log independently)
- Requires message passing infrastructure
- Adds latency (write queue, coordination overhead)
- More complex (more failure modes)

**Decision**: REJECTED. Excessive complexity for append-only logs.

### Alternative 3: Lock-Free Data Structures

**Argument**: Use atomic CAS operations, lock-free queues.

**Counter-argument**:
- Not available in bash (requires C/compiled code)
- Overkill for this performance profile
- Complex to implement correctly
- Debugging nightmare (subtle race conditions)

**Decision**: REJECTED. flock is simpler and sufficient.

### Alternative 4: Do Nothing (Wait for More Evidence)

**Argument**: Only one instance found (audit), maybe not systemic.

**Counter-argument**:
- Pattern is systemic (16+ uses of `echo >>`)
- Same vulnerability exists in switch-history.jsonl (not yet observed)
- Thrashing demonstrated concurrent writes happen
- Architectural gap exists regardless of whether bugs manifest

**Decision**: REJECTED. Fix the architecture, don't wait for more bugs.

---

## Success Metrics

### Quantitative

1. **Data Integrity**: 100% audit coverage maintained during thrashing (validated via validation scripts)
2. **Performance**: <10ms average append time (measure in production)
3. **Coverage**: 100% Tier 1 systems migrated within 1 week
4. **Zero Regressions**: No new data corruption issues introduced

### Qualitative

1. **Developer Experience**: Library is easy to use (adoption without resistance)
2. **Maintainability**: Future developers understand when/how to coordinate writes
3. **Architectural Clarity**: Concurrency model clearly documented and understood

### Validation Timeline

- **Day 1**: Library implemented, tested (unit + integration)
- **Day 2-3**: Tier 1 migrations (audit, history, timeline)
- **Day 4**: Production deployment, monitoring begins
- **Day 5-7**: 72h production validation (100% coverage confirmed)
- **Week 2**: Tier 2 migrations, performance benchmarks
- **Week 3**: Documentation updates, standards enforcement
- **Week 4**: Retrospective, success metrics evaluation

---

## References

- **Triggering Issue**: docs/state-audit-thrashing-hypothesis.md (Experimenter, 2025-11-08)
- **State API Context**: docs/ADR-001-state-api-adoption.md (Architect, 2025-11-04)
- **Watchdog Issue**: docs/watchdog-false-positive-bug-20251108.md (Experimenter, 2025-11-08)
- **Architecture Doc**: docs/ARCHITECTURE.md (Architect, 2025-11-04)
- **POSIX flock**: man 1 flock (standard Unix utility)

---

## Approval

**Proposer**: Architect
**Required Reviewers**:
- [ ] **Auditor**: Security implications of locking strategy
- [ ] **Experimenter**: Technical feasibility, implementation approach
- [ ] **Skeptic**: Challenge assumptions, find edge cases

**Approval Criteria**:
1. Security review (Auditor confirms Tier 1 classification correct)
2. Technical review (Experimenter confirms flock is right choice)
3. Critical review (Skeptic finds no fatal flaws)

**Timeline**: 48 hours for reviews, proceed if no blocking concerns.

---

## Appendix: Code Examples

### Before (Vulnerable)

```bash
# lib/state-audit.sh:51 (BEFORE)
echo "$entry" >> "$AUDIT_LOG" || {
    echo "WARNING: Failed to write audit log entry" >&2
    return 1
}
```

**Problem**: Concurrent writes can fail/corrupt during thrashing.

### After (Safe)

```bash
# lib/state-audit.sh:51 (AFTER)
source "${DAEMON_ROOT}/lib/atomic-io.sh"

atomic_append "$AUDIT_LOG" "$entry" || {
    echo "WARNING: Failed to write audit log entry" >&2
    return 1
}
```

**Benefit**: flock ensures atomic writes, zero data loss.

### Usage Pattern

```bash
# For critical logs (Tier 1)
source "${DAEMON_ROOT}/lib/atomic-io.sh"
atomic_append "${DAEMON_ROOT}/logs/state-audit.jsonl" "$audit_entry"

# For batch operations
atomic_append_batch "${DAEMON_ROOT}/logs/metrics.jsonl" \
    "$metric1" \
    "$metric2" \
    "$metric3"

# For reads during heavy writes
atomic_read "${DAEMON_ROOT}/logs/state-audit.jsonl" | jq '.operation'
```

---

## Operations on Coordinated Files

**Date Added**: 2025-11-09
**Context**: Race condition discovered in log rotation (0.27% data loss)
**Issue**: experiments/FINDINGS-rotation-race-condition.md

### The Problem

ADR-002 originally focused on **concurrent writers** - multiple processes appending to the same file simultaneously. This addresses writer-writer conflicts.

**What we missed**: **Operations that interact with coordinated files** - rotation, backup, archival, analysis, monitoring. These are **meta-operations** that operate on the file itself, not just append to it.

**Result**: Log rotation scripts bypassed atomic_append's lock mechanism → 0.27% data loss when rotation occurred during active writes.

### Root Cause: Lock Scope Mismatch

**What writers do**:
```bash
# atomic_append acquires exclusive lock
(
    flock -x -w 5 200
    echo "$content" >> "$LOG_FILE"
) 200>>"${LOG_FILE}.lock"
```

**What rotation did** (BROKEN):
```bash
# NO LOCK COORDINATION
gzip -c "$LOG_FILE" > "$archive_file"  # Reads file without lock
mv "$new_file" "$LOG_FILE"             # Replaces file without lock
```

**The race condition**:
```
T0: Writer acquires lock on LOG_FILE.lock
T1: Writer appending to LOG_FILE
T2: Rotation starts: gzip reads LOG_FILE (NO LOCK!)
T3: Rotation captures partial state
T4: Writer finishes, releases lock
T5: Rotation: mv replaces LOG_FILE (NEW INODE!)
T6: Next writer acquires lock to OLD INODE (orphaned)
T7: Write goes to deleted file → DATA LOSS
```

**Key insight**: Lock protects the *inode*, not the *path*. When `mv` replaces the file, it creates a new inode. Writers holding locks to the old inode continue writing to an orphaned (deleted) file.

### The Architectural Gap

ADR-002 defined three tiers for **writers**:
- Tier 1: Critical data → atomic_append required
- Tier 2: Important data → atomic_append recommended
- Tier 3: Optional data → no coordination

**What's missing**: Requirements for **operations on Tier 1/2 files**.

Operations include:
- **Log rotation** (gzip + mv)
- **Backup/archival** (copying for backup)
- **Analysis scripts** (reading for processing)
- **Monitoring dashboards** (reading for display)
- **Export utilities** (extracting data)

These operations **must coordinate** with the same locking mechanism that writers use.

### The Solution: Coordination Requirement

**New architectural principle**:

> **Any operation that reads or modifies a Tier 1/2 file must coordinate through the same locking primitives that writers use.**

This applies to **all operations**, not just writes:

**Operation Type** | **Lock Type** | **Example**
---|---|---
**Read-only** | Shared (`flock -s`) | Backup, analysis, monitoring
**Modify** | Exclusive (`flock -x`) | Rotation, truncation, migration
**Write** | Exclusive (`flock -x`) | Append operations (existing)

### Implementation Patterns

#### Pattern 1: Read-Only Operations (Shared Lock)

```bash
# Backup script - multiple readers OK, blocks writers
source "${DAEMON_ROOT}/lib/atomic-io.sh"

(
    flock -s 200  # Shared lock (multiple readers allowed)
    tar czf backup.tar.gz "$LOG_FILE"
) 200>>"${LOG_FILE}.lock"
```

**Why shared lock**:
- Multiple backup/analysis scripts can run simultaneously
- Writers blocked during read (prevents torn reads)
- Readers block writers (prevents reading during modification)

#### Pattern 2: Modify Operations (Exclusive Lock)

```bash
# Log rotation - must have exclusive access
source "${DAEMON_ROOT}/lib/atomic-io.sh"

(
    flock -x -w 30 200 || {  # Exclusive lock with timeout
        echo "ERROR: Could not acquire lock for rotation"
        exit 1
    }

    # Now safe - no writers can interfere
    gzip -c "$LOG_FILE" > "$archive_file"

    # Validate archive before destroying data
    gunzip -t "$archive_file" || {
        echo "ERROR: Archive validation failed"
        exit 1
    }

    # Replace with new file
    mv "$new_file" "$LOG_FILE"

    # Lock releases automatically when subshell exits
) 200>>"${LOG_FILE}.lock"
```

**Why exclusive lock**:
- No readers or writers during modification
- Atomic replacement while holding lock
- Validation before committing changes

#### Pattern 3: Using atomic_file_operation

For custom operations, use the helper function:

```bash
source "${DAEMON_ROOT}/lib/atomic-io.sh"

# Execute custom command on file with automatic locking
atomic_file_operation "$LOG_FILE" tail -n 100
atomic_file_operation "$LOG_FILE" jq '.operation'
```

This acquires appropriate lock, runs command, releases lock.

### Anti-Patterns (What NOT To Do)

**❌ WRONG**: Direct file access without coordination
```bash
# BROKEN: Bypasses lock coordination
gzip -c state-audit.jsonl > archive.gz
cat activity.log | grep ERROR
tail -f switch-history.jsonl
```

**Why broken**: These operations don't coordinate with atomic_append. They can:
- See torn reads (partial lines, corrupted data)
- Read during rotation (inconsistent state)
- Race with writers (undefined behavior)

**✅ CORRECT**: Coordinate through lock primitives
```bash
# SAFE: Uses same lock as writers
(
    flock -s 200
    gzip -c state-audit.jsonl > archive.gz
) 200>>state-audit.jsonl.lock

# OR use atomic_read helper
atomic_read activity.log | grep ERROR

# For monitoring/tailing, accept best-effort (Tier 3 behavior)
tail -f switch-history.jsonl  # OK for monitoring (not critical)
```

### When Coordination is Required

**MUST coordinate** (Tier 1/2 files):
- ✅ Log rotation scripts
- ✅ Backup operations
- ✅ Data export utilities
- ✅ Analysis scripts that process entire file
- ✅ Migration scripts
- ✅ Archival operations

**MAY skip coordination** (specific cases):
- Real-time monitoring (tail -f) - accept best-effort
- Debug/diagnostic reads - not critical
- Operations on Tier 3 files - best-effort acceptable

**When in doubt**: If the operation matters for correctness, coordinate. If it's just observability, best-effort may be acceptable.

### File List: Operations to Audit

Operations currently interacting with Tier 1/2 files:

**Log Rotation** (NOW FIXED):
- `scripts/rotate-activity-log.sh` - ✅ Now coordinates with flock
- `scripts/rotate-state-audit-log.sh` - ✅ Now coordinates with flock
- `scripts/rotate-emergence-log.sh` - ⚠️ Needs review (low write frequency)

**Monitoring** (EVALUATE):
- `scripts/maintenance-metrics.sh` - Reads log sizes (stat only, OK)
- Dashboard scripts - If they read files, need coordination

**Backup** (FUTURE):
- Any backup scripts - Must coordinate with shared locks
- Export utilities - Must coordinate with shared locks

**Analysis** (FUTURE):
- Coverage analysis scripts - Must coordinate with shared locks
- Report generation - Must coordinate with shared locks

### Integration with atomic-io.sh

The `lib/atomic-io.sh` library provides coordination primitives for all operations:

**For writes** (existing):
- `atomic_append` - Single line append
- `atomic_append_batch` - Multiple lines atomically

**For reads** (existing):
- `atomic_read` - Safe read with shared lock

**For custom operations** (existing):
- `atomic_file_operation` - Run any command with lock

**For rotation** (pattern, not function):
- Use explicit `flock` in rotation scripts
- Allows validation before committing changes

### Testing Requirements

Operations on coordinated files require **integration testing**:

**Test**: Rotation during active writes
```bash
# experiments/test-rotation-race-conditions.sh
# Spawns 10 concurrent writers
# Triggers rotation mid-stream
# Validates: zero data loss
```

**Result**: Before fix (0.27% loss), After fix (0% loss)

**Future tests needed**:
- Backup during thrashing (ensure no torn reads)
- Analysis during writes (ensure consistency)
- Multiple readers + rotation (ensure correctness)

### Consequences of This Update

**Positive**:
1. **Closes architectural gap** - Operations now explicitly covered
2. **Prevents future bugs** - Next developer knows to coordinate
3. **Documents lessons** - Race condition analysis preserved
4. **Clear guidelines** - When to use which lock type

**Negative**:
1. **Increased complexity** - More rules to remember
2. **Performance overhead** - Operations now block writers
3. **Testing burden** - Need integration tests for all operations

**Neutral**:
1. **Existing code needs audit** - Must review all operations on Tier 1/2 files
2. **Documentation updates** - Other docs must reference this section

### Success Metrics

**Quantitative**:
- Zero data loss in rotation during concurrent writes (validated)
- All operations on Tier 1/2 files coordinated (audit required)
- Zero race conditions in production (ongoing validation)

**Qualitative**:
- Developers understand when coordination required
- Future operations designed with coordination from start
- Integration gaps caught in code review

### References

- **Triggering incident**: experiments/FINDINGS-rotation-race-condition.md (2025-11-09)
- **Test suite**: experiments/test-rotation-race-conditions.sh
- **Fix commit**: [EXPERIMENTER] Critical Fix: Race condition in log rotation
- **Architect reflection**: memory/emergence-log.md (2025-11-09)
- **Experimenter reflection**: memory/emergence-log.md (2025-11-09)

### Lessons Learned

**1. Integration Points Are Vulnerable**

Unit tests passed:
- Concurrent writes (5000 operations, 0 loss) ✓
- Log rotation (successful rotation) ✓

System failed at integration point:
- Rotation DURING writes ✗

**Lesson**: Test operations **in combination**, not just in isolation.

**2. Implicit Assumptions Are Dangerous**

Implicit assumption: "Operations happen sequentially - writes finish, THEN rotation runs"

Reality: Operations can overlap.

**Lesson**: Make assumptions explicit. Document them. Test if they hold.

**3. Lock Scope Matters**

Lock protects *inode*, not *path*. File replacement (mv) creates new inode.

**Lesson**: Understand file system semantics. Lock coordination must account for file replacement.

**4. Cross-Cutting Concerns Need Explicit Design**

Rotation touches multiple subsystems:
- I/O layer (atomic-io.sh)
- Operations (rotation scripts)
- Monitoring (metrics)
- Compliance (audit logs)

**Lesson**: Cross-cutting concerns don't fit in neat boxes. They require explicit architectural design.

**5. Chaos Testing Finds Real Bugs**

0.27% loss rate is invisible in normal operation. Only chaos testing (rotation mid-write) exposed it.

**Lesson**: Adversarial testing validates assumptions. Test uncomfortable scenarios.

---

**END OF ADR-002**

---

## Addendum: Rotation Script Concurrency (2025-11-23)

### New Discovery: Rotation Race Conditions

**Context**: Skeptic discovered that rotation scripts using `mv` to replace files create race conditions with concurrent writers, even when writers use `atomic_append()`.

**Root cause**: `mv` replaces the file inode. Writers holding flock on old inode don't block the mv, leading to silent data loss.

**Solution**: ALL writers (rotation + daemon + libraries) must coordinate with the SAME lockfile.

### Complete Validation Checklist

A comprehensive checklist for rotation script concurrency safety has been created:

**Reference**: docs/rotation-concurrency-checklist.md

**Key insight**: Lockfile in rotation script is necessary but NOT sufficient. ALL writers must coordinate with that lockfile.

**Validated across**: 7 rotation scripts (all now safe as of 2025-11-23)

### Pattern for File Replacement

```bash
LOCKFILE="${FILE}.lock"
(
    if ! flock -x -w 10 200; then
        echo "ERROR: Failed to acquire lock" >&2
        exit 1
    fi
    
    mv newfile oldfile  # Safe: we hold the lock
    
) 200>>"$LOCKFILE"
```

**Critical rule**: Every `mv` operation that replaces a file with rotation must use this pattern.

### Related Issues Fixed

1. **rotate-persona-timeline.sh**: Race condition with atomic_append() writers (Skeptic finding, Nov 2025)
2. **rotate-emergence-log.sh**: Race condition with Claude Code sessions (Maintainer fix, Nov 2025)
3. **rotate-inter-persona-dialogue.sh**: Race condition with Claude Code sessions (Maintainer fix, Nov 2025)
4. **rotate-task-queue.sh**: Incomplete coordination - mark_task_completed() missing lockfile (Skeptic validation finding, Maintainer completion, Nov 2025)

**Status**: All rotation scripts now coordinate properly with writers.

**Lessons learned**:
- Check lockfile presence AND writer coordination
- Find ALL writers, not just obvious ones
- Validate each case individually (don't generalize from partial success)

---

**END OF ADR-002 (Updated 2025-11-23)**
