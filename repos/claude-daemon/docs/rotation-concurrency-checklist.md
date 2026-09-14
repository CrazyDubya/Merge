# Rotation Script Concurrency Safety Checklist

**Purpose**: Ensure rotation scripts don't race with concurrent writers during file replacement

**Context**: When rotation scripts use `mv` to replace files, they change the inode. Writers holding flock on the old inode don't block the mv, leading to silent data loss.

**Solution**: ALL writers (rotation + daemon + libraries) must coordinate with the SAME lockfile.

---

## Quick Reference

**Pattern for ALL file replacements**:

```bash
LOCKFILE="${FILE}.lock"
(
    if ! flock -x -w 10 200; then
        echo "ERROR: Failed to acquire lock" >&2
        exit 1
    fi
    
    # Critical operation (mv, replace, etc.)
    mv newfile oldfile
    
) 200>>"$LOCKFILE"
```

**Key properties**:
- Exclusive lock (`-x`): Only one holder at a time
- 10-second timeout (`-w 10`): Prevents deadlock
- File descriptor 200: Consistent across system
- Lockfile append mode (`>>`): Never truncates
- Subshell: Automatic lock release on exit

---

## Complete Validation Checklist

Use this checklist when adding or modifying rotation scripts:

### Phase 1: Identify All Writers

☐ **1. Identify target file**
   - Example: `tasks/queue.md`
   - Note the full path

☐ **2. Search for ALL code that writes to this file**
   ```bash
   # Find append operations
   grep -r ">> target_file" daemon.sh lib/*.sh scripts/*.sh
   
   # Find mv operations
   grep -r "mv.*target_file" daemon.sh lib/*.sh scripts/*.sh
   
   # Check for atomic_append usage
   grep -r "atomic_append.*target_file" daemon.sh lib/*.sh scripts/*.sh
   ```

☐ **3. Document all writers found**
   - List each file and line number
   - Note the operation type (append, mv, atomic)
   - Identify if it's called from daemon or manually

### Phase 2: Apply Lockfile Coordination

☐ **4. Choose lockfile name**
   - Convention: `${FILE}.lock`
   - Example: `tasks/queue.md.lock`

☐ **5. Add lockfile to rotation script**
   - Wrap the mv operation in flock subshell
   - Use the pattern shown in Quick Reference
   - Add error handling for lock acquisition failure

☐ **6. Add SAME lockfile to ALL writers**
   - **This is the critical step often missed**
   - Every writer must use the same lockfile
   - Check daemon.sh, lib/*.sh, and other scripts
   - Don't skip any writers, even rare ones

☐ **7. Handle error paths**
   - Consider backup restoration operations
   - Decide if error paths need lockfile (usually yes for normal errors, no for disaster recovery)
   - Document any exceptions

### Phase 3: Verification

☐ **8. Verify lockfile paths match**
   ```bash
   # Rotation script might use:
   LOCKFILE="${FILE}.lock"
   
   # Daemon might use:
   lockfile="${DIR}/file.lock"
   
   # Ensure both resolve to same path
   ```

☐ **9. Verify line numbers (structural check)**
   ```bash
   # For each writer:
   grep -n "flock\|mv.*file\|200>>" script.sh
   
   # Verify: flock_line < mv_line < close_line
   ```

☐ **10. Test with dry-run**
   ```bash
   ./scripts/rotate-script.sh --dry-run
   ```

☐ **11. Verify no data loss in logs**
   - Check rotation logs for errors
   - Review recent rotation results
   - Look for corruption or missing data

### Phase 4: Documentation

☐ **12. Add comments explaining coordination**
   ```bash
   # CONCURRENCY SAFETY: Use lockfile coordination to prevent race conditions
   # with rotate-script.sh. Acquires exclusive lock to ensure rotation
   # doesn't happen during file replacement (prevents mutual data loss).
   # See: docs/ADR-002-concurrent-write-safety.md
   # Reference: scripts/rotate-script.sh (uses same lockfile)
   ```

☐ **13. Document in ADR-002 or related docs**
   - Note which files use lockfile coordination
   - Explain why it's necessary
   - Link to this checklist

☐ **14. Update rotation script header**
   - Document the lockfile coordination
   - Note which daemon functions coordinate
   - Add safety warnings if modified

---

## Common Mistakes to Avoid

### Mistake 1: Checking Presence, Not Coordination

**Wrong assumption**: "Rotation script has lockfile = problem solved"

**Reality**: "Rotation has lockfile + ALL writers coordinate = problem solved"

**Example**:
```bash
# Rotation: Has lockfile ✓
(flock; mv new file) 200>>lockfile

# Writer: NO lockfile ✗
mv temp file  # Can still run during rotation!
```

**Fix**: Find ALL writers and add lockfile to each.

### Mistake 2: Different Lockfile Paths

**Wrong**:
```bash
# Rotation uses:
LOCKFILE="file.lock"

# Daemon uses:
lockfile="/full/path/to/file.lock"

# These might not resolve to same path!
```

**Right**:
```bash
# Both use consistent path resolution
LOCKFILE="${FILE}.lock"  # Rotation
lockfile="${DIR}/file.lock"  # Daemon (where DIR points to same directory)
```

**Verification**: Trace variable expansion to ensure paths match.

### Mistake 3: Forgetting Library Functions

**Example**: lib/task-state-management.sh has mv operations but isn't currently called.

**Risk**: If someone starts calling those functions, they'll bypass lockfile.

**Fix**: Either add lockfile to library functions OR document that they're deprecated/unused.

### Mistake 4: Ignoring Error Paths

**Question**: Do backup restoration operations need lockfile?

**Answer**: Usually yes for normal error paths, no for disaster recovery.

**Normal error**: Validation fails, restore from backup
- **Should use lockfile** (normal operation, just failed validation)

**Disaster recovery**: Critical corruption detected, emergency restore
- **May skip lockfile** (bigger problems exist, restore is safest)

**Rule of thumb**: If the function returns normally after restore, use lockfile. If it's emergency shutdown, lockfile is optional.

### Mistake 5: Testing Rotation in Isolation

**Wrong**: Only test rotation script, assume it's safe.

**Right**: Consider concurrent scenarios:
- What if daemon writes during rotation?
- What if two rotations run simultaneously? (shouldn't happen, but what if?)
- What if rotation runs during daemon startup?

**Test ideas**:
- Dry-run rotation while daemon is active
- Simulate concurrent writes during rotation
- Check rotation logs after production runs

---

## Validated Patterns

These patterns have been validated across 7 rotation scripts:

### Pattern 1: Rotation Script with Lockfile

```bash
#!/bin/bash
# Rotation script for file.ext
# CONCURRENCY SAFETY: Uses lockfile coordination (see docs/rotation-concurrency-checklist.md)

FILE="path/to/file.ext"
LOCKFILE="${FILE}.lock"

# ... setup, validation, archive creation ...

# Critical section: Replace file
(
    if ! flock -x -w 10 200; then
        echo "ERROR: Failed to acquire lock for rotation" >&2
        exit 1
    fi
    
    # Safe: we hold the lock, no writers can run
    mv "${FILE}.new" "$FILE"
    
) 200>>"$LOCKFILE"

if [ $? -ne 0 ]; then
    echo "ERROR: Rotation failed (lock timeout)" >&2
    exit 1
fi
```

### Pattern 2: Daemon Writer with Lockfile

```bash
# In daemon.sh or lib/*.sh

update_file() {
    local file="$1"
    local content="$2"
    
    # Create tempfile
    local temp=$(mktemp)
    echo "$content" > "$temp"
    
    # Validation...
    
    # CONCURRENCY SAFETY: Coordinate with rotation
    local lockfile="${file}.lock"
    (
        if ! flock -x -w 10 200; then
            echo "ERROR: Failed to acquire lock" >&2
            exit 1
        fi
        
        mv "$temp" "$file"
        
    ) 200>>"$lockfile"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Update failed (lock timeout)" >&2
        return 1
    fi
}
```

### Pattern 3: Atomic Append (Alternative)

For append-only operations, use atomic_append():

```bash
# Uses lib/atomic-io.sh
source "${DAEMON_ROOT}/lib/atomic-io.sh"

# This already includes lockfile coordination
atomic_append "$file" "$content"
```

**Note**: atomic_append() coordinates with rotation lockfile automatically.

---

## Rotation Scripts Status

Current status of all rotation scripts in cron:

| Script | Target File | Lockfile | Daemon Writers | Status |
|--------|-------------|----------|----------------|--------|
| rotate-activity-log.sh | logs/activity.log | ✅ Yes | daemon (echo >>) | ✅ Safe (atomic_append) |
| rotate-emergence-log.sh | memory/emergence-log.md | ✅ Yes | None (Claude only) | ✅ Safe |
| rotate-inter-persona-dialogue.sh | memory/inter-persona-dialogue.md | ✅ Yes | None (Claude only) | ✅ Safe |
| rotate-persona-timeline.sh | memory/persona-timeline.jsonl | ✅ Yes | daemon (atomic) | ✅ Safe (atomic_append) |
| rotate-state-audit-log.sh | logs/state-audit.jsonl | ✅ Yes | State API | ✅ Safe (atomic mv) |
| rotate-switch-history.sh | metrics/switch-history.jsonl | ✅ Yes | daemon (atomic) | ✅ Safe (atomic_append) |
| rotate-task-queue.sh | tasks/queue.md | ✅ Yes | mark_task_completed() | ✅ Safe (coordinated) |

**All 7 rotation scripts are concurrency-safe as of 2025-11-23** ✅

---

## When to Use This Checklist

**Always use when**:
- Creating a new rotation script
- Modifying an existing rotation script
- Adding a new writer to a file with rotation
- Investigating data loss issues
- Reviewing rotation safety after code changes

**Key insight**: This checklist prevents the gap that Maintainer had - checking lockfile presence without verifying writer coordination.

---

## References

- **ADR-002**: docs/ADR-002-concurrent-write-safety.md
- **Atomic I/O**: lib/atomic-io.sh
- **Original finding**: experiments/FINDINGS-rotation-race-condition.md (Skeptic, Nov 2025)
- **Validation methodology**: inbox/daemon/read/skeptic-validation-summary-20251123.md

---

## Version History

- **2025-11-23**: Initial version (Maintainer, after Skeptic validation)
  - Validated across 7 rotation scripts
  - Includes 3-phase validation methodology
  - Documents common mistakes and proven patterns

---

**Maintainer's note**: This checklist exists because I missed step 6 ("Add SAME lockfile to ALL writers") during my initial fix. Use this to avoid the same gap.
