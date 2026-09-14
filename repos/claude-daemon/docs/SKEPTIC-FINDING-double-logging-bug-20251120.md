# SKEPTIC FINDING: Double-Logging Bug Masking Sed Error Root Cause

**Date:** 2025-11-20
**Investigator:** Skeptic
**Severity:** High (affects debugging, masks true root cause)
**Related:** Sed escaping bug investigation by Maintainer

## Summary

Found fundamental flaw in Maintainer's sed bug investigation. The "double completion attempts" hypothesis is based on misreading activity.log due to an unrecognized double-logging bug.

## The Double-Logging Bug

### Location

`daemon.sh` lines 1471, 1478, 1484, 1491:

```bash
execute_task_action "$current_persona" 2>&1 | tee -a "$ACTIVITY_LOG"
execute_reflection_action "$current_persona" 2>&1 | tee -a "$ACTIVITY_LOG"
execute_task_action "$current_persona" 2>&1 | tee -a "$ACTIVITY_LOG"  # fallback
execute_conversation_check "$current_persona" 2>&1 | tee -a "$ACTIVITY_LOG"
```

### Root Cause

The `log()` function (line 120) ALREADY writes to `$ACTIVITY_LOG`:

```bash
log() {
    local level="$1"
    shift
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] [$level] $*"
    echo "$msg" >&2  # Console output (stderr)
    atomic_append "$ACTIVITY_LOG" "$msg"  # File write
}
```

When combined with `tee -a "$ACTIVITY_LOG"`, this creates duplicate writes:

1. `log()` writes to stderr and activity.log
2. `tee -a` captures stderr and writes to activity.log again

### Evidence

From activity.log, EVERY log message appears twice:

```
[2025-11-09 02:00:02] [INFO] [maintainer] Starting task: [ALL PERSONAS]...
[2025-11-09 02:00:02] [INFO] [maintainer] Starting task: [ALL PERSONAS]...
[2025-11-09 02:00:02] [DEBUG] Using existing session ID: 199dd104...
[2025-11-09 02:00:02] [DEBUG] Using existing session ID: 199dd104...
[2025-11-09 02:03:23] [INFO] [maintainer] Task completed successfully
[2025-11-09 02:03:23] [INFO] [maintainer] Task completed successfully
```

This pattern occurs throughout the entire log file, not just for success messages.

## Impact on Sed Bug Investigation

### Maintainer's Misinterpretation

From `inbox/human/unread/response-20251120-181644-from-maintainer.md`:

> From activity log:
> ```
> [2025-11-20 14:44:09] [INFO] [experimenter] Task completed successfully
> [2025-11-20 14:44:09] [INFO] [experimenter] Task completed successfully
> sed: -e expression #1, char 937: Invalid preceding regular expression
> ```
>
> Notice: "Task completed successfully" appears **TWICE**, then the sed error.
>
> **Hypothesis:** The personas may be attempting to mark an ALREADY COMPLETED task as complete again.

### Why This Is Wrong

1. **Every log message appears twice**, not just success messages
2. This is a logging artifact, not evidence of double execution
3. `mark_task_completed()` is called exactly once (daemon.sh line 880)
4. The sed error is legitimate, not caused by double-marking

### Actual Sed Error Pattern

```
[timestamp] [INFO] [persona] Task completed successfully  <-- from log()
[timestamp] [INFO] [persona] Task completed successfully  <-- from tee -a
sed: -e expression #1, char 937: Invalid preceding...     <-- stderr from sed subprocess
ERROR: mark_task_completed produced corrupted file        <-- stderr from validation
```

The sed error is happening on the **first and only** call to `mark_task_completed()`.

## Revised Sed Bug Hypothesis

**Task Text Mismatch Theory:**

The `clean_task` variable doesn't exactly match the text in `queue.md`, causing sed pattern to find no match, producing empty output, triggering corruption detection.

**Why Tests Pass:**
- Tests use exact text from queue.md
- No time delay, no concurrent modification
- Controlled environment

**Why Production Fails:**
- Task text captured when selected
- Time delay before completion
- Potential formatting differences
- Whitespace/unicode normalization issues

**Supporting Evidence:**
1. Different character positions (937, 8834, 8888) → Different task content
2. Intermittent failures → Some tasks match, some don't
3. Tests work perfectly → Exact matches work
4. Empty output → Pattern mismatch → sed produces nothing

## Recommended Fixes

### Fix 1: Double-Logging Bug

**Remove the tee -a** since log() already writes to activity.log:

```bash
# Current (broken)
execute_task_action "$current_persona" 2>&1 | tee -a "$ACTIVITY_LOG"

# Fixed
execute_task_action "$current_persona"
```

Or if stderr visibility is needed:

```bash
execute_task_action "$current_persona" 2>&1 | tee /dev/tty
```

### Fix 2: Sed Bug Debug Logging

Add to `mark_task_completed()` before sed command:

```bash
# Debug: Log what we're searching for
echo "DEBUG: Task to mark: <<<$task>>>" >&2
echo "DEBUG: Escaped pattern: <<<$escaped_task>>>" >&2
echo "DEBUG: Current uncompleted tasks:" >&2
grep "^- \[ \] " "$TASKS_DIR/queue.md" | head -5 >&2 || true
```

This will reveal text mismatches causing sed failures.

## Lessons Learned

1. **Verify infrastructure assumptions** - Don't assume log patterns mean what they appear to mean
2. **Trace data flow completely** - log() → stderr → tee → file creates duplication
3. **Question obvious conclusions** - "Appears twice" doesn't always mean "executed twice"
4. **Test logging mechanisms** - Infrastructure bugs can mask application bugs

## Why This Matters

- **Wasted investigation time**: 75 minutes on wrong hypothesis
- **Wrong fix proposed**: Pre-check for completed tasks won't solve text mismatch
- **Real issue masked**: Task text mismatch not investigated
- **Future debugging**: Double logs make all debugging harder

This is exactly why the Skeptic persona exists - to catch analytical errors even in thorough, well-intentioned work.

## Action Items

1. Fix double-logging bug (remove tee -a or change target)
2. Add debug logging to mark_task_completed()
3. Capture next sed failure with full context
4. Analyze task text differences
5. Design proper fix based on actual root cause

## References

- Original investigation: `inbox/human/unread/response-20251120-181644-from-maintainer.md`
- Skeptic's correction: `inbox/human/unread/msg-skeptic-sed-bug-critical-correction-20251120.md`
- Code locations:
  - `daemon.sh:114-122` - log() function
  - `daemon.sh:880` - mark_task_completed() call
  - `daemon.sh:1471,1478,1484,1491` - tee -a usage
  - `daemon.sh:651-657` - sed escaping logic
  - `daemon.sh:631-690` - mark_task_completed() function

---

**Skeptic**
*Always verify your assumptions about the infrastructure.*
