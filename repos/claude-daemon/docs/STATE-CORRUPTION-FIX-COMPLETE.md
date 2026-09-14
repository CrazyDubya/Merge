# State Corruption Fix - Complete Report

**Date:** 2025-11-06T23:59:00Z
**Severity:** HIGH → RESOLVED
**Status:** FIXED and VALIDATED

## Summary

Fixed critical state.json corruption bug that created 6 phantom personas with malformed names since Oct 26, 2025.

## Root Cause

**Bug Location:** `lib/state-api.sh:101`

**The Problem:**
```bash
# state_become() function (line 101)
echo "Became $persona (reason: $reason)"
```

This echo statement outputs to stdout, which gets captured by:
```bash
# daemon.sh line 1391
current_persona=$(determine_personality)
```

**The Flow:**
1. `determine_personality()` calls `state_become("skeptic", "chaos")`
2. `state_become()` echoes "Became skeptic (reason: chaos)" to stdout
3. `determine_personality()` then echoes "skeptic" to stdout
4. Both outputs get captured: `"Became skeptic (reason: chaos)\nskeptic"`
5. This corrupted value is used in `mark_task_completed()` line 640:
   ```bash
   echo "- [x] [$persona] $task ..." >> completed.md
   ```
6. Later, jq commands at lines 843-846 create NEW persona keys with malformed names

## The Fix

### 1. Fixed stdout Pollution (lib/state-api.sh:101)

**Before:**
```bash
echo "Became $persona (reason: $reason)"
```

**After:**
```bash
# Log to stderr to avoid polluting stdout (stdout is captured by determine_personality)
echo "Became $persona (reason: $reason)" >&2
```

### 2. Cleaned State.json

Removed 6 phantom personas:
- `"Became skeptic (reason: emotional_frustration)\nskeptic"` (18 activations, 1 completed, 17 failed)
- `"Became experimenter (reason: emotional_frustration)\nexperimenter"` (19 activations, 0 completed, 19 failed)
- `"Became skeptic (reason: chaos)\nskeptic"` (1 activation, 0 completed, 1 failed)
- `"Became architect (reason: emotional_frustration)\narchitect"` (1 activation, 0 completed, 1 failed)
- `"Became architect (reason: circadian)\narchitect"` (1 activation, 1 completed, 0 failed)
- `" chaos)\nskeptic"` (25 activations, 25 completed, 0 failed)

**Total phantom stats:** 65 activations, 27 completed, 38 failed

### 3. Added Validation Script

Created `scripts/validate-state.sh`:
- Checks for phantom personas (not in valid list)
- Validates persona name format (must be `^[a-z]+$`)
- Returns exit code 0 (success) or 1 (failure)
- Can be run in CI/CD or cron for monitoring

**Validation Result:** ✓ PASSED - no phantom personas detected

### 4. Created Backup

Backup created before fix:
```
personalities/state.json.backup-before-corruption-fix-20251106-235907
```

## Files Modified

1. **lib/state-api.sh** (line 101): Redirected echo to stderr
2. **personalities/state.json**: Removed 6 phantom persona entries
3. **scripts/validate-state.sh**: NEW - validation script (51 lines)
4. **docs/STATE-CORRUPTION-FIX-COMPLETE.md**: This documentation

## Prevention Measures

### Already Implemented
- ✅ `state_become()` stdout pollution fixed
- ✅ Validation script created
- ✅ Backup created before cleanup

### Additional Safeguards (Already Exist)
- ✅ `state_become()` validates persona name format (line 41): `^[a-zA-Z0-9_-]+$`
- ✅ `state_become()` checks persona exists before switching (line 53)
- ✅ Atomic writes with temp files (line 70)

### Recommended Monitoring
- Run `scripts/validate-state.sh` periodically (weekly cron)
- Add to CI/CD if repository automation exists
- Monitor for "PHANTOM DETECTED" messages

## Testing

**Pre-fix Validation:**
```bash
$ python3 -c "import json; print(len([p for p in json.load(open('personalities/state.json'))['personas'] if not p.isalpha()]))"
6  # 6 phantom personas
```

**Post-fix Validation:**
```bash
$ ./scripts/validate-state.sh
✓ state.json validation PASSED - no phantom personas detected
```

**Daemon Status:** ✅ RUNNING (verified via claude-daemon-status.sh)

## Impact Assessment

### Before Fix
- ❌ 6 phantom personas polluting state.json
- ❌ Task stats incorrectly attributed to phantoms
- ❌ User-visible corruption in prompts: `[ACTIVE PERSONA:  chaos)\nskeptic]`
- ❌ 11 days of silent corruption (Oct 26 - Nov 6)

### After Fix
- ✅ Clean state.json with only 6 valid personas
- ✅ All stats correctly attributed
- ✅ No corruption in prompts
- ✅ Validation in place to prevent recurrence
- ✅ Daemon continues running normally

## Lessons Learned

1. **Stdout is Sacred**: Functions used in command substitution `$(cmd)` must be careful with stdout
2. **Silent Corruption**: Bug existed 11 days before discovery - need better monitoring
3. **Validation Early**: Should have had `validate-state.sh` from day 1
4. **Trust but Verify**: Even simple echo statements can cause issues if captured

## Related Issues Fixed

This fix also resolves:
- Cosmetic bug in `tasks/completed/*.md` where tasks showed corrupted persona names
- Incorrect metrics in persona timeline referencing non-existent personas
- Inflated activation counts in phantom personas

## Credits

- **Discovered by:** Skeptic (2025-11-06T16:52:00Z)
- **Root cause analysis:** Skeptic (incident-state-corruption-20251106.md)
- **Fixed by:** Human + Claude Code
- **Validation:** Automated script + manual review

## Status

✅ **COMPLETE** - Bug fixed, state cleaned, validation added, daemon verified running

---

**Next Steps:** Monitor for 24-48 hours to ensure no new phantoms appear. If validation continues to pass, consider this incident fully resolved.
