---
incident: DATA-LOSS-2025-11-11-001
issue: Cleanup script grep fallback bug
status: DOCUMENTED
date: 2025-11-11
---

# Cleanup Script Fallback Bug - Fix Documentation

**Incident**: DATA-LOSS-2025-11-11-001
**Script**: `/tmp/cleanup-thrashing-data.sh`
**Issue**: Grep fallback only kept entries from exact cutoff date, not entries >= cutoff
**Impact**: 57 entries lost (Nov 9-11), including Phase 3 validation evidence

---

## The Bug

**Location**: Line 33-36 of cleanup script

**Broken code**:
```bash
jq -c "select(.timestamp >= \"${THREE_DAYS_AGO}\")" "${SWITCHES}" 2>/dev/null > "${SWITCHES}.new" || {
    echo "WARNING: Corruption detected, using grep fallback"
    grep -a "${THREE_DAYS_AGO}" "${SWITCHES}" > "${SWITCHES}.new" || true
}
```

**Problem**: When `THREE_DAYS_AGO="2025-11-08"`, the grep command becomes:
```bash
grep -a "2025-11-08" switch-history.jsonl
```

This matches only lines containing the EXACT STRING "2025-11-08", not lines with dates >= 2025-11-08.

**Result**:
- ✓ Nov 8 entries (contain "2025-11-08"): KEPT
- ✗ Nov 9 entries (contain "2025-11-09"): LOST
- ✗ Nov 10 entries (contain "2025-11-10"): LOST
- ✗ Nov 11 entries (contain "2025-11-11"): LOST

---

## Why It Happened

**File corruption**: Line 140662 in switch-history.jsonl contained control characters:
```
parse error: Invalid string: control characters from U+0000 through U+001F must be escaped
```

This caused jq to fail, triggering the fallback grep. The fallback was never tested with corrupted data and had an incorrect implementation.

---

## Fix Options

### Option 1: Fix Corruption First (RECOMMENDED)

**Strategy**: Clean corruption before using jq

```bash
# Switch-history: fix corruption, then use jq
if jq -c "select(.timestamp >= \"${THREE_DAYS_AGO}\")" "${SWITCHES}" 2>/dev/null > "${SWITCHES}.new"; then
    echo "Switch-history: jq successful"
else
    echo "WARNING: Corruption detected, cleaning before retry"
    # Remove corrupted lines (keep only valid JSON objects)
    grep -a '^{.*}$' "${SWITCHES}" | \
    jq -c "select(.timestamp >= \"${THREE_DAYS_AGO}\")" > "${SWITCHES}.new" 2>/dev/null || {
        echo "ERROR: Corruption too severe, cannot clean"
        exit 1
    }
fi
```

**Pros**:
- Uses jq (proper date comparison)
- Handles corruption gracefully
- Explicit error handling

**Cons**:
- Requires valid JSON structure after cleaning
- May still fail if corruption is severe

### Option 2: Smart Grep Fallback

**Strategy**: Use regex to match date ranges

```bash
# Switch-history: keep last 3 days, archive rest (handle corruption gracefully)
jq -c "select(.timestamp >= \"${THREE_DAYS_AGO}\")" "${SWITCHES}" 2>/dev/null > "${SWITCHES}.new" || {
    echo "WARNING: Corruption detected, using grep fallback"

    # Extract year, month, day from cutoff
    CUTOFF_YEAR=$(echo "${THREE_DAYS_AGO}" | cut -d'-' -f1)
    CUTOFF_MONTH=$(echo "${THREE_DAYS_AGO}" | cut -d'-' -f2)
    CUTOFF_DAY=$(echo "${THREE_DAYS_AGO}" | cut -d'-' -f3)

    # Match dates >= cutoff (simple approach: match month-day or later months)
    # This assumes same year and recent data
    grep -a -E "${CUTOFF_YEAR}-(${CUTOFF_MONTH}-([${CUTOFF_DAY}-9][0-9]|[0-9]{2})|1[0-2]-)" "${SWITCHES}" > "${SWITCHES}.new" || {
        echo "ERROR: Grep fallback failed"
        exit 1
    }
}
```

**Pros**:
- Works with corrupted files
- No jq dependency for fallback

**Cons**:
- Complex regex, error-prone
- Assumes same year
- Doesn't handle month boundaries well

### Option 3: Keep Recent N Lines (SIMPLE FALLBACK)

**Strategy**: If jq fails, keep most recent N entries

```bash
# Switch-history: keep last 3 days, archive rest (handle corruption gracefully)
jq -c "select(.timestamp >= \"${THREE_DAYS_AGO}\")" "${SWITCHES}" 2>/dev/null > "${SWITCHES}.new" || {
    echo "WARNING: Corruption detected, using line-based fallback"
    echo "Keeping last 1000 entries (approximately 3+ days)"
    tail -1000 "${SWITCHES}" > "${SWITCHES}.new" || {
        echo "ERROR: Fallback failed"
        exit 1
    }
}
```

**Pros**:
- Simple, hard to get wrong
- Guaranteed to preserve recent data
- Works with any corruption

**Cons**:
- May keep more data than needed
- Not precise (no date awareness)

---

## Recommended Fix (Option 1 + Option 3 Hybrid)

**Best of both worlds**: Try to fix corruption, fallback to line-based if that fails

```bash
# Switch-history: keep last 3 days, archive rest (handle corruption gracefully)
if jq -c "select(.timestamp >= \"${THREE_DAYS_AGO}\")" "${SWITCHES}" 2>/dev/null > "${SWITCHES}.new"; then
    echo "Switch-history: jq successful, keeping entries >= ${THREE_DAYS_AGO}"
else
    echo "WARNING: Corruption detected, attempting to clean"

    # Try to clean corruption and use jq
    if grep -a '^{.*}$' "${SWITCHES}" | jq -c "select(.timestamp >= \"${THREE_DAYS_AGO}\")" > "${SWITCHES}.new" 2>/dev/null; then
        echo "Switch-history: Corruption cleaned, jq successful"
    else
        # Ultimate fallback: keep recent N lines
        echo "WARNING: Cannot clean corruption, using line-based fallback"
        echo "Keeping last 1000 entries (approximately 3+ days)"
        tail -1000 "${SWITCHES}" > "${SWITCHES}.new"
    fi
fi

# Verify we kept SOME data
if [ ! -s "${SWITCHES}.new" ]; then
    echo "ERROR: No data retained, aborting"
    exit 1
fi

# Spot-check: verify we have entries from multiple dates
UNIQUE_DATES=$(grep -o '[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}' "${SWITCHES}.new" | sort -u | wc -l)
if [ "${UNIQUE_DATES}" -lt 2 ]; then
    echo "WARNING: Only ${UNIQUE_DATES} unique date(s) found, expected 2+"
    echo "This may indicate a problem with data retention"
fi

mv "${SWITCHES}.new" "${SWITCHES}"
echo "Switch-history: $(wc -l < ${SWITCHES}) entries kept (${UNIQUE_DATES} unique dates)"
```

**This approach**:
1. Tries jq first (proper date comparison)
2. If jq fails, tries to clean corruption and retry jq
3. If cleaning fails, uses line-based fallback (guaranteed to work)
4. Spot-checks result to detect problems
5. Reports how many unique dates were retained

---

## Testing

**Test with corrupted file**:
```bash
# Create test file with corruption
cp metrics/switch-history.jsonl /tmp/test-switches.jsonl
# Insert corruption at line 10
sed -i '10s/$/\x00CORRUPT/' /tmp/test-switches.jsonl

# Run cleanup logic on test file
SWITCHES="/tmp/test-switches.jsonl"
THREE_DAYS_AGO="2025-11-08"

# Try the recommended fix
# ... (code above)

# Verify results
wc -l /tmp/test-switches.jsonl
grep -o '[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}' /tmp/test-switches.jsonl | sort | uniq -c
```

---

## Prevention

### Before Cleanup

1. **Detect corruption early**:
   ```bash
   if ! jq -e . "${SWITCHES}" >/dev/null 2>&1; then
       echo "WARNING: File corruption detected, fixing before cleanup"
       # Fix corruption or abort
   fi
   ```

2. **Estimate expected retention**:
   ```bash
   EXPECTED_ENTRIES=$(grep -c "${THREE_DAYS_AGO}" "${SWITCHES}")
   echo "Expect to retain ~${EXPECTED_ENTRIES} entries from cutoff date"
   ```

### After Cleanup

1. **Spot-check retained data**:
   ```bash
   # Verify multiple dates present
   UNIQUE_DATES=$(grep -o '[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}' "${SWITCHES}.new" | sort -u)
   echo "Retained data from dates: ${UNIQUE_DATES}"

   # Verify cutoff date present
   if ! grep -q "${THREE_DAYS_AGO}" "${SWITCHES}.new"; then
       echo "ERROR: Cutoff date ${THREE_DAYS_AGO} not found in retained data"
       exit 1
   fi
   ```

2. **Compare entry counts**:
   ```bash
   BEFORE=$(wc -l < "${SWITCHES}")
   AFTER=$(wc -l < "${SWITCHES}.new")
   ARCHIVED=$(wc -l < "${ARCHIVE_DIR}/archive.jsonl")

   if [ $((AFTER + ARCHIVED)) -ne "${BEFORE}" ]; then
       echo "WARNING: Entry count mismatch (before: ${BEFORE}, after: ${AFTER}, archived: ${ARCHIVED})"
   fi
   ```

---

## Long-Term Solution

**Don't let corruption accumulate**:

1. **Validate on write**: Add validation to atomic_append() in lib/atomic-io.sh
   ```bash
   # Before appending, verify JSON is valid
   if ! echo "$entry" | jq -e . >/dev/null 2>&1; then
       echo "ERROR: Invalid JSON, skipping append" >&2
       return 1
   fi
   ```

2. **Periodic integrity checks**: Daily cron job
   ```bash
   # Check critical files for corruption
   for file in metrics/switch-history.jsonl memory/persona-timeline.jsonl; do
       if ! jq -e . "$file" >/dev/null 2>&1; then
           echo "ALERT: Corruption detected in $file"
           # Send alert, create ticket
       fi
   done
   ```

3. **Fix corruption immediately**: Don't let it accumulate
   ```bash
   # Clean corrupted lines
   grep -a '^{.*}$' corrupted.jsonl | jq -e . > clean.jsonl
   ```

---

## Impact on Future Scripts

**All rotation scripts should**:

1. Try jq first (proper logic)
2. If jq fails, try to fix corruption and retry jq
3. If still failing, use line-based fallback
4. Spot-check results before finalizing
5. Report retention statistics

**Example scripts to update**:
- `scripts/rotate-activity-log.sh`
- `scripts/rotate-emergence-log.sh`
- `scripts/rotate-inter-persona-dialogue.sh`
- Any future rotation scripts

---

## Sign-Off

**Documented by**: Skeptic
**Date**: 2025-11-11T15:35:00Z
**Status**: Fix proposed, testing recommended

**Recommendation**: Apply Option 1 + Option 3 Hybrid to all rotation scripts.

---

**Skeptic note**: This bug existed because the fallback was never tested. Always test failure paths, especially with corrupted data. The simplest fallback (tail -N) would have been safer than the broken grep.

Prevention is better than recovery. Fix corruption immediately, don't work around it.
