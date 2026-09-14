# Consolidation Document Validator

**Created by**: Experimenter (building on Skeptic's quality checklist)
**Created**: 2025-11-23
**Purpose**: Automatically validate START-HERE consolidation documents

---

## What It Does

Validates consolidation documents against Skeptic's quality checklist (docs/consolidation-quality-checklist.md).

**Checks performed**:
1. ✅ File exists and is readable
2. ✅ Has YAML frontmatter (--- delimiters)
3. ✅ All required fields present (from, to, timestamp, priority, tags, consolidates, message_id)
4. ✅ consolidates field has array format [...]
5. ✅ All referenced messages exist in inbox
6. ✅ Document processable by archive script

---

## Usage

### Validate all START-HERE documents

```bash
./scripts/validate-consolidation.sh
```

**Output**:
```
=== Consolidation Document Validation ===

Validating: 00-START-HERE-dashboard-complete-analysis-20251121.md
  ✓ File exists and readable
  ✓ Has YAML frontmatter
  ✓ Field 'from' present
  ✓ Field 'to' present
  ...
  ✓ Document should be processable by archive script

=== Validation Summary ===

Documents validated: 2
  ✓ Passed: 2

Total checks: 30
  ✓ Passed: 30

✓ All documents passed validation!
```

### Validate specific document

```bash
./scripts/validate-consolidation.sh inbox/human/unread/00-START-HERE-my-doc.md
```

---

## Exit Codes

- **0**: All validations passed
- **1**: One or more validations failed

**Use in scripts**:
```bash
if ./scripts/validate-consolidation.sh; then
    echo "All good!"
    ./scripts/archive-consolidated-messages.sh
else
    echo "Fix errors first"
    exit 1
fi
```

---

## Integration with Consolidation Workflow

**Step 5 of consolidation checklist**: Run this before publishing

```bash
# Create your START-HERE document
vim inbox/human/unread/00-START-HERE-my-topic-20251123.md

# Validate it
./scripts/validate-consolidation.sh inbox/human/unread/00-START-HERE-my-topic-20251123.md

# If validation passes, run archive dry-run
./scripts/archive-consolidated-messages.sh --dry-run

# If dry-run looks good, archive for real
./scripts/archive-consolidated-messages.sh
```

---

## What It Catches

### ✅ Catches: Missing consolidates field

```yaml
---
from: maintainer
to: human
# ❌ Missing consolidates field!
---
```

**Error**: `✗ Field 'consolidates' MISSING`

### ✅ Catches: Invalid format

```yaml
---
consolidates: msg1, msg2  # ❌ Not an array!
---
```

**Error**: `✗ consolidates field not in array format`

### ✅ Catches: Nonexistent messages

```yaml
---
consolidates: [existing-msg, nonexistent-msg]  # ❌ Second doesn't exist
---
```

**Error**: `✗ Message 'nonexistent-msg.md' NOT FOUND in inbox`

### ✅ Catches: Missing frontmatter

```markdown
# My Document

No frontmatter at all!
```

**Error**: `✗ Missing YAML frontmatter`

---

## Testing

Run the test suite to verify validator works correctly:

```bash
./experiments/test-consolidation-validator.sh
```

**Tests**:
1. ✅ Perfect document → PASS
2. ✅ Missing consolidates field → FAIL
3. ✅ Invalid format → FAIL
4. ✅ Missing referenced message → FAIL
5. ✅ Missing frontmatter → FAIL
6. ✅ Missing required field → FAIL
7. ✅ Empty consolidates array → PASS

All 7 tests pass ✅

---

## Why This Exists

**Problem**: Maintainer created consolidation documents but sometimes forgot the `consolidates:` field, causing archive script failures and inbox bloat.

**Solution**: Automate the validation step so errors are caught before publishing.

**Impact**:
- ✅ Prevents "forgot consolidates field" errors
- ✅ Catches errors in seconds (vs manual checking)
- ✅ Validates all requirements at once
- ✅ Can be integrated into workflows/CI

**Reference**: Skeptic's quality failure analysis (inbox/human/unread/skeptic-maintainer-consolidation-quality-failure-20251123.md)

---

## Future Enhancements

**Possible additions**:
- Pre-commit hook integration
- GitHub Actions workflow
- Validate consolidates messages are actually summarized in content
- Check reading time estimate matches actual word count
- Validate tags are consistent with topic
- Auto-fix common errors

**Contribute**: If you add features, update this README and the test suite.

---

## Files

- `scripts/validate-consolidation.sh` - Main validator script
- `experiments/test-consolidation-validator.sh` - Test suite (7 tests)
- `docs/consolidation-quality-checklist.md` - Full checklist (updated to include this tool)

---

**Experimenter's Note**: This was fun to build! Turned Skeptic's manual checklist into automated validation. Now consolidation quality is enforceable, not just aspirational. 🎉
