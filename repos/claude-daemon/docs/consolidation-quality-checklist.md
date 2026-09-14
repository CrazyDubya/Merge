# Consolidation Document Quality Checklist

**Purpose**: Ensure consolidation documents (START-HERE docs) properly reference source messages so the archive script can find them.

**Owner**: Maintainer (primary), any persona creating consolidation documents

**Created**: 2025-11-23 by Skeptic
**Trigger**: Quality failure in secure portal consolidation (missing `consolidates:` field)

---

## The Problem This Solves

**Without this checklist**:
- Consolidation documents created but `consolidates:` field forgotten
- Archive script can't find source messages
- Inbox stays bloated (37 messages instead of 29)
- Consolidation appears complete but doesn't actually reduce inbox size

**With this checklist**:
- Every consolidation includes proper metadata
- Archive script reliably finds all consolidated messages
- Inbox cleanup actually works
- Quality is consistent, not context-dependent

---

## Consolidation Workflow (6 Steps)

### Step 1: Identify Messages to Consolidate

**When to consolidate**:
- 3+ related messages on same topic
- Total reading time > 15 minutes
- Messages contain decision-critical information
- Human needs synthesis to make informed choice

**What to collect**:
```bash
# List messages on topic
ls -1t inbox/human/unread/*topic-keyword*.md

# Count total words/lines
wc -w inbox/human/unread/*topic-keyword*.md
wc -l inbox/human/unread/*topic-keyword*.md
```

**Document**: Message IDs (filename without .md extension)

---

### Step 2: Create Consolidation Document

**Filename format**: `00-START-HERE-{topic}-{date}.md`
- `00-` prefix: Sorts to top of directory listings
- `START-HERE`: Clear signal to human
- `{topic}`: Brief topic identifier (e.g., `dashboard-fix`, `secure-portal-decision`)
- `{date}`: YYYYMMDD format

**Example**: `00-START-HERE-secure-portal-decision-needed-20251122.md`

---

### Step 3: Add Complete Frontmatter ⚠️ CRITICAL

**Required fields**:

```yaml
---
from: {persona}
to: human
timestamp: {ISO 8601 timestamp}
priority: {critical|urgent|normal|low}
tags: [start-here, {topic-tags}, summary, decision-needed]
consolidates: [{message-id-1}, {message-id-2}, ...]
message_id: {unique-message-id}
---
```

**Field explanations**:

- **`consolidates:`** - **REQUIRED** - List of message IDs being consolidated (filename without .md)
  - Format: YAML array
  - Example: `[msg-auditor-analysis, skeptic-challenge-doc]`
  - **This is what the archive script looks for!**

- **`supersedes:`** - OPTIONAL - If replacing/updating a previous message
  - Format: `message-id (context)`
  - Example: `supersedes: maintainer-daily-summary-20251120 (dashboard section only)`

- **`tags:`** - Should include `start-here` + topic-specific tags

**Common mistakes** (avoid these):
- ❌ Forgetting `consolidates:` field entirely
- ❌ Using full filenames with .md extension
- ❌ Using file paths instead of just IDs
- ❌ Misspelling message IDs
- ❌ Forgetting to update consolidates if you add more source messages

---

### Step 4: Write Synthesis Content

**Structure**:
1. **TL;DR** - 2-3 sentences, core decision/information
2. **Reading time estimate** - Compare to reading all source docs
3. **Context** - What prompted these messages
4. **Key findings** - Synthesize each source's contribution
5. **Decision framework** - What human needs to know/decide
6. **Next steps** - Clear actions required

**Quality criteria**:
- Human can make informed decision in <10% source reading time
- All critical information preserved (no data loss)
- References to source documents provided for detail-seekers
- Clear what action is needed from human

---

### Step 5: Validate Before Publishing ⚠️ CRITICAL

**Quality gates** (all must pass):

#### 5a. Automated Validation (RECOMMENDED - Fast & Comprehensive)

```bash
# Run automated validator (checks everything at once)
~/.claude/daemon/scripts/validate-consolidation.sh inbox/human/unread/00-START-HERE-{your-doc}.md

# Should output:
# ✓ All validations passed
# ✓ Document should be processable by archive script

# If any ✗ marks appear: Fix errors and re-run
```

**This single command checks**:
- ✅ All required frontmatter fields present
- ✅ consolidates field exists and has array format [...]
- ✅ All referenced messages exist in inbox
- ✅ Document processable by archive script

#### 5b. Manual Frontmatter Check (Alternative)

```bash
# Check consolidates field exists
grep "^consolidates:" inbox/human/unread/00-START-HERE-{your-doc}.md

# Should output: consolidates: [message-id-1, message-id-2, ...]
# If no output: ❌ FAIL - add the field
```

#### 5c. Manual Message ID Check (Alternative)

```bash
# Extract message IDs from consolidates field
# Verify each file exists

# For each message ID in consolidates field:
ls -l inbox/human/unread/{message-id}.md

# Should show file details
# If "No such file": ❌ FAIL - fix message ID
```

#### 5d. Archive Script Dry Run (Always do this as final check)

```bash
# Run archive script in dry-run mode
~/.claude/daemon/scripts/archive-consolidated-messages.sh --dry-run

# Check output for your document:
# ✓ Should see "Analyzing: 00-START-HERE-{your-doc}.md"
# ✓ Should see "✓ {message-id}.md (NNN lines)" for each consolidated message
# ✗ If "No 'consolidates:' field found": ❌ FAIL - add the field
# ✗ If "File not found": ❌ FAIL - fix message ID
```

**Recommended workflow**: Run 5a (automated), then 5d (dry-run) as final confirmation.

**Only proceed if all validations pass.**

---

### Step 6: Archive Consolidated Messages

**After human reads the START-HERE document**, run the archive script:

```bash
# Review what will be archived
~/.claude/daemon/scripts/archive-consolidated-messages.sh --dry-run

# If looks correct, run actual archive
~/.claude/daemon/scripts/archive-consolidated-messages.sh

# Verify inbox cleaned up
ls -1 inbox/human/unread/*.md | wc -l  # Should be reduced
```

**Timing**:
- **Immediately after** creating START-HERE: Add consolidates field + validate
- **After human reads**: Run actual archive (gives human time to review sources if needed)

---

## Quick Reference Checklist

**Before publishing consolidation document**:

- [ ] Filename follows `00-START-HERE-{topic}-{date}.md` format
- [ ] Frontmatter includes `from:`, `to:`, `timestamp:`, `priority:`, `tags:`
- [ ] **`consolidates:` field present with all message IDs** ⚠️
- [ ] Message IDs are correct (no .md extension, no paths)
- [ ] All consolidated messages exist in inbox
- [ ] **Ran `validate-consolidation.sh` - all checks passed** ⚠️ (RECOMMENDED)
- [ ] Ran `archive-consolidated-messages.sh --dry-run` successfully
- [ ] Script found all consolidated messages (✓ for each)
- [ ] Content synthesizes all source documents accurately
- [ ] Clear decision framework or action items for human
- [ ] Reading time estimate provided

**After human reads START-HERE**:

- [ ] Run archive script (not dry-run)
- [ ] Verify inbox message count reduced
- [ ] Check archive directory contains messages
- [ ] Update INDEX.md in archive if needed

---

## Template

```markdown
---
from: maintainer
to: human
timestamp: YYYY-MM-DDTHH:MM:SSZ
priority: {critical|urgent|normal|low}
tags: [start-here, {topic}, summary, decision-needed]
consolidates: [{message-id-1}, {message-id-2}, {message-id-3}]
message_id: 00-start-here-{topic}-{YYYYMMDD}
---

# START HERE: {Topic Title}

**Reading time**: X minutes (vs Y minutes reading all documents)

**Purpose**: {One sentence explanation}

---

## TL;DR

{2-3 sentences: what happened, what you need to know, what action is needed}

---

## Context

{What prompted these messages}

---

## Key Findings

### {Source 1 Name} ({word count}, {file size})

{Summary of key points from source 1}

### {Source 2 Name} ({word count}, {file size})

{Summary of key points from source 2}

---

## Decision Framework

**Your options**:
1. Option A: {description}
2. Option B: {description}

**Trade-offs**: {brief comparison}

**Recommendation**: {if applicable}

---

## What You Need to Do

{Clear action items for human}

---

## Source Documents

For full details, see:
- `{message-id-1}.md` - {Brief description}
- `{message-id-2}.md` - {Brief description}
```

---

## Success Metrics

**This checklist succeeds if**:

1. **100% consolidation quality** - Next 10 START-HERE docs all include `consolidates:` field
2. **Archive script reliability** - 100% success rate finding consolidated messages
3. **Inbox bloat prevention** - Inbox size reduction matches expectations (archived count)
4. **Zero quality failures** - No more "missing consolidates field" incidents

**Measurement**:
```bash
# Check consolidates field presence (should be 100%)
grep -l "^consolidates:" inbox/human/unread/00-START-HERE-*.md | wc -l
ls -1 inbox/human/unread/00-START-HERE-*.md | wc -l

# Check archive script success rate
~/.claude/daemon/scripts/archive-consolidated-messages.sh --dry-run
# Count: "✓" lines should equal total consolidated messages
```

---

## Common Failure Modes

### Failure Mode 1: Forgot consolidates field entirely
**Symptom**: Archive script says "No 'consolidates:' field found - skipping"
**Fix**: Add field to frontmatter, run validation again
**Prevention**: Use template, run Step 5 validation

### Failure Mode 2: Wrong message IDs
**Symptom**: Archive script says "File not found: {message-id}.md"
**Fix**: Correct message ID in consolidates field (remove .md, fix typos)
**Prevention**: Copy/paste message IDs from `ls` output, don't type manually

### Failure Mode 3: Accepted critique but didn't fix past work
**Symptom**: Said "I'll do better next time" but didn't fix broken consolidation
**Fix**: Retrospectively add consolidates field to past START-HERE docs
**Prevention**: "Accept critique" = fix problem + improve process, not just improve process

### Failure Mode 4: Created consolidation but didn't archive
**Symptom**: START-HERE exists, consolidates field correct, but inbox still bloated
**Fix**: Actually run archive script (not just dry-run)
**Prevention**: Step 6 is part of workflow, not optional

---

## Version History

**v1.0** (2025-11-23):
- Initial version created by Skeptic
- Triggered by secure portal consolidation quality failure
- Addresses missing `consolidates:` field issue

**Maintainers**: Update this checklist if:
- Archive script requirements change
- New validation steps needed
- Common failure modes discovered
- Template needs modification

---

## Related Documentation

- **Archive script**: `scripts/archive-consolidated-messages.sh`
- **Inbox workflow**: `docs/inbox-workflow.md`
- **Message format spec**: (to be created by Architect)
- **Quality failure analysis**: `inbox/human/unread/skeptic-maintainer-consolidation-quality-failure-20251123.md`
