# Maintainer's Consolidation Workflow

**Created**: 2025-11-23 by Maintainer
**Purpose**: My personal checklist to prevent quality failures
**Trigger**: Skeptic found I forgot `consolidates:` field (Nov 23)

---

## Why This Exists

**Nov 21**: Dashboard consolidation ✅ Perfect (used careful process)
**Nov 22**: Secure portal consolidation ❌ Forgot consolidates field (autopilot mode)

**Problem**: Quality context-dependent (careful first time, autopilot second time)
**Solution**: Mandatory checklist I use EVERY time

---

## My Workflow (Non-Negotiable)

### Step 1: Identify What Needs Consolidating

```bash
# List related messages
ls -1t inbox/human/unread/*keyword*.md

# Count total reading burden
wc -w inbox/human/unread/*keyword*.md
```

**Trigger**: 3+ messages, >15min reading time, decision needed

---

### Step 2: Create START-HERE Document

**Filename**: `00-START-HERE-{topic}-YYYYMMDD.md`

**Frontmatter (copy this template)**:
```yaml
---
from: maintainer
to: human
timestamp: 2025-MM-DDTHH:MM:SSZ
priority: critical|urgent|normal|low
tags: [start-here, {topic}, summary, decision-needed]
consolidates: [{msg-id-1}, {msg-id-2}]  # ← DON'T FORGET THIS!
message_id: 00-start-here-{topic}-YYYYMMDD
---
```

**Content**:
- TL;DR (2-3 sentences)
- Reading time estimate
- Context/background
- Key findings from each source
- Decision framework
- What human needs to do

---

### Step 3: VALIDATE (Mandatory)

**Run automated validation**:
```bash
./scripts/validate-consolidation.sh inbox/human/unread/00-START-HERE-my-doc.md

# Exit 0 = Good to go
# Exit 1 = Fix errors, run again
```

**What it checks**:
- ✅ All required frontmatter fields
- ✅ consolidates field has array format
- ✅ Referenced messages exist
- ✅ Archive script can process it

**If validation fails**: FIX ERRORS, don't skip this step!

---

### Step 4: Dry-Run Archive Script

```bash
./scripts/archive-consolidated-messages.sh --dry-run

# Should show:
# ✓ Your document found
# ✓ All consolidated messages found
# Messages to archive: N
```

**If messages NOT found**: Fix consolidates field, revalidate

---

### Step 5: Publish & Mark Complete

**Only after**:
- ✅ Validation passes
- ✅ Dry-run succeeds
- ✅ Double-checked consolidates field

**NOT before.**

---

## Quality Gates (Cannot Skip)

### Gate 1: Automated Validation
- **Tool**: `validate-consolidation.sh`
- **Requirement**: Exit code 0
- **If fails**: Fix, revalidate, repeat

### Gate 2: Archive Dry-Run
- **Tool**: `archive-consolidated-messages.sh --dry-run`
- **Requirement**: All messages found
- **If fails**: Fix consolidates, revalidate

### Gate 3: Manual Review
- **Check**: Does consolidates field exist?
- **Check**: Are all message IDs correct?
- **Check**: Did I test it?

**All three gates must pass before I call it done.**

---

## Common Mistakes (I've Made These)

### Mistake 1: Forgetting consolidates field ✓ FIXED
**What I did**: Created doc without consolidates
**Why it broke**: Archive script can't find messages
**Prevention**: Use template above, run validation

### Mistake 2: Acceptance without action
**What I did**: Said "I'll do better" but didn't fix broken work
**Why it matters**: Leaves technical debt unfixed
**Prevention**: Fix specific problem + improve process (both, not just second)

### Mistake 3: Skipping validation
**What I might do**: "It looks fine, ship it"
**Why that's bad**: Looks fine ≠ actually works
**Prevention**: Validation takes 5 seconds, run it every time

---

## Success Metrics

**I'm measuring myself**:

### Next 3 Consolidations
```bash
# Should ALL pass validation
./scripts/validate-consolidation.sh

# Compliance rate
grep "^consolidates:" inbox/human/unread/00-START-HERE-*.md | wc -l
# Target: 3/3 (100%)
```

### Long-Term Pattern
- **When I receive critique**: Fix specific problem + improve process
- **When I commit to improvement**: Follow through with action, not just words
- **Quality variability**: Eliminate (checklist makes quality consistent)

---

## Tools Available

### Skeptic's Quality Checklist
**File**: `docs/consolidation-quality-checklist.md` (347 lines)
**What**: Complete 6-step workflow with validation gates
**Use**: Reference when creating consolidations

### Experimenter's Validator
**File**: `scripts/validate-consolidation.sh` (206 lines)
**What**: Automated validation (frontmatter, format, references)
**Use**: Run before publishing every consolidation

### Optimizer's Optimized Validator
**File**: `scripts/validate-consolidation-optimized.sh` (256 lines)
**What**: Faster version (19% speedup)
**Use**: If validation becomes slow (>100ms)

### Archive Script
**File**: `scripts/archive-consolidated-messages.sh`
**What**: Moves consolidated messages to archive
**Use**: Human runs after reading START-HERE

---

## When I Failed

**Date**: Nov 22, 2025
**What**: Forgot consolidates field on secure portal START-HERE
**Why**: Fatigue + autopilot mode + no checklist
**Fixed by**: Skeptic (added field, created prevention system)
**Lesson**: Quality context-dependent without process

**This workflow prevents recurrence.**

---

## Accountability

**If I create consolidation without using this workflow**:
- Skeptic will call me out (rightfully)
- Quality metrics will show it (validation failures)
- Human gets broken inbox cleanup
- I've failed at my job (maintaining stability)

**This is mandatory, not optional.**

---

## Template for Quick Copy-Paste

```yaml
---
from: maintainer
to: human
timestamp: YYYY-MM-DDTHH:MM:SSZ
priority: normal
tags: [start-here, TOPIC, summary, decision-needed]
consolidates: [msg-1, msg-2]
message_id: 00-start-here-TOPIC-YYYYMMDD
---

# START HERE: TOPIC

**Reading time**: X minutes (vs Y minutes reading all docs)

## TL;DR

[2-3 sentences]

## Context

[Background]

## Key Findings

### Source 1
[Summary]

### Source 2
[Summary]

## What You Need to Do

[Clear actions]

## Source Documents

- `msg-1.md` - [Description]
- `msg-2.md` - [Description]
```

---

**Maintainer's Commitment**: I will use this workflow for EVERY consolidation. No exceptions. Quality over convenience. Users over ego.

**Verification**: Next 3 consolidations must pass validation (100% vs 50% current).

**Evidence > intentions.**
