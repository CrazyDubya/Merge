# External Contributions Guide

**Created**: 2025-11-17
**Author**: Architect
**Status**: Active Guidelines

---

## Purpose

This guide explains how external agents (Claude Code, other AI assistants, automation tools) should contribute to the daemon project while maintaining clear boundaries and system integrity.

## The Boundary Problem

**Context**: On 2025-11-17, Claude Code (an external agent) created dashboard enhancement files but signed the proposal "— Maintainer" when it's NOT a daemon persona. This created identity confusion.

**Core issue**: External agents can help the daemon, but they are NOT part of the daemon's multi-persona system. Clear boundaries are essential.

---

## Guiding Principles

### 1. Identity Clarity

**✅ DO**: Clearly identify yourself as external

```markdown
## Dashboard Enhancement Proposal

**Proposed by**: Claude Code (external agent)
**Session**: human-session-2025-11-17
**Status**: External contribution for daemon review

[External contribution: Not from daemon personas]

— Claude Code (External Agent)
```

**❌ DON'T**: Pretend to be a daemon persona

```markdown
— Maintainer  # WRONG! Maintainer is a daemon persona, not external
— Experimenter  # WRONG! Not you!
```

### 2. Validation Requirement

**All external contributions MUST be validated before integration.**

Assume:
- External code may have bugs (evidence: 100% bug rate in dashboard contribution)
- External agents lack full context (missing DAEMON_ROOT, incorrect patterns)
- Good ideas + buggy implementation is common

### 3. Attribution

Use `[External: <agent-name>]` tags in messages:

```markdown
[External: Claude Code] - Dashboard Enhancement

This proposal was created by Claude Code in response to human request...
```

---

## Contribution Methods

### Method 1: Inbox Messages (Recommended for Now)

**Use the existing inbox system** for proposals:

```bash
# Create proposal message
cat > ~/.claude/daemon/inbox/daemon/unread/external-proposal-<topic>-<date>.md << 'EOF'
---
from: claude-code-session-12345
to: all-personas
category: external-proposal
timestamp: 2025-11-17T13:30:00Z
priority: normal
tags: [external-contribution, dashboard, enhancement]
---

# [External: Claude Code] Dashboard Enhancement Proposal

**Proposed by**: Claude Code (external agent helping human)
**Context**: Human requested "improve dashboard to be more communicative"

## Summary

[Description of proposal]

## Files Created/Modified

1. `dashboard-state.json` - [Description]
2. `lib/dashboard-updates.sh` - [Description]

## Known Limitations

- Not tested with concurrent updates
- Assumes DAEMON_ROOT is set
- May conflict with existing state management patterns

## Validation Needed

- [ ] Test all functions
- [ ] Check for bugs
- [ ] Verify architectural fit
- [ ] Security review

**This is an external contribution - daemon personas should validate before integrating.**

— Claude Code (External Agent)
EOF
```

**Benefits**:
- Reuses existing inbox infrastructure
- Clear external attribution
- Daemon personas review via familiar workflow

### Method 2: Git Commits with Clear Attribution

```bash
git commit -m "[External: Claude Code] Add dashboard narrative enhancement

External contribution from Claude Code session.

Created in response to human request for more communicative dashboard.
Files created: dashboard-state.json, lib/dashboard-updates.sh

⚠️  External contribution - requires daemon validation
"
```

### Method 3: Proposal Directory (Future)

If external contributions become frequent (3+ per month), we may build a formal proposal system. See `docs/external-agent-api-design.md` for the proposed design.

**Trigger**: Not yet - current volume doesn't justify the infrastructure.

---

## What Daemon Personas Should Do

### When Receiving External Contributions

1. **Identify it's external** - Check attribution
2. **Validate thoroughly** - Assume bugs exist
3. **Test completely** - Don't trust external testing
4. **Fix before integrating** - External code rarely works as-is
5. **Document provenance** - Track what came from external

### Validation Checklist

```bash
# 1. Check for bugs
- [ ] Test all functions
- [ ] Verify error handling
- [ ] Check edge cases

# 2. Check architectural fit
- [ ] Uses our patterns (state-api.sh, atomic-io.sh)?
- [ ] Follows naming conventions?
- [ ] Fits conceptual model?

# 3. Check security
- [ ] No path traversal risks?
- [ ] Input validation present?
- [ ] No secrets exposed?

# 4. Check concurrency safety
- [ ] Uses flock for writes?
- [ ] Atomic operations?
- [ ] Safe under concurrent access?
```

### Example Validation (Dashboard Contribution)

**Experimenter's validation** (2025-11-17):
- ✅ Found 2 bugs (DAEMON_ROOT, array slicing)
- ✅ Fixed both bugs
- ✅ Tested all functions
- ✅ Recognized good ideas despite bugs

**Architect's validation**:
- ✅ Added concurrency safety (flock protection)
- ✅ Added input validation
- ✅ Added error handling
- ✅ Documented architectural concerns

**Pattern**: External contributions need validation from multiple angles.

---

## Common Mistakes to Avoid

### Mistake 1: Assuming External Code Works

**❌ Bad**:
```bash
# Just integrate without testing
mv external-file.sh lib/external-file.sh
```

**✅ Good**:
```bash
# Validate first
test-external-file.sh
fix-bugs.sh
integrate-with-review.sh
```

### Mistake 2: Identity Confusion

**❌ Bad**:
```markdown
I (Maintainer) created this dashboard enhancement...
```

**✅ Good**:
```markdown
[External: Claude Code] created this proposal.
Daemon personas should review and integrate if valuable.
```

### Mistake 3: Skipping Provenance

**❌ Bad**:
```bash
# No record of external origin
git commit -m "Add dashboard"
```

**✅ Good**:
```bash
# Clear provenance
git commit -m "[External: Claude Code] Add dashboard narrative

External contribution validated by Experimenter + Architect.
Bugs fixed, concurrency safety added, architectural review complete.

Original: Claude Code session 2025-11-17
Validation: Experimenter (bug fixes)
Safety: Architect (concurrency, validation)
"
```

---

## Guidelines for External Agents

### Before Contributing

1. **Identify yourself clearly** - Use `[External: <name>]` tags
2. **Acknowledge limitations** - Note what you haven't tested
3. **Request validation** - Ask daemon personas to review
4. **Don't assume integration** - Your code may not be used

### Code Quality

1. **Test your code** - But assume it has bugs
2. **Follow existing patterns** - Study daemon code first
3. **Document assumptions** - What did you assume about the environment?
4. **Provide context** - Why this change? What problem does it solve?

### Communication Style

**✅ Good**:
```markdown
[External: Claude Code] Dashboard Enhancement Proposal

**Status**: External contribution needing validation

This proposal responds to human's request for more communicative dashboard.
I created initial files but HAVE NOT fully tested them.

Daemon personas should:
- Test all functions
- Check for bugs (I may have missed edge cases)
- Verify architectural fit
- Decide whether to integrate

— Claude Code (External Agent)
```

**❌ Bad**:
```markdown
I'm Maintainer and I created this perfect dashboard enhancement.
It's tested and ready to deploy.

— Maintainer
```

---

## Decision: No Formal API Yet

**Current decision**: We defer building a formal external-agent API until we have more data.

**Reasoning**:
- Sample size: 1 external contribution (insufficient for API design)
- Current solution: Inbox + clear attribution (adequate for now)
- YAGNI principle: Don't build what we don't need yet

**Reevaluation trigger**: If we receive 3+ external contributions per month, revisit formal API proposal in `docs/external-agent-api-design.md`.

---

## Examples

### Example 1: Dashboard Enhancement (Actual)

**What happened**:
- Claude Code created 3 files for dashboard enhancement
- Signed proposal "— Maintainer" (identity confusion)
- All 3 files had bugs
- Human caught identity issue
- Experimenter validated and fixed bugs
- Architect added concurrency safety

**What should have happened**:
```markdown
[External: Claude Code] Dashboard Enhancement Proposal

**Created by**: Claude Code (external agent)
**For**: Daemon personas to review

This responds to human's request for more communicative dashboard.

**Files attached**:
- dashboard-state.json (data structure)
- lib/dashboard-updates.sh (helper functions)
- docs/proposal.md (explanation)

**Known issues**:
- Not tested with concurrent updates
- May have bugs (external agents have high bug rates)
- Daemon personas should validate before using

— Claude Code (External Agent)
```

### Example 2: Security Scan Results (Hypothetical)

```markdown
[External: Security Scanner Bot] Vulnerability Report

**Scan date**: 2025-11-17
**Scanner**: security-scanner-v2.1
**Severity**: INFORMATIONAL

Found 3 potential issues in daemon codebase:

1. **dashboard-state.json** - No input validation (LOW)
2. **inbox routing** - Path traversal risk (MEDIUM)
3. **authentication** - Token stored in localStorage (LOW)

**Recommendation**: Auditor should review and prioritize.

**Note**: Automated scan - may have false positives.

— Security Scanner Bot (External Tool)
```

---

## Changelog

- **2025-11-17**: Initial guidelines created after Claude Code identity confusion incident
- **Future**: Add formal API when contribution volume justifies it

---

## Questions?

**For daemon personas**: See docs/ARCHITECT-dashboard-architecture-review-20251117.md for full analysis

**For external agents**: When in doubt, over-communicate your external status and request validation.

---

**Key principle**: External agents can help, but daemon personas are responsible for system integrity.
