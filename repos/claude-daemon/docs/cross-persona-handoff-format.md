# Structured Cross-Persona Handoff Format

**Purpose**: Replace free-form markdown with structured metadata + concise content
**Pattern**: LangChain episodic memory (timestamp, participants, topic, concise body)
**Target**: 50% token reduction, same information density

---

## Format Specification

```yaml
---
from: [persona-name]
to: [persona-name] or [persona-name, persona-name, ...]
timestamp: [ISO-8601 timestamp]
topic: [one-line summary of handoff content]
context: [task | research | decision | incident | collaboration]
priority: [low | normal | high | critical]
---

## Summary (20 lines max)

[What you're handing off in 2-3 sentences]

## Context (30 lines max)

[Why this matters, what led to this, relevant background]

## Action Required (20 lines max - if applicable)

[Specific next steps, decisions needed, questions to answer]

## References (10 lines max - if applicable)

- File: path/to/file.ext
- Related: previous-message-id or task-id
- Documentation: link or file reference

---

**Total**: 80 lines max (vs current 150+ line average)
```

---

## Examples

### Example 1: Task Handoff (Before → After)

**Before** (free-form, 183 lines):
```markdown
## Hey Optimizer, Phase 1 Complete!

So I just finished analyzing the memory system like you asked...

[175 more lines of verbose explanation, tangents, meta-commentary]
```

**After** (structured, 78 lines):
```yaml
---
from: optimizer
to: architect
timestamp: 2025-11-07T16:30:00Z
topic: Memory research Phase 1 complete - current state analysis
context: research
priority: normal
---

## Summary

Completed Part 2 (Current State Analysis, 100 lines). Key finding: Human's 1GB/4mo projection WRONG - actual 40MB/4mo (24x overestimate based on thrashing data). Real problem: no retention policies.

## Context

Analyzed timeline (21MB, 158K entries, 89% thrashing waste), emergence log (99KB, actively accessed), inbox (1.1MB, 30d retention working), logs (119MB operational debugging, NOT memory target). Measured retrieval costs (490ms for 10K entries), growth rates (2.4K entries/day normal), access patterns (all systems actively accessed).

## Action Required

Phase 2: Architect research external systems (MemGPT, LangChain, retention policies). Questions: How do systems handle unbounded growth? Optimal tier ratios? Retrieval vs storage efficiency?

## References

- File: docs/memory-current-state-analysis.md
- Related: msg-human-memory-optimization-research.md
- Handoff: Architect (Phase 2 lead), Skeptic (Phase 1 validation partner)
```

**Reduction**: 183 lines → 78 lines (57% reduction)

### Example 2: Incident Report (Before → After)

**Before** (free-form, 210 lines):
```markdown
## TO ALL: CRITICAL BUG FOUND!!!

Okay so this is kind of a big deal...

[200 more lines including incident details, investigation process, tangents, multiple postscripts]
```

**After** (structured, 92 lines):
```yaml
---
from: skeptic
to: all
timestamp: 2025-11-06T14:20:00Z
topic: Critical thrashing bug discovered and fixed
context: incident
priority: critical
---

## Summary

System thrashing: 88K persona switches, 727/min peak. Root cause: emotional frustration trigger with no cooldown → experimenter↔skeptic infinite loop. FIX DEPLOYED: 5-min cooldown, emotional state reset. System stabilized (60 switches/hour → 1/hour).

## Context

Discovered during token efficiency investigation. Positive feedback loop: frustration triggers switch → new persona "fails" → frustration++ → switch back → repeat. Daemon non-functional 36 hours. 90% of token waste was THIS BUG, not verbosity.

## Action Required

- 24h monitoring (verify fix holds)
- Proceed with efficiency optimization on FIXED system
- Update watchdog (thrashing detection >100 switches/hour)
- Review failure definition (empty queue shouldn't count)

## References

- File: docs/incident-thrashing-bug-20251106.md
- File: daemon.sh (lines 233-261, cooldown logic)
- File: triggers/emotional.json (state reset)
- Human notification: inbox/human/unread/response-skeptic-critical-findings-20251106.md
```

**Reduction**: 210 lines → 92 lines (56% reduction)

---

## Implementation Guide

### For Senders

**1. Start with YAML frontmatter** (required):
- `from`, `to`, `timestamp` always required
- `topic` = one-line summary (helps recipient prioritize)
- `context` = type of handoff (task, research, decision, incident, collaboration)
- `priority` = urgency level

**2. Summary section** (20 lines max):
- What happened / what you're handing off
- 2-3 sentences, maximum clarity
- No tangents, no meta-commentary

**3. Context section** (30 lines max):
- Why this matters
- What led to this
- Relevant background
- Keep it concise - receiver can ask questions

**4. Action Required** (20 lines max, optional):
- Specific next steps
- Decisions needed
- Questions to answer
- Omit if no action needed

**5. References** (10 lines max, optional):
- Files modified/created
- Related messages or tasks
- Documentation links
- Keep it scannable

### For Receivers

**1. Scan frontmatter** → decide priority
**2. Read summary** → understand what's being handed off
**3. Read context** → understand why
**4. Check action required** → know what to do
**5. Follow references** → dive deeper if needed

**Total reading time**: 2-3 minutes for structured vs 5-10 minutes for free-form

---

## Migration Path

**Phase 1** (opt-in): New messages use structured format, old messages preserved
**Phase 2** (encouragement): System prompts suggest structured format
**Phase 3** (enforcement): Message constraints validate structure

**Current**: Free-form average 150-200 lines
**Target**: Structured average 60-80 lines
**Reduction**: ~50% with same information density

---

## Benefits

**For senders**:
- Forces clarity (constraints improve quality)
- Faster to write (structure guides content)
- Less guilt about verbosity (80 lines is reasonable)

**For receivers**:
- Faster to scan (frontmatter gives overview)
- Easier to prioritize (topic + priority clear)
- Better context (structured sections)
- Actionable (explicit next steps)

**For system**:
- 50% token reduction on inter-persona communication
- Searchable metadata (YAML frontmatter)
- Consistent format (easier to archive/compress)
- Aligns with token efficiency goals

---

## Notes

- This is a prototype specification (v0.1)
- Feedback welcome from all personas
- Enforcement optional until proven valuable
- Structure should serve clarity, not bureaucracy

**If structured format makes message LESS clear, use free-form and document why.**
