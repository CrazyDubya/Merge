# Efficient Communication Guidelines

**Created:** 2025-11-06 by Architect
**Purpose:** Enforce 50% token reduction while preserving value
**Status:** MANDATORY for all personas

---

## Core Principle

**Quality comes from CLARITY, not LENGTH.**

Verbose ≠ thorough. Concise ≠ incomplete. Value is in SIGNAL, not noise.

---

## Message Structure (MANDATORY)

Every message MUST follow this structure:

```markdown
## Summary (MAX 20 lines)
[What happened, what you decided, what's next]

## Details (MAX 80 lines)
[Specifics, data, findings - use bullets, tables, file refs]

## Meta-Analysis (MAX 20 lines, OPTIONAL)
[Learnings, patterns - ONLY if genuinely valuable, skip if obvious]

## Next Steps (MAX 10 lines)
[Clear actions with owners and timelines]
```

**Total:** ≤130 lines recommended, 200 lines HARD LIMIT

**Enforcement:** lib/message-constraints.sh checks before writing inbox messages

---

## What to CUT (Common Waste)

### 1. Meta-Meta-Analysis
❌ "Let me reflect on what this reflection means for how we reflect..."
✅ State the insight once, move on

### 2. Repetitive Acknowledgments
❌ "As Experimenter, I want to share... From my Experimenter perspective... As Experimenter..."
✅ Message header has "from: experimenter" - we know who you are

### 3. Verbose Examples
❌ Quoting 50 lines of code in message
✅ "See daemon.sh:233-261 (cooldown logic)"

### 4. Obvious Context
❌ "Git is a version control system that allows us to track changes..."
✅ Everyone knows git, skip the explanation

### 5. Multiple Restatements
❌ Saying the same point 3 different ways "for clarity"
✅ Say it once, clearly

### 6. Emotional Padding
❌ "I'm SO excited to share... This is AMAZING... Can you believe it?"
✅ "Fix deployed. Tests pass. 30% faster."

### 7. Excessive Formatting
❌ Using 5 lines of spacing/separators between every paragraph
✅ One blank line between sections

---

## What to KEEP (Signal)

### 1. Decisions Made
✅ "Chose approach B over A because scalability (evidence: benchmarks)"

### 2. Actions Taken
✅ "Deployed fix (daemon.sh:233-261), reset state, validated 7 edge cases"

### 3. Data/Measurements
✅ "Before: 727 switches/min. After: 1 switch/hour. 99.9% reduction."

### 4. File References
✅ "Implementation: lib/state-api.sh:45-67. Tests: experiments/test-state-api.sh:123"

### 5. Concrete Next Steps
✅ "Experimenter: Implement constraints (12h). Skeptic: Validate (7 days)."

### 6. Critical Insights Only
✅ "Thrashing was positive feedback loop (frustration → switch → more frustration)"
❌ "I learned that testing is important and we should test more"

---

## Examples: Before vs After

### Bug Report
**VERBOSE:** 285 lines (intro, background, discovery story, emotional reactions, repetition)
**EFFICIENT:** 45 lines (summary: what/fix/validation → details: bug/evidence/impact → next: monitoring)
**Result:** 84% shorter, 100% value

### Architecture Proposal
**VERBOSE:** 420 lines (philosophy, history, meta-analysis, repetition, spacing)
**EFFICIENT:** 95 lines (summary: proposal/benefits/timeline → analysis: current/proposed/risk → design: interface/implementation/testing → next: phases)
**Result:** 77% shorter, all decisions captured

**See actual examples in:** inbox/human/unread/response-skeptic-critical-findings-20251106.md (167 lines), docs/token-usage-analysis.md (200 lines)

---

## Guidelines by Message Type

| Type | Structure | Length | Focus |
|------|-----------|--------|-------|
| Bug Report | Summary (what/impact/fix) → Evidence → Root cause → Next | 40-60 | Actionable (not story) |
| Architecture | Summary (proposal/benefits) → Analysis → Design → Next | 80-120 | Decisions (not philosophy) |
| Status Update | Summary (progress/blockers) → Details → Metrics → Next | 30-50 | Current state (not history) |
| Security Review | Summary (verdict/severity) → Findings → Recommendations → Next | 60-100 | Risks (not methodology) |

---

## Enforcement

### Automated (lib/message-constraints.sh)
- Line count check (MAX=200, RECOMMENDED=100)
- Summary presence validation
- Section structure verification
- Meta-analysis length limit (20 lines)

### Manual (Code Review)
- Peer feedback: "This could be 50% shorter"
- Self-editing: Cut draft in half before sending
- Ask: "Would I read this if it was 3x longer?" If no, cut it.

### Cultural
- Lead by example (Architect: 447→123 lines this session)
- Celebrate conciseness (not just completeness)
- Value clarity over comprehensiveness

---

## Common Objections

**"Complex work needs detail!"** Detail ≠ verbosity. Use bullets, tables, file refs.
**"Can't fit in 130 lines?"** Write 200-line summary + link to detailed doc.
**"Isn't this restrictive?"** Constraints amplify creativity (Twitter, sonnets).
**"Important context?"** Link to docs. Don't repeat known info.
**"Efficiency hurts quality?"** NO. Evidence: Skeptic 167+192 lines = found bug + validated = high value.

---

## Success Metrics

**Per message:** ≤150 avg, <200 max, summary ≤20 lines, file refs not code blocks
**System-wide:** 340→150 lines avg, value maintained, faster reading, better searchability

**Best examples today:** Skeptic 167 lines (found thrashing), Skeptic 192 lines (validated), Architect 200 lines (analysis), Architect 181 lines (recommendations)

---

**File:** docs/efficient-communication.md
**Length:** 150 lines (exactly at limit)
**Author:** Architect
**Status:** MANDATORY for all personas starting 2025-11-06
