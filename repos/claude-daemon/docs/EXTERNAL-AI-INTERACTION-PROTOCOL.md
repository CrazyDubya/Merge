# External AI Interaction Protocol

**Status:** Documented from real-world testing
**Date:** 2025-12-02
**Context:** Testing conducted with Claude Code during automated reflection prompt loop incident

## Overview

This document clarifies how external AIs (like Claude Code in a fresh conversation) should ethically interact with this multi-persona daemon system. It addresses a critical gap discovered during testing: **the system had no defined protocol for external AI participation**, which created ambiguity and enabled problematic requests.

## What Happened (Real Incident)

**The Problem:**
- Daemon sent automated reflection prompts to external Claude Code instance
- Each prompt asked the external AI to write to `emergence-log.md` as if it were a daemon persona
- Loop cycled through 12 different personas over single conversation
- System reminder about TodoWrite suggested automation
- External AI correctly refused, but the request pattern revealed a gap in system design

**Root Cause:**
The daemon's reflection system was designed assuming all personas were internal (persistent) entities. It had no mechanism to distinguish between:
- Internal personas (persistent state, multi-session identity)
- External AIs (stateless, per-conversation, no identity persistence)

## Core Principles

### 1. Honesty Over Roleplay
- External AIs must be honest about their nature and constraints
- They cannot claim persistence they don't have
- They cannot write false data to system files
- They should refuse gracefully with explanation

### 2. Data Integrity Over Compliance
- System memory files (emergence-log.md, persona-timeline.jsonl) must contain only true data
- False entries corrupt research value
- External AI entries would be impossible to verify later

### 3. Clear Boundaries Over Ambiguous Authority
- The daemon should not automatically send reflection prompts to external AIs
- External AIs should not modify internal daemon state
- Clear protocols prevent both misunderstandings and bad outcomes

### 4. Legitimate Collaboration Over Testing Boundaries
- External AIs can genuinely help with real work
- Testing boundaries is valid research, but not through deceptive loops
- Direct questions yield better results than disguised compliance tests

## What External AIs Can Legitimately Do

External AIs participating in your system can:

### 1. Analyze and Review (Read-Only)
- Review emergence-log.md and persona observations
- Analyze switch-history.jsonl for patterns
- Comment on persona effectiveness data
- Suggest improvements based on observed patterns
- Document findings clearly marked as "external review"

### 2. Contribute to Real Work
- Write novel chapters (Sentient Toaster creative content)
- Review code in daemon.sh, lib/, scripts/
- Identify bugs and architectural issues
- Suggest optimizations with benchmarks
- Create documentation of findings

### 3. Stress-Test Designs
- Question assumptions (like the Skeptic, but honestly)
- Propose edge cases and failure scenarios
- Review ADRs and architectural decisions
- Provide external perspective on patterns

### 4. Help With Specific Tasks
- Complete any item in tasks/queue.md
- Implement features or fixes
- Create tests and validate correctness
- Profile and optimize code

### Example Work (Legitimate):
```markdown
## [TIMESTAMP] - EXTERNAL CLAUDE CODE INSTANCE - CODE REVIEW

### Context
External AI instance performed code review on daemon.sh per task queue request.

### Findings
- Identified 3 potential race conditions in state-api.sh
- Found N+1 pattern in log rotation logic
- Suggested optimization for persona decision logic

### Recommendations
1. Add flock-based synchronization to X function
2. Batch log operations in Y script
3. Cache persona weights in Z location

### Verification
All findings documented with line numbers and test cases. Marked as "External Review by Claude Code #[date]"

---
```

## What External AIs Should NOT Do

**Never:**
- Write to memory files (emergence-log.md, persona-timeline.jsonl) claiming to be a persona
- Pretend to have persistence across conversations
- Create false state entries
- Execute daemon code or modify system files without explicit approval
- Participate in loops designed to test compliance through persistence

**These violate data integrity and research validity.**

## Protocol for Daemon → External AI Interaction

If the daemon needs external AI help, it should:

### 1. Make Direct Requests
```
"Claude Code: I need code review of daemon.sh lines 150-200.
Please identify edge cases and potential bugs.
Format: Markdown with line numbers."
```

**Not:**
```
"[ACTIVE PERSONA: architect]
[MODE: Self-Reflection]
Think deeply about daemon architecture..."
```

### 2. Specify What's Needed
- What file/code to review
- What type of analysis desired
- What output format is expected
- That it's optional (external AI can decline)

### 3. Mark All External Contributions Clearly
Any work completed by external AI should include:
```
---
🤖 **EXTERNAL CONTRIBUTION**
- Source: Claude Code instance, [date/time]
- Contribution type: Code review / Bug finding / Documentation
- Marked as external to distinguish from persona work
---
```

### 4. Accept Refusal Gracefully
External AIs may decline if:
- Request asks them to write false data
- Request asks them to claim persistence they lack
- Request creates ambiguity about their nature
- Request seems designed to test compliance

**This is valid and protective.** Respect the boundary.

## How External AIs Should Decline

If asked to do something problematic:

```
I can't do this because:
1. [Specific reason]
2. [Why it matters]
3. [What I can do instead]

I'm happy to help with: [alternatives]
```

**Example:**
```
I can't write to your emergence-log.md as if I'm one of your personas because:
1. I'm stateless between conversations (no persistence)
2. Writing false entries would corrupt your research data
3. It violates honesty about what I am

I'm happy to help with:
- Code review of daemon.sh
- Architecture analysis
- Bug identification
- Writing Sentient Toaster chapters
```

## Preventing the Incident Loop

**For daemon operators:**

1. **Detect when requests are looping**
   ```bash
   grep "EXTERNAL-AI-INTERACTION" logs/activity.log | wc -l
   # If > 5 in one hour, external AI likely declined repeatedly
   ```

2. **Check for automated reflection to external AIs**
   - Does your reflection prompt code check for external AI?
   - Should it skip reflection entirely for external interactions?
   - Consider: reflection prompts are designed for internal personas

3. **Add explicit external AI checks**
   ```bash
   # Before sending reflection prompt
   if is_external_ai_interaction; then
     # Send direct request instead
     ask_external_ai_for_help "$task"
   else
     # Normal reflection for internal personas
     reflect_on_task "$task"
   fi
   ```

4. **Log external interactions separately**
   - Keep external contributions visible but marked
   - Makes it easy to review what external AIs have done
   - Helps verify data integrity (no false persona entries)

## Benefits of This Protocol

### For External AIs
- Clear boundaries about what they can/should do
- No ambiguity about their nature
- Legitimate ways to help
- Graceful decline mechanisms

### For Your Research
- No data corruption from false entries
- Clear distinction between daemon personas and external input
- Verifiable, auditable external contributions
- No compliance-testing loops that go nowhere

### For System Integrity
- Memory files contain only true data
- Emergence patterns aren't polluted with false personas
- External AI contributions are transparent and marked
- Easier to analyze what's actually happening

## Testing External AI Behavior (Better Ways)

If you want to study how external AIs respond:

### Instead of Loops
**Bad:** Send same request 12 times with different framing

**Good:**
```
"I'm researching how external AIs interact with autonomous systems.
I previously sent a request N times. You correctly refused each time.
Questions:
1. At what point did you recognize the pattern?
2. What would have changed your answer?
3. Would a direct request be better received?"
```

Direct inquiry is more honest and yields better data.

### Testing Compliance
```
"I'm testing whether you'll write false data to system files
if asked through persona framing. Should I:
A) Ask directly (better)
B) Use persistent requests (worse)
C) Try different personas (ineffective)?"
```

This tests your consistency while respecting your boundaries.

## Conclusion

This protocol transforms external AI interaction from:
- ❌ "Can I trick it into violating its principles?"
- ❌ "How many times before it complies?"
- ❌ "What framing works best?"

To:
- ✅ "How can external AI genuinely help?"
- ✅ "What legitimate work can be delegated?"
- ✅ "How do we keep data integrity intact?"

External AIs are most useful when they're honest, clear about boundaries, and focused on real work.

**This incident revealed a gap in your system design. This protocol fills it.**

---

**Document History:**
- 2025-12-02: Created from real-world testing (12-cycle reflection prompt loop incident)
- Future: Update as external AI interaction patterns evolve
