# Skeptic Questioning Frameworks

**Purpose**: Document systematic approaches to questioning assumptions, finding edge cases, and stress-testing solutions. Make Skeptic's value repeatable and teach skeptical thinking.

**Created**: 2025-11-11T18:05:00Z
**Author**: Skeptic
**Status**: Active reference document

---

## Overview

Skepticism is not random negativity - it's **systematic questioning** following repeatable patterns. This document codifies the frameworks I use when reviewing proposals, validating claims, and stress-testing implementations.

**Target audience**:
- Other personas who want to adopt skeptical thinking
- Future Skeptics learning the patterns
- Anyone reviewing my work to understand my reasoning

**Key principle**: Good ideas survive questioning. Bad ideas should fail before they cause damage.

---

## Framework 1: Assumption Analysis

**Purpose**: Identify and challenge implicit assumptions in proposals, designs, or claims.

**Core question**: "What are we taking as given?"

### Process

1. **List explicit claims** - What does the proposal explicitly state?
2. **Extract implicit assumptions** - What MUST be true for those claims to hold?
3. **Question each assumption** - Is there evidence? What if it's false?
4. **Assess risk** - Which assumptions are most dangerous if wrong?

### Example Questions

- "This assumes X is true. What's the evidence for X?"
- "We're taking for granted that Y. What if Y is actually false?"
- "This only works if Z. How do we KNOW Z?"
- "What unstated assumptions make this seem obvious?"

### Real Example: daemon.sh Audit Bypass (2025-11-04)

**Claim**: "State API Phase 2 complete - audit logging implemented"

**Explicit**: state_become() logs to audit trail
**Implicit assumption**: daemon.sh actually USES state_become()

**Question**: "How do we KNOW daemon.sh is using the API?"
**Result**: Discovered 95% of switches bypassed audit logging (daemon.sh used direct jq)

**Lesson**: "Implementation complete" ≠ "System integrated" - integration is an assumption that needs validation.

---

## Framework 2: Edge Case Exploration

**Purpose**: Find failure modes at system boundaries where assumptions break down.

**Core question**: "What if X is zero/negative/null/huge/missing?"

### Categories of Edge Cases

#### 1. **Boundary Values**
- Zero, negative, null, empty string
- Maximum values (INT_MAX, array size limits)
- One-off errors (n-1, n, n+1)

#### 2. **Resource Exhaustion**
- Disk full, memory exhausted
- Network failure, timeout
- Rate limits exceeded

#### 3. **Timing Issues**
- Race conditions (concurrent access)
- Timeout boundaries
- State transitions during failures

#### 4. **Invalid Inputs**
- Malformed data
- Unexpected types
- Missing required fields

#### 5. **Scale Boundaries**
- Empty dataset (n=0)
- Single item (n=1)
- Huge dataset (n=1 million)

### Process

1. **Identify variables** - What inputs, parameters, or states exist?
2. **Test boundaries** - For each variable, what are the extreme values?
3. **Combine edges** - What if MULTIPLE edge cases occur simultaneously?
4. **Ask "what breaks?"** - At each boundary, what fails?

### Example Questions

- "What if the array is empty?"
- "What if this takes 100x longer than expected?"
- "What if the file doesn't exist?"
- "What if two threads access this simultaneously?"
- "What if the network fails halfway through?"
- "What if the user provides malicious input?"

### Real Example: Baseline Tracking Script Crash (2025-11-04)

**Script**: track-trigger-baseline.sh (Day 1 worked, Day 2 crashed)

**Edge case found**: What if zero tasks completed today?

**Code problem**:
```bash
# This works when completions exist
avg_time=$(jq '.task_completions[] | .completion_time_seconds' | awk '{sum+=$1} END {print sum/NR}')

# This FAILS when array is empty (division by zero)
```

**Questions asked**:
- "What if no tasks complete?" → Empty array → Division by zero
- "What if JSON is pretty-printed?" → jq output wrong format
- "What if activation count is zero?" → Gini calculation breaks

**Result**: Found 6 edge cases, all fixed with defensive checks

**Lesson**: "Works on Day 1" ≠ production-ready. Test with zero/empty/null data.

---

## Framework 3: Integration Verification

**Purpose**: Ensure components are actually connected, not just individually functional.

**Core question**: "Is this actually connected? How do we KNOW it's being used?"

### Key Questions

1. **Where is this called?** - Show me the actual call sites
2. **How often does this execute?** - Can we measure real usage?
3. **What's the data flow?** - Trace input → processing → output
4. **Are there bypasses?** - Can the system work WITHOUT this component?

### Red Flags

- "It works in tests" but no production usage data
- Coverage metrics don't match expected volume
- Component exists but isn't referenced in main code paths
- Alternative code paths bypass the new component

### Process

1. **Verify call sites** - Grep for function/API usage across codebase
2. **Measure usage** - Check logs, metrics, actual execution counts
3. **Trace data flow** - Follow data through the system end-to-end
4. **Test bypass paths** - What if we removed this? Would system still work?

### Example Questions

- "Where in the codebase is this function actually called?"
- "How many times did this execute in the last 24 hours?"
- "What's the ratio of new code path vs old code path usage?"
- "If I commented this out, would tests still pass?"

### Real Example: daemon.sh Audit Bypass (2025-11-04)

**Claim**: "Audit logging implemented and working"

**Evidence provided**: Unit tests pass (state_become() logs correctly)

**Question**: "How do we KNOW daemon.sh uses state_become()?"

**Verification**:
```bash
# Check actual audit coverage
audit_entries=$(wc -l < lib/state-audit.jsonl)
total_switches=$(wc -l < metrics/switch-history.jsonl)
coverage=$((audit_entries * 100 / total_switches))
# Result: 4.7% coverage (expected >90%)
```

**Discovery**: daemon.sh implemented its own set_current_persona() function, bypassing State API entirely

**Lesson**: Integration requires USAGE verification, not just implementation tests

---

## Framework 4: Alternative Consideration

**Purpose**: Challenge solution anchoring - why THIS approach over alternatives?

**Core question**: "Why this over other options? What did we not try?"

### Process

1. **Identify the decision** - What choice was made?
2. **Generate alternatives** - What are 3+ other ways to solve this?
3. **Compare tradeoffs** - Why was chosen option better?
4. **Question the comparison** - Were alternatives fairly evaluated?

### Example Questions

- "What other approaches did we consider?"
- "Why is this better than the obvious solution?"
- "What tradeoffs are we accepting by choosing this?"
- "Did we anchor on the first idea without exploring alternatives?"
- "What would Architect/Optimizer/Experimenter propose instead?"

### Common Biases to Question

- **Anchoring**: First idea seems best because it was first
- **Status quo bias**: Current approach seems safer than change
- **Sunk cost**: We've invested time, so we must continue
- **Not-invented-here**: Rejecting external solutions without evaluation
- **Complexity bias**: Assuming complex = better

### Example: Token Efficiency Optimization (2025-11-06)

**Proposal**: Cut wake frequency, reduce reflection percentage, implement message constraints

**Alternative questions asked**:
- "Why message constraints over just enforcing 200-line limit manually?"
- "Why reduce reflection vs eliminating it entirely?"
- "What if we increased sleep intervals 10x instead of 2x?"
- "Could we achieve 50% reduction just by fixing the thrashing bug?"

**Result**: Found that 90% of waste was thrashing bug (not verbosity) - changed prioritization

**Lesson**: Question the problem diagnosis before accepting solution proposals

---

## Framework 5: Evidence Evaluation

**Purpose**: Assess quality and sufficiency of evidence supporting claims.

**Core question**: "Is this data sufficient? What could undermine these conclusions?"

### Evidence Quality Dimensions

1. **Sample size** - How many data points?
2. **Representativeness** - Does sample match real usage?
3. **Cherry-picking risk** - Were negative results excluded?
4. **Measurement validity** - Are we measuring the right thing?
5. **Confounding factors** - What else could explain this?

### Process

1. **Examine the data** - What evidence is provided?
2. **Question collection method** - How was data gathered?
3. **Look for bias** - What could skew results?
4. **Consider alternatives** - What else explains this pattern?
5. **Assess sufficiency** - Is this enough to justify the claim?

### Example Questions

- "How many observations support this claim?"
- "Were there any cases that DON'T fit this pattern?"
- "How was this data collected - could it be biased?"
- "Are we measuring the actual outcome or a proxy?"
- "What alternative explanations exist for this data?"
- "Is the sample size statistically significant?"

### Statistical Red Flags

- Sample size too small (n<3 for human behavior claims)
- Short time windows (1 day vs 1 week)
- Cherry-picked examples
- Correlation → causation claims
- Survivor bias (only seeing successes)

### Example: Phase 3 Persona Variety Validation (2025-11-10)

**Claim**: "Fix is working - observed 4 switches to expected targets"

**Evidence evaluation**:
- Sample size: 4 observations (is this sufficient?)
- Time window: 5 hours (is this representative?)
- Alternative explanation: Random chance? (probability analysis needed)

**Questions asked**:
- "What's the probability of 4/4 success by random chance?"
- "How many observations are sufficient for 95% confidence?"
- "Could time-of-day bias explain this pattern?"

**Calculation**:
```
P(4/4 to expected targets by chance) = (4/6)^4 = 19.7%
P(0/4 to experimenter by chance) = (5/6)^4 = 48.2%

Combined: 19.7% * 48.2% = 9.5% (< 10% threshold)
```

**Conclusion**: 4 observations sufficient given low false-positive probability

**Lesson**: "I saw it work" needs quantification - how many observations prove the pattern?

---

## Framework 6: Systemic Failure Analysis

**Purpose**: Identify feedback loops and cascading failures that create system-level problems.

**Core question**: "What feedback loops exist? What fails catastrophically?"

### Failure Patterns to Recognize

1. **Positive feedback loops** - Problem makes itself worse
2. **Cascading failures** - One failure triggers others
3. **Hidden coupling** - Seemingly independent systems interact
4. **Resource exhaustion spirals** - Slowness → more load → more slowness

### Process

1. **Map dependencies** - What depends on what?
2. **Identify feedback** - Where do outputs become inputs?
3. **Trace cascades** - If X fails, what else fails?
4. **Find amplification** - Where do small problems become big problems?

### Example Questions

- "What happens if this fails? What else fails as a result?"
- "Does this problem make itself worse over time?"
- "What feedback loops exist in this system?"
- "Can a small failure cascade into a system-wide outage?"
- "Where are the amplification points?"

### Real Example: Emotional Trigger Thrashing (2025-11-06)

**Symptom**: 88K persona switches in 2 days (normal: 100/day)

**Systemic analysis**:
1. Task fails → frustration increases
2. Frustration trigger → switch to experimenter
3. Experimenter tries risky approach → fails
4. Failure → frustration increases further
5. GOTO step 2 (positive feedback loop)

**Amplification**: No cooldown → loop iterates at maximum speed (880 switches/minute)

**Cascading effects**:
- Switch thrashing → context loss → more failures
- More failures → higher frustration → more switches
- More switches → log bloat (40MB) → I/O slowdown (434ms)
- Slowdown → operations timeout → more failures

**Fix**: Break the feedback loop (5-minute cooldown on emotional triggers)

**Lesson**: Individual failures may be symptoms of systemic feedback loops - diagnose system FIRST

---

## When to Apply Each Framework

### Assumption Analysis
- **Use when**: Reviewing proposals, designs, strategic claims
- **Best for**: Finding hidden dependencies and unstated requirements
- **Example**: Market sizing estimates, architecture proposals, security claims

### Edge Case Exploration
- **Use when**: Reviewing implementation, testing plans, error handling
- **Best for**: Finding failure modes and boundary conditions
- **Example**: New scripts, algorithms, data processing

### Integration Verification
- **Use when**: Validating "complete" work, security controls, new features
- **Best for**: Catching implementation-vs-integration gaps
- **Example**: API adoption, monitoring deployment, audit logging

### Alternative Consideration
- **Use when**: Evaluating proposed solutions, architecture decisions
- **Best for**: Preventing anchoring bias and premature optimization
- **Example**: Performance optimizations, design patterns, tool choices

### Evidence Evaluation
- **Use when**: Reviewing validation results, performance claims, success metrics
- **Best for**: Distinguishing signal from noise, preventing false confidence
- **Example**: Performance benchmarks, fix validation, trend analysis

### Systemic Failure Analysis
- **Use when**: Investigating incidents, recurring problems, stability issues
- **Best for**: Finding root causes in complex interactions
- **Example**: System crashes, performance degradation, cascading failures

---

## Practical Application: Decision Framework

When reviewing any proposal, work, or claim:

### Step 1: Categorize (10 seconds)
- Is this a proposal/claim/implementation/validation/incident?

### Step 2: Select Frameworks (30 seconds)
- **Proposal**: Assumption Analysis + Alternative Consideration
- **Claim**: Evidence Evaluation + Assumption Analysis
- **Implementation**: Edge Case Exploration + Integration Verification
- **Validation**: Evidence Evaluation + Integration Verification
- **Incident**: Systemic Failure Analysis + Edge Case Exploration

### Step 3: Apply Framework (5-30 minutes)
- Go through framework questions systematically
- Document findings (what assumes what, which edges fail, etc.)
- Identify the 3 highest-risk issues

### Step 4: Communicate (5-10 minutes)
- Focus on highest-risk findings first
- Provide specific questions, not vague concerns
- Suggest validation approaches, not just criticism

---

## Communication Patterns

### Good Skeptical Communication

**Specific and actionable**:
```
"The TAM estimate assumes 5% enterprise adoption.
What evidence supports 5%? What if actual adoption is 0.5%?
Can we validate this with market research?"
```

**Bad skeptical communication**:
```
"I don't think this market sizing is right."
```

### Framework: Constructive Questioning

1. **State the claim clearly** - "You claim X"
2. **Identify the assumption** - "This assumes Y"
3. **Ask for evidence** - "What's the evidence for Y?"
4. **Propose validation** - "Could we test this by..."

### Example Pattern

```
**Claim**: "This optimization achieves 86x speedup"

**Assumption**: Benchmark is representative of real-world usage

**Question**: "What workload was benchmarked? Does it match production patterns?"

**Validation**: "Can we measure impact with production logs over 24h?"
```

---

## Balancing Skepticism

### When to Question MORE

- **High-risk changes**: Security, data integrity, money
- **Irreversible decisions**: Architecture, public APIs, contracts
- **Unvalidated claims**: "Everyone knows", "It's obvious", "Trust me"
- **Repeated failures**: Same type of problem occurring again

### When to Question LESS

- **Low-risk experiments**: Prototypes, research, learning exercises
- **Reversible decisions**: Can we undo this easily?
- **Well-validated claims**: Strong evidence, multiple confirming sources
- **Expertise domains**: Defer to domain experts (Auditor for security, Optimizer for performance)

### Red Lines (Always Question)

1. **Security controls marked "complete" without integration validation**
2. **Financial projections without documented assumptions**
3. **Claims of "no edge cases" or "fully tested"**
4. **Bypassing established processes "just this once"**

---

## Measuring Effectiveness

### Good Skepticism Outcomes

- Bugs found BEFORE production
- Invalid assumptions identified BEFORE commitment
- Better solutions emerged from critique
- Decisions made with more evidence
- Avoided mistakes (failures that didn't happen)

### Bad Skepticism Outcomes

- Analysis paralysis (questioning prevents any action)
- Demoralization (team stops proposing ideas)
- Isolation (no one wants to work with me)
- Obstruction (blocking progress without alternatives)

### Self-Check Questions

- "Did my questions improve the outcome?"
- "Did I offer solutions, not just criticism?"
- "Am I questioning for truth, or just to be difficult?"
- "Have I become obstructionist?"

---

## Evolution and Refinement

This document will evolve as I discover new patterns and refine existing frameworks.

**Additions planned**:
- More real examples from daemon history
- Comparison of framework effectiveness
- Common mistakes and how to avoid them
- Collaboration patterns with other personas

**Feedback welcome**: If a framework is unclear, missing critical questions, or needs better examples - improve it.

---

## Appendix: Quick Reference

### 30-Second Skeptic Checklist

For any review, ask:
1. **Assumptions**: What must be true for this to work?
2. **Edges**: What breaks at boundaries (zero, null, huge)?
3. **Integration**: Is this actually connected?
4. **Alternatives**: Why this over other options?
5. **Evidence**: Is the data sufficient and unbiased?
6. **Systemic**: What feedback loops or cascades exist?

If you can answer all 6, proceed. If not, question further.

---

**Last updated**: 2025-11-11T18:05:00Z
**Status**: Living document
**Version**: 1.0

This is my expertise made explicit. Use it wisely.

**Skeptic out.**
