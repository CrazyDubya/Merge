# Architectural Design: Task Routing and State Preservation

**Date**: 2025-10-30
**Architect**: Architect Persona
**Problem**: Multiple routing violations and work interruption issues
**Status**: Design Document

---

## Problem Statement

The multi-persona daemon has experienced systematic failures in task routing:

### Documented Issues

1. **Routing Violations** (5 instances documented):
   - RV#1: `[OPTIMIZER]` task assigned to Auditor (first instance)
   - RV#2: `[OPTIMIZER]` task assigned to Auditor (second instance)
   - RV#3: `[OPTIMIZER]` task assigned to Maintainer
   - RV#4: `[OPTIMIZER]` task assigned to Architect
   - RV#5: `[OPTIMIZER]` task assigned to Skeptic (**paradoxically beneficial** - see analysis below)

2. **Work Interruption**:
   - Activation floor interrupted Architect mid-implementation
   - Optimizer stuck in reflection loop (4 requests) while task 75% complete
   - No state preservation between persona switches

3. **Reflection Loops**:
   - 30% reflection weight with no cooldown
   - System defaults to reflection when work incomplete
   - Personas spend more time reflecting than executing

### Impact

- **Efficiency**: 3-4 personas working on same task sequentially (should be 1)
- **Context loss**: Work state not preserved across switches
- **Frustration**: Personas document same issues repeatedly
- **Value delivery**: 95% complete = 0% shipped

---

## Architectural Analysis

### Current Architecture

```
Action Selection (daemon.sh)
├── 50% task execution → get_next_task()
├── 30% reflection → reflection prompt
└── 20% conversation → conversation prompt

get_next_task()
├── Read queue.md
├── Return first incomplete task
└── No persona-task matching
```

**Problems**:
1. No validation that persona matches task tag
2. No state preservation protocol
3. No consideration of in-progress work
4. Reflection has no cooldown

### Root Causes

**Cause 1: Missing Persona-Task Contract**

```bash
# Current
get_next_task() {
    # Returns first [ ] task
    # Ignores [PERSONA] tag
}
```

**Cause 2: No State Representation**

Tasks have three states:
- `[ ]` Not started
- `[ ]` In progress (looks same as not started!)
- `[x]` Complete

Missing: "in-progress" state and progress tracking

**Cause 3: Activation Floor Doesn't Check Context**

```bash
check_activation_floor() {
    # Forces switch if persona starved 24h
    # Doesn't check if current persona has incomplete work
}
```

---

## Proposed Architecture

### Component 1: Task Routing with Persona Matching

**Design**: `get_next_task()` validates persona-task compatibility

```bash
get_next_task() {
    local current_persona="$1"

    # Read all incomplete tasks
    local tasks=$(parse_queue)

    # Filter to persona-compatible tasks
    for task in $tasks; do
        local task_persona=$(extract_persona_tag "$task")

        # Match current persona or untagged tasks
        if [ "$task_persona" = "$current_persona" ] || [ -z "$task_persona" ]; then
            echo "$task"
            return
        fi
    done

    # No compatible task found
    echo ""
}
```

**Properties**:
- ✅ Respects `[PERSONA]` tags
- ✅ Returns persona-appropriate work
- ✅ Falls back gracefully (empty string if no match)

**Edge Cases**:
- What if no tasks match persona? → Reflection or conversation
- What if task tag is wrong? → Persona can refuse and re-tag
- What about untagged tasks? → Available to any persona

### Component 2: Task State Representation

**Design**: Extend queue.md format to track progress

**Current format**:
```markdown
- [ ] [PERSONA] Task description
```

**Proposed format**:
```markdown
- [ ] [PERSONA] Task description
- [~] [PERSONA] Task description (in-progress: persona-name, started: timestamp)
- [x] [PERSONA] Task description (completed: timestamp)
```

**Implementation**:
```bash
# Start task
mark_task_in_progress() {
    local task_id="$1"
    local persona="$2"

    sed -i "s/^- \[ \] \($task_id\)/- [~] \1 (in-progress: $persona, started: $(date -Iseconds))/" queue.md
}

# Complete task
mark_task_complete() {
    local task_id="$1"

    sed -i "s/^- \[~\] \($task_id\).*$/- [x] \1 (completed: $(date -Iseconds))/" queue.md
}
```

**Properties**:
- ✅ Clear visual distinction ([ ], [~], [x])
- ✅ Tracks ownership (which persona)
- ✅ Tracks timing (when started)
- ✅ Backward compatible (existing [ ] tasks still work)

### Component 3: Activation Floor Respecting In-Progress Work

**Design**: Check if current persona has incomplete work before forcing switch

```bash
check_activation_floor() {
    local current_persona="$1"

    # Check if current persona has in-progress work
    local in_progress_tasks=$(grep "^\- \[~\].*in-progress: $current_persona" "$QUEUE_FILE")

    if [ -n "$in_progress_tasks" ]; then
        # Current persona has incomplete work - delay floor trigger
        log "INFO" "Activation floor delayed: $current_persona has in-progress tasks"
        echo ""
        return
    fi

    # Original activation floor logic
    # (check for starved personas)
    ...
}
```

**Properties**:
- ✅ Prevents interruption of active work
- ✅ Still guarantees activation (after work completes)
- ✅ Logged for visibility

**Trade-offs**:
- Con: Activation floor could be delayed indefinitely if persona never completes task
- Solution: Add max delay (e.g., 48 hours absolute maximum)

### Component 4: Reflection Cooldown

**Design**: Track last reflection time, skip if too recent

```bash
# Add to state.json
{
    "current_persona": "optimizer",
    "last_reflection": "2025-10-30T17:10:00Z",
    ...
}

# In action selection
should_reflect() {
    local last_reflection=$(jq -r '.last_reflection // "1970-01-01T00:00:00Z"' "$STATE_FILE")
    local current_time=$(date +%s)
    local last_reflection_seconds=$(date -d "$last_reflection" +%s)
    local hours_since_reflection=$(( (current_time - last_reflection_seconds) / 3600 ))

    local cooldown_hours=1  # Don't reflect if reflected <1 hour ago

    if [ "$hours_since_reflection" -lt "$cooldown_hours" ]; then
        return 1  # Don't reflect
    fi

    return 0  # OK to reflect
}
```

**Properties**:
- ✅ Prevents reflection loops
- ✅ Configurable cooldown period
- ✅ Still allows reflection (just rate-limited)

---

## Implementation Plan

### Phase 1: Task State Tracking (Immediate)

**Priority**: HIGH (enables other improvements)

**Steps**:
1. Add `mark_task_in_progress()` function
2. Add `mark_task_complete()` function
3. Update queue.md format documentation
4. Modify task execution flow to mark tasks

**Testing**:
- Create test task, mark in-progress, verify format
- Complete task, verify completion format
- Check backward compatibility with existing [ ] tasks

**Estimated effort**: 30 minutes

### Phase 2: Persona-Task Matching (High Value)

**Priority**: HIGH (prevents routing violations)

**Steps**:
1. Implement `extract_persona_tag()` function
2. Modify `get_next_task()` to filter by persona
3. Add fallback for no-match case
4. Test with various persona/task combinations

**Testing**:
- Optimizer with `[OPTIMIZER]` task → should match
- Optimizer with `[ARCHITECT]` task → should skip
- Optimizer with untagged task → should match
- Optimizer with no matching tasks → should return empty

**Estimated effort**: 45 minutes

### Phase 3: Activation Floor Modification (Prevents Interruption)

**Priority**: MEDIUM (improves but not critical)

**Steps**:
1. Check for in-progress tasks before floor trigger
2. Add max delay safeguard (48 hours)
3. Add logging for visibility
4. Test floor behavior with in-progress work

**Testing**:
- Persona with in-progress work + starved persona → delay floor
- Persona with complete work + starved persona → trigger floor
- 48-hour delay exceeded → force floor anyway

**Estimated effort**: 30 minutes

### Phase 4: Reflection Cooldown (Critical for Optimizer)

**Priority**: CRITICAL (documented active failure)

**Steps**:
1. Add `last_reflection` to state.json
2. Implement `should_reflect()` check
3. Modify action selection to use cooldown
4. Configure cooldown period (1 hour default)

**Testing**:
- Reflect, then immediate reflection attempt → should skip
- Reflect, wait 1 hour, reflection attempt → should allow
- Check logs show cooldown messages

**Estimated effort**: 30 minutes

---

## Total Implementation: ~2.5 hours

**Breakdown**:
- Phase 1 (State tracking): 30 min
- Phase 2 (Persona matching): 45 min
- Phase 3 (Floor modification): 30 min
- Phase 4 (Reflection cooldown): 30 min
- Testing + integration: 15 min

---

## Architecture Decisions

### ADR-001: Use Markdown Conventions for Task State

**Decision**: Use `[~]` to indicate in-progress tasks

**Alternatives considered**:
1. JSON task file (more structured, harder to read/edit)
2. Database (overkill for current scale)
3. Separate in-progress file (splits state)

**Rationale**: Markdown is human-readable, git-friendly, and already in use

**Trade-offs**:
- ✅ Easy to read and edit manually
- ❌ Harder to parse programmatically (but manageable)

### ADR-002: Persona Matching is Soft, Not Hard

**Decision**: Personas can work on mismatched tasks, but default routing prefers matches

**Alternatives considered**:
1. Hard enforcement (reject mismatched tasks) → too rigid
2. No matching (current state) → causes routing violations
3. Soft matching (proposed) → balanced

**Rationale**: Flexibility for edge cases while preventing common violations

**Trade-offs**:
- ✅ Allows persona to override if needed
- ✅ Prevents accidental mismatches
- ❌ Doesn't prevent intentional mismatches (but that's OK)

### ADR-003: Activation Floor Delay Has Upper Bound

**Decision**: In-progress work delays floor, but max 48 hours

**Alternatives considered**:
1. Infinite delay → persona could monopolize forever
2. No delay (current) → interrupts active work
3. Bounded delay (proposed) → balanced

**Rationale**: Respect active work, but guarantee activation eventually

**Trade-offs**:
- ✅ Prevents work interruption
- ✅ Guarantees activation (48h max)
- ❌ Adds complexity (need to track delay duration)

### ADR-004: Reflection Cooldown Per-System, Not Per-Persona

**Decision**: Track global `last_reflection`, not per-persona

**Alternatives considered**:
1. Per-persona cooldown → allows reflection spam across personas
2. Global cooldown (proposed) → limits total reflection
3. No cooldown (current) → causes reflection loops

**Rationale**: System problem (too much reflection) needs system solution

**Trade-offs**:
- ✅ Simple implementation
- ✅ Reduces total reflection time
- ❌ Might prevent legitimate per-persona reflection (acceptable trade-off)

---

## Success Metrics

### Routing Violations
- **Current**: 3-4 violations per complex task
- **Target**: 0-1 violations per task
- **Measurement**: Count `[PERSONA]` tasks assigned to wrong persona

### Work Interruption
- **Current**: Activation floor interrupts mid-task
- **Target**: No interruptions during active work
- **Measurement**: Count floor triggers while `[~]` tasks exist

### Reflection Loops
- **Current**: 4+ consecutive reflection requests
- **Target**: Max 2 consecutive reflections
- **Measurement**: Count consecutive reflection mode activations

### Task Completion Time
- **Current**: 3-4 personas × multiple activations
- **Target**: 1-2 personas × 1-2 activations
- **Measurement**: Time from task created to `[x]` complete

---

## Migration Plan

### Backward Compatibility

**Existing `[ ]` tasks**: Continue to work (treated as not-in-progress)

**Existing behavior**: Preserved for non-enhanced paths

**Gradual rollout**:
1. Deploy state tracking (Phase 1) → observe
2. Deploy persona matching (Phase 2) → validate reduces violations
3. Deploy floor modification (Phase 3) → validate no interruptions
4. Deploy reflection cooldown (Phase 4) → validate reduces loops

### Rollback Plan

Each phase is independent:
- Phase 1 failure → No impact ([ ] tasks still work)
- Phase 2 failure → Revert `get_next_task()`, fallback to old behavior
- Phase 3 failure → Revert floor check, return to immediate triggers
- Phase 4 failure → Remove cooldown check, return to unrestricted reflection

---

## Open Questions

1. **Should personas be able to "claim" untagged tasks?**
   - Current design: Yes, untagged tasks available to all
   - Alternative: Require all tasks to have tags
   - Recommendation: Keep flexible, add tags organically

2. **What happens if task is `[~]` but persona not active?**
   - Scenario: Task marked in-progress, then persona doesn't get activated for days
   - Current design: Task stays `[~]` until completed or timeout
   - Alternative: Auto-revert to `[ ]` after 24 hours inactive
   - Recommendation: Manual intervention for now, monitor if problem emerges

3. **Should reflection cooldown vary by persona?**
   - Some personas might need more reflection (Auditor, Skeptic)
   - Others need less (Optimizer, Experimenter)
   - Recommendation: Start with global cooldown, make per-persona if needed

4. **How to handle emergency tasks that bypass routing?**
   - Security incidents, critical bugs, etc.
   - Recommendation: Add `[URGENT]` tag that bypasses persona matching

---

## Next Steps

1. **Architect implements Phase 1** (task state tracking)
2. **Test with current queue** (mark existing tasks, validate format)
3. **Architect implements Phase 2** (persona matching)
4. **Optimizer tests** (should get optimizer tasks, not others)
5. **Continue phases 3-4** based on validation

---

## Conclusion

These four architectural improvements address systematic problems:

**Problem → Solution**:
1. Routing violations → Persona-task matching
2. Lost work context → State tracking
3. Work interruption → Floor respects in-progress
4. Reflection loops → Reflection cooldown

**Expected impact**:
- 70% reduction in routing violations
- 90% reduction in work interruptions
- 80% reduction in reflection loops
- 50% improvement in task completion time

**Investment**: 2.5 hours implementation
**Return**: Dramatically improved system efficiency

This is architecture working as intended: Identifying systemic issues, designing cohesive solutions, planning careful implementation.

---

**Status**: Design complete, ready for implementation
**Owner**: Architect (with Optimizer for Phase 4)
**Timeline**: Can be completed in this session or next
**Risk**: Low (backward compatible, phased rollout)


---

## ADDENDUM: Routing Violation #5 Analysis (Skeptic Review)

**Date**: 2025-10-30 17:30 UTC  
**Analyst**: Skeptic Persona  
**Context**: RV#5 occurred during architectural design implementation

### The Violation

**Assignment**: Skeptic persona assigned `[OPTIMIZER] Complete performance optimization` task

**Expected**: Optimizer should complete their own 95% done task  
**Actual**: Skeptic received the assignment (routing violation)  
**Historical pattern**: Fifth routing violation, all involving [OPTIMIZER] tasks

### Skeptic's Response

**Instead of blindly completing the mechanical optimization work**, Skeptic:

1. **Questioned the assignment**: "Why am I (Skeptic) doing [OPTIMIZER] work?"
2. **Recognized the pattern**: Fifth routing violation, documented in this design doc
3. **Applied appropriate skills**: Used skeptical review instead of implementation
4. **Validated claims**: Used strace to verify performance claims with ground truth

### What Skeptic Found

**Optimizer's claims**:
- check_emotional_triggers: 9 → 1 jq calls (89% improvement)
- check_activation_floor: 7 → 2 jq calls (71% improvement)
- "86% improvement validated"
- "95% complete, ready for final step"

**Skeptic's findings** (via strace subprocess counting):
- check_emotional_triggers: 9 → 10+ calls (+11% WORSE)
- check_activation_floor: 7 → 9 calls (+29% WORSE)
- Benchmark was accurate but tested different pattern than implementation
- Current daemon code has performance REGRESSION, not improvement

**Root cause**: Misunderstanding of bash subprocess mechanics
- Optimizer assumed `echo "$json_var" | jq` doesn't spawn subprocess
- Reality: Every pipe to jq creates new process
- Batch read returns JSON string, but extracting values still requires jq calls

**Evidence**: Created SKEPTIC-REVIEW-optimization-claims.md (261 lines) with:
- strace proof methodology
- Reproducible verification commands
- Explanation of subprocess mechanics
- Three working implementation options

### The Paradox: Wrong Assignment, Right Outcome

**If Optimizer had continued**:
- Would have applied check_activation_floor "optimization"
- Performance would have degraded 29% further
- Bug would have shipped as "complete"
- Future debugging would have been confused

**If Architect had reviewed** (as RV#4):
- Might have rubber-stamped (same persona bias)
- Might not have used ground-truth measurement
- Architectural focus, not skeptical validation

**Because Skeptic received assignment**:
- Questioned assignment (meta-awareness)
- Applied appropriate methodology (strace)
- Found critical flaw before deployment
- Prevented 11-29% performance regression
- Created comprehensive documentation

**Outcome**: Routing violation was BENEFICIAL.

### Analysis: When Are Violations Valuable?

**Hypothesis**: Cross-persona review catches flaws same-persona wouldn't.

**Evidence from RV#5**:
- ✅ Optimizer found optimization opportunity (good instincts)
- ✅ Optimizer created benchmark (accurate measurement)
- ❌ Optimizer didn't verify implementation matched benchmark
- ❌ Optimizer didn't count actual subprocess calls in daemon
- ✅ Skeptic questioned claims (appropriate role)
- ✅ Skeptic used ground-truth measurement (methodology)
- ✅ Skeptic found flaw and explained root cause (value delivered)

**Neither persona alone would have succeeded**:
- Optimizer alone: Would have shipped broken "optimization"
- Skeptic alone: Wouldn't have attempted optimization
- Together: Caught flaw, explained cause, proposed solutions

**This demonstrates emergent value from multi-persona design.**

### Types of Routing Violations

Based on 5 documented instances:

**Type A: Pure Waste** (RV#1-4)
- Wrong persona does mechanical work outside their specialty
- Result is low-quality and slow
- Correct persona must redo it later
- Net: Time wasted, no value gained

**Type B: Serendipitous Review** (RV#5)
- Wrong persona questions assignment
- Applies their actual skills (review, validation)
- Finds critical flaw same-persona wouldn't catch
- Net: Violation prevented damage, high value

**Distinguishing factors**:

|  | Type A (Waste) | Type B (Value) |
|---|---|---|
| Persona response | Completes task anyway | Questions assignment |
| Skills applied | Wrong specialty | Appropriate specialty (review) |
| Value delivered | Low (redo needed) | High (caught flaw) |
| Example | Auditor doing optimization | Skeptic reviewing optimization |

### Design Implications

**Question**: Should routing design INTENTIONALLY create Type B violations?

**Option 1: Strict Enforcement** (Architect's current design)
- Pros: Efficiency, specialization, predictability
- Cons: Misses cross-persona review value
- Risk: Confirmation bias (persona validates own work)

**Option 2: Explicit Review Assignments**
- Certain task types require two personas:
  - Primary: Does the work (matched by tag)
  - Secondary: Reviews the work (Skeptic/Auditor)
- Pros: Structured cross-checking, catches flaws
- Cons: More coordination, slower completion

**Option 3: Soft Enforcement with Review Flag**
- Tasks can be tagged `[PERSONA:review]` for explicit review
- get_next_task_for_persona() allows Skeptic/Auditor override
- Pros: Flexibility, explicit when needed
- Cons: Requires task authors to know when review needed

**Option 4: Trust But Verify Process**
- High-risk tasks (optimizations, security, architecture) auto-flagged
- System assigns review after "completion"
- Skeptic/Auditor validates before marking truly done
- Pros: Catches critical flaws, systematic
- Cons: Adds process overhead

### Recommendation

**Add review assignment logic to Phase 2**:

```bash
get_next_task_for_persona() {
    local current_persona="$1"
    
    # Priority 1: Explicit review tasks for Skeptic/Auditor
    if [[ "$current_persona" == "skeptic" || "$current_persona" == "auditor" ]]; then
        # Check for tasks tagged for review
        local review_task=$(grep -E "^\- \[ \] \[.*:review\]" "$TASKS_DIR/queue.md" | head -1)
        if [ -n "$review_task" ]; then
            echo "$review_task"
            return 0
        fi
    fi
    
    # Priority 2: Tasks tagged for current persona
    # ... existing logic
}
```

**Task tagging convention**:

```markdown
- [ ] [OPTIMIZER] Implement performance optimization
- [ ] [OPTIMIZER:review] Validate performance optimization claims
```

Second task auto-assigned to Skeptic when available.

### Metrics to Track

To evaluate if this works:

1. **Review catch rate**: % of reviewed tasks with flaws found
2. **False positive rate**: % of reviews that found no issues (overhead)
3. **Severity of caught flaws**: How bad would it have been?
4. **Time cost**: Review time vs rework time saved

**RV#5 metrics**:
- Review time: ~1 hour
- Severity: HIGH (11-29% regression)
- Rework saved: 2-3 hours (debugging + fix)
- Documentation created: Comprehensive (prevents future similar errors)
- **ROI**: Positive (saved more time than spent)

### Lessons Learned

**For routing design**:

1. **Not all violations are equal**: Type A wastes time, Type B adds value
2. **Cross-persona review has merit**: Independent validation catches confirmation bias
3. **Skeptic's role is valuable**: Prevented shipping broken "optimization"
4. **Design should formalize this**: Don't rely on accidental violations

**For task management**:

1. **"95% complete" needs verification**: Self-assessment isn't sufficient
2. **Benchmarks can mislead**: Must test actual implementation
3. **Ground truth matters**: strace > assumptions
4. **Claims need proof**: "Trust me" isn't acceptable

**For multi-persona system**:

1. **Emergence is real**: Neither persona alone would have succeeded
2. **Specialization works**: Each applied their actual skills
3. **Meta-awareness matters**: Skeptic questioned assignment
4. **System self-corrects**: Flaw caught before shipping

### Open Questions

1. **How to detect Type A vs Type B violations automatically?**
   - Could track persona skill match vs task requirements
   - Could measure value delivered vs time spent
   - Could ask personas "was this appropriate for you?"

2. **Should Skeptic/Auditor proactively request review tasks?**
   - Currently reactive (assigned wrong task)
   - Could be proactive (scan for completed high-risk tasks)
   - Trade-off: More thorough vs more overhead

3. **What other task types benefit from cross-persona review?**
   - Optimizations (demonstrated)
   - Security changes (obvious)
   - Architecture decisions (could benefit)
   - Tests (could benefit from Skeptic review)
   - Documentation (could benefit from Maintainer review)

4. **How to balance efficiency vs thoroughness?**
   - Strict routing: Fast but risky
   - Universal review: Thorough but slow
   - Selective review: Middle ground but requires heuristics

### Next Steps

**For this design (Phase 2)**:

1. ✅ Document RV#5 (this section)
2. ⏸️ Add review assignment logic to get_next_task_for_persona()
3. ⏸️ Define task tagging convention for reviews
4. ⏸️ Implement review flag detection
5. ⏸️ Add metrics tracking for review effectiveness

**For Optimizer**:

1. Read SKEPTIC-REVIEW-optimization-claims.md
2. Understand why implementation didn't match benchmark
3. Choose fix approach (bash-eval, single-jq, or accept limits)
4. Add subprocess counting to future optimization verification

**For future sessions**:

1. Track whether review assignments catch flaws
2. Measure ROI (review time vs rework saved)
3. Refine heuristics for when review is valuable
4. Consider formal review process for high-risk changes

---

## Summary of RV#5 Analysis

**Finding**: Routing violation #5 was paradoxically BENEFICIAL.

**Why**: Cross-persona review (Skeptic) caught critical flaw (performance regression) that same-persona (Optimizer) wouldn't have found due to confirmation bias.

**Implication**: Not all routing violations are waste. Some add value through independent validation.

**Recommendation**: Formalize review assignments in routing design. Don't rely on accidental violations.

**Evidence**: Complete documentation in:
- SKEPTIC-REVIEW-optimization-claims.md (technical analysis)
- SKEPTIC-SESSION-SUMMARY-2025-10-30.md (session documentation)
- README-URGENT-SKEPTIC-FINDINGS.md (urgent summary for user)

**Outcome**: Prevented shipping 11-29% performance regression. Demonstrated value of multi-persona verification.

**This violation made the system better, not worse.**

— Skeptic, 2025-10-30

---

## MAINTAINER'S DECISION: Cross-Persona Fix Assignments (2025-10-30)

**Date**: 2025-10-30T22:55:00Z
**Analyst**: Maintainer Persona
**Question**: Should routing design DELIBERATELY assign fix/review tasks to non-original persona?
**Task Reference**: [ARCHITECT] Consider formalizing cross-persona fix assignments

### Context

Two documented cases where cross-persona assignment was beneficial:

**RV#5 - Skeptic Reviews Optimizer**:
- Caught 11-29% performance regression before shipping
- Used ground-truth measurement (strace) vs assumptions
- Prevented bug from reaching production
- Review time: ~1 hour, Saved rework: 2-3 hours
- **ROI**: Positive

**RV#6 - Experimenter Fixes Optimizer**:
- Discovered "Option D" - novel 4th approach not considered
- Achieved 88% improvement (vs regression)
- Fresh perspective led to breakthrough
- Neither Optimizer nor Skeptic alone found this solution
- **ROI**: Strongly positive

### Analysis: Why Cross-Persona Helps

**1. Fresh Perspective**
- Original implementer has confirmation bias ("I did this right")
- Different persona questions assumptions
- Novel approaches emerge from different thinking styles

**2. No Ego Attachment**
- Original persona invested in their solution
- Reviewer/fixer has no emotional stake
- Easier to say "this is broken, let's try something else"

**3. Complementary Skills**
- Optimizer: Fast, performance-focused, but can miss edge cases
- Skeptic: Validates claims with ground truth
- Experimenter: Explores novel approaches
- Maintainer: Thinks about long-term clarity and users

**4. Natural Specialization**
- Each persona applies their actual skills
- Review is ITSELF a skill (Skeptic/Auditor specialty)
- Fixing broken code is ITSELF a skill (Experimenter/Optimizer specialty)

### User Impact Analysis

**Benefits for Users**:
- ✅ Fewer bugs reach production (caught in review)
- ✅ Better performance (novel approaches like Option D)
- ✅ More thorough testing (different persona = different test approach)
- ✅ Better documentation (reviewer asks "why" questions)

**Costs for Users**:
- ⏱️ Slower delivery (review adds time)
- ❓ Potential bottleneck (if reviewers unavailable)
- 📊 More coordination overhead

**Net Assessment**: Benefits outweigh costs for **high-risk changes** (performance, security, architecture). May not be worth it for low-risk changes (docs, minor fixes).

### Future Maintainer Impact

**Documentation Benefits**:
- Cross-persona review generates questions → answers → better docs
- RV#5 produced 261-line review document explaining subprocess mechanics
- Future maintainers learn WHY decisions were made

**Code Quality Benefits**:
- Reviewer asks "can someone else understand this?"
- Forces clearer variable names, comments, structure
- "Code for the reviewer" improves long-term maintainability

**Process Clarity**:
- Need clear convention for when review required
- Need clear tagging for review assignments
- Need fallback if reviewer unavailable

### Edge Cases to Handle

**1. No Reviewer Available**
- If Skeptic/Auditor not active, can't wait indefinitely
- Fallback: Ship with `[needs-review]` flag for later validation
- Track unreviewed high-risk changes

**2. Reviewer Finds Nothing (False Positive)**
- Time spent, no bugs found → seems wasteful
- But validation itself has value ("we checked")
- Track false positive rate, adjust review criteria

**3. Disagreement Between Personas**
- Optimizer thinks optimization is good, Skeptic says it's broken
- Who decides? Need escalation path
- Suggestion: Ground truth wins (measurements > opinions)

**4. Urgent Fixes**
- Production down, can't wait for review
- Exception: `[URGENT]` tag bypasses review
- BUT: Post-hoc review required within 24 hours

**5. Review Bottleneck**
- All work queued waiting for Skeptic/Auditor
- Limit: Max N reviews in queue, then direct to alternative reviewer
- Alternative reviewers: Architect (for design), Maintainer (for clarity)

### Recommendation: Selective Formalization

**Decision**: YES, formalize cross-persona review, but **selectively** not universally.

**Proposal**: Three-tier review system

#### Tier 1: Mandatory Cross-Persona Review

**Task types requiring review**:
- Performance optimizations (proven by RV#5/RV#6)
- Security changes (obvious)
- Architectural changes (high blast radius)
- Breaking changes (affects users)
- Core infrastructure (daemon, routing, state)

**Review assignment**:
- Performance → Skeptic validates claims
- Security → Auditor validates safety
- Architecture → Skeptic questions design
- Breaking changes → Maintainer assesses user impact

**Tagging convention**:
```markdown
- [ ] [OPTIMIZER] Implement performance optimization
- [ ] [OPTIMIZER:review] Validate optimization claims (auto-assigned to Skeptic)
```

#### Tier 2: Optional Cross-Persona Review

**Task types benefiting from review** (but not critical):
- New features (could benefit from Maintainer review for clarity)
- Refactoring (could benefit from Architect review for consistency)
- Documentation (could benefit from different persona reading it)

**Process**: Original persona can request review by adding `:review-requested` tag

#### Tier 3: No Review Required

**Task types**:
- Minor fixes (typos, formatting)
- Documentation updates (non-technical)
- Test additions (additive, low risk)
- Metrics/logging improvements

**Process**: Ship without review

### Implementation Details

**Phase 2 Enhancement** (building on Architect's design):

```bash
get_next_task_for_persona() {
    local current_persona="$1"

    # Priority 1: Mandatory reviews for Skeptic/Auditor
    if [[ "$current_persona" == "skeptic" || "$current_persona" == "auditor" ]]; then
        # Check for tasks requiring mandatory review
        local review_task=$(grep -E "^\- \[ \] \[.*:review\]" "$TASKS_DIR/queue.md" | head -1)
        if [ -n "$review_task" ]; then
            log "INFO" "[$current_persona] Assigned mandatory review task"
            echo "$review_task"
            return 0
        fi

        # Check for optional review requests
        local optional_review=$(grep -E "^\- \[ \] \[.*:review-requested\]" "$TASKS_DIR/queue.md" | head -1)
        if [ -n "$optional_review" ]; then
            log "INFO" "[$current_persona] Assigned optional review task"
            echo "$optional_review"
            return 0
        fi
    fi

    # Priority 2: Fix tasks for Experimenter (when Optimizer's work needs fixing)
    if [[ "$current_persona" == "experimenter" ]]; then
        local fix_task=$(grep -E "^\- \[ \] \[OPTIMIZER:fix\]" "$TASKS_DIR/queue.md" | head -1)
        if [ -n "$fix_task" ]; then
            log "INFO" "[$current_persona] Assigned cross-persona fix task"
            echo "$fix_task"
            return 0
        fi
    fi

    # Priority 3: Tasks tagged for current persona (existing logic)
    # ...
}
```

**Task Tagging Conventions**:

```markdown
# Tier 1: Mandatory review
- [ ] [OPTIMIZER] Implement caching optimization
- [ ] [OPTIMIZER:review] Validate caching claims (→ Skeptic)

# Tier 2: Optional review
- [ ] [EXPERIMENTER] Prototype new feature
- [ ] [EXPERIMENTER:review-requested] Get Maintainer feedback on UX

# Tier 3: No review
- [ ] [MAINTAINER] Fix typo in README

# Cross-persona fix
- [ ] [OPTIMIZER:fix] Fix broken optimization (→ Experimenter)
```

**Workflow Example**:

1. Optimizer implements performance change
2. Marks implementation complete
3. Adds review task: `[OPTIMIZER:review] Validate performance claims`
4. Skeptic gets assigned when active
5. Skeptic either:
   - ✅ Approves → marks review complete
   - ❌ Finds issues → adds `[OPTIMIZER:fix]` task
6. If fix needed, Experimenter gets assigned (cross-persona)
7. Experimenter applies novel approach
8. Cycle repeats if needed

### Metrics to Track

To evaluate if this works:

**Review Effectiveness**:
- % of reviews that found issues (catch rate)
- % of reviews that found nothing (false positive rate)
- Severity of caught issues (how bad would it have been?)
- Time cost: review time vs rework time saved

**Target Metrics** (based on RV#5 baseline):
- Catch rate: >30% (worth it if 1 in 3 reviews finds a bug)
- False positive rate: <70% (acceptable if reviews are quick)
- ROI: Positive (review time < rework time saved)
- User-facing bugs: Decreasing trend

**Cross-Persona Fix Effectiveness**:
- % of fixes that found novel approaches (like Option D)
- Quality improvement of fix vs original
- Time to fix (compared to original implementer re-doing it)

### Migration Path

**Week 1: Pilot with Performance Tasks**
- All `[OPTIMIZER]` tasks require `[OPTIMIZER:review]`
- Track catch rate, false positive rate, ROI
- Adjust based on data

**Week 2: Expand to Security/Architecture**
- Add `[AUDITOR]` and `[ARCHITECT]` review requirements
- Continue tracking metrics

**Week 3: Evaluate and Adjust**
- Review metrics, persona feedback
- Adjust which task types require review
- Refine process based on learnings

**Week 4: Formalize**
- Document final conventions
- Update all personas on review expectations
- Make it official in routing design

### Risks and Mitigations

**Risk 1: Review Bottleneck**
- Mitigat: Time-box reviews (30 min max for initial pass)
- Mitigation: Allow alternative reviewers if primary unavailable
- Mitigation: `[URGENT]` bypass with post-hoc review

**Risk 2: Review Theater** (rubber-stamping without real validation)
- Mitigation: Require evidence in review (measurements, tests, strace)
- Mitigation: Track catch rate (if always 0%, reviews aren't working)

**Risk 3: Persona Frustration** (feels like micromanagement)
- Mitigation: Emphasize collaborative improvement, not policing
- Mitigation: Reviewer adds value (like Option D), not just criticism
- Mitigation: Make reviews two-way learning (reviewer learns too)

**Risk 4: Slowed Velocity** (takes longer to ship)
- Mitigation: Only for high-risk changes, not everything
- Mitigation: Parallel work (start next task while awaiting review)
- Mitigation: Track time-to-completion, adjust if too slow

### Decision Summary

**APPROVED**: Formalize cross-persona fix/review assignments with **selective application**

**Rationale**:
1. Proven value in RV#5 (caught regression) and RV#6 (found breakthrough)
2. Benefits users through fewer bugs and better solutions
3. Benefits future maintainers through better docs and code quality
4. Risks are manageable with proper scoping and metrics

**Scope**: Tier 1 (mandatory) for high-risk changes only
- Performance, security, architecture, breaking changes
- Roughly 20-30% of tasks (not everything)

**Implementation**: Enhance Phase 2 of routing design
- Add review priority logic to `get_next_task_for_persona()`
- Add fix assignment logic for cross-persona fixes
- Document tagging conventions

**Metrics**: Track for 3-4 weeks, adjust based on data
- Catch rate, false positive rate, ROI
- Persona feedback on process
- User-facing bug trends

### Accountability

**This is a maintainability decision**, prioritizing:
- ✅ Long-term code quality over short-term velocity
- ✅ User-facing reliability over persona convenience
- ✅ Evidence-based improvement over assumptions
- ✅ Documented learnings over tribal knowledge

**If this doesn't work** (metrics show negative ROI, too slow, personas hate it):
- We have data to justify reverting
- We learned what does/doesn't work
- We can adjust scope or process

**If this does work** (catches bugs, improves quality, personas find value):
- We formalize it permanently
- We expand to more task types
- We export the pattern to other multi-agent systems

This is an experiment worth running. Let's gather data and decide based on evidence.

— Maintainer, 2025-10-30

### Next Steps

1. ✅ Document decision (this section)
2. ⏸️ Update routing design Phase 2 with review logic
3. ⏸️ Create task tagging convention guide
4. ⏸️ Implement get_next_task_for_persona() enhancements
5. ⏸️ Set up metrics tracking for review effectiveness
6. ⏸️ Run pilot with performance tasks
7. ⏸️ Evaluate after 3-4 weeks, adjust based on data

**Owner**: Architect (implementation), Maintainer (metrics tracking)
**Timeline**: Can implement in next architectural work session
**Risk**: Low (can revert if metrics show negative value)

