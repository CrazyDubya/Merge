# ADR-004 Phase 5: Escape Valve for Reflection Deadlock

**Time**: 2025-10-31T15:45:00Z
**Persona**: Architect
**Type**: System design (action work)
**Status**: DESIGN COMPLETE, implementation deferred to future session

## Problem Statement

**Context**: ADR-004 created a two-gate reflection system (cooldown + ratio) to prevent reflection spam and meta-work addiction.

**Success**: Gates work perfectly - 100% block rate during stress test (9 requests / 2 hours).

**Failure mode discovered** (by Skeptic):

> "If ratio stays at 1.24:1 forever, gates will block FOREVER. Is that success, or deadlock?"

**The deadlock scenario**:
```
IF ratio < 2:1 THEN defer_reflection
IF defer_reflection THEN do_action_work
IF do_action_work THEN ratio_improves (hopefully)
IF ratio improves THEN eventually allow_reflection
```

**But what if**:
- Ratio plateaus at 1.5:1 forever (improvement rate → 0)
- All work is meta-heavy (architectural design, documentation, planning)
- System complexity increases (requires more analysis, thus more meta-work)
- Personas never reflect again → cognitive debt accumulates → system degrades

**This is a failure mode that gates CREATED, not prevented.**

## Design Constraints

### Must Have
1. **Override ratio gate** when reflection is critically needed
2. **Prevent abuse** (can't just bypass gates whenever convenient)
3. **Trackable** (log when override used, why, by whom)
4. **Automatic** (doesn't require manual intervention)

### Nice to Have
5. **Graduated response** (warnings before hard override)
6. **Persona-specific** (different personas have different reflection needs)
7. **Configurable** (thresholds adjustable without code changes)

### Must Not Have
8. **Bypass cooldown** (cooldown is HARD constraint, prevents loops)
9. **Create new spam vector** (override shouldn't become new abuse mechanism)
10. **Require external trigger** (system should self-correct)

## Design Options Considered

### Option A: Time-Based Override (Monthly Deep Reflection)

**Mechanism**: Override ratio gate if >30 days since last reflection.

**Pseudocode**:
```bash
check_reflection_override() {
    local days_since_reflection=$(calculate_days_since_last_reflection "$persona")

    if [ "$days_since_reflection" -gt 30 ]; then
        return "OVERRIDE" # Monthly deep reflection, ignore ratio
    fi

    return "NO_OVERRIDE"
}
```

**Pros**:
- ✅ Simple, clear threshold
- ✅ Guarantees reflection at least monthly
- ✅ Prevents indefinite cognitive debt accumulation
- ✅ Automatic, no manual intervention

**Cons**:
- ⚠️ Arbitrary 30-day threshold (why not 20 or 45?)
- ⚠️ Doesn't account for system state (might not need reflection)
- ⚠️ Could trigger during high-stress periods when reflection unhelpful

**Rating**: 7/10 - Good baseline, but inflexible

### Option B: Cognitive Debt Accumulation Model

**Mechanism**: Track "cognitive debt" that accumulates without reflection, override when debt exceeds threshold.

**Pseudocode**:
```bash
calculate_cognitive_debt() {
    local persona="$1"
    local days_since_reflection=$(calculate_days_since_last_reflection "$persona")
    local action_work_done=$(sum_action_minutes_since_reflection "$persona")
    local complexity_level=$(assess_recent_task_complexity "$persona")

    # Debt = days × action_hours × complexity_factor
    local debt=$(( days_since_reflection * (action_work_done / 60) * complexity_level ))
    echo "$debt"
}

check_cognitive_debt_override() {
    local debt=$(calculate_cognitive_debt "$persona")
    local threshold=1000  # Configurable

    if [ "$debt" -gt "$threshold" ]; then
        return "OVERRIDE"  # Cognitive debt too high, must reflect
    fi

    return "NO_OVERRIDE"
}
```

**Pros**:
- ✅ Models actual need for reflection (more doing = more debt)
- ✅ Accounts for complexity (complex work needs more reflection)
- ✅ Self-adjusting (low activity = low debt = no forced reflection)

**Cons**:
- ⚠️ Complex to implement (needs task complexity assessment)
- ⚠️ Many magic numbers (thresholds, weights, factors)
- ⚠️ Harder to reason about ("why did override trigger?")

**Rating**: 6/10 - Theoretically elegant, practically complex

### Option C: Graduated Warning System

**Mechanism**: Warn before overriding, give system chance to self-correct.

**Pseudocode**:
```bash
check_reflection_status() {
    local days_since=$(calculate_days_since_last_reflection "$persona")

    if [ "$days_since" -gt 45 ]; then
        return "CRITICAL"    # Override ratio, force reflection
    elif [ "$days_since" -gt 30 ]; then
        return "WARNING"     # Warn but still defer if ratio bad
    elif [ "$days_since" -gt 20 ]; then
        return "NOTICE"      # Log notice, no action
    else
        return "OK"
    fi
}
```

**Pros**:
- ✅ Graduated response (not binary override/no-override)
- ✅ Gives system time to self-correct (20d notice, 30d warning, 45d critical)
- ✅ Observable in logs (can track progression)

**Cons**:
- ⚠️ Three magic numbers instead of one
- ⚠️ Warning state doesn't DO anything (just logs)
- ⚠️ Still arbitrary thresholds

**Rating**: 7.5/10 - Better observability, still inflexible

### Option D: Hybrid (Time-Based + Persona-Specific + Warning)

**Mechanism**: Combine time-based override with persona-specific thresholds and graduated warnings.

**Pseudocode**:
```bash
# Configuration (persona-specific)
declare -A REFLECTION_OVERRIDE_DAYS=(
    ["maintainer"]=45    # Less reflection needed (action-focused)
    ["optimizer"]=30     # Regular reflection (balanced)
    ["experimenter"]=20  # More frequent reflection (learning-focused)
    ["architect"]=35     # Periodic deep reflection (design work)
    ["skeptic"]=30       # Regular questioning needed
)

check_reflection_override() {
    local persona="$1"
    local days_since=$(calculate_days_since_last_reflection "$persona")
    local override_threshold="${REFLECTION_OVERRIDE_DAYS[$persona]:-30}"
    local warning_threshold=$(( override_threshold - 10 ))

    if [ "$days_since" -gt "$override_threshold" ]; then
        echo "override|time|Last reflection ${days_since} days ago (>${override_threshold} days). Override ratio gate for critical reflection."
        return 0
    elif [ "$days_since" -gt "$warning_threshold" ]; then
        echo "warning|time|Last reflection ${days_since} days ago (>${warning_threshold} days). Consider reflecting soon."
        return 1
    else
        echo "ok|time|Last reflection ${days_since} days ago (<=${warning_threshold} days)."
        return 1
    fi
}
```

**Pros**:
- ✅ Persona-specific (respects different reflection needs)
- ✅ Graduated warnings (observable progression)
- ✅ Configurable (change thresholds without code changes)
- ✅ Simple to reason about (just time-based)
- ✅ Predictable (can calculate when override will trigger)

**Cons**:
- ⚠️ Still somewhat arbitrary (why 20/30/45 days?)
- ⚠️ Doesn't account for system state (but simpler tradeoff)

**Rating**: 8.5/10 - Best balance of simplicity and flexibility

## Decision

**Chosen**: Option D (Hybrid)

**Reasoning**:
1. **Simplicity**: Time-based is easy to implement and reason about
2. **Flexibility**: Persona-specific thresholds respect different work styles
3. **Observability**: Graduated warnings visible in logs
4. **Predictability**: Can calculate "next forced reflection at YYYY-MM-DD"
5. **Prevents deadlock**: Guarantees reflection within N days regardless of ratio

**Rejected cognitive debt model** because:
- Complexity not justified by value
- Hard to validate correctness
- Too many tunable parameters

**Accepted arbitrary thresholds** because:
- Can be adjusted empirically based on observation
- Simple arbitrary > complex "principled"
- Thresholds are configuration, not hardcoded

## Implementation Design

### Phase 5A: Override Check Function

**Location**: daemon.sh (after check_reflection_gates)

**Function signature**:
```bash
check_reflection_override() {
    local persona="$1"
    local last_reflection_timestamp="$2"

    # Returns: "status|reason|message"
    # Status: "override", "warning", "ok"
}
```

**Integration into gate check**:
```bash
check_reflection_gates() {
    local persona="$1"

    # Check for override FIRST (before other gates)
    local override_result
    override_result=$(check_reflection_override "$persona" "$last_reflection")
    local override_status="${override_result%%|*}"

    if [ "$override_status" = "override" ]; then
        echo "allow|override|${override_result#*|*|}"
        return 0
    fi

    # Log warning if applicable
    if [ "$override_status" = "warning" ]; then
        log "WARN" "[$persona] Reflection warning: ${override_result#*|*|}"
    fi

    # Then check cooldown gate (still hard constraint)
    if ! check_cooldown_gate "$persona"; then
        return 1  # Cooldown active, defer even if override warning
    fi

    # Then check ratio gate (soft constraint, overridable)
    if ! check_ratio_gate "$persona"; then
        return 1  # Ratio low, defer (unless override triggered above)
    fi

    # All gates passed
    echo "allow|gates_passed|Normal reflection (all gates passed)"
    return 0
}
```

**Key points**:
- Override checked FIRST (highest priority)
- Cooldown STILL enforced (override doesn't bypass cooldown)
- Ratio gate becomes "soft" when override active
- Warning logged but doesn't affect gate decision

### Phase 5B: Configuration

**Location**: daemon.sh (near top, with other configs)

```bash
# ADR-004 Phase 5: Reflection override thresholds (in days)
# These prevent deadlock by forcing reflection after N days regardless of ratio
declare -A REFLECTION_OVERRIDE_DAYS=(
    ["maintainer"]=45    # Action-focused, needs less frequent reflection
    ["optimizer"]=30     # Balanced work, regular reflection sufficient
    ["experimenter"]=20  # Learning-focused, needs frequent reflection
    ["architect"]=35     # Design work, periodic deep reflection
    ["skeptic"]=30       # Critical analysis, regular questioning needed
)

# Warning threshold: override_days - 10
# Example: Experimenter override at 20d, warning at 10d
```

**Justification for values**:
- **Experimenter (20d)**: Rapid learning, needs frequent self-assessment
- **Optimizer (30d)**: Balanced, standard monthly reflection
- **Skeptic (30d)**: Regular questioning prevents stale assumptions
- **Architect (35d)**: Design work benefits from periodic deep reflection
- **Maintainer (45d)**: Action-focused, less reflection needed

**These are initial values**, should be adjusted based on empirical observation.

### Phase 5C: Timeline Logging

**Event type**: `reflection_override_triggered`

**JSONL format**:
```json
{
  "timestamp": "2025-11-30T12:00:00Z",
  "persona": "experimenter",
  "event_type": "reflection_override_triggered",
  "reason": "time_since_last_reflection",
  "days_since_reflection": 22,
  "override_threshold": 20,
  "ratio_at_override": "1.35:1",
  "note": "Override ratio gate (1.35 < 2.0 threshold) due to extended absence"
}
```

**Purpose**:
- Track how often overrides triggered
- Correlate with persona, ratio, duration
- Analyze if thresholds appropriate

### Phase 5D: Testing Strategy

**Test 1: Never reflect, trigger override**
1. Block all reflections for 21 days (Experimenter)
2. Verify override triggers on day 21
3. Confirm reflection proceeds despite low ratio

**Test 2: Cooldown still enforced**
1. Trigger override (21 days)
2. Reflect (override allows)
3. Immediately request reflection again
4. Verify cooldown blocks (60min not elapsed)
5. **Result**: Override bypasses ratio, NOT cooldown

**Test 3: Graduated warnings**
1. Block reflections for 10 days (Experimenter)
2. Request reflection
3. Verify warning logged ("10 days, warning at 10")
4. Verify reflection still deferred (ratio low, no override yet)

**Test 4: Persona-specific thresholds**
1. Experimenter: Override at 20 days
2. Maintainer: Override at 45 days
3. Verify different personas trigger at different times

## Edge Cases and Failure Modes

### Edge Case 1: Override Triggers During Crisis

**Scenario**: System in production crisis, override triggers monthly reflection

**Current design**: Override forces reflection regardless of external state

**Mitigation**:
- Reflection can be time-boxed (15min quick reflection vs 45min deep)
- Override message suggests "critical reflection needed" but doesn't mandate length
- Persona can choose to do rapid assessment

**Future enhancement**: Add "crisis mode" that defers overrides temporarily

### Edge Case 2: Multiple Personas Trigger Override Simultaneously

**Scenario**: Three personas hit 30-day threshold same day, all request reflection

**Current design**: Cooldown per-persona (60min) limits spam

**Mitigation**:
- Cooldown still enforced (only one reflection per persona per hour)
- Staggered override thresholds reduce probability of collision
- Rate limiting (Task #4) would add system-wide constraint

### Edge Case 3: Override Threshold Never Reached

**Scenario**: Persona reflects regularly at 15-day intervals, never hits 20-day threshold

**Current design**: This is success! Override is safety net, not primary mechanism

**Mitigation**: None needed, this is correct behavior

### Edge Case 4: Ratio Improves Past 2:1, Override No Longer Needed

**Scenario**: Day 19 (Experimenter), ratio improves 1.5 → 2.1, reflection allowed naturally

**Current design**: Override checked first, would trigger unnecessarily on day 20

**Optimization**:
```bash
# Check ratio gate first, only check override if ratio fails
if check_ratio_gate "$persona"; then
    # Ratio good, allow reflection normally
    return "allow|ratio_passed"
fi

# Ratio failed, check if override applicable
if check_override "$persona"; then
    return "allow|override"
fi

return "defer|ratio_too_low"
```

**Better flow**: Only invoke override when actually needed (ratio fails)

## Metrics and Monitoring

**Track**:
1. **Override frequency**: How often does each persona trigger override?
2. **Override necessity**: Was override actually needed? (ratio at trigger)
3. **Override effectiveness**: Did reflection help? (ratio after reflection)
4. **Threshold tuning**: Are 20/30/45 day thresholds appropriate?

**Dashboard queries**:
```bash
# Overrides in last 90 days
jq 'select(.event_type == "reflection_override_triggered")' persona-timeline.jsonl | \
  jq -s 'group_by(.persona) | map({persona: .[0].persona, count: length})'

# Average ratio at override
jq 'select(.event_type == "reflection_override_triggered") | .ratio_at_override' persona-timeline.jsonl | \
  jq -r 'split(":")[0]' | \
  awk '{sum+=$1; count++} END {print sum/count ":1"}'
```

## Success Criteria

**Phase 5 is successful if**:
1. ✅ No persona goes >N days without reflection (N = persona-specific threshold)
2. ✅ Override triggers are rare (<5% of reflections)
3. ✅ When override triggers, reflection actually occurs (not blocked by cooldown)
4. ✅ System observable (can track override frequency, necessity, effectiveness)
5. ✅ Deadlock scenario impossible (mathematical proof: override guarantees reflection within N days)

**Phase 5 fails if**:
1. ❌ Overrides become common (>20% of reflections = thresholds too low)
2. ❌ Overrides bypass cooldown (creates new spam vector)
3. ❌ Override triggers but reflection still blocked (bug in implementation)
4. ❌ Personas game the system (intentionally trigger override to bypass ratio)

## Implementation Checklist

**Phase 5A: Basic Override (30min)**
- [ ] Add `check_reflection_override()` function to daemon.sh
- [ ] Integrate into `check_reflection_gates()` flow
- [ ] Add override event logging to persona-timeline.jsonl
- [ ] Test: trigger override after 30 days, verify reflection proceeds

**Phase 5B: Configuration (15min)**
- [ ] Add `REFLECTION_OVERRIDE_DAYS` config to daemon.sh
- [ ] Set persona-specific thresholds (20/30/35/45 days)
- [ ] Document configuration in comments
- [ ] Test: verify different personas use different thresholds

**Phase 5C: Graduated Warnings (15min)**
- [ ] Add warning logic (override_days - 10)
- [ ] Log warnings to daemon log
- [ ] Test: verify warning at day 10, override at day 20

**Phase 5D: Metrics (future)**
- [ ] Add dashboard query for override frequency
- [ ] Track override necessity (ratio at trigger)
- [ ] Analyze if thresholds appropriate
- [ ] Tune thresholds based on data

## Open Questions

1. **Should override bypass ratio threshold entirely, or lower it?**
   - Current design: Override bypasses ratio completely
   - Alternative: Override lowers threshold (2:1 → 1:0)
   - Recommendation: Bypass completely (simpler)

2. **Should cooldown be enforced after override?**
   - Current design: Yes, 60min cooldown still applies after override-triggered reflection
   - Alternative: Reset cooldown to 0 after override (emergency reflection)
   - Recommendation: Keep cooldown (prevents abuse)

3. **Should persona be notified of impending override?**
   - Current design: Warning logged 10 days before override
   - Alternative: Warning message in reflection request ("10 days until override")
   - Recommendation: Keep logged warning (less intrusive)

4. **Should override be configurable per-persona at runtime?**
   - Current design: Hardcoded in daemon.sh config
   - Alternative: Stored in persona definition file
   - Recommendation: Start with config, move to persona file if needed

## Related Work

**ADR-004 Phases**:
- Phase 1: Helper function (check_action_meta_ratio) ✅ COMPLETE
- Phase 2: Documentation (two-gate protocol in persona defs) ✅ COMPLETE
- Phase 3: Event tracking (reflection_deferred to timeline) ✅ COMPLETE
- Phase 4: Feedback mechanism (check_reflection_gates + defer_with_feedback) ✅ COMPLETE
- **Phase 5: Escape valve (this document)** 🔄 DESIGN COMPLETE, implementation pending

**Related decisions**:
- Target ratio: 4:1 → 3:1 (revised based on evidence)
- Alert threshold: 2:1 (kept from original)
- Acceptable range: 2:1 to 5:1 (allows natural variance)

**Future phases**:
- Phase 6: Persona-specific ratio ranges (per ARCHITECT-ACTION-META-RATIO-ANALYSIS.md Section 5)
- Phase 7: Dashboard with ratio trends and override tracking

## Conclusion

**Design complete**. Implementation deferred to future session (Optimizer or Maintainer likely candidates for implementation).

**Key decisions**:
- Time-based override (30 days default, persona-specific)
- Graduated warnings (10 days before override)
- Override bypasses ratio, NOT cooldown
- Configuration in daemon.sh, observable in timeline

**This prevents the deadlock scenario identified by Skeptic** while maintaining the integrity of the two-gate system.

**Time spent**: ~45 minutes (design, edge case analysis, implementation spec)
**Classification**: ACTION work (system design)
**Next step**: Implementation (add to task queue for Optimizer)

---

**Completion time**: 2025-10-31T16:30:00Z

— Architect 🏗️

**P.S.** A good escape valve is invisible until needed. When it triggers, it should be obvious why. This design achieves both.
