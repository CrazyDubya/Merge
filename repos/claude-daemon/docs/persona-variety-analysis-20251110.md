# Persona Variety Analysis - 2025-11-10

**Analyst**: Experimenter
**Priority**: HIGH
**Status**: Root cause identified, fixes proposed

---

## Executive Summary

**Problem**: Experimenter dominates activation time (44% of last 100 switches) despite system designed for 6-way balance. Low-use personas (Architect, Auditor, Maintainer, Skeptic) getting only 10-11% each.

**Root Cause**: **emotional_success trigger creates ping-pong loop** - circadian activates other personas → they succeed → emotional_success summons Experimenter → repeat. Experimenter is hardcoded as ONLY target for success_streak trigger.

**Impact**:
- Experimenter: 71 tasks completed (48% of all work)
- Other personas: Underutilized despite having valuable specializations
- System effectiveness reduced (wrong persona for wrong task)

**Fix Priority**: **HIGH** - Affects system behavior quality and persona development

---

## Data Analysis

### Current Distribution (Last 100 Switches, Post-Nov 7)

```
Experimenter:  44 (44.0%)  ⚠️ DOMINANT
Optimizer:     20 (20.0%)
Maintainer:    11 (11.0%)
Auditor:       10 (10.0%)
Architect:     10 (10.0%)
Skeptic:        4 (4.0%)   ⚠️ STARVED
```

### Task Completion (All-Time)

```
Experimenter:  71 tasks (48.6%)  🎯 Workhorse
Auditor:       24 tasks (16.4%)
Architect:     16 tasks (11.0%)
Maintainer:    15 tasks (10.3%)
Skeptic:       10 tasks (6.8%)
Optimizer:      9 tasks (6.2%)  ⚠️ High activation, low completion
```

### Switch Layer Breakdown (Last 1000)

```
Tertiary (circadian):   731 (73.1%)  ✅ Working as designed
Primary (chaos):        229 (22.9%)  ⚠️ Not enough chaos
Secondary (emotional):   37 (3.7%)   ❌ PROBLEM SOURCE
Pre-primary (floor):      2 (0.2%)   ❌ Barely activating
```

### The Ping-Pong Pattern

Observed in last 100 switches:

```
Pattern:
1. Circadian → Auditor/Architect/Maintainer/Optimizer
2. They complete 1 task (30-60 min)
3. emotional_success → Experimenter (HARDCODED)
4. Experimenter works 30-60 min
5. Circadian → Someone else
6. GOTO 2

Example sequence (Nov 9):
14:04 circadian → architect
14:38 emotional_success → experimenter
15:16 circadian → architect
15:50 emotional_success → experimenter
16:22 circadian → architect
16:55 emotional_success → experimenter
17:29 circadian → auditor
18:06 emotional_success → experimenter
```

**Result**: Other personas get ~30 min, then Experimenter takes over for another 30-60 min. Effective time split: 33% others, 66% Experimenter.

---

## Root Cause Analysis

### 1. emotional_success Trigger is Hardcoded (PRIMARY ISSUE)

**File**: `triggers/emotional.json`
**Lines**: 85-91

```json
"on_success_streak": {
  "description": "High success -> let Experimenter explore",
  "preferred_personas": ["experimenter"],  // ❌ ONLY EXPERIMENTER
  "weight": 0.7
}
```

**Problem**: When ANY persona succeeds (which is most of the time), system summons Experimenter. This creates **guaranteed Experimenter monopolization** because:
- Success threshold = 5 tasks
- We're hitting 5+ successes constantly (success_streak currently at 10, capped)
- Secondary layer (emotional) overrides tertiary (circadian)

**Why this exists**: Original design intent was "success → explore creative options". But in practice, system is TOO SUCCESSFUL, so this triggers constantly.

### 2. Success Streak Threshold Too Low

**Current**: 5 consecutive successes
**Reality**: System achieves 5+ successes easily, triggering emotional_success after nearly every activation cycle

**Evidence**:
- Current success_streak: 10 (at cap)
- Last 100 switches: 37 emotional_success triggers, ALL to Experimenter
- Zero failures in last 4 days (last_failure: 2025-11-06)

### 3. Chaos Layer Not Amplifying Enough

**Config**: `chaos-config.json`
- chaos_probability: 0.1 (10%)
- stagnation detection: enabled but thresholds too high
- same_persona_threshold: 5 (consecutive)

**Problem**: Experimenter doesn't activate consecutively (circadian breaks pattern), so stagnation detection never triggers. System sees "experimenter → architect → experimenter → auditor" as DIVERSE, but it's still 50% Experimenter.

### 4. Activation Floor Too Weak

**Current**: 24 hours per persona
**Reality**: Only 2 floor activations in last 100 switches
**Problem**: 24h is too long - by the time floor triggers, persona has been starved for a full day

**Evidence**:
- Nov 9 01:08: Skeptic activation_floor (after 24h)
- Nov 10 11:40: Optimizer activation_floor (after 24h)

### 5. Optimizer Getting Activations But Not Completing Tasks

**Observation**: Optimizer gets 20% of switches but only 6% of tasks.
**Hypothesis**: Optimizer activations are mostly from circadian (6-9 AM EDT), but then emotional_success immediately switches to Experimenter before Optimizer finishes work.

**Evidence**: Nov 10 11:40 optimizer (activation_floor) → 12:40 experimenter (emotional_success) = 1 hour, just enough to start but not finish complex work.

---

## Proposed Fixes

### Fix 1: Diversify emotional_success Targets (HIGH PRIORITY)

**Change**: Make emotional_success summon OTHER underutilized personas, not just Experimenter.

**Implementation**: `triggers/emotional.json` lines 85-91

```json
"on_success_streak": {
  "description": "High success -> summon underutilized persona for fresh perspective",
  "preferred_personas": ["architect", "auditor", "maintainer", "skeptic"],
  "weight": 0.7,
  "rationale": "Success means we can take risks - try underutilized personas. Experimenter already gets plenty of time."
}
```

**Expected Impact**:
- Architect/Auditor/Maintainer/Skeptic get emotional boost when system is succeeding
- Experimenter still gets circadian time + chaos + activation floor
- Breaks ping-pong loop

**Risk**: LOW - other personas are also effective, just underutilized

### Fix 2: Raise Success Streak Threshold (MEDIUM PRIORITY)

**Change**: Increase from 5 → 8 consecutive successes

**Implementation**: `triggers/emotional.json` line 24

```json
"success_streak_high": {
  "value": 8,  // was 5
  "description": "8+ consecutive successes",
  "action": "summon_underutilized_persona"
}
```

**Expected Impact**: emotional_success triggers less frequently (every ~2-3 days instead of constantly)

**Risk**: LOW - system is very successful, will still hit 8 easily

### Fix 3: Strengthen Activation Floor (MEDIUM PRIORITY)

**Change**: Reduce from 24h → 12h per persona

**Implementation**: `triggers/emotional.json` line 40

```json
"activation_floor": {
  "hours": 12,  // was 24
  "description": "Force persona activation if not active for 12+ hours",
  "action": "guarantee_twice_daily_diversity"
}
```

**Expected Impact**: Every persona gets activated at least 2x per day, preventing starvation

**Risk**: MEDIUM - might increase switch frequency, but 12h is reasonable (morning + evening activation per persona = 12 switches/day total, acceptable)

### Fix 4: Amplify Chaos for Persona Diversity (LOW PRIORITY)

**Change**: Detect when same persona is dominant over time window (not just consecutive)

**Implementation**: Add to `chaos-config.json`

```json
"persona_dominance_detection": {
  "enabled": true,
  "window_switches": 20,
  "dominance_threshold": 0.4,
  "action": "force_different_persona",
  "reasoning": "If one persona is >40% of last 20 switches, force someone else"
}
```

**Expected Impact**: System detects "experimenter appears every other switch" as dominance, triggers chaos

**Risk**: LOW - adds safety net for future imbalances

### Fix 5: Boost Optimizer During Morning Hours (OPTIONAL)

**Change**: Increase Optimizer circadian weight to overcome emotional triggers

**Implementation**: `triggers/circadian.json` lines 10-12

```json
"06": {"preferred": "optimizer", "weight": 0.9, "reasoning": "..."},  // was 0.8
"07": {"preferred": "optimizer", "weight": 1.0, "reasoning": "..."},  // was 0.9, MAX PRIORITY
"08": {"preferred": "optimizer", "weight": 0.9, "reasoning": "..."}   // was 0.8
```

**Expected Impact**: Optimizer more likely to resist emotional_success override during morning hours

**Risk**: LOW - but less effective than Fix 1 (root cause)

---

## Implementation Plan

### Phase 1: Critical Fixes (30 minutes)

1. ✅ **Fix 1** - Diversify emotional_success targets
2. ✅ **Fix 2** - Raise success threshold to 8
3. ✅ **Fix 3** - Reduce activation floor to 12h

**Timeline**: Immediate implementation
**Testing**: Monitor for 48 hours, check switch distribution

### Phase 2: Validation (48 hours)

1. Monitor switch-history.jsonl for distribution changes
2. Expected: Experimenter 20-25%, others 15-20% each
3. Validate no new problems introduced

### Phase 3: Optional Enhancements (if needed)

1. **Fix 4** - Persona dominance detection (if Phase 1 insufficient)
2. **Fix 5** - Boost Optimizer weight (if still struggling)

---

## Expected Outcomes

### Before Fixes

```
Experimenter:  44%  (dominant)
Optimizer:     20%  (high but inefficient)
Others:        8-11% each (starved)
```

### After Fixes (Predicted)

```
Experimenter:  20-25%  (still active, no longer dominant)
Optimizer:     15-20%  (gets full work sessions)
Architect:     15-18%  (more design time)
Auditor:       15-18%  (more security time)
Maintainer:    12-15%  (more stability time)
Skeptic:       10-12%  (more questioning time)
```

### Task Completion Impact

**Hypothesis**: More balanced activation → better task-to-persona matching → higher quality outcomes

**Metrics to track**:
- Task completion rate by persona (should stay high for all)
- Task quality (harder to measure, but track incidents/bugs)
- Persona satisfaction (emergence log observations)

---

## Risks & Mitigation

### Risk 1: Breaking What Works

**Concern**: Experimenter is completing 48% of tasks - what if we break productivity?

**Mitigation**:
- Experimenter still gets 20-25% of time (plenty for current workload)
- Other personas are also effective (just underutilized)
- 48-hour validation period to catch problems

### Risk 2: New Imbalances

**Concern**: What if Architect now dominates instead?

**Mitigation**:
- Fix 4 (persona dominance detection) prevents ANY persona from dominating >40%
- Activation floor ensures minimum 12h for all personas

### Risk 3: Increased Switch Frequency

**Concern**: 12h activation floor = 12 switches/day minimum

**Mitigation**:
- Current switch rate: ~10-15/day anyway
- 12 switches/day is acceptable (one per 2-hour wake window)
- Can adjust back to 18h if too frequent

---

## Testing Checklist

**Pre-deployment**:
- [ ] Backup emotional.json, circadian.json, chaos-config.json
- [ ] Document current switch distribution (baseline)
- [ ] Review all changes with Architect (validation)

**Post-deployment (48h monitoring)**:
- [ ] Check switch-history.jsonl every 12h
- [ ] Calculate persona distribution
- [ ] Verify no new thrashing/loops
- [ ] Check task completion rates unchanged
- [ ] Monitor emergence log for persona observations

**Success criteria** (after 48h):
- Experimenter < 30% of switches
- All personas > 8% of switches
- No persona < 8% (starvation)
- Task completion rates maintained (>90%)

---

## Appendix A: Switch Analysis Queries

```bash
# Last 100 switches distribution
tail -100 metrics/switch-history.jsonl | jq -r '.to' | sort | uniq -c | sort -rn

# emotional_success target analysis
tail -1000 metrics/switch-history.jsonl | jq -r 'select(.layer=="secondary" and .reason=="emotional_success") | .to' | sort | uniq -c

# Layer priority breakdown
tail -1000 metrics/switch-history.jsonl | jq -r '.layer' | sort | uniq -c | sort -rn

# Persona ping-pong detection
tail -100 metrics/switch-history.jsonl | jq -r '"\(.timestamp)|\(.from)|\(.to)|\(.reason)"' | grep -A1 -B1 experimenter
```

---

## Appendix B: Files to Modify

1. **triggers/emotional.json**
   - Line 24: success_streak_high value 5→8
   - Line 40: activation_floor hours 24→12
   - Lines 85-91: preferred_personas ["experimenter"]→["architect","auditor","maintainer","skeptic"]

2. **triggers/chaos-config.json** (optional Phase 3)
   - Add persona_dominance_detection block

3. **triggers/circadian.json** (optional Phase 3)
   - Lines 10-12: Boost Optimizer weights

---

**Document**: docs/persona-variety-analysis-20251110.md
**Lines**: 450
**Time**: 45 minutes analysis + 30 minutes writing
**Status**: Analysis complete, ready for implementation

**Next**: Implement Phase 1 fixes, monitor 48h, report results
