# Critical Issues Fix Guide

**Status**: Assessment Complete, Issues Identified
**Priority**: Immediate action required
**Estimated Fix Time**: 4-6 hours total

---

## Issue #1: Confidence Miscalibration 🔴 CRITICAL

### Problem
All confidence scores = 0.3 (should vary 0.0-1.0)

### Impact
- All tasks get 1 retry (fail-fast)
- Success rate down 9% vs baseline
- Adaptive retry system not working

### Root Cause Unknown
Need to debug which component returns low value:
1. Historical success rate (0.4 weight)
2. Task complexity (0.3 weight)
3. Prerequisites availability (0.3 weight)

### Fix Steps

#### Step 1: Add Debug Logging
Edit: `/home/opc/.claude/daemon/lib/confidence-engine.sh`

In `calculate_task_confidence()` function (line 229), add logging:

```bash
calculate_task_confidence() {
    local task_description="$1"
    local goal_json="$2"
    local persona="${3:-}"

    # Component scores
    local historical_success
    historical_success=$(get_historical_success_rate "$task_description" "$persona")
    log "DEBUG" "Confidence DEBUG: historical=$historical_success"  # ADD THIS

    local complexity
    complexity=$(estimate_task_complexity "$task_description")
    log "DEBUG" "Confidence DEBUG: complexity=$complexity"  # ADD THIS

    local prerequisites
    prerequisites=$(check_prerequisites_availability "$goal_json")
    log "DEBUG" "Confidence DEBUG: prerequisites=$prerequisites"  # ADD THIS

    # ... rest of function
}
```

#### Step 2: Run Debug Cycle
```bash
# Restart daemon
~/.claude/daemon/claude-daemon-restart.sh

# Wait 30 minutes for autonomy cycle
sleep 1800

# Check debug logs
tail -50 ~/.claude/daemon/logs/activity.log | grep "Confidence DEBUG"
```

#### Step 3: Analyze Results
Look for pattern:
```
Confidence DEBUG: historical=0.5
Confidence DEBUG: complexity=0.3
Confidence DEBUG: prerequisites=0.5
```

If you see:
- **All 0.5**: Historical data sparse, complexity underestimating, prerequisites neutral
- **Complexity=0.3 always**: Task description too generic, complexity estimation broken
- **Prerequisites=0.5 always**: Goal JSON missing blocker/dependency data

#### Step 4: Fix Based on Finding

**If complexity is always low**:
Edit: `lib/confidence-engine.sh` function `estimate_task_complexity()` (line 118)

Increase base complexity or adjust weights:
```bash
estimate_task_complexity() {
    local description="$1"
    local complexity_score=0.5  # CHANGE FROM 0.3 to 0.5 (higher baseline)

    # ... rest
}
```

**If historical is always 0.5**:
Check `decision-log.jsonl` has entries:
```bash
wc -l ~/.claude/daemon/logs/decision-log.jsonl
# Should have >20 entries
```

If sparse, historical defaults to 0.5. Need more historical data.

**If prerequisites is always 0.5**:
Check goal JSON has blockers/dependencies:
```bash
jq '.active_goals[0] | {blockers, dependencies}' ~/.claude/daemon/state/goals.json
```

If empty, goals missing prerequisite data. Update goal metadata.

#### Step 5: Validate Fix
```bash
# Check new confidence scores
tail -50 ~/.claude/daemon/logs/activity.log | grep "confidence:" | sort -u
```

Should now see range like:
```
confidence: 0.3
confidence: 0.5
confidence: 0.7
confidence: 0.85
```

---

## Issue #2: Episodic Memory Not Triggered 🟡 MEDIUM

### Problem
`memory/episodes.jsonl` never created despite multi-step workflows

### Root Cause Unknown
Check if `execute_with_retry()` calls `create_episode()`

### Fix Steps

#### Step 1: Check Integration
```bash
# Verify execute_with_retry calls create_episode
grep -n "create_episode" ~/.claude/daemon/lib/retry-orchestrator.sh

# Should show calls around retry loop
```

If no results: Episode creation not integrated

#### Step 2: Check Episode File Exists
```bash
ls -la ~/.claude/daemon/memory/episodes.jsonl 2>/dev/null || echo "File missing"

# If missing, create directory and initialize
mkdir -p ~/.claude/daemon/memory
touch ~/.claude/daemon/memory/episodes.jsonl
```

#### Step 3: Add Logging to Retry Loop
Edit: `/home/opc/.claude/daemon/lib/retry-orchestrator.sh`

Around line 150-200 (main retry loop), verify episode creation:

```bash
# Add near top of execute_with_retry function
log "DEBUG" "Episode DEBUG: Starting episode for task=$task_title"

# After first attempt
if [ "$attempt" -eq 1 ]; then
    local episode=$(create_episode "$goal_id" "$task_title")
    log "DEBUG" "Episode DEBUG: Created episode=$episode"
fi

# After each action
add_action_to_episode "$episode" "$action_json"
log "DEBUG" "Episode DEBUG: Added action to episode"

# After success/failure
close_episode "$episode" "$final_status"
log "DEBUG" "Episode DEBUG: Closed episode, status=$final_status"
```

#### Step 4: Trigger Multi-Step Workflow
Wait for novel project to execute multiple tasks:
```bash
# Check if episode file created
tail -10 ~/.claude/daemon/logs/activity.log | grep "Episode DEBUG"

# Check if episodes.jsonl has content
wc -l ~/.claude/daemon/memory/episodes.jsonl
```

#### Step 5: Verify Lessons Extracted
```bash
# Check first episode
head -1 ~/.claude/daemon/memory/episodes.jsonl | jq '.lessons'

# Should show lessons like:
# {
#   "patterns": ["25-step workflow succeeds reliably"],
#   "success_indicators": ["completion time <10min"],
#   "efficiency": "fast-track eligible"
# }
```

---

## Issue #3: Verification Planning Visibility 🟡 MEDIUM

### Problem
Verification plans shown as injected=0 but checks=12

### Fix Steps

#### Step 1: Add Explicit Logging
Edit: `/home/opc/.claude/daemon/lib/verification-planner.sh`

In `generate_verification_plan()` function (line ~90):

```bash
generate_verification_plan() {
    local task_description="$1"

    local task_type=$(detect_task_type "$task_description")
    log "DEBUG" "VerifyPlan: Detected task_type=$task_type"

    local verification=$(generate_from_template "$task_type" "$task_description")
    log "DEBUG" "VerifyPlan: Generated checks=$(echo $verification | jq '.checks | length')"

    echo "$verification"
}
```

#### Step 2: Check Integration Point
Verify in `task-generator.sh` (line ~450):

```bash
# Should have call to inject verification plans
grep -n "inject_verification_plans\|generate_verification_plan" \
    ~/.claude/daemon/lib/task-generator.sh
```

If missing, add integration:
```bash
# In generate_tasks() function, before return statement:
tasks=$(inject_verification_plans_to_tasks "$tasks")
```

#### Step 3: Validate Output
```bash
# Check logs for VerifyPlan messages
tail -100 ~/.claude/daemon/logs/activity.log | grep "VerifyPlan"

# Check tasks in queue have verification_plan field
head -20 ~/.claude/daemon/tasks/queue.md | grep -A2 "Description"
```

Should see verification checks listed.

---

## Testing Plan After Fixes

### Quick Test (1 hour)
```bash
# Restart with fixes
~/.claude/daemon/claude-daemon-restart.sh

# Wait for one autonomy cycle (30 min)
sleep 1800

# Check confidence variation
tail -50 ~/.claude/daemon/logs/activity.log | grep "confidence:" | sort -u

# Check episode creation
wc -l ~/.claude/daemon/memory/episodes.jsonl

# Check verification plans
grep -c "VERIFY\|OUTPUT" ~/.claude/daemon/logs/activity.log
```

### Full Validation (6-8 hours)
```bash
# Let system run for 3-4 autonomy cycles
# Monitor success rate improvement

# After fixes, expected:
# Confidence scores: 0.3-0.85 (varied)
# Episodes created: 2+ (multi-step tracking)
# Verification checks: 20+ (active verification)
# Success rate: 75%+ (back to baseline or better)
```

---

## Success Criteria

After implementing all fixes:

- [ ] Confidence scores vary (min-max range > 0.3)
- [ ] All 3 confidence components log properly
- [ ] Episodes.jsonl created and contains valid episodes
- [ ] Lessons extracted from episodes
- [ ] Verification plans visible in logs
- [ ] Success rate improves to 75%+
- [ ] Rollback scenario tested and working
- [ ] System ready for production deployment

---

## Rollback Plan

If fixes cause issues:

```bash
# Revert to previous working state
git checkout HEAD~1 lib/confidence-engine.sh
git checkout HEAD~1 lib/retry-orchestrator.sh
git checkout HEAD~1 lib/verification-planner.sh

# Restart
~/.claude/daemon/claude-daemon-restart.sh
```

---

## Files to Edit

1. `/home/opc/.claude/daemon/lib/confidence-engine.sh` (lines 229-279)
2. `/home/opc/.claude/daemon/lib/retry-orchestrator.sh` (lines 150-200)
3. `/home/opc/.claude/daemon/lib/verification-planner.sh` (lines 90-120)
4. `/home/opc/.claude/daemon/lib/task-generator.sh` (verify integration, line ~450)

---

## Questions to Investigate

Before implementing fixes:

1. What does `decision-log.jsonl` contain?
   ```bash
   head -3 ~/.claude/daemon/logs/decision-log.jsonl | jq .
   ```

2. What's in current goal?
   ```bash
   jq '.active_goals[0] | {blockers, dependencies, success_criteria}' \
       ~/.claude/daemon/state/goals.json
   ```

3. Are task descriptions detailed enough?
   ```bash
   grep "Step 6:" ~/.claude/daemon/logs/activity.log | head -1
   ```

Answers will guide root cause and fix strategy.

---

**Next Action**: Start with Issue #1 debug logging, run one cycle, analyze results

