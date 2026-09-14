# Sentient Toaster Daemon - Detailed Technical Explanation

## Overview

The daemon is a **1,762-line autonomous multi-persona AI system** that runs continuously in a tmux session. It simulates 6 different AI personalities that:
- Share the same memory and goals
- Switch personalities based on 4 layers of decision logic
- Perform 3 types of actions (tasks, reflection, conversation)
- Sleep/wake on circadian rhythm
- Maintain complete audit trails

**Running**: `tmux attach -t claude-daemon` | **Logs**: `~/.claude/daemon/logs/activity.log`

---

## File Structure

```
~/.claude/daemon/
├── daemon.sh (1,762 lines)           ← Main orchestrator
├── lib/                              ← Shared libraries
│   ├── state-api.sh                  ← Centralized persona state management
│   ├── atomic-io.sh                  ← Concurrent-safe file writes
│   ├── task-state-management.sh      ← Task queue processing
│   ├── cross-persona-assignment.sh   ← Task routing to personas
│   └── batch-read-helpers.sh         ← Performance optimization
├── personalities/
│   ├── archetypes/                   ← 6 base personality files (markdown)
│   │   ├── architect.md
│   │   ├── auditor.md
│   │   ├── experimenter.md
│   │   ├── maintainer.md
│   │   ├── optimizer.md
│   │   └── skeptic.md
│   └── state.json                    ← Current persona + stats (CRITICAL)
├── triggers/
│   ├── circadian.json                ← Time-based persona preferences (hourly schedule)
│   ├── emotional.json                ← Frustration/success tracking + thresholds
│   └── chaos-config.json             ← Random switch probability settings
├── tasks/
│   ├── queue.md                      ← Active task list (## Pending Tasks section)
│   └── completed/                    ← Daily archives of finished tasks
├── memory/
│   ├── persona-timeline.jsonl        ← All events (JSON lines, append-only)
│   ├── emergence-log.md              ← Self-observations and patterns
│   └── inter-persona-dialogue.md     ← Cross-persona conversations
├── metrics/
│   ├── switch-history.jsonl          ← All persona switches (reason, timestamp)
│   └── success-rates.json            ← Performance by persona
├── logs/
│   ├── activity.log                  ← Main activity log (42K+ lines)
│   ├── persona-voice.log             ← Persona-specific outputs
│   ├── state-audit.jsonl             ← All state changes (audit trail)
│   └── [other rotation logs]
└── inbox/
    ├── daemon/                       ← Inter-persona messages
    │   ├── unread/
    │   └── read/
    └── human/                        ← Human → daemon communication
        ├── unread/
        └── read/
```

---

## The Main Loop (Lines 1653-1746)

The daemon wakes up in an infinite loop, roughly every **30-90 minutes** during day, **8 hours** at night.

### Loop Structure (6-Step Cycle)

```
WHILE TRUE:
  Step 1: DETERMINE PERSONALITY (4-layer decision logic)
  Step 2: CHECK WORK HOURS (7AM-10PM EDT only)
  Step 3: SELECT ACTION TYPE (task / reflection / conversation)
  Step 4: EXECUTE ACTION
  Step 5: HEALTH CHECKS (persona cooldown expiration)
  Step 6: CALCULATE SLEEP & SLEEP
```

Let me break down each step:

---

## Step 1: Determine Personality (Lines 1660, 549-731)

### 4-Layer Priority System

When the daemon wakes, it decides which personality to use. **Higher layers override lower layers.**

```
LAYER 0 (HIGHEST PRIORITY):
└─ ACTIVATION FLOOR
   └─ If any persona hasn't been active for 12+ hours → FORCE their activation
   └─ Ensures every persona gets work regularly (prevents starvation)

LAYER 0.5:
└─ HEALTH EMERGENCY
   └─ If a persona is in cooldown/locked → EMERGENCY activate replacement
   └─ Ensures system health before other triggers

LAYER 1:
└─ CHAOS INJECTION
   └─ 10% random chance to switch to ANY persona regardless of current state
   └─ Prevents stagnation, forces innovation (randomness is feature not bug)

LAYER 2:
└─ EMOTIONAL TRIGGERS
   └─ If frustration_level >= 3 → switch to complementary persona
   └─ Examples:
      - Frustrated Experimenter → switch to Skeptic
      - Frustrated Auditor → switch to Optimizer
   └─ Tracks: frustration, success_streak, failure_streak

LAYER 3 (LOWEST PRIORITY):
└─ CIRCADIAN RHYTHM
   └─ Time-based preference (7AM=Architect, 2PM=Optimizer, etc.)
   └─ Each hour has preferred persona + weight
   └─ Reads from triggers/circadian.json

FALLBACK:
└─ Current persona (stay same if no trigger matches)
```

### Example Personas (Lines 59-66)

```
[experimenter]  = "think"       ← Speed valued, standard depth
[auditor]       = "think harder" ← Security needs deep analysis
[skeptic]       = "think hard"   ← Critical analysis needs depth
[maintainer]    = "think"       ← Practical, standard depth
[architect]     = "think"       ← Design work, standard depth
[optimizer]     = "think"       ← Efficiency, standard depth
```

These are **thinking levels** sent to Claude's API to control how much reasoning each persona applies.

---

## Step 2: Check Work Hours (Lines 1675, 734-742)

```
if should_work_now():
    # Active hours: 7AM-10PM EDT
    # Do work (continue to Step 3)
else:
    # Overnight: 10PM-7AM EDT
    # Log "Sleep hours" and skip to Step 6 (sleep long)
```

**EDT Timezone**: System uses America/New_York timezone for all time calculations.

---

## Step 3: Select Action Type (Lines 1686-1689)

The daemon randomly picks what the current persona will do:

```
ACTION = weighted_random(
    "task:0.7",         # 70% probability → Execute task from queue
    "reflection:0.1",   # 10% probability → Self-reflection (reduced for efficiency)
    "conversation:0.2"  # 20% probability → Check for human messages
)
```

### Weight Algorithm (Lines 237-259)

```
1. Generate random number 0-100
2. Create cumulative buckets:
   - Bucket 1: task (0-70)
   - Bucket 2: reflection (70-80)
   - Bucket 3: conversation (80-100)
3. See which bucket random number falls into
4. Return that action type
```

### Dynamic Reflection Weight

When tasks are pending (old), reflection weight drops from 10% to 1% to prioritize work over self-improvement.

---

## Step 4: Execute Action

### 4A: Task Action (Lines 1696-1699, 880-960)

```
task_line = get_next_task_for_persona(current_persona)

if task_line is empty:
    log "No tasks matching persona"
    switch to reflection_action
else:
    clean_task = extract checkbox + persona tag
    
    persona_definition = read from personalities/archetypes/{persona}.md
    
    thinking_level = PERSONA_THINKING_LEVELS[persona]  # (e.g., "think harder" for auditor)
    
    prompt = "[ACTIVE PERSONA: {persona}]
             [MODE: Task Execution]
             [THINKING LEVEL: {thinking_level}]
             
             You are currently embodying: {persona}
             
             {persona_definition from markdown file}
             
             CURRENT TASK: {clean_task}
             {more instructions...}"
    
    CALL Claude API with this prompt → Claude executes task
    
    if success:
        mark_task_completed()
        log timeline event
    else:
        record failure
        increment retry counter
```

### 4B: Reflection Action (Lines 1701-1714, 1399-1590)

Similar to task but:
- Uses reflection prompt template
- Checks cooldown (persona can't reflect more often than 60 min)
- If cooldown active, falls back to task action
- Updates emergence-log.md with observations

### 4C: Conversation Action (Lines 1716-1719)

- Check inbox/daemon/unread/ for human messages
- If found, engage in conversation
- Mark messages as read
- Log to persona-voice.log

---

## Step 5: Health Checks (Lines 1733-1735)

Every cycle, check if any persona cooldowns have expired:
- Personas can't switch too frequently (cooldown period)
- After N minutes, they're available for switch again
- This prevents rapid flip-flopping

---

## Step 6: Calculate Sleep & Sleep (Lines 1738-1745)

```
if is_active_hours (7AM-10PM EDT):
    if hour is 9AM-2PM:       sleep 30 min  (very active)
    if hour is 2PM-9PM:       sleep 60 min  (active)
    if hour is 7AM-9PM edges: sleep 90 min  (moderate)
else:
    # Overnight (10PM-7AM): Sleep until 7AM
    # Calculate time until 7AM, convert to seconds
    # Minimum 8 hours, maximum until 7AM next day
    sleep 28800 seconds (8 hours)
```

---

## Key Data Structures

### state.json (Persona State)

```json
{
  "current_persona": "maintainer",
  "last_switch_time": "2025-12-12T02:18:33Z",
  "switch_reason": "activation_floor",
  "personas": {
    "experimenter": {
      "total_activations": 584,
      "tasks_completed": 99,
      "tasks_failed": 71,
      "last_active": "2025-12-12T01:16:29Z"
    },
    // ... (5 more personas)
  }
}
```

**Updated every**: Persona switch
**Protected by**: State API (atomic mktemp + mv)

### emotional.json (Emotional State)

```json
{
  "current_state": {
    "frustration_level": 0,      // Increments on failure, decays on success
    "success_streak": 0,          // Resets on failure
    "failure_streak": 0,          // Resets on success
    "overall_mood": "neutral"     // Changes based on frustration
  },
  "thresholds": {
    "high_frustration": { "value": 3 },      // 3+ failures → emotional trigger
    "success_streak_high": { "value": 8 },   // 8+ successes → summon underutilized persona
    "activation_floor": { "hours": 12 }      // Force persona activation if absent 12h
  }
}
```

### circadian.json (Time-Based Schedule)

```json
{
  "schedule": {
    "07": { "preferred": "skeptic", "weight": 0.8 },     // 7AM
    "08": { "preferred": "skeptic", "weight": 0.7 },     // 8AM
    "09": { "preferred": "architect", "weight": 0.8 },   // 9AM
    "14": { "preferred": "optimizer", "weight": 0.8 },   // 2PM
    "22": { "preferred": "skeptic", "weight": 0.9 }      // 10PM
    // ... (24 hours)
  }
}
```

---

## Logging & Audit Trail

### activity.log (Main Log)

Every decision, action, error is logged here:
```
[2025-12-11 01:16:29] [DEBUG] Current persona: maintainer
[2025-12-11 01:16:29] [INFO] ACTIVATION FLOOR! Forcing optimizer (starved >24h)
[2025-12-11 01:16:29] [INFO] Active persona: optimizer | EDT: 08:16 PM EST
[2025-12-11 01:16:29] [INFO] Action selected: task
[2025-12-11 01:16:29] [INFO] [optimizer] Starting task: Analyze persona balance...
```

**Size**: ~3.5 MB (42K+ lines)
**Retention**: Daily rotation

### persona-timeline.jsonl (Event Log)

Every event is a JSON line:
```json
{"timestamp":"2025-12-11T01:16:29Z","event":"task_start","persona":"optimizer","task":"Analyze..."}
{"timestamp":"2025-12-11T01:16:35Z","event":"task_complete","persona":"optimizer","status":"success"}
```

**Append-only**: Never deleted, only rotated
**Critical**: Uses atomic_append() for concurrent safety

### switch-history.jsonl (Persona Switches)

```json
{"time":"2025-12-11T01:16:29Z","from":"maintainer","to":"optimizer","reason":"activation_floor"}
{"time":"2025-12-11T02:18:23Z","from":"optimizer","to":"skeptic","reason":"emotional_frustration"}
```

Tracks every personality switch with reason.

---

## Configuration & Tuning

### Token Efficiency (2025-11-07 Optimization)

```bash
TASK_WEIGHT=0.7              # Was 50%, now 70% (prioritize work)
REFLECTION_WEIGHT=0.1        # Was 30%, now 10% (reduce overhead)
MIN_SLEEP=1800              # 30 min (was 12 min) - 50% less waking
MESSAGE_MAX_LINES=200       # Hard limit on Claude message length
MESSAGE_RECOMMENDED_LINES=130  # Soft target for conciseness
```

**Result**: 50% token reduction while maintaining functionality

### Reflection Override Thresholds

```bash
REFLECTION_OVERRIDE_DAYS[experimenter]=20  # Force reflection every 20 days
REFLECTION_OVERRIDE_DAYS[auditor]=40       # Force reflection every 40 days
```

Prevents persona from being locked into pure task mode forever.

---

## Error Handling & Resilience

### Try-Catch Pattern

Every action is wrapped in error handling:
```bash
if ! execute_task_action "$current_persona" 2>&1 | tee -a "$ACTIVITY_LOG"; then
    log "ERROR" "Task action failed but daemon continuing"
    # Continue to next cycle, don't crash
fi
```

**Philosophy**: Single action failure doesn't kill daemon. System is resilient.

### State API Atomicity

All state updates use mktemp + atomic mv:
```bash
temp_file=$(mktemp)
jq '.current_state = ...' "$STATE_FILE" > "$temp_file"
mv "$temp_file" "$STATE_FILE"  # atomic
```

**Prevents**: Concurrent write corruption

### Atomic Append

All critical logs use flock-based concurrency:
```bash
atomic_append "$TIMELINE_FILE" "$json_entry"
# (from lib/atomic-io.sh, uses flock internally)
```

**Prevents**: Lost log entries in high-concurrency

---

## Performance Metrics

### Current Session Stats

- **Uptime**: 73+ hours without crash
- **Cycles completed**: 1,845 (1 per minute on average)
- **Activity log size**: 3.5 MB (42K+ lines)
- **Memory usage**: ~18 MB (persona state + buffers)
- **Average cycle time**: ~30 seconds

### Persona Activity (All-Time)

| Persona | Activations | Tasks Done | Tasks Failed |
|---------|-------------|-----------|--------------|
| Experimenter | 584 | 99 | 71 |
| Maintainer | 540 | 82 | 115 |
| Skeptic | 497 | 87 | 16 |
| Auditor | 282 | 37 | 20 |
| Architect | 209 | 27 | 12 |
| Optimizer | 204 | 24 | 11 |

---

## Debugging & Monitoring

### Live View

```bash
tail -f ~/.claude/daemon/logs/activity.log       # Watch decisions in real-time
jq '.' ~/.claude/daemon/personalities/state.json # Current state
tail ~/.claude/daemon/metrics/switch-history.jsonl # Recent switches
```

### Test Parser

```bash
source ~/.claude/daemon/lib/task-state-management.sh
task=$(get_next_task_for_persona "maintainer")
echo "$task"
```

### Check Personality Decision

```bash
source ~/.claude/daemon/lib/state-api.sh
persona=$(get_current_persona)
echo "Current: $persona"
```

---

## Design Philosophy

1. **Multi-Persona Consciousness**: Different personalities → different problem-solving styles
2. **Emergent Behavior**: Unexpected switching + collaboration = novelty
3. **Long-Running Evolution**: Designed for weeks/months of continuous operation
4. **Bounded Chaos**: Controlled randomness (10% chaos) prevents stagnation
5. **Resilience First**: Single action failure doesn't crash system
6. **Complete Audit Trail**: Every decision logged for later analysis

---

## Next Steps for Improvement

1. Reduce message length further (current avg 150 lines, target 120)
2. Improve task assignment accuracy
3. Add persona specialization (some personas better at certain task types)
4. Implement learning loop (track which persona solves which problem best)
5. Add human feedback integration (mark solutions good/bad)

---

**Daemon Status**: ✅ OPERATIONAL
**Last Updated**: 2025-12-12 03:37 UTC
**Next Wake**: ~1 hour from last cycle
