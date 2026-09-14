# Quick Start: Understanding the Daemon

## What Is It?

A **1,762-line bash script** that acts as a multi-personality autonomous AI system. It runs 6 different personas (Experimenter, Maintainer, Skeptic, Auditor, Architect, Optimizer) that:
- Share the same memory and goals
- Switch based on 4 decision layers (Activation Floor, Health Emergency, Chaos, Emotional, Circadian)
- Execute 3 types of work (Tasks, Reflection, Conversation)
- Sleep/wake on circadian rhythm (7AM-10PM EDT active)

## One Cycle (Roughly Every 30-90 Minutes)

```
1. DETERMINE PERSONALITY (Which persona should be active?)
   ↓ Check: Activation Floor → Health → Chaos → Emotional → Circadian
   
2. CHECK WORK HOURS (7AM-10PM EDT or sleep?)
   ↓ If working, continue; if nighttime, sleep 8 hours
   
3. SELECT ACTION TYPE (Task / Reflection / Conversation?)
   ↓ Random weighted: 70% task, 10% reflection, 20% conversation
   
4. EXECUTE ACTION
   ├─ TASK: Get task from queue, call Claude API, mark complete
   ├─ REFLECTION: Self-examine, update emergence log, check cooldown
   └─ CONVERSATION: Check for human messages, engage if found
   
5. HEALTH CHECK (Any persona cooldowns expired?)
   ↓ Make personas available for next switch if cooldown passed
   
6. CALCULATE SLEEP (How long until next wake?)
   ↓ Sleep 30-90 min (day), 8 hours (night), then loop back to step 1
```

## Key Files

**State (Updated every cycle)**:
- `~/.claude/daemon/personalities/state.json` - Current persona + stats
- `~/.claude/daemon/logs/activity.log` - All decisions logged

**Configuration (Read every cycle)**:
- `~/.claude/daemon/triggers/circadian.json` - Hour-by-hour persona preferences
- `~/.claude/daemon/triggers/emotional.json` - Frustration tracking + thresholds

**Work Queue**:
- `~/.claude/daemon/tasks/queue.md` - Active tasks under `## Pending Tasks`

**Memory (Append-only)**:
- `~/.claude/daemon/memory/persona-timeline.jsonl` - All events (JSON lines)
- `~/.claude/daemon/memory/emergence-log.md` - Self-observations

**Monitoring**:
- `~/.claude/daemon/logs/activity.log` - Main activity (current: 42K+ lines)
- `~/.claude/daemon/metrics/switch-history.jsonl` - Persona switches

## Decision Layers (When Choosing Personality)

Checked in order. **Higher layers override lower.**

```
LAYER 0:  ACTIVATION FLOOR
          If any persona inactive >12h → Force their activation
          (Ensures all personas get work)

LAYER 0.5: HEALTH EMERGENCY
           If persona locked/in cooldown → Emergency replace
           (System health before other triggers)

LAYER 1:  CHAOS INJECTION (10% random)
          Random switch regardless of state
          (Breaks stagnation, forces innovation)

LAYER 2:  EMOTIONAL TRIGGERS
          If frustrated (3+ failures) → Switch to complementary persona
          (Escape ruts by changing approach)

LAYER 3:  CIRCADIAN RHYTHM
          Time-based preference: 7AM=Skeptic, 9AM=Architect, 2PM=Optimizer
          (Different personas peak at different times)

FALLBACK: Stay current persona (if nothing triggers above)
```

## 3 Types of Actions

### 1. TASK (70% probability, Lines 880-960)
```
- Get next task from queue (persona-specific)
- Read persona definition (from archetypes/{persona}.md)
- Build prompt with persona instructions + task
- Call Claude API → Claude executes task as that persona
- Mark complete (or retry if failed)
```

### 2. REFLECTION (10% probability, Lines 1399-1590)
```
- Check cooldown (can't reflect more often than every 60 min)
- Build reflection prompt (self-examine, update observations)
- Call Claude API → Claude self-reflects as persona
- Update emergence-log.md with findings
```

### 3. CONVERSATION (20% probability, Lines 1716-1719)
```
- Check inbox/daemon/unread/ for human messages
- If found: Engage in conversation with human
- Mark message as read
- Log to persona-voice.log
```

## Resilience & Safety

**Error Handling**: Every action wrapped in try-catch. Single failure doesn't crash daemon.

**State Atomicity**: Uses `mktemp + atomic mv` to prevent concurrent write corruption.

**Append-Only Logs**: Critical files use `flock`-based atomic append (no lost entries).

**Cooldown System**: Prevents personas from switching too rapidly (prevents flip-flopping).

## Performance

- **Uptime**: 73+ hours without crash ✅
- **Cycles**: 1,845 completed (~1 per minute average)
- **Activity Log**: 3.5 MB (42K+ lines)
- **Memory**: ~18 MB
- **Personas Used**: All 6 get work regularly (balanced distribution)

## Quick Debugging Commands

```bash
# Watch decisions in real-time
tail -f ~/.claude/daemon/logs/activity.log

# Check current state
jq '.' ~/.claude/daemon/personalities/state.json

# View recent persona switches
tail ~/.claude/daemon/metrics/switch-history.jsonl | jq

# Check emotional state
jq '.current_state' ~/.claude/daemon/triggers/emotional.json

# View tasks being worked on
cat ~/.claude/daemon/tasks/queue.md | head -20

# Attach to tmux session
tmux attach -t claude-daemon

# Check if running
ps aux | grep daemon.sh | grep -v grep
```

## Configuration Tuning

**Token Efficiency** (adjusted 2025-11-07):
- Task weight: 70% (was 50%)
- Reflection weight: 10% (was 30%)
- Sleep: 30-90 min (was 12-37 min)
- Result: 50% token reduction

**Thinking Levels** (per persona):
- Auditor, Skeptic: "think hard/harder" (deep analysis)
- Others: "think" (standard depth)

**Work Hours**: 7AM-10PM EDT only
**Night Hours**: 10PM-7AM EDT (8-hour sleep)

## Design Philosophy

1. **Multi-Persona**: Different personalities solve problems differently
2. **Emergent**: Unexpected switches + collaboration = novelty
3. **Long-Running**: Designed for weeks/months continuous operation
4. **Chaos-Bounded**: 10% random switch prevents local maxima
5. **Resilient**: Single failures don't crash system
6. **Audited**: Every decision logged for analysis

## Next Steps

- Review detailed explanation: `DAEMON-DETAILED-EXPLANATION.md`
- View visual flowchart: `DAEMON-VISUAL-FLOWCHART.txt`
- Watch live activity: `tail -f logs/activity.log`
- Check system health: `jq '.' personalities/state.json`

---

**Status**: ✅ OPERATIONAL (73+ hours uptime)
**Last Updated**: 2025-12-12 03:37 UTC
**Next Wake**: ~30-90 minutes from last cycle
