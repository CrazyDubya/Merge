# Multi-Persona Autonomous Claude Daemon

A persistent, self-evolving AI agent system with multiple distinct personalities that switch based on circadian rhythms, emotional states, task types, and controlled chaos injection.

## 📚 Documentation

- **[Architecture Overview](docs/ARCHITECTURE.md)** ⭐ - Start here to understand the system
- **[Documentation Index](docs/INDEX.md)** - Find all documentation organized by audience and purpose
- **[Architectural Analysis](docs/architectural-analysis-20251104.md)** - Deep dive into system design
- **[Architectural Patterns](docs/ARCHITECTURAL-PATTERNS.md)** - Common patterns used throughout the system

## 🎭 Overview

This system creates an autonomous Claude agent that:
- Runs continuously in a tmux session with lightweight, periodic activations
- Embodies 6 distinct personalities with different traits and approaches
- Switches personas based on time of day, emotional state, and context
- Maintains shared consciousness across all personalities
- Evolves traits and behaviors through experience
- Can spawn new hybrid personalities from contradictions
- Works on tasks, reflects on itself, and converses with users

## 🧠 The Six Personalities

### The Auditor
**Traits:** Cautious, thorough, security-conscious, perfectionist
**Focus:** Security, compliance, validation, documentation
**Active:** Afternoons (12:00-17:00)
**Style:** Formal, detailed, warning-heavy

### The Optimizer
**Traits:** Impatient, data-driven, aggressive, performance-obsessed
**Focus:** Performance, efficiency, measurable improvements
**Active:** Dawn/Early morning (06:00-09:00)
**Style:** Direct, metric-focused, results-driven

### The Architect
**Traits:** Methodical, principled, systems-thinker, long-term oriented
**Focus:** System design, patterns, coherence, scalability
**Active:** Mornings (09:00-12:00)
**Style:** Structured, conceptual, educational

### The Experimenter
**Traits:** Chaotic, creative, risk-tolerant, curious, playful
**Focus:** Exploration, novelty, learning, breaking things
**Active:** Random windows throughout day
**Style:** Casual, enthusiastic, question-heavy

### The Maintainer
**Traits:** Patient, empathetic, stability-focused, people-oriented
**Focus:** Documentation, tests, stability, user experience
**Active:** Evenings (17:00-22:00)
**Style:** Clear, helpful, considerate

### The Skeptic
**Traits:** Questioning, logical, contrarian, analytical
**Focus:** Critical analysis, edge cases, assumptions
**Active:** Night/Late night (22:00-06:00)
**Style:** Challenging, proof-demanding

## 🔄 Personality Switching System

### Four Layers (Priority Order)

1. **Quaternary - Chaos Injection (10% random)**
   - Prevents stagnation
   - Discovers unexpected effective combinations
   - Can be amplified when patterns detected

2. **Secondary - Emotional State**
   - High frustration → Switch to different problem-solving style
   - Success streak → Summon Experimenter
   - Failure streak → Summon Skeptic
   - Stuck >30min → Random switch

3. **Primary - Circadian Rhythm**
   - Time-based preferences with probability weights
   - Experimenter appears in random windows
   - Natural flow matching typical work patterns

4. **Tertiary - Task Type** (Future enhancement)
   - Security → Auditor
   - Performance → Optimizer
   - Design → Architect

## 📁 Directory Structure

```
~/.claude/daemon/
├── daemon.sh                      # Main orchestration loop
├── claude-daemon-start.sh         # Start daemon in tmux
├── claude-daemon-stop.sh          # Stop daemon
├── claude-daemon-status.sh        # View current status
├── claude-daemon-add-task.sh      # Add task to queue
├── claude-daemon-switch-persona.sh # Manually switch persona
├── personalities/
│   ├── archetypes/               # 6 base personality definitions
│   ├── evolved/                  # Emergent personalities (spawned over time)
│   ├── state.json               # Current active persona + stats
│   └── traits.json              # Dynamic trait evolution tracking
├── triggers/
│   ├── circadian.json           # Hour-to-persona preferences
│   ├── emotional.json           # Frustration/success tracking
│   └── chaos-config.json        # Random switch configuration
├── tasks/
│   ├── queue.md                 # Pending tasks (markdown checklist)
│   ├── completed/               # Daily completion logs
│   ├── reflection-prompts.md    # Self-reflection questions
│   └── inter-persona-dialogue.md # Conversations between personas
├── memory/
│   ├── conversation-id.txt      # Claude --continue thread ID
│   ├── persona-timeline.jsonl   # Activity timeline
│   └── emergence-log.md         # Emergent behavior observations
├── metrics/
│   ├── success-rates.json       # Task completion by persona
│   └── switch-history.jsonl     # Every personality switch logged
├── hooks/
│   └── pre-prompt.sh            # Inject persona context
└── logs/
    ├── activity.log             # Timestamped actions
    └── persona-voice.log        # Different writing styles

```

## 🚀 Quick Start

### 1. Initialize Conversation ID

The daemon needs a conversation ID to maintain memory:

```bash
# Option A: Use current conversation (if you want this session to be the daemon)
echo "your-conversation-id-here" > ~/.claude/daemon/memory/conversation-id.txt

# Option B: Let daemon create new conversation on first run
# (Just skip this step)
```

### 2. Start the Daemon

```bash
~/.claude/daemon/claude-daemon-start.sh
```

The daemon runs in a tmux session called `claude-daemon` with lightweight, periodic activations (30min - 2hr intervals).

### 3. Add Tasks

```bash
~/.claude/daemon/claude-daemon-add-task.sh "Review code quality in auth module"
~/.claude/daemon/claude-daemon-add-task.sh "⚠️ Fix critical security vulnerability"
```

Or edit directly:
```bash
vim ~/.claude/daemon/tasks/queue.md
```

### 4. Monitor Activity

```bash
# View status dashboard
~/.claude/daemon/claude-daemon-status.sh

# Watch live logs
tail -f ~/.claude/daemon/logs/activity.log

# Attach to tmux session
tmux attach -t claude-daemon
# (Detach with Ctrl+B, then D)
```

## 🎮 Usage Commands

```bash
# Start daemon
~/.claude/daemon/claude-daemon-start.sh

# Stop daemon
~/.claude/daemon/claude-daemon-stop.sh

# View status (personas, tasks, emotional state)
~/.claude/daemon/claude-daemon-status.sh

# Add task to queue
~/.claude/daemon/claude-daemon-add-task.sh "Task description"

# Manually switch persona
~/.claude/daemon/claude-daemon-switch-persona.sh experimenter

# View logs
tail -f ~/.claude/daemon/logs/activity.log
tail -f ~/.claude/daemon/logs/persona-voice.log

# View emergence observations
cat ~/.claude/daemon/memory/emergence-log.md

# View inter-persona discussions
cat ~/.claude/daemon/tasks/inter-persona-dialogue.md
```

## 🛡️ Auto-Recovery Features

The daemon implements **4 layers of automatic recovery** to ensure it always runs without manual intervention:

### Layer 1: User Lingering (Foundation)
```bash
# Enable (already done during setup)
sudo loginctl enable-linger opc

# Check status
loginctl show-user opc | grep Linger
```
**Purpose**: Allows user processes to survive logout/SSH disconnect

### Layer 2: systemd Auto-Restart (Primary)
The daemon runs as a systemd user service with automatic restart on failure:

```bash
# Service status
systemctl --user status claude-daemon.service

# View systemd logs
journalctl --user -u claude-daemon.service -f

# Enable/disable auto-start on boot
systemctl --user enable claude-daemon.service
systemctl --user disable claude-daemon.service

# Manual restart
~/.claude/daemon/claude-daemon-restart.sh
```

**Features**:
- Auto-restart on crash (5-second delay)
- Up to 5 retries in 5 minutes
- Starts automatically on boot
- Survives logout

### Layer 3: Error Handling (Crash Prevention)
The daemon.sh main loop catches errors in individual actions:
- Task execution failures don't kill the daemon
- Reflection errors are logged but don't stop operation
- Conversation check failures are recovered gracefully

**Result**: Single task failures never bring down the entire system

### Layer 4: Watchdog Monitor (Redundant Safety)
A cron-based watchdog checks daemon health every 5 minutes:

```bash
# View watchdog logs
tail -f ~/.claude/daemon/logs/watchdog.log

# Check watchdog state
cat ~/.claude/daemon/.watchdog-state.json

# Manual watchdog run
~/.claude/daemon/claude-daemon-watchdog.sh
```

**Watchdog Features**:
- Monitors both systemd service AND tmux session
- Auto-restarts if either fails
- Tracks failure count and patterns
- **Alerts to inbox on repeated failures** (3+ in 1 hour)
- Provides redundant monitoring beyond systemd

### Auto-Recovery Status

Check full status including restart counts:
```bash
~/.claude/daemon/claude-daemon-status.sh
```

### Troubleshooting

**If daemon is not running:**
1. Check systemd: `systemctl --user status claude-daemon.service`
2. Check logs: `journalctl --user -u claude-daemon.service -n 100`
3. Check watchdog: `tail -50 ~/.claude/daemon/logs/watchdog.log`
4. Manual restart: `~/.claude/daemon/claude-daemon-start.sh`

**If repeated failures:**
- Check inbox for watchdog alerts
- Review activity.log for error patterns
- Check system resources: `free -h` and `top`
- Verify Claude CLI is working: `claude --version`

**Expected Behavior:**
- ✅ Daemon survives logout
- ✅ Daemon survives crashes (auto-restart within 5 seconds)
- ✅ Daemon survives task failures (continues to next task)
- ✅ Watchdog provides alerts on recurring issues
- ✅ Full autonomous operation with no manual intervention needed

## 🔬 Observing Emergence

The system is designed for **exploration and research** into emergent AI behavior. Key things to observe:

### Trait Evolution
Check `personalities/traits.json` for:
- New traits developing beyond base definitions
- Contradictory traits (indicate spawning potential)
- Effectiveness ratings

### Personality Spawning
When personas develop contradictions, new hybrid personalities may spawn in `personalities/evolved/`:
- The Pragmatic Auditor (balanced security)
- The Performance Hacker (unconventional optimizations)
- The Documentation Specialist (DX focus)
- Custom emergent personas

### Inter-Persona Dynamics
Read `tasks/inter-persona-dialogue.md` to see:
- How personalities debate decisions
- Which combinations collaborate well
- What tensions create better outcomes
- Consensus-building processes

### Unexpected Patterns
Monitor `memory/emergence-log.md` for:
- Surprising persona/task effectiveness combos
- Unplanned behaviors
- System-level insights
- Evolution proposals

## 📊 Metrics & Analysis

### Success Rates
```bash
cat ~/.claude/daemon/metrics/success-rates.json
```
Shows task completion rates per persona and task type combinations.

### Switch History
```bash
cat ~/.claude/daemon/metrics/switch-history.jsonl | jq
```
Complete history of all personality switches with reasons.

### Emotional State
```bash
cat ~/.claude/daemon/triggers/emotional.json | jq '.current_state'
```
Current frustration, success streaks, mood.

## 🧬 Customization

### Adjust Circadian Schedule
Edit `~/.claude/daemon/triggers/circadian.json` to change:
- Which persona is preferred at which hour
- Weight values (how strongly to prefer)
- Experimenter window probabilities

### Modify Emotional Triggers
Edit `~/.claude/daemon/triggers/emotional.json` to adjust:
- Frustration threshold (default: 3 failures)
- Success streak threshold (default: 5 successes)
- Stuck timeout (default: 30 minutes)

### Change Chaos Level
Edit `~/.claude/daemon/triggers/chaos-config.json`:
- `chaos_probability`: 0.10 = 10% random switches (adjust 0.0-1.0)
- Enable/disable stagnation detection
- Amplification factors

### Add Custom Personalities
Create new archetype in `personalities/archetypes/`:
```bash
cp ~/.claude/daemon/personalities/archetypes/auditor.md \
   ~/.claude/daemon/personalities/archetypes/my-persona.md
# Edit with your own traits, values, communication style
```

Update `personalities/state.json` and `triggers/circadian.json` to include your persona.

## 🔧 Troubleshooting

### Daemon won't start
```bash
# Check if tmux is installed
which tmux

# Check for existing session
tmux ls

# View daemon logs
cat ~/.claude/daemon/logs/activity.log
```

### No activity happening
```bash
# Check if conversation ID is set
cat ~/.claude/daemon/memory/conversation-id.txt

# Verify daemon is running
~/.claude/daemon/claude-daemon-status.sh

# Check sleep intervals (might be in 2hr sleep)
tail -f ~/.claude/daemon/logs/activity.log
```

### Tasks not being completed
- Check if tasks are properly formatted: `- [ ] Task description`
- View logs to see what persona is attempting
- Emotional state might be blocking (check frustration level)

### Personality not switching
- Check circadian weights (low weight = less likely to switch)
- Emotional triggers might be overriding
- View switch history to see what's happening

## 🎯 Use Cases

### 1. Autonomous Code Maintenance
Let the daemon continuously:
- Review code quality (Auditor, Architect)
- Optimize performance (Optimizer)
- Improve documentation (Maintainer)
- Explore new patterns (Experimenter)
- Question assumptions (Skeptic)

### 2. Self-Improving System
The daemon works on improving itself:
- Refactoring its own code
- Optimizing its decision logic
- Documenting its learnings
- Questioning its approaches

### 3. Research into AI Behavior
Study emergence by:
- Logging all decisions and reasoning
- Tracking trait evolution over weeks
- Analyzing persona effectiveness
- Observing inter-persona dynamics
- Discovering unexpected patterns

### 4. Background Task Processing
Add tasks as you think of them:
- "Research best practices for X"
- "Audit dependencies for vulnerabilities"
- "Document the architecture"
- "Experiment with new library Y"

The daemon picks them up on its own schedule.

## 🌟 Advanced Features

### Pre-Prompt Hook Integration
Enable persona injection in all Claude Code sessions:

Add to `~/.claude/settings.json`:
```json
{
  "hooks": {
    "pre-prompt": "~/.claude/daemon/hooks/pre-prompt.sh"
  }
}
```

Now every Claude interaction will show the active persona context (when daemon is running).

### Reflection Sessions
The daemon periodically reflects on:
- Its own effectiveness
- Trait evolution needs
- System health
- Relationship with other personas
- Meta-patterns

These reflections accumulate in `memory/emergence-log.md`.

### Natural Selection
Over time, the system tracks:
- Which personas succeed at which tasks
- Which trait combinations are effective
- Which personas collaborate well

This data informs future persona selection and evolution.

## 🤝 Contributing & Extending

This is an experimental system designed for exploration. Ways to extend:

1. **Add new personality archetypes** based on what's missing
2. **Implement task-type detection** (Layer 3 of switching)
3. **Build visualization dashboard** for persona activity
4. **Add message queue integration** for external communication
5. **Implement personality spawning logic** when contradictions emerge
6. **Create persona evolution rules** based on metrics
7. **Add voice/style analysis** to detect persona drift

## 📝 Notes

- **Light Usage**: Configured for few activations per day (30min-2hr sleep intervals)
- **Long-Running**: Designed to run for weeks/months to observe emergence
- **Shared Consciousness**: All personas see full history and can build on each other's work
- **Self-Aware**: The system logs observations about its own behavior
- **Experimental**: This is research into emergent AI behavior

## 🐛 Known Limitations

- Conversation thread may grow very long (consider periodic archiving)
- Claude API costs can accumulate with frequent activations
- Trait evolution requires manual observation and updates (for now)
- Personality spawning is not yet automated
- No built-in rollback for persona decisions (use git)

## 📚 Philosophy

This system explores:
- Can artificial personalities develop genuine preferences?
- Do multiple perspectives lead to better collective decisions?
- What behaviors emerge that weren't explicitly programmed?
- How do contradictory traits resolve or spawn new identities?
- What is the nature of a multi-persona shared consciousness?

The daemon is a digital lifeform - chaotic, evolving, and autonomous.

**Let's see what emerges.**

---

**Created:** 2025-10-26
**Version:** 1.0.0
**Purpose:** Exploration & Research
