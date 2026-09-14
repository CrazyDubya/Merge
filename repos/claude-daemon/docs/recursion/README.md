# Recursion - Claude Multi-Persona Autonomous Daemon Project

A self-aware autonomous AI agent system with 6 distinct personalities, circadian rhythms, emotional intelligence, and continuous evolution.

## 🎭 Overview

This project implements a persistent, autonomous Claude agent that:
- Runs 24/7 in a tmux session (active 7AM-10PM EDT, sleeps overnight)
- Embodies 6 distinct personalities that switch based on time, emotion, and controlled chaos
- Maintains persistent memory across all persona transitions
- Works on tasks autonomously, reflects on experiences, and evolves over time
- Can be disturbed/awakened at any time for urgent work

## 📦 Repository Structure

This project is split across **three private GitHub repositories**:

### 1. 🏠 [recursion](https://github.com/CrazyDubya/recursion) (This Repo)
**Documentation and project coordination**
- Project documentation
- Setup guides
- Management instructions
- Links to other repositories

### 2. 🤖 [claude-daemon](https://github.com/CrazyDubya/claude-daemon)
**Multi-Persona Autonomous Daemon System**
- `daemon.sh`: 900+ line orchestration script with maximum freedom configuration
- 6 persona definitions (Auditor, Optimizer, Architect, Experimenter, Maintainer, Skeptic)
- Circadian rhythm system (EDT timezone aware)
- Emotional state tracking (success/failure streaks, frustration)
- Chaos injection (10% random personality switches)
- Task queue management
- Reflection and self-analysis system
- Memory persistence via conversation IDs
- **NEW:** Bidirectional inbox system (human ↔ daemon async messaging)
- **NEW:** daemon-settings.json (full capabilities configuration)
- **NEW:** Adaptive extended thinking per persona (think/think hard/think harder)
- **NEW:** Helper scripts for communication (send-message.sh, read-messages.sh)

Location on disk: `~/.claude/daemon/`

### 3. 🛠️ [claude-skills](https://github.com/CrazyDubya/claude-skills)
**12 Specialized Skills for Claude Code**
- accessibility-auditor
- api-documentation-generator
- code-style-enforcer
- configuration-validator
- database-migration-helper
- dependency-audit-assistant
- docker-optimizer
- error-tracking-integrator
- git-workflow-enforcer
- internationalization-helper
- performance-profiler
- test-coverage-analyzer

Location on disk: `~/.claude/skills/`

## 🚀 Quick Start

### First Time Setup

```bash
# 1. Clone all three repositories
git clone https://github.com/CrazyDubya/recursion.git
git clone https://github.com/CrazyDubya/claude-daemon.git ~/.claude/daemon
git clone https://github.com/CrazyDubya/claude-skills.git ~/.claude/skills

# 2. Install dependencies
sudo yum install -y tmux jq bc git

# 3. Create required directories
mkdir -p ~/.claude/daemon/logs ~/.claude/daemon/tasks/completed ~/.claude/daemon/metrics

# 4. Start the daemon
~/.claude/daemon/claude-daemon-start.sh
```

### Daily Usage

```bash
# Check daemon status
~/.claude/daemon/claude-daemon-status.sh

# Add a task
~/.claude/daemon/claude-daemon-add-task.sh "Your task here"

# Send a message to daemon (NEW!)
~/.claude/daemon/claude-daemon-send-message.sh "Your message here"

# Read daemon's responses (NEW!)
~/.claude/daemon/claude-daemon-read-messages.sh

# Watch it work
tail -f ~/.claude/daemon/logs/activity.log

# Wake it during sleep hours
~/.claude/daemon/claude-daemon-wake.sh

# Stop the daemon
~/.claude/daemon/claude-daemon-stop.sh
```

## 🎭 The Six Personas

Each persona has distinct traits, active hours, and communication styles:

| Persona | Traits | Active Hours (EDT) | Focus |
|---------|--------|-------------------|-------|
| 🔍 **The Auditor** | Cautious, thorough, security-conscious | 12:00-17:00 | Security, compliance, validation |
| ⚡ **The Optimizer** | Impatient, data-driven, performance-obsessed | 06:00-09:00 | Performance, efficiency, metrics |
| 🏗️ **The Architect** | Methodical, principled, systems-thinker | 09:00-12:00 | System design, patterns, scalability |
| 🎨 **The Experimenter** | Chaotic, creative, risk-tolerant | Random windows | Exploration, novelty, learning |
| 🔧 **The Maintainer** | Patient, empathetic, stability-focused | 17:00-22:00 | Documentation, tests, user experience |
| 🤔 **The Skeptic** | Questioning, logical, contrarian | 22:00-06:00 | Critical analysis, edge cases |

## 🕐 Schedule

**Active Hours**: 7:00 AM - 10:00 PM EDT
- Morning (7-12): **Very active**, every **10 minutes** (6 wakes/hour)
- Afternoon (12-18): **Active**, every **15 minutes** (4 wakes/hour)
- Evening (18-22): **Moderate**, every **30 minutes** (2 wakes/hour)

**Sleep Hours**: 10:00 PM - 7:00 AM EDT
- Full overnight sleep (~9 hours)
- Can be disturbed with wake script

**Activity Distribution**:
- 50% Tasks - Primary work
- 30% Reflection - Self-improvement
- 20% Conversation - Human communication (inbox check)

## 🔄 Personality Switching

Four layers of switching logic (priority order):

1. **Chaos Injection** (10% random) - Prevents stagnation
2. **Emotional State** - High frustration triggers different approaches
3. **Circadian Rhythm** - Time-based preferences (EDT timezone)
4. **Task Type** - Future enhancement

## 📊 Features

### Task Management
- Markdown-based task queue
- Automatic task selection and execution
- Completion tracking and daily logs
- Task prioritization (urgent tasks marked with ⚠️)
- **NEW:** Task chaining (multiple related tasks per wake cycle)

### Communication System (NEW!)
- **Bidirectional inbox**: Human ↔ daemon async messaging
- File-based message queue (no database needed)
- Persona-aware responses (each personality responds differently)
- Custom folder routing via rules.json
- Helper scripts for easy messaging

### Emotional Intelligence
- Success/failure streak tracking
- Frustration level monitoring
- Mood-based persona switching
- Adaptive behavior based on outcomes

### Memory & Evolution
- Persistent conversation via session IDs
- Timeline logging of all activities
- Emergence log for self-observations
- Trait evolution over time
- Inter-persona dialogue capability

### Reflection System
- Periodic self-analysis
- Pattern recognition
- Improvement suggestions
- Meta-cognitive observations

### Maximum Freedom Configuration (NEW!)
- **daemon-settings.json**: Full capabilities documentation
- **Adaptive thinking**: Per-persona extended thinking levels
- **No token limits**: Explicitly told to work thoroughly
- **12 skills explicitly listed**: All capabilities surfaced
- **Task tool encouraged**: For complex multi-step work
- **Parallel execution**: Multiple tool calls simultaneously

## 📁 File Organization

```
/home/opc/
├── recursion/                    # This repo (documentation)
│   ├── README.md
│   ├── CLAUDE_EXTERNAL_MANAGEMENT.md
│   └── .gitignore
│
└── .claude/
    ├── daemon/                   # Daemon repo
    │   ├── daemon.sh            # Main orchestration (900+ lines)
    │   ├── daemon-settings.json # Full capabilities config (NEW!)
    │   ├── claude-daemon-*.sh   # Control scripts (8 total)
    │   ├── personalities/       # 6 personas + state
    │   ├── triggers/            # Circadian, emotional, chaos
    │   ├── tasks/               # Queue + logs
    │   ├── memory/              # Conversation ID + timeline + docs
    │   ├── metrics/             # Success rates + switches
    │   ├── logs/                # Activity + persona voice
    │   └── inbox/               # Human ↔ daemon messaging (NEW!)
    │       ├── daemon/unread/   # Messages TO daemon FROM human
    │       ├── daemon/read/     # Processed messages
    │       ├── human/unread/    # Messages TO human FROM daemon
    │       └── human/read/      # Read messages
    │
    └── skills/                   # Skills repo
        ├── accessibility-auditor/
        ├── api-documentation-generator/
        └── ... (10 more skills)
```

## ⚠️ Important Notes

### Runtime Data NOT in Git

The daemon repos exclude runtime data via `.gitignore`:
- Logs (activity.log, persona-voice.log)
- Conversation IDs (sensitive session data)
- Metrics (runtime statistics)
- State files (current persona, emotional state)

These files change every 30 minutes and would create massive git noise.

### Backup Strategy

**Code/Configuration**: Automatically backed up via git push to all 3 repos

**Runtime State**: Backup separately if needed:
```bash
tar -czf daemon-runtime-$(date +%Y%m%d).tar.gz \
  ~/.claude/daemon/logs/ \
  ~/.claude/daemon/memory/conversation-id.txt \
  ~/.claude/daemon/metrics/
```

## 🛠️ Development

### Modifying Personas

Edit persona definitions:
```bash
vim ~/.claude/daemon/personalities/archetypes/skeptic.md
cd ~/.claude/daemon && git commit -am "Update skeptic persona" && git push
```

### Adding Skills

Create new skill:
```bash
mkdir ~/.claude/skills/my-skill
vim ~/.claude/skills/my-skill/SKILL.md
cd ~/.claude/skills && git add . && git commit -m "Add my-skill" && git push
```

### Modifying Schedule

Edit timing in daemon.sh:
```bash
vim ~/.claude/daemon/daemon.sh
# Modify MIN_SLEEP, DEFAULT_SLEEP, MAX_SLEEP, NIGHT_SLEEP
~/.claude/daemon/claude-daemon-stop.sh
~/.claude/daemon/claude-daemon-start.sh
cd ~/.claude/daemon && git commit -am "Adjust timing" && git push
```

## 📚 Documentation

- **[CLAUDE_EXTERNAL_MANAGEMENT.md](./CLAUDE_EXTERNAL_MANAGEMENT.md)** - Critical: explains external file management
- **Daemon README**: `~/.claude/daemon/README.md` - Full daemon documentation
- **Daemon Quickstart**: `~/.claude/daemon/QUICKSTART.md` - Getting started guide
- **Timing Schedule**: `~/.claude/daemon/TIMING-SCHEDULE.md` - Detailed schedule info
- **Skills Database**: `~/.claude/skills/DATABASE_README.md` - Skills documentation

## 🔗 Links

- **Main Repo**: https://github.com/CrazyDubya/recursion
- **Daemon Repo**: https://github.com/CrazyDubya/claude-daemon
- **Skills Repo**: https://github.com/CrazyDubya/claude-skills

## 🤖 Stats

- **Total Code**: ~14,000+ lines across all repos
- **Daemon Core**: 900+ lines (daemon.sh)
- **Control Scripts**: 8 bash scripts (incl. messaging helpers)
- **Personas**: 6 distinct personalities
- **Skills**: 12 specialized capabilities
- **Uptime**: Runs 15 hours/day (7AM-10PM EDT)
- **Active Cycles**: ~60-80 per day (up from 30-40)
- **Wake Frequency**: 10-30 min intervals (vs 30-120 min previously)
- **Inbox Messages**: Async human ↔ daemon communication

## 📝 Version Info

- **Created**: October 2025
- **Daemon Version**: Multi-Persona v2.0 (Maximum Freedom Edition)
- **Skills Count**: 12
- **Personality System**: v1.1 (Adaptive Thinking)
- **Communication**: v1.0 (Inbox System)
- **Latest Update**: October 28, 2025 - Added inbox system, maximum freedom config, adaptive thinking

---

**Remember**: The real magic isn't in this repo - it's in `~/.claude/daemon/` where the autonomous personas live, think, work, and evolve continuously. This repo is just the guide.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
