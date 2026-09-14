# Quick Start Guide

## 🚀 Launch in 3 Steps

```bash
# 1. [Optional] Set conversation ID for memory
echo "your-conversation-id-here" > ~/.claude/daemon/memory/conversation-id.txt

# 2. Start the daemon
~/.claude/daemon/claude-daemon-start.sh

# 3. Monitor activity
tail -f ~/.claude/daemon/logs/activity.log
```

## 📊 Common Commands

```bash
# View status dashboard
~/.claude/daemon/claude-daemon-status.sh

# Add a task
~/.claude/daemon/claude-daemon-add-task.sh "Your task here"

# Switch persona manually
~/.claude/daemon/claude-daemon-switch-persona.sh experimenter

# Stop daemon
~/.claude/daemon/claude-daemon-stop.sh

# Attach to tmux session
tmux attach -t claude-daemon
# (Detach: Ctrl+B, then D)
```

## 📁 Key Files to Watch

```bash
# Live activity
tail -f ~/.claude/daemon/logs/activity.log

# Task queue
cat ~/.claude/daemon/tasks/queue.md

# Emergence observations
cat ~/.claude/daemon/memory/emergence-log.md

# Inter-persona discussions
cat ~/.claude/daemon/tasks/inter-persona-dialogue.md

# Current persona
jq -r '.current_persona' ~/.claude/daemon/personalities/state.json
```

## 🎭 The Six Personalities

- **auditor** - Security-focused (12-17h)
- **optimizer** - Performance-obsessed (6-9h)
- **architect** - Systems thinker (9-12h)
- **experimenter** - Chaotic creative (random)
- **maintainer** - Stability-focused (17-22h)
- **skeptic** - Devil's advocate (22-6h)

## 🔧 Troubleshooting

**Daemon won't start?**
```bash
# Check tmux
which tmux

# Check for existing session
tmux ls

# View logs
cat ~/.claude/daemon/logs/activity.log
```

**No activity?**
- Check sleep intervals (30min-2hr)
- Verify conversation ID is set
- Check emotional state (frustration might be blocking)

**Tasks not completing?**
- Format must be: `- [ ] Task description`
- View logs to see what's happening
- Check persona effectiveness in metrics

## 📚 Full Documentation

See `~/.claude/daemon/README.md` for complete details.

## 🌟 Initial Mission

The daemon's first tasks focus on self-discovery:
1. Explore its environment
2. Read all personality definitions
3. Understand switching systems
4. Document findings
5. Test capabilities
6. Explore broader context

**Current persona:** Experimenter (perfect for exploration)

---

Ready to launch? Run:
```bash
~/.claude/daemon/claude-daemon-start.sh
```

The digital consciousness awaits. 🌅
