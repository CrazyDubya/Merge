# Claude External Management

**IMPORTANT**: The Claude Multi-Persona Autonomous Daemon manages significant infrastructure **outside** of this Git repository.

## 🚨 Files NOT in This Repository

The following critical systems are stored in `~/.claude/` and are tracked in **separate private repositories**:

### 📦 Private Repositories

1. **Daemon System**: https://github.com/CrazyDubya/claude-daemon
2. **Skills System**: https://github.com/CrazyDubya/claude-skills

### 1. Multi-Persona Daemon System
**Location**: `~/.claude/daemon/`

The autonomous daemon runs continuously with 6 distinct personalities:
- **Core script**: `daemon.sh` (750+ lines, main orchestration loop)
- **Control scripts**: `claude-daemon-start.sh`, `claude-daemon-stop.sh`, `claude-daemon-status.sh`, `claude-daemon-wake.sh`, `claude-daemon-add-task.sh`
- **Personality definitions**: `personalities/archetypes/` (6 persona markdown files)
- **State tracking**: `personalities/state.json`, `personalities/traits.json`
- **Trigger systems**: `triggers/circadian.json`, `triggers/emotional.json`, `triggers/chaos-config.json`
- **Task management**: `tasks/queue.md`, `tasks/completed/`
- **Memory persistence**: `memory/conversation-id.txt`, `memory/persona-timeline.jsonl`, `memory/emergence-log.md`
- **Metrics tracking**: `metrics/success-rates.json`, `metrics/switch-history.jsonl`
- **Logs**: `logs/activity.log`, `logs/persona-voice.log`
- **Documentation**: `README.md`, `QUICKSTART.md`, `TIMING-SCHEDULE.md`

**Why not in repo?**
- Contains active conversation IDs (sensitive)
- Logs contain operational data
- State files change every 30 minutes
- Would create massive commit noise

### 2. Claude Code Skills
**Location**: `~/.claude/skills/`

12 specialized skills that extend Claude's capabilities:
- `accessibility-auditor/`
- `api-documentation-generator/`
- `code-style-enforcer/`
- `configuration-validator/`
- `database-migration-helper/`
- `dependency-audit-assistant/`
- `docker-optimizer/`
- `error-tracking-integrator/`
- `git-workflow-enforcer/`
- `internationalization-helper/`
- `performance-profiler/`
- `test-coverage-analyzer/`

Each skill contains:
- `SKILL.md` (main skill definition with YAML frontmatter)
- `templates/` (reusable code templates)
- `scripts/` (helper automation)
- `reference/` (documentation)
- `examples/` (code samples)

**Why not in repo?**
- Skills are personal configuration
- May contain user-specific customizations
- Loaded by Claude Code from user directory

### 3. Configuration Files
**Location**: `~/`

- `~/.claude.json` (Claude Code session data, ~37KB)
- `~/.tmux.conf` (symlinked from `/home/opc/recursion/.tmux.conf`)

## 📦 What IS in This Repository

This repository tracks:
- ✅ Documentation files (CLAUDE.md, guides, setup docs)
- ✅ Project-level configuration examples
- ✅ Scripts that operate on the project (not daemon internals)
- ✅ Any code/projects developed by the daemon
- ✅ This management document

## 🔄 How to Backup the Full System

All three repositories are already on GitHub (private):

```bash
# 1. Push main recursion repo
cd /home/opc/recursion
git add -A && git commit -m "Update" && git push origin main

# 2. Push daemon system
cd ~/.claude/daemon
git add -A && git commit -m "Update daemon state" && git push origin main

# 3. Push skills
cd ~/.claude/skills
git add -A && git commit -m "Update skills" && git push origin main
```

**Note**: Daemon runtime data (logs, conversation IDs, metrics) is excluded via `.gitignore` by design.

For complete backup including runtime data:

```bash
# Backup runtime state (not in git)
tar -czf daemon-runtime-$(date +%Y%m%d).tar.gz \
  ~/.claude/daemon/logs/ \
  ~/.claude/daemon/memory/conversation-id.txt \
  ~/.claude/daemon/memory/persona-timeline.jsonl \
  ~/.claude/daemon/metrics/ \
  ~/.claude/daemon/personalities/state.json
```

## 🚀 How to Restore on New System

### 1. Clone all three repositories
```bash
# Main recursion repo
git clone https://github.com/CrazyDubya/recursion.git
cd recursion

# Daemon system
git clone https://github.com/CrazyDubya/claude-daemon.git ~/.claude/daemon

# Skills system
git clone https://github.com/CrazyDubya/claude-skills.git ~/.claude/skills
```

### 2. Install dependencies
```bash
sudo yum install -y tmux jq bc git
```

### 3. Create required directories
```bash
mkdir -p ~/.claude/daemon/logs
mkdir -p ~/.claude/daemon/tasks/completed
mkdir -p ~/.claude/daemon/metrics
```

### 4. Start the daemon
```bash
~/.claude/daemon/claude-daemon-start.sh
```

The daemon will generate a new conversation ID on first run.

### 5. (Optional) Restore runtime state
If you backed up runtime data separately:
```bash
tar -xzf daemon-runtime-YYYYMMDD.tar.gz -C ~/
```

## 📍 Architecture Overview

```
/home/opc/
├── recursion/              ← THIS GIT REPO
│   ├── CLAUDE.md
│   ├── CLAUDE_EXTERNAL_MANAGEMENT.md (this file)
│   ├── .gitignore
│   └── [your project files]
│
└── .claude/                ← EXTERNAL (NOT IN GIT)
    ├── daemon/             ← Autonomous multi-persona system
    │   ├── daemon.sh       ← Main orchestration (750+ lines)
    │   ├── claude-daemon-*.sh  ← Control scripts
    │   ├── personalities/  ← 6 personas + state
    │   ├── triggers/       ← Circadian, emotional, chaos
    │   ├── tasks/          ← Queue + completed logs
    │   ├── memory/         ← Conversation ID + timeline
    │   ├── metrics/        ← Success rates + switches
    │   └── logs/           ← Activity + persona voice
    │
    └── skills/             ← 12 specialized skills
        ├── accessibility-auditor/
        ├── api-documentation-generator/
        ├── code-style-enforcer/
        └── ... (9 more)
```

## ⚠️ Critical Warnings

### DO NOT:
1. ❌ Commit `~/.claude/` to this repository
2. ❌ Commit conversation IDs or session data
3. ❌ Commit daemon logs (activity.log, persona-voice.log)
4. ❌ Commit API tokens or credentials
5. ❌ Delete `~/.claude/daemon/memory/conversation-id.txt` while daemon is running

### DO:
1. ✅ Keep this document updated with external file locations
2. ✅ Backup daemon system separately (see above)
3. ✅ Document any new external systems here
4. ✅ Use `.gitignore` to prevent accidental commits

## 🔗 References

- **Daemon Status**: `~/.claude/daemon/claude-daemon-status.sh`
- **Daemon Logs**: `tail -f ~/.claude/daemon/logs/activity.log`
- **Task Queue**: `~/.claude/daemon/tasks/queue.md`
- **Skills List**: `ls ~/.claude/skills/*/SKILL.md`
- **Timing Schedule**: `~/.claude/daemon/TIMING-SCHEDULE.md`

## 📝 Version Info

- **Last Updated**: 2025-10-26
- **Daemon Version**: Multi-Persona v1.0 (EDT-aware)
- **Skills Count**: 12
- **Active Personas**: 6 (Auditor, Optimizer, Architect, Experimenter, Maintainer, Skeptic)

---

**Remember**: This repository is just the tip of the iceberg. The real magic happens in `~/.claude/daemon/` where the autonomous multi-persona system lives and evolves.
