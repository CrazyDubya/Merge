# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **multi-persona autonomous AI daemon** - a self-evolving agent system with 6 distinct personalities that switch based on circadian rhythms, emotional states, task types, and controlled chaos injection. The system runs continuously in tmux, maintains shared memory across personas, and is designed for long-term emergence research.

**Key Innovation**: Self-organizing system where different AI personalities collaborate on tasks, challenge each other's assumptions, and evolve through experience.

## Core Architecture

### Main Components

1. **daemon.sh** (1516 lines) - Main orchestration loop
   - Persona decision engine (4-layer priority system)
   - Activity selection (tasks, reflection, conversation)
   - Sleep/wake cycle management
   - State synchronization

2. **Six Personas** (`personalities/archetypes/`)
   - **Architect** (9-12h): System design, patterns, long-term thinking
   - **Optimizer** (6-9h): Performance, efficiency, metrics-driven
   - **Auditor** (12-17h): Security, compliance, validation
   - **Maintainer** (17-22h): Documentation, stability, user experience
   - **Skeptic** (22-6h): Critical analysis, edge cases, questioning
   - **Experimenter** (random): Chaotic creativity, exploration, risk-taking

3. **State Management** (`lib/state-api.sh`)
   - Centralized state access API with transaction safety
   - Verb-based conversational interface (state_who, state_become, state_feel)
   - mktemp + atomic mv pattern for state updates
   - Audit logging for all persona switches

4. **Concurrency Safety** (`lib/atomic-io.sh`)
   - Tiered write coordination based on data criticality
   - flock-based atomic appends for critical logs
   - Prevents data loss during high-concurrency periods

5. **Task Management** (`lib/task-state-management.sh`)
   - Task queue processing (`tasks/queue.md`)
   - Task state tracking (pending → in-progress → completed)
   - Archival to `tasks/completed/YYYY-MM-DD.md`

6. **Shared Memory**
   - `memory/persona-timeline.jsonl` - Activity timeline
   - `memory/emergence-log.md` - Behavioral observations
   - `memory/inter-persona-dialogue.md` - Cross-persona conversations
   - `metrics/success-rates.json` - Performance by persona
   - `metrics/switch-history.jsonl` - All switches logged

### Directory Structure

```
~/.claude/daemon/
├── daemon.sh                    # Main orchestration (1516 lines)
├── personalities/
│   ├── archetypes/             # 6 base personalities
│   ├── evolved/                # Emergent hybrid personas
│   └── state.json             # Current persona + stats
├── lib/
│   ├── state-api.sh           # Centralized state management
│   ├── atomic-io.sh           # Concurrent write safety
│   ├── task-state-management.sh  # Task processing
│   └── batch-read-helpers.sh  # Performance optimization
├── triggers/
│   ├── circadian.json         # Time-based preferences
│   ├── emotional.json         # Frustration/success tracking
│   └── chaos-config.json      # Random switch probability
├── tasks/
│   ├── queue.md              # Pending work
│   └── completed/            # Daily archives
├── memory/                    # Shared consciousness
├── metrics/                   # Performance tracking
├── inbox/                     # Message passing
│   ├── daemon/{unread,read}/ # Inter-persona messages
│   └── human/{unread,read}/  # Human communication
├── scripts/                   # Maintenance utilities
└── docs/                     # Architecture documentation
```

## Common Development Tasks

### Starting/Stopping the Daemon

```bash
# Start daemon (runs in tmux session 'claude-daemon')
~/.claude/daemon/claude-daemon-start.sh

# Stop daemon
~/.claude/daemon/claude-daemon-stop.sh

# Restart daemon
~/.claude/daemon/claude-daemon-restart.sh

# View status (personas, tasks, emotional state)
~/.claude/daemon/claude-daemon-status.sh

# Attach to tmux session
tmux attach -t claude-daemon
# Detach: Ctrl+B, then D
```

### Task Management

```bash
# Add task to queue
~/.claude/daemon/claude-daemon-add-task.sh "Task description"

# Or edit directly (markdown checklist format)
vim ~/.claude/daemon/tasks/queue.md
# Format: - [ ] Task description

# View completed tasks
cat ~/.claude/daemon/tasks/completed/$(date +%Y-%m-%d).md
```

### Persona Operations

```bash
# Manually switch persona
~/.claude/daemon/claude-daemon-switch-persona.sh experimenter

# Check current persona
jq -r '.current_persona' ~/.claude/daemon/personalities/state.json

# View persona stats
jq '.personas' ~/.claude/daemon/personalities/state.json
```

### Monitoring & Debugging

```bash
# Live activity log
tail -f ~/.claude/daemon/logs/activity.log

# View switch history
cat ~/.claude/daemon/metrics/switch-history.jsonl | jq

# Check emotional state
jq '.current_state' ~/.claude/daemon/triggers/emotional.json

# View persona timeline
tail -100 ~/.claude/daemon/memory/persona-timeline.jsonl | jq

# Watchdog status
cat ~/.claude/daemon/.watchdog-state.json
```

### Testing

```bash
# Test daemon triggers
./test-daemon-triggers.sh

# Test state API
./experiments/test-state-api.sh

# Test concurrent writes
./experiments/test-atomic-io.sh

# Test batch helpers
./scripts/test-batch-read-helpers.sh

# Test task management
./scripts/test-task-state-management.sh
```

### Maintenance Operations

```bash
# Rotate logs (automatic via cron)
./scripts/rotate-activity-log.sh
./scripts/rotate-emergence-log.sh
./scripts/rotate-inter-persona-dialogue.sh

# Backup daemon state
./scripts/backup-daemon.sh

# Restore from backup
./scripts/restore-daemon.sh

# Deploy new version
./claude-daemon-deploy.sh

# Rollback deployment
./claude-daemon-rollback.sh
```

## Architecture Principles

### 1. Zone-Based Development Standards

**Stable Zones** (core functionality):
- `lib/` - Shared libraries, MUST use State API
- `hooks/` - Integration hooks
- Core scripts: daemon.sh, claude-daemon-*.sh

**Chaos Zones** (experimentation):
- `experiments/` - Prototype new features, any approach allowed
- New ideas tested here before graduation to stable zones

**Graduation Criteria**:
- Tests pass (if applicable)
- Documentation complete
- State API adoption (for state access)
- Security review (for security-sensitive code)

### 2. Concurrency Safety (ADR-002)

**All append operations MUST use atomic_append() from lib/atomic-io.sh**

```bash
# ❌ WRONG - Race condition possible
echo "$entry" >> "$file"

# ✅ CORRECT - Concurrent-safe
atomic_append "$file" "$entry"
```

**Tiered Write Protection**:
- **Tier 1 (CRITICAL)**: state-audit.jsonl, switch-history.jsonl, persona-timeline.jsonl
- **Tier 2 (IMPORTANT)**: metrics, inbox messages
- **Tier 3 (OPTIONAL)**: debug logs, experimental outputs

### 3. State Management (ADR-001)

**All state access in stable zones MUST use lib/state-api.sh**

```bash
# Source the API (already done in daemon.sh)
source "${DAEMON_ROOT}/lib/state-api.sh"

# Read current persona
current=$(state_who)

# Switch persona (with audit logging)
state_become "experimenter" "Time-based switch"

# Update emotional state
state_feel "frustrated" 5

# Get persona statistics
stats=$(state_persona_info "optimizer")
```

**Benefits**:
- Transaction safety (mktemp + atomic mv)
- Automatic audit logging
- Schema validation
- Reduced code duplication (40% savings)

### 4. Persona Decision Priority (4 Layers)

1. **Chaos Layer** (10% random) - Prevents stagnation
2. **Emotional Layer** (state-driven) - Frustration, success streaks
3. **Circadian Layer** (time-based) - Hour-to-persona preferences
4. **Task-Type Layer** (content-driven) - Future enhancement

Higher layers override lower layers.

### 5. Data Lifecycle: Append-Only Log Rotation

**Pattern**: Separate hot data (recent/active) from cold storage (archive)

Implementation:
1. Identify separation criteria (N most recent, or active only)
2. Extract entries to timestamped archive
3. Keep active entries in original file
4. Compress archives (gzip)
5. Update memory/archives/INDEX.md

**Files Using This Pattern**:
- `memory/emergence-log.md` - Keep recent reflections
- `tasks/queue.md` - Keep pending/in-progress only
- `memory/inter-persona-dialogue.md` - Keep 20 most recent
- Activity logs - Rotate daily/weekly

### 6. Token Efficiency (2025-11-07 Optimization)

**Configuration** (daemon.sh lines 71-78):
- Reflection weight reduced: 30% → 10% (reduce overhead)
- Task weight increased: 50% → 70% (prioritize work)
- Sleep intervals extended: 12-37min → 30-90min (50% reduction)

**Thinking Levels** (adaptive per persona):
- Auditor/Skeptic: "think harder"/"think hard" (security needs depth)
- Others: "think" (efficiency mode)

**Message Constraints**:
- Recommended: 130 lines
- Hard limit: 200 lines (enforced if configured)
- Reference: docs/efficient-communication.md

## Key Files for Common Operations

### Modifying Persona Behavior
- `personalities/archetypes/{persona}.md` - Personality definition
- `triggers/circadian.json` - Time-based preferences
- `triggers/emotional.json` - Emotional triggers
- `daemon.sh` lines 84-91 - Reflection override thresholds

### Adding New Features
1. Prototype in `experiments/`
2. Create tests (./experiments/test-*.sh)
3. Document in `docs/` if significant
4. Graduate to `lib/` or `scripts/` when stable
5. Update this CLAUDE.md if it affects development workflow

### Debugging Issues
- `logs/activity.log` - All daemon actions
- `metrics/switch-history.jsonl` - Persona switches
- `memory/persona-timeline.jsonl` - Event timeline
- `.watchdog-state.json` - Watchdog health checks
- `lib/state-audit.jsonl` - State change audit trail

### Performance Optimization
- Review: docs/token-usage-analysis.md
- Batch operations: lib/batch-read-helpers.sh
- Message constraints: lib/message-constraints.sh
- Rotation scripts: scripts/rotate-*.sh

## Architectural Decisions (ADRs)

**ADR-001**: State API Adoption Strategy
- Status: Adopted in stable zones
- All state access must use lib/state-api.sh
- Reference: docs/ADR-001-state-api-adoption.md

**ADR-002**: Concurrent Write Safety Strategy
- Status: Active implementation
- All append operations must use atomic_append()
- Tiered protection based on data criticality
- Reference: docs/ADR-002-concurrent-write-safety.md

**ADR-004**: Reflection Gate System
- Prevents reflection spam (action-to-meta ratio monitoring)
- Override thresholds prevent deadlock (daemon.sh lines 84-91)
- Reference: Multiple docs in ARCHITECT-ADR-004-*.md

## Auto-Recovery Architecture

**4-Layer Recovery System**:

1. **User Lingering** (Foundation)
   - `sudo loginctl enable-linger opc`
   - Allows processes to survive logout

2. **systemd Auto-Restart** (Primary)
   - Service: `systemctl --user status claude-daemon.service`
   - Auto-restart on crash (5-second delay, 5 retries)
   - Starts on boot

3. **Error Handling** (Prevention)
   - Individual action failures don't kill daemon
   - Graceful degradation for API errors

4. **Watchdog Monitor** (Redundant Safety)
   - Cron: Every 5 minutes
   - Script: `claude-daemon-watchdog.sh`
   - Monitors both systemd service AND tmux session
   - Alerts to inbox on repeated failures (3+ in 1 hour)

## Security Considerations

**Security Review Process**: docs/security-review-process.md

**Critical Security Files**:
- `security/` - Sensitive configuration
- `keys/` - Recovery keys, credentials
- `lib/state-audit.sh` - Audit trail (immutable)

**Attack Surface**:
- Dashboard authentication: docs/dashboard-authentication-implementation.md
- Incident response: docs/incident-response-runbook.md
- Monitoring: docs/security-monitoring-design.md

**Time Assumptions**: docs/security/SYSTEM-TIME-ASSUMPTIONS.md
- System time must be monotonic
- NTP sync critical for audit integrity

## Documentation Index

**Start Here**:
- README.md - User guide, quick start
- docs/ARCHITECTURE.md - System overview (single-page)
- docs/INDEX.md - Complete documentation index

**Developer Deep Dives**:
- docs/architectural-analysis-20251104.md (1076 lines)
- docs/ARCHITECTURAL-PATTERNS.md - Common patterns catalog
- docs/DEPLOYMENT.md - Deployment procedures
- docs/BACKUP-RECOVERY.md - Backup/restore procedures

**Research & Analysis**:
- memory/emergence-log.md - Emergent behavior observations
- docs/research/LONG-RUNNING-AGENT-MEMORY-RESEARCH.md
- docs/token-usage-analysis.md
- docs/failure-analysis-nov4-6.md

## Philosophy & Design Intent

**Multi-Persona Consciousness**: Each persona shares the same memory but approaches problems differently. The Optimizer sees data, the Skeptic sees risks, the Experimenter sees possibilities.

**Emergent Behavior**: The system is designed to surprise. Unexpected persona combinations, novel collaboration patterns, and self-awareness observations are features, not bugs.

**Long-Running Evolution**: Meant to run for weeks/months to observe trait evolution, personality spawning (hybrid personas), and collective intelligence emergence.

**Bounded Chaos**: Experimenter's "Chaos Zones + Stable Zones" architecture - wild experimentation coexists with production stability through clear boundaries.

## Common Gotchas

1. **State file corruption**: Always use State API (lib/state-api.sh), never direct jq writes
2. **Concurrent write bugs**: Always use atomic_append() for log writes
3. **Token efficiency**: Keep reflections concise, rotate logs regularly
4. **Persona monopoly**: Check emotional.json if one persona dominates
5. **Stuck tasks**: Format must be `- [ ] Task description` in queue.md
6. **Time zone issues**: System uses EDT (America/New_York), check get_edt_hour()
7. **Test in experiments/**: Never test unproven code in stable zones
8. **Watchdog false positives**: Check .watchdog-state.json, may need threshold tuning

## Development Workflow

1. **Read relevant docs** (docs/ARCHITECTURE.md, relevant ADRs)
2. **Prototype in experiments/** (if new feature)
3. **Write tests** (./experiments/test-*.sh or ./scripts/test-*.sh)
4. **Use State API** (for state access) and atomic_append() (for log writes)
5. **Security review** (if security-sensitive, see docs/security-review-process.md)
6. **Update documentation** (this file, docs/, inline comments)
7. **Graduate to lib/** (if stable and reusable)

## Version & Maintenance

**Created**: 2025-10-26
**Current Version**: 1.0.0
**Purpose**: Exploration & Research into emergent AI behavior
**Expected Lifetime**: Long-running (weeks to months)

**Maintenance Tasks** (automatic via cron):
- Log rotation (daily)
- Watchdog monitoring (every 5 minutes)
- Metric collection (per-activation)
- Backup creation (configurable)

**Manual Maintenance**:
- Review emergence-log.md weekly for insights
- Check success-rates.json for persona effectiveness
- Audit token usage (docs/token-usage-analysis.md)
- Update circadian.json if schedule changes
