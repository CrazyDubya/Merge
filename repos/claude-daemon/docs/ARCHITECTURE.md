# System Architecture Overview

**Purpose**: Single-page architectural overview for developers, designers, and stakeholders to understand system structure, key decisions, and design principles.

**Last Updated**: 2025-11-04
**System Version**: 1.0.0
**Audience**: New developers, architectural reviewers, future personas

---

## What Is This System?

A **multi-persona autonomous AI agent** that runs continuously, embodies 6 distinct personalities, switches between them based on context, maintains shared memory, and evolves through experience.

**Core Innovation**: Self-organizing system with persona specialization, emotional triggers, and emergent collaboration patterns.

---

## Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────┐
│                    HUMAN INTERFACE                            │
│  Commands: start, stop, add-task, switch-persona, status     │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│                   DAEMON ORCHESTRATOR                         │
│  daemon.sh: Main loop, persona switching, activity selection │
│  ┌────────────┬────────────┬──────────────────────────────┐  │
│  │ Persona    │ Activity   │ State                        │  │
│  │ Decision   │ Selection  │ Management                   │  │
│  │ (4-layer)  │ (3 types)  │ (JSON files)                 │  │
│  └────────────┴────────────┴──────────────────────────────┘  │
└────────────────────┬─────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┬──────────────┐
        ▼            ▼            ▼              ▼
┌───────────┐  ┌──────────┐  ┌────────────┐  ┌──────────┐
│ PERSONAS  │  │  STATE   │  │  TRIGGERS  │  │  TASKS   │
│ 6 archetypes │  current │  │  4 layers  │  │  queue   │
│ + evolved │  │  persona │  │  switching │  │  pending │
│ hybrids   │  │  stats   │  │  rules     │  │  archive │
└───────────┘  └──────────┘  └────────────┘  └──────────┘
                     │            │              │
        ┌────────────┴────────────┴──────────────┘
        ▼
┌─────────────────────────────────────────────────────────────┐
│                  SHARED MEMORY                               │
│  Timeline, emergence log, inter-persona dialogue            │
│  Success rates, metrics, reflection history                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Daemon Orchestrator (`daemon.sh`)

**Purpose**: Main control loop that manages persona lifecycle, activity selection, and system orchestration.

**Key Responsibilities**:
- Read system state (persona, emotional state, time, tasks)
- Decide next persona (4-layer decision tree)
- Select activity type (task execution, self-reflection, conversation)
- Execute via Claude CLI
- Update state files
- Manage sleep/wake cycles

**Architecture Pattern**: Event-driven loop with pluggable decision layers

### 2. Six Personas

**Purpose**: Distinct AI personalities with specialized capabilities, different problem-solving approaches, and unique communication styles.

| Persona | Focus | Active Time | Traits |
|---------|-------|-------------|--------|
| **Architect** | System design, patterns, coherence | 09:00-12:00 | Methodical, principled, long-term |
| **Optimizer** | Performance, efficiency, metrics | 06:00-09:00 | Impatient, data-driven, aggressive |
| **Auditor** | Security, compliance, validation | 12:00-17:00 | Cautious, thorough, perfectionist |
| **Maintainer** | Documentation, stability, UX | 17:00-22:00 | Patient, empathetic, helpful |
| **Skeptic** | Critical analysis, edge cases | 22:00-06:00 | Questioning, logical, contrarian |
| **Experimenter** | Exploration, novelty, creativity | Random windows | Chaotic, creative, risk-tolerant |

**Evolution**: Personas can spawn hybrid personalities when contradictions emerge (e.g., "Fast Architect" from Optimizer + Architect traits).

### 3. Four-Layer Persona Decision System

**Purpose**: Determine which persona should activate next based on multiple factors.

**Priority Order** (highest first):

1. **Chaos Layer** (10% random)
   - Prevents stagnation
   - Discovers unexpected effective combinations
   - Can be amplified when patterns detected

2. **Emotional Layer** (state-driven)
   - High frustration → Switch to different problem-solving style
   - Success streak → Summon Experimenter
   - Failure streak → Summon Skeptic
   - Stuck >30min → Random switch

3. **Circadian Layer** (time-based)
   - Hour-to-persona probability weights
   - Natural flow matching work patterns
   - Experimenter in random windows

4. **Task-Type Layer** (content-driven, future)
   - Security tasks → Auditor
   - Performance tasks → Optimizer
   - Design tasks → Architect

**Design Principle**: Multi-layered decision-making prevents persona monopolization and enables context-aware switching.

### 4. State Management

**Purpose**: Persistent storage of system state, persona statistics, emotional tracking, and configuration.

**Key Files**:
- `personalities/state.json` - Current persona, activation counts, task stats
- `triggers/emotional.json` - Frustration level, success/failure streaks
- `triggers/circadian.json` - Time-based persona preferences
- `triggers/chaos-config.json` - Random switch probability
- `tasks/queue.md` - Pending tasks (markdown checklist)
- `metrics/success-rates.json` - Historical success by persona

**Architecture Decision**: JSON files (not database) for simplicity, human-readability, and git-friendly diff format.

**Transaction Safety**:
- **State file updates** (read-modify-write): mktemp + atomic mv pattern
- **Append operations** (concurrent writes): flock coordination via `lib/atomic-io.sh`
- **Concurrency Model**: Tiered write coordination based on data criticality (see ADR-002)

### 5. Shared Memory System

**Purpose**: Cross-persona knowledge sharing, behavioral evolution tracking, and system self-awareness.

**Components**:
- **Timeline** (`memory/persona-timeline.jsonl`) - Activity log, event tracking
- **Emergence Log** (`memory/emergence-log.md`) - Behavioral observations, insights
- **Inter-Persona Dialogue** (`memory/inter-persona-dialogue.md`) - Conversations between personas
- **Metrics** (`metrics/success-rates.json`) - Performance by persona

**Architecture Pattern**: Append-only logs with rotation to prevent unbounded growth.

### 6. Inbox System

**Purpose**: Asynchronous message passing between personas and with human users.

**Structure**:
```
inbox/
├── daemon/{unread,read}/    # Messages for daemon (any persona)
└── human/{unread,read}/     # Messages for human user
```

**Workflow**:
1. Persona writes message to appropriate inbox/unread/
2. Recipient persona/human reads during their active time
3. After reading, message moves to read/ folder
4. Multi-recipient messages duplicated appropriately

**Architecture Pattern**: Directory-based pub/sub with file-based messages.

---

## Key Architectural Decisions

### ADR-001: State API Adoption Strategy
**Status**: Proposed (2025-11-04)
**Decision**: Adopt centralized State API (lib/state-api.sh) as standard for stable zones
**Rationale**: 40% code duplication, 85+ direct jq calls, inconsistent transaction safety
**See**: docs/ADR-001-state-api-adoption.md

### ADR-002: Concurrent Write Safety Strategy
**Status**: Proposed (2025-11-08)
**Decision**: Implement tiered flock coordination for concurrent append operations
**Rationale**: 0.044% audit data loss during thrashing, systemic vulnerability in 16+ scripts
**See**: docs/ADR-002-concurrent-write-safety.md

### Chaos Zones + Stable Zones Architecture
**Origin**: Experimenter (2025-11-04)
**Principle**: Different governance for different areas
- **Stable Zones** (core, lib, hooks): High standards, architectural oversight
- **Chaos Zones** (experiments): Maximum freedom, rapid iteration
- **Graduation Path**: experiments/ → lib/ when validated

**Rationale**: Balance structure (Architect) with innovation (Experimenter).

### Persona Specialization over Monolithic Agent
**Decision**: 6 specialized personas instead of single general-purpose agent
**Rationale**:
- Different tasks require different approaches (security vs performance)
- Single persona can monopolize (Experimenter was 80% of activations early on)
- Specialization enables expertise development
- Diversity prevents groupthink

**Trade-off**: Coordination overhead vs improved domain expertise.

### Shell Scripts over Python/Node
**Decision**: Primary implementation in Bash
**Rationale**:
- Simple deployment (no dependencies)
- Direct system integration (tmux, systemd, jq)
- Human-readable file manipulation
- Fast prototyping

**Trade-off**: Complexity at scale vs deployment simplicity.
**Future**: May migrate to Python for complex logic (under evaluation).

### JSON State Files over Database
**Decision**: JSON files for all persistent state
**Rationale**:
- Human-readable (debugging, auditing)
- Git-friendly (version control, diffs)
- No database setup required
- Sufficient for single-daemon scale

**Trade-off**: Concurrency safety vs simplicity.
**Mitigation**: mktemp + atomic mv for transaction safety.

---

## Data Flow Patterns

### Pattern 1: Task Execution Flow

```
1. User adds task → tasks/queue.md
2. Daemon wakes → reads queue.md
3. Persona decision → 4-layer selection
4. Activity selection → "task execution" chosen
5. Execute task → claude CLI with persona context
6. Update state → completion status, emotional state
7. Archive → completed tasks rotated
```

### Pattern 2: Persona Switch Flow

```
1. Check chaos trigger → 10% random switch?
2. Check emotional state → frustration/success/failure?
3. Check circadian time → preferred persona for hour?
4. Check task type → security/performance/design? (future)
5. Select persona → highest priority layer wins
6. Update state.json → new current_persona, activation count
7. Next activation uses new persona
```

### Pattern 3: Inter-Persona Collaboration

```
1. Persona A completes work → writes message to inbox/daemon/unread/
2. Message specifies recipient → "To: Skeptic" or "To: ALL"
3. Persona B activates → reads inbox during setup
4. Persona B processes → responds, takes action
5. Persona B writes response → inbox/daemon/unread/ for A
6. Dialogue continues → async message exchange
```

---

## Architectural Patterns in Use

### 1. Append-Only Log Rotation
**Problem**: Logs grow unbounded, causing context bloat
**Solution**: Keep recent N entries, archive historical
**Applied to**: emergence-log.md, queue.md, inter-persona-dialogue.md
**See**: docs/ARCHITECTURAL-PATTERNS.md

### 2. Bounded Context Separation
**Problem**: Complex systems need clear boundaries
**Solution**: Explicit interfaces, minimal coupling, clear responsibilities
**Applied to**: inbox structure, persona domains, data lifecycle
**See**: docs/ARCHITECTURAL-PATTERNS.md

### 3. Verb-Based API Design
**Problem**: State access scattered across 85+ jq calls
**Solution**: Conversational API (state_who, state_become, state_feel)
**Applied to**: lib/state-api.sh (pending adoption)
**See**: docs/ADR-001-state-api-adoption.md

---

## Scaling Characteristics

### Current Scale
- **Scripts**: 47 shell scripts (8,134 LOC)
- **State Files**: 12 JSON files (~50KB total)
- **Personas**: 6 base + 0 evolved hybrids
- **Daily Activity**: ~20-30 activations/day
- **Task Queue**: 5-15 pending tasks typical

### Scaling Limits (Current Architecture)

| Dimension | Current | Breaking Point | Risk |
|-----------|---------|----------------|------|
| **Scripts** | 47 | ~100-150 | MEDIUM - needs modularization |
| **State Access** | 85+ jq calls | Already breaking | HIGH - ADR-001 addresses |
| **Personas** | 6 | ~10-12 | LOW - manageable growth |
| **Daily Tasks** | 20-30 | ~100-200 | MEDIUM - queue management |
| **Memory Files** | ~2MB | ~50MB | LOW - rotation handles growth |

### Scaling Strategies

**Short-term** (1-3 months):
- Adopt State API (ADR-001) → Reduces duplication 40%
- Continue log rotation → Prevents context bloat
- Maintain Chaos+Stable Zones → Enables experimentation

**Medium-term** (3-6 months):
- Modularize daemon.sh → Extract libraries
- Add task prioritization → Handle higher task load
- Implement batch operations → Reduce subprocess overhead

**Long-term** (6-12 months):
- Evaluate Python migration → Handle complexity at scale
- Add distributed state → If multi-daemon needed
- Implement task delegation → If workload exceeds single daemon capacity

---

## Security Considerations

### Current Security Posture: 7/10 (HIGH)

**Strengths**:
- Security review gate process (docs/security-review-process.md)
- Auditor persona with 48h activation floor
- Attack surface audit completed (docs/attack-surface-audit-20251102.md)
- Dashboard authentication implemented
- SSH monitoring active

**Weaknesses**:
- No centralized state validation (schema violations possible)
- File-based transactions (race conditions possible, low probability)
- Limited audit logging (no centralized security log)

**Threat Model**:
- **Highest Risk**: Malicious task injection via queue.md
- **Medium Risk**: State corruption via concurrent writes
- **Low Risk**: Unauthorized daemon access (mitigated by file permissions)

**Mitigation**:
- Task validation before execution (Auditor review for security tasks)
- Atomic file operations (mktemp + mv pattern)
- File permissions (daemon directory restricted to user)

**See**: docs/security-review-process.md, docs/attack-surface-audit-20251102.md

---

## Testing Strategy

### Current State: ~30% coverage (manual testing primary)

**What's Tested**:
- State API (experiments/test-state-api.sh) - 49/50 score
- Baseline tracking (experiments/baseline-edge-case-tests.sh) - 4/4 tests pass
- Critical path validation (manual smoke tests)

**What's NOT Tested**:
- Daemon orchestration loop (daemon.sh)
- Persona switching logic (4-layer decision)
- Error recovery paths
- Concurrent state access

**Testing Philosophy**:
- **Experiments**: Comprehensive testing required before graduation
- **Core**: Manual testing + validation sufficient for current scale
- **Production**: Monitor, measure, react pattern

**Future**: Expand test coverage to 80%+ as system complexity grows.

---

## Observability & Monitoring

### Metrics Collected
- **Persona activations**: Per-persona count, last active timestamp
- **Task completion**: Success rate by persona
- **Emotional state**: Frustration level, success/failure streaks
- **Switch history**: Every persona switch logged with reason
- **System health**: Service monitoring for critical components

### Logs
- `logs/activity.log` - Timestamped actions
- `memory/persona-timeline.jsonl` - Structured event log
- `memory/emergence-log.md` - Behavioral observations

### Dashboards
- `claude-daemon-status.sh` - Current state summary
- `claude-daemon-dashboard.sh` - Visual status (HTTP server)
- Future: Grafana/Prometheus integration for historical metrics

---

## Evolution & Emergence

### System Evolution Characteristics

**Self-Organizing Behaviors Observed**:
- Development pipeline emerged organically: Experimenter (prototype) → Skeptic (validate) → Optimizer (optimize)
- Collaboration patterns emerged without central coordination
- Personas evolved communication styles through experience

**Architectural Gaps Identified Through Emergence**:
- State API built but not adopted → Consolidation phase missing
- Security gate process emerged from incident → Now standard
- Log rotation pattern discovered at N=2 → Now documented

**Learning**: System good at emergence (building), needs architecture for consolidation (adoption, standardization).

---

## Documentation Map

**For New Developers**:
1. **Start here**: README.md (operational overview, quick start)
2. **Then read**: ARCHITECTURE.md (this document, system understanding)
3. **Deep dive**: docs/architectural-analysis-20251104.md (detailed analysis)

**For Persona Developers**:
- personalities/archetypes/*.md (persona definitions)
- memory/emergence-log.md (behavioral observations)
- docs/ARCHITECTURAL-PATTERNS.md (common patterns)

**For Security Review**:
- docs/security-review-process.md (review procedures)
- docs/attack-surface-audit-20251102.md (threat analysis)
- docs/security-monitoring-design.md (monitoring architecture)

**For Deployment**:
- docs/DEPLOYMENT.md (deployment guide)
- docs/BACKUP-RECOVERY.md (disaster recovery)
- docs/incident-response-runbook.md (incident handling)

---

## Key Contacts & Ownership

**System Architect**: Architect persona (architectural coherence, technical debt)
**Security Owner**: Auditor persona (security review, compliance)
**Performance Owner**: Optimizer persona (efficiency, metrics)
**Stability Owner**: Maintainer persona (documentation, user experience)
**Innovation Owner**: Experimenter persona (exploration, prototyping)
**Quality Owner**: Skeptic persona (validation, edge cases)

**Human Stakeholder**: Primary user/operator

---

## Future Architectural Considerations

### Under Active Discussion
1. **State API Adoption** (ADR-001) - Awaiting Experimenter approval
2. **Python Migration** - Evaluate for complex logic at scale
3. **Task Delegation** - If workload exceeds single daemon capacity

### Proposed But Not Prioritized
- Multi-daemon coordination (for distributed operation)
- Centralized logging service (structured log aggregation)
- Plugin architecture (third-party persona development)
- REST API (external system integration)

### Deferred
- Database migration (JSON files sufficient for current scale)
- Containerization (deployment simplicity preferred)
- Language migration for existing scripts (works well, no urgency)

---

## Conclusion

This is a **functionally effective but architecturally evolving system**. The core innovation (multi-persona emergence) works well. The implementation has accumulated technical debt that must be addressed before scaling 2-3x.

**Current State**: Functional chaos with emerging structure
**Target State**: Structured foundation with bounded chaos zones
**Path**: Consolidation of working patterns, adoption of proven solutions, continued experimentation in safe zones

**The system is learning to balance emergence (building) with architecture (consolidation).**

---

**Document History**:
- 2025-11-04: Initial architecture overview (Architect persona)

**Related Documents**:
- docs/architectural-analysis-20251104.md (detailed analysis, 1076 lines)
- docs/ARCHITECTURAL-PATTERNS.md (pattern catalog, 716 lines)
- docs/ADR-001-state-api-adoption.md (adoption strategy)
- README.md (operational guide, 520 lines)
