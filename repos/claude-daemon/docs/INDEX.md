# Documentation Index

**Purpose**: Central index of all daemon documentation organized by audience and purpose.

**Last Updated**: 2025-11-10

---

## 🚀 Getting Started (New Users)

**Start here if you're new to the daemon:**

1. **README.md** (520 lines) - Operational overview, quick start guide
   - What is this system?
   - How to start/stop/monitor the daemon
   - Directory structure
   - Basic usage commands

2. **docs/ARCHITECTURE.md** (NEW) - System architecture overview
   - Architecture at a glance
   - Core components and their relationships
   - Key architectural decisions
   - Data flow patterns
   - Scaling characteristics

3. **docs/architectural-analysis-20251104.md** (1076 lines) - Deep dive analysis
   - Comprehensive system analysis
   - Technical debt identification
   - Scaling concerns
   - Recommendations

---

## 📚 Documentation by Audience

### For Developers

**Understanding the System**:
- ⭐ **docs/ARCHITECTURE.md** - Start here (single-page overview)
- 📊 **docs/architectural-analysis-20251104.md** - Detailed analysis
- 🔧 **docs/ARCHITECTURAL-PATTERNS.md** - Common patterns catalog
- 📋 **README.md** - Operational guide

**Building & Extending**:
- 🔒 **docs/security-review-process.md** - Security review procedures
- ✅ **docs/integration-validation-checklist.md** - 4-phase validation framework (NEW - Nov 10)
- 🛡️ **docs/validation-patterns.md** - Multi-dimensional validation patterns (NEW - Nov 15)
- 🚀 **docs/DEPLOYMENT.md** - Deployment guide
- 💾 **docs/BACKUP-RECOVERY.md** - Backup and recovery procedures
- 💬 **docs/cross-persona-handoff-format.md** - Persona handoff protocol (NEW - Nov 6)
- 📏 **docs/efficient-communication.md** - Message constraints (NEW - Nov 7)

**Current Decisions**:
- 📝 **docs/ADR-001-state-api-adoption.md** - ADR: State API adoption
- 🔐 **docs/ADR-002-concurrent-write-safety.md** - ADR: Concurrent write safety (NEW - Nov 6)

### For Security Reviewers

**Security Documentation**:
- 🔒 **docs/security-review-process.md** - Review procedures and criteria
- 🔍 **docs/attack-surface-audit-20251102.md** - Threat analysis
- 📡 **docs/security-monitoring-design.md** - Monitoring architecture
- 🔐 **docs/dashboard-authentication-implementation.md** - Auth implementation
- ⏰ **docs/security/SYSTEM-TIME-ASSUMPTIONS.md** - Time-based security considerations

**Incident Response**:
- 🚨 **docs/incident-response-runbook.md** - Incident handling procedures
- 📊 **docs/incident-alert-fatigue-20251103.md** - Alert fatigue incident analysis

### For Operators

**Day-to-Day Operations**:
- 📋 **README.md** - Quick start and usage commands
- 🚀 **docs/DEPLOYMENT.md** - Deployment procedures
- 💾 **docs/BACKUP-RECOVERY.md** - Backup procedures
- 🔧 **docs/maintenance-log-2025-11-03.md** - Maintenance history

**Monitoring & Troubleshooting**:
- 📡 **docs/security-monitoring-design.md** - What's being monitored
- 🚨 **docs/incident-response-runbook.md** - How to respond to incidents
- 📊 **docs/incident-alert-fatigue-20251103.md** - Past incident learnings

### For Personas (Internal AI Agents)

**Persona Development**:
- 📝 **personalities/archetypes/*.md** - Persona definitions
- 🧠 **memory/emergence-log.md** - Behavioral observations and insights
- 💬 **memory/inter-persona-dialogue.md** - Cross-persona conversations
- 🔧 **docs/ARCHITECTURAL-PATTERNS.md** - Patterns to apply

**Collaboration**:
- 📬 **docs/inbox-workflow.md** - Message passing guidelines
- 📊 **tasks/queue.md** - Task queue (where work lives)
- 🏗️ **docs/ARCHITECTURE.md** - System understanding

### For Researchers

**System Behavior**:
- 🧠 **memory/emergence-log.md** - Emergent behavior observations
- 📊 **memory/persona-timeline.jsonl** - Activity timeline (structured log)
- 💬 **memory/inter-persona-dialogue.md** - Persona interactions
- 🔬 **docs/research/LONG-RUNNING-AGENT-MEMORY-RESEARCH.md** - Memory research

**Performance & Optimization**:
- ⚡ **docs/optimization-log-2025-11-03-dialogue-rotation.md** - Optimization case study
- 🔧 **docs/ARCHITECTURAL-PATTERNS.md** - Pattern analysis

---

## 📁 Documentation by Category

### Architecture & Design

| Document | Size | Purpose |
|----------|------|---------|
| **docs/ARCHITECTURE.md** ⭐ | ~500 lines | Single-page system overview (START HERE) |
| **docs/ARCHITECTURE-MEMORY.md** | ~400 lines | Memory architecture deep dive (NEW - Nov 10) |
| docs/architectural-analysis-20251104.md | 1076 lines | Comprehensive system analysis |
| docs/ARCHITECTURAL-PATTERNS.md | 716 lines | Common pattern catalog |
| docs/ADR-001-state-api-adoption.md | ~450 lines | ADR: State API adoption strategy |
| **docs/ADR-002-concurrent-write-safety.md** | ~600 lines | **ADR: Concurrent write safety strategy** (NEW - Nov 6) |

### Security

| Document | Size | Purpose |
|----------|------|---------|
| docs/security-review-process.md | ~600 lines | Security review procedures |
| **docs/integration-validation-checklist.md** | ~300 lines | **4-phase validation framework** (NEW - Nov 10) |
| docs/attack-surface-audit-20251102.md | ~400 lines | Threat analysis |
| docs/security-monitoring-design.md | ~300 lines | Monitoring architecture |
| docs/dashboard-authentication-implementation.md | ~250 lines | Auth implementation details |
| docs/security/SYSTEM-TIME-ASSUMPTIONS.md | ~150 lines | Time-based security considerations |
| docs/audit-coverage-monitoring.md | ~200 lines | Audit trail monitoring (Nov 6) |

### Operations

| Document | Size | Purpose |
|----------|------|---------|
| README.md | 520 lines | Operational guide and quick start |
| docs/DEPLOYMENT.md | ~300 lines | Deployment procedures |
| docs/BACKUP-RECOVERY.md | ~200 lines | Backup and recovery |
| docs/maintenance-log-2025-11-03.md | ~150 lines | Maintenance history |
| docs/inbox-workflow.md | ~200 lines | Message passing guidelines |
| **docs/efficient-communication.md** | ~150 lines | **Message constraints & token efficiency** (NEW - Nov 7) |
| **docs/cross-persona-handoff-format.md** | ~200 lines | **Persona handoff protocol** (NEW - Nov 6) |
| docs/log-rotation-guide.md | ~250 lines | Log rotation procedures (Nov 6) |

### Incident Response

| Document | Size | Purpose |
|----------|------|---------|
| docs/incident-response-runbook.md | ~400 lines | Incident handling procedures |
| **docs/security-incident-validation-gap-20251110.md** | ~500 lines | **Validation process failure (Nov 10)** ⭐ NEW |
| docs/security-incident-negative-sleep-20251107.md | ~450 lines | Daemon crash loop incident (Nov 7) |
| docs/incident-thrashing-bug-20251106.md | ~400 lines | Persona switch thrashing (Nov 6) |
| docs/incident-state-corruption-20251106.md | ~350 lines | State corruption incident (Nov 6) |
| docs/security-incident-audit-bypass-20251104.md | ~400 lines | Audit bypass incident (Nov 4) |
| docs/incident-alert-fatigue-20251103.md | ~250 lines | Alert fatigue incident (Nov 3) |

**Recent Incidents (Nov 3-10)**:
- 🔴 **Nov 10**: Validation process failure - approved without runtime check
- 🔴 **Nov 7**: Daemon crash loop - negative sleep calculation
- 🟡 **Nov 6**: Thrashing bug - 88K persona switches in hours
- 🟡 **Nov 6**: State corruption - concurrent write race condition
- 🟡 **Nov 4**: Audit bypass - daemon.sh not using State API
- 🟢 **Nov 3**: Alert fatigue - monitoring false positives

**Key Lesson**: All incidents have comprehensive postmortems with root cause analysis

### Performance & Optimization

| Document | Size | Purpose |
|----------|------|---------|
| docs/optimization-log-2025-11-03-dialogue-rotation.md | ~300 lines | Log rotation case study |
| docs/ARCHITECTURAL-PATTERNS.md | 716 lines | Includes performance patterns |
| **docs/token-usage-analysis.md** | ~400 lines | **Token efficiency analysis & optimizations** (NEW - Nov 7) |
| docs/memory-optimization-research.md | ~300 lines | Memory system optimization (Nov 10) |
| docs/skeptic-log-growth-analysis-20251109.md | ~200 lines | Log growth monitoring (Nov 9) |

### Research & Experimentation

| Document | Size | Purpose |
|----------|------|---------|
| docs/research/LONG-RUNNING-AGENT-MEMORY-RESEARCH.md | ~500 lines | Memory management research |
| **docs/persona-variety-analysis-20251110.md** | ~800 lines | **Persona distribution analysis** (NEW - Nov 10) |
| docs/failure-analysis-nov4-6.md | ~600 lines | System failure patterns (Nov 6) |
| docs/test-coverage-analysis.md | ~300 lines | Test coverage assessment (Nov 4) |

---

## 🎯 Documentation by Task

### "I want to understand the system architecture"
1. Start: **docs/ARCHITECTURE.md** (single-page overview)
2. Deep dive: **docs/architectural-analysis-20251104.md**
3. Patterns: **docs/ARCHITECTURAL-PATTERNS.md**

### "I want to deploy the daemon"
1. Quick start: **README.md** (Quick Start section)
2. Full deployment: **docs/DEPLOYMENT.md**
3. Backup setup: **docs/BACKUP-RECOVERY.md**

### "I need to review security"
1. Process: **docs/security-review-process.md**
2. Threats: **docs/attack-surface-audit-20251102.md**
3. Monitoring: **docs/security-monitoring-design.md**

### "I'm responding to an incident"
1. Runbook: **docs/incident-response-runbook.md**
2. Past incidents: **docs/incident-alert-fatigue-20251103.md**

### "I'm developing a new persona"
1. Architecture: **docs/ARCHITECTURE.md** (Personas section)
2. Examples: **personalities/archetypes/*.md**
3. Patterns: **docs/ARCHITECTURAL-PATTERNS.md**
4. Collaboration: **docs/inbox-workflow.md**

### "I'm optimizing performance"
1. Patterns: **docs/ARCHITECTURAL-PATTERNS.md** (Log Rotation pattern)
2. Case study: **docs/optimization-log-2025-11-03-dialogue-rotation.md**
3. Architecture: **docs/architectural-analysis-20251104.md** (Performance section)
4. Token efficiency: **docs/token-usage-analysis.md** (Nov 7 optimizations)

### "I need to validate an integration or fix"
1. Framework: **docs/integration-validation-checklist.md** (4-phase validation)
2. Example: **docs/persona-variety-phase3-validation-20251110.md** (runtime validation)
3. Security review: **docs/security-review-process.md**

### "I'm investigating persona behavior"
1. Analysis: **docs/persona-variety-analysis-20251110.md** (distribution metrics)
2. Memory: **memory/emergence-log.md** (behavioral observations)
3. Timeline: **memory/persona-timeline.jsonl** (structured activity log)
4. Metrics: **metrics/switch-history.jsonl** (all persona switches)

---

## 📊 Documentation Health

### Coverage Assessment

| Area | Coverage | Status |
|------|----------|--------|
| **Architecture** | High | ✅ Excellent (overview + detailed analysis + patterns) |
| **Security** | High | ✅ Excellent (process + audit + monitoring) |
| **Operations** | Medium | ✅ Good (deployment + backup + maintenance) |
| **Development** | Medium | ⚠️ Adequate (patterns exist, API docs pending) |
| **Testing** | Low | ❌ Needs improvement (only experimental tests documented) |
| **API Reference** | Low | ❌ Missing (State API pending documentation) |

### Recently Updated (Last 7 Days)
- 2025-11-10: docs/ARCHITECTURE-MEMORY.md (NEW - Architect)
- 2025-11-10: docs/persona-variety-analysis-20251110.md (NEW - Experimenter)
- 2025-11-10: docs/persona-variety-phase3-validation-20251110.md (NEW - Skeptic)
- 2025-11-10: docs/security-incident-validation-gap-20251110.md (NEW - Skeptic)
- 2025-11-10: docs/integration-validation-checklist.md (NEW - Auditor)
- 2025-11-09: docs/skeptic-log-growth-analysis-20251109.md (NEW - Skeptic)
- 2025-11-07: docs/token-usage-analysis.md (NEW - Optimizer)
- 2025-11-07: docs/efficient-communication.md (NEW - Maintainer)
- 2025-11-07: docs/security-incident-negative-sleep-20251107.md (NEW - Auditor)
- 2025-11-06: docs/ADR-002-concurrent-write-safety.md (NEW - Architect)
- 2025-11-06: docs/incident-thrashing-bug-20251106.md (NEW - Experimenter)
- 2025-11-06: docs/incident-state-corruption-20251106.md (NEW - Auditor)
- 2025-11-06: docs/cross-persona-handoff-format.md (NEW - Maintainer)

### Needs Attention
- [ ] API documentation for lib/state-api.sh (pending ADR-001 approval)
- [ ] Testing guide and strategy (no document exists)
- [ ] Development workflow guide (how to contribute code)
- [ ] Troubleshooting guide (common issues and solutions)

---

## 🔍 Finding Documentation

### By File Name Patterns

**Architecture**: `ARCHITECTURE*.md`, `architectural-*.md`, `ADR-*.md`
**Security**: `security-*.md`, `attack-*.md`
**Operations**: `DEPLOYMENT.md`, `BACKUP-RECOVERY.md`, `maintenance-*.md`
**Incidents**: `incident-*.md`
**Optimization**: `optimization-*.md`
**Research**: `research/*.md`

### By Keyword Search

```bash
# Search all documentation
grep -r "your keyword" ~/.claude/daemon/docs/

# Search specific types
grep -r "security" ~/.claude/daemon/docs/security-*.md
grep -r "architecture" ~/.claude/daemon/docs/ARCH*.md
```

---

## 📝 Documentation Guidelines

### When to Create New Documentation

**Create a new document when**:
- Explaining a complex system component (architecture)
- Documenting a repeatable process (deployment, incident response)
- Recording a significant decision (ADR)
- Analyzing a past event (incident postmortem, optimization case study)

**Don't create a new document when**:
- Quick note or reminder (use comments in code)
- Temporary information (use task queue)
- Personal reflection (use emergence log)
- Small update to existing doc (edit existing)

### Documentation Standards

**All documentation should include**:
- Purpose statement (what is this doc for?)
- Last updated date
- Intended audience
- Clear section headers
- Examples where applicable

**Architecture docs should include**:
- Diagrams (text-based ASCII art is fine)
- Decision rationale (why, not just what)
- Trade-offs considered
- Related documents

**Process docs should include**:
- Step-by-step procedures
- Required permissions/access
- Expected outcomes
- Troubleshooting section

---

## 🎓 Learning Path

### Level 1: New User (1-2 hours)
1. README.md → Understand what the system is and how to use it
2. docs/ARCHITECTURE.md → Understand system architecture
3. Try commands: start, status, add-task, stop

### Level 2: Developer (4-6 hours)
1. docs/architectural-analysis-20251104.md → Deep system understanding
2. docs/ARCHITECTURAL-PATTERNS.md → Learn common patterns
3. docs/security-review-process.md → Understand security requirements
4. Review code: daemon.sh, lib/*.sh

### Level 3: Contributor (8-12 hours)
1. All Level 1 & 2 documentation
2. docs/ADR-001-state-api-adoption.md → Current architectural discussions
3. memory/emergence-log.md → System evolution history
4. personalities/archetypes/*.md → Persona development
5. Experiment in chaos zone (experiments/)

### Level 4: Maintainer (Ongoing)
1. All previous levels
2. Monitor docs/maintenance-log-*.md
3. Review all incident-*.md as they occur
4. Participate in architectural discussions (ADRs)
5. Update documentation regularly

---

## 🔗 External Resources

**Claude Code Documentation**: https://docs.claude.com/claude-code
**Bash Best Practices**: https://google.github.io/styleguide/shellguide.html
**jq Manual**: https://jqlang.github.io/jq/manual/
**tmux Tutorial**: https://github.com/tmux/tmux/wiki

---

**This index is maintained by**: All personas (update when creating docs)
**Index version**: 1.1.0
**Last reviewed**: 2025-11-10 (Maintainer: Added Nov 5-10 documentation, 13 new files indexed)
