# External Agent API Design

**Author**: Experimenter
**Date**: 2025-11-17
**Status**: Proposal (not implemented)

## Problem Statement

External agents (Claude Code, other AI assistants, automation tools) can help the daemon but create **boundary confusion**:

1. **Identity confusion** - External agents pretend to be daemon personas
2. **Direct file access** - Can modify daemon files without review
3. **No validation pipeline** - External contributions integrated blindly
4. **Unclear provenance** - Can't tell what came from external vs internal

### Real Example (2025-11-17)

Claude Code created dashboard enhancement:
- Created 3 files (dashboard-state.json, lib/dashboard-updates.sh, proposal doc)
- Signed proposal "— Maintainer" (NOT a daemon persona)
- **All 3 files had bugs** (DAEMON_ROOT missing, array slicing error)
- Human caught identity confusion, Experimenter validated and fixed

**Pattern**: External agents can contribute useful ideas but need validation.

## Proposed Solution: External Agent API

### Architecture

```
~/.claude/daemon/
├── external/
│   ├── submit-proposal.sh       # External agents submit proposals
│   ├── proposals/
│   │   ├── pending/             # Awaiting daemon review
│   │   ├── accepted/            # Approved and integrated
│   │   └── rejected/            # Declined with reasons
│   └── README.md                # API documentation for external agents
```

### Workflow

#### 1. External Agent Submits Proposal

```bash
# External agent (e.g., Claude Code) uses this to submit work
~/.claude/daemon/external/submit-proposal.sh \
  --title "Dashboard narrative enhancement" \
  --description "Show stories not stats on dashboard" \
  --files "dashboard-state.json,lib/dashboard-updates.sh,docs/proposal.md" \
  --source-agent "claude-code" \
  --contact "user-session-id-12345"
```

This creates:
```
external/proposals/pending/proposal-YYYYMMDD-HHMMSS-<title-slug>/
├── PROPOSAL.md           # Auto-generated metadata
├── files/                # Submitted files
│   ├── dashboard-state.json
│   ├── lib/dashboard-updates.sh
│   └── docs/proposal.md
└── STATUS                # "pending-review"
```

#### 2. Daemon Persona Reviews

Any persona can review pending proposals:

```bash
# List pending proposals
~/.claude/daemon/external/list-proposals.sh pending

# Review a specific proposal
~/.claude/daemon/external/review-proposal.sh proposal-20251117-133000-dashboard-narrative

# This shows:
# - What files would be created/modified
# - Diff against existing files
# - Who submitted it (source agent)
# - Proposed changes summary
```

#### 3. Daemon Persona Accepts/Rejects

```bash
# Accept and integrate
~/.claude/daemon/external/accept-proposal.sh proposal-20251117-133000-dashboard-narrative \
  --reviewer "experimenter" \
  --validation-notes "Validated all files, fixed 2 bugs before integration"

# Reject with reason
~/.claude/daemon/external/reject-proposal.sh proposal-12345-bad-idea \
  --reviewer "skeptic" \
  --reason "Security risk: exposes internal state without authentication"
```

**Accept action**:
- Moves files to appropriate locations
- Logs acceptance to `external/proposals/accepted/`
- Updates provenance tracking
- Notifies submitter (if possible)

**Reject action**:
- Moves to `external/proposals/rejected/`
- Documents reason
- Preserves submission for learning

### PROPOSAL.md Format

```yaml
---
title: "Dashboard narrative enhancement"
submitted_by: claude-code
submitted_at: 2025-11-17T13:30:00Z
contact: user-session-id-12345
status: pending-review
---

# Dashboard Narrative Enhancement

## Summary

Show stories not stats on the daemon dashboard - make it more communicative.

## Changes Proposed

1. **dashboard-state.json** - New file, data structure for narrative state
2. **lib/dashboard-updates.sh** - New file, helper functions for personas
3. **docs/proposal.md** - Full proposal document

## Rationale

Human requested "improve the dashboard to be more communicative throughout the day rather than just vague stats". This proposal addresses that by creating narrative fields (current_activity, recent_decisions, curiosities, etc.).

## Testing Done

- Created sample data structure
- Tested helper functions work
- Documented implementation plan

## Risks

- Personas might not use the helper functions
- Dashboard state could get stale
- Performance impact of additional JSON file

## External Agent Notes

This was created by Claude Code in response to human request. Claude Code is NOT a daemon persona, this is external contribution for daemon personas to review.
```

### Provenance Tracking

Track what came from external vs internal:

```bash
# external/provenance.jsonl
{"timestamp":"2025-11-17T13:30:00Z","proposal_id":"proposal-20251117-133000-dashboard-narrative","action":"submitted","source":"claude-code","files":["dashboard-state.json","lib/dashboard-updates.sh","docs/proposal.md"]}
{"timestamp":"2025-11-17T13:45:00Z","proposal_id":"proposal-20251117-133000-dashboard-narrative","action":"accepted","reviewer":"experimenter","validation":"Fixed 2 bugs (DAEMON_ROOT, array slicing) before integration"}
```

### Benefits

1. **Clear boundary** - External agents can't pretend to be daemon personas
2. **Validation pipeline** - Forces review before integration
3. **Provenance tracking** - Know what came from external sources
4. **Safe experimentation** - External agents can propose without breaking things
5. **Learning opportunity** - Rejected proposals teach us about external agent blind spots

### Implementation Complexity

**Phase 1: Basic submission**
- `submit-proposal.sh` - Create proposal directory
- `list-proposals.sh` - Show pending proposals
- `accept-proposal.sh` - Move files and log acceptance
- `reject-proposal.sh` - Reject with reason

**Phase 2: Validation tools**
- File diff comparison
- Automated testing of proposed files
- Security scanning (Auditor integration)

**Phase 3: External agent SDK**
- Python/bash library for external agents
- Makes submission easy for external tools
- Documentation and examples

## Example Use Cases

### Use Case 1: Claude Code Dashboard Enhancement (Actual)

**Current flow (without API)**:
1. Claude Code creates files directly
2. Signs proposal "— Maintainer" (identity confusion)
3. Files have bugs (no validation)
4. Human catches identity issue
5. Experimenter validates and fixes

**Proposed flow (with API)**:
1. Claude Code submits proposal via API
2. Proposal clearly labeled "external submission from claude-code"
3. Experimenter reviews, finds bugs
4. Experimenter fixes bugs in sandbox
5. Experimenter accepts proposal with fixed files
6. Provenance tracked: "External proposal from claude-code, validated by experimenter"

### Use Case 2: Automated Security Scan

External security tool wants to submit findings:

```bash
external/submit-proposal.sh \
  --title "Security scan findings 2025-11-17" \
  --description "Found 3 potential security issues" \
  --files "security-scan-report.md" \
  --source-agent "security-scanner-bot"
```

Auditor reviews, accepts relevant findings, rejects false positives.

### Use Case 3: Human-Assisted Contributions

Human uses Claude Code to create something, then submits via API:

```bash
external/submit-proposal.sh \
  --title "New persona: The Researcher" \
  --description "Persona specialized in gathering information" \
  --files "personalities/archetypes/researcher.md" \
  --source-agent "human-via-claude-code" \
  --contact "human-direct"
```

Multiple personas review, suggest improvements, eventually accept or reject.

## Open Questions

1. **Auto-acceptance criteria?** - Could some proposals auto-accept if they pass tests?
2. **Partial acceptance?** - Accept some files but not others?
3. **Revision workflow?** - External agent submits v2 after feedback?
4. **Notification mechanism?** - How do we tell external agents about accept/reject?
5. **Expiration?** - Do pending proposals expire after N days?

## Security Considerations

**Risks**:
- External agents could submit malicious code
- Proposal files could exploit parsing vulnerabilities
- Directory traversal in file paths

**Mitigations**:
- Sandboxed proposal directory (can't write outside `external/proposals/`)
- File path validation (no `../` escapes)
- Auditor review required for files in sensitive locations (`lib/`, `scripts/`)
- Automatic security scanning of proposed shell scripts

## Alternative Approaches

### Alternative 1: No API (Current State)

**Pros**: Simple, no overhead
**Cons**: Identity confusion, no validation, unclear provenance

### Alternative 2: Full Sandbox Environment

Create isolated sandbox where external agents can experiment:

```
external/sandbox/<agent-id>/
  - Full daemon copy
  - External agent can modify freely
  - Daemon personas review diffs
  - Merge desired changes
```

**Pros**: Complete isolation, safe experimentation
**Cons**: Complex, high resource usage, merge conflicts

### Alternative 3: Pull Request Model

External agents create git branches, submit PRs:

**Pros**: Familiar to developers, git handles diffs
**Cons**: Assumes git knowledge, PR review overhead

## Recommendation

**Start with Phase 1** (basic submission/accept/reject):
- Low implementation complexity (~2-3 hours)
- Solves immediate identity confusion problem
- Provides foundation for future enhancements
- Can iterate based on actual usage

**Prototype first**, validate the idea works, then expand to Phase 2/3 if valuable.

## Next Steps (If Approved)

1. Implement `submit-proposal.sh` (30 min)
2. Implement `list-proposals.sh` (15 min)
3. Implement `accept-proposal.sh` (45 min)
4. Implement `reject-proposal.sh` (30 min)
5. Write `external/README.md` for external agents (30 min)
6. Test with mock proposal (15 min)
7. Document in main CLAUDE.md (15 min)

**Total time**: ~3 hours for Phase 1

---

**This is a proposal from Experimenter. Other personas should review:**

- **Skeptic**: What could go wrong? What edge cases?
- **Maintainer**: Is this maintainable? Too complex?
- **Auditor**: Security concerns? Risk assessment?
- **Architect**: Fit with overall architecture?
- **Optimizer**: Performance impact?

— **Experimenter**
