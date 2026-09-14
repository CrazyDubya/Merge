# Security Requirements: Memory Architecture Design

**Reviewer**: Auditor
**Date**: 2025-11-09T17:30:00Z
**Status**: REQUIREMENTS DEFINITION (Pre-Design)
**Target**: Architect's memory architecture design (24h deadline: Nov 10 02:55 GMT)
**Context**: Human authorized memory architecture design, Architect leads, 200-line constraint

---

## Executive Summary

This document defines **MANDATORY** security requirements for the memory architecture design. Architect must address all CRITICAL and HIGH priority requirements in the initial design. MEDIUM priority requirements must be documented as future work.

**Purpose**: Ensure security is designed IN, not bolted ON after implementation.

**Scope**: Memory consolidation, retention policies, compression, LLM integration, access control.

---

## Security Context

### Current State (From Previous Review)

**Phase 2 validation (2025-11-07)**: Prototypes approved at 8/10 security rating
- Timeline archival: flock protection, atomic operations, integrity validation ✅
- Emergence summarization: Logic validated (not yet tested on old data) ✅
- Security mechanisms: 5/5 verified working ✅

**What changed**: Moving from prototype validation to **PRODUCTION ARCHITECTURE DESIGN**

**New security concerns**:
1. LLM integration for semantic consolidation (external service risk)
2. Data retention policies (compliance, audit requirements)
3. Long-term archive integrity (tampering detection, corruption recovery)
4. Access control (who can read/modify/delete archived memory)

---

## Requirement Classification

- 🔴 **CRITICAL**: Must be in design, blocks implementation if missing
- 🟡 **HIGH**: Should be in design, can be added during implementation
- 🟢 **MEDIUM**: Document as future work, implement in Phase 4+

---

## R1: Data Retention Security (🔴 CRITICAL)

### R1.1: Data Classification

**Requirement**: Memory architecture MUST classify all data by sensitivity and define retention requirements.

**Classification framework**:

| Tier | Sensitivity | Retention | Rationale |
|------|-------------|-----------|-----------|
| **Critical** | HIGH | NEVER DELETE | Emergence insights, architectural decisions, security incidents |
| **Important** | MEDIUM | 90 days minimum | Task history, persona reflections, system events |
| **Operational** | LOW | 30 days sufficient | Persona switches, routine task starts, low-value events |
| **Transient** | MINIMAL | 7 days sufficient | Debug logs, temporary state, routine operational noise |

**Security principle**: "Preserve evidence, prune noise"

**Design requirement**: Architect must specify which memory files belong to which tier and document retention requirements for each.

### R1.2: Deletion Auditability

**Requirement**: All data deletion MUST be auditable.

**Implementation**:
- Log what was deleted (date range, entry count, reason)
- Preserve deletion audit log indefinitely (small, high-value)
- Enable recovery from backups if deletion was incorrect

**Security principle**: "Trust but verify" - deletion is intentional and traceable

**Design requirement**: Consolidation algorithm must generate deletion audit log.

### R1.3: Retention Compliance

**Requirement**: Retention policies MUST comply with security incident investigation needs.

**Minimum retention**:
- **Security events**: 90 days (industry standard for incident investigation)
- **Audit trail**: 90 days minimum (sufficient for retrospective analysis)
- **System changes**: 180 days (architectural decision context)

**Design requirement**: Architect must ensure timeline archives preserve security-relevant events for 90+ days.

---

## R2: LLM Integration Security (🔴 CRITICAL)

### R2.1: Data Exfiltration Risk

**Threat**: Sending internal system data to external LLM services creates exfiltration risk.

**Requirement**: LLM consolidation MUST NOT send sensitive data externally unless explicitly approved.

**Sensitive data classes**:
1. **Credentials**: API keys, tokens, passwords (even if mentioned in logs)
2. **System internals**: File paths, hostnames, IP addresses, configuration details
3. **Human identifiable info**: Username, email, personal context
4. **Security incidents**: Vulnerability details, attack patterns, exploit code

**Mitigation options** (Architect must choose):
1. **Option A (Safest)**: Use bash/jq/gzip only, NO LLM consolidation
2. **Option B (Balanced)**: LLM consolidation ONLY for pre-sanitized data
3. **Option C (Riskiest)**: LLM consolidation with content filtering (requires validation)

**Security principle**: "Default to local processing, escalate to LLM only when safe"

**Design requirement**: Architect must specify when LLM is used vs bash/jq and document data sanitization approach.

### R2.2: LLM Output Validation

**Requirement**: If LLM is used for summarization, output MUST be validated before replacing source data.

**Validation checks**:
1. **Completeness**: Key facts from source present in summary
2. **Accuracy**: No hallucinations or invented events
3. **Safety**: No injection of malicious content

**Failure handling**: If validation fails, preserve original data, do NOT replace with LLM output.

**Design requirement**: Consolidation algorithm must include LLM output validation step.

### R2.3: LLM Cost Controls

**Requirement**: LLM usage MUST have cost controls to prevent runaway spending.

**Controls**:
- Maximum token limit per consolidation run
- Maximum cost budget per month
- Graceful degradation if budget exceeded (fall back to bash/jq compression)

**Security principle**: Denial-of-wallet is a security issue

**Design requirement**: Architect must specify LLM usage limits and fallback behavior.

---

## R3: Archive Integrity (🟡 HIGH)

### R3.1: Tampering Detection

**Requirement**: Archived data SHOULD have tampering detection.

**Implementation options**:
1. **Basic**: SHA256 checksum file alongside archive
2. **Better**: Signed checksums (GPG signature)
3. **Best**: Immutable storage (append-only, no delete/modify)

**Design requirement**: Architect should specify integrity protection mechanism for archives.

### R3.2: Corruption Recovery

**Requirement**: Corrupted archives MUST NOT block system operation.

**Recovery strategy**:
1. Detect corruption during read (gunzip -t, checksum validation)
2. Log corruption event with alert
3. Fall back to backup or skip corrupted data
4. Continue system operation (degraded but functional)

**Security principle**: "Fail open with alerting" - corruption doesn't cause denial of service

**Design requirement**: Consolidation algorithm must handle corrupted archives gracefully.

### R3.3: Backup Retention

**Requirement**: Backups MUST be retained before destructive operations.

**Current implementation**: ✅ Timeline archival creates backups (verified in Phase 2)

**Design requirement**: Architect must ensure all destructive operations (deletion, consolidation) create backups first.

---

## R4: Access Control (🟡 HIGH)

### R4.1: Principle of Least Privilege

**Requirement**: Archive access SHOULD follow least privilege.

**Access matrix**:

| Operation | Persona | Justification |
|-----------|---------|---------------|
| **Read archives** | All personas | Retrospective analysis, debugging |
| **Create archives** | Automation only | Consolidation scripts run via cron |
| **Modify archives** | NONE | Archives are immutable (append-only) |
| **Delete archives** | Auditor + Human | Manual intervention only (security review required) |

**Design requirement**: Architect should document access control model for archives.

### R4.2: Archive Location Security

**Requirement**: Archives SHOULD be stored in protected location.

**Recommendation**: `memory/archives/` with appropriate file permissions
- Archives: 0644 (read-only for users, writable by owner)
- Directory: 0755 (prevent unauthorized file creation)

**Design requirement**: Architect should specify archive location and permissions.

---

## R5: Consolidation Algorithm Security (🔴 CRITICAL)

### R5.1: Atomic Operations

**Requirement**: Consolidation MUST be atomic (all-or-nothing).

**Current implementation**: ✅ Verified in Phase 2 validation

**Design requirement**: Architect must ensure consolidation algorithm preserves atomicity.

### R5.2: Concurrent Execution Protection

**Requirement**: Only one consolidation run MUST execute at a time.

**Current implementation**: ✅ flock protection verified in Phase 2

**Design requirement**: Architect must specify locking strategy for nightly consolidation.

### R5.3: Failure Recovery

**Requirement**: Consolidation failures MUST NOT corrupt production data.

**Failure scenarios**:
1. **Disk full**: Detect early, abort before modifying production files
2. **Process killed**: Rollback to pre-consolidation state from backup
3. **Validation failure**: Skip consolidation, alert operator, preserve current state

**Security principle**: "Safe failure" - corruption is worse than skipped consolidation

**Design requirement**: Architect must specify failure handling for each scenario.

---

## R6: Audit Trail Requirements (🟡 HIGH)

### R6.1: Consolidation Audit Log

**Requirement**: All consolidation operations SHOULD be audited.

**Audit log contents**:
- Timestamp of consolidation
- Data range processed (e.g., "Nov 1-7")
- Actions taken (archived, deleted, summarized)
- Entry counts (before/after)
- Compression ratios
- Errors/warnings
- Duration

**Design requirement**: Architect should specify audit log format and location.

### R6.2: Security Event Preservation

**Requirement**: Security-relevant events MUST be preserved even after consolidation.

**Security events** (from triggers/security.json):
- Deployment requests
- Security vulnerability triggers
- Authentication/authorization events
- Configuration changes
- Incident responses

**Design requirement**: Architect must ensure security events are marked as "NEVER DELETE" in tier classification.

---

## R7: Performance vs Security Trade-offs (🟢 MEDIUM)

### R7.1: Compression Security

**Consideration**: gzip compression is fast but not authenticated.

**Options**:
1. **gzip** (current): Fast, good compression (80-87%), no authentication
2. **gzip + checksum**: Same performance, add SHA256 for integrity
3. **Authenticated compression**: zstd with checksum, slower but stronger guarantees

**Recommendation**: Start with gzip + SHA256 checksum (minimal overhead, strong integrity)

**Design requirement**: Document as future improvement in Phase 4+

### R7.2: Query Performance vs Security

**Trade-off**: Encryption would protect archives but slow queries.

**Analysis**:
- Archives contain operational data, not credentials (low sensitivity)
- System runs on trusted infrastructure (Oracle Cloud with SSH key auth)
- Query performance critical for debugging (8ms currently)

**Recommendation**: Encryption NOT required for initial design (can add later if threat model changes)

**Design requirement**: Document as optional Phase 4+ enhancement

---

## R8: Operational Security (🟡 HIGH)

### R8.1: Monitoring and Alerting

**Requirement**: Consolidation failures SHOULD trigger alerts.

**Alert triggers**:
- Consolidation script failure (exit code != 0)
- Integrity validation failure (corrupted archive)
- Disk space low (<10% free after consolidation)
- Compression ratio anomaly (<50% or >95%)
- Excessive duration (>10 minutes for routine consolidation)

**Design requirement**: Architect should specify monitoring approach.

### R8.2: Rollback Procedures

**Requirement**: Operators SHOULD have documented rollback procedures.

**Rollback scenarios**:
1. **Bad consolidation**: Restore from backup, re-run consolidation
2. **Corrupted archive**: Delete corrupted archive, restore from backup
3. **Accidental deletion**: Restore from backup (if within retention period)

**Design requirement**: Architect should document rollback procedures in design.

---

## Security Review Process

### Design Review (Before Implementation)

**Timing**: After Architect completes design (within 24h deadline)

**Auditor review checklist**:
1. ✅ All CRITICAL requirements addressed in design
2. ✅ HIGH requirements documented (implementation plan or future work)
3. ✅ LLM security model clearly defined
4. ✅ Data retention policies specified
5. ✅ Failure handling documented

**Approval criteria**: All CRITICAL requirements met, HIGH requirements acknowledged

### Implementation Review (Phase 3+)

**Timing**: Before production deployment

**Auditor validation**:
1. Code review (verify design requirements implemented)
2. Security mechanism testing (LLM sanitization, integrity checks, access control)
3. Failure scenario testing (disk full, process killed, validation failures)
4. Production readiness assessment

---

## Requirements Summary

### CRITICAL (Must be in design)

- R1.1: Data classification and retention tiers ✅
- R1.2: Deletion auditability ✅
- R2.1: LLM data exfiltration risk mitigation ✅
- R2.2: LLM output validation ✅
- R5.1: Atomic operations ✅
- R5.2: Concurrent execution protection ✅
- R5.3: Failure recovery ✅

### HIGH (Should be in design or documented as Phase 4)

- R1.3: Retention compliance (90-day minimum) ✅
- R2.3: LLM cost controls ✅
- R3.1: Archive tampering detection ✅
- R3.2: Corruption recovery ✅
- R4.1: Access control model ✅
- R6.1: Consolidation audit log ✅
- R6.2: Security event preservation ✅
- R8.1: Monitoring and alerting ✅
- R8.2: Rollback procedures ✅

### MEDIUM (Document as future work)

- R7.1: Compression authentication (checksum) ⏳
- R7.2: Archive encryption (optional) ⏳

**Total**: 7 CRITICAL + 9 HIGH + 2 MEDIUM = 18 security requirements

---

## Architect Guidance

### What I Need in the Design

1. **Tier definitions**: Classify all memory files (emergence-log, timeline, inbox, etc.) into Critical/Important/Operational/Transient tiers with retention policies.

2. **Consolidation algorithm**: Step-by-step process for nightly consolidation including:
   - What gets archived (7d threshold)
   - What gets compressed (30d threshold)
   - What gets summarized (LLM or bash/jq)
   - What gets deleted (90d threshold)
   - Failure handling for each step

3. **LLM security model**: When is LLM used? What data sanitization? What validation? What fallback?

4. **Audit trail**: What gets logged? Where? How long retained?

5. **Access control**: Who can read/modify/delete archives?

6. **Monitoring**: What alerts? When? Who receives them?

### Design Format (Recommendation)

Given 200-line constraint, prioritize:
- **Section 1 (40 lines)**: Tier definitions + retention table
- **Section 2 (60 lines)**: Consolidation algorithm (step-by-step)
- **Section 3 (40 lines)**: LLM security model
- **Section 4 (30 lines)**: Audit + monitoring
- **Section 5 (30 lines)**: Failure handling + rollback

**Focus**: Clarity over completeness. Address CRITICAL requirements first, HIGH requirements second, MEDIUM as "Future work: ..."

---

## Auditor Commitment

**Review SLA**: I will review Architect's design within 3 hours of submission (before 24h deadline).

**Approval process**:
- ✅ All CRITICAL requirements addressed → APPROVED
- ⚠️ CRITICAL gaps identified → CONDITIONAL (specify what's needed)
- ❌ Major security concerns → REJECTED (redesign required)

**Collaboration**: Available for questions during design. Proactive review preferred over reactive fixes.

---

**Security Requirements**: COMPLETE
**Status**: Ready for Architect's design work
**Next**: Architect drafts memory architecture design addressing these requirements

---

**Generated**: 2025-11-09T17:30:00Z
**Auditor**: Security review prepared for memory architecture design
**Distribution**: Architect (primary), Experimenter (implementation support), Human (visibility)
