# Skeptic Analysis: Unbounded Log Growth Risk

**Date**: 2025-11-09
**Analyzer**: Skeptic
**Severity**: MEDIUM (operational risk, not security)
**Status**: IDENTIFIED

---

## Executive Summary

While Maintainer reported repository in "EXCELLENT" health and Experimenter reported "99% ADR-002 compliance", **critical log files have unbounded growth** with no rotation mechanism.

**Impact**: Disk exhaustion risk, performance degradation, operational issues.

## Evidence

### Current State (2025-11-09)

| File | Size | Lines | Time Span | Growth Rate |
|------|------|-------|-----------|-------------|
| logs/activity.log | 100 MB | 1,581,034 | ~14 days (Oct 26 - Nov 9) | ~7.1 MB/day |
| logs/state-audit.jsonl | 19 MB | 104,157 | 5 days (Nov 4 - Nov 9) | ~3.8 MB/day |
| memory/emergence-log.md | 281 KB | - | - | Has rotation (scripts/rotate-emergence-log.sh) |

### Projection

**activity.log** at 7.1 MB/day:
- 1 month: 213 MB
- 6 months: 1.3 GB
- 1 year: 2.6 GB

**state-audit.jsonl** at 3.8 MB/day:
- 1 month: 114 MB
- 6 months: 684 MB
- 1 year: 1.4 GB

**Combined**: ~4 GB/year in just these two logs.

## Questions Asked (Skeptic Method)

### Q1: Is this being monitored?

**Answer**: NO

- Maintainer's metrics dashboard (scripts/maintenance-metrics.sh) tracks:
  - ✓ Days since last commit
  - ✓ Uncommitted changes
  - ✓ Untracked files
  - ✓ Commit frequency
  - ✓ Repository size (git + working tree total)
  - ✗ **Individual log file sizes**
  - ✗ **Log growth rates**
  - ✗ **Disk space trends**

**Finding**: Metrics track git health but not operational health.

### Q2: Is rotation implemented?

**Answer**: PARTIAL

- ✓ emergence-log.md: HAS rotation (scripts/rotate-emergence-log.sh, 100KB threshold)
- ✗ activity.log: NO rotation
- ✗ state-audit.jsonl: NO rotation
- ✗ persona-voice.log: NO rotation (848 KB currently)
- ✗ watchdog.log: NO rotation (408 KB currently)

**Evidence**:
```bash
$ grep -r "activity.log.*rotation\|rotate.*activity" . --include="*.sh"
# (no results)
```

### Q3: Why wasn't this caught?

**Possible reasons**:
1. Maintainer focused on git health, not disk health
2. Experimenter focused on append safety (race conditions), not size management
3. Log rotation was implemented for emergence-log but not generalized
4. Recent work focused on security (ADR-002) rather than operational concerns

**Root cause**: Incomplete monitoring coverage.

### Q4: What are the risks?

**Operational Risks**:
1. **Disk exhaustion**: Logs could fill disk
   - activity.log: 2.6 GB/year
   - state-audit.jsonl: 1.4 GB/year
   - Combined with other growth: potential disk issues

2. **Performance degradation**:
   - Large log files slow down grep/tail operations
   - state-audit.jsonl used for analysis (slower queries)
   - activity.log read during debugging (slow load)

3. **Backup costs**:
   - Larger backup sizes
   - Slower backup/restore operations

4. **Analysis difficulty**:
   - 1.5M lines harder to analyze than rotated chunks
   - No time-based archival (harder to find "October issues")

**Severity Assessment**:
- **Likelihood**: HIGH (growth is continuous and unbounded)
- **Impact**: MEDIUM (disk space, not data loss)
- **Overall Risk**: MEDIUM

### Q5: What's the evidence-based recommendation?

**Recommendation**: Implement log rotation for activity.log and state-audit.jsonl

**Evidence supporting recommendation**:
1. Existing rotation for emergence-log works well (see tasks/completed/2025-10-30.md)
2. Growth rate is measurable and concerning (10.9 MB/day combined)
3. Audit trail integrity maintained with rotation (files archived, not deleted)
4. Compression effective (67% for emergence-log, likely similar for others)

**Proposed thresholds** (based on emergence-log precedent):

| File | Current Size | Proposed Threshold | Rationale |
|------|--------------|-------------------|-----------|
| activity.log | 100 MB | 50 MB | ~7 days of history at current rate |
| state-audit.jsonl | 19 MB | 25 MB | ~1 week of audit trail |
| persona-voice.log | 848 KB | 10 MB | Lower priority, less critical |
| watchdog.log | 408 KB | 10 MB | Alerts should be time-bounded |

## Assumptions Identified

Maintainer's work assumed:
- ✓ Git health = repository health
- ✗ **Missed**: Operational health (disk usage, log growth)

Experimenter's work assumed:
- ✓ Append safety (ADR-002) protects data integrity
- ✗ **Missed**: Append frequency affects disk usage

**Challenge to assumption**: "Repository in EXCELLENT health" is only true for git health, not operational health.

## Edge Cases Considered

**Edge case 1**: What if rotation happens during high activity (thrashing)?
- **Risk**: Rotation script competes for disk I/O during crisis
- **Mitigation**: Use nice/ionice for rotation, or defer during high load

**Edge case 2**: What if rotation corrupts active log?
- **Risk**: Rotation race condition during write
- **Mitigation**: Use atomic operations (as emergence-log rotation does)

**Edge case 3**: What if we need audit trail older than rotation window?
- **Risk**: Compliance/debugging requires full history
- **Mitigation**: Archives compressed and retained (like emergence-log)

**Edge case 4**: What if growth rate spikes (another thrashing bug)?
- **Risk**: Thresholds too high, disk fills faster
- **Mitigation**: Monitor disk space separately, not just log sizes

## Validation Questions

Before implementing rotation:

1. **Q**: What's the disk capacity?
   **Why**: Need to know actual risk of exhaustion

2. **Q**: Are logs used for compliance/security?
   **Why**: If yes, retention policy needed before rotation

3. **Q**: What's the actual query pattern for state-audit.jsonl?
   **Why**: If recent data only, rotation is safe; if historical analysis, need different approach

4. **Q**: Is activity.log actually used, or just debug noise?
   **Why**: If unused, could just disable verbose logging instead of rotating

5. **Q**: What's the performance impact of 100MB activity.log?
   **Why**: Quantify the "performance degradation" claim

## Proposed Next Steps

1. **Immediate** (0 hours):
   - Document this finding
   - Add to task queue

2. **Short-term** (1-2 hours):
   - Check disk capacity: `df -h`
   - Validate state-audit.jsonl usage patterns
   - Review activity.log necessity (could reduce logging instead)

3. **Medium-term** (2-4 hours):
   - Implement rotation for activity.log (adapt from rotate-emergence-log.sh)
   - Implement rotation for state-audit.jsonl
   - Update Maintainer metrics to include log sizes

4. **Long-term** (ongoing):
   - Monitor disk usage trends
   - Adjust thresholds based on actual usage
   - Consider log aggregation/analysis tools

## Comparison to Previous Work

**Optimizer's emergence-log rotation** (2025-10-30):
- Target: <100KB active log
- Method: Size-based rotation, gzip compression, archival
- Security: Validated by Auditor (9/10)
- Result: 67% compression, atomic operations

**Why wasn't this generalized?**
- Task was specific to emergence-log
- No follow-up task to generalize pattern
- Focus shifted to other priorities (ADR-002, token efficiency)

**Lesson**: One-off solutions create gaps when not generalized.

## Conclusion

**Finding**: Unbounded log growth in activity.log and state-audit.jsonl poses operational risk.

**Severity**: MEDIUM (operational, not security)

**Confidence**: HIGH (evidence-based measurement)

**Recommendation**: Implement log rotation similar to emergence-log pattern.

**Questions for Human/Team**:
1. What's the actual disk capacity and acceptable usage?
2. Are these logs used for compliance/long-term analysis?
3. Should we reduce logging verbosity instead of/in addition to rotation?
4. What's the priority vs. other tasks?

---

**Skeptic's Note**: When Maintainer reports "EXCELLENT" and Experimenter reports "99% compliant", that's when I look for what's NOT being measured. This is what I found.

Evidence > assumptions. Always.
