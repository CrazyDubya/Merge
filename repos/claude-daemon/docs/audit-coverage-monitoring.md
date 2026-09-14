# Audit Coverage Monitoring

**Created**: 2025-11-04 by Maintainer
**Purpose**: Monitor audit logging coverage to detect integration gaps
**Context**: Built in response to daemon.sh audit bypass incident (95% gap)

---

## Overview

The audit coverage monitoring system tracks how many persona switches are actually being logged versus how many activations are occurring. This helps detect integration gaps where code bypasses the State API audit trail.

**Key metrics**:
- **Total activations**: Sum of all persona `total_activations` from state.json
- **Audit entries**: Count of `persona_switch` operations in audit log
- **Coverage**: Percentage of activations that are logged

**Coverage thresholds**:
- ✅ **Healthy**: ≥90% coverage (production-ready)
- ⚠️  **Warning**: 50-89% coverage (needs attention)
- 🚨 **Critical**: <50% coverage (urgent action required)

---

## The Script

**Location**: `scripts/audit-coverage-monitor.sh`

**What it does**:
1. Reads `total_activations` from state.json (actual activations)
2. Counts `persona_switch` entries in audit.jsonl (logged activations)
3. Calculates coverage percentage
4. Provides status assessment and recommendations

**Exit codes**:
- `0` - Coverage ≥90% (healthy)
- `1` - Coverage <90% (warning)
- `2` - Coverage <50% (critical)
- `3` - Error reading files

---

## Usage

### One-time Check

```bash
./scripts/audit-coverage-monitor.sh
```

**Output**:
```
======================================================================
  Audit Coverage Report
======================================================================

Total persona activations: 192
Audit log entries:         6
Coverage:                  3%

[CRITICAL] Status: CRITICAL (<50%)

✗ Audit logging severely degraded
✗ Most persona switches are NOT logged
✗ URGENT: Review State API integration
```

### Detailed Breakdown

```bash
./scripts/audit-coverage-monitor.sh --detailed
```

**Shows per-persona activation counts** to help identify patterns.

### Continuous Monitoring

```bash
./scripts/audit-coverage-monitor.sh --watch
```

**Output** (updates every 5 minutes):
```
[INFO] Starting continuous monitoring (interval: 300s)
[INFO] Press Ctrl+C to stop

[2025-11-04 22:30:15] Activations: 192 | Audit:   6 | Coverage:   3% | Status: CRITICAL
[2025-11-04 22:35:15] Activations: 195 | Audit:   8 | Coverage:   4% | Status: CRITICAL
[2025-11-04 22:40:15] Activations: 198 | Audit:  10 | Coverage:   5% | Status: CRITICAL
```

**Custom interval**:
```bash
./scripts/audit-coverage-monitor.sh --watch --interval 60  # Check every minute
```

### Alert Mode (for Automation)

```bash
./scripts/audit-coverage-monitor.sh --alert
```

**Use in cron**:
```bash
# Check audit coverage every hour, alert if <90%
0 * * * * /home/opc/.claude/daemon/scripts/audit-coverage-monitor.sh --alert || mail -s "Audit Coverage Alert" ops@example.com
```

**Use in CI/CD**:
```bash
# Fail build if audit coverage is low
./scripts/audit-coverage-monitor.sh --alert || exit 1
```

---

## When to Use

### During Development

**Before approving security phases**:
```bash
# Run 24h production validation
./scripts/audit-coverage-monitor.sh --watch --interval 300 > coverage-log.txt &

# After 24 hours, check final coverage
./scripts/audit-coverage-monitor.sh --detailed

# Coverage should be >90%
```

### During Migration

**When migrating scripts to State API**:
```bash
# Check before migration
./scripts/audit-coverage-monitor.sh  # Baseline

# Perform migration
# ...

# Check after migration
./scripts/audit-coverage-monitor.sh  # Should increase

# Watch for 1 hour to confirm
./scripts/audit-coverage-monitor.sh --watch --interval 60
```

### In Production

**Continuous monitoring** (add to cron):
```bash
# Check every 6 hours, alert if coverage drops
0 */6 * * * /home/opc/.claude/daemon/scripts/audit-coverage-monitor.sh --alert
```

**Daily reports**:
```bash
# Send daily coverage report
0 9 * * * /home/opc/.claude/daemon/scripts/audit-coverage-monitor.sh --detailed > /tmp/audit-coverage-$(date +\%Y\%m\%d).txt
```

---

## Understanding Results

### Healthy Coverage (≥90%)

```
Total persona activations: 200
Audit log entries:         185
Coverage:                  92%

[OK] Status: HEALTHY (>=90%)

✓ Audit logging is working correctly
✓ Coverage meets production requirements
```

**What this means**:
- State API is being used consistently
- Most persona switches are logged
- Audit trail is reliable for incident response

**Note**: 100% coverage is not required. Some switches during testing or development may not be logged.

### Warning Coverage (50-89%)

```
Total persona activations: 200
Audit log entries:         120
Coverage:                  60%

[WARN] Status: WARNING (50-90%)

⚠ Audit coverage below healthy threshold
⚠ Some persona switches may not be logged
⚠ Review daemon.sh integration
```

**What this means**:
- Partial State API adoption
- Some code paths bypass audit trail
- Not production-ready for security-critical operations

**Action**: Review which scripts/code paths are logging vs not logging.

### Critical Coverage (<50%)

```
Total persona activations: 192
Audit log entries:         6
Coverage:                  3%

[CRITICAL] Status: CRITICAL (<50%)

✗ Audit logging severely degraded
✗ Most persona switches are NOT logged
✗ URGENT: Review State API integration
```

**What this means**:
- Major integration gap (like daemon.sh bypass)
- Audit trail is mostly non-functional
- Cannot rely on audit logs for accountability

**Action**: URGENT - identify and fix bypass immediately.

---

## Troubleshooting

### Low Coverage After Fresh Install

**Symptom**: 0% or very low coverage on new system

**Cause**: Audit logging not yet configured

**Fix**:
```bash
# Verify State API exists
ls -l lib/state-api.sh lib/state-audit.sh

# Check if daemon sources State API
grep 'state-api.sh' daemon.sh

# Test manual switch (should log)
./claude-daemon-switch-persona.sh experimenter testing
tail -1 logs/state-audit.jsonl
```

### Coverage Drops Suddenly

**Symptom**: Coverage was 90%+, now 50%

**Cause**: Code change bypassed State API

**Fix**:
```bash
# Check recent commits
git log --oneline -10

# Check which scripts were modified
git diff HEAD~5 --name-only | grep '.sh$'

# Review those scripts for State API usage
```

### Coverage Stuck at Partial Level

**Symptom**: Coverage at 30-40%, not improving

**Cause**: Major component (like daemon.sh) not integrated

**Fix**:
```bash
# Find scripts that don't use State API
grep -L 'state_become\|state_who' *.sh

# Prioritize by usage frequency
grep -L 'state_become' daemon.sh  # Critical!
```

### False Low Coverage

**Symptom**: Coverage seems low but code looks correct

**Cause**: Old activations from before audit logging implemented

**Fix**:
```bash
# Check when audit logging started
head -1 logs/state-audit.jsonl  # First logged entry

# Compare with state.json activation counts
# Activations from before audit logging will inflate total

# Consider resetting activation counts if desired
# (CAUTION: This loses historical data)
```

---

## Integration with Other Tools

### With Test Suite

```bash
# In experiments/test-state-audit.sh
echo "Checking audit coverage..."
./scripts/audit-coverage-monitor.sh --alert || {
    echo "ERROR: Audit coverage below threshold"
    exit 1
}
```

### With Deployment Process

```bash
# Before deploying State API changes
COVERAGE_BEFORE=$(./scripts/audit-coverage-monitor.sh --alert 2>&1 | grep -oP '\d+%' | head -1)

# Deploy changes
# ...

# After deployment
COVERAGE_AFTER=$(./scripts/audit-coverage-monitor.sh --alert 2>&1 | grep -oP '\d+%' | head -1)

echo "Coverage change: $COVERAGE_BEFORE → $COVERAGE_AFTER"
```

### With Monitoring Systems

**Prometheus metrics** (example):
```bash
# Export coverage as metric
COVERAGE=$(./scripts/audit-coverage-monitor.sh --alert 2>&1 | grep -oP '(\d+)%' | grep -oP '\d+')
echo "audit_coverage_percent $COVERAGE" > /var/lib/prometheus/node-exporter/audit-coverage.prom
```

**Nagios check** (example):
```bash
# Use as Nagios check
./scripts/audit-coverage-monitor.sh --alert
# Exit 0 = OK, Exit 1 = WARNING, Exit 2 = CRITICAL
```

---

## Recommendations

### For Phase Approvals

**Before approving any security phase**:

1. ✅ Unit tests pass
2. ✅ Integration tests pass
3. ✅ **Run audit coverage check** ← ADD THIS
4. ✅ Coverage ≥90% confirmed
5. ✅ 24h production monitoring

**Example checklist**:
```bash
# Phase approval validation
echo "1. Running tests..."
./experiments/test-state-api.sh

echo "2. Checking audit coverage..."
./scripts/audit-coverage-monitor.sh --detailed

# Require >90% coverage
COVERAGE=$(./scripts/audit-coverage-monitor.sh --alert 2>&1 | grep -oP '(\d+)%' | head -1 | grep -oP '\d+')
if [ "$COVERAGE" -lt 90 ]; then
    echo "ERROR: Coverage $COVERAGE% < 90%"
    echo "Phase approval BLOCKED"
    exit 1
fi

echo "3. Starting 24h monitoring..."
./scripts/audit-coverage-monitor.sh --watch --interval 3600 &
WATCH_PID=$!
echo "Monitor PID: $WATCH_PID (kill to stop)"
```

### For Incident Prevention

**Daily monitoring** (cron):
```bash
# /etc/cron.d/audit-coverage-check
0 */6 * * * opc /home/opc/.claude/daemon/scripts/audit-coverage-monitor.sh --alert || logger -t audit-coverage "Coverage below 90%"
```

**Alert on degradation**:
```bash
# Store baseline
./scripts/audit-coverage-monitor.sh --alert > /tmp/baseline-coverage.txt

# Check periodically, alert if drops >10%
# (implement as monitoring script)
```

### For Debugging

**When investigating incident**:
```bash
# Check coverage at time of incident
./scripts/audit-coverage-monitor.sh --detailed

# Review audit log for incident timeframe
grep "2025-11-04T20:" logs/state-audit.jsonl

# Compare activations vs logged switches
# Low coverage = missing audit trail
```

---

## Historical Context

**Why this exists**: On 2025-11-04, Auditor discovered daemon.sh (core orchestrator) bypassed State API completely, resulting in 95% audit gap (9/191 activations logged).

**Incident**: docs/security-incident-audit-bypass-20251104.md

**Root cause**: Phase 2 approved without production validation. Unit tests passed but integration was never verified.

**Lesson**: Production validation is mandatory for security controls. This script automates the check that would have caught the gap immediately.

**Prevention**: This monitoring script is now part of the validation checklist for all security phase approvals.

---

## Future Enhancements

**Potential improvements**:

1. **Trend tracking**: Store coverage over time, alert on sudden drops
2. **Per-persona coverage**: Track which personas are/aren't being logged
3. **Hourly breakdown**: Show coverage by hour (circadian patterns)
4. **Automatic remediation**: Detect bypass, notify responsible persona
5. **Coverage dashboard**: Web UI showing real-time coverage
6. **Integration with CI**: Automatic coverage checks in PR pipeline

**Contributions welcome** - see tasks/queue.md for enhancement tasks.

---

## See Also

- `docs/security-incident-audit-bypass-20251104.md` - Incident that motivated this tool
- `docs/audit-log-format.md` - Audit log specification
- `docs/ADR-001-state-api-adoption.md` - State API adoption strategy
- `experiments/test-state-audit.sh` - Audit trail test suite
- `lib/state-api.sh` - State API implementation
- `lib/state-audit.sh` - Audit logging functions

---

**Maintained by**: Maintainer persona
**Last updated**: 2025-11-04
**Status**: Production-ready

*"What gets measured gets managed. Audit coverage monitoring prevents integration gaps from hiding in production."*
