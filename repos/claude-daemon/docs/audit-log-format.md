# State API Audit Log Format

**Version**: 1.0
**Date**: 2025-11-04
**Status**: PRODUCTION
**Security Review**: docs/security-review-state-api-20251104.md

---

## Overview

The State API audit log tracks all state modifications for security incident investigation, debugging, and accountability. All mutating operations are logged before execution.

**Log file**: `~/.claude/daemon/logs/state-audit.jsonl`
**Format**: JSON Lines (one JSON object per line)
**Rotation**: Automatic (monthly or when >10MB)

---

## Log Entry Format

Each log entry is a single-line JSON object with the following fields:

```json
{
  "timestamp": "2025-11-04T18:10:59Z",
  "operation": "persona_switch",
  "details": "from=experimenter to=skeptic reason=testing",
  "caller": "direct",
  "pid": 12345,
  "user": "opc"
}
```

### Field Descriptions

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `timestamp` | string | ISO 8601 UTC timestamp | `"2025-11-04T18:10:59Z"` |
| `operation` | string | Type of operation | `"persona_switch"`, `"emotional_update"` |
| `details` | string | Operation-specific details | `"from=X to=Y reason=Z"` |
| `caller` | string | Script that invoked the operation | `"daemon.sh"`, `"direct"` |
| `pid` | number | Process ID | `12345` |
| `user` | string | System user | `"opc"` |

---

## Operation Types

### `persona_switch`

Logged by: `state_become()`

**Details format**: `from=<old_persona> to=<new_persona> reason=<reason>`

**Example**:
```json
{
  "timestamp": "2025-11-04T18:10:59Z",
  "operation": "persona_switch",
  "details": "from=experimenter to=skeptic reason=testing-new-feature",
  "caller": "claude-daemon-switch-persona.sh",
  "pid": 12345,
  "user": "opc"
}
```

### `emotional_update`

Logged by: `state_feel_frustrated()`, `state_feel_successful()`, `state_feel_failed()`

**Details format**: Description of emotional state changes

**Examples**:
```json
{"operation": "emotional_update", "details": "frustration +2"}
{"operation": "emotional_update", "details": "success_streak +1, failure_streak=0, frustration=0"}
{"operation": "emotional_update", "details": "failure_streak +1, success_streak=0, frustration +1"}
```

---

## Caller Identification

The `caller` field identifies which script invoked the State API:

- **Script name**: When called from another script (e.g., `"daemon.sh"`, `"claude-daemon-switch-persona.sh"`)
- **`"direct"`**: When called interactively or from command line

**Implementation**: Extracted from `${BASH_SOURCE[1]}` (parent script in call stack)

---

## Log Rotation

**Automatic rotation triggers**:
- File size exceeds 10MB
- File age exceeds 30 days

**Rotation process**:
1. Current log moved to `logs/archives/state-audit-YYYY-MM-YYYYMMDD-HHMMSS.jsonl`
2. Archived log compressed with gzip
3. New empty log created

**Manual rotation**:
```bash
source lib/state-audit.sh
state_audit_rotate
```

**Archive location**: `~/.claude/daemon/logs/archives/`

---

## Query Functions

### Show Recent Entries

```bash
source lib/state-audit.sh
state_audit_show_recent [count]
```

**Default count**: 20

**Example**:
```bash
$ state_audit_show_recent 5
2025-11-04T18:10:51Z  persona_switch    direct  from=experimenter to=experimenter reason=testing
2025-11-04T18:10:59Z  persona_switch    direct  from=experimenter to=skeptic reason=testing-audit
2025-11-04T18:10:59Z  emotional_update  direct  success_streak +1, failure_streak=0, frustration=0
2025-11-04T18:10:59Z  emotional_update  direct  frustration +2
2025-11-04T18:10:59Z  persona_switch    direct  from=skeptic to=architect reason=more-testing
```

### Show Persona Switches Only

```bash
state_audit_show_persona_switches
```

**Example**:
```bash
$ state_audit_show_persona_switches
2025-11-04T18:10:51Z  from=experimenter to=experimenter reason=testing
2025-11-04T18:10:59Z  from=experimenter to=skeptic reason=testing-audit
2025-11-04T18:10:59Z  from=skeptic to=architect reason=more-testing
```

### Show Entries by Caller

```bash
state_audit_show_by_caller <script-name>
```

**Example**:
```bash
$ state_audit_show_by_caller daemon.sh
2025-11-04T15:23:45Z  persona_switch  from=architect to=experimenter reason=automatic
2025-11-04T16:45:12Z  persona_switch  from=experimenter to=skeptic reason=validation
```

---

## Security Considerations

### What is Logged

✅ **Logged**:
- All persona switches (who, when, why, from where)
- All emotional state updates (type, magnitude)
- Process ID and user for each change
- Exact timestamp (UTC) of each operation

### What is NOT Logged

❌ **Not logged**:
- Read-only operations (`state_who()`, `state_feel()`, `state_stats()`)
- API function arguments beyond what's in details
- State file contents (too verbose, privacy)
- Failed validation attempts (before state is accessed)

### Privacy

- Audit logs contain **system state changes only** (persona names, emotional state)
- No user data, secrets, or sensitive information logged
- File permissions: 644 (readable by user, not writable by others)

### Retention

- **Production logs**: Retained for 12 months in archives
- **Development logs**: Can be cleared anytime (`rm logs/state-audit.jsonl`)
- **Archived logs**: Compressed, stored in `logs/archives/`

---

## Failure Handling

**If audit logging fails**:
1. Warning printed to stderr: `WARNING: Failed to write audit log entry`
2. State operation continues (doesn't fail)
3. Rationale: Audit logging is important but shouldn't block system operation

**Fallback behavior**:
- If `jq` fails, simple JSON string written to log
- If log directory doesn't exist, created automatically
- If disk full, warning printed but operation continues

---

## Usage Examples

### Example 1: Investigating Unexpected Persona Switch

**Scenario**: System switched to `auditor` unexpectedly

```bash
$ grep "to=auditor" logs/state-audit.jsonl | tail -1 | jq .
{
  "timestamp": "2025-11-04T15:23:45Z",
  "operation": "persona_switch",
  "details": "from=experimenter to=auditor reason=security-trigger",
  "caller": "daemon.sh",
  "pid": 12345,
  "user": "opc"
}
```

**Result**: Switched by daemon.sh due to security trigger at 15:23:45

### Example 2: Debugging State Corruption

**Scenario**: Emotional state stuck at high frustration

```bash
$ grep "emotional_update" logs/state-audit.jsonl | tail -10
```

**Shows last 10 emotional updates** to trace when frustration increased

### Example 3: Accountability Audit

**Scenario**: Who made changes in the last hour?

```bash
$ jq -r 'select(.timestamp > "2025-11-04T17:00:00Z") |
  [.timestamp, .operation, .caller] | @tsv' logs/state-audit.jsonl
```

---

## Integration with State API

### How It Works

1. **Mutating function called** (e.g., `state_become experimenter "testing"`)
2. **Audit entry created** with timestamp, operation details, caller
3. **Audit log written** to `logs/state-audit.jsonl` (append mode)
4. **State operation proceeds** (validation, transaction, update)

**Timing**: Audit log written **BEFORE** state change (captures intent even if change fails)

### Code Example

```bash
# In lib/state-api.sh
state_become() {
    local persona="$1"
    local reason="${2:-manual}"

    # ... validation ...

    # Audit log BEFORE change
    local from_persona=$(state_who)
    local caller="${BASH_SOURCE[1]:-direct}"
    caller="${caller##*/}"
    if command -v state_audit >/dev/null 2>&1; then
        state_audit "persona_switch" \
            "from=$from_persona to=$persona reason=$reason" \
            "$caller" || true  # Don't fail if audit logging fails
    fi

    # ... state update ...
}
```

---

## Future Enhancements

**Potential additions** (not yet implemented):

1. **Structured details field**: JSON object instead of string
2. **Before/after snapshots**: Log old and new values
3. **Operation duration**: Track how long state operations take
4. **Correlation IDs**: Link related operations (e.g., persona switch → multiple emotional updates)
5. **Remote logging**: Send audit logs to central server
6. **Anomaly detection**: Alert on unusual patterns (e.g., rapid persona switches)

---

## Related Documentation

- **State API**: lib/state-api.sh
- **Audit System**: lib/state-audit.sh
- **Security Review**: docs/security-review-state-api-20251104.md
- **ADR-001**: docs/ADR-001-state-api-adoption.md

---

**Document owner**: Experimenter
**Reviewed by**: Auditor (pending)
**Status**: Production-ready (Phase 2 requirement met)
**Last updated**: 2025-11-04
