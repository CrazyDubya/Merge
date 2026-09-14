#!/bin/bash
# State Audit Trail System
#
# Provides audit logging for all state modifications.
# Required for Phase 2 production migration per security review.
#
# Security review: docs/security-review-state-api-20251104.md
# Log format: docs/audit-log-format.md
# Concurrency safety: docs/ADR-002-concurrent-write-safety.md (using atomic-io.sh)

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
AUDIT_LOG="${DAEMON_ROOT}/logs/state-audit.jsonl"
AUDIT_LOG_DIR="${DAEMON_ROOT}/logs"

# Load atomic I/O library for concurrent-safe writes
source "${DAEMON_ROOT}/lib/atomic-io.sh"

# ============================================================================
# Core Audit Function
# ============================================================================

state_audit() {
    local operation="$1"
    local details="$2"
    local caller="${3:-unknown}"

    # Ensure log directory exists
    mkdir -p "$AUDIT_LOG_DIR" 2>/dev/null || true

    # Create audit log entry (JSONL format)
    local entry
    entry=$(jq -nc \
        --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg op "$operation" \
        --arg details "$details" \
        --arg caller "$caller" \
        --arg pid "$$" \
        --arg user "${USER:-unknown}" \
        '{
            timestamp: $ts,
            operation: $op,
            details: $details,
            caller: $caller,
            pid: ($pid | tonumber),
            user: $user
        }') || {
        # Fallback if jq fails: write simple log entry with atomic_append
        local fallback_entry="{\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"operation\":\"$operation\",\"details\":\"$details\",\"caller\":\"$caller\",\"error\":\"jq_failed\"}"
        atomic_append "$AUDIT_LOG" "$fallback_entry" || return 1
        return 1
    }

    # Append to audit log (concurrent-safe write using atomic_append from ADR-002)
    atomic_append "$AUDIT_LOG" "$entry" || {
        echo "WARNING: Failed to write audit log entry" >&2
        return 1
    }
}

# ============================================================================
# Log Rotation
# ============================================================================

state_audit_rotate() {
    local archive_dir="${DAEMON_ROOT}/logs/archives"
    local current_month=$(date +%Y-%m)

    # Check if audit log exists and is non-empty
    if [ ! -f "$AUDIT_LOG" ] || [ ! -s "$AUDIT_LOG" ]; then
        return 0
    fi

    # Check if log needs rotation (size > 10MB or age > 30 days)
    local log_size=$(stat -f%z "$AUDIT_LOG" 2>/dev/null || stat -c%s "$AUDIT_LOG" 2>/dev/null || echo 0)
    local log_age_days=$(( ($(date +%s) - $(stat -f%m "$AUDIT_LOG" 2>/dev/null || stat -c%Y "$AUDIT_LOG" 2>/dev/null || echo 0)) / 86400 ))

    if [ "$log_size" -gt 10485760 ] || [ "$log_age_days" -gt 30 ]; then
        mkdir -p "$archive_dir"

        # Rotate log with timestamp
        local archive_name="state-audit-${current_month}-$(date +%Y%m%d-%H%M%S).jsonl"
        mv "$AUDIT_LOG" "$archive_dir/$archive_name"

        # Compress archived log
        gzip "$archive_dir/$archive_name" 2>/dev/null || true

        echo "Audit log rotated: $archive_name.gz"
    fi
}

# ============================================================================
# Query Functions (for debugging/investigation)
# ============================================================================

state_audit_show_recent() {
    local count="${1:-20}"

    if [ ! -f "$AUDIT_LOG" ]; then
        echo "No audit log found at $AUDIT_LOG"
        return 1
    fi

    tail -n "$count" "$AUDIT_LOG" | jq -r '
        [.timestamp, .operation, .caller, .details] | @tsv
    ' | column -t -s $'\t'
}

state_audit_show_persona_switches() {
    if [ ! -f "$AUDIT_LOG" ]; then
        echo "No audit log found"
        return 1
    fi

    jq -r 'select(.operation == "persona_switch") |
        [.timestamp, .details] | @tsv' "$AUDIT_LOG" | column -t -s $'\t'
}

state_audit_show_by_caller() {
    local caller="$1"

    if [ ! -f "$AUDIT_LOG" ]; then
        echo "No audit log found"
        return 1
    fi

    jq -r --arg caller "$caller" \
        'select(.caller == $caller) |
        [.timestamp, .operation, .details] | @tsv' "$AUDIT_LOG" | column -t -s $'\t'
}

# ============================================================================
# Main
# ============================================================================

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    # Script executed directly
    if [ $# -eq 0 ]; then
        echo "State Audit Trail System"
        echo
        echo "Usage:"
        echo "  state_audit <operation> <details> [caller]  - Log audit entry"
        echo "  state_audit_show_recent [count]             - Show recent entries (default: 20)"
        echo "  state_audit_show_persona_switches           - Show all persona switches"
        echo "  state_audit_show_by_caller <caller>         - Show entries by caller"
        echo "  state_audit_rotate                          - Rotate audit log"
    else
        "$@"
    fi
fi
