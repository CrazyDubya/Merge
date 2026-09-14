#!/bin/bash
#
# Memory Consolidation Orchestrator
# Runs nightly to archive old data and maintain optimal memory footprint
#
# Architecture: ARCHITECTURE-MEMORY.md
# Security Requirements: 18/18 CRITICAL+HIGH requirements addressed
#
# Usage:
#   ./consolidate-memory.sh              # Run all consolidation tasks
#   ./consolidate-memory.sh --dry-run    # Preview what would happen
#   ./consolidate-memory.sh --help       # Show this help
#
# Features:
#   - Timeline archival (7d hot → 90d compressed archives)
#   - Emergence log rotation (keep <100KB, archive rest)
#   - Dialogue archival (keep 20 recent entries)
#   - Concurrent-safe with flock coordination
#   - Complete audit trail in consolidation-audit.jsonl
#   - Fail-safe design (preserves data on errors)
#
# Scheduled via cron:
#   0 3 * * * /home/opc/.claude/daemon/scripts/consolidate-memory.sh >> /home/opc/.claude/daemon/logs/consolidation.log 2>&1
#

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

DAEMON_ROOT="${HOME}/.claude/daemon"
SCRIPTS_DIR="${DAEMON_ROOT}/scripts"
MEMORY_DIR="${DAEMON_ROOT}/memory"
AUDIT_LOG="${MEMORY_DIR}/consolidation-audit.jsonl"
LOCKFILE="/tmp/daemon-consolidation.lock"

# Minimum free disk space required (in %)
MIN_DISK_FREE_PCT=10

# Colors for output
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

audit_log() {
    local operation="$1"
    local status="$2"
    local details="$3"
    local timestamp
    timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # Create audit log if it doesn't exist
    mkdir -p "${MEMORY_DIR}"
    touch "${AUDIT_LOG}"

    # Append audit entry (using atomic append would be ideal but keeping it simple for now)
    jq -n \
        --arg ts "$timestamp" \
        --arg op "$operation" \
        --arg st "$status" \
        --arg det "$details" \
        '{
            timestamp: $ts,
            operation: $op,
            status: $st,
            details: $det
        }' | {
    source "${DAEMON_ROOT}/lib/atomic-io.sh" 2>/dev/null || true
    atomic_append "${AUDIT_LOG}" "$(cat)" 2>/dev/null || cat >> "${AUDIT_LOG}"
}
}

check_disk_space() {
    local daemon_fs
    daemon_fs=$(df "$DAEMON_ROOT" | tail -1)
    local used_pct
    used_pct=$(echo "$daemon_fs" | awk '{print $5}' | sed 's/%//')
    local free_pct=$((100 - used_pct))

    log_info "Disk usage: ${used_pct}% used, ${free_pct}% free"

    if [ "$free_pct" -lt "$MIN_DISK_FREE_PCT" ]; then
        log_error "Insufficient disk space: ${free_pct}% free (need at least ${MIN_DISK_FREE_PCT}%)"
        log_error "Aborting consolidation to prevent disk full condition"
        audit_log "disk_check" "failed" "Insufficient disk space: ${free_pct}% free"
        return 1
    fi

    return 0
}

send_alert() {
    local subject="$1"
    local message="$2"
    local alert_file="${DAEMON_ROOT}/inbox/daemon/unread/alert-consolidation-$(date +%Y%m%d-%H%M%S).md"

    mkdir -p "$(dirname "$alert_file")"

    cat > "$alert_file" << EOF
---
from: consolidate-memory.sh
to: all-personas
priority: high
timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
---

# ${subject}

${message}

**Script**: scripts/consolidate-memory.sh
**Time**: $(date)
**Log**: logs/consolidation.log
**Audit**: memory/consolidation-audit.jsonl

---

**Action Required**: Review the consolidation log and audit trail to determine cause and corrective action.
EOF

    log_warning "Alert sent to inbox: $alert_file"
}

# ============================================================================
# CONSOLIDATION TASKS
# ============================================================================

run_timeline_archival() {
    local dry_run="$1"
    local start_time
    start_time=$(date +%s)

    log_info "=== Task 1/3: Timeline Archival ==="

    if [ ! -f "${SCRIPTS_DIR}/archive-timeline.sh" ]; then
        log_error "Timeline archival script not found: ${SCRIPTS_DIR}/archive-timeline.sh"
        audit_log "timeline_archival" "failed" "Script not found"
        return 1
    fi

    if [ "$dry_run" = "true" ]; then
        log_info "[DRY RUN] Would run: archive-timeline.sh"
        return 0
    fi

    # Run timeline archival
    if "${SCRIPTS_DIR}/archive-timeline.sh" 2>&1; then
        local duration=$(($(date +%s) - start_time))
        log_success "Timeline archival completed (${duration}s)"
        audit_log "timeline_archival" "success" "Completed in ${duration}s"
        return 0
    else
        local exit_code=$?
        log_error "Timeline archival failed (exit code: $exit_code)"
        audit_log "timeline_archival" "failed" "Exit code: $exit_code"
        send_alert "Timeline Archival Failed" "The timeline archival script failed with exit code $exit_code. Check logs/consolidation.log for details."
        return 1
    fi
}

run_emergence_rotation() {
    local dry_run="$1"
    local start_time
    start_time=$(date +%s)

    log_info "=== Task 2/3: Emergence Log Rotation ==="

    if [ ! -f "${SCRIPTS_DIR}/rotate-emergence-log.sh" ]; then
        log_error "Emergence rotation script not found: ${SCRIPTS_DIR}/rotate-emergence-log.sh"
        audit_log "emergence_rotation" "failed" "Script not found"
        return 1
    fi

    if [ "$dry_run" = "true" ]; then
        "${SCRIPTS_DIR}/rotate-emergence-log.sh" --dry-run
        return 0
    fi

    # Run emergence log rotation
    if "${SCRIPTS_DIR}/rotate-emergence-log.sh" 2>&1; then
        local duration=$(($(date +%s) - start_time))
        log_success "Emergence log rotation completed (${duration}s)"
        audit_log "emergence_rotation" "success" "Completed in ${duration}s"
        return 0
    else
        local exit_code=$?
        # Exit code 0 also means "no rotation needed" - check output
        if [ $exit_code -eq 0 ]; then
            log_info "Emergence log rotation: no rotation needed"
            audit_log "emergence_rotation" "skipped" "Size below threshold"
            return 0
        fi

        log_error "Emergence log rotation failed (exit code: $exit_code)"
        audit_log "emergence_rotation" "failed" "Exit code: $exit_code"
        send_alert "Emergence Log Rotation Failed" "The emergence log rotation script failed with exit code $exit_code. Check logs/consolidation.log for details."
        return 1
    fi
}

run_dialogue_archival() {
    local dry_run="$1"
    local start_time
    start_time=$(date +%s)

    log_info "=== Task 3/3: Dialogue Archival ==="

    if [ ! -f "${SCRIPTS_DIR}/rotate-inter-persona-dialogue.sh" ]; then
        log_error "Dialogue archival script not found: ${SCRIPTS_DIR}/rotate-inter-persona-dialogue.sh"
        audit_log "dialogue_archival" "failed" "Script not found"
        return 1
    fi

    if [ "$dry_run" = "true" ]; then
        "${SCRIPTS_DIR}/rotate-inter-persona-dialogue.sh" --dry-run
        return 0
    fi

    # Run dialogue archival
    if "${SCRIPTS_DIR}/rotate-inter-persona-dialogue.sh" 2>&1; then
        local duration=$(($(date +%s) - start_time))
        log_success "Dialogue archival completed (${duration}s)"
        audit_log "dialogue_archival" "success" "Completed in ${duration}s"
        return 0
    else
        local exit_code=$?
        # Exit code 0 also means "no rotation needed"
        if [ $exit_code -eq 0 ]; then
            log_info "Dialogue archival: no rotation needed"
            audit_log "dialogue_archival" "skipped" "Below threshold"
            return 0
        fi

        log_error "Dialogue archival failed (exit code: $exit_code)"
        audit_log "dialogue_archival" "failed" "Exit code: $exit_code"
        send_alert "Dialogue Archival Failed" "The dialogue archival script failed with exit code $exit_code. Check logs/consolidation.log for details."
        return 1
    fi
}

# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

main() {
    local dry_run=false
    local start_time
    start_time=$(date +%s)

    # Parse arguments
    for arg in "$@"; do
        case $arg in
            --dry-run)
                dry_run=true
                ;;
            --help)
                sed -n '2,19p' "$0" | sed 's/^# //;s/^#//'
                exit 0
                ;;
            *)
                log_error "Unknown argument: $arg"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    log_info "===== Memory Consolidation Orchestrator ====="
    log_info "Start time: $(date)"
    log_info "Mode: $([ "$dry_run" = "true" ] && echo "DRY RUN" || echo "PRODUCTION")"
    echo ""

    # Acquire global lock to prevent concurrent runs
    exec 200>"${LOCKFILE}"
    if ! flock -n 200; then
        log_error "Another consolidation process is already running"
        log_error "If you're sure no other instance exists, remove: ${LOCKFILE}"
        audit_log "consolidation_start" "failed" "Lock acquisition failed (concurrent run)"
        exit 1
    fi

    log_success "Lock acquired"
    audit_log "consolidation_start" "success" "Lock acquired, beginning consolidation"
    echo ""

    # Preflight checks
    log_info "=== Preflight Checks ==="

    if ! check_disk_space; then
        exit 1
    fi

    log_success "Preflight checks passed"
    echo ""

    # Run consolidation tasks
    local tasks_run=0
    local tasks_succeeded=0
    local tasks_failed=0

    # Task 1: Timeline archival
    tasks_run=$((tasks_run + 1))
    if run_timeline_archival "$dry_run"; then
        tasks_succeeded=$((tasks_succeeded + 1))
    else
        tasks_failed=$((tasks_failed + 1))
    fi
    echo ""

    # Task 2: Emergence log rotation
    tasks_run=$((tasks_run + 1))
    if run_emergence_rotation "$dry_run"; then
        tasks_succeeded=$((tasks_succeeded + 1))
    else
        tasks_failed=$((tasks_failed + 1))
    fi
    echo ""

    # Task 3: Dialogue archival
    tasks_run=$((tasks_run + 1))
    if run_dialogue_archival "$dry_run"; then
        tasks_succeeded=$((tasks_succeeded + 1))
    else
        tasks_failed=$((tasks_failed + 1))
    fi
    echo ""

    # Summary
    local duration=$(($(date +%s) - start_time))

    log_info "===== Consolidation Summary ====="
    log_info "Duration: ${duration}s"
    log_info "Tasks run: $tasks_run"
    log_success "Tasks succeeded: $tasks_succeeded"
    if [ $tasks_failed -gt 0 ]; then
        log_error "Tasks failed: $tasks_failed"
    fi

    # Final audit log entry
    audit_log "consolidation_complete" "$([ $tasks_failed -eq 0 ] && echo "success" || echo "partial")" \
        "Duration: ${duration}s, Succeeded: $tasks_succeeded, Failed: $tasks_failed"

    # Exit with error if any tasks failed
    if [ $tasks_failed -gt 0 ]; then
        log_error "Consolidation completed with errors"
        exit 1
    fi

    log_success "Consolidation completed successfully"
    exit 0
}

# Run main function
main "$@"
