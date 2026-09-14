#!/bin/bash
#
# Audit Coverage Monitoring Script
#
# Purpose: Monitors audit logging coverage by comparing actual persona activations
#          against audit log entries. Detects gaps in audit trail.
#
# Created: 2025-11-04 by Maintainer
# Context: Built in response to daemon.sh audit bypass incident (95% gap discovered)
#
# Usage:
#   ./audit-coverage-monitor.sh              # One-time check with summary
#   ./audit-coverage-monitor.sh --watch      # Continuous monitoring (5 min intervals)
#   ./audit-coverage-monitor.sh --alert      # Check and exit 1 if coverage < 90%
#   ./audit-coverage-monitor.sh --detailed   # Show per-persona breakdown
#
# Exit codes:
#   0 - Coverage >= 90% (healthy)
#   1 - Coverage < 90% (alert)
#   2 - Coverage < 50% (critical)
#   3 - Error reading files

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
STATE_FILE="${DAEMON_ROOT}/personalities/state.json"
AUDIT_LOG="${DAEMON_ROOT}/logs/state-audit.jsonl"

# Coverage thresholds
COVERAGE_HEALTHY=90    # >= 90% = healthy
COVERAGE_WARNING=50    # 50-89% = warning
COVERAGE_CRITICAL=50   # < 50% = critical

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_critical() {
    echo -e "${RED}${BOLD}[CRITICAL]${NC} $*"
}

# ============================================================================
# Core Monitoring Functions
# ============================================================================

# Get total persona activations from state.json
get_total_activations() {
    if [ ! -f "$STATE_FILE" ]; then
        log_error "State file not found: $STATE_FILE"
        exit 3
    fi

    jq '[.personas[].total_activations] | add' "$STATE_FILE"
}

# Get per-persona activation counts
get_persona_activations() {
    if [ ! -f "$STATE_FILE" ]; then
        log_error "State file not found: $STATE_FILE"
        exit 3
    fi

    jq -r '.personas | to_entries[] | "\(.key):\(.value.total_activations)"' "$STATE_FILE"
}

# Count total audit log entries
get_audit_entries_total() {
    if [ ! -f "$AUDIT_LOG" ]; then
        log_warning "Audit log not found: $AUDIT_LOG"
        echo "0"
        return
    fi

    wc -l < "$AUDIT_LOG" | tr -d ' '
}

# Count persona switch audit entries specifically
get_audit_persona_switches() {
    if [ ! -f "$AUDIT_LOG" ]; then
        echo "0"
        return
    fi

    grep -c '"operation":"persona_switch"' "$AUDIT_LOG" 2>/dev/null || echo "0"
}

# Calculate coverage percentage
calculate_coverage() {
    local total_activations="$1"
    local audit_entries="$2"

    if [ "$total_activations" -eq 0 ]; then
        echo "0"
        return
    fi

    # Calculate percentage (bash integer math)
    echo $(( (audit_entries * 100) / total_activations ))
}

# Get coverage status (healthy/warning/critical)
get_coverage_status() {
    local coverage="$1"

    if [ "$coverage" -ge "$COVERAGE_HEALTHY" ]; then
        echo "HEALTHY"
    elif [ "$coverage" -ge "$COVERAGE_WARNING" ]; then
        echo "WARNING"
    else
        echo "CRITICAL"
    fi
}

# ============================================================================
# Reporting Functions
# ============================================================================

# Print basic coverage summary
print_summary() {
    local total_activations="$1"
    local audit_entries="$2"
    local coverage="$3"
    local status="$4"

    echo ""
    echo "======================================================================"
    echo "  Audit Coverage Report"
    echo "======================================================================"
    echo ""
    echo "Total persona activations: $total_activations"
    echo "Audit log entries:         $audit_entries"
    echo "Coverage:                  ${coverage}%"
    echo ""

    case "$status" in
        HEALTHY)
            log_success "Status: HEALTHY (>=${COVERAGE_HEALTHY}%)"
            echo ""
            echo "✓ Audit logging is working correctly"
            echo "✓ Coverage meets production requirements"
            ;;
        WARNING)
            log_warning "Status: WARNING (${COVERAGE_WARNING}-${COVERAGE_HEALTHY}%)"
            echo ""
            echo "⚠ Audit coverage below healthy threshold"
            echo "⚠ Some persona switches may not be logged"
            echo "⚠ Review daemon.sh integration"
            ;;
        CRITICAL)
            log_critical "Status: CRITICAL (<${COVERAGE_CRITICAL}%)"
            echo ""
            echo "✗ Audit logging severely degraded"
            echo "✗ Most persona switches are NOT logged"
            echo "✗ URGENT: Review State API integration"
            echo ""
            echo "Likely causes:"
            echo "  1. daemon.sh not using State API"
            echo "  2. Scripts bypassing state_become()"
            echo "  3. Audit logging disabled or failing"
            ;;
    esac

    echo ""
    echo "======================================================================"
}

# Print detailed per-persona breakdown
print_detailed() {
    local audit_switches="$1"

    echo ""
    echo "======================================================================"
    echo "  Per-Persona Activation Breakdown"
    echo "======================================================================"
    echo ""
    printf "%-15s %12s\n" "Persona" "Activations"
    echo "----------------------------------------------------------------------"

    while IFS=: read -r persona count; do
        printf "%-15s %12s\n" "$persona" "$count"
    done < <(get_persona_activations)

    echo "======================================================================"
    echo ""
    echo "NOTE: Persona switches logged in audit trail: $audit_switches"
    echo "      (Audit trail may also contain emotional_update entries)"
    echo ""
}

# Print recommendations based on coverage
print_recommendations() {
    local coverage="$1"
    local status="$2"

    if [ "$status" = "HEALTHY" ]; then
        return  # No recommendations needed
    fi

    echo "======================================================================"
    echo "  Recommendations"
    echo "======================================================================"
    echo ""

    if [ "$coverage" -lt 50 ]; then
        echo "IMMEDIATE ACTIONS:"
        echo ""
        echo "1. Check if daemon.sh uses State API:"
        echo "   grep -n 'state_become' $DAEMON_ROOT/daemon.sh"
        echo ""
        echo "2. Verify State API is sourced:"
        echo "   grep -n 'state-api.sh' $DAEMON_ROOT/daemon.sh"
        echo ""
        echo "3. Review audit log for recent entries:"
        echo "   tail -20 $AUDIT_LOG"
        echo ""
        echo "4. Test manual persona switch (should log):"
        echo "   ./claude-daemon-switch-persona.sh experimenter testing"
        echo "   tail -1 $AUDIT_LOG  # Should show new entry"
        echo ""
    else
        echo "RECOMMENDED ACTIONS:"
        echo ""
        echo "1. Review recent audit log entries:"
        echo "   tail -50 $AUDIT_LOG"
        echo ""
        echo "2. Check for daemon-driven vs manual switches:"
        echo "   grep 'daemon' $AUDIT_LOG"
        echo "   grep 'claude-daemon-switch-persona' $AUDIT_LOG"
        echo ""
        echo "3. Monitor coverage over 24 hours:"
        echo "   $0 --watch"
        echo ""
    fi

    echo "======================================================================"
}

# ============================================================================
# Monitoring Modes
# ============================================================================

# One-time check
mode_check() {
    local detailed="${1:-false}"

    local total_activations
    local audit_entries
    local audit_switches
    local coverage
    local status

    total_activations=$(get_total_activations)
    audit_entries=$(get_audit_entries_total)
    audit_switches=$(get_audit_persona_switches)
    coverage=$(calculate_coverage "$total_activations" "$audit_switches")
    status=$(get_coverage_status "$coverage")

    print_summary "$total_activations" "$audit_switches" "$coverage" "$status"

    if [ "$detailed" = "true" ]; then
        print_detailed "$audit_switches"
    fi

    if [ "$status" != "HEALTHY" ]; then
        print_recommendations "$coverage" "$status"
    fi

    # Exit with appropriate code
    case "$status" in
        HEALTHY)   exit 0 ;;
        WARNING)   exit 1 ;;
        CRITICAL)  exit 2 ;;
    esac
}

# Continuous monitoring (watch mode)
mode_watch() {
    local interval="${1:-300}"  # 5 minutes default

    log_info "Starting continuous monitoring (interval: ${interval}s)"
    log_info "Press Ctrl+C to stop"
    echo ""

    while true; do
        local timestamp
        timestamp=$(date +'%Y-%m-%d %H:%M:%S')

        local total_activations
        local audit_switches
        local coverage
        local status

        total_activations=$(get_total_activations)
        audit_switches=$(get_audit_persona_switches)
        coverage=$(calculate_coverage "$total_activations" "$audit_switches")
        status=$(get_coverage_status "$coverage")

        printf "[%s] Activations: %3d | Audit: %3d | Coverage: %3d%% | Status: %-8s\n" \
            "$timestamp" "$total_activations" "$audit_switches" "$coverage" "$status"

        if [ "$status" = "CRITICAL" ]; then
            log_critical "Coverage critically low! Investigate immediately."
        elif [ "$status" = "WARNING" ]; then
            log_warning "Coverage below threshold. Monitor closely."
        fi

        sleep "$interval"
    done
}

# Alert mode (for CI/monitoring systems)
mode_alert() {
    local total_activations
    local audit_switches
    local coverage

    total_activations=$(get_total_activations)
    audit_switches=$(get_audit_persona_switches)
    coverage=$(calculate_coverage "$total_activations" "$audit_switches")

    if [ "$coverage" -lt "$COVERAGE_HEALTHY" ]; then
        echo "ALERT: Audit coverage at ${coverage}% (threshold: ${COVERAGE_HEALTHY}%)"
        echo "Total activations: $total_activations"
        echo "Audit entries: $audit_switches"
        exit 1
    else
        echo "OK: Audit coverage at ${coverage}%"
        exit 0
    fi
}

# ============================================================================
# Main
# ============================================================================

main() {
    local mode="check"
    local detailed="false"

    # Parse arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            --watch)
                mode="watch"
                shift
                ;;
            --alert)
                mode="alert"
                shift
                ;;
            --detailed|-d)
                detailed="true"
                shift
                ;;
            --interval)
                WATCH_INTERVAL="$2"
                shift 2
                ;;
            --help|-h)
                cat <<EOF
Audit Coverage Monitoring Script

Usage:
  $0 [OPTIONS]

Options:
  --watch              Continuous monitoring mode (5 min intervals)
  --alert              Alert mode (exit 1 if coverage < 90%)
  --detailed, -d       Show detailed per-persona breakdown
  --interval SECONDS   Set watch interval (default: 300)
  --help, -h           Show this help message

Examples:
  $0                   # One-time check
  $0 --detailed        # One-time check with breakdown
  $0 --watch           # Continuous monitoring
  $0 --alert           # Alert mode (for cron/CI)

Exit codes:
  0 - Coverage >= 90% (healthy)
  1 - Coverage < 90% (warning)
  2 - Coverage < 50% (critical)
  3 - Error reading files

Coverage thresholds:
  Healthy:   >= ${COVERAGE_HEALTHY}%
  Warning:   ${COVERAGE_WARNING}-$(($COVERAGE_HEALTHY-1))%
  Critical:  < ${COVERAGE_CRITICAL}%

Created: 2025-11-04 by Maintainer
Context: Response to daemon.sh audit bypass incident
EOF
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 3
                ;;
        esac
    done

    # Execute requested mode
    case "$mode" in
        check)
            mode_check "$detailed"
            ;;
        watch)
            mode_watch "${WATCH_INTERVAL:-300}"
            ;;
        alert)
            mode_alert
            ;;
    esac
}

# Run main function
main "$@"
