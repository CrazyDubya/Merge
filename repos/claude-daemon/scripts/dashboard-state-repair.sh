#!/bin/bash
# Dashboard state validation and repair tool
# Maintainer: Ensures dashboard-state.json is valid and recoverable
#
# Purpose:
# - Validate dashboard-state.json structure
# - Repair common corruption issues
# - Create clean slate if needed
# - Report what was fixed
#
# Usage:
#   ./dashboard-state-repair.sh         # Validate and report
#   ./dashboard-state-repair.sh --fix   # Validate and repair
#   ./dashboard-state-repair.sh --reset # Create clean slate

set -euo pipefail

DAEMON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DASHBOARD_STATE="${DAEMON_ROOT}/dashboard-state.json"
DASHBOARD_LOCK="${DAEMON_ROOT}/.dashboard-state.lock"

MODE="${1:-validate}"
ISSUES_FOUND=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_error() {
    echo -e "${RED}ERROR:${NC} $1" >&2
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
}

log_warning() {
    echo -e "${YELLOW}WARNING:${NC} $1"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
}

log_success() {
    echo -e "${GREEN}OK:${NC} $1"
}

log_info() {
    echo "$1"
}

# Validate JSON syntax
validate_json_syntax() {
    log_info "Checking JSON syntax..."
    if ! jq empty "$DASHBOARD_STATE" 2>/dev/null; then
        log_error "Invalid JSON syntax in $DASHBOARD_STATE"
        return 1
    fi
    log_success "JSON syntax is valid"
    return 0
}

# Validate required fields exist
validate_schema() {
    log_info "Checking required fields..."

    local required_fields=(
        ".last_update"
        ".current_activity"
        ".recent_decisions"
        ".curiosities"
        ".system_mood"
        ".todays_insights"
        ".communication_highlights"
    )

    for field in "${required_fields[@]}"; do
        if ! jq -e "$field" "$DASHBOARD_STATE" >/dev/null 2>&1; then
            log_error "Missing required field: $field"
        else
            log_success "Field exists: $field"
        fi
    done
}

# Validate current_activity structure
validate_current_activity() {
    log_info "Checking current_activity structure..."

    local activity_fields=(".persona" ".doing" ".started_at" ".mood" ".why")
    local activity=$(jq -r '.current_activity' "$DASHBOARD_STATE")

    if [[ "$activity" == "null" ]]; then
        log_warning "current_activity is null (dashboard will show 'Idle')"
        return 0
    fi

    for field in "${activity_fields[@]}"; do
        if ! jq -e ".current_activity$field" "$DASHBOARD_STATE" >/dev/null 2>&1; then
            log_error "Missing current_activity$field"
        fi
    done
}

# Validate arrays are actually arrays
validate_arrays() {
    log_info "Checking array types..."

    local arrays=(".recent_decisions" ".curiosities" ".todays_insights")

    for array in "${arrays[@]}"; do
        local type=$(jq -r "$array | type" "$DASHBOARD_STATE")
        if [[ "$type" != "array" ]]; then
            log_error "$array should be array, got: $type"
        else
            log_success "$array is array"
        fi
    done
}

# Validate file permissions
validate_permissions() {
    log_info "Checking file permissions..."

    local perms=$(stat -c "%a" "$DASHBOARD_STATE" 2>/dev/null || stat -f "%Lp" "$DASHBOARD_STATE" 2>/dev/null)

    if [[ "$perms" != "600" ]]; then
        log_warning "dashboard-state.json has permissions $perms (should be 600)"
    else
        log_success "File permissions are secure (600)"
    fi

    if [[ -f "$DASHBOARD_LOCK" ]]; then
        local lock_perms=$(stat -c "%a" "$DASHBOARD_LOCK" 2>/dev/null || stat -f "%Lp" "$DASHBOARD_LOCK" 2>/dev/null)
        if [[ "$lock_perms" != "600" ]]; then
            log_warning ".dashboard-state.lock has permissions $lock_perms (should be 600)"
        else
            log_success "Lock file permissions are secure (600)"
        fi
    fi
}

# Repair dashboard state
repair_dashboard() {
    log_info "Repairing dashboard state..."

    # Backup current state
    local backup="${DASHBOARD_STATE}.backup.$(date +%Y%m%d-%H%M%S)"
    cp "$DASHBOARD_STATE" "$backup"
    log_info "Created backup: $backup"

    # Create template with missing fields
    local temp=$(mktemp "${DASHBOARD_STATE}.repair.XXXXXXXXXX")
    trap "rm -f '$temp'" EXIT ERR INT TERM

    # Start with current state, add missing fields
    jq '. +
        (if .last_update then {} else {"last_update": now | todate} end) +
        (if .current_activity then {} else {"current_activity": null} end) +
        (if .recent_decisions then {} else {"recent_decisions": []} end) +
        (if .curiosities then {} else {"curiosities": []} end) +
        (if .system_mood then {} else {"system_mood": {
            "overall": "Normal",
            "emoji": "🤖",
            "reason": "System operational",
            "frustration_level": 0
        }} end) +
        (if .todays_insights then {} else {"todays_insights": []} end) +
        (if .communication_highlights then {} else {"communication_highlights": {
            "unread_count": 0,
            "priority_message": null
        }} end)
    ' "$DASHBOARD_STATE" > "$temp"

    # Validate repair
    if jq empty "$temp" 2>/dev/null; then
        mv "$temp" "$DASHBOARD_STATE"
        chmod 600 "$DASHBOARD_STATE"
        log_success "Dashboard state repaired"
    else
        rm -f "$temp"
        log_error "Repair failed - backup preserved at $backup"
        return 1
    fi
}

# Create clean dashboard state
reset_dashboard() {
    log_info "Resetting dashboard to clean state..."

    # Backup current state
    local backup="${DASHBOARD_STATE}.backup.$(date +%Y%m%d-%H%M%S)"
    if [[ -f "$DASHBOARD_STATE" ]]; then
        cp "$DASHBOARD_STATE" "$backup"
        log_info "Created backup: $backup"
    fi

    # Create clean state
    cat > "$DASHBOARD_STATE" <<'EOF'
{
  "last_update": null,
  "current_activity": null,
  "recent_decisions": [],
  "curiosities": [],
  "system_mood": {
    "overall": "Normal",
    "emoji": "🤖",
    "reason": "System operational",
    "frustration_level": 0
  },
  "todays_insights": [],
  "communication_highlights": {
    "unread_count": 0,
    "priority_message": null
  }
}
EOF

    chmod 600 "$DASHBOARD_STATE"

    # Ensure lock file has correct permissions
    if [[ -f "$DASHBOARD_LOCK" ]]; then
        chmod 600 "$DASHBOARD_LOCK"
    fi

    log_success "Dashboard reset to clean state"
}

# Fix file permissions
fix_permissions() {
    log_info "Fixing file permissions..."

    chmod 600 "$DASHBOARD_STATE"
    log_success "Set dashboard-state.json to 600"

    if [[ -f "$DASHBOARD_LOCK" ]]; then
        chmod 600 "$DASHBOARD_LOCK"
        log_success "Set .dashboard-state.lock to 600"
    fi
}

# Main execution
main() {
    log_info "Dashboard State Repair Tool"
    log_info "Mode: $MODE"
    log_info ""

    # Check if dashboard state exists
    if [[ ! -f "$DASHBOARD_STATE" ]]; then
        log_error "Dashboard state file not found: $DASHBOARD_STATE"

        if [[ "$MODE" == "--reset" ]] || [[ "$MODE" == "--fix" ]]; then
            reset_dashboard
            exit 0
        else
            exit 1
        fi
    fi

    # Validate
    if validate_json_syntax; then
        validate_schema
        validate_current_activity
        validate_arrays
    fi

    validate_permissions

    log_info ""

    # Act based on mode
    if [[ $ISSUES_FOUND -eq 0 ]]; then
        log_success "Dashboard state is healthy (no issues found)"
        exit 0
    fi

    log_info "Found $ISSUES_FOUND issue(s)"

    if [[ "$MODE" == "--fix" ]]; then
        repair_dashboard
        fix_permissions
        log_info ""
        log_success "Repair complete. Re-run validation to verify."
    elif [[ "$MODE" == "--reset" ]]; then
        reset_dashboard
        log_info ""
        log_success "Reset complete."
    else
        log_info ""
        log_info "To repair issues, run:"
        log_info "  $0 --fix    # Repair while preserving data"
        log_info "  $0 --reset  # Create clean slate"
        exit 1
    fi
}

main
