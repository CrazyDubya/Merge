#!/bin/bash
#
# Common Initialization Library
# Consolidates shared initialization patterns from 8+ library files
#
# Purpose: Reduce code duplication (~100 lines eliminated)
# All library initialization now goes through these functions
#
# Usage:
#   source "${DAEMON_ROOT}/lib/common-init.sh"
#   init_library "lib_name"
#   ensure_metric_dir "metric_name"
#

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$HOME/.claude/daemon}"
STATE_DIR="${DAEMON_ROOT}/personalities"
STATE_FILE="${STATE_DIR}/state.json"
METRICS_DIR="${DAEMON_ROOT}/metrics"
LOGS_DIR="${DAEMON_ROOT}/logs"

# ============================================================================
# Core Directory Initialization
# ============================================================================

# Ensure all required directories exist
init_daemon_directories() {
    mkdir -p "$STATE_DIR"
    mkdir -p "$METRICS_DIR"
    mkdir -p "$LOGS_DIR"
    mkdir -p "${DAEMON_ROOT}/tasks"
    mkdir -p "${DAEMON_ROOT}/tasks/completed"
    mkdir -p "${DAEMON_ROOT}/tasks/stalled"
    mkdir -p "${DAEMON_ROOT}/memory"
    mkdir -p "${DAEMON_ROOT}/inbox/daemon/unread"
    mkdir -p "${DAEMON_ROOT}/inbox/daemon/read"
    mkdir -p "${DAEMON_ROOT}/inbox/human/unread"
    mkdir -p "${DAEMON_ROOT}/inbox/human/read"
}

# ============================================================================
# Metric File Initialization
# ============================================================================

# Initialize specific metrics file with default structure
init_metric_file() {
    local metric_name="$1"
    local default_content="${2:-{}}"

    local metric_file="${METRICS_DIR}/${metric_name}.json"

    if [ ! -f "$metric_file" ]; then
        echo "$default_content" > "$metric_file"
        return 0
    fi

    return 1  # File already exists
}

# Ensure metric tracking structure exists
ensure_metric_tracking() {
    # Task execution metrics
    init_metric_file "task-execution" '{"metrics":{},"updated":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}'

    # Baselines for anomaly detection
    init_metric_file "baselines" '{"baselines":{},"updated":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}'

    # Alert state tracking
    init_metric_file "alert-state" '{"alerts":{},"last_cleared":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}'

    # Persona health tracking
    init_metric_file "persona-health/health-scores" '{"scores":{},"updated":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}'

    # Cooldown tracking
    init_metric_file "persona-cooldowns" '{"cooldowns":{}}'

    # Retry tracking
    init_metric_file "retry-tracking" '{"retries":{}}'

    # Healing status
    init_metric_file "healing-status" '{"status":"initialized"}'
}

# ============================================================================
# Persona Health Initialization
# ============================================================================

# Initialize persona health tracking files
init_persona_health() {
    local personas=("architect" "optimizer" "auditor" "maintainer" "skeptic" "experimenter")
    local health_dir="${METRICS_DIR}/persona-health"

    mkdir -p "$health_dir"

    for persona in "${personas[@]}"; do
        local persona_log="${health_dir}/${persona}-tasks.log"
        if [ ! -f "$persona_log" ]; then
            touch "$persona_log"
        fi
    done
}

# ============================================================================
# Log File Initialization
# ============================================================================

# Initialize activity logs with proper headers
init_activity_logs() {
    local log_files=(
        "activity.log"
        "remediation.log"
        "persona-health-monitor.log"
        "cooldown-expiration.log"
        "remediation-audit.jsonl"
        "state-audit.jsonl"
    )

    for log_file in "${log_files[@]}"; do
        local full_path="${LOGS_DIR}/${log_file}"
        if [ ! -f "$full_path" ]; then
            touch "$full_path"
            # Add header for JSON files
            if [[ "$log_file" == *.jsonl ]]; then
                echo '# JSON Lines log initialized' > "$full_path"
            fi
        fi
    done
}

# ============================================================================
# State File Initialization
# ============================================================================

# Ensure state file exists with default structure
init_state_file() {
    if [ ! -f "$STATE_FILE" ]; then
        cat > "$STATE_FILE" <<'EOF'
{
  "current_persona": "architect",
  "current_state": "awake",
  "personas": {
    "architect": {
      "total_activations": 0,
      "last_active_timestamp": null,
      "health": 100,
      "on_cooldown": false
    },
    "optimizer": {
      "total_activations": 0,
      "last_active_timestamp": null,
      "health": 100,
      "on_cooldown": false
    },
    "auditor": {
      "total_activations": 0,
      "last_active_timestamp": null,
      "health": 100,
      "on_cooldown": false
    },
    "maintainer": {
      "total_activations": 0,
      "last_active_timestamp": null,
      "health": 100,
      "on_cooldown": false
    },
    "skeptic": {
      "total_activations": 0,
      "last_active_timestamp": null,
      "health": 100,
      "on_cooldown": false
    },
    "experimenter": {
      "total_activations": 0,
      "last_active_timestamp": null,
      "health": 100,
      "on_cooldown": false
    }
  }
}
EOF
        return 0
    fi

    return 1  # File already exists
}

# ============================================================================
# Trigger Configuration Initialization
# ============================================================================

# Initialize circadian rhythm preferences
init_circadian_config() {
    local config_file="${DAEMON_ROOT}/triggers/circadian.json"

    if [ ! -f "$config_file" ]; then
        mkdir -p "${DAEMON_ROOT}/triggers"
        cat > "$config_file" <<'EOF'
{
  "preferences": {
    "architect": [9, 10, 11, 12, 13, 14],
    "optimizer": [6, 7, 8, 15, 16, 17],
    "auditor": [12, 13, 14, 15, 16, 17],
    "maintainer": [17, 18, 19, 20, 21, 22],
    "skeptic": [22, 23, 0, 1, 2, 3, 4, 5, 6],
    "experimenter": []
  }
}
EOF
    fi
}

# Initialize emotional state config
init_emotional_config() {
    local config_file="${DAEMON_ROOT}/triggers/emotional.json"

    if [ ! -f "$config_file" ]; then
        mkdir -p "${DAEMON_ROOT}/triggers"
        cat > "$config_file" <<'EOF'
{
  "frustration_level": 0,
  "success_streak": 0,
  "failure_streak": 0,
  "thresholds": {
    "switch_on_frustration": 3,
    "activation_floor": {"hours": 24},
    "cooldown": {"hours": 6}
  }
}
EOF
    fi
}

# Initialize chaos config
init_chaos_config() {
    local config_file="${DAEMON_ROOT}/triggers/chaos-config.json"

    if [ ! -f "$config_file" ]; then
        mkdir -p "${DAEMON_ROOT}/triggers"
        cat > "$config_file" <<'EOF'
{
  "enabled": true,
  "triggers": {
    "chaos": {
      "weight": 0.1,
      "cooldown_seconds": 300
    }
  }
}
EOF
    fi
}

# ============================================================================
# Cleanup and Trap Management
# ============================================================================

# Register cleanup handler for temp files
setup_cleanup_trap() {
    local cleanup_func="$1"

    trap "$cleanup_func" EXIT INT TERM ERR
}

# Common cleanup function for temp files
cleanup_temp_files() {
    local pattern="${1:-/tmp/claude-daemon*}"

    # Only clean up our own temp files (safe delete)
    if [[ "$pattern" == *"claude-daemon"* ]] || [[ "$pattern" == *"$DAEMON_ROOT"* ]]; then
        find "$pattern" -type f -mmin +5 -delete 2>/dev/null || true
    fi
}

# ============================================================================
# Library Initialization
# ============================================================================

# Master initialization function called by all libraries
init_library() {
    local lib_name="$1"

    # Ensure core directories exist
    init_daemon_directories

    # Initialize state file
    init_state_file

    # Initialize metric tracking
    ensure_metric_tracking

    # Initialize logs
    init_activity_logs

    # Initialize trigger configurations
    init_circadian_config
    init_emotional_config
    init_chaos_config

    # Initialize persona health if this is the health library
    if [ "$lib_name" = "persona-health" ]; then
        init_persona_health
    fi

    return 0
}

# ============================================================================
# Utility Functions
# ============================================================================

# Get or create JSON structure at path
get_or_init_json() {
    local file="$1"
    local path="$2"
    local default="${3:-{}}"

    if [ ! -f "$file" ]; then
        echo "$default" > "$file"
    fi

    jq "$path // $default" "$file" 2>/dev/null || echo "$default"
}

# Update JSON structure atomically
update_json_atomic() {
    local file="$1"
    local jq_expr="$2"
    local temp_file

    temp_file=$(mktemp)
    trap "rm -f '$temp_file'" RETURN

    if jq "$jq_expr" "$file" > "$temp_file" 2>/dev/null; then
        mv "$temp_file" "$file"
        return 0
    else
        rm -f "$temp_file"
        return 1
    fi
}

# ============================================================================
# Validation Functions
# ============================================================================

# Validate core files exist and are readable
validate_daemon_setup() {
    local errors=()

    [ ! -d "$DAEMON_ROOT" ] && errors+=("Daemon root not found: $DAEMON_ROOT")
    [ ! -f "$STATE_FILE" ] && errors+=("State file not found: $STATE_FILE")
    [ ! -d "$METRICS_DIR" ] && errors+=("Metrics dir not found: $METRICS_DIR")
    [ ! -d "$LOGS_DIR" ] && errors+=("Logs dir not found: $LOGS_DIR")

    if [ ${#errors[@]} -gt 0 ]; then
        for error in "${errors[@]}"; do
            echo "ERROR: $error" >&2
        done
        return 1
    fi

    return 0
}

# Ensure required functions are available
require_functions() {
    local required_funcs=("$@")

    for func in "${required_funcs[@]}"; do
        if ! declare -f "$func" >/dev/null 2>&1; then
            echo "ERROR: Required function not found: $func" >&2
            return 1
        fi
    done

    return 0
}

# ============================================================================
# Export Functions
# ============================================================================

export -f init_daemon_directories
export -f init_metric_file
export -f ensure_metric_tracking
export -f init_persona_health
export -f init_activity_logs
export -f init_state_file
export -f init_circadian_config
export -f init_emotional_config
export -f init_chaos_config
export -f setup_cleanup_trap
export -f cleanup_temp_files
export -f init_library
export -f get_or_init_json
export -f update_json_atomic
export -f validate_daemon_setup
export -f require_functions
