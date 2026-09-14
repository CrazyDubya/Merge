#!/bin/bash
#
# Performance Metrics Collector
# Tracks task execution metrics to establish baselines and detect anomalies
#
# Usage:
#   source "${DAEMON_ROOT}/lib/performance-metrics.sh"
#   record_task_metrics "$persona" "$task" "$duration" "$success" "$validation_passed"
#

set -euo pipefail

# Initialize metrics tracking files if needed
init_performance_metrics() {
    mkdir -p "$DAEMON_ROOT/metrics"
    touch "$DAEMON_ROOT/metrics/task-execution.log"

    # Initialize baselines if not present
    if [ ! -f "$DAEMON_ROOT/metrics/baselines.json" ]; then
        cat > "$DAEMON_ROOT/metrics/baselines.json" <<'BASELINE_EOF'
{
  "description": "Performance baselines calculated from historical task execution data",
  "last_updated": null,
  "data_window_days": 7,
  "minimum_samples_required": 10,
  "metrics": {
    "task_duration": {
      "overall": {
        "mean": null,
        "stddev": null,
        "p50": null,
        "p95": null,
        "samples": 0
      },
      "by_persona": {}
    },
    "success_rate": {
      "overall": {
        "rate_24h": null,
        "rate_7d": null,
        "rate_lifetime": null,
        "samples": 0
      },
      "by_persona": {}
    },
    "validation_failure_rate": {
      "overall": {
        "rate_24h": null,
        "rate_7d": null,
        "samples": 0
      },
      "by_persona": {}
    }
  }
}
BASELINE_EOF
    fi
}

# Record task execution metrics to JSONL log
# Args: persona, task (clean), duration_seconds, success (0/1), validation_passed (0/1)
record_task_metrics() {
    local persona="$1"
    local task="$2"
    local duration_seconds="$3"
    local success="${4:-0}"  # 0=success, 1=failure
    local validation_passed="${5:-0}"  # 0=passed, 1=failed

    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # Construct metrics entry
    local entry=$(cat <<EOF
{"timestamp":"$timestamp","persona":"$persona","task":"$task","duration_seconds":$duration_seconds,"success":$success,"validation_passed":$validation_passed}
EOF
)

    # Append to metrics log using atomic append (if available)
    if declare -f atomic_append > /dev/null 2>&1; then
        atomic_append "$DAEMON_ROOT/metrics/task-execution.log" "$entry"
    else
        echo "$entry" >> "$DAEMON_ROOT/metrics/task-execution.log"
    fi

    if declare -f log > /dev/null 2>&1; then
        log "DEBUG" "Recorded metrics: $persona | ${task:0:40} | ${duration_seconds}s | success=$success"
    fi
}

# Get task execution metrics from log (last N entries or time window)
get_task_metrics() {
    local limit="${1:-1000}"  # Last N entries

    if [ -f "$DAEMON_ROOT/metrics/task-execution.log" ]; then
        tail -"$limit" "$DAEMON_ROOT/metrics/task-execution.log" 2>/dev/null || echo ""
    fi
}

# Calculate baseline metrics for a given time window (in days)
calculate_baseline() {
    local window_days="${1:-7}"  # Default 7-day window
    local min_samples="${2:-10}"  # Minimum samples required

    if declare -f log > /dev/null 2>&1; then
        log "INFO" "Calculating baselines (window: ${window_days}d, min samples: $min_samples)"
    fi

    local since=$(date -d "-${window_days} days" -u +%Y-%m-%dT%H:%M:%SZ)
    local metrics_file="$DAEMON_ROOT/metrics/task-execution.log"

    if [ ! -f "$metrics_file" ]; then
        if declare -f log > /dev/null 2>&1; then
            log "WARN" "No metrics file yet, cannot calculate baselines"
        fi
        return 1
    fi

    # Extract metrics within time window
    local window_data=$(jq -s --arg since "$since" \
        '[.[] | select(.timestamp > $since)]' "$metrics_file" 2>/dev/null || echo "[]")

    # Calculate statistics using jq
    local stats=$(jq -n --arg since "$since" --slurpfile data <(echo "$window_data") \
        '
        $data[0] as $metrics |
        {
          "last_updated": (now | todate),
          "data_window_days": 7,
          "metrics": {
            "task_duration": {
              "overall": {
                "mean": ($metrics | map(.duration_seconds) | add / length),
                "p50": ($metrics | map(.duration_seconds) | sort | .[(length / 2 | floor)]),
                "p95": ($metrics | map(.duration_seconds) | sort | .[(length * 0.95 | floor)]),
                "samples": ($metrics | length)
              }
            },
            "success_rate": {
              "overall": {
                "rate_7d": (($metrics | map(select(.success == 0)) | length) / ($metrics | length) * 100),
                "samples": ($metrics | length)
              }
            },
            "validation_failure_rate": {
              "overall": {
                "rate_7d": (($metrics | map(select(.validation_passed == 1)) | length) / ($metrics | length) * 100),
                "samples": ($metrics | length)
              }
            }
          }
        }
        ' 2>/dev/null || echo "{}")

    # Update baselines.json
    if [ -n "$stats" ] && [ "$stats" != "{}" ]; then
        local temp_file=$(mktemp)
        trap "rm -f '$temp_file'" RETURN
        jq --argjson stats "$stats" '.last_updated = $stats.last_updated |
                                      .metrics.task_duration.overall = $stats.metrics.task_duration.overall |
                                      .metrics.success_rate.overall = $stats.metrics.success_rate.overall |
                                      .metrics.validation_failure_rate.overall = $stats.metrics.validation_failure_rate.overall' \
            "$DAEMON_ROOT/metrics/baselines.json" > "$temp_file"
        mv "$temp_file" "$DAEMON_ROOT/metrics/baselines.json"

        if declare -f log > /dev/null 2>&1; then
            log "INFO" "Baselines updated: $(jq -r '.metrics.task_duration.overall.samples' "$DAEMON_ROOT/metrics/baselines.json") samples"
        fi
        return 0
    else
        if declare -f log > /dev/null 2>&1; then
            log "WARN" "Could not calculate baselines - insufficient data"
        fi
        return 1
    fi
}

# Detect anomalies against baseline
# Returns: 0=normal, 1=warning, 2=critical
detect_anomaly() {
    local metric_type="$1"  # duration, success_rate, validation_rate
    local current_value="$2"

    local baselines="$DAEMON_ROOT/metrics/baselines.json"

    if [ ! -f "$baselines" ]; then
        return 0  # No baseline yet, can't detect anomalies
    fi

    # Get baseline and stddev
    local baseline=$(jq -r --arg type "$metric_type" \
        '.metrics[$type].overall.mean // 0' "$baselines")

    # Simple anomaly detection: 2σ threshold
    # For now, just check if significantly different from baseline
    if [ -z "$baseline" ] || [ "$baseline" = "null" ] || [ "$baseline" = "0" ]; then
        return 0  # No baseline data
    fi

    # If current value is 3x baseline, flag as warning
    if [ $(echo "$current_value > ($baseline * 3)" | bc) -eq 1 ]; then
        if declare -f log > /dev/null 2>&1; then
            log "WARN" "Anomaly detected: $metric_type = $current_value (baseline: $baseline)"
        fi
        return 1  # Warning
    fi

    # If current value is 5x baseline, flag as critical
    if [ $(echo "$current_value > ($baseline * 5)" | bc) -eq 1 ]; then
        if declare -f log > /dev/null 2>&1; then
            log "ERROR" "Critical anomaly detected: $metric_type = $current_value (baseline: $baseline)"
        fi
        return 2  # Critical
    fi

    return 0  # Normal
}

# Get baseline value for a metric
get_baseline() {
    local metric_type="$1"
    local dimension="${2:-overall}"  # overall, or persona name

    local baselines="$DAEMON_ROOT/metrics/baselines.json"

    if [ ! -f "$baselines" ]; then
        echo "0"
        return
    fi

    jq -r --arg type "$metric_type" --arg dim "$dimension" \
        '.metrics[$type][$dim].mean // .metrics[$type][$dim].rate_7d // 0' "$baselines" 2>/dev/null || echo "0"
}

# Get standard deviation for a metric
get_stddev() {
    local metric_type="$1"

    local baselines="$DAEMON_ROOT/metrics/baselines.json"

    if [ ! -f "$baselines" ]; then
        echo "0"
        return
    fi

    jq -r --arg type "$metric_type" \
        '.metrics[$type].overall.stddev // 0' "$baselines" 2>/dev/null || echo "0"
}

# Get success rate for last N hours
get_success_rate() {
    local hours="${1:-24}"

    local metrics_file="$DAEMON_ROOT/metrics/task-execution.log"

    if [ ! -f "$metrics_file" ]; then
        echo "0"
        return
    fi

    local since=$(date -d "-${hours} hours" -u +%Y-%m-%dT%H:%M:%SZ)

    jq -s --arg since "$since" \
        'map(select(.timestamp > $since)) |
         if length == 0 then 0
         else (map(select(.success == 0)) | length) as $success |
              ($success / length * 100 | floor)
         end' "$metrics_file" 2>/dev/null || echo "0"
}

# Get average task duration for last N hours
get_avg_duration() {
    local hours="${1:-24}"

    local metrics_file="$DAEMON_ROOT/metrics/task-execution.log"

    if [ ! -f "$metrics_file" ]; then
        echo "0"
        return
    fi

    local since=$(date -d "-${hours} hours" -u +%Y-%m-%dT%H:%M:%SZ)

    jq -s --arg since "$since" \
        'map(select(.timestamp > $since) | .duration_seconds) |
         if length == 0 then 0
         else (add / length | round)
         end' "$metrics_file" 2>/dev/null || echo "0"
}

# Initialize on source
init_performance_metrics

export -f init_performance_metrics
export -f record_task_metrics
export -f get_task_metrics
export -f calculate_baseline
export -f detect_anomaly
export -f get_baseline
export -f get_stddev
export -f get_success_rate
export -f get_avg_duration
