#!/usr/bin/env bash
# Leonardo.ai Integration Health Monitor
# Checks cost, error rates, disk usage, and performance
# Can be run manually or via cron for automated monitoring
#
# Usage:
#   ./monitoring/check-health.sh                    # Run all checks, show summary
#   ./monitoring/check-health.sh --alert            # Send alerts to daemon inbox if thresholds exceeded
#   ./monitoring/check-health.sh --cost-only        # Only check costs
#   ./monitoring/check-health.sh --errors-only      # Only check error rates
#   ./monitoring/check-health.sh --disk-only        # Only check disk usage
#   ./monitoring/check-health.sh --performance-only # Only check performance

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEGRATION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DAEMON_ROOT="${HOME}/.claude/daemon"

# Default thresholds (can be overridden via config file)
COST_DAILY_THRESHOLD=${LEONARDO_COST_DAILY_THRESHOLD:-10.00}  # $10/day default
COST_WEEKLY_THRESHOLD=${LEONARDO_COST_WEEKLY_THRESHOLD:-50.00} # $50/week default
ERROR_RATE_THRESHOLD=${LEONARDO_ERROR_RATE_THRESHOLD:-10}      # 10% error rate
DISK_WARNING_MB=${LEONARDO_DISK_WARNING_MB:-500}               # 500MB warning
DISK_CRITICAL_MB=${LEONARDO_DISK_CRITICAL_MB:-1000}            # 1GB critical
PERF_SLOW_SECONDS=${LEONARDO_PERF_SLOW_SECONDS:-30}            # 30s is slow

# Alert mode
ALERT_MODE=false
CHECK_ALL=true
CHECK_COST=false
CHECK_ERRORS=false
CHECK_DISK=false
CHECK_PERFORMANCE=false

# Parse arguments
for arg in "$@"; do
  case $arg in
    --alert)
      ALERT_MODE=true
      shift
      ;;
    --cost-only)
      CHECK_ALL=false
      CHECK_COST=true
      shift
      ;;
    --errors-only)
      CHECK_ALL=false
      CHECK_ERRORS=true
      shift
      ;;
    --disk-only)
      CHECK_ALL=false
      CHECK_DISK=true
      shift
      ;;
    --performance-only)
      CHECK_ALL=false
      CHECK_PERFORMANCE=true
      shift
      ;;
    --help|-h)
      echo "Leonardo.ai Health Monitor"
      echo ""
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --alert            Send alerts to daemon inbox if thresholds exceeded"
      echo "  --cost-only        Only check costs"
      echo "  --errors-only      Only check error rates"
      echo "  --disk-only        Only check disk usage"
      echo "  --performance-only Only check performance"
      echo "  --help             Show this help"
      echo ""
      echo "Environment variables (configure thresholds):"
      echo "  LEONARDO_COST_DAILY_THRESHOLD    Daily cost limit (default: \$10.00)"
      echo "  LEONARDO_COST_WEEKLY_THRESHOLD   Weekly cost limit (default: \$50.00)"
      echo "  LEONARDO_ERROR_RATE_THRESHOLD    Error rate % (default: 10)"
      echo "  LEONARDO_DISK_WARNING_MB         Disk warning MB (default: 500)"
      echo "  LEONARDO_DISK_CRITICAL_MB        Disk critical MB (default: 1000)"
      echo "  LEONARDO_PERF_SLOW_SECONDS       Slow generation threshold (default: 30)"
      echo ""
      exit 0
      ;;
  esac
done

# If CHECK_ALL, enable all checks
if [[ "$CHECK_ALL" = true ]]; then
  CHECK_COST=true
  CHECK_ERRORS=true
  CHECK_DISK=true
  CHECK_PERFORMANCE=true
fi

cd "$INTEGRATION_ROOT"

# Color output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Alert storage
declare -a ALERTS=()

log_info() {
  echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
  echo -e "${GREEN}[OK]${NC} $*"
}

log_warning() {
  echo -e "${YELLOW}[WARN]${NC} $*"
  if [[ "$ALERT_MODE" = true ]]; then
    ALERTS+=("⚠️  $*")
  fi
}

log_critical() {
  echo -e "${RED}[CRIT]${NC} $*"
  if [[ "$ALERT_MODE" = true ]]; then
    ALERTS+=("🔴 $*")
  fi
}

# ============================================================================
# COST MONITORING
# ============================================================================

check_costs() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "💰 Cost Monitoring"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if [[ ! -f cost-tracking.jsonl ]]; then
    log_info "No cost tracking data yet"
    return 0
  fi

  # Calculate costs for different time periods
  local today=$(date +%Y-%m-%d)
  local week_ago=$(date -d '7 days ago' +%Y-%m-%d 2>/dev/null || date -v-7d +%Y-%m-%d 2>/dev/null)

  # Daily cost
  local daily_cost=$(jq -r "select(.timestamp | startswith(\"$today\")) | .estimated_cost" cost-tracking.jsonl 2>/dev/null | \
    awk '{sum+=$1} END {printf "%.2f", sum}')
  daily_cost=${daily_cost:-0.00}

  # Weekly cost
  local weekly_cost=$(jq -r "select(.timestamp >= \"$week_ago\") | .estimated_cost" cost-tracking.jsonl 2>/dev/null | \
    awk '{sum+=$1} END {printf "%.2f", sum}')
  weekly_cost=${weekly_cost:-0.00}

  # Total cost
  local total_cost=$(jq -r '.estimated_cost' cost-tracking.jsonl 2>/dev/null | \
    awk '{sum+=$1} END {printf "%.2f", sum}')
  total_cost=${total_cost:-0.00}

  # Generation count
  local total_generations=$(wc -l < cost-tracking.jsonl 2>/dev/null || echo 0)

  echo "  Today:       \$$daily_cost"
  echo "  Last 7 days: \$$weekly_cost"
  echo "  All time:    \$$total_cost"
  echo "  Generations: $total_generations"
  echo ""

  # Check thresholds
  local daily_exceeded=$(awk -v daily="$daily_cost" -v threshold="$COST_DAILY_THRESHOLD" \
    'BEGIN {print (daily > threshold) ? 1 : 0}')
  local weekly_exceeded=$(awk -v weekly="$weekly_cost" -v threshold="$COST_WEEKLY_THRESHOLD" \
    'BEGIN {print (weekly > threshold) ? 1 : 0}')

  if [[ "$daily_exceeded" = "1" ]]; then
    log_critical "Daily cost (\$$daily_cost) exceeds threshold (\$$COST_DAILY_THRESHOLD)"
  elif [[ "$weekly_exceeded" = "1" ]]; then
    log_warning "Weekly cost (\$$weekly_cost) exceeds threshold (\$$COST_WEEKLY_THRESHOLD)"
  else
    log_success "Costs within acceptable limits"
  fi
}

# ============================================================================
# ERROR RATE MONITORING
# ============================================================================

check_error_rates() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "⚠️  Error Rate Monitoring"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if [[ ! -f generations.jsonl ]]; then
    log_info "No generation data yet"
    return 0
  fi

  local total_attempts=$(wc -l < generations.jsonl 2>/dev/null || echo 0)

  if [[ "$total_attempts" = "0" ]]; then
    log_info "No generation attempts yet"
    return 0
  fi

  # Count successes and failures
  # Success: has image_path field
  # Failure: has error field or missing image_path
  local successes=$(jq -r 'select(.image_path != null and .image_path != "") | .image_path' generations.jsonl 2>/dev/null | wc -l || echo 0)
  local failures=$((total_attempts - successes))

  local error_rate=0
  if [[ "$total_attempts" -gt 0 ]]; then
    error_rate=$(awk -v failures="$failures" -v total="$total_attempts" \
      'BEGIN {printf "%.1f", (failures / total) * 100}')
  fi

  echo "  Total attempts: $total_attempts"
  echo "  Successes:      $successes"
  echo "  Failures:       $failures"
  echo "  Error rate:     ${error_rate}%"
  echo ""

  local rate_exceeded=$(awk -v rate="$error_rate" -v threshold="$ERROR_RATE_THRESHOLD" \
    'BEGIN {print (rate > threshold) ? 1 : 0}')

  if [[ "$rate_exceeded" = "1" ]]; then
    log_critical "Error rate (${error_rate}%) exceeds threshold (${ERROR_RATE_THRESHOLD}%)"

    # Show recent errors
    echo ""
    echo "Recent errors (last 5):"
    jq -r 'select(.error != null) | "\(.timestamp): \(.error)"' generations.jsonl 2>/dev/null | tail -5 || echo "  (no error details available)"
  else
    log_success "Error rate within acceptable limits"
  fi
}

# ============================================================================
# DISK USAGE MONITORING
# ============================================================================

check_disk_usage() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "💾 Disk Usage Monitoring"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if [[ ! -d cache ]]; then
    log_info "No cache directory yet"
    return 0
  fi

  # Get disk usage in MB
  local cache_kb=$(du -sk cache 2>/dev/null | cut -f1)
  local cache_mb=$((cache_kb / 1024))

  # Count files
  local image_count=$(find cache -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" -o -name "*.webp" \) 2>/dev/null | wc -l || echo 0)

  echo "  Cache size:    ${cache_mb} MB"
  echo "  Image count:   $image_count"
  echo "  Location:      cache/"
  echo ""

  # Check thresholds
  if [[ "$cache_mb" -ge "$DISK_CRITICAL_MB" ]]; then
    log_critical "Disk usage (${cache_mb}MB) exceeds critical threshold (${DISK_CRITICAL_MB}MB)"
  elif [[ "$cache_mb" -ge "$DISK_WARNING_MB" ]]; then
    log_warning "Disk usage (${cache_mb}MB) exceeds warning threshold (${DISK_WARNING_MB}MB)"
  else
    log_success "Disk usage within acceptable limits"
  fi

  # Oldest and newest images
  if [[ "$image_count" -gt 0 ]]; then
    echo "  Oldest image:  $(find cache -type f -printf '%T+ %p\n' 2>/dev/null | sort | head -1 | awk '{print $1}' || echo 'unknown')"
    echo "  Newest image:  $(find cache -type f -printf '%T+ %p\n' 2>/dev/null | sort | tail -1 | awk '{print $1}' || echo 'unknown')"
  fi
}

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

check_performance() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "⚡ Performance Monitoring"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if [[ ! -f generations.jsonl ]]; then
    log_info "No generation data yet"
    return 0
  fi

  # Check if we have generation_time_seconds field
  local has_timing=$(jq -r 'select(.generation_time_seconds != null) | .generation_time_seconds' generations.jsonl 2>/dev/null | head -1)

  if [[ -z "$has_timing" ]]; then
    log_info "No timing data available (generation_time_seconds not logged)"
    return 0
  fi

  # Calculate average generation time
  local avg_time=$(jq -r 'select(.generation_time_seconds != null) | .generation_time_seconds' generations.jsonl 2>/dev/null | \
    awk '{sum+=$1; count++} END {if (count>0) printf "%.1f", sum/count; else print "0"}')

  # Find slowest generation
  local slowest=$(jq -r 'select(.generation_time_seconds != null) | .generation_time_seconds' generations.jsonl 2>/dev/null | \
    sort -n | tail -1)

  # Count slow generations
  local slow_count=$(jq -r "select(.generation_time_seconds > $PERF_SLOW_SECONDS) | .generation_time_seconds" generations.jsonl 2>/dev/null | wc -l || echo 0)

  echo "  Average time:   ${avg_time}s"
  echo "  Slowest:        ${slowest}s"
  echo "  Slow (>${PERF_SLOW_SECONDS}s): $slow_count"
  echo ""

  # Check if average is concerning
  local avg_slow=$(awk -v avg="$avg_time" -v threshold="$PERF_SLOW_SECONDS" \
    'BEGIN {print (avg > threshold) ? 1 : 0}')

  if [[ "$avg_slow" = "1" ]]; then
    log_warning "Average generation time (${avg_time}s) exceeds threshold (${PERF_SLOW_SECONDS}s)"
  else
    log_success "Performance within acceptable limits"
  fi
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏥 Leonardo.ai Integration Health Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Location: $INTEGRATION_ROOT"
echo "Time:     $(date '+%Y-%m-%d %H:%M:%S')"

if [[ "$CHECK_COST" = true ]]; then
  check_costs
fi

if [[ "$CHECK_ERRORS" = true ]]; then
  check_error_rates
fi

if [[ "$CHECK_DISK" = true ]]; then
  check_disk_usage
fi

if [[ "$CHECK_PERFORMANCE" = true ]]; then
  check_performance
fi

# ============================================================================
# ALERT DELIVERY
# ============================================================================

if [[ "$ALERT_MODE" = true ]] && [[ ${#ALERTS[@]} -gt 0 ]]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "📨 Sending Alerts"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Create alert message
  local alert_file="$DAEMON_ROOT/inbox/daemon/unread/leonardo-health-alert-$(date +%Y%m%d-%H%M%S).md"

  cat > "$alert_file" <<EOF
# Leonardo.ai Health Alert

**Date**: $(date '+%Y-%m-%d %H:%M:%S')
**Source**: Leonardo.ai Integration Monitoring
**Severity**: ${#ALERTS[@]} issue(s) detected

---

## Issues Detected

EOF

  for alert in "${ALERTS[@]}"; do
    echo "- $alert" >> "$alert_file"
  done

  cat >> "$alert_file" <<EOF

---

## Current Status Summary

**Daily cost**: \$$daily_cost (threshold: \$$COST_DAILY_THRESHOLD)
**Weekly cost**: \$$weekly_cost (threshold: \$$COST_WEEKLY_THRESHOLD)
**Error rate**: ${error_rate}% (threshold: ${ERROR_RATE_THRESHOLD}%)
**Disk usage**: ${cache_mb}MB (warning: ${DISK_WARNING_MB}MB, critical: ${DISK_CRITICAL_MB}MB)

---

## Recommended Actions

1. Review cost tracking: \`cat integrations/leonardo-ai/cost-tracking.jsonl\`
2. Check recent errors: \`cat integrations/leonardo-ai/generations.jsonl | jq 'select(.error != null)'\`
3. Review disk usage: \`du -sh integrations/leonardo-ai/cache/\`
4. Run health check: \`./integrations/leonardo-ai/monitoring/check-health.sh\`

---

**Generated by**: Leonardo.ai Health Monitor
**Report**: Run \`./integrations/leonardo-ai/monitoring/check-health.sh\` for full details
EOF

  log_success "Alert sent to daemon inbox: $(basename "$alert_file")"
else
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  if [[ ${#ALERTS[@]} -eq 0 ]]; then
    log_success "All checks passed! No issues detected."
  else
    log_info "Issues detected but --alert not enabled. Use --alert to send to daemon inbox."
  fi
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

echo ""

# Exit with error code if critical alerts exist
if [[ ${#ALERTS[@]} -gt 0 ]]; then
  exit 1
else
  exit 0
fi
