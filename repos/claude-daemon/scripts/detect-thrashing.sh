#!/bin/bash
#
# Thrashing Detection Script
#
# Monitors persona switch frequency and kills daemon if thrashing detected.
# Thrashing = >100 persona switches per hour (catastrophic positive feedback loop)
#
# CONTEXT: 2025-11-06 thrashing incident: 88K switches, 727 switches/min peak,
#          system non-functional for 36 hours. Watchdog couldn't break loop.
#
# SOLUTION: Automated detection + intervention (kill daemon, alert human)
#
# USAGE:
#   ./detect-thrashing.sh                    # Check current state
#   crontab: */15 * * * * /path/to/detect-thrashing.sh  # Run every 15 min
#

set -euo pipefail

# Configuration
DAEMON_ROOT="${HOME}/.claude/daemon"
SWITCH_HISTORY="${DAEMON_ROOT}/metrics/switch-history.jsonl"
INBOX_DIR="${DAEMON_ROOT}/inbox/human/unread"
METRICS_DIR="${DAEMON_ROOT}/metrics"
ACTIVITY_LOG="${DAEMON_ROOT}/logs/activity.log"

# Thresholds
THRASHING_THRESHOLD=100  # switches per hour = catastrophic
WARNING_THRESHOLD=50     # switches per hour = concerning
LOOKBACK_MINUTES=60      # analyze last hour

# Ensure required directories exist
mkdir -p "$INBOX_DIR" "$METRICS_DIR"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$ACTIVITY_LOG"
}

count_recent_switches() {
    # Count persona switches in last N minutes from switch history
    local minutes="$1"
    local cutoff_time
    cutoff_time=$(date -u -d "$minutes minutes ago" +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -u -v-${minutes}M +%Y-%m-%dT%H:%M:%S)

    if [ ! -f "$SWITCH_HISTORY" ]; then
        echo "0"
        return
    fi

    # Count switches after cutoff time
    # Switch history format: {"timestamp":"2025-11-06T14:00:02Z",...}
    local count
    count=$(awk -v cutoff="$cutoff_time" '
        {
            if (match($0, /"timestamp":"([^"]+)"/, ts)) {
                if (ts[1] >= cutoff) count++
            }
        }
        END { print count+0 }
    ' "$SWITCH_HISTORY" 2>/dev/null || echo "0")
    echo "$count"
}

get_daemon_pid() {
    # Find daemon.sh process
    pgrep -f "daemon.sh$" | head -1
}

kill_daemon() {
    local pid="$1"
    log "CRITICAL: Killing daemon process $pid due to thrashing detection"
    kill "$pid"
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        log "CRITICAL: Daemon didn't die, using SIGKILL"
        kill -9 "$pid"
    fi
}

alert_human() {
    local switches_per_hour="$1"
    local alert_file="${INBOX_DIR}/thrashing-alert-$(date +%Y%m%d-%H%M%S).md"

    cat > "$alert_file" <<EOF
# 🚨 CRITICAL ALERT: Persona Switch Thrashing Detected

**Priority:** CRITICAL
**Time:** $(date -u +%Y-%m-%dT%H:%M:%SZ)
**Status:** Daemon killed automatically

## Detection

**Switches in last hour:** $switches_per_hour
**Threshold:** $THRASHING_THRESHOLD switches/hour
**Verdict:** THRASHING (positive feedback loop)

## What Happened

The system detected catastrophic persona switch thrashing - a positive feedback loop
where personas switch rapidly without making progress. This is similar to the
2025-11-06 incident that caused 88,161 switches and system paralysis.

## Automatic Response

1. ✅ Daemon process killed (prevents further damage)
2. ✅ Human alerted (this message)
3. ⏳ Awaiting human investigation

## What You Should Do

1. **Review recent logs:**
   - \`tail -200 logs/activity.log\` (switch patterns)
   - \`tail -100 logs/state-audit.jsonl\` (audit trail)
   - \`jq -r '.reason' metrics/switch-history.jsonl | tail -100 | sort | uniq -c\` (trigger frequency)

2. **Identify root cause:**
   - Emotional trigger without cooldown? (check triggers/emotional.json)
   - Activation floor bug? (check daemon.sh:382-449)
   - Task-based trigger loop? (check task assignments)
   - Chaos probability too high? (check triggers/chaos-config.json)

3. **Fix before restart:**
   - Apply cooldown mechanism if missing
   - Reset emotional state if corrupted
   - Disable problematic trigger temporarily
   - Review and test fix

4. **Restart daemon:**
   - \`tmux new-session -d -s claude-daemon ./daemon.sh\`
   - Monitor for 15-30 minutes to verify stability

## Historical Context

**Previous incident:** 2025-11-06T12:59:00Z
- **Switches:** 88,161 total (727 in one minute)
- **Duration:** 36 hours
- **Root cause:** Emotional frustration trigger with no cooldown
- **Fix:** 5-minute cooldown added to emotional triggers (daemon.sh:233-261)
- **Incident report:** docs/incident-thrashing-bug-20251106.md

## Detection Metrics

**Current switch rate:** $switches_per_hour switches/hour
**Warning threshold:** $WARNING_THRESHOLD switches/hour
**Critical threshold:** $THRASHING_THRESHOLD switches/hour

---

**This alert was generated automatically by scripts/detect-thrashing.sh**
EOF

    log "CRITICAL: Human alert created at $alert_file"
}

log_detection_event() {
    local switches_per_hour="$1"
    local action="$2"
    local event_file="${METRICS_DIR}/thrashing-detections.jsonl"

    echo "{\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"switches_per_hour\":$switches_per_hour,\"threshold\":$THRASHING_THRESHOLD,\"action\":\"$action\"}" >> "$event_file"
}

# Main detection logic
main() {
    log "INFO: Starting thrashing detection check (threshold: $THRASHING_THRESHOLD/hour)"

    # Count recent switches
    local switches_last_hour
    switches_last_hour=$(count_recent_switches "$LOOKBACK_MINUTES")

    log "INFO: Detected $switches_last_hour switches in last $LOOKBACK_MINUTES minutes"

    # Check thresholds
    if [ "$switches_last_hour" -ge "$THRASHING_THRESHOLD" ]; then
        log "CRITICAL: THRASHING DETECTED! $switches_last_hour switches/hour >= $THRASHING_THRESHOLD threshold"

        # Find and kill daemon
        local daemon_pid
        daemon_pid=$(get_daemon_pid)

        if [ -n "$daemon_pid" ]; then
            kill_daemon "$daemon_pid"
            alert_human "$switches_last_hour"
            log_detection_event "$switches_last_hour" "daemon_killed"
            echo "CRITICAL: Thrashing detected and daemon killed. Check $INBOX_DIR for alert."
            exit 2
        else
            log "WARNING: Thrashing detected but daemon not running"
            alert_human "$switches_last_hour"
            log_detection_event "$switches_last_hour" "daemon_not_running"
            echo "WARNING: Thrashing detected but daemon already stopped."
            exit 1
        fi

    elif [ "$switches_last_hour" -ge "$WARNING_THRESHOLD" ]; then
        log "WARNING: High switch rate: $switches_last_hour switches/hour (warning threshold: $WARNING_THRESHOLD)"
        log_detection_event "$switches_last_hour" "warning_logged"
        echo "WARNING: High switch rate detected ($switches_last_hour/hour). Monitor closely."
        exit 0

    else
        log "INFO: Switch rate normal: $switches_last_hour switches/hour"
        echo "OK: Switch rate normal ($switches_last_hour/hour)"
        exit 0
    fi
}

main "$@"
