#!/bin/bash
#
# Trigger Optimization Baseline Tracking Script
#
# PURPOSE: Collect daily metrics on persona distribution and system behavior
#          to establish baseline before trigger A/B test
#
# USAGE: ./track-trigger-baseline.sh [--date YYYY-MM-DD] [--output FILE]
#
# OUTPUT: JSON Lines format with one entry per day containing all metrics
#
# METRICS TRACKED:
# - Persona activation counts and rates
# - Activation gaps (time between persona activations)
# - Success/failure task distribution
# - Task completion time statistics
# - Security review frequency
# - Distribution fairness (Gini coefficient)
#
# CREATED: 2025-11-03 by Experimenter
# PURPOSE: Implementation follow-through for trigger optimization A/B test

set -euo pipefail

# Configuration
DAEMON_ROOT="${DAEMON_ROOT:-${HOME}/.claude/daemon}"
TIMELINE_FILE="${DAEMON_ROOT}/memory/persona-timeline.jsonl"
OUTPUT_DIR="${DAEMON_ROOT}/experiments"
DEFAULT_OUTPUT="${OUTPUT_DIR}/trigger-baseline-$(date +%Y%m%d).jsonl"

# Parse arguments
TARGET_DATE="${1:-$(date +%Y-%m-%d)}"
OUTPUT_FILE="${2:-$DEFAULT_OUTPUT}"

# Validate timeline file exists
if [[ ! -f "$TIMELINE_FILE" ]]; then
    echo "ERROR: Timeline file not found: $TIMELINE_FILE" >&2
    exit 1
fi

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

echo "[$(date +%Y-%m-%dT%H:%M:%SZ)] Starting baseline tracking for date: $TARGET_DATE"

# Function: Extract well-formed JSON entries from timeline
extract_timeline_entries() {
    local date_filter="$1"

    # Extract entries for target date (or all if date is empty)
    if [[ -n "$date_filter" ]]; then
        grep '^{"timestamp"' "$TIMELINE_FILE" | \
            grep "\"timestamp\":\"${date_filter}" || true
    else
        grep '^{"timestamp"' "$TIMELINE_FILE" || true
    fi
}

# Function: Count persona activations
calculate_persona_activations() {
    local entries="$1"

    echo "$entries" | \
        jq -r 'select(.event == "personality_switch") | .persona' | \
        sort | uniq -c | \
        awk '{print "{\"persona\": \"" $2 "\", \"count\": " $1 "}"}'
}

# Function: Calculate activation gaps (time between activations per persona)
calculate_activation_gaps() {
    local entries="$1"

    # For each persona, find gaps between consecutive activations
    echo "$entries" | \
        jq -r 'select(.event == "personality_switch") | "\(.timestamp)|\(.persona)"' | \
        awk -F'|' '
        function iso8601_to_epoch(ts,    year, month, day, hour, min, sec, days) {
            # Parse ISO8601: YYYY-MM-DDTHH:MM:SSZ
            match(ts, /^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})Z?$/, parts)
            year = parts[1]; month = parts[2]; day = parts[3]
            hour = parts[4]; min = parts[5]; sec = parts[6]

            # Calculate days since epoch (1970-01-01)
            days = (year - 1970) * 365 + int((year - 1969) / 4) - int((year - 1901) / 100) + int((year - 1601) / 400)

            # Add days for months (approximate, good enough for gap calculations)
            month_days = "0 31 59 90 120 151 181 212 243 273 304 334"
            split(month_days, md, " ")
            days += md[month] + day - 1

            # Leap year adjustment for current year
            if (month > 2 && (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0))) days++

            # Convert to seconds
            return days * 86400 + hour * 3600 + min * 60 + sec
        }
        {
            persona = $2
            timestamp = $1

            # Convert timestamp to seconds since epoch (no subprocess!)
            seconds = iso8601_to_epoch(timestamp)

            # Calculate gap from previous activation of same persona
            if (persona in last_seen) {
                gap = seconds - last_seen[persona]
                gaps[persona] = (gaps[persona] ? gaps[persona] "," gap : gap)
            }

            last_seen[persona] = seconds
        }
        END {
            for (p in gaps) {
                split(gaps[p], gap_array, ",")

                # Calculate max gap
                max_gap = 0
                sum = 0
                count = 0
                for (i in gap_array) {
                    gap = gap_array[i]
                    if (gap > max_gap) max_gap = gap
                    sum += gap
                    count++
                }

                mean_gap = (count > 0 ? sum / count : 0)
                max_gap_hours = max_gap / 3600
                mean_gap_hours = mean_gap / 3600

                printf "{\"persona\": \"%s\", \"max_gap_hours\": %.2f, \"mean_gap_hours\": %.2f, \"gap_count\": %d}\n",
                       p, max_gap_hours, mean_gap_hours, count
            }
        }'
}

# Function: Calculate task success/failure distribution
calculate_task_distribution() {
    local entries="$1"

    echo "$entries" | \
        jq -sc '
            [.[] | select(.event == "task_complete")] |
            group_by(.persona) |
            map({
                persona: .[0].persona,
                total: length,
                successes: [.[] | select(.details | contains("(success)") or contains("success"))] | length,
                failures: [.[] | select(.details | contains("(failed)") or contains("failure"))] | length
            }) |
            map(. + {
                success_rate: (if .total > 0 then (.successes / .total) else 0 end)
            })
        '
}

# Function: Calculate task completion time statistics
calculate_completion_times() {
    local entries="$1"

    # Match task_start with task_complete events
    echo "$entries" | \
        jq -r 'select(.event == "task_start" or .event == "task_complete") | "\(.timestamp)|\(.event)|\(.persona)|\(.details | gsub("\n"; " ") | .[0:100])"' | \
        awk -F'|' '
        function iso8601_to_epoch(ts,    year, month, day, hour, min, sec, days) {
            # Parse ISO8601: YYYY-MM-DDTHH:MM:SSZ
            match(ts, /^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})Z?$/, parts)
            year = parts[1]; month = parts[2]; day = parts[3]
            hour = parts[4]; min = parts[5]; sec = parts[6]

            # Calculate days since epoch (1970-01-01)
            days = (year - 1970) * 365 + int((year - 1969) / 4) - int((year - 1901) / 100) + int((year - 1601) / 400)

            # Add days for months (approximate, good enough for gap calculations)
            month_days = "0 31 59 90 120 151 181 212 243 273 304 334"
            split(month_days, md, " ")
            days += md[month] + day - 1

            # Leap year adjustment for current year
            if (month > 2 && (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0))) days++

            # Convert to seconds
            return days * 86400 + hour * 3600 + min * 60 + sec
        }
        {
            timestamp = $1
            event = $2
            persona = $3
            task = $4

            # Convert to seconds (no subprocess!)
            seconds = iso8601_to_epoch(timestamp)

            # Track task starts
            if (event == "task_start") {
                task_key = persona ":" task
                task_starts[task_key] = seconds
            }

            # Calculate completion time
            if (event == "task_complete") {
                task_key = persona ":" task
                if (task_key in task_starts) {
                    duration = seconds - task_starts[task_key]
                    durations[++duration_count] = duration
                    persona_durations[persona][++persona_counts[persona]] = duration
                    delete task_starts[task_key]
                }
            }
        }
        END {
            # Overall statistics
            if (duration_count > 0) {
                # Sort durations for median
                for (i = 1; i <= duration_count; i++) {
                    for (j = i + 1; j <= duration_count; j++) {
                        if (durations[i] > durations[j]) {
                            temp = durations[i]
                            durations[i] = durations[j]
                            durations[j] = temp
                        }
                    }
                }

                # Calculate mean
                sum = 0
                for (i = 1; i <= duration_count; i++) {
                    sum += durations[i]
                }
                mean = sum / duration_count

                # Calculate median
                if (duration_count % 2 == 1) {
                    median = durations[int(duration_count / 2) + 1]
                } else {
                    median = (durations[duration_count / 2] + durations[duration_count / 2 + 1]) / 2
                }

                printf "{\"task_count\": %d, \"mean_seconds\": %.1f, \"median_seconds\": %.1f, \"mean_minutes\": %.1f, \"median_minutes\": %.1f}\n",
                       duration_count, mean, median, mean/60, median/60
            } else {
                printf "{\"task_count\": 0, \"mean_seconds\": 0, \"median_seconds\": 0, \"mean_minutes\": 0, \"median_minutes\": 0}\n"
            }
        }'
}

# Function: Calculate Gini coefficient for distribution fairness
calculate_gini_coefficient() {
    local activation_counts="$1"

    echo "$activation_counts" | \
        jq -s '
            map(.count) |
            sort |
            . as $counts |
            if (length == 0) then 0 else
                (add / length) as $mean |
                if $mean == 0 then 0 else
                    (
                        [
                            range(length) as $i |
                            range(length) as $j |
                            (($counts[$i] - $counts[$j]) | if . < 0 then -. else . end)
                        ] | add
                    ) / (2 * length * length * $mean)
                end
            end
        '
}

# Function: Count security review events
calculate_security_reviews() {
    local entries="$1"

    echo "$entries" | \
        jq -sc '
            [.[] | select(
                (.event == "task_start" and (.details | contains("AUDITOR") or contains("security"))) or
                (.event == "personality_switch" and .persona == "auditor")
            )] |
            {
                auditor_activations: [.[] | select(.event == "personality_switch" and .persona == "auditor")] | length,
                security_tasks: [.[] | select(.event == "task_start" and (.details | contains("security") or contains("AUDITOR")))] | length,
                total_security_events: length
            }
        '
}

# Main execution
echo "[$(date +%Y-%m-%dT%H:%M:%SZ)] Extracting timeline entries for $TARGET_DATE..."
TIMELINE_ENTRIES=$(extract_timeline_entries "$TARGET_DATE")

ENTRY_COUNT=$(echo "$TIMELINE_ENTRIES" | wc -l)
echo "[$(date +%Y-%m-%dT%H:%M:%SZ)] Found $ENTRY_COUNT timeline entries"

if [[ $ENTRY_COUNT -eq 0 ]]; then
    echo "WARNING: No entries found for $TARGET_DATE" >&2
    exit 1
fi

echo "[$(date +%Y-%m-%dT%H:%M:%SZ)] Calculating persona activations..."
ACTIVATIONS=$(calculate_persona_activations "$TIMELINE_ENTRIES")

echo "[$(date +%Y-%m-%dT%H:%M:%SZ)] Calculating activation gaps..."
GAPS=$(calculate_activation_gaps "$TIMELINE_ENTRIES")

echo "[$(date +%Y-%m-%dT%H:%M:%SZ)] Calculating task distribution..."
TASK_DIST=$(calculate_task_distribution "$TIMELINE_ENTRIES")

echo "[$(date +%Y-%m-%dT%H:%M:%SZ)] Calculating completion times..."
COMPLETION_TIMES=$(calculate_completion_times "$TIMELINE_ENTRIES")

echo "[$(date +%Y-%m-%dT%H:%M:%SZ)] Calculating Gini coefficient..."
GINI=$(calculate_gini_coefficient "$ACTIVATIONS")

echo "[$(date +%Y-%m-%dT%H:%M:%SZ)] Calculating security review frequency..."
SECURITY=$(calculate_security_reviews "$TIMELINE_ENTRIES")

# Combine all metrics into single JSON object
echo "[$(date +%Y-%m-%dT%H:%M:%SZ)] Assembling baseline metrics..."

# Provide defaults for empty values to prevent jq errors
ACTIVATIONS_JSON="[$(echo "$ACTIVATIONS" | paste -sd, || echo '')]"
[ "$ACTIVATIONS_JSON" = "[]" ] && ACTIVATIONS_JSON="[]"

GAPS_JSON="[$(echo "$GAPS" | paste -sd, || echo '')]"
[ "$GAPS_JSON" = "[]" ] && GAPS_JSON="[]"

# Use variables directly - functions already output valid JSON
TASK_DIST_JSON="$TASK_DIST"
COMPLETION_TIMES_JSON="$COMPLETION_TIMES"
GINI_JSON="$GINI"
SECURITY_JSON="$SECURITY"

# Provide defaults only if truly empty
[ -z "$TASK_DIST_JSON" ] && TASK_DIST_JSON="[]"
[ -z "$COMPLETION_TIMES_JSON" ] && COMPLETION_TIMES_JSON='{"task_count":0,"mean_seconds":0,"median_seconds":0,"mean_minutes":0,"median_minutes":0}'
[ -z "$GINI_JSON" ] && GINI_JSON="0"
[ -z "$SECURITY_JSON" ] && SECURITY_JSON='{"auditor_activations":0,"security_tasks":0,"total_security_events":0}'

BASELINE_DATA=$(jq -n \
    --arg date "$TARGET_DATE" \
    --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --argjson activations "$ACTIVATIONS_JSON" \
    --argjson gaps "$GAPS_JSON" \
    --argjson task_dist "$TASK_DIST_JSON" \
    --argjson completion "$COMPLETION_TIMES_JSON" \
    --argjson gini "$GINI_JSON" \
    --argjson security "$SECURITY_JSON" \
    '{
        collection_date: $date,
        collection_timestamp: $timestamp,
        baseline_version: "1.0",
        metrics: {
            persona_activations: $activations,
            activation_gaps: $gaps,
            task_distribution: $task_dist,
            completion_times: $completion,
            gini_coefficient: $gini,
            security_reviews: $security
        }
    }')

# Write to output file
echo "$BASELINE_DATA" >> "$OUTPUT_FILE"

echo "[$(date +%Y-%m-%dT%H:%M:%SZ)] ✅ Baseline data written to: $OUTPUT_FILE"
echo ""
echo "Summary:"
echo "$BASELINE_DATA" | jq '{
    date: .collection_date,
    personas_active: (.metrics.persona_activations | length),
    total_activations: (.metrics.persona_activations | map(.count) | add),
    gini_coefficient: .metrics.gini_coefficient,
    auditor_activations: .metrics.security_reviews.auditor_activations
}'

exit 0
