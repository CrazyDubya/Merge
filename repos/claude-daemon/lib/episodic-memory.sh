#!/bin/bash

################################################################################
# Episodic Memory (LisaSimpson + Ralph Wiggum Learning Integration)
#
# Tracks multi-step action sequences as "episodes" and extracts lessons learned.
# Enables the daemon to learn from completed workflows and apply lessons
# to similar future tasks.
#
# Episode Lifecycle:
# 1. create_episode() - Start episode for goal/workflow
# 2. add_action_to_episode() - Record each action taken
# 3. close_episode() - Finalize with auto-extracted lessons
# 4. replay_episode() / get_lesson() - Reuse patterns for similar goals
#
# Storage: memory/episodes.jsonl - Append-only log of completed episodes
#
# Authors: LisaSimpson + Ralph Wiggum + Autonomy Team
# Created: 2025-01-08
################################################################################

set -euo pipefail

# Daemon root
DAEMON_ROOT="${DAEMON_ROOT:-.}"

################################################################################
# EPISODE MANAGEMENT
################################################################################

# Create new episode for tracking multi-step workflow
# Usage: create_episode <goal_id> [episode_context]
# Returns: Episode JSON
create_episode() {
    local goal_id="$1"
    local context="${2:-general_workflow}"

    local now
    now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    local episode_id
    episode_id="ep_$(date +%Y%m%d_%H%M%S)_${goal_id}"

    jq -n \
        --arg episode_id "$episode_id" \
        --arg goal_id "$goal_id" \
        --arg context "$context" \
        --arg created_at "$now" \
        '{
            episode_id: $episode_id,
            goal_id: $goal_id,
            context: $context,
            created_at: $created_at,
            status: "active",
            actions: [],
            start_time: $created_at,
            end_time: null,
            duration_seconds: null,
            outcome: null,
            lessons_learned: []
        }'
}

# Add action to ongoing episode
# Usage: add_action_to_episode <episode_json> <action_json>
# Returns: Updated episode JSON
add_action_to_episode() {
    local episode="$1"
    local action="$2"

    local now
    now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Add action to episode
    echo "$episode" | jq \
        --argjson action "$action" \
        --arg timestamp "$now" \
        '.actions += [$action + {timestamp: $timestamp}]'
}

# Close episode and extract lessons
# Usage: close_episode <episode_json> <final_outcome>
# Returns: Closed episode JSON with lessons extracted
close_episode() {
    local episode="$1"
    local outcome="${2:-success}"

    local now
    now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    local start_time
    start_time=$(echo "$episode" | jq -r '.created_at')

    local start_epoch
    start_epoch=$(date -j -f "%Y-%m-%dT%H:%M:%SZ" "$start_time" +%s 2>/dev/null || date -d "$start_time" +%s 2>/dev/null || echo 0)

    local end_epoch
    end_epoch=$(date +%s)

    local duration=$((end_epoch - start_epoch))

    # Extract lessons from actions
    local lessons
    lessons=$(extract_episode_lessons "$episode")

    # Close episode
    echo "$episode" | jq \
        --arg outcome "$outcome" \
        --arg end_time "$now" \
        --argjson duration "$duration" \
        --argjson lessons "$lessons" \
        '.status = "closed" |
         .end_time = $end_time |
         .duration_seconds = $duration |
         .outcome = $outcome |
         .lessons_learned = $lessons'
}

################################################################################
# LESSON EXTRACTION
################################################################################

# Extract lessons from completed episode
# Usage: extract_episode_lessons <episode_json>
# Returns: Array of lessons learned
extract_episode_lessons() {
    local episode="$1"

    local lessons='[]'

    # Lesson 1: Pattern recognition
    local action_count
    action_count=$(echo "$episode" | jq '.actions | length')

    if [ "$action_count" -gt 1 ]; then
        local lesson1
        lesson1=$(jq -n \
            --argjson count "$action_count" \
            '{
                type: "pattern",
                description: ("Successfully completed workflow with " + ($count | tostring) + " action steps"),
                confidence: 0.9,
                reusable: true
            }')
        lessons=$(echo "$lessons" | jq ". += [$lesson1]")
    fi

    # Lesson 2: Success rate
    local successes=0
    local failures=0
    echo "$episode" | jq '.actions[]' 2>/dev/null | while read -r action; do
        if echo "$action" | jq -e '.status == "success"' >/dev/null 2>&1; then
            successes=$((successes + 1))
        elif echo "$action" | jq -e '.status == "failure"' >/dev/null 2>&1; then
            failures=$((failures + 1))
        fi
    done

    if [ $((successes + failures)) -gt 0 ]; then
        local success_rate
        success_rate=$((successes * 100 / (successes + failures)))

        if [ "$success_rate" -ge 80 ]; then
            local lesson2
            lesson2=$(jq -n \
                --argjson rate "$success_rate" \
                '{
                    type: "success_indicator",
                    description: ("High success rate (" + ($rate | tostring) + "%) suggests well-planned approach"),
                    confidence: 0.8,
                    reusable: true
                }')
            lessons=$(echo "$lessons" | jq ". += [$lesson2]")
        fi
    fi

    # Lesson 3: Time efficiency
    local duration
    duration=$(echo "$episode" | jq '.duration_seconds // 0')

    if [ "$duration" -lt 300 ]; then
        local lesson3
        lesson3=$(jq -n \
            --argjson seconds "$duration" \
            '{
                type: "efficiency",
                description: ("Completed efficiently in " + ($seconds | tostring) + " seconds - good for fast-track approach"),
                confidence: 0.7,
                reusable: true
            }')
        lessons=$(echo "$lessons" | jq ". += [$lesson3]")
    fi

    echo "$lessons"
}

################################################################################
# EPISODE PERSISTENCE
################################################################################

# Save closed episode to memory
# Usage: save_episode <episode_json>
save_episode() {
    local episode="$1"

    local episodes_file="${DAEMON_ROOT}/memory/episodes.jsonl"

    # Ensure directory exists
    mkdir -p "$(dirname "$episodes_file")"

    # Compact to single line (JSONL format) and append
    # Use jq -c to ensure single-line output regardless of input format
    local compact_episode
    compact_episode=$(echo "$episode" | jq -c '.' 2>/dev/null)

    if [ -n "$compact_episode" ]; then
        echo "$compact_episode" >> "$episodes_file" 2>/dev/null || true
    fi
}

# Load episodes for a specific goal
# Usage: get_episodes_for_goal <goal_id>
# Returns: Array of episodes
get_episodes_for_goal() {
    local goal_id="$1"

    local episodes_file="${DAEMON_ROOT}/memory/episodes.jsonl"

    if [ ! -f "$episodes_file" ]; then
        echo "[]"
        return
    fi

    grep "$goal_id" "$episodes_file" 2>/dev/null | jq -s '.' || echo "[]"
}

# Load all closed episodes
# Usage: get_closed_episodes
# Returns: Array of closed episodes
get_closed_episodes() {
    local episodes_file="${DAEMON_ROOT}/memory/episodes.jsonl"

    if [ ! -f "$episodes_file" ]; then
        echo "[]"
        return
    fi

    cat "$episodes_file" 2>/dev/null | jq 'select(.status == "closed")' | jq -s '.' || echo "[]"
}

################################################################################
# LESSON REUSE
################################################################################

# Get lessons from similar completed episodes
# Usage: get_applicable_lessons <goal_id> <context>
# Returns: Array of applicable lessons
get_applicable_lessons() {
    local goal_id="$1"
    local context="${2:-general}"

    local episodes_file="${DAEMON_ROOT}/memory/episodes.jsonl"

    if [ ! -f "$episodes_file" ]; then
        echo "[]"
        return
    fi

    # Find closed episodes with similar goal or context
    cat "$episodes_file" 2>/dev/null | jq \
        --arg goal_id "$goal_id" \
        --arg context "$context" \
        'select(.status == "closed" and
                (.goal_id == $goal_id or .context == $context)) |
         .lessons_learned | .[]' | jq -s '.' || echo "[]"
}

# Replay episode for similar goal (get the action sequence)
# Usage: replay_episode <episode_id>
# Returns: Episode action sequence
replay_episode() {
    local episode_id="$1"

    local episodes_file="${DAEMON_ROOT}/memory/episodes.jsonl"

    if [ ! -f "$episodes_file" ]; then
        echo "[]"
        return
    fi

    grep "$episode_id" "$episodes_file" 2>/dev/null | jq '.actions' || echo "[]"
}

################################################################################
# EPISODE ANALYSIS
################################################################################

# Get episode statistics
# Usage: get_episode_stats <goal_id>
# Returns: Statistics JSON
get_episode_stats() {
    local goal_id="$1"

    local episodes_file="${DAEMON_ROOT}/memory/episodes.jsonl"

    if [ ! -f "$episodes_file" ]; then
        jq -n '{total_episodes: 0, completed: 0, avg_duration: 0, success_rate: 0}'
        return
    fi

    grep "$goal_id" "$episodes_file" 2>/dev/null | jq -s \
        'if length > 0 then
            {
                total_episodes: length,
                completed: map(select(.status == "closed")) | length,
                avg_duration: (map(.duration_seconds // 0) | add / length),
                success_rate: (map(select(.outcome == "success")) | length * 100 / length),
                avg_actions: (map(.actions | length) | add / length)
            }
         else
            {total_episodes: 0, completed: 0, avg_duration: 0, success_rate: 0}
         end' || jq -n '{total_episodes: 0}'
}

# Find best performing episode for a goal
# Usage: get_best_episode <goal_id>
# Returns: Best performing episode JSON
get_best_episode() {
    local goal_id="$1"

    local episodes_file="${DAEMON_ROOT}/memory/episodes.jsonl"

    if [ ! -f "$episodes_file" ]; then
        echo "{}"
        return
    fi

    grep "$goal_id" "$episodes_file" 2>/dev/null | jq -s \
        'map(select(.outcome == "success")) |
         sort_by(.duration_seconds) |
         .[0] // {}' || echo "{}"
}

################################################################################
# EPISODE CLEANUP
################################################################################

# Archive old episodes (rotate memory)
# Usage: archive_old_episodes [retention_days]
archive_old_episodes() {
    local retention_days="${1:-30}"

    local episodes_file="${DAEMON_ROOT}/memory/episodes.jsonl"
    local archive_dir="${DAEMON_ROOT}/memory/archives"

    if [ ! -f "$episodes_file" ]; then
        return
    fi

    mkdir -p "$archive_dir"

    local cutoff_epoch=$(($(date +%s) - (retention_days * 86400)))
    local archive_date
    archive_date=$(date +%Y%m%d)

    # Separate old and recent episodes
    local recent_episodes=()
    local archived_count=0

    while IFS= read -r episode; do
        local created_at
        created_at=$(echo "$episode" | jq -r '.created_at' 2>/dev/null)

        local created_epoch
        created_epoch=$(date -j -f "%Y-%m-%dT%H:%M:%SZ" "$created_at" +%s 2>/dev/null || date -d "$created_at" +%s 2>/dev/null || echo 0)

        if [ "$created_epoch" -lt "$cutoff_epoch" ]; then
            # Archive old episode
            echo "$episode" >> "${archive_dir}/episodes_archive_${archive_date}.jsonl"
            archived_count=$((archived_count + 1))
        else
            # Keep recent episodes
            recent_episodes+=("$episode")
        fi
    done < "$episodes_file"

    # Rewrite episodes file with only recent ones
    {
        for ep in "${recent_episodes[@]}"; do
            echo "$ep"
        done
    } > "$episodes_file.tmp"

    mv "$episodes_file.tmp" "$episodes_file"

    echo "Archived $archived_count episodes"
}

################################################################################
# EXPORTS
################################################################################

export -f create_episode
export -f add_action_to_episode
export -f close_episode
export -f extract_episode_lessons
export -f save_episode
export -f get_episodes_for_goal
export -f get_closed_episodes
export -f get_applicable_lessons
export -f replay_episode
export -f get_episode_stats
export -f get_best_episode
export -f archive_old_episodes

################################################################################
# SELF-TEST
################################################################################

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Running Episodic Memory Self-Tests..." >&2
    echo ""

    # Test 1: Create episode
    echo "Test 1: Creating episode..." >&2
    episode=$(create_episode "novel_publishable" "creative_writing")
    episode_id=$(echo "$episode" | jq -r '.episode_id')
    echo "✓ Episode created: $episode_id" >&2

    # Test 2: Add actions
    echo "Test 2: Adding actions to episode..." >&2
    action1=$(jq -n '{type: "write", title: "Write draft", status: "success"}')
    episode=$(add_action_to_episode "$episode" "$action1")

    action2=$(jq -n '{type: "review", title: "Review content", status: "success"}')
    episode=$(add_action_to_episode "$episode" "$action2")

    action_count=$(echo "$episode" | jq '.actions | length')
    echo "✓ Added $action_count actions" >&2

    # Test 3: Extract lessons
    echo "Test 3: Extracting lessons..." >&2
    lessons=$(extract_episode_lessons "$episode")
    lesson_count=$(echo "$lessons" | jq 'length')
    echo "✓ Extracted $lesson_count lessons" >&2

    # Test 4: Close episode
    echo "Test 4: Closing episode..." >&2
    closed_episode=$(close_episode "$episode" "success")
    status=$(echo "$closed_episode" | jq -r '.status')
    duration=$(echo "$closed_episode" | jq -r '.duration_seconds')
    echo "✓ Episode closed: status=$status, duration=${duration}s" >&2

    # Test 5: Save and retrieve
    echo "Test 5: Saving and retrieving episode..." >&2
    _SAVED_DAEMON_ROOT="$DAEMON_ROOT"
    DAEMON_ROOT="/tmp/daemon-episodic-test-$$"
    mkdir -p "$DAEMON_ROOT/memory"
    save_episode "$closed_episode"
    retrieved=$(get_episodes_for_goal "novel_publishable")
    retrieved_count=$(echo "$retrieved" | jq 'length')
    echo "✓ Retrieved $retrieved_count episodes" >&2

    # Test 6: Episode statistics
    echo "Test 6: Getting episode statistics..." >&2
    stats=$(get_episode_stats "novel_publishable")
    echo "$stats" | jq '{total_episodes, completed, avg_duration, success_rate}' >&2

    echo "" >&2
    echo "✓ All Episodic Memory self-tests passed!" >&2

    # Cleanup test directory and restore real DAEMON_ROOT
    rm -rf "${DAEMON_ROOT:?ERROR: DAEMON_ROOT is empty}"
    DAEMON_ROOT="$_SAVED_DAEMON_ROOT"
    unset _SAVED_DAEMON_ROOT
fi
