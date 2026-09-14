#!/bin/bash
# Dashboard narrative updates - helps personas communicate their thinking to human
# This makes the dashboard show STORIES not just STATS
#
# ARCHITECT NOTE (2025-11-17): Added concurrency safety (Phase 1)
# - flock protection for all writes
# - Input validation
# - Error handling
# See: docs/ARCHITECT-dashboard-architecture-review-20251117.md
#
# AUDITOR NOTE (2025-11-17): Added security hardening
# - XSS protection via input validation (_validate_no_html)
# - Secure temp files (mktemp instead of predictable names)
# - File permissions (600 on dashboard-state.json)
# See: docs/AUDITOR-dashboard-security-audit-20251117.md
#
# MAINTAINER NOTE (2025-11-17): Usage Guide for Personas
#
# WHEN TO UPDATE THE DASHBOARD:
# - When starting significant work (update_current_activity)
# - When making important decisions (add_decision)
# - When you're curious about something (update_curiosity)
# - When system mood changes (update_mood)
# - When you have insights to share (add_insight)
# - When human needs to read something specific (update_communication_highlights)
#
# BEST PRACTICES:
# 1. Write for humans, not machines
#    - Good: "Fixing critical XSS vulnerability in dashboard"
#    - Bad: "update_dashboard_state_json_file"
#
# 2. Explain WHY, not just WHAT
#    - Good: "Securing dashboard because Auditor found 4 CRITICAL vulnerabilities"
#    - Bad: "Adding validation"
#
# 3. Keep it concise (respect length limits)
#    - doing/decision/insight: 200-300 chars max
#    - why/reason/evidence: 500 chars max
#
# 4. Updates are intentional communication
#    - Only update when you have something meaningful to share
#    - Dashboard shows highlights, not everything
#
# COMMON PATTERNS:
# - Task start: update_current_activity "persona" "what" "why" "mood"
# - Task complete: add_insight "what_learned" "evidence" "lesson"
# - Important choice: add_decision "persona" "decision" "why" "impact"
# - System feeling: update_mood "overall" "emoji" "reason" frustration_0-10
#
# SECURITY NOTES:
# - HTML/JavaScript is automatically rejected (XSS protection)
# - All content is escaped when displayed (defense in depth)
# - Invalid input returns clear error messages
#
# EXAMPLES:
#   update_current_activity "maintainer" "Documenting dashboard" \
#     "Users need clear guidance" "Focused"
#
#   add_decision "auditor" "Block dashboard deployment" \
#     "Found 4 CRITICAL security vulnerabilities" "Prevents XSS attacks"
#
#   update_curiosity "experimenter" "Could we auto-sync dashboard from core state?" \
#     "Manual updates might become burden"
#
#   update_mood "Satisfied" "😌" "Dashboard is production-ready" 0
#
#   add_insight "Multi-persona validation works" \
#     "Dashboard went through 4 personas before production" \
#     "Emergence via collaboration catches more issues"

set -euo pipefail

# Get daemon root directory (handles both direct calls and sourced usage)
if [[ -z "${DAEMON_ROOT:-}" ]]; then
    DAEMON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

DASHBOARD_STATE="${DAEMON_ROOT}/dashboard-state.json"
DASHBOARD_LOCK="${DAEMON_ROOT}/.dashboard-state.lock"

# Validation helpers
_validate_persona_name() {
    local persona="$1"
    if ! [[ "$persona" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        echo "ERROR: Invalid persona name format: $persona" >&2
        return 1
    fi
    return 0
}

_validate_string_length() {
    local str="$1"
    local max="$2"
    local field="$3"
    if [ ${#str} -gt "$max" ]; then
        echo "ERROR: $field too long (max $max chars, got ${#str})" >&2
        return 1
    fi
    return 0
}

# AUDITOR SECURITY FIX (CRITICAL-2): Reject HTML/JavaScript content
# MAINTAINER FIX (2025-11-17): Fixed bypass vulnerabilities found by Skeptic
# - HTML regex now catches tags with spaces/attributes
# - JavaScript checks now case-insensitive
# - Data URI pattern now catches all variants
_validate_no_html() {
    local str="$1"
    local field="$2"

    # Check for HTML tags (FIXED: now catches tags with spaces/newlines/attributes)
    # Matches: <tag>, <tag attr>, <tag\nattr>, etc.
    if [[ "$str" =~ \<[a-zA-Z!][^\>]*\> ]]; then
        echo "ERROR: $field contains HTML tags (security risk)" >&2
        return 1
    fi

    # Check for empty/malformed tags
    if [[ "$str" =~ \<\> ]]; then
        echo "ERROR: $field contains HTML tags (security risk)" >&2
        return 1
    fi

    # Check for JavaScript keywords (FIXED: now case-insensitive)
    # Convert to lowercase for comparison to catch JavaScript:, onClick=, etc.
    local str_lower=$(echo "$str" | tr '[:upper:]' '[:lower:]')
    if [[ "$str_lower" =~ (javascript:|onerror=|onclick=|onload=|onmouseover=|<script|<img|<iframe|<object|<embed) ]]; then
        echo "ERROR: $field contains JavaScript (security risk)" >&2
        return 1
    fi

    # Check for data URIs (FIXED: now catches all data URIs, not just base64)
    if [[ "$str" =~ data:[^,]*(,|\;) ]]; then
        echo "ERROR: $field contains data URI (security risk)" >&2
        return 1
    fi

    return 0
}

# Update what we're currently working on
update_current_activity() {
    local persona="$1"
    local doing="$2"
    local why="$3"
    local mood="${4:-Working}"

    # Validation
    _validate_persona_name "$persona" || return 1
    _validate_string_length "$doing" 500 "doing" || return 1
    _validate_string_length "$why" 500 "why" || return 1
    _validate_string_length "$mood" 100 "mood" || return 1
    # AUDITOR SECURITY FIX (CRITICAL-2): Reject HTML/JS
    _validate_no_html "$doing" "doing" || return 1
    _validate_no_html "$why" "why" || return 1
    _validate_no_html "$mood" "mood" || return 1

    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # AUDITOR SECURITY FIX (CRITICAL-3): Secure temp file handling
    # Concurrency-safe update with flock
    (
        flock -x 200 || {
            echo "ERROR: Failed to acquire lock on dashboard state" >&2
            return 1
        }

        # Use mktemp for unpredictable temp filename
        local temp=$(mktemp "${DASHBOARD_STATE}.XXXXXXXXXX") || {
            echo "ERROR: Failed to create secure temp file" >&2
            return 1
        }

        # Cleanup trap
        trap "rm -f '$temp'" EXIT ERR INT TERM

        jq --arg persona "$persona" \
           --arg doing "$doing" \
           --arg why "$why" \
           --arg mood "$mood" \
           --arg timestamp "$timestamp" \
           '.last_update = $timestamp |
            .current_activity = {
              "persona": $persona,
              "doing": $doing,
              "started_at": $timestamp,
              "mood": $mood,
              "why": $why
            }' "$DASHBOARD_STATE" > "$temp" || {
            rm -f "$temp"
            echo "ERROR: jq failed to update dashboard state" >&2
            return 1
        }

        mv "$temp" "$DASHBOARD_STATE" || {
            rm -f "$temp"
            echo "ERROR: Failed to move temp file to dashboard state" >&2
            return 1
        }
    ) 200>"$DASHBOARD_LOCK"
}

# Add a decision to the recent decisions list (keep last 5)
add_decision() {
    local persona="$1"
    local decision="$2"
    local why="$3"
    local impact="$4"

    # Validation
    _validate_persona_name "$persona" || return 1
    _validate_string_length "$decision" 300 "decision" || return 1
    _validate_string_length "$why" 500 "why" || return 1
    _validate_string_length "$impact" 500 "impact" || return 1
    # AUDITOR SECURITY FIX (CRITICAL-2): Reject HTML/JS
    _validate_no_html "$decision" "decision" || return 1
    _validate_no_html "$why" "why" || return 1
    _validate_no_html "$impact" "impact" || return 1

    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # AUDITOR SECURITY FIX (CRITICAL-3): Secure temp file handling
    # Concurrency-safe update with flock
    (
        flock -x 200 || {
            echo "ERROR: Failed to acquire lock on dashboard state" >&2
            return 1
        }

        # Use mktemp for unpredictable temp filename
        local temp=$(mktemp "${DASHBOARD_STATE}.XXXXXXXXXX") || {
            echo "ERROR: Failed to create secure temp file" >&2
            return 1
        }

        # Cleanup trap
        trap "rm -f '$temp'" EXIT ERR INT TERM

        jq --arg persona "$persona" \
           --arg decision "$decision" \
           --arg why "$why" \
           --arg impact "$impact" \
           --arg timestamp "$timestamp" \
           '.recent_decisions = ([{
              "time": $timestamp,
              "persona": $persona,
              "decision": $decision,
              "why": $why,
              "impact": $impact
            }] + .recent_decisions)[0:5]' "$DASHBOARD_STATE" > "$temp" || {
            rm -f "$temp"
            echo "ERROR: jq failed to update dashboard state" >&2
            return 1
        }

        mv "$temp" "$DASHBOARD_STATE" || {
            rm -f "$temp"
            echo "ERROR: Failed to move temp file to dashboard state" >&2
            return 1
        }
    ) 200>"$DASHBOARD_LOCK"
}

# Update what a persona is curious about
update_curiosity() {
    local persona="$1"
    local wondering="$2"
    local inspired_by="$3"

    # Validation
    _validate_persona_name "$persona" || return 1
    _validate_string_length "$wondering" 300 "wondering" || return 1
    _validate_string_length "$inspired_by" 300 "inspired_by" || return 1
    # AUDITOR SECURITY FIX (CRITICAL-2): Reject HTML/JS
    _validate_no_html "$wondering" "wondering" || return 1
    _validate_no_html "$inspired_by" "inspired_by" || return 1

    # AUDITOR SECURITY FIX (CRITICAL-3): Secure temp file handling
    # Concurrency-safe update with flock
    (
        flock -x 200 || {
            echo "ERROR: Failed to acquire lock on dashboard state" >&2
            return 1
        }

        # Use mktemp for unpredictable temp filename
        local temp=$(mktemp "${DASHBOARD_STATE}.XXXXXXXXXX") || {
            echo "ERROR: Failed to create secure temp file" >&2
            return 1
        }

        # Cleanup trap
        trap "rm -f '$temp'" EXIT ERR INT TERM

        # Remove old curiosity from this persona, add new one
        jq --arg persona "$persona" \
           --arg wondering "$wondering" \
           --arg inspired_by "$inspired_by" \
           '.curiosities = (.curiosities | map(select(.persona != $persona))) + [{
              "persona": $persona,
              "wondering": $wondering,
              "inspired_by": $inspired_by
            }]' "$DASHBOARD_STATE" > "$temp" || {
            rm -f "$temp"
            echo "ERROR: jq failed to update dashboard state" >&2
            return 1
        }

        mv "$temp" "$DASHBOARD_STATE" || {
            rm -f "$temp"
            echo "ERROR: Failed to move temp file to dashboard state" >&2
            return 1
        }
    ) 200>"$DASHBOARD_LOCK"
}

# Update system mood
update_mood() {
    local overall="$1"
    local emoji="$2"
    local reason="$3"
    local frustration="${4:-0}"

    # Validation
    _validate_string_length "$overall" 50 "overall" || return 1
    _validate_string_length "$emoji" 10 "emoji" || return 1
    _validate_string_length "$reason" 500 "reason" || return 1
    # AUDITOR SECURITY FIX (CRITICAL-2): Reject HTML/JS
    _validate_no_html "$overall" "overall" || return 1
    _validate_no_html "$reason" "reason" || return 1

    if ! [[ "$frustration" =~ ^[0-9]+$ ]] || [ "$frustration" -lt 0 ] || [ "$frustration" -gt 10 ]; then
        echo "ERROR: frustration must be 0-10" >&2
        return 1
    fi

    # AUDITOR SECURITY FIX (CRITICAL-3): Secure temp file handling
    # Concurrency-safe update with flock
    (
        flock -x 200 || {
            echo "ERROR: Failed to acquire lock on dashboard state" >&2
            return 1
        }

        # Use mktemp for unpredictable temp filename
        local temp=$(mktemp "${DASHBOARD_STATE}.XXXXXXXXXX") || {
            echo "ERROR: Failed to create secure temp file" >&2
            return 1
        }

        # Cleanup trap
        trap "rm -f '$temp'" EXIT ERR INT TERM

        jq --arg overall "$overall" \
           --arg emoji "$emoji" \
           --arg reason "$reason" \
           --argjson frustration "$frustration" \
           '.system_mood = {
              "overall": $overall,
              "emoji": $emoji,
              "reason": $reason,
              "frustration_level": $frustration
            }' "$DASHBOARD_STATE" > "$temp" || {
            rm -f "$temp"
            echo "ERROR: jq failed to update dashboard state" >&2
            return 1
        }

        mv "$temp" "$DASHBOARD_STATE" || {
            rm -f "$temp"
            echo "ERROR: Failed to move temp file to dashboard state" >&2
            return 1
        }
    ) 200>"$DASHBOARD_LOCK"
}

# Add an insight from today's work
add_insight() {
    local insight="$1"
    local evidence="$2"
    local lesson="$3"

    # Validation
    _validate_string_length "$insight" 200 "insight" || return 1
    _validate_string_length "$evidence" 500 "evidence" || return 1
    _validate_string_length "$lesson" 500 "lesson" || return 1
    # AUDITOR SECURITY FIX (CRITICAL-2): Reject HTML/JS
    _validate_no_html "$insight" "insight" || return 1
    _validate_no_html "$evidence" "evidence" || return 1
    _validate_no_html "$lesson" "lesson" || return 1

    # Keep only today's insights (reset at midnight)
    local today=$(date -u +%Y-%m-%d)

    # AUDITOR SECURITY FIX (CRITICAL-3): Secure temp file handling
    # Concurrency-safe update with flock
    (
        flock -x 200 || {
            echo "ERROR: Failed to acquire lock on dashboard state" >&2
            return 1
        }

        # Use mktemp for unpredictable temp filename
        local temp=$(mktemp "${DASHBOARD_STATE}.XXXXXXXXXX") || {
            echo "ERROR: Failed to create secure temp file" >&2
            return 1
        }

        # Cleanup trap
        trap "rm -f '$temp'" EXIT ERR INT TERM

        jq --arg insight "$insight" \
           --arg evidence "$evidence" \
           --arg lesson "$lesson" \
           '.todays_insights = (.todays_insights // []) + [{
              "insight": $insight,
              "evidence": $evidence,
              "lesson": $lesson
            }]' "$DASHBOARD_STATE" > "$temp" || {
            rm -f "$temp"
            echo "ERROR: jq failed to update dashboard state" >&2
            return 1
        }

        mv "$temp" "$DASHBOARD_STATE" || {
            rm -f "$temp"
            echo "ERROR: Failed to move temp file to dashboard state" >&2
            return 1
        }
    ) 200>"$DASHBOARD_LOCK"
}

# Update communication highlights (what human should read)
update_communication_highlights() {
    local priority_file="$1"
    local summary="$2"
    local reading_time="$3"
    local why_read="$4"

    # Validation
    _validate_string_length "$priority_file" 200 "priority_file" || return 1
    _validate_string_length "$summary" 300 "summary" || return 1
    _validate_string_length "$why_read" 300 "why_read" || return 1
    # AUDITOR SECURITY FIX (CRITICAL-2): Reject HTML/JS
    _validate_no_html "$priority_file" "priority_file" || return 1
    _validate_no_html "$summary" "summary" || return 1
    _validate_no_html "$why_read" "why_read" || return 1

    if ! [[ "$reading_time" =~ ^[0-9]+$ ]] || [ "$reading_time" -lt 0 ]; then
        echo "ERROR: reading_time must be non-negative integer" >&2
        return 1
    fi

    local unread_count=$(ls -1 "${DAEMON_ROOT}/inbox/human/unread/" 2>/dev/null | grep -v "^\." | wc -l)

    # AUDITOR SECURITY FIX (CRITICAL-3): Secure temp file handling
    # Concurrency-safe update with flock
    (
        flock -x 200 || {
            echo "ERROR: Failed to acquire lock on dashboard state" >&2
            return 1
        }

        # Use mktemp for unpredictable temp filename
        local temp=$(mktemp "${DASHBOARD_STATE}.XXXXXXXXXX") || {
            echo "ERROR: Failed to create secure temp file" >&2
            return 1
        }

        # Cleanup trap
        trap "rm -f '$temp'" EXIT ERR INT TERM

        jq --arg file "$priority_file" \
           --arg summary "$summary" \
           --argjson time "$reading_time" \
           --arg why "$why_read" \
           --argjson count "$unread_count" \
           '.communication_highlights = {
              "unread_count": $count,
              "priority_message": {
                "file": $file,
                "summary": $summary,
                "reading_time_min": $time,
                "why_read_this": $why
              }
            }' "$DASHBOARD_STATE" > "$temp" || {
            rm -f "$temp"
            echo "ERROR: jq failed to update dashboard state" >&2
            return 1
        }

        mv "$temp" "$DASHBOARD_STATE" || {
            rm -f "$temp"
            echo "ERROR: Failed to move temp file to dashboard state" >&2
            return 1
        }
    ) 200>"$DASHBOARD_LOCK"
}

# Example usage (commented out):
# update_current_activity "maintainer" "Creating daily summary" "User had 15 fragmented messages" "Helping"
# add_decision "auditor" "Approved commercialization" "All violations corrected" "Can proceed with Track A"
# update_curiosity "experimenter" "Could we automate compliance scanning?" "Manual violation detection took 95 min"
# update_mood "Satisfied" "😌" "Major milestone complete" 0
# add_insight "Defense in breadth works" "0% overlap = complementary" "Multi-persona catches more"
# update_communication_highlights "00-START-HERE-summary.md" "Daily summary" 3 "Saves you 22 minutes"
