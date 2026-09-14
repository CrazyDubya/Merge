#!/bin/bash
# Maintenance Metrics Dashboard
# Tracks repository health to prevent maintenance debt accumulation
#
# Usage: ./scripts/maintenance-metrics.sh
#
# Metrics tracked:
# - Days since last commit
# - Untracked file count (excluding gitignored)
# - Uncommitted changes
# - Commit frequency trend
# - Largest uncommitted project

set -e

cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo "  Maintenance Metrics Dashboard"
echo "=========================================="
echo ""
echo "Generated: $(date +'%Y-%m-%d %H:%M:%S')"
echo ""

# ==================================
# 1. Days Since Last Commit
# ==================================
echo "📊 GIT COMMIT HEALTH"
echo "----------------------------------------"

last_commit_date=$(git log -1 --format=%cd --date=short 2>/dev/null || echo "unknown")
last_commit_time=$(git log -1 --format=%cd --date=iso 2>/dev/null || echo "unknown")

if [[ "$last_commit_date" != "unknown" ]]; then
    last_commit_epoch=$(date -d "$last_commit_time" +%s 2>/dev/null || echo "0")
    now_epoch=$(date +%s)
    days_since_commit=$(( (now_epoch - last_commit_epoch) / 86400 ))
    hours_since_commit=$(( (now_epoch - last_commit_epoch) / 3600 ))

    echo "Last commit: $last_commit_date ($hours_since_commit hours ago)"
    echo "Days since last commit: $days_since_commit"

    # Alert thresholds
    if [[ $days_since_commit -gt 2 ]]; then
        echo "⚠️  WARNING: $days_since_commit days without commit (threshold: 2 days)"
        echo "   Recommendation: Review uncommitted work and commit soon"
    elif [[ $hours_since_commit -gt 24 ]]; then
        echo "⚠️  NOTICE: Over 24 hours since last commit"
        echo "   Recommendation: Consider daily commit routine"
    else
        echo "✅ HEALTHY: Recent commit activity"
    fi
else
    echo "❌ ERROR: Could not determine last commit date"
fi

echo ""

# ==================================
# 2. Uncommitted Changes
# ==================================
echo "📝 UNCOMMITTED CHANGES"
echo "----------------------------------------"

# Modified files
modified_count=$(git diff --name-only | wc -l)
staged_count=$(git diff --cached --name-only | wc -l)

echo "Modified files (unstaged): $modified_count"
echo "Staged files: $staged_count"

if [[ $modified_count -gt 0 ]]; then
    echo ""
    echo "Modified files:"
    git diff --name-only | sed 's/^/  - /'
fi

if [[ $staged_count -gt 0 ]]; then
    echo ""
    echo "Staged files:"
    git diff --cached --name-only | sed 's/^/  - /'
fi

echo ""

# ==================================
# 3. Untracked Files
# ==================================
echo "📁 UNTRACKED FILES"
echo "----------------------------------------"

untracked_count=$(git ls-files --others --exclude-standard | wc -l)

echo "Untracked files (not in .gitignore): $untracked_count"

if [[ $untracked_count -gt 20 ]]; then
    echo "⚠️  WARNING: $untracked_count untracked files (threshold: 20)"
    echo "   Recommendation: Review and commit or update .gitignore"
elif [[ $untracked_count -gt 10 ]]; then
    echo "⚠️  NOTICE: $untracked_count untracked files"
    echo "   Recommendation: Consider reviewing soon"
elif [[ $untracked_count -gt 0 ]]; then
    echo "✅ HEALTHY: $untracked_count untracked files (manageable)"
else
    echo "✅ EXCELLENT: No untracked files"
fi

if [[ $untracked_count -gt 0 ]] && [[ $untracked_count -le 20 ]]; then
    echo ""
    echo "Untracked files:"
    git ls-files --others --exclude-standard | head -20 | sed 's/^/  - /'
    if [[ $untracked_count -gt 20 ]]; then
        echo "  ... and $((untracked_count - 20)) more"
    fi
fi

echo ""

# ==================================
# 4. Commit Frequency (Last 7 Days)
# ==================================
echo "📈 COMMIT FREQUENCY (Last 7 Days)"
echo "----------------------------------------"

commits_last_7=$(git log --since="7 days ago" --oneline | wc -l)
commits_per_day=$(echo "scale=1; $commits_last_7 / 7" | bc 2>/dev/null || echo "0")

echo "Commits in last 7 days: $commits_last_7"
echo "Average per day: $commits_per_day"

if (( $(echo "$commits_per_day >= 1.0" | bc -l) )); then
    echo "✅ EXCELLENT: Daily commit pattern established"
elif (( $(echo "$commits_per_day >= 0.5" | bc -l) )); then
    echo "✅ GOOD: Regular commit activity"
elif (( $(echo "$commits_per_day > 0" | bc -l) )); then
    echo "⚠️  NOTICE: Infrequent commits"
    echo "   Recommendation: Aim for daily commits"
else
    echo "❌ WARNING: No commits in last 7 days"
    echo "   Recommendation: Review and commit work"
fi

echo ""

# ==================================
# 5. Repository Size Trends
# ==================================
echo "💾 REPOSITORY SIZE"
echo "----------------------------------------"

repo_size=$(du -sh .git 2>/dev/null | cut -f1)
working_tree_size=$(du -sh --exclude=.git . 2>/dev/null | cut -f1)

echo "Git repository (.git): $repo_size"
echo "Working tree: $working_tree_size"

echo ""

# ==================================
# 6. Log File Health
# ==================================
echo "📋 LOG FILE HEALTH"
echo "----------------------------------------"

get_log_size_mb() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "0"
        return
    fi
    local size_bytes
    size_bytes=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo "0")
    echo $((size_bytes / 1024 / 1024))
}

# Check key log files
activity_size=$(get_log_size_mb "logs/activity.log")
audit_size=$(get_log_size_mb "logs/state-audit.jsonl")
emergence_size=$(get_log_size_mb "memory/emergence-log.md")

echo "activity.log: ${activity_size} MB (threshold: 50 MB)"
echo "state-audit.jsonl: ${audit_size} MB (threshold: 25 MB)"
echo "emergence-log.md: ${emergence_size} MB (threshold: 0 MB - auto-rotates at 0.1 MB)"

# Check if rotation needed
logs_need_rotation=false
if [[ $activity_size -ge 50 ]]; then
    echo "⚠️  WARNING: activity.log needs rotation (${activity_size} MB >= 50 MB)"
    echo "   Run: ./scripts/rotate-activity-log.sh"
    logs_need_rotation=true
elif [[ $activity_size -ge 40 ]]; then
    echo "⚠️  NOTICE: activity.log approaching threshold (${activity_size} MB / 50 MB)"
fi

if [[ $audit_size -ge 25 ]]; then
    echo "⚠️  WARNING: state-audit.jsonl needs rotation (${audit_size} MB >= 25 MB)"
    echo "   Run: ./scripts/rotate-state-audit-log.sh"
    logs_need_rotation=true
elif [[ $audit_size -ge 20 ]]; then
    echo "⚠️  NOTICE: state-audit.jsonl approaching threshold (${audit_size} MB / 25 MB)"
fi

if [[ $logs_need_rotation == false ]]; then
    if [[ $activity_size -lt 40 ]] && [[ $audit_size -lt 20 ]]; then
        echo "✅ EXCELLENT: All logs within healthy ranges"
    else
        echo "✅ HEALTHY: Logs under threshold (monitoring recommended)"
    fi
fi

echo ""

# ==================================
# 7. Summary & Recommendations
# ==================================
echo "=========================================="
echo "  SUMMARY & RECOMMENDATIONS"
echo "=========================================="
echo ""

total_warnings=0
total_notices=0

# Count issues
if [[ $days_since_commit -gt 2 ]]; then
    ((total_warnings++))
elif [[ $hours_since_commit -gt 24 ]]; then
    ((total_notices++))
fi

if [[ $untracked_count -gt 20 ]]; then
    ((total_warnings++))
elif [[ $untracked_count -gt 10 ]]; then
    ((total_notices++))
fi

if (( $(echo "$commits_per_day < 0.5" | bc -l) )); then
    ((total_notices++))
fi

if [[ $modified_count -gt 0 ]] || [[ $staged_count -gt 0 ]]; then
    ((total_notices++))
fi

if [[ $logs_need_rotation == true ]]; then
    ((total_warnings++))
fi

# Report status
if [[ $total_warnings -gt 0 ]]; then
    echo "⚠️  STATUS: NEEDS ATTENTION ($total_warnings warnings, $total_notices notices)"
    echo ""
    echo "Priority actions:"
    if [[ $days_since_commit -gt 2 ]]; then
        echo "  1. Review and commit uncommitted work (no commits in $days_since_commit days)"
    fi
    if [[ $untracked_count -gt 20 ]]; then
        echo "  2. Review $untracked_count untracked files (commit or gitignore)"
    fi
elif [[ $total_notices -gt 0 ]]; then
    echo "✅ STATUS: HEALTHY ($total_notices notices)"
    echo ""
    echo "Recommended actions:"
    if [[ $hours_since_commit -gt 24 ]]; then
        echo "  - Consider committing recent work"
    fi
    if [[ $untracked_count -gt 10 ]]; then
        echo "  - Review untracked files when convenient"
    fi
    if [[ $modified_count -gt 0 ]] || [[ $staged_count -gt 0 ]]; then
        echo "  - Uncommitted changes present (review when ready)"
    fi
else
    echo "✅ STATUS: EXCELLENT"
    echo ""
    echo "Repository is in excellent shape!"
    echo "  - Recent commits"
    echo "  - Clean working tree"
    echo "  - Regular commit frequency"
fi

echo ""
echo "Run this script at the start of Maintainer sessions to track health."
echo ""

exit 0
