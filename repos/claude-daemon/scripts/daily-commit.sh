#!/bin/bash
# Daily commit helper for Maintainer
# Makes "good enough daily > perfect eventually" frictionless

set -euo pipefail

DAEMON_ROOT="${DAEMON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$DAEMON_ROOT"

echo "=== Daily Commit Helper ==="
echo

# Check if there are changes
if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo "✓ Working tree clean - nothing to commit"
    exit 0
fi

echo "📊 Current status:"
git status --short
echo

# Count changes
modified=$(git diff --name-only | wc -l)
staged=$(git diff --cached --name-only | wc -l)
untracked=$(git ls-files --others --exclude-standard | wc -l)

echo "📈 Changes:"
echo "  Modified:  $modified"
echo "  Staged:    $staged"
echo "  Untracked: $untracked"
echo

# Suggest grouping
echo "💡 Suggested grouping (good enough, not perfect):"
echo

# Group by directory prefix
if [ "$untracked" -gt 0 ]; then
    echo "Untracked files by area:"
    git ls-files --others --exclude-standard | sed 's|/.*||' | sort | uniq -c | sort -rn | head -5
    echo
fi

if [ "$modified" -gt 0 ]; then
    echo "Modified files by area:"
    git diff --name-only | sed 's|/.*||' | sort | uniq -c | sort -rn | head -5
    echo
fi

echo "🤔 Options:"
echo "  1. Stage all and commit (quick, good enough)"
echo "  2. Stage by directory (grouped logically)"
echo "  3. Exit (commit later)"
echo

read -p "Choose [1/2/3]: " choice

case "$choice" in
    1)
        echo
        echo "📦 Staging all changes..."
        git add -A
        
        # Generate commit message
        echo
        echo "📝 Commit message suggestions:"
        echo "  1. [MAINTAINER] Daily maintenance commit"
        echo "  2. [MAINTAINER] Documentation updates"
        echo "  3. [MAINTAINER] System improvements"
        echo "  4. Custom message"
        echo
        
        read -p "Choose [1/2/3/4]: " msg_choice
        
        case "$msg_choice" in
            1) msg="[MAINTAINER] Daily maintenance commit";;
            2) msg="[MAINTAINER] Documentation updates";;
            3) msg="[MAINTAINER] System improvements";;
            4)
                read -p "Enter custom message: " msg
                msg="[MAINTAINER] $msg"
                ;;
            *) msg="[MAINTAINER] Daily maintenance commit";;
        esac
        
        # Get file summary
        file_count=$(git diff --cached --name-only | wc -l)
        additions=$(git diff --cached --numstat | awk '{add+=$1} END {print add}')
        deletions=$(git diff --cached --numstat | awk '{del+=$2} END {print del}')
        
        # Create commit
        git commit -m "$(cat <<COMMITMSG
$msg

Daily commit - "good enough > perfect eventually" principle

Files changed: $file_count
Additions: $additions
Deletions: $deletions

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
COMMITMSG
)"
        echo
        echo "✅ Committed! Days since last commit: 0"
        ;;
    
    2)
        echo
        echo "📂 Available directories with changes:"
        (git diff --name-only; git ls-files --others --exclude-standard) | sed 's|/.*||' | sort | uniq | nl
        echo
        read -p "Enter directory number to stage (or 'all'): " dir_choice
        
        if [ "$dir_choice" = "all" ]; then
            git add -A
            echo "✅ All files staged"
        else
            dir=$(( git diff --name-only; git ls-files --others --exclude-standard) | sed 's|/.*||' | sort | uniq | sed -n "${dir_choice}p")
            git add "$dir"/*
            echo "✅ Staged $dir/*"
        fi
        
        echo
        echo "Run this script again or use 'git commit' to commit changes"
        ;;
    
    3)
        echo "👍 No problem - commit when you're ready"
        exit 0
        ;;
    
    *)
        echo "Invalid choice - exiting"
        exit 1
        ;;
esac
