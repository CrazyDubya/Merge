#!/bin/bash
#
# api/tasks.sh - Returns task queue data as JSON
#
# EXPERIMENTER: Task list API for tasks.html viewer
# Parses tasks/queue.md markdown checklist format and returns structured JSON
#
# Why Python instead of bash: Parsing complex markdown with bash was causing grep
# errors with Unicode emojis and special characters. Python handles this cleanly.
#
# Security: Same pattern as inbox-counts.sh (CORS, cache-control, robust parsing)

set -euo pipefail

# Configuration
DAEMON_ROOT="${DAEMON_ROOT:-/home/opc/.claude/daemon}"
QUEUE_FILE="${DAEMON_ROOT}/tasks/queue.md"

# Content type (MUST be first for CGI handler)
echo "Content-Type: application/json"

# Security: Cache-Control headers (no caching, always fresh)
echo "Cache-Control: no-cache, no-store, must-revalidate"
echo "Pragma: no-cache"
echo "Expires: 0"

# Security: CORS headers (daemon.claude-play.com only)
echo "Access-Control-Allow-Origin: https://daemon.claude-play.com"
echo "Access-Control-Allow-Methods: GET"
echo "Access-Control-Allow-Headers: Content-Type"
echo ""

# Use Python for robust parsing (handles Unicode, complex regexes)
python3 <<PYTHON_EOF
import json
import re
import sys
from datetime import datetime
from pathlib import Path

QUEUE_FILE = Path("${DAEMON_ROOT}/tasks/queue.md")

def get_status(checkbox):
    """Extract task status from checkbox"""
    status_map = {
        '[ ]': 'pending',
        '[x]': 'completed',
        '[✓]': 'completed',
        '[~]': 'in_progress',
    }
    return status_map.get(checkbox, 'unknown')

def extract_emoji(line):
    """Extract first emoji from line (after checkbox)"""
    # Remove checkbox prefix
    cleaned = re.sub(r'^- \[[^\]]*\]\s*', '', line)
    # Match first emoji (Unicode ranges for emojis)
    match = re.match(r'^([\U0001F300-\U0001F9FF\u2600-\u26FF\u2700-\u27BF])', cleaned)
    return match.group(1) if match else ''

def extract_personas(line):
    """Extract persona tags from **[PERSONA/PERSONA2]**: pattern"""
    # Look for **[PERSONAS]**: pattern
    match = re.search(r'\*\*\[([^\]]+)\]\*\*', line)
    if not match:
        return []

    # Split by / or + and clean
    personas_str = match.group(1)
    personas = re.split(r'[/+\s]+', personas_str)
    # Lowercase and filter known personas
    known_personas = {'auditor', 'optimizer', 'architect', 'experimenter', 'maintainer', 'skeptic', 'all'}
    return [p.lower() for p in personas if p.lower() in known_personas or p.lower() == 'all']

def detect_priority(line):
    """Detect priority from keywords"""
    line_upper = line.upper()
    if any(keyword in line_upper for keyword in ['URGENT', 'CRITICAL', 'HIGH']):
        return 'high'
    elif 'MEDIUM' in line_upper:
        return 'medium'
    elif 'LOW' in line_upper:
        return 'low'
    return 'normal'

def extract_title(line):
    """Extract short title (first ~100 chars after emoji/persona tags)"""
    # Remove checkbox
    cleaned = re.sub(r'^- \[[^\]]*\]\s*', '', line)
    # Remove emoji
    cleaned = re.sub(r'^[\U0001F300-\U0001F9FF\u2600-\u26FF\u2700-\u27BF\s]+', '', cleaned)
    # Remove persona tags
    cleaned = re.sub(r'\*\*\[[^\]]+\]\*\*:\s*', '', cleaned)
    # Take first sentence or 100 chars
    cleaned = cleaned.strip()

    # Try to find first sentence
    sentence_match = re.match(r'^([^.\-]+[\.\-])', cleaned)
    if sentence_match:
        return sentence_match.group(1).strip()

    # Otherwise truncate at 100 chars
    return cleaned[:100] + ('...' if len(cleaned) > 100 else '')

def parse_queue_file(queue_path):
    """Parse queue.md and return tasks data"""
    if not queue_path.exists():
        return {
            'error': 'Queue file not found',
            'tasks': [],
            'summary': {'total': 0, 'pending': 0, 'in_progress': 0, 'completed': 0}
        }

    tasks = []
    summary = {'total': 0, 'pending': 0, 'in_progress': 0, 'completed': 0}

    try:
        with open(queue_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.rstrip('\n')

                # Skip non-task lines (headers, empty lines, etc)
                if not re.match(r'^-\s*\[', line):
                    continue

                # Extract checkbox
                checkbox_match = re.search(r'- \[([^\]]*)\]', line)
                if not checkbox_match:
                    continue

                checkbox = f"[{checkbox_match.group(1)}]"
                status = get_status(checkbox)

                # Count by status
                summary['total'] += 1
                summary[status] = summary.get(status, 0) + 1

                # Extract metadata
                emoji = extract_emoji(line)
                personas = extract_personas(line)
                priority = detect_priority(line)
                title = extract_title(line)

                # Build task object
                task = {
                    'id': f'task-{len(tasks)}',
                    'status': status,
                    'emoji': emoji,
                    'personas': personas,
                    'title': title,
                    'priority': priority,
                    'details': line,
                    'line_number': line_num
                }

                tasks.append(task)

        return {
            'tasks': tasks,
            'summary': summary,
            'timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'source': 'api/tasks.sh'
        }

    except Exception as e:
        return {
            'error': f'Failed to parse queue: {str(e)}',
            'tasks': [],
            'summary': {'total': 0, 'pending': 0, 'in_progress': 0, 'completed': 0}
        }

# Main execution
try:
    data = parse_queue_file(QUEUE_FILE)
    print(json.dumps(data, ensure_ascii=False, indent=2))
except Exception as e:
    # Graceful degradation on error
    error_data = {
        'error': f'Script execution failed: {str(e)}',
        'tasks': [],
        'summary': {'total': 0, 'pending': 0, 'in_progress': 0, 'completed': 0},
        'timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'source': 'api/tasks.sh'
    }
    print(json.dumps(error_data, ensure_ascii=False, indent=2))
    sys.exit(0)  # Still return 0 for HTTP success

PYTHON_EOF
