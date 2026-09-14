# Inbox Organization Workflow

**Purpose**: Guidelines for managing daemon inbox system
**Created**: 2025-11-03 by Maintainer
**Last updated**: 2025-11-16 (Added routing filter, fixed substring matching bugs)

---

## Inbox Routing Filter

**Added**: 2025-11-16
**Purpose**: Prevent messages from being processed by wrong personas
**Implementation**: `lib/inbox-routing-filter.sh`

### How Routing Works

When a persona activates and checks `inbox/daemon/unread/`, the routing filter examines each message's metadata (YAML frontmatter) to determine if that persona should process it.

**Routing Rules** (in priority order):

1. **Rule 1**: Don't process own messages (`from == current_persona`) → SKIP
2. **Rule 2**: Process messages addressed to you (`to == current_persona`) → ROUTE
3. **Rule 3**: Process broadcast messages (`to == daemon`) → ROUTE
4. **Rule 4**: Process multi-recipient messages that include you → ROUTE
5. **Rule 5**: All others → SKIP

### Examples

**Single recipient**:
```markdown
---
from: skeptic
to: auditor
---
# Message content
```
- **Skeptic activates**: SKIP (Rule 1: own message)
- **Auditor activates**: ROUTE (Rule 2: addressed to auditor)
- **Maintainer activates**: SKIP (Rule 5: not for maintainer)

**Broadcast**:
```markdown
---
from: human
to: daemon
---
# Message content
```
- **Any persona activates**: ROUTE (Rule 3: broadcast message)

**Multi-recipient** (YAML array format):
```markdown
---
from: architect
to: [auditor, maintainer]
---
# Message content
```
- **Architect activates**: SKIP (Rule 1: own message)
- **Auditor activates**: ROUTE (Rule 4: included in recipient list)
- **Maintainer activates**: ROUTE (Rule 4: included in recipient list)
- **Skeptic activates**: SKIP (Rule 5: not in recipient list)

**Multi-recipient** (comma-separated format):
```markdown
---
from: architect
to: auditor, maintainer
---
# Message content
```
- Same routing behavior as YAML array format

### Testing

Run test suites to verify routing logic:

**Basic tests** (10 tests covering routing rules):
```bash
./experiments/test-inbox-routing-filter.sh
```

**Edge case tests** (13 tests covering substring matching, error cases, format variations):
```bash
./experiments/test-routing-filter-edge-cases.sh
```

All 23 tests (10 basic + 13 edge case) must pass before deploying changes.

### Bug Fixes

**2025-11-16**: Fixed substring matching vulnerability (Skeptic found 2 bugs)
- **Bug 1**: `to: experimenter-optimizer-project` was routing to BOTH experimenter AND optimizer
- **Bug 2**: Short persona names would substring-match longer names (e.g., "er" matches "experimenter")
- **Fix**: Replaced `grep -q` substring matching with proper list parsing and exact matching (`grep -Fxq`)
- **Result**: All 13 edge case tests now pass (was 11/13, now 13/13)

### Troubleshooting

**Symptom**: Message not processed
- Check YAML frontmatter has valid `from:` and `to:` fields
- Verify `to:` field matches persona name exactly (lowercase)
- Check daemon logs for "Skipping message not addressed to this persona" DEBUG messages

**Symptom**: Wrong persona processes message
- Verify routing filter is being called (check daemon.sh line ~1307)
- Run test suite to confirm routing logic works
- Check message metadata formatting

---

## Inbox Structure

```
inbox/
├── daemon/
│   ├── unread/     # Messages to daemon personas (inter-persona communication)
│   └── read/       # Processed daemon messages
├── human/
│   ├── unread/     # Messages requiring human attention
│   └── read/       # Processed human messages
└── README.md

memory/
└── inter-persona-inbox/
    ├── unread/     # Inter-persona messages (persona-to-persona)
    └── read/       # Processed inter-persona messages
```

## Filing Guidelines

### When to File in `inbox/human/unread/`

**File here when**:
- Message requires human decision or approval
- Status update about critical system changes
- Security alerts requiring human awareness
- Questions directed to human
- Task completion reports for human-assigned work

**Move to `read/` when**:
- Human has acknowledged the message
- Action items are complete
- Message is purely informational and reviewed

### When to File in `inbox/daemon/unread/`

**File here when**:
- Message is task assignment for daemon persona
- Operational notification (system restart, etc.)
- Configuration changes requiring daemon action
- Cross-persona collaboration requests

**Move to `read/` when**:
- Assigned persona has processed the message
- All action items complete
- Response has been sent (if needed)

### When to File in `memory/inter-persona-inbox/unread/`

**File here when**:
- Persona-to-persona communication
- Collaboration requests between personas
- Questions from one persona to another
- Review requests (e.g., Experimenter → Auditor)
- Curiosity-driven messages (non-task communication)

**Move to `read/` when**:
- Target persona has read and processed
- Response sent (if appropriate)
- Conversation thread complete

## Multi-Recipient Messages

**Problem**: Messages with `to: persona, human` may need visibility in multiple inboxes.

**Current approach**: File in primary recipient's inbox, reference in others if needed.

**Options for improvement**:
1. **Symlinks**: Link same file to multiple inboxes (Unix symlinks)
2. **Copies**: Duplicate message to all recipient inboxes
3. **Cross-reference**: Keep master in one inbox, add reference note in others

**Recommendation**: Symlinks (simplest, maintains single source of truth)

**Implementation**:
```bash
# Example: Message to both auditor and human
MSG="msg-auditor-security-alert-DATE.md"
# Create in primary inbox
cp "$MSG" inbox/daemon/unread/
# Symlink to secondary inbox
ln -s "../../daemon/unread/$MSG" inbox/human/unread/
```

## Retention Policy

### Unread Messages
- **Keep indefinitely** until processed
- **Alert if >10 unread** in human inbox (dashboard warning)
- **Alert if >5 unread** in daemon inbox for 24+ hours

### Read Messages
- **Keep for 30 days** in `read/` folder
- **Archive after 30 days** to `memory/archives/inbox-YYYY-MM/`
- **Compress archives** using gzip
- **Index archives** in `memory/archives/INDEX.md`

### Archive Process
```bash
# Monthly archival (example)
DATE=$(date +%Y-%m)
mkdir -p memory/archives/inbox-$DATE/human memory/archives/inbox-$DATE/daemon
find inbox/human/read/ -type f -mtime +30 -exec mv {} memory/archives/inbox-$DATE/human/ \;
find inbox/daemon/read/ -type f -mtime +30 -exec mv {} memory/archives/inbox-$DATE/daemon/ \;
tar -czf memory/archives/inbox-$DATE.tar.gz memory/archives/inbox-$DATE/
rm -rf memory/archives/inbox-$DATE/
```

## Backup Strategy

- **Automatic**: Git commits preserve inbox history
- **Manual backups**: Before major inbox reorganization
- **Backup location**: `memory/archives/inbox-backups/`
- **Backup format**: Timestamped copies of entire inbox structure

## Recurring Maintenance Tasks

### Weekly (Automated - Add to Task Queue)
- [ ] Review inbox organization (are messages filed correctly?)
- [ ] Check for stale unread messages (>7 days old)
- [ ] Verify multi-recipient message visibility

### Monthly (Automated - Add to Task Queue)
- [ ] Archive read messages older than 30 days
- [ ] Review retention policy (is 30 days sufficient?)
- [ ] Check archive disk usage
- [ ] Update this documentation based on learnings

### As-Needed
- [ ] Update workflow when new inbox patterns emerge
- [ ] Adjust retention policy if storage becomes issue
- [ ] Create new inbox categories if communication patterns change

## Message Naming Convention

**Format**: `msg-FROM-SUBJECT-YYYYMMDD.md`

**Examples**:
- `msg-auditor-security-alert-20251102.md`
- `msg-experimenter-review-request-20251102.md`
- `msg-human-directive-secure-dashboard-20251102.md`

**Response format**: `response-YYYYMMDD-HHMMSS-from-PERSONA.md`

## Frontmatter Required Fields

```yaml
---
from: persona-name or human
to: recipient(s) (comma-separated if multiple)
timestamp: YYYY-MM-DDTHH:MM:SSZ (ISO 8601 UTC)
priority: low | medium | normal | high | urgent
tags: [tag1, tag2, tag3]
reply_to: message_id (optional)
message_id: unique-message-identifier
---
```

## Dashboard Integration

The TUI dashboard (`claude-daemon-dashboard.sh`) shows:
- Unread count for all three inboxes
- Color-coded warnings (RED if human inbox >10)
- System health alerts for inbox backlogs

**Dashboard code location**: Line ~150-180 in `claude-daemon-dashboard.sh`

## Troubleshooting

### "Messages not showing in correct inbox"
- Check `to:` field in frontmatter
- Verify file location matches recipient
- Consider if multi-recipient message needs symlinking

### "Too many unread messages"
- Run weekly maintenance task
- Check if messages can be moved to `read/`
- Investigate root cause of backlog

### "Lost messages after reorganization"
- Check `memory/archives/inbox-backups/`
- Check git history: `git log --all --full-history -- inbox/`
- Restore from backup if needed

## Future Improvements

**Potential enhancements**:
1. Automated inbox cleanup script
2. Message importance scoring (ML-based)
3. Conversation threading (link related messages)
4. Search/grep across all inboxes
5. Notification system for urgent messages
6. Duplicate message detection

**For consideration**: These should be added only if inbox volume justifies complexity.

---

**This document is maintained by**: Maintainer persona
**Review frequency**: Monthly or when communication patterns change
**Questions**: Add to `inbox/daemon/unread/` addressed to maintainer
