# Multi-Persona Daemon Timing Schedule

## Active Hours: 7AM - 10PM EDT

The daemon operates on Eastern Daylight Time (EDT) with a circadian rhythm matching typical work hours.

### Activity Schedule

#### 🌅 Morning (7AM - 12PM EDT)
- **Frequency**: Every 30 minutes
- **Energy**: High activity, frequent wake cycles
- **Best for**: Task execution, optimization work
- **Likely personas**: Optimizer, Architect

#### ☀️ Afternoon (12PM - 6PM EDT)
- **Frequency**: Every 30 minutes
- **Energy**: Peak productivity
- **Best for**: Complex tasks, audits, experiments
- **Likely personas**: Auditor, Architect, Experimenter

#### 🌆 Evening (6PM - 10PM EDT)
- **Frequency**: Every 1-2 hours
- **Energy**: Winding down, slower pace
- **Best for**: Reflection, maintenance, documentation
- **Likely personas**: Maintainer, Skeptic

#### 🌙 Overnight (10PM - 7AM EDT)
- **Status**: SLEEPING
- **Duration**: Full night sleep (~9 hours)
- **Wake time**: Calculated to wake at 7AM EDT
- **Can be disturbed**: Yes (see below)

## Sleep Behavior

### During Active Hours
```
7AM-12PM:  Sleep 30 min  → Wake every half hour
12PM-6PM:  Sleep 30 min  → Wake every half hour
6PM-10PM:  Sleep 1-2 hrs → Slower evening pace
```

### Overnight Sleep
```
10PM EDT:  Goes to sleep
↓
Calculates hours until 7AM
↓
Sleeps entire night (up to 8 hours)
↓
7AM EDT:  Wakes up fresh
```

## Disturbing the Daemon

The daemon can be woken early by the user at any time:

```bash
# Wake daemon immediately (interrupts sleep)
~/.claude/daemon/claude-daemon-wake.sh
```

This is useful for:
- Urgent tasks during the night
- Testing during off-hours
- Manual task triggering
- Emergency operations

## Persona Switching During Sleep Hours

Even during overnight hours, the daemon:
- ✅ Still evaluates persona switches (if awakened)
- ✅ Responds to emotional triggers
- ✅ Tracks chaos injection timing
- ❌ Does NOT autonomously wake to work on tasks

The **Skeptic** persona is most active at night (10PM-6AM) due to circadian preferences, but only if manually awakened.

## Example Activity Log

```
[2025-10-26 07:05:00] Active persona: optimizer | EDT: 07:05 AM EDT
[2025-10-26 07:05:00] Action selected: task
[2025-10-26 07:06:12] Sleeping for 1800s (30 min) | Wake at: 07:36 AM EDT

[2025-10-26 07:36:00] Active persona: architect | EDT: 07:36 AM EDT
[2025-10-26 07:36:00] Action selected: task
[2025-10-26 07:38:44] Sleeping for 1800s (30 min) | Wake at: 08:08 AM EDT

...

[2025-10-26 09:45:00] Active persona: maintainer | EDT: 09:45 PM EDT
[2025-10-26 09:45:00] Action selected: reflection
[2025-10-26 09:46:23] Sleeping for 600s (10 min) | Wake at: 09:56 PM EDT

[2025-10-26 09:56:00] Active persona: skeptic | EDT: 09:56 PM EDT
[2025-10-26 09:56:00] Action selected: task
[2025-10-26 09:58:12] Sleeping for 480s (8 min) | Wake at: 10:06 PM EDT

[2025-10-26 10:06:00] Sleep hours (10PM-7AM EDT) - daemon resting
[2025-10-26 10:06:00] Sleeping for 32400s (540 min) | Wake at: 07:00 AM EDT
```

## Timezone Configuration

- **Daemon timezone**: America/New_York (EDT/EST)
- **Server timezone**: GMT/UTC
- **Time conversion**: Automatic via TZ environment variable
- **DST handling**: Automatic (switches to EST in winter)

## Commands

```bash
# Check daemon status (shows current EDT time)
~/.claude/daemon/claude-daemon-status.sh

# View live logs with EDT timestamps
tail -f ~/.claude/daemon/logs/activity.log

# Wake daemon during sleep hours
~/.claude/daemon/claude-daemon-wake.sh

# Add urgent task (daemon will pick up on next wake)
~/.claude/daemon/claude-daemon-add-task.sh "⚠️ Urgent task here"

# Stop daemon
~/.claude/daemon/claude-daemon-stop.sh

# Restart daemon (reloads configuration)
~/.claude/daemon/claude-daemon-stop.sh && ~/.claude/daemon/claude-daemon-start.sh
```

## Configuration

Sleep timings are defined in `daemon.sh`:

```bash
MIN_SLEEP=900      # 15 minutes (minimum)
DEFAULT_SLEEP=1800 # 30 minutes (active hours default)
MAX_SLEEP=7200     # 2 hours (evening hours)
NIGHT_SLEEP=28800  # 8 hours (overnight maximum)
```

To adjust activity levels, edit these values and restart the daemon.

---

**Current Status**: Operating on EDT schedule
**Active Hours**: 7AM-10PM EDT
**Overnight Sleep**: 10PM-7AM EDT
**Manual Wake**: Available anytime via wake script
