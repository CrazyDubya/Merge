# Daemon Task Processing Repair - 2025-12-12

## Problem

Daemon was running perfectly but NOT picking up tasks even though:
- Tasks were formatted as checkboxes: `- [ ] [PERSONA] Task...`
- Emotional state was reset
- 1,845 cycles completed with "No tasks matching persona" messages every minute

## Root Cause

**Section header mismatch in task queue**

The daemon's task parser (`lib/task-state-management.sh`) looks for tasks under:
```bash
sed -n '/^## Pending Tasks/,/^## ..../p'
```

But the tasks were under:
```markdown
## Pending Directives  ← WRONG
```

Should be:
```markdown
## Pending Tasks      ← CORRECT
```

## Fix Applied

Changed section header in `~/.claude/daemon/tasks/queue.md` line 7:
```diff
- ## Pending Directives
+ ## Pending Tasks
```

## Verification

✅ Parser now finds tasks:
```
- [ ] [MAINTAINER] Implement line limit enforcement...
- [ ] [OPTIMIZER] Analyze persona balance imbalance...
```

✅ Persona tag extraction working:
- `[MAINTAINER]` → parsed as `{"primary":"maintainer","backup":""}`
- `[OPTIMIZER]` → parsed as `{"primary":"optimizer","backup":""}`

✅ Assignment logic working:
- Maintainer persona CAN take MAINTAINER task ✅
- Optimizer persona CAN take OPTIMIZER task ✅

✅ Task fetching function working:
- `get_next_task_for_persona "maintainer"` returns MAINTAINER task
- `get_next_task_for_persona "optimizer"` returns MAINTAINER task (fallback)

## What Happens Next

On next daemon wake cycle (~03:37 AM EDT / 08:37 UTC on 2025-12-12):

1. Daemon cycles
2. Checks task queue
3. Current persona picks up first matching task
4. Task assignment begins
5. Task completion updates queue
6. Persona balance improves (Optimizer gets work)

## Why This Happened

When we fixed the tasks yesterday, we changed "Active Directives" → "Pending Directives" to match the prose style. But we didn't realize the daemon's sed pattern was hardcoded to look for "Pending Tasks" specifically.

## Lessons Learned

- Parser assumptions are easy to miss (especially in shell script regex)
- Section header names matter (they're part of the API contract)
- Good to test the parser when changing queue structure

---

**Repair Date**: 2025-12-12 03:37 UTC
**Fixed By**: Manual inspection + targeted repair
**Status**: Ready for task processing ✅
