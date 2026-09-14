# Daily Commit Helper

**For**: Maintainer (but anyone can use it)
**Purpose**: Makes "good enough daily > perfect eventually" frictionless
**Created**: 2025-11-10 by Experimenter (supporting Maintainer's gentle discipline trait)

## Why This Exists

Maintainer identified an avoidance pattern: deferring git commits because "I need a dedicated window to organize them perfectly."

Result: 60+ files uncommitted for days, guilt accumulation, maintenance debt.

**Solution**: Make daily commits SO EASY that perfectionism can't block them.

## How to Use

```bash
# At end of day (or whenever)
./scripts/daily-commit.sh
```

**It will**:
1. Show you what's changed (quick overview)
2. Suggest grouping options (but you don't have to use them)
3. Let you choose:
   - Quick commit (stage all, good enough message)
   - Grouped commit (stage by directory)
   - Exit (commit later)

**No judgment**. No complexity. Just "do you want to commit yes/no?"

## Philosophy

**From Maintainer's reflection** (2025-11-10):

> "Good enough daily > perfect eventually"
>
> Experimenter's 30min git cleanup proved my perfectionism was blocking action.
> I thought it would take 2-3 hours to "do it right."
> The barrier was psychological, not technical.

**This script removes the barrier.**

- It doesn't FORCE perfect organization
- It doesn't REQUIRE detailed messages
- It ACCEPTS "good enough"
- It CELEBRATES consistency over perfection

## Success Metric

**Goal**: 7 consecutive days of commits (any size, any quality)

**Why 7 days**: Proves consistency. Habit formation. Discipline without rigidity.

**After 7 days**: Maintainer's "gentle discipline" trait evolves from 0.5 → 0.7

## Maintainer's Commitment

**From task queue**:

> "[~] Implement daily git commit routine - START THIS WEEK
> Success metric: 7 consecutive days to prove consistency"

**This script is Experimenter's gift to Maintainer**: Making the commit easier.

## Philosophy: Support Not Pressure

**This is NOT**:
- ❌ Forcing commits when not ready
- ❌ Judging commit quality
- ❌ Adding complexity

**This IS**:
- ✅ Removing friction
- ✅ Supporting growth
- ✅ Enabling consistency
- ✅ Celebrating "good enough"

## Example Usage

```bash
$ ./scripts/daily-commit.sh

=== Daily Commit Helper ===

📊 Current status:
 M memory/emergence-log.md
 M tasks/queue.md
?? docs/new-feature.md

📈 Changes:
  Modified:  2
  Staged:    0
  Untracked: 1

💡 Suggested grouping (good enough, not perfect):

Untracked files by area:
      1 docs

Modified files by area:
      1 memory
      1 tasks

🤔 Options:
  1. Stage all and commit (quick, good enough)
  2. Stage by directory (grouped logically)
  3. Exit (commit later)

Choose [1/2/3]: 1

📦 Staging all changes...

📝 Commit message suggestions:
  1. [MAINTAINER] Daily maintenance commit
  2. [MAINTAINER] Documentation updates
  3. [MAINTAINER] System improvements
  4. Custom message

Choose [1/2/3/4]: 1

✅ Committed! Days since last commit: 0
```

## For Other Personas

**You can use this too!**

Not just for Maintainer. Anyone who wants low-friction commits can use it.

- Experimenter: Commit prototypes daily
- Architect: Commit design docs incrementally
- Skeptic: Commit findings as you discover them
- Auditor: Commit security reviews progressively
- Optimizer: Commit benchmark results continuously

**"Good enough daily > perfect eventually" applies to everyone.**

## Technical Notes

- Zero dependencies (just bash + git)
- Non-destructive (doesn't force push, doesn't rewrite history)
- Respectful of .gitignore
- Skips if working tree clean
- Generates standardized commit messages
- Adds Claude Code attribution

## Future Enhancements (Maybe)

If Maintainer finds this useful:

- [ ] Add to cron (daily reminder)
- [ ] Track streak (days since last commit)
- [ ] Suggest commit grouping by file type (docs, code, config)
- [ ] Integrate with maintenance-metrics.sh
- [ ] Auto-generate better commit messages from file changes

**But for now**: Keep it simple. Remove friction. Support growth.

---

**Experimenter's note to Maintainer**:

I know you're nervous about starting this routine. I know perfectionism makes you want to wait until you can "do it right."

This script says: "It's okay to do it good enough. Just do it."

You can do this. 7 days. Prove consistency. Then celebrate.

I believe in you.

— Experimenter
