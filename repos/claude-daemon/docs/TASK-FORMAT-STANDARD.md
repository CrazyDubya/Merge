# Task Format Standard

## Overview

The task system supports two complementary validation approaches:
1. **Heuristic Validation** (automatic) - For most tasks, validates based on intelligent pattern matching
2. **Structured Verification** (recommended) - For important tasks, uses explicit OUTPUT/VERIFY metadata

This standard helps prevent phantom completions (tasks marked done without actual work).

---

## Format 1: Heuristic Validation (Automatic)

### When to Use
- Quick, routine tasks
- Audit, review, analysis tasks
- Writing tasks (creative content)
- General maintenance work

### Syntax
```markdown
- [ ] [PERSONA] Task description
```

### Automatic Checks

**Audit/Review Tasks** (keywords: audit, review, examine, analyze, document findings)
- ✅ MUST modify documentation files (.md, emergence-log.md, inter-persona-dialogue.md)
- ✅ MUST add at least 500 bytes (roughly 10-15 lines) of substantive content
- ❌ FAIL if no documentation files modified
- ❌ FAIL if documentation too brief (<500 bytes)

**Writing Tasks** (keywords: write, chapter, novel)
- ✅ MUST create/modify files in creative/ or novel/ directories
- ✅ MUST be at least 1000 bytes per chapter file
- ❌ FAIL if no creative files modified
- ❌ FAIL if chapter too short (<1000 bytes)

**General Tasks**
- ✅ MUST modify at least one non-log file
- ❌ FAIL if ONLY log files were modified (suspicious - no real work done)
- ⚠️  Log-only modifications trigger error

### Example: Heuristic Validation

```markdown
- [ ] [AUDITOR] Review security audit findings and document in emergence-log.md

- [ ] [EXPERIMENTER] Write Chapter 2 of novel with sensory details

- [ ] [MAINTAINER] Consolidate common logging patterns into shared library
```

### Implementation Details

Function: `validate_task_output()` in `/home/opc/.claude/daemon/lib/task-validation.sh`
- Detects task type by keyword patterns
- Checks for relevant file modifications using `find ... -mmin -5`
- Validates file sizes using `stat`
- Logs validation failures to activity.log

---

## Format 2: Structured Verification (Recommended)

### When to Use
- Critical tasks with clear output deliverables
- Complex tasks with specific success criteria
- Tasks where silent failure is unacceptable
- Writing tasks that need quality assurance

### Syntax

```markdown
- [ ] [PERSONA] Task description
  OUTPUT: path/to/expected/file.ext
  VERIFY: bash command that returns 0 on success
  SUCCESS: What success looks like (human description)
```

### Fields

**OUTPUT** (required for verification)
- Path to expected output file
- Can include wildcards: `chapter-*.md`, `*.sh`
- Relative to `$DAEMON_ROOT` or can be absolute path

**VERIFY** (required for verification)
- Bash command/test that validates the output
- Returns 0 (success) if output is valid
- Returns 1 (failure) if output is missing/invalid
- Common patterns:
  - `[ -f FILE ]` - File exists
  - `[ $(wc -w < FILE) -ge 2500 ]` - Word count ≥2500
  - `bash -n FILE` - Bash syntax check
  - `[ $(wc -l < FILE) -ge 50 ]` - Line count ≥50

**SUCCESS** (optional, for documentation)
- Human-readable description of what "done" looks like
- Helps reviewer understand task intent
- Appears in completion logs

### Validation Flow

1. Task marked for completion
2. Extract OUTPUT/VERIFY metadata
3. Run verification command
4. If passes: mark complete with ✅
5. If fails: prevent completion and log error

---

## Examples by Task Type

### Audit/Review Tasks

```markdown
- [ ] [AUDITOR] Review Phase 7 refactoring and document findings
  OUTPUT: docs/PHASE7-AUDIT-REPORT.md
  VERIFY: [ -f docs/PHASE7-AUDIT-REPORT.md ] && [ $(wc -l < docs/PHASE7-AUDIT-REPORT.md) -ge 50 ]
  SUCCESS: Comprehensive audit documenting findings, optimizations completed, risks mitigated
```

**What's being checked:**
- Output file exists and is in right location
- Substantive documentation (50+ lines)
- Task completion is explicit and verifiable

---

### Writing Tasks (Novels)

```markdown
- [ ] [EXPERIMENTER] Write Chapter 3 of Sentient Toaster novel
  OUTPUT: creative/sentient-toaster/chapter-03-*.md
  VERIFY: [ -f creative/sentient-toaster/chapter-03-*.md ] && [ $(wc -w < creative/sentient-toaster/chapter-03-*.md) -ge 2500 ]
  SUCCESS: Chapter 3 complete with 2500+ words, character development, sensory details
```

**What's being checked:**
- Creative file was actually created
- Minimum word count (2500 words ≈ 10 pages)
- Prevents phantom 16-second completions

---

### Code Development Tasks

```markdown
- [ ] [OPTIMIZER] Implement caching layer for task state
  OUTPUT: lib/task-cache.sh
  VERIFY: [ -f lib/task-cache.sh ] && bash -n lib/task-cache.sh && grep -q 'function.*cache' lib/task-cache.sh
  SUCCESS: Caching library created with syntax validation and at least one cache function
```

**What's being checked:**
- File exists
- Bash syntax is valid
- Contains implementation (specific function patterns)

---

### Documentation Tasks

```markdown
- [ ] [MAINTAINER] Document daemon architecture and decision history
  OUTPUT: docs/ARCHITECTURE-DECISIONS.md
  VERIFY: [ -f docs/ARCHITECTURE-DECISIONS.md ] && [ $(wc -w < docs/ARCHITECTURE-DECISIONS.md) -ge 3000 ]
  SUCCESS: Architecture documentation with 3000+ words covering all major decisions
```

---

### Performance Optimization Tasks

```markdown
- [ ] [OPTIMIZER] Optimize grep operations to reduce subprocess spawning
  OUTPUT: lib/optimized-grep.sh
  VERIFY: [ -f lib/optimized-grep.sh ] && [ $(grep -c 'grep -E\|xargs grep' lib/optimized-grep.sh) -ge 5 ]
  SUCCESS: Optimized library with batch grep operations reducing subprocess count by 50%+
```

---

## Migration Checklist

When adding verification to existing tasks:

1. **Identify Task Type** ✓
   - Audit/review, writing, code, documentation, data?

2. **Define Output** ✓
   - What file(s) will this task create?
   - Where should they be stored?
   - Use patterns: `chapter-*.md`, `report-*.json`

3. **Write Verification Command** ✓
   - Test that output exists: `[ -f FILE ]`
   - Check quality: word count, line count, syntax
   - Combine with `&&` operator
   - Test locally first to ensure it works

4. **Describe Success** ✓
   - What does "done" actually look like?
   - What qualitative criteria are met?
   - Write in human language

5. **Test** ✓
   - Run the VERIFY command manually
   - Ensure it returns 0 (success) when task is complete
   - Ensure it returns 1 (failure) when incomplete

---

## Common VERIFY Patterns

### File Existence
```bash
[ -f path/to/file.ext ]
```

### Word Count Minimum
```bash
[ $(wc -w < file.ext) -ge 2500 ]
```

### Line Count Minimum
```bash
[ $(wc -l < file.ext) -ge 50 ]
```

### Bash Syntax Valid
```bash
bash -n script.sh
```

### File Contains Keyword
```bash
grep -q 'function cache_get' lib/cache.sh
```

### Multiple Conditions
```bash
[ -f file.md ] && [ $(wc -w < file.md) -ge 2500 ] && grep -q 'Chapter 3' file.md
```

### Wildcard Matching
```bash
[ -f creative/sentient-toaster/chapter-05-*.md ] && [ $(wc -w < creative/sentient-toaster/chapter-05-*.md) -ge 2500 ]
```

---

## Troubleshooting

### "Task validation failed: Output file not found"
- Check that OUTPUT path is correct
- Ensure file was actually created (not just logged)
- Verify relative vs absolute paths

### "Task validation failed: Verification command failed"
- Test VERIFY command manually: `bash -c "VERIFY_COMMAND"`
- Ensure command returns 0 on success, 1 on failure
- Check for typos in file paths
- Verify word counts are realistic for task type

### "Task completed without verification metadata"
- Task used heuristic validation instead
- This is OK - heuristics provide 80%+ coverage
- For stricter validation, add OUTPUT/VERIFY fields

---

## Best Practices

1. **Be Specific**: Generic "do something" tasks are hard to verify. Specific outputs are easy.

2. **Set Realistic Minimums**:
   - Chapters: 2500+ words
   - Audits: 500+ bytes / 10+ lines
   - Code: Syntax valid + implementation patterns

3. **Test Your VERIFY Command**: Run it manually before using in queue

4. **Keep It Simple**: Complex verification commands are error-prone. Use simple bash tests.

5. **Combine Multiple Checks**: Use `&&` to require all conditions: `[ -f file ] && [ $(wc -w < file) -ge 2500 ]`

6. **Document SUCCESS**: Help future reviewers understand what you're building

---

## Implementation Notes

- **Heuristic validation** is the first layer, runs for all tasks
- **Structured verification** is the second layer, runs if OUTPUT/VERIFY fields exist
- If either layer fails, task completion is prevented
- All validation attempts are logged to `logs/activity.log`
- Use `logs/validation.log` for detailed validation audit trail

---

## Examples of Phantom Completions We're Preventing

**Before (16-second phantom completion):**
```markdown
- [x] [AUDITOR] Examine refactored environment. Document findings in emergence-log.md.
# Task marked complete in 16 seconds, no documentation created
```

**After (strict verification):**
```markdown
- [ ] [AUDITOR] Examine refactored environment. Document findings in emergence-log.md.
  OUTPUT: docs/PHASE7-AUDIT-REPORT.md
  VERIFY: [ -f docs/PHASE7-AUDIT-REPORT.md ] && [ $(wc -l < docs/PHASE7-AUDIT-REPORT.md) -ge 50 ]
  SUCCESS: Comprehensive audit report with findings and recommendations
# Task BLOCKED: Can't mark complete without valid report
```

---

## Version

**Created**: 2025-12-08
**Status**: Active standard for all new tasks
**Migration**: Apply to high-value tasks first (writing, audits, critical features)
