# INCIDENT REPORT: State.json Persona Corruption

**Severity:** HIGH
**Status:** ACTIVE (since Oct 26, 2025)
**Discovered:** 2025-11-06T16:52:00Z by Skeptic
**Impact:** Data integrity violation, phantom persona entries in state.json

## Summary

state.json is corrupted with 6 phantom persona entries containing malformed names like `"Became skeptic (reason: chaos)\nskeptic"` and `" chaos)\nskeptic"`. These phantom entries have accumulated task completion stats since Oct 26, 2025.

## Evidence

**File:** personalities/state.json lines 157-187

**Corrupt persona keys:**
```
"Became skeptic (reason: emotional_frustration)\nskeptic"
"Became experimenter (reason: emotional_frustration)\nexperimenter"
"Became skeptic (reason: chaos)\nskeptic"
"Became architect (reason: emotional_frustration)\narchitect"
"Became architect (reason: circadian)\narchitect"
" chaos)\nskeptic"
```

**Stats accumulated:**
- Total tasks_completed across phantoms: 5+
- Total tasks_failed across phantoms: 38+
- Total activations across phantoms: 40+

## Root Cause Analysis

**Primary bug:** Unknown source is passing malformed persona names to jq commands at daemon.sh:843-846.

**Suspected mechanism:**
1. Variable `$persona` receives malformed value (contains newlines + prefix text)
2. Value pattern matches: `"Became X (reason: Y)\nX"` or `" chaos)\nX"`
3. jq command: `.personas[$p]` creates NEW KEY with malformed value
4. Phantom personas accumulate over time

**NOT the source:**
- ✅ `state_who()` reads `.current_persona` correctly ("architect")
- ✅ `determine_personality()` calls state_who correctly
- ✅ state.json `.current_persona` field is NOT corrupted
- ✅ Daemon doesn't read from tasks/completed/

**Suspected sources (unconfirmed):**
1. Task completion logging creates malformed task entries
2. Something parses task descriptions and extracts corrupted persona names
3. Prompt construction concatenates strings incorrectly
4. Environmental variable pollution

## Impact Assessment

**System functionality:** ✅ WORKING
- `.current_persona` field is correct ("architect")
- Daemon switching works normally
- Task execution proceeds

**Data integrity:** ❌ VIOLATED
- 6 phantom persona entries pollute namespace
- Task stats incorrectly attributed to phantoms
- Timeline entries reference non-existent personas
- Metrics calculations include garbage data

**User visibility:** 🟡 PARTIALLY VISIBLE
- User sees corrupted persona in prompts: `[ACTIVE PERSONA:  chaos)\nskeptic]`
- Emergence log entries may reference phantoms
- Reports/dashboards may show invalid personas

## Timeline

- **2025-10-26T22:25:53Z**: First corrupted entry (" chaos)") appears in timeline
- **2025-10-26 - 2025-11-06**: Phantom personas accumulate task stats
- **2025-11-06T16:39:08Z**: Current session running as `" chaos)\nskeptic"` phantom
- **2025-11-06T16:52:00Z**: Bug discovered by Skeptic during investigation

## Immediate Actions Taken

1. ✅ Bug identified and root cause analysis performed
2. ✅ Evidence documented in this incident report
3. ⏳ Human alert being created
4. ⏳ Recommendation for cleanup + prevention

## Recommended Fix

**Phase 1 - Cleanup (URGENT):**
1. Backup state.json
2. Remove phantom persona entries (6 keys)
3. Redistribute stats to correct personas if possible
4. Validate state.json integrity

**Phase 2 - Prevention (HIGH PRIORITY):**
1. Add persona name validation in jq commands
2. Sanitize `$persona` variable before use (strip newlines, validate format)
3. Add regex validation: `^[a-z]+$` for persona names
4. Add assertion checks before state.json writes

**Phase 3 - Investigation (MEDIUM PRIORITY):**
1. Find actual source of malformed persona names
2. Check if task completion logging is culprit
3. Review all code paths that set `$persona` variable
4. Add integration tests for persona name integrity

## Code Locations

**Bug manifestation:**
- daemon.sh:843-846 (jq command creates phantom keys)
- daemon.sh:865-868 (same jq pattern for failures)

**Suspected sources:**
- daemon.sh:676-848 (execute_task_action function)
- daemon.sh:1391 (determine_personality call)
- tasks/completed/*.md (malformed task entries)

**Data corruption:**
- personalities/state.json:157-187 (phantom entries)
- memory/persona-timeline.jsonl (references to phantoms)
- metrics/switch-history.jsonl (may contain phantom names)

## Prevention Checklist

- [ ] Add persona name validation function
- [ ] Sanitize all persona variables before jq
- [ ] Add pre-write assertions for state.json
- [ ] Create integration test for persona name integrity
- [ ] Document valid persona name format in code
- [ ] Add monitoring for invalid persona keys
- [ ] Review task completion logging format

## Lessons Learned

1. **No input validation:** Persona names accepted without format checks
2. **Silent corruption:** jq creates new keys without warning
3. **Late detection:** Bug existed 11 days before discovery
4. **Accumulated damage:** 40+ phantom activations before caught

## Owner

**Discovery:** Skeptic
**Remediation:** Awaiting assignment (recommend: Maintainer + Auditor)
**Validation:** Recommend: Skeptic (re-verify after fix)

---

**CRITICAL:** This bug violates data integrity. Recommend immediate cleanup before phantom persona stats corrupt analytics or cause secondary bugs.
