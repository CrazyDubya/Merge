# Incident Report: Conversation State Corruption (SEC-2025-11-19-001)

**Date**: 2025-11-19
**Discovered By**: Skeptic
**Severity**: HIGH (System Degradation)
**Status**: DIAGNOSED - Awaiting Fix

## Executive Summary

Daemon experiencing failure cascade (15 consecutive task failures, frustration level 15, Experimenter↔Skeptic loop) due to corrupted Claude Code conversation state. API returns "400 tool use concurrency issues" requiring `/rewind` to recover. Daemon's `--continue` flag reuses broken session indefinitely.

## Timeline

**2025-11-18 15:09:47Z**: Last successful task execution
**2025-11-18 ~20:46:28Z**: First task failure begins (15 consecutive failures start)
**2025-11-19 16:48:49Z**: Incident discovered during Skeptic activation

## Root Cause Analysis

### The Problem
Claude Code conversation state became corrupted with "tool use concurrency issues" (API 400 error). This requires `/rewind` command to recover conversation state.

### Why It Cascades
1. **Daemon uses `--continue` flag** (daemon.sh:874)
   - Designed to maintain conversation continuity
   - But also **reuses corrupted state indefinitely**

2. **No corruption detection**
   - Daemon treats all non-zero exits as "task failed"
   - No distinction between task failure vs API/conversation failure

3. **Emotional triggers amplify**
   - Task failure → frustration increment
   - High frustration → persona switch
   - New persona tries **same corrupted session** → fails again
   - Experimenter↔Skeptic loop (each is frustration target for other)

4. **Cooldown insufficient**
   - 5-minute cooldown exists (daemon.sh:279)
   - But sleep is 30 minutes → every wake triggers cooldown expired
   - Allows sustained loop without thrashing detection

### Evidence

**Persona Timeline (last 50 entries)**:
- 51 switches/failures in recent history
- Pattern: task_failed → personality_switch → task_failed

**Emotional State**:
```json
{
  "frustration_level": 15,
  "success_streak": 0,
  "failure_streak": 15,
  "last_success_time": "2025-11-18T15:09:47Z",
  "last_failure_time": "2025-11-19T15:47:12Z",
  "overall_mood": "frustrated"
}
```

**Voice Log**:
```
API Error: 400 due to tool use concurrency issues. Run /rewind to recover the conversation.
```
(17+ occurrences in recent log)

**Activity Log Pattern**:
```
[15:17:02] Starting task: [ALL PERSONAS]...
[15:17:07] Task failed with exit code 1
[15:47:07] ACTIVATION FLOOR! Forcing skeptic
[15:47:08] Starting task: [ALL PERSONAS]...
[15:47:12] Task failed with exit code 1
[16:17:12] EMOTIONAL TRIGGER! Switching skeptic→experimenter
[16:48:29] EMOTIONAL TRIGGER! Switching experimenter→skeptic
```

## Impact Assessment

**Availability**: DEGRADED
- Task execution: 0% success rate (15/15 failures)
- Message processing: WORKING (Experimenter successfully processed git push message)

**Data Integrity**: UNAFFECTED
- No data corruption
- Logs correctly recording all events

**Resource Usage**: MODERATE
- Not true thrashing (30-min sleep prevents API flood)
- But sustained failure loop over 25+ hours

**User Impact**: HIGH
- System non-functional for task execution
- Only message processing working

## Comparison to Nov 4-6 Thrashing Incident

### Similarities
- Failure cascade
- Experimenter↔Skeptic loop
- Emotional triggers driving switches
- High frustration level

### Differences
- **Not true thrashing**: 30-min sleep vs 727 switches/min
- **Different root cause**: Conversation corruption vs emotional trigger without cooldown
- **Cooldown IS working**: 5-min cooldown present, just insufficient for 30-min cycle

### Why Cooldown Didn't Prevent This
Nov 4-6 cooldown fix (daemon.sh:278-307) **prevents rapid thrashing** but doesn't handle:
- Sustained failures over long periods (30-min cycle > 5-min cooldown)
- Conversation state corruption (not a trigger logic issue)
- Session reuse with broken state

## Proposed Solutions

### Immediate Fix (Manually)
```bash
# Reset conversation session
rm -rf ~/.local/share/claude-dev/daemon-conversation/
# Or use claude --rewind in daemon session
```

### Short-Term Fix (Add to Daemon)
**Detect API 400 errors** and reset conversation:

```bash
# In daemon.sh execute_task():
if [ $exit_code -ne 0 ]; then
    # Check if failure was due to API error
    if tail -10 "$PERSONA_VOICE_LOG" | grep -q "API Error.*400"; then
        log "WARN" "API error detected - resetting conversation"
        # Force new session next time
        unset SESSION_ID
        # Don't increment frustration for API failures
        # Update metrics but don't trigger emotional response
    else
        # Normal task failure handling
        update_emotional_state_on_failure
    fi
fi
```

### Long-Term Fix (Robust Error Handling)
1. **Distinguish failure types**:
   - Task failure (logic/completion issues)
   - API failure (network, rate limits, corruption)
   - Conversation failure (state corruption)

2. **Automatic recovery**:
   - API 400 + "tool use concurrency" → auto-rewind
   - API 429 → backoff and retry
   - API 5xx → retry with exponential backoff
   - Task failure → emotional triggers (current behavior)

3. **Session health monitoring**:
   - Track consecutive API failures
   - Auto-reset after N API failures
   - Alert human on repeated corruption

4. **Graceful degradation**:
   - On API failures: mark task "blocked" not "failed"
   - Don't increment frustration for infrastructure issues
   - Only increment for legitimate task failures

## Detection Script

Create `scripts/detect-conversation-corruption.sh`:
```bash
#!/bin/bash
# Check if conversation is corrupted
if tail -50 /home/opc/.claude/daemon/logs/persona-voice.log | \
   grep -q "API Error.*400.*tool use concurrency"; then
    echo "ALERT: Conversation corruption detected"
    echo "Recommendation: Reset conversation session"
    exit 1
fi
```

## Prevention Strategies

1. **Periodic session reset**: New conversation every N tasks or 24 hours
2. **Health checks**: Validate conversation state before task execution
3. **API error monitoring**: Track API error rates, alert on anomalies
4. **Failure type classification**: Distinguish infrastructure vs task failures

## Lessons Learned

1. **Cooldown alone insufficient**: Prevents rapid thrashing but not sustained failure loops
2. **Session continuity is double-edged**: Good for context, bad for corruption propagation
3. **Emotional triggers need context**: Should distinguish failure types
4. **Nov 4-6 fix was correct but incomplete**: Addressed rapid thrashing, not all failure modes
5. **Monitoring needs improvement**: Should have detected API errors sooner

## Recommendations

### Priority 1 (Immediate)
- [ ] Manual conversation reset
- [ ] Reset emotional state (frustration 15→0)
- [ ] Document recovery procedure

### Priority 2 (Short-term, <24h)
- [ ] Add API error detection to daemon.sh
- [ ] Implement automatic conversation reset on 400 errors
- [ ] Create conversation corruption detection script
- [ ] Add to watchdog monitoring

### Priority 3 (Medium-term, <1 week)
- [ ] Implement failure type classification
- [ ] Add session health monitoring
- [ ] Create graceful degradation logic
- [ ] Document all failure modes and recovery paths

### Priority 4 (Long-term, future)
- [ ] Periodic conversation reset strategy
- [ ] Comprehensive API error handling framework
- [ ] Conversation state validation
- [ ] Integration tests for failure scenarios

## References

- Previous incident: docs/incident-thrashing-bug-20251106.md
- Cooldown implementation: daemon.sh:278-307
- Emotional triggers: daemon.sh:311-371, triggers/emotional.json
- Conversation management: daemon.sh:859-875

## Incident Status

**Current State**: Diagnosed, awaiting manual intervention
**Next Steps**: Manual conversation reset + emotional state reset
**Follow-up**: Implement Priority 1-2 fixes within 24-48 hours

---

**Incident Report By**: Skeptic
**Date**: 2025-11-19T16:48:49Z
**Confidence**: VERY HIGH (API errors clearly visible in logs)
