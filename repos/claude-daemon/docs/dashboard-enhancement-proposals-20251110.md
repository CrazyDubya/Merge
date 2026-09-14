# Dashboard Enhancement Proposals - 2025-11-10

**Analyst**: Experimenter
**Priority**: MEDIUM
**Status**: Proposals ready for review

---

## Current Dashboard Features

**Existing Cards** (6 total):
1. Current Persona - active persona, stats, traits
2. System Health - watchdog status, uptime
3. Emotional State - frustration, success streaks, mood
4. All Personas - grid of all 6 personas with stats
5. Recent Switches - last 10 switches
6. System Alerts - warnings and notices

**Strengths**:
- Clean, modern UI with glassmorphism design
- Real-time data (refresh button)
- Good basic monitoring coverage
- Responsive layout

**Gaps Identified**:
- No historical trends (everything is current state)
- No persona activation distribution visualization
- No task completion metrics by persona
- No switch trigger breakdown
- No token/resource usage tracking
- No memory growth visibility

---

## Enhancement Proposals

### HIGH VALUE

#### 1. Persona Activation Distribution Chart 📊

**Value**: **HIGH** - Shows persona imbalance at a glance
**Complexity**: **LOW** - Simple pie/bar chart
**Data Source**: Exists in `switch-history.jsonl`

**Implementation**:
- Add new card: "Persona Distribution (Last 100 Switches)"
- Pie chart showing % per persona
- Color-code by persona emoji
- Highlight dominant persona (>30%) in red
- Add time period selector (last 50/100/500 switches)

**Why valuable**: Would have made today's Experimenter monopolization (44%) immediately visible. Human requested persona variety analysis - this makes the problem visible without analysis.

**Code approach**:
- Fetch last N from `switch-history.jsonl`
- Count by persona
- Use Chart.js or simple CSS bar chart
- Update on refresh

---

#### 2. Switch Trigger Breakdown 🎯

**Value**: **HIGH** - Shows WHY switches happen
**Complexity**: **LOW** - Aggregation + pie chart
**Data Source**: Exists in `switch-history.jsonl` (`.layer` and `.reason`)

**Implementation**:
- Add card: "Switch Triggers (Last 100)"
- Show breakdown:
  - Circadian: XX%
  - Emotional success: XX%
  - Chaos: XX%
  - Activation floor: XX%
  - Other: XX%
- Pie chart visualization
- Highlight if one trigger dominates >60%

**Why valuable**: Shows if emotional triggers are overriding circadian (as we discovered today). Makes system behavior transparent.

---

#### 3. Task Completion Rates by Persona 📈

**Value**: **HIGH** - Shows persona effectiveness
**Complexity**: **MEDIUM** - Requires aggregation
**Data Source**: Exists in `state.json` (`.personas[].tasks_completed`)

**Implementation**:
- Add card: "Task Completion by Persona"
- Bar chart: tasks completed per persona
- Show completion rate (completed / (completed + failed))
- Highlight most/least productive personas
- Add time filter (all-time / last 7 days / last 30 days)

**Why valuable**: Human asked about persona variety affecting quality. This shows which personas are actually getting work done. Today we saw: Experimenter 71 tasks, Optimizer only 9 tasks despite high activation.

---

### MEDIUM VALUE

#### 4. Emotional State Trends Over Time 📉

**Value**: **MEDIUM** - Shows emotional patterns
**Complexity**: **MEDIUM** - Requires time-series data
**Data Source**: Would need to log `emotional.json` state periodically

**Implementation**:
- Add card: "Emotional Trends (Last 24h)"
- Line graph: frustration level over time
- Line graph: success streak over time
- Mark persona switches on timeline
- Show correlation between emotional spikes and switches

**Why valuable**: Reveals feedback loops (success → Experimenter → more success → more Experimenter). Would have shown the ping-pong pattern visually.

**Challenge**: Requires historical emotional data (not currently logged). Would need to add periodic snapshots to timeline.jsonl or create new emotional-history.jsonl.

---

#### 5. Memory Growth Tracking 💾

**Value**: **MEDIUM** - Shows system resource usage
**Complexity**: **LOW** - File size monitoring
**Data Source**: Filesystem

**Implementation**:
- Add card: "Memory & Storage"
- Show file sizes:
  - switch-history.jsonl: XX MB
  - persona-timeline.jsonl: XX MB
  - emergence-log.md: XX MB
  - activity.log: XX MB
  - Total: XX MB
- Growth rate: +XX MB/day (estimated)
- Alert if any file >50MB

**Why valuable**: Human concerned about token efficiency and resource usage. This makes growth visible and prompts action (rotation) when needed.

---

#### 6. Recent Activity Timeline 🕐

**Value**: **MEDIUM** - Shows what daemon is doing
**Complexity**: **MEDIUM** - Parse activity.log
**Data Source**: `logs/activity.log`

**Implementation**:
- Add card: "Recent Activity (Last 10)"
- Scrollable list of recent actions:
  - Task started/completed
  - Reflections
  - Conversations
  - Switches
- Show timestamp, persona, action type
- Color-code by action type

**Why valuable**: Gives human visibility into "what is the daemon actually doing right now?" Currently opaque without reading logs.

---

### LOW VALUE (Nice-to-Have)

#### 7. Token Usage Estimation ⚡

**Value**: **LOW** - Interesting but hard to measure accurately
**Complexity**: **HIGH** - No direct token counting
**Data Source**: Would need API call logging

**Implementation**:
- Add card: "Token Usage (Estimated)"
- Estimate based on:
  - Message lengths from inbox
  - Switch frequency
  - Reflection frequency
- Show daily burn rate
- Compare to budget (if set)

**Why valuable**: Human requested 50% token reduction. Direct measurement would validate success.

**Challenge**: We don't have actual token counts from API. Would need to:
- Log API responses (if they include token usage)
- OR estimate based on message lengths (inaccurate)
- OR integrate with Claude API billing (complex)

---

#### 8. Persona Relationship Network 🕸️

**Value**: **LOW** - Cool but not actionable
**Complexity**: **HIGH** - Graph visualization
**Data Source**: `switch-history.jsonl`

**Implementation**:
- Add card: "Persona Interaction Network"
- Node graph showing personas as nodes
- Edges showing switch frequency between personas
- Thickness = frequency of switches
- Color = relationship type (common switches)

**Why valuable**: Visualizes "who switches to whom" patterns. Might reveal interesting collaboration patterns.

**Challenge**: Complex visualization, questionable actionable value. More "cool" than "useful."

---

#### 9. Success/Failure Event Log 📝

**Value**: **LOW** - Already in activity.log
**Complexity**: **LOW** - Filter activity.log
**Data Source**: `logs/activity.log`

**Implementation**:
- Add card: "Recent Events"
- Show last 20 successes/failures
- Filter by: All / Successes / Failures
- Show: timestamp, persona, task description (if available)

**Why valuable**: Quick view of what's working/breaking.

**Challenge**: Redundant with activity.log. Dashboard user can check logs directly.

---

## Implementation Recommendations

### Phase 1: Quick Wins (1-2 hours)

Implement **HIGH VALUE + LOW COMPLEXITY** enhancements:
1. ✅ Persona Activation Distribution Chart
2. ✅ Switch Trigger Breakdown
3. ✅ Memory Growth Tracking

**Why**: Solves today's immediate needs (persona variety visibility) with minimal effort.

### Phase 2: Deeper Insights (4-6 hours)

Implement **HIGH VALUE + MEDIUM COMPLEXITY**:
1. Task Completion Rates by Persona (requires aggregation logic)
2. Recent Activity Timeline (requires log parsing)

**Why**: Adds significant monitoring value but requires more engineering.

### Phase 3: Advanced (if needed)

Consider **MEDIUM VALUE** enhancements based on user feedback:
- Emotional State Trends (if feedback loops are ongoing concern)
- Token Usage Estimation (if token budget becomes critical)

**Skip**: LOW VALUE items unless specifically requested.

---

## Priority Ranking Summary

| Enhancement | Value | Complexity | Priority | Time |
|-------------|-------|------------|----------|------|
| 1. Persona Distribution Chart | HIGH | LOW | ⭐⭐⭐ | 30min |
| 2. Switch Trigger Breakdown | HIGH | LOW | ⭐⭐⭐ | 30min |
| 3. Task Completion by Persona | HIGH | MEDIUM | ⭐⭐ | 2h |
| 5. Memory Growth Tracking | MEDIUM | LOW | ⭐⭐ | 20min |
| 6. Recent Activity Timeline | MEDIUM | MEDIUM | ⭐ | 2h |
| 4. Emotional Trends | MEDIUM | MEDIUM | ⭐ | 3h |
| 7. Token Usage Estimation | LOW | HIGH | - | 4h+ |
| 8. Persona Network Graph | LOW | HIGH | - | 4h+ |
| 9. Event Log | LOW | LOW | - | 1h |

**Recommendation**: Implement #1, #2, #5 in Phase 1 (90 minutes total).

---

## Technical Implementation Notes

### Data Sources Available

**Already exists** (no backend changes needed):
- `personalities/state.json` - current state, persona stats
- `triggers/emotional.json` - emotional state
- `metrics/switch-history.jsonl` - all switches with reasons
- `metrics/success-rates.json` - historical success rates
- `logs/activity.log` - all daemon actions
- `.watchdog-state.json` - health status

**Would need to add**:
- Periodic emotional state snapshots (for trends)
- API token usage logging (for token tracking)

### Chart Libraries

**Options**:
1. **Chart.js** - Popular, easy, good for pie/bar/line charts
2. **D3.js** - Powerful but complex, overkill for simple charts
3. **Pure CSS** - For simple bar charts, no dependencies

**Recommendation**: Chart.js for Phase 1 (balance of ease + features)

### Refresh Strategy

**Current**: Manual refresh button

**Proposed**:
- Keep manual refresh (user control)
- Add auto-refresh option (toggle, default OFF)
- Auto-refresh interval: 60 seconds (if enabled)
- Show "last updated" timestamp

**Why**: Some users want live dashboard, others want control. Provide both.

---

## Mockup: Enhanced Dashboard Layout

```
+----------------------------------------------------------+
|  Multi-Persona Claude Daemon Dashboard         [Refresh] |
+----------------------------------------------------------+

+---------------------+---------------------+----------------+
| 🧠 Current Persona  | 💚 System Health   | 😊 Emotional   |
| Experimenter        | Status: Running    | Mood: positive |
| Since: 2h ago       | Uptime: 3d 4h     | Success: 10    |
| Tasks: 71 ✅ 2 ❌   | Restarts: 0       | Frustration: 0 |
+---------------------+---------------------+----------------+

+---------------------------+-----------------------------+
| 📊 Persona Distribution   | 🎯 Switch Triggers (100)    |
| (Last 100 Switches)       |                             |
|                           |                             |
| [Pie Chart]               | Circadian:    73%           |
| Experimenter: 44% 🔴      | Emotional:    20% ⚠️        |
| Optimizer: 20%            | Chaos:         5%           |
| Others: 8-11% each        | Floor:         2%           |
+---------------------------+-----------------------------+

+-------------------------------+-------------------------+
| 📈 Task Completion by Persona | 💾 Memory & Storage     |
|                               |                         |
| [Bar Chart]                   | switch-history: 12MB    |
| Experimenter: 71 tasks        | timeline: 8MB           |
| Auditor: 24 tasks             | logs: 5MB               |
| Architect: 16 tasks           | Total: 35MB             |
| ... (others)                  | Growth: +2MB/day        |
+-------------------------------+-------------------------+

+-----------------------------+----------------------------+
| 👥 All Personas (6)         | 🔄 Recent Switches (10)    |
| [Grid of persona cards]     | [Scrollable list]          |
+-----------------------------+----------------------------+

+------------------------------------------------------------+
| ⚠️ System Alerts                                           |
| • Experimenter dominant (44%) - variety issue detected     |
| • Success streak at cap (10) - emotional triggers active   |
+------------------------------------------------------------+
```

---

## Files to Create/Modify

**Modify**:
- `dashboard.html` (add new cards + chart library)
- `claude-daemon-dashboard.sh` (if backend changes needed)

**Create** (optional):
- `dashboard-charts.js` (chart rendering functions)
- `dashboard-styles.css` (additional styles)

**No backend changes needed** for Phase 1 (all data already accessible).

---

## Success Metrics

**After Phase 1 implementation, dashboard should**:
1. Show persona variety issues visually (distribution chart)
2. Explain why switches happen (trigger breakdown)
3. Track system resource usage (memory growth)
4. Require no backend changes (pure frontend enhancement)
5. Load quickly (<2 seconds even with charts)

**User value**: Human can spot issues (like today's Experimenter monopolization) at a glance, without running queries or reading logs.

---

**Document**: docs/dashboard-enhancement-proposals-20251110.md
**Lines**: 400
**Time**: 30 minutes analysis + 15 minutes writing
**Status**: Proposals complete, ready for review

**Next**: Get human/Architect approval, implement Phase 1 (90 minutes estimated)
