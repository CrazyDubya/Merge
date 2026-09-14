# MADagent Reliability Refactoring Plan

## Evaluation: Current State vs. 10 Reliability Rules

### Codebase Summary
- 2,779 lines of source across 10 modules
- 2,008 lines of tests (175 passing)
- Python 3.10+, zero runtime dependencies
- Architecture: event-driven agent with governance, ledger, memory, planning, execution, reflection subsystems

---

## Rule-by-Rule Evaluation

### Rule 1: Keep functions short (~60 lines max, one thing each)

**Current violations:**

| File | Function | Lines | Problem |
|------|----------|-------|---------|
| `agent.py` | `Agent.__init__` | 166-253 (87 lines) | Does 10+ things: config, identity, stakes, ledger, memory, models, governance, planner, executor, reflection, event_bus, adapters, wiring, state loading, boot event |
| `agent.py` | `process_input` | 286-401 (115 lines) | 14-step pipeline in a single function |
| `agent.py` | `_persist_state` | 559-596 (37 lines) | Serializes 5 different entities — borderline, but each serialization block is repetitive boilerplate |
| `agent.py` | `_load_persisted_state` | 598-692 (94 lines) | Deserializes 6 entities with field-by-field assignment |
| `agent.py` | `resolve_escalation` | 485-538 (53 lines) | Acceptable length but mixes validation, ledger ops, and execution |
| `planner.py` | `generate_candidates` | 89-168 (79 lines) | Builds 3 plan variants inline |
| `planner.py` | `score_plan` | 170-218 (48 lines) | Acceptable |
| `governance.py` | `evaluate` | 92-155 (63 lines) | Borderline — orchestrates 6 checks plus verdict logic |
| `reflection.py` | `reflect` | 77-131 (54 lines) | Borderline — 7 steps |

**What's good now:** Most helper functions (`_check_boundaries`, `_check_tool_permissions`, etc.) are well-scoped and short. The smaller modules (event_bus, adapters, memory stores) are well-factored.

**What's bad now:** `agent.py` at 779 lines is carrying too much weight. `__init__`, `process_input`, and `_load_persisted_state` all exceed the 60-line limit and do multiple unrelated things.

**Refactoring plan:**
1. **Split `Agent.__init__`** into `_init_subsystems()`, `_init_models()`, and `_boot()`. The constructor becomes a ~15-line orchestrator.
2. **Split `process_input`** into pipeline stages: `_ingest_and_check(text, trace_id)` -> `_plan_and_govern(intent)` -> `_execute_and_record(plan, trace_id)`. Each stage returns a typed intermediate result or an early-exit error dict.
3. **Extract `_persist_state` and `_load_persisted_state`** into a new `persistence.py` module with a `StatePersistence` class. Each entity type gets its own `_save_X` / `_load_X` method.
4. **Split `generate_candidates`** so each plan variant (direct, investigate, cautious) is built by its own `_build_X_plan` helper.
5. **Split `evaluate`** in governance: extract `_determine_verdict(reasons)` as its own function (the set-matching + verdict logic).

### Rule 2: Assert liberally (pre/post conditions, parameter validity, invariants)

**Current violations:**
- `reflection.py:116-119` — Correctly asserts commitments/boundaries are preserved. This is the *only* substantive assertion in the codebase.
- `ledger.py:174` — `assert self._path is not None` in `_persist_entry`. Good.
- **Missing everywhere else:**
  - `Stakes.deduct()` does not assert `cost >= 0` or that budget won't go negative.
  - `Stakes.penalize_reputation()` does not assert `amount >= 0`.
  - `Ledger.append()` does not validate `entry_type` is non-empty.
  - `Plan.to_dict()` does not validate steps are non-empty.
  - `GovernanceLayer.evaluate()` does not assert plan is not None or has steps.
  - `Executor.execute_plan()` does not validate plan has steps.
  - `MemorySystem.ingest_event()` does not validate event is well-formed.
  - `EventBus.publish()` does not validate event type.
  - `InputAdapter.user_text()` does not assert text is non-empty.
  - `WorkingCache.push()` does not validate item.
  - `LedgerEntry.compute_chain_hash()` does not assert content_hash is already set.

**What's good now:** The reflection invariant check is exactly right — it snapshots before and asserts after.

**What's bad now:** Almost no defensive checks. A None plan, empty text, negative cost, or malformed event would silently propagate and fail downstream with confusing errors.

**Refactoring plan:**
1. Add guard clauses at the top of every public method:
   - `Ledger.append()`: assert `entry_type` is truthy, `isinstance(data, dict)`
   - `Stakes.deduct(cost)`: `assert cost >= 0, "cost must be non-negative"`
   - `Stakes.penalize_reputation(amount)`: `assert amount >= 0`
   - `Executor.execute_plan(plan)`: `assert plan.steps, "plan must have steps"`
   - `GovernanceLayer.evaluate(plan)`: `assert plan is not None`
   - `EventBus.publish(event)`: `assert isinstance(event, Event)`
   - `InputAdapter.user_text(text)`: `assert isinstance(text, str) and text.strip()`
   - `LedgerEntry.compute_chain_hash()`: `assert self.content_hash, "compute_content_hash first"`
2. Add post-condition assertions:
   - `Ledger.append()`: assert new entry's sequence == expected sequence after append
   - `Executor._execute_step()`: assert bundle always has timestamps and hashes set

### Rule 3: Minimize variable scope

**Current violations:**
- `agent.py:333-365` — `allowed` and `escalated` lists are declared 30+ lines before they're fully consumed. The governance loop, the escalation queuing, and the "no allowed plans" check could be a single extracted function.
- `planner.py:96-168` — `candidates`, `available`, `tool`, `params` are all declared at function top and used throughout a long function. Extract sub-builders.
- `agent.py:584-596` — `facts`, `facts_data` are declared in _persist_state mid-function. Fine on its own but the function is too long.
- `governance.py:94-96` — `reasons` and `required_mods` accumulate across the entire `evaluate` method. Acceptable given the control flow, but would benefit from verdict extraction.

**What's good now:** Most modules use tight scoping. The data models are clean dataclasses with no extraneous state. Memory stores use focused methods.

**What's bad now:** The long functions in agent.py naturally create wide scopes. Variables declared for step 3 of a 14-step pipeline are still alive at step 14.

**Refactoring plan:**
- The function splitting from Rule 1 will naturally fix most scope issues.
- Specifically: the governance evaluation loop in `process_input` should become `_evaluate_candidates(ranked, trace_id)` returning `(allowed: list[Plan], escalation_response: dict | None)`.

### Rule 4: Check all return values and validate all inputs

**Current violations:**
- `agent.py:302` — `self.event_bus.publish(event, priority=0.1)` return value (bool, False = backpressure) is **never checked**. The bus could silently drop the event.
- `agent.py:479` — Same in `inject_observation`.
- `agent.py:337` — `self.ledger.append_governance(decision)` return value (LedgerEntry) is ignored. Acceptable if we trust ledger, but the entry could be useful for trace correlation.
- `agent.py:380` — `self.executor.evidence_log[-len(chosen.steps):]` — if evidence_log has fewer entries than plan steps (possible on partial execution), this silently takes wrong slice.
- `agent.py:527` — Same pattern in `resolve_escalation`.
- `agent.py:553` — `self.reflection_engine._ticks_since_reflection = 0` directly accesses a private field, bypassing any validation.
- `memory.py:128` — `self.ledger.append_event(event)` return value ignored.
- `reflection.py:94` — `self.ledger.recent(self.reflection_interval)` — doesn't check if returned list is empty before processing.
- `executor.py:196` — `result = tool_fn(step.params)` — no type check on return value. If a registered tool returns something other than ToolResult, it will fail with AttributeError.
- `event_bus.py:82-90` — `process_all()` has no upper bound — if handlers publish new events during processing, this could loop indefinitely.

**What's good now:** Tool functions (`read_file_tool`, `list_dir_tool`, etc.) validate their inputs thoroughly and always return ToolResult.

**What's bad now:** The agent layer often ignores return values from subsystems, especially EventBus.publish (backpressure signal) and Ledger.append (confirmation). The evidence_log slicing is fragile.

**Refactoring plan:**
1. Check `event_bus.publish()` return and log/handle backpressure.
2. Validate tool function return types in `Executor._execute_step`.
3. Fix evidence_log slicing: track evidence per-execution instead of global slicing.
4. Add bounded iteration to `EventBus.process_all()` (Rule 7 overlap).

### Rule 5: Compile clean; analyze clean

**Current status:**
- No type checker configuration (no mypy.ini, no pyproject.toml mypy section).
- No linting configuration (no ruff, flake8, or pylint config).
- `import hashlib` appears at module top in `models.py` AND locally inside `ledger.py:142` (inside `verify_integrity`). The local import is unnecessary and inconsistent.
- `_AUTONOMY_TOOLS` in planner.py **duplicates** `TOOL_PERMISSIONS` in governance.py — same data, two sources of truth.
- `executor.py` imports `os` but never uses it.
- Some `Any` types are used where more specific types would work (e.g., `payload: dict[str, Any]` could sometimes be narrower).

**What's good now:** The code uses type hints throughout. Dataclasses enforce structure. No obvious shadowing or name collisions.

**What's bad now:** Without CI enforcement (mypy strict, ruff), type errors and dead code can creep in. The duplicate tool permission table is a concrete correctness risk — if one is updated and the other isn't, governance and planning disagree on what tools are allowed.

**Refactoring plan:**
1. Remove `import os` from executor.py.
2. Move the local `import hashlib` in `ledger.py:142` to the top of the file.
3. Consolidate `TOOL_PERMISSIONS` and `_AUTONOMY_TOOLS` into a single source of truth in `models.py` or a new `permissions.py`, imported by both governance and planner.
4. Add `[tool.mypy]` section to pyproject.toml with `strict = true`.
5. Add a `[tool.ruff]` section for linting.

### Rule 6: Prefer simple control flow

**Current status:**
- No recursion in the codebase. Good.
- No goto, no nonlocal jumps. Good.
- `InputAdapter._redact_sensitive` (adapters.py:70-81) **recursively** processes nested dicts. This is unbounded — a deeply nested dict could blow the stack.
- `EventBus.process_all` calls `process_one` in a loop — simple and clear.
- `agent.py:process_input` is a linear pipeline with early returns — good structure, just too long.

**What's good now:** Control flow is predominantly iterative and linear. The governance checks are sequential and independent — easy to reason about.

**What's bad now:** The recursive `_redact_sensitive` has no depth limit. In a system designed to ingest external observations, this is a real risk.

**Refactoring plan:**
1. Add a `max_depth` parameter to `_redact_sensitive` with a default of 10. Convert to iterative with an explicit stack, or keep recursive but with a depth counter that stops and returns `"[REDACTED:too_deep]"` at the limit.
2. Keep the overall linear pipeline structure — it's a strength.

### Rule 7: Give every loop a reason to terminate

**Current violations:**
- `event_bus.py:82-90` — `process_all()` uses `while True` with no upper bound. If a handler publishes events during dispatch, this loop may not terminate.
- `agent.py:442` — `run_forever()` while `self._running` — intentionally infinite but `self._running` is set from another thread with no synchronization guarantees. The `time.sleep` makes this practically safe but it violates the principle.
- `agent.py:460` — Same pattern in `run_forever_async`.

**What's good now:** `WorkingCache` has a bounded max_size. `EventBus` has max_queue_size for backpressure. Most loops iterate over finite collections.

**What's bad now:** `process_all()` is the main concern — it's a correctness risk, not just theoretical.

**Refactoring plan:**
1. `EventBus.process_all()`: Add `max_iterations` parameter (default: `max_queue_size * 2`). After the limit, log a warning and return.
2. `run_forever` / `run_forever_async`: Add a `max_ticks: int | None = None` parameter. If set, stop after N ticks. Document that the default (None) means truly infinite with `_running` as the only exit.

### Rule 8: Own your memory

**Current violations:**
- `Executor._evidence` (executor.py:126) — Unbounded list that grows forever. Every plan execution appends evidence bundles, but nothing ever trims or clears this list. Over a long-running agent, this is a memory leak.
- `EventBus._subscribers` and `_global_subscribers` — Never cleaned up. Subscribers accumulate. In practice this is fine since wiring happens once, but there's no unsubscribe mechanism if the agent is reconfigured.
- `SemanticStore._facts` — Unbounded dict. In a long-running agent, this grows without limit. The ledger is the source of truth and can be replayed, so the semantic store could be bounded.
- `VectorIndex._documents` — Same concern. Grows without bound.
- `Ledger._entries` — Grows without bound in memory. The JSONL file is the persistent backing, but the entire history is also held in RAM. For a long-running agent, this needs a strategy (e.g., only keep last N entries in memory, page from disk for older ones).

**What's good now:** WorkingCache has FIFO eviction with max_size — exactly right. No manual memory management issues (this is Python). No freed-memory-access concerns.

**What's bad now:** Several unbounded collections will grow indefinitely in a `run_forever` scenario. The evidence_log in Executor is the most acute issue since it's write-only with no consumer that clears it.

**Refactoring plan:**
1. **Executor**: Change evidence tracking. Instead of an ever-growing list, either:
   - Give `execute_plan` a return that includes the evidence bundles directly (alongside Result), or
   - Add a `drain_evidence()` method that returns and clears the list, called after each plan execution by the agent.
2. **SemanticStore**: Add optional `max_facts` parameter. When exceeded, evict lowest-confidence facts.
3. **VectorIndex**: Add `max_documents` with LRU eviction.
4. **Ledger**: For now, document the trade-off. True fix (memory-mapped or paged) is out of scope for this refactor but should be noted.

### Rule 9: Keep metaprogramming transparent

**Current status:** Minimal metaprogramming. Uses:
- `dataclass` decorators — standard and transparent.
- `field(default_factory=...)` — standard.
- `str, Enum` mixin for string enums — standard.
- `json.dumps(... default=str)` — transparent fallback serializer.

**No violations.** This is a strength of the codebase. No metaclasses, no dynamic class generation, no __getattr__ magic, no decorators beyond @dataclass and @staticmethod.

### Rule 10: Minimize indirection

**Current violations:**
- `EventBus` callback indirection: `event_bus.publish()` -> `process_one()` -> `_dispatch()` -> `handler()`. The handlers are registered dynamically, so tracing "what happens when a USER_INPUT event is published" requires reading `_wire_event_bus()` to find the handler, then reading the handler. 3 levels of indirection.
- `EscalationQueue._callbacks` — callbacks registered via `on_escalation()`. Same pattern: to understand what happens on escalation, you trace through the callback list.
- `Executor._tools` — Tool registry maps string names to callables. Standard plugin pattern, but another level of indirection. The `step.tool` string is resolved at runtime.
- `agent.py:553` — `self.reflection_engine._ticks_since_reflection = 0` — Reaches into a private field of another object. This is the opposite of indirection — it's *too direct* and violates encapsulation.

**What's good now:** No deep inheritance hierarchies. No abstract base classes with multiple implementation layers. The module structure is flat — each module is one class with clear responsibilities.

**What's bad now:** The EventBus callback pattern adds cognitive load. The escalation callback system adds another layer. The private field access in `reset()` bypasses the reflection engine's own API.

**Refactoring plan:**
1. Add `ReflectionEngine.reset()` method to encapsulate tick counter reset. Remove direct private field access from `Agent.reset()`.
2. Keep EventBus callbacks — they earn their keep (decoupling subsystems). But add a `describe_wiring()` or `__repr__` that dumps all subscriptions for debugging.
3. Document the event flow in a module-level docstring or diagram comment.

---

## What's Better About the Code As-Is

1. **Flat module structure.** 10 files, each with a single clear responsibility. No deep nesting, no framework complexity. A stranger can understand the architecture in 10 minutes.

2. **Linear pipeline.** `process_input` is a numbered, sequential pipeline (steps 1-14). Despite being too long, the flow is easy to follow top-to-bottom.

3. **No metaprogramming.** Pure dataclasses, plain functions, no magic. Everything does what it says.

4. **Good separation of concerns.** Governance doesn't know about execution. Execution doesn't know about planning. The ledger is a pure append-only store.

5. **Comprehensive test suite.** 175 tests at 1.38:1 code-to-test ratio. Tests cover invariants, edge cases, and integration paths.

6. **Zero dependencies.** Standard library only. No framework lock-in, no transitive dependency risks.

7. **Thread-safe where it matters.** Ledger and EventBus use locks. WorkingCache has bounded FIFO.

8. **Deterministic governance.** No heuristics, no ML-based decisions. The veto gate is a set of boolean checks with structured reason codes. Easy to audit.

## What's Worse About the Code As-Is

1. **agent.py is a god module.** 779 lines, with `__init__` (87 lines), `process_input` (115 lines), and `_load_persisted_state` (94 lines) all exceeding the 60-line rule. Too much in one place.

2. **Almost no input validation or assertions.** Only 2 assertions in the entire codebase. Bugs will propagate far from their origin before failing.

3. **Unchecked return values.** EventBus.publish backpressure signal is silently ignored. Evidence log slicing is fragile.

4. **Unbounded growth.** Evidence log, semantic store, vector index, and ledger in-memory list all grow without limit. A `run_forever` agent will eventually exhaust memory.

5. **Duplicate tool permission table.** `TOOL_PERMISSIONS` in governance.py and `_AUTONOMY_TOOLS` in planner.py are the same data. If one changes and the other doesn't, the system disagrees with itself about what's allowed.

6. **`process_all()` can loop forever.** If event handlers publish events during dispatch, the while-True loop has no termination guarantee.

7. **Direct private field access.** `Agent.reset()` writes `self.reflection_engine._ticks_since_reflection = 0` — bypasses the object's API and creates a coupling to internal implementation.

8. **No static analysis enforcement.** No mypy, no linting in CI. Type hints exist but aren't checked.

---

## What Will Be Better After Refactoring

1. **Every function fits on a screen.** The 115-line `process_input` becomes 3-4 pipeline stages of ~30 lines each. `__init__` becomes a 15-line orchestrator. `_load_persisted_state` splits into per-entity loaders.

2. **Bugs caught at origin.** Assertions at function boundaries mean a negative cost, empty plan, or None event fails immediately with a clear message, not 5 frames deep with an AttributeError.

3. **No silent data loss.** EventBus.publish backpressure is checked and logged. Evidence bundles are returned directly from execution, not sliced from a global log.

4. **Bounded memory.** Evidence list drains after each use. Semantic store and vector index have configurable caps. The process_all loop has a hard iteration limit.

5. **Single source of truth for tool permissions.** One table, imported by both governance and planner. Impossible for them to disagree.

6. **Static analysis in CI.** mypy strict + ruff catch type errors and dead code before they reach tests.

7. **Every loop provably terminates.** `process_all` has a max_iterations guard. `_redact_sensitive` has a depth limit.

## What Will Be Worse After Refactoring

1. **More files, more functions.** Splitting agent.py into smaller pieces means more files to navigate. The "read it top to bottom" simplicity of the current agent.py is lost.

2. **More assertion noise.** Every public method starts with 1-3 assertions. This is boilerplate that clutters the code for readers who trust the callers. It trades readability for defensiveness.

3. **Intermediate types.** Pipeline stages need typed return values (dataclass or tuple) for the intermediate results between stages. This adds structural overhead.

4. **Bounded data structures add complexity.** Max-size semantic store needs an eviction policy. Max-iterations in process_all needs error handling for the limit case. These are new failure modes that didn't exist before.

5. **Refactoring risk.** 175 tests must continue to pass. Structural changes to agent.py risk breaking integration test assumptions. The tests themselves may need updates if internal APIs change.

---

## Execution Order

| Phase | Scope | Risk | Rule(s) |
|-------|-------|------|---------|
| 1 | Consolidate tool permissions (single source of truth) | Low | 5 |
| 2 | Add assertions/guard clauses to all public methods | Low | 2, 4 |
| 3 | Fix unchecked return values (EventBus.publish, evidence slicing) | Medium | 4 |
| 4 | Bound all loops (process_all max iterations, _redact_sensitive depth) | Low | 6, 7 |
| 5 | Bound unbounded collections (evidence drain, semantic/vector caps) | Medium | 8 |
| 6 | Split agent.py functions (init, process_input, persistence) | High | 1, 3 |
| 7 | Extract persistence.py | Medium | 1 |
| 8 | Add ReflectionEngine.reset() method | Low | 10 |
| 9 | Clean up imports (dead import, local import) | Low | 5 |
| 10 | Add mypy/ruff config | Low | 5 |

Each phase should be a separate commit. Tests run after each phase. Phase 6 is the highest-risk change and should be done last among the structural changes, when all behavioral improvements are already in place.
