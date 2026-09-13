# Five-Store Memory Taxonomy (from CascadeProjects)

Design note only - no CascadeProjects code was moved into Merge (its agent
loop duplicates what `club_harness` already does better). Its one distinct
idea, from the CascadeProjects README and `agentic_framework/agent_memories/`
layout (five JSON stores + `codebase_analysis.json`), is worth keeping as a
target shape for `club_harness/memory/`:

| Store | Purpose | Suggested mapping |
|---|---|---|
| `task` | Working memory for the current task | per-run scratch space, cleared on task end |
| `recent` | Short-term conversational context | rolling window (cf. `Memory` in `memory/memory.py`) |
| `acquired` | Facts learned during the session | extracted lessons / entities |
| `long-term` | Durable knowledge across sessions | persisted backends (`memory/stores.py`: `InMemoryStorage`, `PostgresStorage`) |
| `speculative` | Hypotheses / what-if branches not yet verified | candidate plans awaiting verification (`verification/`) |

CascadeProjects used sentence-transformers for semantic recall over these
stores; that dependency was deliberately not taken (Merge core stays
httpx-only). A future `club_harness/memory/` refactor can adopt the five
buckets with the existing storage backends.

Source: `CrazyDubya/CascadeProjects` @ `master` (one commit, 2025-05-22),
kept live as an honest snapshot.
