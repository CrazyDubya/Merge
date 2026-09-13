# Provenance: Agent-Framework Consolidation

Date: 2026-09-12. Merge strategy: `git subtree add` (history preserved) for
the two absorbed repos, per Stephen's decision. Clean-copy refresh for the
vendored TinyTroupe.

## Subtree merges (full history in-repo)

| Prefix | Source repo | Source commit | Branch |
|---|---|---|---|
| `repos/Village/` | `CrazyDubya/Village` | `55bf5da05fa5efca07b5ea6bce8581e40e4c362f` | `main` |
| `repos/role-based-llm-framework/` | `CrazyDubya/role-based-llm-framework` | `6faf2e5b6c0bcb6ea9ec712a90998bbe2eab877e` | `master` |

## Ported into club_harness (selective, adapted)

### From Village (`repos/Village/`, commit `55bf5da`)

| New file | Source file(s) | Notes |
|---|---|---|
| `club_harness/llm/providers/base.py` | `village/llm/base.py` | verbatim; `village.exceptions` -> `club_harness.core.errors` |
| `club_harness/llm/providers/openai.py` | `village/llm/openai.py` | lazy `openai` SDK import kept; optional extra |
| `club_harness/llm/providers/anthropic.py` | `village/llm/anthropic.py` | lazy `anthropic` SDK import kept; optional extra |
| `club_harness/llm/providers/google.py` | `village/llm/google.py` | lazy `google-generativeai` import kept; optional extra |
| `club_harness/llm/providers/__init__.py` | new | re-exports; imports never require SDKs |
| `club_harness/memory/stores.py` | `village/storage/base.py` + `memory.py` + `postgres.py` | combined; asyncpg lazy in postgres backend |
| `club_harness/core/rate_limit.py` | `village/utils/rate_limiter.py` | + `acquire_nowait()` sync helper; **fixed a deadlock**: original `use_quota()` held the asyncio lock while calling `check_quota()` (which re-acquires it). Split `_check_quota_locked()`. |
| `club_harness/orchestration/village.py` | `village/core/village.py` (`collaborate()`) | adapted: orchestrates `club_harness.core.Agent` via `chat()`; added `collaborate_async()` |
| `requirements-village.txt` | `village` pyproject deps | optional extra; core stays httpx-only |

Village's `examples/`, `cli.py`, `utils/config.py`, `utils/metrics.py`,
`utils/logging.py` were not ported (Merge has equivalents or they are
app-specific). Village's test suite was partially ported to
`test_village_providers.py` (import paths rewritten).

### From role-based-llm-framework ("ChipCliff", `repos/role-based-llm-framework/`, commit `6faf2e5`)

| New file | Source file(s) | Notes |
|---|---|---|
| `club_harness/orchestration/roles/classifier.py` | `pm_algorithm.py` (classifier part) | torch/transformers path lazy; keyword-heuristic fallback; no import-time model download (original downloaded DistilBERT at import) |
| `club_harness/orchestration/roles/pm.py` | `pm_algorithm.py` (dispatch part) | `ProjectManager`: classify -> assign -> track; in-memory task store (XML persistence stays in dashboard app) |
| `club_harness/orchestration/roles/coder.py` | `coder_algorithm.py` | generates via `LLMRouter` instead of static HTML stub; `test_code()` compile-checks instead of opening a browser |
| `club_harness/orchestration/roles/researcher.py` | `researcher_algorithm.py` | query enhancement via `LLMRouter` (falls back to base queries); requests/bs4 lazy |
| `club_harness/orchestration/roles/__init__.py` | new | re-exports |
| `apps/dashboard/` | `main.py`, `main2.py`, `ui.py`, `utils.py`, `xml_utils.py`, `llm_integration.py`, `templates/`, `static/`, `config/` | kept verbatim as optional app (Stephen: keep dashboard); `requirements-dashboard.txt` from source `requirements-minimal.txt` |

ChipCliff's torch/transformers/fastapi/uvicorn stack was deliberately NOT
taken into core. `xml_utils.py` (XML task persistence) stays dashboard-only.

## Vendored reference copies

| Prefix | State |
|---|---|
| `repos/TinyTroupe/` | refreshed from `CrazyDubya/TinyTropute` fork `main` (was 9 commits behind; picked up Ollama provider + offline-test stabilization) |
| `repos/Village/` | new (subtree, see above) |
| `repos/role-based-llm-framework/` | new (subtree, see above) |

## Idea-only contributions (no code moved)

- **CascadeProjects** five-memory-store taxonomy (task / recent / acquired /
  long-term / speculative) documented in `docs/memory-taxonomy.md` and
  referenced from `club_harness/memory/stores.py`.
- **cards** (private): excluded entirely - no code moved without declassification.
- **DoghouseLLM**: excluded - application, not framework; left live as-is.

## Dependency discipline

- Merge core remains **httpx-only** (`requirements.txt` unchanged).
- Village provider SDKs -> `requirements-village.txt` (optional extra).
- Dashboard -> `apps/dashboard/requirements-dashboard.txt` (optional app).
- No torch/transformers/fastapi anywhere in `club_harness/`.
