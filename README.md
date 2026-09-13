# Merge: Agent Framework Consolidation

A unified agent orchestration framework combining best practices from multiple agent repositories.

## Consolidated Repositories

| Repository | Source | Purpose |
|------------|--------|---------|
| TinyTroupe | Microsoft | Persona simulation, memory systems |
| llm-council | CrazyDubya | Multi-LLM consensus mechanisms |
| LisaSimpson | CrazyDubya | Deliberative planning with verification |
| Agentic-Hub | CrazyDubya | Agent orchestration, sandboxing |
| 12-factor-agents | HumanLayer | Production agent principles |
| hivey | CrazyDubya | Swarm intelligence, cost-aware routing |
| qwen-code | Qwen | Coding agent architecture baseline |
| Village | CrazyDubya | Provider abstraction, storage backends, rate limiting, collaborate loop (merged 2026-09-12 via `git subtree`; full history in `repos/Village/`) |
| role-based-llm-framework | CrazyDubya | PM/coder/researcher role algorithms + dashboard (merged 2026-09-12 via `git subtree`; full history in `repos/role-based-llm-framework/`) |

## What's new (2026-09-12 consolidation)

- **`club_harness/llm/providers/`** - OpenAI/Anthropic/Google provider backends from Village (optional extra: `requirements-village.txt`; core stays httpx-only)
- **`club_harness/memory/stores.py`** - storage backends from Village (in-memory + Postgres)
- **`club_harness/core/rate_limit.py`** - token-bucket rate limiter + quota manager from Village; `Agent(..., rate_limiter=...)` enforces it in `chat()`
- **`club_harness/orchestration/village.py`** - `VillageTeam.collaborate()` multi-agent loop adapted to `Agent`
- **`club_harness/orchestration/roles/`** - PM/coder/researcher roles from ChipCliff, running on the club_harness LLM router (torch classifier optional, heuristic fallback built in)
- **`apps/dashboard/`** - the ChipCliff FastAPI dashboard, kept as an optional app
- **`docs/PROVENANCE.md`** - file-by-file source mapping; **`docs/memory-taxonomy.md`** - five-store memory design note from CascadeProjects

## Club Harness

The `club_harness/` directory contains the unified agent framework:

```bash
# Run tests
python test_harness.py

# Quick usage
from club_harness.core.agent import Agent, AgentBuilder

agent = (
    AgentBuilder("MyAgent")
    .with_instructions("You are a helpful assistant.")
    .with_tier("free")  # Use free OpenRouter models
    .build()
)

response = agent.chat("Hello!")
```

## Features

- **Multi-Provider LLM Support**: OpenRouter (400+ models), Ollama, direct APIs
- **Cost-Aware Routing**: Intelligent model tier selection (free, cheap, standard, reasoning, advanced)
- **Persona System**: TinyTroupe-inspired agent personalities
- **Confidence Tracking**: LisaSimpson-inspired provenance tracking
- **12-Factor Compliance**: Production-ready patterns
- **Multi-Turn Conversations**: Context-aware chat sessions

## Setup

```bash
pip install httpx
export OPENROUTER_API_KEY="your-key-here"
python test_harness.py
```

## Architecture

See [CONSOLIDATION_PLAN.md](./CONSOLIDATION_PLAN.md) for the detailed multi-dimensional plan combining all repositories.
