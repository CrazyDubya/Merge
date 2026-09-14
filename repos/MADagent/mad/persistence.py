"""State persistence: save and load agent state to/from disk.

Each entity type (identity, stakes, world, self_model, semantic_store)
has its own save/load function for clarity and testability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mad.memory import MemorySystem, SemanticFact
from mad.models import (
    IdentityCore,
    SelfModel,
    Stakes,
    WorldModel,
)


def save_state(
    data_dir: Path,
    identity: IdentityCore,
    stakes: Stakes,
    world: WorldModel,
    self_model: SelfModel,
    memory: MemorySystem,
) -> None:
    """Persist all agent state to disk."""
    data_dir.mkdir(parents=True, exist_ok=True)
    _save_json(data_dir / "identity.json", identity.to_dict())
    _save_json(data_dir / "stakes.json", stakes.to_dict())
    _save_json(data_dir / "world.json", world.to_dict())
    _save_json(data_dir / "self_model.json", self_model.to_dict())
    _save_semantic_store(data_dir / "semantic_store.json", memory)


def load_state(
    data_dir: Path,
    identity: IdentityCore,
    stakes: Stakes,
    world: WorldModel,
    self_model: SelfModel,
    memory: MemorySystem,
    ledger_length: int,
) -> None:
    """Load all agent state from disk, if files exist."""
    load_identity(data_dir, identity)
    load_stakes(data_dir, stakes)
    load_world(data_dir, world)
    load_self_model(data_dir, self_model)

    if ledger_length > 0:
        memory.rebuild_from_ledger()

    # Semantic store loaded AFTER rebuild so persisted facts survive
    load_semantic_store(data_dir, memory)


# ------------------------------------------------------------------
# Save helpers
# ------------------------------------------------------------------

def _save_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _save_semantic_store(path: Path, memory: MemorySystem) -> None:
    facts = memory.semantic.all_facts()
    facts_data = [
        {
            "fact_id": f.fact_id,
            "content": f.content,
            "source_refs": f.source_refs,
            "tags": f.tags,
            "confidence": f.confidence,
            "created_at": f.created_at,
        }
        for f in facts
    ]
    _save_json(path, facts_data)


# ------------------------------------------------------------------
# Load helpers
# ------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_identity(data_dir: Path, identity: IdentityCore) -> None:
    d = _load_json(data_dir / "identity.json")
    if d is None:
        return
    identity.id = d.get("id", identity.id)
    identity.commitments = d.get("commitments", identity.commitments)
    identity.preferences = d.get("preferences", identity.preferences)
    identity.boundaries = d.get("boundaries", identity.boundaries)
    identity.narrative_thread = d.get("narrative_thread", identity.narrative_thread)


def load_stakes(data_dir: Path, stakes: Stakes) -> None:
    d = _load_json(data_dir / "stakes.json")
    if d is None:
        return
    loaded = Stakes.from_dict(d)
    stakes.continuity_cost = loaded.continuity_cost
    stakes.reputation_value = loaded.reputation_value
    stakes.resource_budget = loaded.resource_budget
    stakes.autonomy_level = loaded.autonomy_level
    stakes.daily_tool_calls = loaded.daily_tool_calls
    stakes.daily_tool_limit = loaded.daily_tool_limit


def load_world(data_dir: Path, world: WorldModel) -> None:
    d = _load_json(data_dir / "world.json")
    if d is None:
        return
    world.belief_state = d.get("belief_state", world.belief_state)
    world.uncertainty = d.get("uncertainty", world.uncertainty)
    world.causal_graph = d.get("causal_graph", world.causal_graph)


def load_self_model(data_dir: Path, self_model: SelfModel) -> None:
    d = _load_json(data_dir / "self_model.json")
    if d is None:
        return
    self_model.capabilities = d.get("capabilities", self_model.capabilities)
    self_model.vulnerabilities = d.get("vulnerabilities", self_model.vulnerabilities)
    self_model.drives = d.get("drives", self_model.drives)
    self_model.self_eval = d.get("self_eval", self_model.self_eval)


def load_semantic_store(data_dir: Path, memory: MemorySystem) -> None:
    facts_data = _load_json(data_dir / "semantic_store.json")
    if facts_data is None:
        return
    for fd in facts_data:
        memory.semantic.add(SemanticFact(
            fact_id=fd["fact_id"],
            content=fd["content"],
            source_refs=fd.get("source_refs", []),
            tags=fd.get("tags", []),
            confidence=fd.get("confidence", 1.0),
            created_at=fd.get("created_at", ""),
        ))
