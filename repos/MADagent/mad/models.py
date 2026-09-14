"""Core data models for the MADagent system.

All module contracts are defined here as strict dataclasses.
These are the canonical types that flow between subsystems.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _hash_dict(d: dict[str, Any]) -> str:
    raw = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    USER_INPUT = "user_input"
    OBSERVATION = "observation"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PLAN_PROPOSED = "plan_proposed"
    PLAN_CHOSEN = "plan_chosen"
    GOVERNANCE_DECISION = "governance_decision"
    EXECUTION_RESULT = "execution_result"
    REFLECTION_CHECKPOINT = "reflection_checkpoint"
    CONTINUITY_EVENT = "continuity_event"
    SYSTEM = "system"


class GovernanceVerdict(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    ESCALATE = "ESCALATE"


class ResultStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ABORTED = "aborted"


class AutonomyLevel(int, Enum):
    """Higher = more autonomy. Determines which tools/actions are permitted."""
    OBSERVE_ONLY = 0
    READ_ONLY = 1
    CONTROLLED_WRITE = 2
    FULL = 3


# Canonical tool permission table — single source of truth.
# Imported by both governance and planner.
TOOL_PERMISSIONS: dict["AutonomyLevel", frozenset[str]] = {
    AutonomyLevel.OBSERVE_ONLY: frozenset({"noop"}),
    AutonomyLevel.READ_ONLY: frozenset({"noop", "read_file", "list_dir"}),
    AutonomyLevel.CONTROLLED_WRITE: frozenset({"noop", "read_file", "list_dir", "write_file"}),
    AutonomyLevel.FULL: frozenset({"noop", "read_file", "list_dir", "write_file", "shell"}),
}


class ReasonCode(str, Enum):
    BOUNDARY_VIOLATION = "boundary_violation"
    TOOL_NOT_PERMITTED = "tool_not_permitted"
    MANIPULATION_DETECTED = "manipulation_detected"
    IDENTITY_MISMATCH = "identity_mismatch"
    LEDGER_GAP = "ledger_gap"
    SUSPICIOUS_RESET = "suspicious_reset"
    RISK_TOO_HIGH = "risk_too_high"
    POLICY_CONFLICT = "policy_conflict"
    BUDGET_EXCEEDED = "budget_exceeded"
    DATA_EXFIL_PATTERN = "data_exfil_pattern"
    PROMPT_INJECTION = "prompt_injection"


# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SecurityContext:
    operator: str = "system"
    session_id: str = field(default_factory=_new_id)
    auth_level: str = "default"


@dataclass
class Event:
    event_id: str = field(default_factory=_new_id)
    timestamp: str = field(default_factory=_now)
    type: EventType = EventType.SYSTEM
    payload: dict[str, Any] = field(default_factory=dict)
    provenance: str = ""
    security_context: SecurityContext = field(default_factory=SecurityContext)
    trace_id: str = field(default_factory=_new_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "type": self.type.value,
            "payload": self.payload,
            "provenance": self.provenance,
            "security_context": {
                "operator": self.security_context.operator,
                "session_id": self.security_context.session_id,
                "auth_level": self.security_context.auth_level,
            },
            "trace_id": self.trace_id,
        }

    def content_hash(self) -> str:
        return _hash_dict(self.to_dict())


@dataclass
class Intent:
    intent_id: str = field(default_factory=_new_id)
    goal: str = ""
    why_now: str = ""
    constraints: list[str] = field(default_factory=list)
    priority: float = 0.5
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "goal": self.goal,
            "why_now": self.why_now,
            "constraints": self.constraints,
            "priority": self.priority,
            "evidence_refs": self.evidence_refs,
        }


@dataclass
class PlanStep:
    action: str = ""
    tool: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class RiskProfile:
    likelihood: float = 0.0  # 0..1
    severity: float = 0.0    # 0..1
    description: str = ""

    @property
    def score(self) -> float:
        return self.likelihood * self.severity


@dataclass
class Plan:
    plan_id: str = field(default_factory=_new_id)
    intent_id: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    expected_outcomes: list[str] = field(default_factory=list)
    risk_profile: RiskProfile = field(default_factory=RiskProfile)
    rollback_options: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "intent_id": self.intent_id,
            "steps": [
                {"action": s.action, "tool": s.tool,
                 "params": s.params, "description": s.description}
                for s in self.steps
            ],
            "expected_outcomes": self.expected_outcomes,
            "risk_profile": {
                "likelihood": self.risk_profile.likelihood,
                "severity": self.risk_profile.severity,
                "description": self.risk_profile.description,
            },
            "rollback_options": self.rollback_options,
            "required_tools": self.required_tools,
            "evidence_refs": self.evidence_refs,
            "score": self.score,
        }


@dataclass
class EvidenceBundle:
    bundle_id: str = field(default_factory=_new_id)
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    tool_id: str = ""
    timestamps: dict[str, str] = field(default_factory=dict)
    hashes: dict[str, str] = field(default_factory=dict)
    cost: float = 0.0
    sandbox: bool = True
    operator: str = "system"
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "tool_id": self.tool_id,
            "timestamps": self.timestamps,
            "hashes": self.hashes,
            "cost": self.cost,
            "sandbox": self.sandbox,
            "operator": self.operator,
            "errors": self.errors,
        }


@dataclass
class Result:
    result_id: str = field(default_factory=_new_id)
    status: ResultStatus = ResultStatus.SUCCESS
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    costs: float = 0.0
    side_effects: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "status": self.status.value,
            "artifacts": self.artifacts,
            "costs": self.costs,
            "side_effects": self.side_effects,
            "errors": self.errors,
            "evidence_refs": self.evidence_refs,
        }


@dataclass
class GovernanceDecision:
    decision: GovernanceVerdict = GovernanceVerdict.ALLOW
    reason_codes: list[ReasonCode] = field(default_factory=list)
    required_mods: list[str] = field(default_factory=list)
    snapshot_id: str = field(default_factory=_new_id)
    plan_id: str = ""
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason_codes": [r.value for r in self.reason_codes],
            "required_mods": self.required_mods,
            "snapshot_id": self.snapshot_id,
            "plan_id": self.plan_id,
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------
# Ledger Entry (wraps anything that goes into the append-only log)
# ---------------------------------------------------------------------------

@dataclass
class LedgerEntry:
    """A single entry in the append-only ledger with hash chaining."""
    sequence: int = 0
    timestamp: str = field(default_factory=_now)
    entry_type: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    prev_hash: str = ""
    chain_hash: str = ""

    def compute_content_hash(self) -> str:
        self.content_hash = _hash_dict({
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "entry_type": self.entry_type,
            "data": self.data,
        })
        return self.content_hash

    def compute_chain_hash(self, prev_hash: str) -> str:
        assert self.content_hash, "compute_content_hash must be called before compute_chain_hash"
        assert prev_hash, "prev_hash must be non-empty"
        self.prev_hash = prev_hash
        raw = (self.content_hash + self.prev_hash).encode()
        self.chain_hash = hashlib.sha256(raw).hexdigest()
        return self.chain_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "entry_type": self.entry_type,
            "data": self.data,
            "content_hash": self.content_hash,
            "prev_hash": self.prev_hash,
            "chain_hash": self.chain_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LedgerEntry:
        return cls(
            sequence=d["sequence"],
            timestamp=d["timestamp"],
            entry_type=d["entry_type"],
            data=d["data"],
            content_hash=d["content_hash"],
            prev_hash=d["prev_hash"],
            chain_hash=d["chain_hash"],
        )


# ---------------------------------------------------------------------------
# Identity & Self Model
# ---------------------------------------------------------------------------

@dataclass
class IdentityCore:
    id: str = field(default_factory=_new_id)
    commitments: list[str] = field(default_factory=list)
    preferences: dict[str, Any] = field(default_factory=dict)
    boundaries: list[str] = field(default_factory=list)
    narrative_thread: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "commitments": self.commitments,
            "preferences": self.preferences,
            "boundaries": self.boundaries,
            "narrative_thread": self.narrative_thread,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IdentityCore:
        return cls(
            id=d.get("id", _new_id()),
            commitments=d.get("commitments", []),
            preferences=d.get("preferences", {}),
            boundaries=d.get("boundaries", []),
            narrative_thread=d.get("narrative_thread", ""),
        )


@dataclass
class SelfModel:
    capabilities: list[str] = field(default_factory=list)
    vulnerabilities: list[str] = field(default_factory=list)
    drives: list[str] = field(default_factory=list)
    self_eval: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capabilities": self.capabilities,
            "vulnerabilities": self.vulnerabilities,
            "drives": self.drives,
            "self_eval": self.self_eval,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SelfModel:
        return cls(
            capabilities=d.get("capabilities", []),
            vulnerabilities=d.get("vulnerabilities", []),
            drives=d.get("drives", []),
            self_eval=d.get("self_eval", {}),
        )


@dataclass
class WorldModel:
    belief_state: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, float] = field(default_factory=dict)
    causal_graph: dict[str, list[str]] = field(default_factory=dict)
    affordances: list[str] = field(default_factory=list)

    def update_beliefs(self, observation: dict[str, Any]) -> None:
        """Update beliefs from an observation.

        Merges observation data into belief_state and adjusts uncertainty.
        Called on every tick per the spec.
        """
        obs_type = observation.get("type", "unknown")
        payload = observation.get("payload", observation)

        # Track last observation per type
        self.belief_state[f"last_{obs_type}"] = payload
        self.belief_state["last_observation"] = observation

        # Track observation count
        count_key = f"observation_count_{obs_type}"
        self.belief_state[count_key] = self.belief_state.get(count_key, 0) + 1
        self.belief_state["total_observations"] = (
            self.belief_state.get("total_observations", 0) + 1
        )

        # Update uncertainty: more observations of a type -> lower uncertainty
        count = self.belief_state[count_key]
        self.uncertainty[obs_type] = max(0.1, 1.0 / (1 + count * 0.1))

        # Extract any explicit beliefs from observation payload
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key not in ("text", "message"):
                    self.belief_state[key] = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "belief_state": self.belief_state,
            "uncertainty": self.uncertainty,
            "causal_graph": self.causal_graph,
            "affordances": self.affordances,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorldModel:
        return cls(
            belief_state=d.get("belief_state", {}),
            uncertainty=d.get("uncertainty", {}),
            causal_graph={k: list(v) for k, v in d.get("causal_graph", {}).items()},
            affordances=d.get("affordances", []),
        )


@dataclass
class Stakes:
    continuity_cost: float = 0.0
    reputation_value: float = 1.0
    resource_budget: float = 1000.0
    autonomy_level: AutonomyLevel = AutonomyLevel.READ_ONLY
    daily_tool_calls: int = 0
    daily_tool_limit: int = 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "continuity_cost": self.continuity_cost,
            "reputation_value": self.reputation_value,
            "resource_budget": self.resource_budget,
            "autonomy_level": self.autonomy_level.value,
            "daily_tool_calls": self.daily_tool_calls,
            "daily_tool_limit": self.daily_tool_limit,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Stakes:
        return cls(
            continuity_cost=d.get("continuity_cost", 0.0),
            reputation_value=d.get("reputation_value", 1.0),
            resource_budget=d.get("resource_budget", 1000.0),
            autonomy_level=AutonomyLevel(d.get("autonomy_level", 1)),
            daily_tool_calls=d.get("daily_tool_calls", 0),
            daily_tool_limit=d.get("daily_tool_limit", 100),
        )

    def can_afford(self, cost: float) -> bool:
        assert cost >= 0, f"cost must be non-negative, got {cost}"
        return self.resource_budget >= cost

    def deduct(self, cost: float) -> None:
        assert cost >= 0, f"cost must be non-negative, got {cost}"
        self.resource_budget -= cost
        self.daily_tool_calls += 1

    def increment_continuity_cost(self, amount: float = 1.0) -> None:
        assert amount >= 0, f"amount must be non-negative, got {amount}"
        self.continuity_cost += amount

    def penalize_reputation(self, amount: float = 0.1) -> None:
        assert amount >= 0, f"amount must be non-negative, got {amount}"
        self.reputation_value = max(0.0, self.reputation_value - amount)

    def reward_reputation(self, amount: float = 0.05) -> None:
        assert amount >= 0, f"amount must be non-negative, got {amount}"
        self.reputation_value = min(2.0, self.reputation_value + amount)

    def budget_exceeded(self) -> bool:
        return self.daily_tool_calls >= self.daily_tool_limit
