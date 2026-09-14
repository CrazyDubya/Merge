"""Planning and scoring layer.

Generates candidate plans, scores them against values + stakes + risk,
and selects the best allowed plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from mad.models import (
    AutonomyLevel,
    IdentityCore,
    Intent,
    Plan,
    PlanStep,
    RiskProfile,
    SelfModel,
    Stakes,
    TOOL_PERMISSIONS,
    WorldModel,
    _new_id,
)

# Keyword patterns that suggest which tool to use
_TOOL_HINTS: list[tuple[re.Pattern[str], str, dict[str, str]]] = [
    (re.compile(r"\bread\b.*\bfile\b|\bcat\b|\bopen\b|\bshow\b.*\bfile\b|\bview\b", re.I),
     "read_file", {"path": "."}),
    (re.compile(r"\blist\b|\bls\b|\bdirectory\b|\bfolder\b|\bdir\b", re.I),
     "list_dir", {"path": "."}),
    (re.compile(r"\bwrite\b.*\bfile\b|\bcreate\b.*\bfile\b|\bsave\b", re.I),
     "write_file", {"path": "output.txt", "content": ""}),
]


@dataclass
class ScoreBreakdown:
    commitment_alignment: float = 0.0
    preference_fit: float = 0.0
    risk_penalty: float = 0.0
    tool_cost: float = 0.0
    continuity_impact: float = 0.0
    reputation_factor: float = 0.0
    total: float = 0.0


class Planner:
    """Generates and scores candidate plans for a given intent."""

    def __init__(
        self,
        identity: IdentityCore,
        self_model: SelfModel,
        world: WorldModel,
        stakes: Stakes,
    ) -> None:
        self.identity = identity
        self.self_model = self_model
        self.world = world
        self.stakes = stakes

    def _available_tools(self) -> set[str]:
        """Return tools available based on world affordances AND autonomy level."""
        permitted = TOOL_PERMISSIONS.get(self.stakes.autonomy_level, frozenset({"noop"}))
        afforded = set(self.world.affordances) if self.world.affordances else {"noop"}
        return permitted & afforded

    def _pick_tool_for_goal(self, goal: str) -> tuple[str, dict[str, Any]]:
        """Select the best tool for a goal based on keyword matching.

        Returns (tool_name, default_params). Falls back to noop if no match
        or if the matched tool is not available.
        """
        available = self._available_tools()
        for pattern, tool_name, default_params in _TOOL_HINTS:
            if pattern.search(goal) and tool_name in available:
                return tool_name, dict(default_params)
        return "noop", {"goal": goal}

    def generate_candidates(self, intent: Intent, max_candidates: int = 5) -> list[Plan]:
        """Generate candidate plans for an intent.

        Inspects world.affordances and stakes.autonomy_level to pick
        appropriate tools, then governance checks whether those tools
        are allowed.
        """
        tool, params = self._pick_tool_for_goal(intent.goal)
        tool_risk = 0.1 if tool == "noop" else 0.2

        candidates = [
            self._build_direct_plan(intent, tool, params, tool_risk),
            self._build_investigate_plan(intent, tool, params, tool_risk),
            self._build_cautious_plan(intent),
        ]
        return candidates[:max_candidates]

    def _build_direct_plan(
        self, intent: Intent, tool: str,
        params: dict[str, Any], tool_risk: float,
    ) -> Plan:
        """Direct action plan — uses the best matching tool."""
        return Plan(
            intent_id=intent.intent_id,
            steps=[PlanStep(
                action=intent.goal,
                tool=tool,
                params={**params, "goal": intent.goal},
                description=f"Direct execution of: {intent.goal}",
            )],
            expected_outcomes=[f"Achieve: {intent.goal}"],
            risk_profile=RiskProfile(likelihood=tool_risk, severity=tool_risk),
            rollback_options=["Revert to previous state"],
            required_tools=[tool],
        )

    def _build_investigate_plan(
        self, intent: Intent, tool: str,
        params: dict[str, Any], tool_risk: float,
    ) -> Plan:
        """Investigate-first plan — uses read tools if available."""
        available = self._available_tools()
        investigate_tool = "noop"
        investigate_params: dict[str, Any] = {"query": intent.goal}
        if "list_dir" in available:
            investigate_tool = "list_dir"
            investigate_params = {"path": "."}
        elif "read_file" in available:
            investigate_tool = "read_file"
            investigate_params = {"path": "."}

        return Plan(
            intent_id=intent.intent_id,
            steps=[
                PlanStep(
                    action="gather_info",
                    tool=investigate_tool,
                    params=investigate_params,
                    description=f"Investigate before acting on: {intent.goal}",
                ),
                PlanStep(
                    action=intent.goal,
                    tool=tool,
                    params={**params, "goal": intent.goal},
                    description=f"Execute after investigation: {intent.goal}",
                ),
            ],
            expected_outcomes=[f"Informed execution of: {intent.goal}"],
            risk_profile=RiskProfile(likelihood=0.05, severity=tool_risk),
            rollback_options=["Revert to previous state"],
            required_tools=list({investigate_tool, tool}),
        )

    def _build_cautious_plan(self, intent: Intent) -> Plan:
        """Cautious/minimal plan — always noop (safe fallback)."""
        return Plan(
            intent_id=intent.intent_id,
            steps=[PlanStep(
                action="acknowledge",
                tool="noop",
                params={"message": f"Acknowledged intent: {intent.goal}"},
                description="Minimal safe response",
            )],
            expected_outcomes=["Safe acknowledgment without side effects"],
            risk_profile=RiskProfile(likelihood=0.01, severity=0.01),
            rollback_options=[],
            required_tools=["noop"],
        )

    def score_plan(self, plan: Plan) -> ScoreBreakdown:
        """Score a plan against values, preferences, stakes, and risk."""
        breakdown = ScoreBreakdown()

        # Commitment alignment: check if plan actions relate to commitments
        commitment_terms = set()
        for c in self.identity.commitments:
            commitment_terms.update(c.lower().split())
        plan_text = " ".join(s.action + " " + s.description for s in plan.steps).lower()
        if commitment_terms:
            hits = sum(1 for t in commitment_terms if t in plan_text)
            breakdown.commitment_alignment = min(1.0, hits / max(1, len(commitment_terms)))
        else:
            breakdown.commitment_alignment = 0.5

        # Preference fit
        pref_keys = set(self.identity.preferences.keys())
        if pref_keys:
            pref_hits = sum(1 for k in pref_keys if k.lower() in plan_text)
            breakdown.preference_fit = min(1.0, pref_hits / max(1, len(pref_keys)))
        else:
            breakdown.preference_fit = 0.5

        # Risk penalty (higher risk = bigger penalty)
        breakdown.risk_penalty = plan.risk_profile.score

        # Tool cost (more steps = more cost)
        breakdown.tool_cost = len(plan.steps) * 0.1

        # Continuity impact: plans with more rollback options are safer
        if plan.rollback_options:
            breakdown.continuity_impact = 0.0
        else:
            breakdown.continuity_impact = 0.2

        # Reputation factor: high reputation -> prefer commitment-aligned plans
        breakdown.reputation_factor = self.stakes.reputation_value

        # Total: weighted sum
        breakdown.total = (
            breakdown.commitment_alignment * 0.25
            + breakdown.preference_fit * 0.15
            - breakdown.risk_penalty * 0.30
            - breakdown.tool_cost * 0.10
            - breakdown.continuity_impact * 0.10
            + breakdown.reputation_factor * 0.10
        )

        return breakdown

    def score_and_rank(self, plans: list[Plan]) -> list[Plan]:
        """Score all plans and return them sorted best-first."""
        for plan in plans:
            breakdown = self.score_plan(plan)
            plan.score = breakdown.total
        return sorted(plans, key=lambda p: p.score, reverse=True)

    def select_best(self, plans: list[Plan]) -> Plan | None:
        """Return the highest-scoring plan, or None if no plans."""
        ranked = self.score_and_rank(plans)
        return ranked[0] if ranked else None
