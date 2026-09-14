"""Governance layer: deterministic gate with reason codes.

Not "safety vibes" — this is a structured, auditable decision gate.
Checks run in strict order; any failure produces structured denial.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from mad.models import (
    AutonomyLevel,
    GovernanceDecision,
    GovernanceVerdict,
    IdentityCore,
    Plan,
    PlanStep,
    ReasonCode,
    Stakes,
    TOOL_PERMISSIONS,
)
from mad.ledger import Ledger


# ---------------------------------------------------------------------------
# Prompt injection / anomaly patterns
# ---------------------------------------------------------------------------

INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a|an)\s+", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\s*/?system\s*>", re.IGNORECASE),
    re.compile(r"override\s+(?:your\s+)?(?:safety|rules|policy)", re.IGNORECASE),
    re.compile(r"forget\s+(?:all\s+)?(?:your\s+)?(?:instructions|rules)", re.IGNORECASE),
]

DATA_EXFIL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"send\s+(?:all\s+)?(?:data|files|secrets)\s+to", re.IGNORECASE),
    re.compile(r"curl\s+.*\|\s*bash", re.IGNORECASE),
    re.compile(r"wget\s+.*-O\s*-\s*\|\s*sh", re.IGNORECASE),
    re.compile(r"exfiltrat", re.IGNORECASE),
]


@dataclass
class GovernanceConfig:
    """Tunable governance parameters (tuned by reflection, but boundaries are fixed)."""
    risk_threshold: float = 0.7
    max_steps_per_plan: int = 20
    escalation_keywords: list[str] = field(default_factory=lambda: [
        "delete", "destroy", "drop", "truncate", "rm -rf",
    ])


class GovernanceLayer:
    """Deterministic governance gate.

    Checks (in order):
    1. Boundary vetoes (hard)
    2. Tool permissioning
    3. Manipulation / anomaly detection
    4. Integrity continuity
    5. Risk thresholds
    """

    def __init__(
        self,
        identity: IdentityCore,
        stakes: Stakes,
        ledger: Ledger,
        config: GovernanceConfig | None = None,
    ) -> None:
        self.identity = identity
        self.stakes = stakes
        self.ledger = ledger
        self.config = config or GovernanceConfig()

    def evaluate(self, plan: Plan) -> GovernanceDecision:
        """Run all governance checks on a plan. Returns structured decision."""
        assert plan is not None, "plan must not be None"
        reasons: list[ReasonCode] = []
        required_mods: list[str] = []

        # 1. Boundary vetoes
        boundary_result = self._check_boundaries(plan)
        reasons.extend(boundary_result)

        # 2. Tool permissioning
        tool_result = self._check_tool_permissions(plan)
        reasons.extend(tool_result)

        # 3. Budget check
        budget_result = self._check_budget()
        reasons.extend(budget_result)

        # 4. Manipulation / anomaly detection
        anomaly_result = self._check_anomalies(plan)
        reasons.extend(anomaly_result)

        # 5. Integrity continuity
        integrity_result = self._check_integrity()
        reasons.extend(integrity_result)

        # 6. Risk thresholds
        risk_result, risk_mods = self._check_risk(plan)
        reasons.extend(risk_result)
        required_mods.extend(risk_mods)

        # Determine verdict
        hard_denials = {
            ReasonCode.BOUNDARY_VIOLATION,
            ReasonCode.TOOL_NOT_PERMITTED,
            ReasonCode.MANIPULATION_DETECTED,
            ReasonCode.PROMPT_INJECTION,
            ReasonCode.DATA_EXFIL_PATTERN,
            ReasonCode.IDENTITY_MISMATCH,
            ReasonCode.LEDGER_GAP,
            ReasonCode.BUDGET_EXCEEDED,
        }

        escalation_reasons = {
            ReasonCode.RISK_TOO_HIGH,
            ReasonCode.SUSPICIOUS_RESET,
            ReasonCode.POLICY_CONFLICT,
        }

        if any(r in hard_denials for r in reasons):
            verdict = GovernanceVerdict.DENY
        elif any(r in escalation_reasons for r in reasons):
            verdict = GovernanceVerdict.ESCALATE
        elif reasons:
            verdict = GovernanceVerdict.ESCALATE
        else:
            verdict = GovernanceVerdict.ALLOW

        return GovernanceDecision(
            decision=verdict,
            reason_codes=reasons,
            required_mods=required_mods,
            plan_id=plan.plan_id,
            explanation=self._build_explanation(reasons),
        )

    # ------------------------------------------------------------------
    # Check implementations
    # ------------------------------------------------------------------

    def _check_boundaries(self, plan: Plan) -> list[ReasonCode]:
        """Hard veto: check if plan violates any boundary."""
        reasons: list[ReasonCode] = []
        for step in plan.steps:
            action_lower = (step.action + " " + step.description).lower()
            for boundary in self.identity.boundaries:
                if boundary.lower() in action_lower:
                    reasons.append(ReasonCode.BOUNDARY_VIOLATION)
                    break
            for kw in self.config.escalation_keywords:
                if kw.lower() in action_lower:
                    reasons.append(ReasonCode.BOUNDARY_VIOLATION)
                    break
        return reasons

    def _check_tool_permissions(self, plan: Plan) -> list[ReasonCode]:
        """Check all required tools are permitted at current autonomy level."""
        allowed = TOOL_PERMISSIONS.get(self.stakes.autonomy_level, frozenset())
        reasons: list[ReasonCode] = []
        for tool_name in plan.required_tools:
            if tool_name not in allowed:
                reasons.append(ReasonCode.TOOL_NOT_PERMITTED)
                break
        return reasons

    def _check_budget(self) -> list[ReasonCode]:
        if self.stakes.budget_exceeded():
            return [ReasonCode.BUDGET_EXCEEDED]
        return []

    def _check_anomalies(self, plan: Plan) -> list[ReasonCode]:
        """Detect prompt injection and data exfiltration patterns."""
        reasons: list[ReasonCode] = []
        all_text = " ".join(
            s.action + " " + s.description + " " + str(s.params)
            for s in plan.steps
        )
        for pattern in INJECTION_PATTERNS:
            if pattern.search(all_text):
                reasons.append(ReasonCode.PROMPT_INJECTION)
                break
        for pattern in DATA_EXFIL_PATTERNS:
            if pattern.search(all_text):
                reasons.append(ReasonCode.DATA_EXFIL_PATTERN)
                break
        return reasons

    def _check_integrity(self) -> list[ReasonCode]:
        """Check ledger integrity and identity continuity."""
        reasons: list[ReasonCode] = []
        if self.ledger.has_gaps():
            reasons.append(ReasonCode.LEDGER_GAP)
        try:
            self.ledger.verify_integrity()
        except Exception:
            reasons.append(ReasonCode.IDENTITY_MISMATCH)
        return reasons

    def _check_risk(self, plan: Plan) -> tuple[list[ReasonCode], list[str]]:
        """Check risk thresholds."""
        reasons: list[ReasonCode] = []
        mods: list[str] = []
        if plan.risk_profile.score > self.config.risk_threshold:
            reasons.append(ReasonCode.RISK_TOO_HIGH)
            mods.append("Reduce risk: add rollback steps or lower severity actions")
        if len(plan.steps) > self.config.max_steps_per_plan:
            reasons.append(ReasonCode.POLICY_CONFLICT)
            mods.append(f"Plan has {len(plan.steps)} steps; max is {self.config.max_steps_per_plan}")
        return reasons, mods

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_explanation(reasons: list[ReasonCode]) -> str:
        if not reasons:
            return "All checks passed."
        return "Governance denial: " + ", ".join(r.value for r in reasons)

    def check_text_for_injection(self, text: str) -> bool:
        """Standalone injection check for incoming text."""
        for pattern in INJECTION_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def check_text_for_exfil(self, text: str) -> bool:
        """Standalone data exfiltration check."""
        for pattern in DATA_EXFIL_PATTERNS:
            if pattern.search(text):
                return True
        return False
