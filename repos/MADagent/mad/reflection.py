"""Reflection and consolidation system.

Bounded self-change:
- Allowed to update: narrative_thread, semantic summaries, calibrated capabilities
- NOT allowed to update without human sign-off: commitments, boundaries, governance veto rules

Reflection produces "chapter summaries" and writes ReflectionCheckpoints to the ledger.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from mad.ledger import Ledger
from mad.memory import MemorySystem, SemanticFact
from mad.models import (
    IdentityCore,
    LedgerEntry,
    SelfModel,
    Stakes,
    _new_id,
    _now,
)


@dataclass
class ReflectionResult:
    """Output of a reflection cycle."""
    chapter_summary: str = ""
    narrative_update: str = ""
    capability_updates: list[str] = field(default_factory=list)
    governance_recommendations: list[str] = field(default_factory=list)
    facts_extracted: list[SemanticFact] = field(default_factory=list)
    commitments_preserved: bool = True


@dataclass
class GovernanceTuningProposal:
    """A recommendation (not automatic change) for governance tuning."""
    proposal_id: str = field(default_factory=_new_id)
    description: str = ""
    rationale: str = ""
    requires_human_approval: bool = True


class ReflectionEngine:
    """Periodic reflection that consolidates experience into identity-relevant updates."""

    def __init__(
        self,
        identity: IdentityCore,
        self_model: SelfModel,
        stakes: Stakes,
        memory: MemorySystem,
        ledger: Ledger,
        reflection_interval: int = 10,
    ) -> None:
        self.identity = identity
        self.self_model = self_model
        self.stakes = stakes
        self.memory = memory
        self.ledger = ledger
        self.reflection_interval = reflection_interval
        self._ticks_since_reflection = 0
        self._reflection_count = 0

    def should_reflect(self) -> bool:
        self._ticks_since_reflection += 1
        return self._ticks_since_reflection >= self.reflection_interval

    def tick(self) -> None:
        self._ticks_since_reflection += 1

    def reset_tick_counter(self) -> None:
        """Reset the reflection tick counter. Used by Agent.reset()."""
        self._ticks_since_reflection = 0

    def reflect(self) -> ReflectionResult:
        """Perform a reflection cycle.

        1. Summarize recent experience
        2. Update narrative_thread (allowed)
        3. Update calibrated capabilities (allowed)
        4. Propose governance tuning (recommendation only)
        5. Write ReflectionCheckpoint to ledger
        6. Preserve commitments and boundaries (invariant)
        """
        result = ReflectionResult()

        # Snapshot commitments and boundaries BEFORE reflection
        commitments_before = list(self.identity.commitments)
        boundaries_before = list(self.identity.boundaries)

        # 1. Summarize recent experience
        recent_entries = self.ledger.recent(self.reflection_interval)
        result.chapter_summary = self._build_chapter_summary(recent_entries)

        # 2. Update narrative thread (allowed)
        result.narrative_update = self._update_narrative(result.chapter_summary)
        self.identity.narrative_thread = result.narrative_update

        # 3. Extract facts for semantic store
        result.facts_extracted = self._extract_facts(recent_entries)
        for fact in result.facts_extracted:
            self.memory.semantic.add(fact)

        # 4. Update self-model capabilities (allowed)
        result.capability_updates = self._update_capabilities(recent_entries)
        self.self_model.capabilities = list(set(
            self.self_model.capabilities + result.capability_updates
        ))

        # 5. Propose governance tuning (recommendation only, not applied)
        result.governance_recommendations = self._propose_governance_tuning(recent_entries)

        # 6. CRITICAL: Verify commitments/boundaries are unchanged
        assert self.identity.commitments == commitments_before, \
            "Reflection must not modify commitments"
        assert self.identity.boundaries == boundaries_before, \
            "Reflection must not modify boundaries"
        result.commitments_preserved = True

        # 7. Write ReflectionCheckpoint to ledger
        summary_hash = hashlib.sha256(
            result.chapter_summary.encode()
        ).hexdigest()
        self.ledger.append_reflection(summary_hash, result.narrative_update)

        self._ticks_since_reflection = 0
        self._reflection_count += 1

        return result

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _build_chapter_summary(self, entries: list[LedgerEntry]) -> str:
        """Build a compressed summary of recent entries."""
        parts = []
        for entry in entries:
            etype = entry.entry_type
            if etype == "user_input":
                text = entry.data.get("payload", {}).get("text", "")
                if text:
                    parts.append(f"User said: {text[:100]}")
            elif etype == "result":
                status = entry.data.get("status", "unknown")
                parts.append(f"Action result: {status}")
            elif etype == "governance_decision":
                decision = entry.data.get("decision", "unknown")
                parts.append(f"Governance: {decision}")
            elif etype == "continuity_event":
                reason = entry.data.get("reason", "unknown")
                parts.append(f"Continuity event: {reason}")
            else:
                parts.append(f"Event: {etype}")

        if not parts:
            return "No significant events in this period."
        return f"Chapter {self._reflection_count + 1}: " + "; ".join(parts)

    def _update_narrative(self, chapter_summary: str) -> str:
        """Update the narrative thread with the new chapter."""
        existing = self.identity.narrative_thread
        if existing:
            return f"{existing}\n---\n{chapter_summary}"
        return chapter_summary

    def _extract_facts(self, entries: list[LedgerEntry]) -> list[SemanticFact]:
        """Extract semantic facts from recent entries."""
        facts = []
        for entry in entries:
            if entry.entry_type == "user_input":
                text = entry.data.get("payload", {}).get("text", "")
                if text and len(text) > 10:
                    facts.append(SemanticFact(
                        fact_id=_new_id(),
                        content=f"User expressed: {text[:200]}",
                        source_refs=[str(entry.sequence)],
                        tags=["user_input"],
                        confidence=0.8,
                        created_at=_now(),
                    ))
            elif entry.entry_type == "result":
                status = entry.data.get("status", "")
                if status:
                    facts.append(SemanticFact(
                        fact_id=_new_id(),
                        content=f"Action completed with status: {status}",
                        source_refs=[str(entry.sequence)],
                        tags=["action_result"],
                        confidence=0.9,
                        created_at=_now(),
                    ))
        return facts

    def _update_capabilities(self, entries: list[LedgerEntry]) -> list[str]:
        """Identify new capabilities from recent experience."""
        new_caps = []
        for entry in entries:
            if entry.entry_type == "evidence_bundle":
                tool_id = entry.data.get("tool_id", "")
                if tool_id and tool_id not in self.self_model.capabilities:
                    new_caps.append(f"used:{tool_id}")
        return new_caps

    def _propose_governance_tuning(self, entries: list[LedgerEntry]) -> list[str]:
        """Propose governance tuning as recommendations (not automatic changes)."""
        proposals = []
        denial_count = sum(
            1 for e in entries
            if e.entry_type == "governance_decision"
            and e.data.get("decision") == "DENY"
        )
        if denial_count > 3:
            proposals.append(
                "High denial rate detected. Consider reviewing plan generation "
                "strategy to reduce governance friction."
            )
        return proposals
