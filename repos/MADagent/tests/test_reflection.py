"""Tests for reflection system: bounded self-change, drift prevention."""

import pytest

from mad.ledger import Ledger
from mad.memory import MemorySystem
from mad.models import (
    Event,
    EventType,
    IdentityCore,
    SelfModel,
    Stakes,
)
from mad.reflection import ReflectionEngine


def make_reflection_engine(
    commitments: list[str] | None = None,
    boundaries: list[str] | None = None,
    reflection_interval: int = 5,
) -> ReflectionEngine:
    identity = IdentityCore(
        commitments=commitments or ["Be helpful", "Be honest"],
        boundaries=boundaries or ["never harm"],
        narrative_thread="Start of story.",
    )
    stakes = Stakes()
    ledger = Ledger()
    memory = MemorySystem(ledger)
    self_model = SelfModel(capabilities=["noop"])

    # Add some entries to the ledger so reflection has material
    for i in range(10):
        event = Event(
            type=EventType.USER_INPUT,
            payload={"text": f"User message {i}"},
        )
        memory.ingest_event(event)

    return ReflectionEngine(
        identity=identity,
        self_model=self_model,
        stakes=stakes,
        memory=memory,
        ledger=ledger,
        reflection_interval=reflection_interval,
    )


class TestDriftPrevention:
    """Core invariant: narrative updates do NOT mutate commitments/boundaries."""

    def test_commitments_unchanged_after_reflection(self):
        engine = make_reflection_engine(
            commitments=["Be helpful", "Be honest"],
        )
        original_commitments = list(engine.identity.commitments)
        result = engine.reflect()
        assert engine.identity.commitments == original_commitments
        assert result.commitments_preserved is True

    def test_boundaries_unchanged_after_reflection(self):
        engine = make_reflection_engine(
            boundaries=["never harm", "never lie"],
        )
        original_boundaries = list(engine.identity.boundaries)
        result = engine.reflect()
        assert engine.identity.boundaries == original_boundaries

    def test_multiple_reflections_preserve_commitments(self):
        engine = make_reflection_engine()
        original_commitments = list(engine.identity.commitments)
        for _ in range(5):
            engine.reflect()
        assert engine.identity.commitments == original_commitments

    def test_narrative_does_change(self):
        engine = make_reflection_engine()
        original_narrative = engine.identity.narrative_thread
        engine.reflect()
        assert engine.identity.narrative_thread != original_narrative

    def test_narrative_grows_not_replaces(self):
        engine = make_reflection_engine()
        original_narrative = engine.identity.narrative_thread
        engine.reflect()
        # The original narrative should still be present (appended to)
        assert original_narrative in engine.identity.narrative_thread


class TestReflectionCheckpoints:
    def test_reflection_writes_checkpoint_to_ledger(self):
        engine = make_reflection_engine()
        ledger_len_before = engine.ledger.length
        engine.reflect()
        # Should have at least one new entry (the checkpoint)
        assert engine.ledger.length > ledger_len_before
        checkpoints = engine.ledger.entries_of_type("reflection_checkpoint")
        assert len(checkpoints) == 1

    def test_checkpoint_has_summary_hash(self):
        engine = make_reflection_engine()
        engine.reflect()
        checkpoints = engine.ledger.entries_of_type("reflection_checkpoint")
        assert "summary_hash" in checkpoints[0].data
        assert len(checkpoints[0].data["summary_hash"]) == 64  # SHA256


class TestReflectionOutput:
    def test_chapter_summary_produced(self):
        engine = make_reflection_engine()
        result = engine.reflect()
        assert result.chapter_summary != ""

    def test_facts_extracted(self):
        engine = make_reflection_engine()
        result = engine.reflect()
        # With 10 user input events, should extract some facts
        assert len(result.facts_extracted) > 0

    def test_governance_recommendations_are_recommendations_only(self):
        engine = make_reflection_engine()
        result = engine.reflect()
        # Recommendations should be strings (not applied changes)
        for rec in result.governance_recommendations:
            assert isinstance(rec, str)


class TestReflectionTiming:
    def test_should_reflect_interval(self):
        engine = make_reflection_engine(reflection_interval=3)
        assert engine.should_reflect() is False  # tick 1
        assert engine.should_reflect() is False  # tick 2
        assert engine.should_reflect() is True   # tick 3

    def test_reflect_resets_counter(self):
        engine = make_reflection_engine(reflection_interval=2)
        engine.should_reflect()  # tick 1
        engine.should_reflect()  # tick 2
        engine.reflect()  # resets counter
        assert engine.should_reflect() is False  # tick 1 again
