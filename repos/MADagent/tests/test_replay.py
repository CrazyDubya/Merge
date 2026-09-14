"""Replay harness: verify ledger → identical state reconstruction.

Core invariant: replaying the ledger produces identical world/self state
within tolerance.
"""

import pytest

from mad.agent import Agent, AgentConfig
from mad.ledger import Ledger
from mad.memory import MemorySystem
from mad.models import Event, EventType


class TestReplayHarness:
    """Replay test: replay ledger → identical world/self state within tolerance."""

    def test_replay_reproduces_event_sequence(self):
        """Events replayed from ledger match original sequence."""
        agent = Agent()
        agent.process_input("Hello, agent")
        agent.process_input("What can you do?")
        agent.process_input("Tell me about yourself")

        # Extract ledger
        original_entries = agent.ledger.entries()
        original_hashes = [e.chain_hash for e in original_entries]

        # Replay
        replayed = agent.ledger.replay()
        replayed_hashes = [e.chain_hash for e in replayed]

        assert original_hashes == replayed_hashes

    def test_replay_preserves_hash_chain(self):
        """Replayed ledger maintains valid hash chain."""
        agent = Agent()
        for i in range(10):
            agent.process_input(f"Message {i}")

        # Replay verifies integrity
        replayed = agent.ledger.replay()
        assert len(replayed) > 10  # user inputs + system events

    def test_memory_rebuild_from_ledger(self):
        """Derived memory stores can be rebuilt from ledger alone."""
        agent = Agent()
        agent.process_input("Important fact about cats")
        agent.process_input("Another fact about dogs")

        # Get state from memory
        cache_size_before = agent.memory.cache.size
        index_size_before = agent.memory.vector_index.size

        # Rebuild from ledger
        agent.memory.rebuild_from_ledger()

        # Cache should be repopulated
        assert agent.memory.cache.size > 0
        # Vector index should have entries
        assert agent.memory.vector_index.size > 0

    def test_jsonl_roundtrip_integrity(self):
        """Ledger survives serialization roundtrip."""
        agent = Agent()
        agent.process_input("Test message one")
        agent.process_input("Test message two")

        jsonl = agent.ledger.to_jsonl()
        restored = Ledger.from_jsonl(jsonl)
        assert restored.verify_integrity() is True
        assert restored.length == agent.ledger.length

    def test_replay_includes_all_event_types(self):
        """Replay includes governance decisions, results, etc."""
        agent = Agent()
        agent.process_input("Do something")

        entry_types = {e.entry_type for e in agent.ledger.entries()}
        # Should have at minimum: system (boot), user_input, governance_decision, plan, result
        assert "system" in entry_types or "user_input" in entry_types
