"""Tests for append-only ledger with hash chaining."""

import json
import tempfile
from pathlib import Path

import pytest

from mad.ledger import GENESIS_HASH, Ledger, TamperError
from mad.models import Event, EventType


class TestLedgerAppend:
    def test_empty_ledger(self):
        ledger = Ledger()
        assert ledger.length == 0
        assert ledger.head_hash == GENESIS_HASH

    def test_single_append(self):
        ledger = Ledger()
        entry = ledger.append("test", {"key": "value"})
        assert ledger.length == 1
        assert entry.sequence == 0
        assert entry.entry_type == "test"
        assert entry.data == {"key": "value"}
        assert entry.content_hash != ""
        assert entry.prev_hash == GENESIS_HASH
        assert entry.chain_hash != ""

    def test_sequential_appends(self):
        ledger = Ledger()
        e1 = ledger.append("a", {"n": 1})
        e2 = ledger.append("b", {"n": 2})
        e3 = ledger.append("c", {"n": 3})

        assert e1.sequence == 0
        assert e2.sequence == 1
        assert e3.sequence == 2
        assert e2.prev_hash == e1.chain_hash
        assert e3.prev_hash == e2.chain_hash
        assert ledger.head_hash == e3.chain_hash

    def test_append_event(self):
        ledger = Ledger()
        event = Event(type=EventType.USER_INPUT, payload={"text": "hello"})
        entry = ledger.append_event(event)
        assert entry.entry_type == "user_input"
        assert entry.data["payload"]["text"] == "hello"

    def test_append_never_rewrites(self):
        ledger = Ledger()
        e1 = ledger.append("a", {"n": 1})
        hash_after_first = e1.chain_hash

        ledger.append("b", {"n": 2})
        # First entry should be unchanged
        assert ledger.get(0).chain_hash == hash_after_first


class TestLedgerIntegrity:
    def test_verify_empty(self):
        ledger = Ledger()
        assert ledger.verify_integrity() is True

    def test_verify_valid_chain(self):
        ledger = Ledger()
        for i in range(10):
            ledger.append("test", {"n": i})
        assert ledger.verify_integrity() is True

    def test_detect_content_tamper(self):
        ledger = Ledger()
        ledger.append("test", {"n": 1})
        ledger.append("test", {"n": 2})

        # Tamper with content
        ledger._entries[0].data = {"n": 999}
        with pytest.raises(TamperError):
            ledger.verify_integrity()

    def test_detect_chain_tamper(self):
        ledger = Ledger()
        ledger.append("test", {"n": 1})
        ledger.append("test", {"n": 2})

        # Tamper with chain hash
        ledger._entries[0].chain_hash = "bogus"
        with pytest.raises(TamperError):
            ledger.verify_integrity()

    def test_no_gaps(self):
        ledger = Ledger()
        for i in range(5):
            ledger.append("test", {"n": i})
        assert ledger.has_gaps() is False

    def test_detect_gaps(self):
        ledger = Ledger()
        for i in range(5):
            ledger.append("test", {"n": i})
        # Artificially create a gap
        ledger._entries[2].sequence = 99
        assert ledger.has_gaps() is True


class TestLedgerReplay:
    def test_replay_returns_all_entries(self):
        ledger = Ledger()
        for i in range(5):
            ledger.append("test", {"n": i})
        replayed = ledger.replay()
        assert len(replayed) == 5
        for i, entry in enumerate(replayed):
            assert entry.sequence == i
            assert entry.data["n"] == i

    def test_replay_verifies_integrity(self):
        ledger = Ledger()
        ledger.append("test", {"n": 1})
        ledger._entries[0].data = {"n": 999}
        with pytest.raises(TamperError):
            ledger.replay()


class TestLedgerPersistence:
    def test_persist_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ledger.jsonl"

            # Write
            ledger1 = Ledger(path=path)
            for i in range(5):
                ledger1.append("test", {"n": i})

            # Load
            ledger2 = Ledger(path=path)
            assert ledger2.length == 5
            for i in range(5):
                assert ledger2.get(i).data["n"] == i

    def test_jsonl_roundtrip(self):
        ledger = Ledger()
        for i in range(5):
            ledger.append("test", {"n": i})
        jsonl = ledger.to_jsonl()
        restored = Ledger.from_jsonl(jsonl)
        assert restored.length == 5
        assert restored.verify_integrity() is True


class TestLedgerQueries:
    def test_get_by_sequence(self):
        ledger = Ledger()
        ledger.append("a", {"x": 1})
        ledger.append("b", {"x": 2})
        assert ledger.get(0).entry_type == "a"
        assert ledger.get(1).entry_type == "b"

    def test_get_invalid_sequence(self):
        ledger = Ledger()
        with pytest.raises(Exception):
            ledger.get(0)

    def test_recent(self):
        ledger = Ledger()
        for i in range(20):
            ledger.append("test", {"n": i})
        recent = ledger.recent(5)
        assert len(recent) == 5
        assert recent[0].data["n"] == 15
        assert recent[-1].data["n"] == 19

    def test_entries_of_type(self):
        ledger = Ledger()
        ledger.append("alpha", {"n": 1})
        ledger.append("beta", {"n": 2})
        ledger.append("alpha", {"n": 3})
        alphas = ledger.entries_of_type("alpha")
        assert len(alphas) == 2
