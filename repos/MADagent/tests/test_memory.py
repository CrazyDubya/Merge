"""Tests for memory system: semantic store, working cache, vector index, retrieval."""

import pytest

from mad.ledger import Ledger
from mad.memory import MemorySystem, SemanticFact, SemanticStore, VectorIndex, WorkingCache
from mad.models import Event, EventType


class TestSemanticStore:
    def test_add_and_get(self):
        store = SemanticStore()
        fact = SemanticFact(fact_id="f1", content="Cats are mammals", tags=["biology"])
        store.add(fact)
        assert store.get("f1") is not None
        assert store.get("f1").content == "Cats are mammals"

    def test_search_by_tag(self):
        store = SemanticStore()
        store.add(SemanticFact(fact_id="f1", content="Cats are mammals", tags=["biology"]))
        store.add(SemanticFact(fact_id="f2", content="Python is a language", tags=["tech"]))
        store.add(SemanticFact(fact_id="f3", content="Dogs are mammals", tags=["biology"]))

        biology = store.search_by_tag("biology")
        assert len(biology) == 2

    def test_search_by_keyword(self):
        store = SemanticStore()
        store.add(SemanticFact(fact_id="f1", content="Cats are great pets"))
        store.add(SemanticFact(fact_id="f2", content="Dogs are loyal friends"))
        store.add(SemanticFact(fact_id="f3", content="Cats need independence"))

        results = store.search_by_keyword("cats")
        assert len(results) == 2

    def test_clear(self):
        store = SemanticStore()
        store.add(SemanticFact(fact_id="f1", content="test"))
        store.clear()
        assert store.get("f1") is None
        assert len(store.all_facts()) == 0


class TestWorkingCache:
    def test_push_and_recent(self):
        cache = WorkingCache(max_size=5)
        for i in range(3):
            cache.push({"n": i})
        recent = cache.recent(10)
        assert len(recent) == 3

    def test_fifo_eviction(self):
        cache = WorkingCache(max_size=3)
        for i in range(5):
            cache.push({"n": i})
        assert cache.size == 3
        recent = cache.recent(10)
        assert recent[0]["n"] == 2
        assert recent[-1]["n"] == 4


class TestVectorIndex:
    def test_index_and_search(self):
        idx = VectorIndex()
        idx.index("d1", "cats and dogs are pets")
        idx.index("d2", "python programming language")
        idx.index("d3", "cats love sleeping")

        results = idx.search("cats pets")
        assert "d1" in results
        assert "d3" in results

    def test_empty_search(self):
        idx = VectorIndex()
        results = idx.search("nonexistent")
        assert results == []


class TestMemorySystem:
    def test_ingest_event(self):
        ledger = Ledger()
        mem = MemorySystem(ledger)
        event = Event(type=EventType.USER_INPUT, payload={"text": "hello world"})
        mem.ingest_event(event)
        assert ledger.length == 1
        assert mem.cache.size == 1

    def test_retrieve_recent(self):
        ledger = Ledger()
        mem = MemorySystem(ledger)
        for i in range(5):
            event = Event(type=EventType.USER_INPUT, payload={"text": f"msg {i}"})
            mem.ingest_event(event)
        recent = mem.retrieve_recent(3)
        assert len(recent) == 3

    def test_retrieve_relevant(self):
        ledger = Ledger()
        mem = MemorySystem(ledger)
        mem.ingest_event(Event(type=EventType.USER_INPUT, payload={"text": "cats are great"}))
        mem.ingest_event(Event(type=EventType.USER_INPUT, payload={"text": "python coding"}))
        mem.ingest_event(Event(type=EventType.USER_INPUT, payload={"text": "cats are fun"}))

        results = mem.retrieve_relevant("cats")
        assert len(results) >= 1

    def test_context_window(self):
        ledger = Ledger()
        mem = MemorySystem(ledger)
        mem.ingest_event(Event(type=EventType.USER_INPUT, payload={"text": "important fact"}))
        context = mem.get_context_window(query="important")
        assert "recent" in context
        assert "relevant" in context
        assert "facts" in context

    def test_rebuild_from_ledger(self):
        ledger = Ledger()
        mem = MemorySystem(ledger)
        mem.ingest_event(Event(type=EventType.USER_INPUT, payload={"text": "hello"}))
        mem.ingest_event(Event(type=EventType.USER_INPUT, payload={"text": "world"}))

        mem.cache.clear()
        mem.vector_index.clear()
        assert mem.cache.size == 0

        mem.rebuild_from_ledger()
        assert mem.cache.size > 0
