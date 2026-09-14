"""Memory system: semantic store, working cache, and retrieval.

Derived stores sit atop the append-only ledger. They are editable
and can be fully reconstructed from the ledger if needed.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from mad.ledger import Ledger
from mad.models import Event, EventType, LedgerEntry


@dataclass
class SemanticFact:
    """A distilled fact extracted from experience."""
    fact_id: str = ""
    content: str = ""
    source_refs: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    confidence: float = 1.0
    created_at: str = ""


class SemanticStore:
    """Distilled knowledge store. Can be rebuilt from ledger."""

    def __init__(self, max_facts: int = 10000) -> None:
        self._facts: dict[str, SemanticFact] = {}
        self._tag_index: dict[str, list[str]] = defaultdict(list)
        self._max_facts = max_facts

    def add(self, fact: SemanticFact) -> None:
        self._facts[fact.fact_id] = fact
        for tag in fact.tags:
            if fact.fact_id not in self._tag_index[tag]:
                self._tag_index[tag].append(fact.fact_id)
        # Evict lowest-confidence facts when over capacity
        if len(self._facts) > self._max_facts:
            self._evict_lowest_confidence()

    def _evict_lowest_confidence(self) -> None:
        """Remove the lowest-confidence fact to stay within max_facts."""
        worst_id = min(self._facts, key=lambda fid: self._facts[fid].confidence)
        del self._facts[worst_id]
        for tag_ids in self._tag_index.values():
            if worst_id in tag_ids:
                tag_ids.remove(worst_id)

    def get(self, fact_id: str) -> SemanticFact | None:
        return self._facts.get(fact_id)

    def search_by_tag(self, tag: str) -> list[SemanticFact]:
        return [self._facts[fid] for fid in self._tag_index.get(tag, [])
                if fid in self._facts]

    def search_by_keyword(self, keyword: str) -> list[SemanticFact]:
        kw = keyword.lower()
        return [f for f in self._facts.values() if kw in f.content.lower()]

    def all_facts(self) -> list[SemanticFact]:
        return list(self._facts.values())

    def clear(self) -> None:
        self._facts.clear()
        self._tag_index.clear()


class WorkingCache:
    """Short-term context window. FIFO with max size."""

    def __init__(self, max_size: int = 50) -> None:
        self._items: list[dict[str, Any]] = []
        self._max_size = max_size

    def push(self, item: dict[str, Any]) -> None:
        self._items.append(item)
        if len(self._items) > self._max_size:
            self._items.pop(0)

    def recent(self, n: int = 10) -> list[dict[str, Any]]:
        return list(self._items[-n:])

    def clear(self) -> None:
        self._items.clear()

    @property
    def size(self) -> int:
        return len(self._items)


class VectorIndex:
    """Minimal keyword-based retrieval index.

    In production this would use actual embeddings; for MVP we use
    simple term frequency matching.
    """

    def __init__(self, max_documents: int = 10000) -> None:
        self._documents: dict[str, str] = {}  # id -> text
        self._insertion_order: list[str] = []
        self._max_documents = max_documents

    def index(self, doc_id: str, text: str) -> None:
        if doc_id not in self._documents:
            self._insertion_order.append(doc_id)
        self._documents[doc_id] = text.lower()
        # Evict oldest documents when over capacity
        while len(self._documents) > self._max_documents and self._insertion_order:
            oldest = self._insertion_order.pop(0)
            self._documents.pop(oldest, None)

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """Return doc_ids ranked by keyword overlap."""
        terms = set(query.lower().split())
        scored: list[tuple[str, int]] = []
        for doc_id, text in self._documents.items():
            score = sum(1 for t in terms if t in text)
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in scored[:top_k]]

    def clear(self) -> None:
        self._documents.clear()

    @property
    def size(self) -> int:
        return len(self._documents)


class MemorySystem:
    """Unified memory interface combining all stores."""

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self.semantic = SemanticStore()
        self.cache = WorkingCache()
        self.vector_index = VectorIndex()

    def ingest_event(self, event: Event) -> None:
        """Write event to ledger and update derived stores."""
        self.ledger.append_event(event)
        self.cache.push(event.to_dict())
        # Index user inputs and observations for retrieval
        if event.type in (EventType.USER_INPUT, EventType.OBSERVATION):
            text = event.payload.get("text", "")
            if text:
                self.vector_index.index(event.event_id, text)

    def retrieve_recent(self, n: int = 10) -> list[dict[str, Any]]:
        """Get recent items from working cache."""
        return self.cache.recent(n)

    def retrieve_relevant(self, query: str, top_k: int = 5) -> list[LedgerEntry]:
        """Semantic retrieval: find relevant ledger entries."""
        doc_ids = self.vector_index.search(query, top_k)
        results = []
        for entry in self.ledger.entries():
            eid = entry.data.get("event_id", "")
            if eid in doc_ids:
                results.append(entry)
        return results

    def get_context_window(self, query: str = "", recent_n: int = 10,
                           relevant_k: int = 5) -> dict[str, Any]:
        """Build a context window combining recent + relevant memory."""
        context: dict[str, Any] = {
            "recent": self.retrieve_recent(recent_n),
            "relevant": [],
            "facts": [],
        }
        if query:
            context["relevant"] = [
                e.to_dict() for e in self.retrieve_relevant(query, relevant_k)
            ]
            context["facts"] = [
                {"content": f.content, "tags": f.tags}
                for f in self.semantic.search_by_keyword(query)
            ]
        return context

    def rebuild_from_ledger(self) -> None:
        """Reconstruct all derived stores from the ledger."""
        self.semantic.clear()
        self.cache.clear()
        self.vector_index.clear()
        for entry in self.ledger.entries():
            data = entry.data
            self.cache.push(data)
            if entry.entry_type in ("user_input", "observation"):
                text = data.get("payload", {}).get("text", "")
                if not text:
                    text = data.get("text", "")
                if text:
                    eid = data.get("event_id", str(entry.sequence))
                    self.vector_index.index(eid, text)
