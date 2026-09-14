"""Append-only ledger with hash chaining and replay support.

WORM semantics: entries are never rewritten, only appended.
Each entry is content-hashed and chain-hashed for tamper evidence.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from mad.models import (
    Event,
    EvidenceBundle,
    GovernanceDecision,
    LedgerEntry,
    Plan,
    Result,
    _hash_dict,
    _now,
)

GENESIS_HASH = "0" * 64


class LedgerError(Exception):
    pass


class TamperError(LedgerError):
    pass


class Ledger:
    """Append-only ledger with hash-chained entries.

    Thread-safe. Supports persistence to JSONL file and full replay.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._entries: list[LedgerEntry] = []
        self._lock = threading.Lock()
        self._path = path
        if path and path.exists():
            self._load(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, entry_type: str, data: dict[str, Any]) -> LedgerEntry:
        """Append a new entry. Returns the committed LedgerEntry."""
        assert entry_type, "entry_type must be non-empty"
        assert isinstance(data, dict), f"data must be a dict, got {type(data).__name__}"
        with self._lock:
            seq = len(self._entries)
            prev_hash = (
                self._entries[-1].chain_hash if self._entries else GENESIS_HASH
            )
            entry = LedgerEntry(
                sequence=seq,
                timestamp=_now(),
                entry_type=entry_type,
                data=data,
            )
            entry.compute_content_hash()
            entry.compute_chain_hash(prev_hash)
            self._entries.append(entry)
            assert self._entries[-1].sequence == seq, "sequence integrity violated"
            if self._path:
                self._persist_entry(entry)
            return entry

    def append_event(self, event: Event) -> LedgerEntry:
        return self.append(event.type.value, event.to_dict())

    def append_plan(self, plan: Plan) -> LedgerEntry:
        return self.append("plan", plan.to_dict())

    def append_governance(self, decision: GovernanceDecision) -> LedgerEntry:
        return self.append("governance_decision", decision.to_dict())

    def append_result(self, result: Result) -> LedgerEntry:
        return self.append("result", result.to_dict())

    def append_evidence(self, bundle: EvidenceBundle) -> LedgerEntry:
        return self.append("evidence_bundle", bundle.to_dict())

    def append_reflection(self, summary_hash: str, narrative: str) -> LedgerEntry:
        return self.append("reflection_checkpoint", {
            "summary_hash": summary_hash,
            "narrative_snapshot": narrative,
        })

    def append_continuity_event(self, reason: str, cost_increment: float) -> LedgerEntry:
        return self.append("continuity_event", {
            "reason": reason,
            "cost_increment": cost_increment,
        })

    @property
    def length(self) -> int:
        return len(self._entries)

    @property
    def head_hash(self) -> str:
        if not self._entries:
            return GENESIS_HASH
        return self._entries[-1].chain_hash

    def get(self, sequence: int) -> LedgerEntry:
        if 0 <= sequence < len(self._entries):
            return self._entries[sequence]
        raise LedgerError(f"No entry at sequence {sequence}")

    def entries(self, start: int = 0, end: int | None = None) -> list[LedgerEntry]:
        return list(self._entries[start:end])

    def recent(self, n: int = 10) -> list[LedgerEntry]:
        return list(self._entries[-n:])

    def entries_of_type(self, entry_type: str) -> list[LedgerEntry]:
        return [e for e in self._entries if e.entry_type == entry_type]

    # ------------------------------------------------------------------
    # Integrity verification
    # ------------------------------------------------------------------

    def verify_integrity(self) -> bool:
        """Verify the full hash chain. Raises TamperError on failure."""
        prev_hash = GENESIS_HASH
        for entry in self._entries:
            expected_content = _hash_dict({
                "sequence": entry.sequence,
                "timestamp": entry.timestamp,
                "entry_type": entry.entry_type,
                "data": entry.data,
            })
            if entry.content_hash != expected_content:
                raise TamperError(
                    f"Content hash mismatch at seq {entry.sequence}"
                )
            expected_chain = hashlib.sha256(
                (entry.content_hash + prev_hash).encode()
            ).hexdigest()
            if entry.chain_hash != expected_chain:
                raise TamperError(
                    f"Chain hash mismatch at seq {entry.sequence}"
                )
            prev_hash = entry.chain_hash
        return True

    def has_gaps(self) -> bool:
        """Check for sequence gaps."""
        for i, entry in enumerate(self._entries):
            if entry.sequence != i:
                return True
        return False

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(self) -> list[LedgerEntry]:
        """Return all entries in order for replay. Verifies integrity first."""
        self.verify_integrity()
        return list(self._entries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_entry(self, entry: LedgerEntry) -> None:
        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def _load(self, path: Path) -> None:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    self._entries.append(LedgerEntry.from_dict(d))

    def to_jsonl(self) -> str:
        """Serialize entire ledger to JSONL string."""
        return "\n".join(json.dumps(e.to_dict()) for e in self._entries)

    @classmethod
    def from_jsonl(cls, data: str) -> Ledger:
        """Reconstruct ledger from JSONL string."""
        ledger = cls()
        for line in data.strip().split("\n"):
            if line:
                d = json.loads(line)
                ledger._entries.append(LedgerEntry.from_dict(d))
        return ledger
