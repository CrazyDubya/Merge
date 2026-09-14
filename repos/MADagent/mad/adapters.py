"""I/O Adapters: normalize messy reality into typed events.

Handles: user text input, observation normalization, redaction,
rate limiting, and schema parsing.
"""

from __future__ import annotations

import re
from typing import Any

from mad.models import Event, EventType, SecurityContext, _new_id, _now


class InputAdapter:
    """Converts raw user input into typed Event objects."""

    def __init__(self, operator: str = "user", session_id: str | None = None) -> None:
        self.operator = operator
        self.session_id = session_id or _new_id()

    def user_text(self, text: str, trace_id: str | None = None) -> Event:
        """Convert user text input to a typed Event."""
        assert isinstance(text, str), f"text must be a string, got {type(text).__name__}"
        assert text.strip(), "text must be non-empty"
        return Event(
            type=EventType.USER_INPUT,
            payload={"text": self._sanitize(text)},
            provenance=f"user:{self.operator}",
            security_context=SecurityContext(
                operator=self.operator,
                session_id=self.session_id,
            ),
            trace_id=trace_id or _new_id(),
        )

    def observation(self, data: dict[str, Any], source: str = "environment",
                    trace_id: str | None = None) -> Event:
        """Convert an external observation to a typed Event."""
        return Event(
            type=EventType.OBSERVATION,
            payload=self._redact_sensitive(data),
            provenance=f"observation:{source}",
            security_context=SecurityContext(
                operator=self.operator,
                session_id=self.session_id,
            ),
            trace_id=trace_id or _new_id(),
        )

    def system_event(self, message: str, trace_id: str | None = None) -> Event:
        """Create a system event."""
        return Event(
            type=EventType.SYSTEM,
            payload={"message": message},
            provenance="system",
            security_context=SecurityContext(
                operator="system",
                session_id=self.session_id,
            ),
            trace_id=trace_id or _new_id(),
        )

    @staticmethod
    def _sanitize(text: str) -> str:
        """Basic input sanitization (remove null bytes, normalize whitespace)."""
        text = text.replace("\x00", "")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    _SENSITIVE_KEYS = frozenset({"password", "secret", "token", "api_key", "private_key"})
    _MAX_REDACT_DEPTH = 10

    @staticmethod
    def _redact_sensitive(
        data: dict[str, Any], _depth: int = 0,
    ) -> dict[str, Any]:
        """Redact known sensitive field patterns.

        Bounded to _MAX_REDACT_DEPTH levels of nesting to prevent
        stack overflow on adversarial input.
        """
        if _depth >= InputAdapter._MAX_REDACT_DEPTH:
            return {"_redacted": "[TOO_DEEP]"}
        redacted = {}
        for k, v in data.items():
            if any(sk in k.lower() for sk in InputAdapter._SENSITIVE_KEYS):
                redacted[k] = "[REDACTED]"
            elif isinstance(v, dict):
                redacted[k] = InputAdapter._redact_sensitive(v, _depth + 1)
            else:
                redacted[k] = v
        return redacted


class OutputAdapter:
    """Formats agent outputs for external consumption."""

    @staticmethod
    def format_text(text: str) -> dict[str, Any]:
        return {"type": "text", "content": text}

    @staticmethod
    def format_error(error: str, code: str = "UNKNOWN") -> dict[str, Any]:
        return {"type": "error", "code": code, "message": error}

    @staticmethod
    def format_escalation(plan_summary: str, reasons: list[str],
                          evidence_refs: list[str]) -> dict[str, Any]:
        return {
            "type": "escalation",
            "plan_summary": plan_summary,
            "reasons": reasons,
            "evidence_refs": evidence_refs,
            "message": "Human review required.",
        }
