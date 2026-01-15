"""
Core type definitions for Club Harness.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ConfidenceSource(Enum):
    """Source of confidence information."""
    INFERENCE = "inference"
    OBSERVATION = "observation"
    ASSUMPTION = "assumption"
    VERIFICATION = "verification"
    MEMORY = "memory"
    LLM = "llm"


@dataclass
class Confidence:
    """Confidence value with provenance tracking."""
    value: float  # 0.0 to 1.0
    source: ConfidenceSource
    timestamp: datetime = field(default_factory=datetime.now)
    evidence: Optional[str] = None

    def decay(self, half_life_hours: float = 24.0) -> "Confidence":
        """Apply time-based decay to confidence."""
        age_hours = (datetime.now() - self.timestamp).total_seconds() / 3600
        decay_factor = 0.5 ** (age_hours / half_life_hours)
        return Confidence(
            value=self.value * decay_factor,
            source=self.source,
            timestamp=self.timestamp,
            evidence=self.evidence,
        )

    def combine(self, other: "Confidence") -> "Confidence":
        """Combine two confidence values (Bayesian-inspired)."""
        combined = 1 - ((1 - self.value) * (1 - other.value))
        return Confidence(
            value=combined,
            source=ConfidenceSource.INFERENCE,
            evidence=f"Combined from {self.source.value} and {other.source.value}",
        )


@dataclass
class Fact:
    """Atomic unit of knowledge with confidence."""
    key: str
    value: Any
    confidence: Confidence


@dataclass
class Goal:
    """Agent goal with verification support."""
    description: str
    predicate: Optional[Callable[["AgentState"], bool]] = None
    dependencies: List["Goal"] = field(default_factory=list)
    priority: int = 0

    def is_satisfied(self, state: "AgentState") -> bool:
        """Check if goal is satisfied."""
        if self.predicate:
            return self.predicate(state)
        return False


@dataclass
class Message:
    """Chat message."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ToolCall:
    """Tool invocation request."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Tool execution result."""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    tool_name: str = ""
    tool_call_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentState:
    """
    Complete agent state (event-sourced design from 12-factor).

    Combines:
    - TinyTroupe: Mental state, persona
    - LisaSimpson: World state, confidence tracking
    - qwen-code: Working memory, task tracking
    """
    # Identity
    name: str
    persona: Dict[str, Any] = field(default_factory=dict)

    # Knowledge (LisaSimpson-inspired)
    facts: Dict[str, Fact] = field(default_factory=dict)

    # Goals
    current_goal: Optional[Goal] = None
    goal_stack: List[Goal] = field(default_factory=list)

    # Conversation
    messages: List[Message] = field(default_factory=list)

    # Task tracking (qwen-code-inspired)
    current_task: Optional[str] = None
    completed_tasks: List[str] = field(default_factory=list)

    # Working memory
    working_memory: Dict[str, Any] = field(default_factory=dict)

    # Metrics
    turn_count: int = 0
    tool_calls_count: int = 0
    total_tokens: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_message(self, message: Message) -> None:
        """Add a message and update timestamp."""
        self.messages.append(message)
        self.updated_at = datetime.now()

    def add_fact(self, key: str, value: Any, confidence: Confidence) -> None:
        """Add or update a fact."""
        self.facts[key] = Fact(key=key, value=value, confidence=confidence)
        self.updated_at = datetime.now()

    def get_fact(self, key: str) -> Optional[Fact]:
        """Get a fact by key."""
        return self.facts.get(key)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "name": self.name,
            "persona": self.persona,
            "current_task": self.current_task,
            "turn_count": self.turn_count,
            "tool_calls_count": self.tool_calls_count,
            "total_tokens": self.total_tokens,
            "message_count": len(self.messages),
        }
