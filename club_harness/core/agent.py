"""
Unified Agent for Club Harness.

Combines patterns from:
- TinyTroupe: Persona-based agents with memory
- LisaSimpson: Deliberative planning with confidence
- qwen-code: Turn-based execution loop
- 12-factor: Stateless reducer pattern
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .types import AgentState, Confidence, ConfidenceSource, Goal, Message
from .config import config

# Avoid circular import
if TYPE_CHECKING:
    from ..llm.router import LLMRouter, LLMResponse


@dataclass
class AgentResult:
    """Result of agent execution."""
    success: bool
    output: str
    state: AgentState
    turns_used: int
    total_tokens: int
    error: Optional[str] = None


class Agent:
    """
    Unified agent combining best practices from multiple frameworks.

    Features:
    - Persona-based identity (TinyTroupe)
    - Goal-oriented execution (LisaSimpson)
    - Turn-based loop (qwen-code)
    - Stateless steps (12-factor)
    - Cost-aware LLM routing (hivey)
    """

    def __init__(
        self,
        name: str,
        instructions: str,
        persona: Optional[Dict[str, Any]] = None,
        llm_router: Optional["LLMRouter"] = None,
        model: Optional[str] = None,
        tier: Optional[str] = None,
    ):
        self.name = name
        self.instructions = instructions
        self.persona = persona or {}

        # Lazy import to avoid circular dependency
        if llm_router is None:
            from ..llm.router import router as default_router
            self.llm_router = default_router
        else:
            self.llm_router = llm_router

        self.model = model
        self.tier = tier or "free"

        # Initialize state
        self.state = AgentState(
            name=name,
            persona=self.persona,
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt from instructions and persona."""
        parts = [f"You are {self.name}."]

        if self.instructions:
            parts.append(f"\nInstructions: {self.instructions}")

        if self.persona:
            parts.append("\nPersona:")
            for key, value in self.persona.items():
                parts.append(f"  - {key}: {value}")

        return "\n".join(parts)

    def _format_messages(
        self,
        user_message: str,
        include_history: bool = True,
    ) -> List[Dict[str, str]]:
        """Format messages for LLM request."""
        messages = [{"role": "system", "content": self._build_system_prompt()}]

        # Include conversation history (respecting context limits)
        if include_history and self.state.messages:
            # 12-factor: Keep context under 40% utilization
            max_history = 20  # Simplified for now
            for msg in self.state.messages[-max_history:]:
                messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": user_message})
        return messages

    def step(self, user_message: str) -> "LLMResponse":
        """
        Execute a single turn (stateless reducer pattern from 12-factor).

        Args:
            user_message: User input for this turn

        Returns:
            LLMResponse with assistant's reply
        """
        # Record user message
        self.state.add_message(Message(role="user", content=user_message))
        self.state.turn_count += 1

        # Build request
        messages = self._format_messages(user_message)

        # Call LLM
        response = self.llm_router.chat(
            messages=messages,
            model=self.model,
            tier=self.tier,
        )

        # Record assistant response
        self.state.add_message(Message(role="assistant", content=response.content))
        self.state.total_tokens += response.total_tokens

        return response

    def chat(self, message: str) -> str:
        """Simple chat interface - single turn."""
        response = self.step(message)
        return response.content

    def run(
        self,
        task: str,
        max_turns: Optional[int] = None,
        stop_condition: Optional[Callable[[AgentState], bool]] = None,
    ) -> AgentResult:
        """
        Run agent on a task with multi-turn loop.

        Args:
            task: Task description
            max_turns: Maximum turns (defaults to config)
            stop_condition: Optional function to check if done

        Returns:
            AgentResult with outcome and state
        """
        max_turns = max_turns or config.agent.max_turns
        self.state.current_task = task

        # Initial turn
        response = self.step(task)
        output = response.content

        turns_used = 1

        # Multi-turn loop (if needed)
        while turns_used < max_turns:
            # Check stop condition
            if stop_condition and stop_condition(self.state):
                break

            # Check if LLM indicates completion
            if self._is_complete(output):
                break

            # Continue conversation (simplified - would need tool handling)
            response = self.step("Continue with the task.")
            output = response.content
            turns_used += 1

        # Mark task complete
        self.state.completed_tasks.append(task)
        self.state.current_task = None

        return AgentResult(
            success=True,
            output=output,
            state=self.state,
            turns_used=turns_used,
            total_tokens=self.state.total_tokens,
        )

    def _is_complete(self, response: str) -> bool:
        """Check if response indicates task completion."""
        completion_markers = [
            "task complete",
            "done",
            "finished",
            "completed",
            "here is the result",
        ]
        response_lower = response.lower()
        return any(marker in response_lower for marker in completion_markers)

    def reset(self) -> None:
        """Reset agent state for a new session."""
        self.state = AgentState(
            name=self.name,
            persona=self.persona,
        )


class AgentBuilder:
    """
    Fluent builder for creating agents.

    Inspired by LisaSimpson's builder pattern.
    """

    def __init__(self, name: str):
        self._name = name
        self._instructions = ""
        self._persona: Dict[str, Any] = {}
        self._model: Optional[str] = None
        self._tier: Optional[str] = None

    def with_instructions(self, instructions: str) -> "AgentBuilder":
        """Set agent instructions."""
        self._instructions = instructions
        return self

    def with_persona(self, **kwargs) -> "AgentBuilder":
        """Add persona attributes."""
        self._persona.update(kwargs)
        return self

    def with_model(self, model: str) -> "AgentBuilder":
        """Set specific model."""
        self._model = model
        return self

    def with_tier(self, tier: str) -> "AgentBuilder":
        """Set model tier (free, cheap, standard, reasoning, advanced)."""
        self._tier = tier
        return self

    def build(self) -> Agent:
        """Build the agent."""
        return Agent(
            name=self._name,
            instructions=self._instructions,
            persona=self._persona,
            model=self._model,
            tier=self._tier,
        )
