"""
LLM Router for Club Harness.

Combines:
- hivey: Cost-aware tier routing
- llm-council: Multi-model orchestration
- qwen-code: ContentGenerator abstraction
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.config import config
from .openrouter import OpenRouterBackend, OpenRouterResponse


@dataclass
class LLMResponse:
    """Unified LLM response."""
    content: str
    model: str
    usage: Dict[str, int]
    tool_calls: Optional[List[Dict[str, Any]]] = None

    @property
    def input_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def output_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


class LLMRouter:
    """
    Intelligent LLM routing with cost awareness.

    Features:
    - Tier-based model selection
    - Automatic fallback on failures
    - Cost tracking
    - Multi-provider support (extensible)
    """

    def __init__(self):
        self._backends: Dict[str, Any] = {}
        self._init_backends()

    def _init_backends(self) -> None:
        """Initialize available backends."""
        # OpenRouter (primary, supports 400+ models)
        if config.llm.api_key:
            self._backends["openrouter"] = OpenRouterBackend(
                api_key=config.llm.api_key,
                default_model=config.llm.model,
            )

    def get_backend(self, provider: str = "openrouter") -> Any:
        """Get a specific backend."""
        if provider not in self._backends:
            raise ValueError(f"Backend '{provider}' not initialized")
        return self._backends[provider]

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        tier: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        provider: str = "openrouter",
    ) -> LLMResponse:
        """
        Send a chat completion request.

        Args:
            messages: Conversation messages
            model: Specific model (overrides tier)
            tier: Model tier (free, cheap, standard, reasoning, advanced)
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            tools: Tool definitions for function calling
            provider: Backend provider to use

        Returns:
            LLMResponse with content and metadata
        """
        # Determine model
        if model is None:
            if tier:
                model = config.get_model_for_tier(tier)
            else:
                model = config.llm.model

        # Get defaults from config
        temperature = temperature if temperature is not None else config.llm.temperature
        max_tokens = max_tokens if max_tokens is not None else config.llm.max_tokens

        # Get backend and make request
        backend = self.get_backend(provider)
        response: OpenRouterResponse = backend.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )

        return LLMResponse(
            content=response.content,
            model=response.model,
            usage=response.usage,
            tool_calls=response.tool_calls,
        )

    async def chat_async(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        tier: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        provider: str = "openrouter",
    ) -> LLMResponse:
        """Async version of chat."""
        if model is None:
            if tier:
                model = config.get_model_for_tier(tier)
            else:
                model = config.llm.model

        temperature = temperature if temperature is not None else config.llm.temperature
        max_tokens = max_tokens if max_tokens is not None else config.llm.max_tokens

        backend = self.get_backend(provider)
        response: OpenRouterResponse = await backend.chat_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )

        return LLMResponse(
            content=response.content,
            model=response.model,
            usage=response.usage,
            tool_calls=response.tool_calls,
        )

    def chat_with_fallback(
        self,
        messages: List[Dict[str, str]],
        tiers: List[str] = ["free", "cheap", "standard"],
        **kwargs,
    ) -> LLMResponse:
        """
        Try chat with automatic tier fallback on failures.

        Useful for handling rate limits and model availability.
        """
        last_error = None

        for tier in tiers:
            try:
                return self.chat(messages=messages, tier=tier, **kwargs)
            except Exception as e:
                last_error = e
                continue

        raise last_error or ValueError("All tiers failed")


# Global router instance
router = LLMRouter()
