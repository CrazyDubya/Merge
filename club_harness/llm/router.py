"""
LLM Router for Club Harness.
"""

import hashlib
import json
import time
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..core.config import config
from .openrouter import OpenRouterBackend, OpenRouterResponse

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Unified LLM response."""
    content: str
    model: str
    usage: Dict[str, int]
    tool_calls: Optional[List[Dict[str, Any]]] = None
    cached: bool = False

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
    Intelligent LLM routing with cost awareness and optional caching.

    Features:
    - Tier-based model selection
    - Automatic fallback on failures
    - Cost tracking
    - Semantic caching (optional)
    - Multi-provider support (extensible)
    """

    def __init__(
        self,
        enable_cache: bool = False,
        cache_config: Optional[Dict[str, Any]] = None,
        embedding_function: Optional[Callable[[str], Any]] = None,
    ):
        """
        Initialize router.

        Args:
            enable_cache: Enable semantic response caching
            cache_config: Configuration for cache (max_entries, similarity_threshold, etc.)
            embedding_function: Function to generate embeddings for semantic matching
        """
        self._backends: Dict[str, Any] = {}
        self._cache = None
        self._embedding_function = embedding_function

        self._init_backends()

        if enable_cache:
            self._init_cache(cache_config or {})

    def _init_backends(self) -> None:
        """Initialize available backends."""
        if config.llm.api_key:
            self._backends["openrouter"] = OpenRouterBackend(
                api_key=config.llm.api_key,
                default_model=config.llm.model,
            )

    def _init_cache(self, cache_config: Dict[str, Any]) -> None:
        """Initialize the semantic cache."""
        try:
            from ..caching.optimized_cache import OptimizedSemanticCache

            self._cache = OptimizedSemanticCache(
                similarity_threshold=cache_config.get("similarity_threshold", 0.85),
                embedding_function=self._embedding_function,
                embedding_dim=cache_config.get("embedding_dim", 1536),
                max_entries=cache_config.get("max_entries", 10000),
                ttl_seconds=cache_config.get("ttl_seconds", 3600),
                use_faiss=cache_config.get("use_faiss", True),
                faiss_threshold=cache_config.get("faiss_threshold", 20000),
            )
            logger.info("Semantic cache initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize cache: {e}")
            self._cache = None

    def get_backend(self, provider: str = "openrouter") -> Any:
        """Get a specific backend."""
        if provider not in self._backends:
            raise ValueError(f"Backend '{provider}' not initialized")
        return self._backends[provider]

    def _create_cache_key(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
    ) -> str:
        """Create unique cache key for request."""
        key_data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def _create_cache_text(self, messages: List[Dict[str, str]]) -> str:
        """Create text representation for semantic matching."""
        return " | ".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in messages
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        tier: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        provider: str = "openrouter",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        use_cache: bool = True,
    ) -> LLMResponse:
        """
        Send a chat completion request with automatic retry and optional caching.

        Args:
            messages: Conversation messages
            model: Specific model (overrides tier)
            tier: Model tier (free, cheap, standard, reasoning, advanced)
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            tools: Tool definitions for function calling
            provider: Backend provider to use
            max_retries: Maximum retry attempts for rate limits
            retry_delay: Base delay between retries (exponential backoff)
            use_cache: Whether to use cache for this request

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

        # Check cache first (skip for tool calls - responses vary)
        if use_cache and self._cache is not None and tools is None:
            cache_key = self._create_cache_key(messages, model, temperature)
            cache_text = self._create_cache_text(messages)

            cached_response = self._cache.get(cache_key, cache_text)
            if cached_response is not None:
                logger.debug(f"Cache hit for request")
                cached_response.cached = True
                return cached_response

        # Get backend and make request with retry logic
        backend = self.get_backend(provider)
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response: OpenRouterResponse = backend.chat(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                )

                llm_response = LLMResponse(
                    content=response.content,
                    model=response.model,
                    usage=response.usage,
                    tool_calls=response.tool_calls,
                    cached=False,
                )

                # Store in cache (skip tool calls)
                if use_cache and self._cache is not None and tools is None:
                    self._cache.set(cache_key, llm_response, cache_text)

                return llm_response

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                if "429" in error_str or "rate" in error_str or "too many" in error_str:
                    if attempt < max_retries:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(
                            f"Rate limited on attempt {attempt + 1}/{max_retries + 1}. "
                            f"Waiting {wait_time:.1f}s before retry..."
                        )
                        time.sleep(wait_time)
                        continue

                raise

        raise last_error or RuntimeError("All retry attempts failed")

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
        """Async version of chat (caching not yet supported in async)."""
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
        """Try chat with automatic tier fallback on failures."""
        last_error = None

        for tier in tiers:
            try:
                return self.chat(messages=messages, tier=tier, **kwargs)
            except Exception as e:
                last_error = e
                continue

        raise last_error or ValueError("All tiers failed")

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics if caching is enabled."""
        if self._cache is None:
            return None
        return self._cache.get_stats()

    def clear_cache(self) -> None:
        """Clear the response cache."""
        if self._cache is not None:
            self._cache.clear()


# Global router instance (without cache by default for backwards compatibility)
router = LLMRouter()


def create_cached_router(
    embedding_function: Optional[Callable[[str], Any]] = None,
    max_entries: int = 10000,
    **cache_config,
) -> LLMRouter:
    """
    Create a router with caching enabled.

    Args:
        embedding_function: Function to generate embeddings for semantic matching
        max_entries: Maximum cache entries
        **cache_config: Additional cache configuration

    Returns:
        LLMRouter with caching enabled
    """
    return LLMRouter(
        enable_cache=True,
        embedding_function=embedding_function,
        cache_config={"max_entries": max_entries, **cache_config},
    )
