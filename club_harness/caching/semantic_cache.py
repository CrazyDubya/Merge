"""
Semantic similarity caching for Club Harness.

Adapted from TinyTroupe's semantic_cache.py.

Provides embedding-based cache lookup alongside exact matching,
enabling fuzzy cache hits for semantically similar LLM queries.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional numpy import for embeddings
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


@dataclass
class CacheEntry:
    """An entry in the cache."""
    key: str
    response: Any
    text_representation: str
    embedding: Optional[Any] = None
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def access(self):
        """Record an access to this entry."""
        self.hit_count += 1
        self.last_accessed = time.time()


class SemanticCache:
    """
    Semantic cache using embeddings for similarity-based lookup.

    Enables fuzzy cache hits when exact match fails but semantically similar
    entry exists. Uses hybrid approach: exact match first, then semantic.

    Features from TinyTroupe:
    - Exact hash matching (fast)
    - Semantic similarity matching (when exact fails)
    - LRU eviction for bounded memory
    - Statistics tracking
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        embedding_function: Optional[Callable[[str], Any]] = None,
        max_entries: int = 1000,
        ttl_seconds: Optional[float] = None,
    ):
        """
        Initialize semantic cache.

        Args:
            similarity_threshold: Minimum cosine similarity for cache hit (0.0-1.0)
            embedding_function: Function to generate embeddings from text
            max_entries: Maximum entries to store
            ttl_seconds: Optional time-to-live for cache entries
        """
        self.similarity_threshold = similarity_threshold
        self.embedding_function = embedding_function
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds

        # Main cache storage
        self._cache: Dict[str, CacheEntry] = {}

        # Statistics
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0
        self.total_lookups = 0

    def set_embedding_function(self, func: Callable[[str], Any]):
        """Set the embedding function to use."""
        self.embedding_function = func

    def get(
        self,
        key: str,
        text_representation: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Get a cached response.

        First tries exact match, then semantic matching if available.

        Args:
            key: Cache key (usually a hash of the request)
            text_representation: Text for semantic matching

        Returns:
            Cached response or None
        """
        self.total_lookups += 1

        # Try exact match first
        if key in self._cache:
            entry = self._cache[key]

            # Check TTL
            if self.ttl_seconds and (time.time() - entry.created_at > self.ttl_seconds):
                del self._cache[key]
            else:
                entry.access()
                self.exact_hits += 1
                logger.debug(f"Cache exact hit: {key[:16]}...")
                return entry.response

        # Try semantic match if we have embedding function and text
        if self.embedding_function and text_representation and HAS_NUMPY:
            result = self._find_similar(text_representation)
            if result:
                similar_key, similarity = result
                entry = self._cache[similar_key]
                entry.access()
                self.semantic_hits += 1
                logger.debug(
                    f"Cache semantic hit: {key[:16]}... -> {similar_key[:16]}... "
                    f"(similarity: {similarity:.3f})"
                )
                return entry.response

        self.misses += 1
        return None

    def set(
        self,
        key: str,
        response: Any,
        text_representation: str,
    ) -> None:
        """
        Store a response in the cache.

        Args:
            key: Cache key
            response: Response to cache
            text_representation: Text for semantic matching
        """
        # Generate embedding if possible
        embedding = None
        if self.embedding_function and HAS_NUMPY:
            try:
                embedding = self._get_embedding(text_representation)
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")

        entry = CacheEntry(
            key=key,
            response=response,
            text_representation=text_representation,
            embedding=embedding,
        )

        self._cache[key] = entry

        # Evict if over capacity
        if len(self._cache) > self.max_entries:
            self._evict_lru()

    def _find_similar(self, text_representation: str) -> Optional[Tuple[str, float]]:
        """Find semantically similar cache entry."""
        if not self.embedding_function or not HAS_NUMPY:
            return None

        try:
            query_embedding = self._get_embedding(text_representation)

            best_key = None
            best_similarity = self.similarity_threshold

            for key, entry in self._cache.items():
                if entry.embedding is None:
                    continue

                # Check TTL
                if self.ttl_seconds and (time.time() - entry.created_at > self.ttl_seconds):
                    continue

                similarity = self._cosine_similarity(query_embedding, entry.embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_key = key

            if best_key:
                return (best_key, best_similarity)

        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")

        return None

    def _get_embedding(self, text: str) -> Any:
        """Get normalized embedding for text."""
        if not self.embedding_function:
            raise RuntimeError("Embedding function not set")

        embedding = self.embedding_function(text)

        # Convert to numpy and normalize
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _cosine_similarity(self, vec1: Any, vec2: Any) -> float:
        """Compute cosine similarity (assumes normalized vectors)."""
        return float(np.dot(vec1, vec2))

    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        # Sort by last accessed time
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed
        )

        # Remove oldest 10%
        num_to_remove = max(1, len(self._cache) // 10)
        for key in sorted_keys[:num_to_remove]:
            del self._cache[key]

        logger.debug(f"Evicted {num_to_remove} cache entries")

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = self.exact_hits + self.semantic_hits
        hit_rate = total_hits / self.total_lookups if self.total_lookups > 0 else 0.0

        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "total_lookups": self.total_lookups,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "semantic_enabled": self.embedding_function is not None and HAS_NUMPY,
            "similarity_threshold": self.similarity_threshold,
        }


class CachedLLMRouter:
    """
    LLM router with semantic caching.

    Wraps an existing router and adds caching layer.
    """

    def __init__(
        self,
        router: Any,
        cache: Optional[SemanticCache] = None,
        cache_tiers: Optional[List[str]] = None,
    ):
        """
        Initialize cached router.

        Args:
            router: Underlying LLM router
            cache: Optional pre-configured cache
            cache_tiers: Which tiers to cache (None = all)
        """
        self.router = router
        self.cache = cache or SemanticCache()
        self.cache_tiers = cache_tiers  # e.g., ["free", "cheap"]

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        tier: Optional[str] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> Any:
        """
        Send a chat request with optional caching.

        Args:
            messages: Chat messages
            model: Model to use
            tier: Tier to use
            use_cache: Whether to use cache
            **kwargs: Additional arguments for router

        Returns:
            LLM response
        """
        # Check if this tier should be cached
        should_cache = use_cache
        if self.cache_tiers and tier and tier not in self.cache_tiers:
            should_cache = False

        if should_cache:
            # Create cache key and text representation
            cache_key = self._create_cache_key(messages, model, tier, kwargs)
            text_repr = self._create_text_representation(messages)

            # Try cache
            cached = self.cache.get(cache_key, text_repr)
            if cached is not None:
                return cached

        # Call underlying router
        response = self.router.chat(
            messages=messages,
            model=model,
            tier=tier,
            **kwargs,
        )

        # Store in cache
        if should_cache:
            self.cache.set(cache_key, response, text_repr)

        return response

    def _create_cache_key(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str],
        tier: Optional[str],
        kwargs: Dict,
    ) -> str:
        """Create a unique cache key for the request."""
        key_data = {
            "messages": messages,
            "model": model,
            "tier": tier,
            "kwargs": {k: v for k, v in kwargs.items() if k not in ["max_retries", "retry_delay"]},
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _create_text_representation(self, messages: List[Dict[str, str]]) -> str:
        """Create text representation for semantic matching."""
        return " | ".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in messages
        )

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


def create_text_representation(function_name: str, *args, **kwargs) -> str:
    """
    Create text representation of a function call for embedding.

    Args:
        function_name: Name of the function
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Text representation suitable for embedding
    """
    parts = [f"Function: {function_name}"]

    if args:
        args_str = ", ".join(str(arg)[:100] for arg in args)
        parts.append(f"Args: {args_str}")

    if kwargs:
        kwargs_items = sorted(kwargs.items())
        kwargs_str = ", ".join(f"{k}={str(v)[:50]}" for k, v in kwargs_items)
        parts.append(f"Kwargs: {kwargs_str}")

    return " | ".join(parts)
