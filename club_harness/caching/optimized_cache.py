"""
Optimized Semantic Cache with tiered architecture.

Implements a 3-tier caching strategy:
- Tier 1: Exact hash match (O(1))
- Tier 2: NumPy batched vector search (O(n) but fast)
- Tier 3: Optional FAISS HNSW for large scale (O(log n))

Scaling characteristics:
- <1K entries: All tiers perform well
- 1K-20K: Tier 2 handles efficiently
- 20K+: Tier 3 (FAISS) recommended
"""

import hashlib
import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    response: Any
    text_hash: str
    embedding_idx: int = -1  # Index in embedding matrix
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    hit_count: int = 0

    def access(self):
        self.hit_count += 1
        self.last_accessed = time.time()


class OptimizedSemanticCache:
    """
    High-performance semantic cache with tiered lookup.

    Architecture:
    1. Exact match via hash (O(1)) - catches identical queries
    2. Batched numpy similarity (O(n)) - fast for <20K entries
    3. FAISS HNSW index (O(log n)) - optional, for 20K+ entries

    Features:
    - Heap-based LRU eviction (O(log n) vs O(n log n))
    - Background TTL cleanup
    - Contiguous embedding storage
    - Thread-safe operations
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        embedding_function: Optional[Callable[[str], Any]] = None,
        embedding_dim: int = 1536,
        max_entries: int = 10000,
        ttl_seconds: Optional[float] = None,
        use_faiss: bool = False,
        faiss_threshold: int = 20000,
        cleanup_interval: float = 60.0,
    ):
        """
        Initialize optimized cache.

        Args:
            similarity_threshold: Min cosine similarity for semantic match
            embedding_function: Function to generate embeddings
            embedding_dim: Dimension of embeddings
            max_entries: Maximum cache entries
            ttl_seconds: Time-to-live for entries (None = no expiry)
            use_faiss: Enable FAISS backend for large scale
            faiss_threshold: Entry count to switch to FAISS
            cleanup_interval: Seconds between background cleanup runs
        """
        self.similarity_threshold = similarity_threshold
        self.embedding_function = embedding_function
        self.embedding_dim = embedding_dim
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.use_faiss = use_faiss and HAS_FAISS
        self.faiss_threshold = faiss_threshold
        self.cleanup_interval = cleanup_interval

        # Tier 1: Exact match cache (hash -> key)
        self._exact_cache: Dict[str, str] = {}

        # Entry storage
        self._entries: Dict[str, CacheEntry] = {}

        # Tier 2: Contiguous embedding storage
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_keys: List[str] = []  # Maps index -> key
        self._next_embedding_idx: int = 0

        # Tier 3: FAISS index (lazy initialized)
        self._faiss_index: Optional[Any] = None
        self._faiss_id_map: Dict[int, str] = {}  # FAISS ID -> key

        # LRU heap: (last_accessed, key)
        self._lru_heap: List[Tuple[float, str]] = []

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "faiss_hits": 0,
            "misses": 0,
            "total_lookups": 0,
        }

        # Background cleanup
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()

        # Initialize embedding storage
        if HAS_NUMPY:
            self._embeddings = np.zeros(
                (max_entries, embedding_dim),
                dtype=np.float32
            )

    def start_background_cleanup(self):
        """Start background TTL cleanup thread."""
        if self._cleanup_thread is not None:
            return

        self._stop_cleanup.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self._cleanup_thread.start()
        logger.debug("Background cleanup started")

    def stop_background_cleanup(self):
        """Stop background cleanup thread."""
        if self._cleanup_thread is None:
            return

        self._stop_cleanup.set()
        self._cleanup_thread.join(timeout=5.0)
        self._cleanup_thread = None
        logger.debug("Background cleanup stopped")

    def _cleanup_loop(self):
        """Background cleanup loop."""
        while not self._stop_cleanup.wait(self.cleanup_interval):
            self._cleanup_expired()

    def _cleanup_expired(self):
        """Remove expired entries."""
        if not self.ttl_seconds:
            return

        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self._entries.items()
                if current_time - entry.created_at > self.ttl_seconds
            ]

            for key in expired_keys:
                self._remove_entry(key)

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

    def _hash_text(self, text: str) -> str:
        """Create hash for exact matching."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def get(
        self,
        key: str,
        text_representation: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Get cached response using tiered lookup.

        Args:
            key: Cache key
            text_representation: Text for semantic matching

        Returns:
            Cached response or None
        """
        with self._lock:
            self.stats["total_lookups"] += 1

            # Tier 1: Exact key match
            if key in self._entries:
                entry = self._entries[key]
                if not self._is_expired(entry):
                    entry.access()
                    self.stats["exact_hits"] += 1
                    return entry.response

            # Tier 1b: Exact text hash match
            if text_representation:
                text_hash = self._hash_text(text_representation)
                if text_hash in self._exact_cache:
                    cached_key = self._exact_cache[text_hash]
                    if cached_key in self._entries:
                        entry = self._entries[cached_key]
                        if not self._is_expired(entry):
                            entry.access()
                            self.stats["exact_hits"] += 1
                            return entry.response

            # Tier 2/3: Semantic search
            if text_representation and self.embedding_function and HAS_NUMPY:
                result = self._semantic_search(text_representation)
                if result:
                    return result

            self.stats["misses"] += 1
            return None

    def _semantic_search(self, text: str) -> Optional[Any]:
        """Perform semantic similarity search."""
        if self._next_embedding_idx == 0:
            return None

        try:
            query_embedding = self._get_normalized_embedding(text)

            # Choose search method based on scale
            if self.use_faiss and self._faiss_index is not None:
                return self._faiss_search(query_embedding)
            else:
                return self._numpy_batch_search(query_embedding)

        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return None

    def _numpy_batch_search(self, query: np.ndarray) -> Optional[Any]:
        """Tier 2: Batched numpy similarity search."""
        if self._next_embedding_idx == 0:
            return None

        # Single matrix multiplication for all similarities
        active_embeddings = self._embeddings[:self._next_embedding_idx]
        similarities = active_embeddings @ query

        # Find best match above threshold
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]

        if best_sim >= self.similarity_threshold:
            key = self._embedding_keys[best_idx]
            if key in self._entries:
                entry = self._entries[key]
                if not self._is_expired(entry):
                    entry.access()
                    self.stats["semantic_hits"] += 1
                    logger.debug(f"Semantic hit: similarity={best_sim:.3f}")
                    return entry.response

        return None

    def _faiss_search(self, query: np.ndarray) -> Optional[Any]:
        """Tier 3: FAISS HNSW search."""
        if self._faiss_index is None:
            return None

        query_batch = query.reshape(1, -1).astype(np.float32)
        distances, indices = self._faiss_index.search(query_batch, 1)

        if indices[0][0] >= 0:
            similarity = distances[0][0]  # Inner product = cosine for normalized
            if similarity >= self.similarity_threshold:
                faiss_id = indices[0][0]
                if faiss_id in self._faiss_id_map:
                    key = self._faiss_id_map[faiss_id]
                    if key in self._entries:
                        entry = self._entries[key]
                        if not self._is_expired(entry):
                            entry.access()
                            self.stats["faiss_hits"] += 1
                            logger.debug(f"FAISS hit: similarity={similarity:.3f}")
                            return entry.response

        return None

    def set(
        self,
        key: str,
        response: Any,
        text_representation: str,
    ) -> None:
        """
        Store response in cache.

        Args:
            key: Cache key
            response: Response to cache
            text_representation: Text for semantic matching
        """
        with self._lock:
            # Evict if at capacity
            while len(self._entries) >= self.max_entries:
                self._evict_lru()

            # Create entry
            text_hash = self._hash_text(text_representation)
            embedding_idx = -1

            # Generate and store embedding
            if self.embedding_function and HAS_NUMPY:
                try:
                    embedding = self._get_normalized_embedding(text_representation)
                    embedding_idx = self._add_embedding(key, embedding)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding: {e}")

            entry = CacheEntry(
                key=key,
                response=response,
                text_hash=text_hash,
                embedding_idx=embedding_idx,
            )

            self._entries[key] = entry
            self._exact_cache[text_hash] = key

            # Add to LRU heap
            heapq.heappush(self._lru_heap, (entry.last_accessed, key))

            # Check if we should switch to FAISS
            if (self.use_faiss and
                len(self._entries) >= self.faiss_threshold and
                self._faiss_index is None):
                self._build_faiss_index()

    def _add_embedding(self, key: str, embedding: np.ndarray) -> int:
        """Add embedding to storage."""
        if self._next_embedding_idx >= self.max_entries:
            # Compact embeddings (remove gaps from deleted entries)
            self._compact_embeddings()

        idx = self._next_embedding_idx
        self._embeddings[idx] = embedding

        if idx < len(self._embedding_keys):
            self._embedding_keys[idx] = key
        else:
            self._embedding_keys.append(key)

        self._next_embedding_idx += 1

        # Update FAISS if active
        if self._faiss_index is not None:
            self._faiss_index.add(embedding.reshape(1, -1))
            self._faiss_id_map[self._faiss_index.ntotal - 1] = key

        return idx

    def _compact_embeddings(self):
        """Compact embedding storage by removing gaps."""
        valid_keys = []
        valid_embeddings = []

        for i, key in enumerate(self._embedding_keys[:self._next_embedding_idx]):
            if key in self._entries:
                valid_keys.append(key)
                valid_embeddings.append(self._embeddings[i])

        if valid_embeddings:
            self._embeddings[:len(valid_embeddings)] = np.array(valid_embeddings)
            self._embedding_keys = valid_keys
            self._next_embedding_idx = len(valid_embeddings)

            # Update entry indices
            for i, key in enumerate(valid_keys):
                if key in self._entries:
                    self._entries[key].embedding_idx = i

        # Rebuild FAISS if active
        if self._faiss_index is not None:
            self._build_faiss_index()

        logger.debug(f"Compacted embeddings: {self._next_embedding_idx} entries")

    def _build_faiss_index(self):
        """Build FAISS HNSW index from current embeddings."""
        if not HAS_FAISS or self._next_embedding_idx == 0:
            return

        logger.info(f"Building FAISS index with {self._next_embedding_idx} entries")

        # Create HNSW index (32 = M parameter, controls accuracy/speed tradeoff)
        self._faiss_index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        self._faiss_index.hnsw.efSearch = 64  # Search quality

        # Add all embeddings
        active_embeddings = self._embeddings[:self._next_embedding_idx].copy()
        self._faiss_index.add(active_embeddings)

        # Build ID map
        self._faiss_id_map = {
            i: key for i, key in enumerate(self._embedding_keys[:self._next_embedding_idx])
        }

        logger.info("FAISS index built successfully")

    def _get_normalized_embedding(self, text: str) -> np.ndarray:
        """Get normalized embedding for text."""
        embedding = self.embedding_function(text)

        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        else:
            embedding = embedding.astype(np.float32)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry is expired."""
        if not self.ttl_seconds:
            return False
        return time.time() - entry.created_at > self.ttl_seconds

    def _evict_lru(self):
        """Evict least recently used entry using heap."""
        while self._lru_heap:
            _, key = heapq.heappop(self._lru_heap)

            # Check if entry still exists and timestamp matches
            if key in self._entries:
                self._remove_entry(key)
                return

        # Heap was stale, fall back to linear scan
        if self._entries:
            oldest_key = min(
                self._entries.keys(),
                key=lambda k: self._entries[k].last_accessed
            )
            self._remove_entry(oldest_key)

    def _remove_entry(self, key: str):
        """Remove entry from all storage structures."""
        if key not in self._entries:
            return

        entry = self._entries[key]

        # Remove from exact cache
        if entry.text_hash in self._exact_cache:
            if self._exact_cache[entry.text_hash] == key:
                del self._exact_cache[entry.text_hash]

        # Mark embedding slot as invalid (will be compacted later)
        if entry.embedding_idx >= 0 and entry.embedding_idx < len(self._embedding_keys):
            self._embedding_keys[entry.embedding_idx] = ""

        # Remove entry
        del self._entries[key]

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._entries.clear()
            self._exact_cache.clear()
            self._embedding_keys.clear()
            self._next_embedding_idx = 0
            self._lru_heap.clear()
            self._faiss_index = None
            self._faiss_id_map.clear()

            if HAS_NUMPY:
                self._embeddings = np.zeros(
                    (self.max_entries, self.embedding_dim),
                    dtype=np.float32
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = (
            self.stats["exact_hits"] +
            self.stats["semantic_hits"] +
            self.stats["faiss_hits"]
        )
        hit_rate = total_hits / self.stats["total_lookups"] if self.stats["total_lookups"] > 0 else 0

        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "embeddings_stored": self._next_embedding_idx,
            **self.stats,
            "hit_rate": hit_rate,
            "faiss_enabled": self._faiss_index is not None,
            "faiss_available": HAS_FAISS,
            "numpy_available": HAS_NUMPY,
        }

    def __enter__(self):
        """Context manager entry."""
        self.start_background_cleanup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_background_cleanup()


# Convenience function to create cache with sensible defaults
def create_optimized_cache(
    embedding_function: Optional[Callable] = None,
    max_entries: int = 10000,
    use_faiss_at_scale: bool = True,
) -> OptimizedSemanticCache:
    """
    Create an optimized cache with sensible defaults.

    Args:
        embedding_function: Function to generate embeddings
        max_entries: Maximum entries to store
        use_faiss_at_scale: Auto-enable FAISS at 20K+ entries

    Returns:
        Configured OptimizedSemanticCache instance
    """
    return OptimizedSemanticCache(
        embedding_function=embedding_function,
        max_entries=max_entries,
        use_faiss=use_faiss_at_scale,
        faiss_threshold=20000,
        ttl_seconds=3600,  # 1 hour default TTL
        cleanup_interval=60.0,
    )
