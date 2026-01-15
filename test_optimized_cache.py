#!/usr/bin/env python3
"""
Test and benchmark the optimized semantic cache.

Compares:
- Original SemanticCache
- OptimizedSemanticCache (new tiered implementation)
"""

import time
import random
import statistics
import sys

try:
    import numpy as np
except ImportError:
    print("NumPy required: pip install numpy")
    sys.exit(1)

from club_harness.caching.semantic_cache import SemanticCache
from club_harness.caching.optimized_cache import OptimizedSemanticCache


def generate_embedding(dim: int = 1536) -> np.ndarray:
    """Generate random normalized embedding."""
    vec = np.random.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def generate_text(length: int = 50) -> str:
    """Generate random text."""
    words = ["test", "query", "cache", "semantic", "search", "vector", "embedding",
             "fast", "scale", "benchmark", "python", "numpy", "faiss"]
    return " ".join(random.choice(words) for _ in range(length))


def benchmark_cache(cache, name: str, n_entries: int, n_queries: int = 100):
    """Benchmark a cache implementation."""
    print(f"\n{'='*50}")
    print(f"Benchmarking: {name}")
    print(f"Entries: {n_entries}, Queries: {n_queries}")
    print(f"{'='*50}")

    dim = 1536

    # Populate cache
    print("Populating cache...")
    populate_start = time.perf_counter()

    for i in range(n_entries):
        key = f"key_{i}"
        text = generate_text()
        cache.set(key, f"response_{i}", text)

    populate_time = time.perf_counter() - populate_start
    print(f"  Population time: {populate_time:.2f}s ({populate_time/n_entries*1000:.2f}ms/entry)")

    # Benchmark exact hits
    print("\nExact match queries (should be O(1))...")
    exact_times = []
    for i in range(min(n_queries, n_entries)):
        key = f"key_{i}"
        start = time.perf_counter()
        result = cache.get(key)
        exact_times.append(time.perf_counter() - start)

    print(f"  Avg: {statistics.mean(exact_times)*1000:.4f}ms")
    print(f"  Median: {statistics.median(exact_times)*1000:.4f}ms")

    # Benchmark semantic search (cache misses that trigger similarity search)
    print("\nSemantic search queries (cache miss + similarity scan)...")
    semantic_times = []
    for _ in range(n_queries):
        text = generate_text()  # New text, will miss exact cache
        start = time.perf_counter()
        result = cache.get("nonexistent_key", text)
        semantic_times.append(time.perf_counter() - start)

    avg_semantic = statistics.mean(semantic_times)
    print(f"  Avg: {avg_semantic*1000:.3f}ms")
    print(f"  Median: {statistics.median(semantic_times)*1000:.3f}ms")
    print(f"  Throughput: {1/avg_semantic:.0f} queries/sec")

    # Get stats
    stats = cache.get_stats()
    print(f"\nCache stats: {stats}")

    return {
        "populate_ms": populate_time * 1000 / n_entries,
        "exact_ms": statistics.mean(exact_times) * 1000,
        "semantic_ms": avg_semantic * 1000,
        "qps": 1 / avg_semantic,
    }


def test_correctness():
    """Test that optimized cache works correctly."""
    print("\n" + "="*50)
    print("CORRECTNESS TESTS")
    print("="*50)

    dim = 256  # Smaller for testing
    cache = OptimizedSemanticCache(
        embedding_function=lambda x: generate_embedding(dim),
        embedding_dim=dim,
        max_entries=100,
        similarity_threshold=0.5,
    )

    # Test 1: Exact match
    print("\n1. Exact match test...")
    cache.set("key1", "response1", "test query one")
    result = cache.get("key1")
    assert result == "response1", f"Expected 'response1', got {result}"
    print("   PASSED")

    # Test 2: Text hash match
    print("2. Text hash match test...")
    cache.set("key2", "response2", "identical query text")
    result = cache.get("different_key", "identical query text")
    assert result == "response2", f"Expected 'response2', got {result}"
    print("   PASSED")

    # Test 3: Multiple entries
    print("3. Multiple entries test...")
    for i in range(50):
        cache.set(f"multi_{i}", f"resp_{i}", f"query number {i}")

    for i in range(50):
        result = cache.get(f"multi_{i}")
        assert result == f"resp_{i}", f"Expected 'resp_{i}', got {result}"
    print("   PASSED")

    # Test 4: LRU eviction
    print("4. LRU eviction test...")
    small_cache = OptimizedSemanticCache(
        embedding_function=lambda x: generate_embedding(dim),
        embedding_dim=dim,
        max_entries=10,
    )

    for i in range(15):
        small_cache.set(f"evict_{i}", f"resp_{i}", f"query {i}")

    assert len(small_cache._entries) <= 10, f"Expected <=10 entries, got {len(small_cache._entries)}"
    print("   PASSED")

    # Test 5: Stats tracking
    print("5. Stats tracking test...")
    stats = cache.get_stats()
    assert "exact_hits" in stats
    assert "semantic_hits" in stats
    assert "entries" in stats
    print("   PASSED")

    print("\n✓ All correctness tests passed!")


def compare_implementations():
    """Compare original vs optimized cache."""
    print("\n" + "#"*60)
    print("# IMPLEMENTATION COMPARISON")
    print("#"*60)

    dim = 1536
    embedding_fn = lambda x: generate_embedding(dim)

    results = {}

    for n_entries in [1000, 5000, 10000]:
        print(f"\n\n{'*'*60}")
        print(f"* SCALE: {n_entries} entries")
        print(f"{'*'*60}")

        # Original cache
        original = SemanticCache(
            similarity_threshold=0.85,
            embedding_function=embedding_fn,
            max_entries=n_entries + 100,
        )
        orig_results = benchmark_cache(original, "Original SemanticCache", n_entries)

        # Optimized cache
        optimized = OptimizedSemanticCache(
            similarity_threshold=0.85,
            embedding_function=embedding_fn,
            embedding_dim=dim,
            max_entries=n_entries + 100,
            use_faiss=False,  # Test numpy-only first
        )
        opt_results = benchmark_cache(optimized, "Optimized (NumPy batched)", n_entries)

        # Calculate speedup
        speedup = orig_results["semantic_ms"] / opt_results["semantic_ms"]
        print(f"\n>>> SPEEDUP: {speedup:.1f}x faster semantic search")

        results[n_entries] = {
            "original": orig_results,
            "optimized": opt_results,
            "speedup": speedup,
        }

    # Summary table
    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Entries':>10} {'Original (ms)':>15} {'Optimized (ms)':>15} {'Speedup':>10}")
    print("-"*55)

    for n, data in results.items():
        print(f"{n:>10} {data['original']['semantic_ms']:>15.2f} {data['optimized']['semantic_ms']:>15.3f} {data['speedup']:>10.1f}x")


def test_faiss_scaling():
    """Test FAISS backend at larger scales."""
    print("\n" + "#"*60)
    print("# FAISS SCALING TEST")
    print("#"*60)

    try:
        import faiss
        print("FAISS available!")
    except ImportError:
        print("FAISS not installed. Skipping FAISS tests.")
        return

    dim = 1536
    embedding_fn = lambda x: generate_embedding(dim)

    for n_entries in [10000, 25000]:
        print(f"\n--- {n_entries} entries ---")

        # NumPy only
        numpy_cache = OptimizedSemanticCache(
            similarity_threshold=0.85,
            embedding_function=embedding_fn,
            embedding_dim=dim,
            max_entries=n_entries + 100,
            use_faiss=False,
        )
        numpy_results = benchmark_cache(numpy_cache, "NumPy batched", n_entries, n_queries=50)

        # With FAISS
        faiss_cache = OptimizedSemanticCache(
            similarity_threshold=0.85,
            embedding_function=embedding_fn,
            embedding_dim=dim,
            max_entries=n_entries + 100,
            use_faiss=True,
            faiss_threshold=5000,  # Enable FAISS early for testing
        )
        faiss_results = benchmark_cache(faiss_cache, "FAISS HNSW", n_entries, n_queries=50)

        speedup = numpy_results["semantic_ms"] / faiss_results["semantic_ms"]
        print(f"\n>>> FAISS speedup at {n_entries} entries: {speedup:.1f}x")


if __name__ == "__main__":
    # Run tests
    test_correctness()

    # Compare implementations
    compare_implementations()

    # Test FAISS scaling
    test_faiss_scaling()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
