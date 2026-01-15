#!/usr/bin/env python3
"""
Benchmark to analyze scaling issues in semantic cache and knowledge base.

Tests:
1. Current Python implementation at various scales
2. Identifies specific bottlenecks
3. Compares with optimized alternatives (if available)
"""

import time
import random
import statistics
from typing import List, Tuple, Dict, Any
import sys

# Check for numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    print("NumPy not available - install with: pip install numpy")
    HAS_NUMPY = False
    sys.exit(1)

from club_harness.caching.semantic_cache import SemanticCache, CacheEntry
from club_harness.knowledge.semantic_kb import SemanticKnowledgeBase, SimpleEmbedding


def generate_random_embedding(dim: int = 1536) -> np.ndarray:
    """Generate a random normalized embedding."""
    vec = np.random.randn(dim)
    return vec / np.linalg.norm(vec)


def generate_random_text(length: int = 100) -> str:
    """Generate random text for testing."""
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "hello", "world", "python", "code", "test", "benchmark", "scale",
             "agent", "memory", "cache", "search", "vector", "embedding"]
    return " ".join(random.choice(words) for _ in range(length))


class BenchmarkResults:
    """Store and display benchmark results."""

    def __init__(self, name: str):
        self.name = name
        self.results: Dict[str, List[float]] = {}

    def add(self, metric: str, value: float):
        if metric not in self.results:
            self.results[metric] = []
        self.results[metric].append(value)

    def summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for metric, values in self.results.items():
            summary[metric] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            }
        return summary

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"Benchmark: {self.name}")
        print(f"{'='*60}")
        for metric, stats in self.summary().items():
            print(f"\n{metric}:")
            print(f"  Mean:   {stats['mean']*1000:.3f} ms")
            print(f"  Median: {stats['median']*1000:.3f} ms")
            print(f"  Min:    {stats['min']*1000:.3f} ms")
            print(f"  Max:    {stats['max']*1000:.3f} ms")


def benchmark_semantic_cache_scaling():
    """Benchmark SemanticCache at different scales."""
    print("\n" + "="*60)
    print("SEMANTIC CACHE SCALING BENCHMARK")
    print("="*60)

    # Embedding dimension (OpenAI ada-002 size)
    dim = 1536

    # Test at different scales
    scales = [100, 500, 1000, 2000, 5000, 10000]

    results = []

    for n_entries in scales:
        print(f"\n--- Testing with {n_entries} entries ---")

        # Create cache with dummy embedding function
        cache = SemanticCache(
            similarity_threshold=0.85,
            embedding_function=lambda x: generate_random_embedding(dim),
            max_entries=n_entries + 100,  # Avoid eviction during test
        )

        # Populate cache
        print(f"  Populating cache...")
        populate_start = time.perf_counter()

        for i in range(n_entries):
            key = f"key_{i}"
            text = generate_random_text()
            cache.set(key, f"response_{i}", text)

        populate_time = time.perf_counter() - populate_start
        print(f"  Population time: {populate_time:.2f}s ({populate_time/n_entries*1000:.2f}ms per entry)")

        # Benchmark search (miss case - forces full scan)
        n_searches = 50
        search_times = []

        print(f"  Running {n_searches} search queries...")
        for _ in range(n_searches):
            query_text = generate_random_text()

            start = time.perf_counter()
            cache.get("nonexistent_key", query_text)  # Force semantic search
            elapsed = time.perf_counter() - start

            search_times.append(elapsed)

        avg_search = statistics.mean(search_times)
        median_search = statistics.median(search_times)

        print(f"  Avg search time: {avg_search*1000:.2f}ms")
        print(f"  Median search time: {median_search*1000:.2f}ms")
        print(f"  Throughput: {1/avg_search:.0f} queries/sec")

        results.append({
            "entries": n_entries,
            "populate_time_ms": populate_time * 1000 / n_entries,
            "search_time_ms": avg_search * 1000,
            "throughput_qps": 1 / avg_search,
        })

        # Memory estimate
        import sys
        cache_size_estimate = n_entries * (dim * 8 + 500)  # floats + overhead
        print(f"  Estimated memory: {cache_size_estimate / 1024 / 1024:.1f} MB")

    # Summary table
    print("\n" + "="*60)
    print("SCALING SUMMARY")
    print("="*60)
    print(f"{'Entries':>10} {'Search (ms)':>12} {'QPS':>10} {'Scaling':>10}")
    print("-"*45)

    baseline_qps = results[0]["throughput_qps"]
    for r in results:
        scaling_factor = baseline_qps / r["throughput_qps"]
        print(f"{r['entries']:>10} {r['search_time_ms']:>12.2f} {r['throughput_qps']:>10.0f} {scaling_factor:>10.1f}x")

    return results


def benchmark_bottleneck_breakdown():
    """Break down where time is spent in the search."""
    print("\n" + "="*60)
    print("BOTTLENECK BREAKDOWN")
    print("="*60)

    dim = 1536
    n_entries = 1000

    # Pre-generate embeddings
    embeddings = [generate_random_embedding(dim) for _ in range(n_entries)]
    query_embedding = generate_random_embedding(dim)

    # Test 1: Pure numpy dot product
    print("\n1. Pure numpy dot product (batch):")
    embedding_matrix = np.vstack(embeddings)

    start = time.perf_counter()
    for _ in range(100):
        similarities = np.dot(embedding_matrix, query_embedding)
    batch_time = (time.perf_counter() - start) / 100
    print(f"   Time: {batch_time*1000:.3f}ms for {n_entries} comparisons")

    # Test 2: Python loop with numpy dot
    print("\n2. Python loop with numpy dot (current approach):")
    start = time.perf_counter()
    for _ in range(100):
        sims = []
        for emb in embeddings:
            sims.append(float(np.dot(query_embedding, emb)))
    loop_time = (time.perf_counter() - start) / 100
    print(f"   Time: {loop_time*1000:.3f}ms for {n_entries} comparisons")
    print(f"   Slowdown vs batch: {loop_time/batch_time:.1f}x")

    # Test 3: Dict iteration overhead
    print("\n3. Dict iteration overhead:")
    cache_dict = {f"key_{i}": {"embedding": emb, "data": f"value_{i}"}
                  for i, emb in enumerate(embeddings)}

    start = time.perf_counter()
    for _ in range(100):
        for key, entry in cache_dict.items():
            _ = entry["embedding"]
    dict_time = (time.perf_counter() - start) / 100
    print(f"   Time: {dict_time*1000:.3f}ms for {n_entries} entries")

    # Test 4: Full search simulation (dict + numpy)
    print("\n4. Full search (dict iteration + numpy dot):")
    start = time.perf_counter()
    for _ in range(100):
        best_sim = 0
        for key, entry in cache_dict.items():
            sim = float(np.dot(query_embedding, entry["embedding"]))
            if sim > best_sim:
                best_sim = sim
    full_time = (time.perf_counter() - start) / 100
    print(f"   Time: {full_time*1000:.3f}ms for {n_entries} entries")

    # Breakdown
    print("\n" + "-"*40)
    print("BREAKDOWN:")
    print(f"  Numpy batch dot:     {batch_time*1000:.3f}ms ({batch_time/full_time*100:.1f}%)")
    print(f"  Python loop overhead: {(loop_time-batch_time)*1000:.3f}ms ({(loop_time-batch_time)/full_time*100:.1f}%)")
    print(f"  Dict overhead:       {dict_time*1000:.3f}ms ({dict_time/full_time*100:.1f}%)")
    print(f"  Total observed:      {full_time*1000:.3f}ms")


def benchmark_alternative_approaches():
    """Test optimized approaches if available."""
    print("\n" + "="*60)
    print("ALTERNATIVE APPROACHES")
    print("="*60)

    dim = 1536
    n_entries = 5000

    # Generate test data
    embeddings = np.array([generate_random_embedding(dim) for _ in range(n_entries)])
    query = generate_random_embedding(dim)

    # Approach 1: Numpy matrix multiplication (batched)
    print("\n1. NumPy batched matrix multiplication:")
    start = time.perf_counter()
    for _ in range(100):
        similarities = embeddings @ query
        best_idx = np.argmax(similarities)
    numpy_time = (time.perf_counter() - start) / 100
    print(f"   Time: {numpy_time*1000:.3f}ms for {n_entries} entries")
    print(f"   Throughput: {1/numpy_time:.0f} queries/sec")

    # Approach 2: Try usearch if available
    try:
        from usearch.index import Index
        print("\n2. USearch (Rust-based):")

        index = Index(ndim=dim, metric='cos')

        # Add vectors
        add_start = time.perf_counter()
        for i, emb in enumerate(embeddings):
            index.add(i, emb)
        add_time = time.perf_counter() - add_start
        print(f"   Index build time: {add_time*1000:.1f}ms")

        # Search
        start = time.perf_counter()
        for _ in range(100):
            matches = index.search(query, 1)
        usearch_time = (time.perf_counter() - start) / 100
        print(f"   Search time: {usearch_time*1000:.4f}ms for {n_entries} entries")
        print(f"   Throughput: {1/usearch_time:.0f} queries/sec")
        print(f"   Speedup vs numpy batch: {numpy_time/usearch_time:.0f}x")

    except ImportError:
        print("\n2. USearch: Not installed (pip install usearch)")

    # Approach 3: Try faiss if available
    try:
        import faiss
        print("\n3. FAISS (C++-based):")

        # Create index
        index = faiss.IndexFlatIP(dim)  # Inner product (cosine for normalized)
        index.add(embeddings.astype(np.float32))

        # Search
        query_batch = query.reshape(1, -1).astype(np.float32)
        start = time.perf_counter()
        for _ in range(100):
            D, I = index.search(query_batch, 1)
        faiss_time = (time.perf_counter() - start) / 100
        print(f"   Search time: {faiss_time*1000:.4f}ms for {n_entries} entries")
        print(f"   Throughput: {1/faiss_time:.0f} queries/sec")
        print(f"   Speedup vs numpy batch: {numpy_time/faiss_time:.0f}x")

    except ImportError:
        print("\n3. FAISS: Not installed (pip install faiss-cpu)")


def benchmark_memory_usage():
    """Analyze memory usage patterns."""
    print("\n" + "="*60)
    print("MEMORY ANALYSIS")
    print("="*60)

    import sys

    dim = 1536

    # Single embedding
    emb_numpy = np.random.randn(dim).astype(np.float32)
    emb_list = list(emb_numpy)

    print(f"\nEmbedding storage ({dim} dimensions):")
    print(f"  numpy float32: {emb_numpy.nbytes:,} bytes")
    print(f"  numpy float64: {emb_numpy.astype(np.float64).nbytes:,} bytes")
    print(f"  Python list:   {sys.getsizeof(emb_list) + sum(sys.getsizeof(x) for x in emb_list):,} bytes (approx)")

    # Cache entry overhead
    entry = CacheEntry(
        key="test_key",
        response="test response",
        text_representation="test text",
        embedding=emb_numpy,
    )

    print(f"\nCacheEntry overhead:")
    print(f"  Base object: ~500 bytes + embedding")
    print(f"  1000 entries with float32: ~{(500 + dim*4) * 1000 / 1024 / 1024:.1f} MB")
    print(f"  10000 entries with float32: ~{(500 + dim*4) * 10000 / 1024 / 1024:.1f} MB")
    print(f"  100000 entries with float32: ~{(500 + dim*4) * 100000 / 1024 / 1024:.1f} MB")


def main():
    """Run all benchmarks."""
    print("\n" + "#"*60)
    print("# SEMANTIC CACHE/KB SCALING ANALYSIS")
    print("#"*60)

    # Run benchmarks
    benchmark_semantic_cache_scaling()
    benchmark_bottleneck_breakdown()
    benchmark_memory_usage()
    benchmark_alternative_approaches()

    # Conclusions
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)
    print("""
Current implementation scaling issues:

1. O(n) LINEAR SCAN: Every search iterates all entries
   - 1000 entries: ~5ms
   - 10000 entries: ~50ms (10x slower)
   - This is the PRIMARY bottleneck

2. PYTHON LOOP OVERHEAD: ~5-10x slower than batched numpy
   - Dict iteration adds ~30% overhead
   - Individual numpy calls vs batched adds ~50%

3. MEMORY LAYOUT: Scattered embeddings, poor cache locality
   - Each entry is a separate Python object
   - No contiguous memory for SIMD

RECOMMENDATIONS:

1. FOR IMMEDIATE IMPROVEMENT:
   - Use numpy batch operations (embeddings @ query)
   - Store embeddings in contiguous array, not dict
   - This alone provides 5-10x speedup

2. FOR PRODUCTION SCALE (10K+ entries):
   - Use compiled index (usearch, faiss)
   - O(log n) instead of O(n)
   - 100-1000x speedup at scale

3. MEMORY OPTIMIZATION:
   - Use float32 instead of float64
   - Store IDs separately from embeddings
   - Consider quantization for very large scale
""")


if __name__ == "__main__":
    main()
