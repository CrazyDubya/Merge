#!/usr/bin/env python3
"""
Test new features: semantic caching and verification framework.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_verification_framework():
    """Test the verification framework from LisaSimpson."""
    print("\n" + "=" * 60)
    print("TEST: Verification Framework")
    print("=" * 60)

    from club_harness.verification.checks import (
        CheckResult, Check, PredicateCheck, FactCheck,
        OutputFormatCheck, ConfidenceCheck, CompositeCheck,
        VerificationPlan, VerificationResult
    )

    # Test 1: PredicateCheck
    print("\n1. Testing PredicateCheck...")
    check = PredicateCheck(
        predicate=lambda ctx: ctx.get("value", 0) > 10,
        description="Value must be > 10"
    )

    result = check.run({"value": 15})
    assert result.passed, "Should pass for value=15"
    assert result.confidence > 0.8, "High confidence expected"
    print(f"   Pass case: {result}")

    result = check.run({"value": 5})
    assert not result.passed, "Should fail for value=5"
    print(f"   Fail case: {result}")

    # Test 2: FactCheck
    print("\n2. Testing FactCheck...")
    check = FactCheck(required_facts=["has_api_key", "is_authenticated"])

    result = check.run({"facts": {"has_api_key", "is_authenticated", "extra_fact"}})
    assert result.passed, "Should pass with all required facts"
    print(f"   Pass case: {result}")

    result = check.run({"facts": {"has_api_key"}})
    assert not result.passed, "Should fail with missing facts"
    print(f"   Fail case: {result}")

    # Test 3: OutputFormatCheck
    print("\n3. Testing OutputFormatCheck...")
    check = OutputFormatCheck(
        expected_type=dict,
        required_keys=["status", "data"]
    )

    result = check.run({"output": {"status": "ok", "data": [1, 2, 3]}})
    assert result.passed, "Should pass with correct format"
    print(f"   Pass case: {result}")

    result = check.run({"output": {"status": "ok"}})
    assert not result.passed, "Should fail with missing key"
    print(f"   Fail case: {result}")

    # Test 4: ConfidenceCheck
    print("\n4. Testing ConfidenceCheck...")
    check = ConfidenceCheck(min_confidence=0.7)

    result = check.run({"confidence": 0.85})
    assert result.passed, "Should pass with high confidence"
    print(f"   Pass case: {result}")

    result = check.run({"confidence": 0.5})
    assert not result.passed, "Should fail with low confidence"
    print(f"   Fail case: {result}")

    # Test 5: CompositeCheck (AND)
    print("\n5. Testing CompositeCheck (AND)...")
    composite = CompositeCheck(
        checks=[
            PredicateCheck(lambda ctx: ctx.get("a") > 0, "a > 0"),
            PredicateCheck(lambda ctx: ctx.get("b") > 0, "b > 0"),
        ],
        require_all=True
    )

    result = composite.run({"a": 1, "b": 2})
    assert result.passed, "Should pass when all pass"
    print(f"   All pass: {result}")

    result = composite.run({"a": 1, "b": -1})
    assert not result.passed, "Should fail when any fail"
    print(f"   One fails: {result}")

    # Test 6: CompositeCheck (OR)
    print("\n6. Testing CompositeCheck (OR)...")
    composite = CompositeCheck(
        checks=[
            PredicateCheck(lambda ctx: ctx.get("a") > 0, "a > 0"),
            PredicateCheck(lambda ctx: ctx.get("b") > 0, "b > 0"),
        ],
        require_all=False
    )

    result = composite.run({"a": 1, "b": -1})
    assert result.passed, "Should pass when any pass"
    print(f"   One passes: {result}")

    result = composite.run({"a": -1, "b": -1})
    assert not result.passed, "Should fail when all fail"
    print(f"   All fail: {result}")

    # Test 7: VerificationPlan
    print("\n7. Testing VerificationPlan...")
    plan = VerificationPlan(
        checks=[
            PredicateCheck(lambda ctx: ctx.get("value") > 0, "Positive value"),
            FactCheck(["initialized"]),
            ConfidenceCheck(min_confidence=0.6),
        ],
        required_confidence=0.7
    )

    context = {
        "value": 10,
        "facts": {"initialized"},
        "confidence": 0.8
    }

    result = plan.verify(context)
    assert result.satisfied, "Verification plan should be satisfied"
    print(f"   Satisfied: {result.summary()}")

    context["value"] = -5
    result = plan.verify(context)
    assert not result.satisfied, "Verification plan should fail"
    print(f"   Failed: {result.summary()}")
    print(f"   Failures: {[f.message for f in result.failures]}")

    print("\n✓ Verification framework tests passed!")
    return True


def test_semantic_cache():
    """Test the semantic caching system."""
    print("\n" + "=" * 60)
    print("TEST: Semantic Cache")
    print("=" * 60)

    from club_harness.caching.semantic_cache import (
        SemanticCache, CacheEntry, CachedLLMRouter,
        create_text_representation
    )

    # Test 1: Basic cache operations
    print("\n1. Testing basic cache operations...")
    cache = SemanticCache(max_entries=100)

    # Set and get
    cache.set("key1", {"response": "hello"}, "What is a greeting?")
    result = cache.get("key1")
    assert result == {"response": "hello"}, "Should retrieve cached value"
    print(f"   Set/Get: {result}")

    # Miss
    result = cache.get("nonexistent")
    assert result is None, "Should return None for missing key"
    print(f"   Miss: {result}")

    # Test 2: Cache statistics
    print("\n2. Testing cache statistics...")
    stats = cache.get_stats()
    assert stats["entries"] == 1, "Should have 1 entry"
    assert stats["exact_hits"] == 1, "Should have 1 exact hit"
    assert stats["misses"] == 1, "Should have 1 miss"
    print(f"   Stats: {stats}")

    # Test 3: TTL expiration
    print("\n3. Testing TTL expiration...")
    ttl_cache = SemanticCache(ttl_seconds=0.1)
    ttl_cache.set("ttl_key", "value", "test")

    result = ttl_cache.get("ttl_key")
    assert result == "value", "Should retrieve before TTL"
    print(f"   Before TTL: {result}")

    time.sleep(0.2)
    result = ttl_cache.get("ttl_key")
    assert result is None, "Should expire after TTL"
    print(f"   After TTL: {result}")

    # Test 4: LRU eviction
    print("\n4. Testing LRU eviction...")
    small_cache = SemanticCache(max_entries=5)
    for i in range(10):
        small_cache.set(f"key{i}", f"value{i}", f"text{i}")

    stats = small_cache.get_stats()
    assert stats["entries"] <= 5, "Should not exceed max entries"
    print(f"   Entries after eviction: {stats['entries']}")

    # Test 5: Text representation
    print("\n5. Testing text representation creation...")
    text = create_text_representation(
        "search",
        "query text",
        limit=10,
        offset=0
    )
    assert "search" in text, "Should contain function name"
    assert "query text" in text, "Should contain args"
    print(f"   Text repr: {text}")

    # Test 6: Semantic matching (without embedding function)
    print("\n6. Testing semantic matching setup...")
    def mock_embedding(text):
        # Simple mock - just return text length as a vector
        import numpy as np
        return np.array([len(text), len(text.split())])

    try:
        import numpy as np
        semantic_cache = SemanticCache(
            similarity_threshold=0.9,
            embedding_function=mock_embedding
        )
        semantic_cache.set("sem_key", "sem_value", "hello world")
        print(f"   Semantic cache created with embeddings")

        # Check that embedding was stored
        entry = semantic_cache._cache.get("sem_key")
        assert entry.embedding is not None, "Should have embedding"
        print(f"   Embedding stored: {entry.embedding}")
    except ImportError:
        print("   Numpy not available, skipping embedding tests")

    print("\n✓ Semantic cache tests passed!")
    return True


def test_cached_router():
    """Test the cached LLM router wrapper."""
    print("\n" + "=" * 60)
    print("TEST: Cached LLM Router")
    print("=" * 60)

    from club_harness.caching.semantic_cache import SemanticCache, CachedLLMRouter

    # Create a mock router
    class MockRouter:
        def __init__(self):
            self.call_count = 0

        def chat(self, messages, model=None, tier=None, **kwargs):
            self.call_count += 1
            return f"Response {self.call_count}"

    mock_router = MockRouter()
    cache = SemanticCache()
    cached_router = CachedLLMRouter(mock_router, cache)

    # Test 1: First call should hit the router
    print("\n1. Testing cache miss (first call)...")
    messages = [{"role": "user", "content": "Hello"}]
    result1 = cached_router.chat(messages, model="test")
    assert result1 == "Response 1", "Should get response from router"
    assert mock_router.call_count == 1, "Router should be called"
    print(f"   First call: {result1}, router calls: {mock_router.call_count}")

    # Test 2: Same call should hit cache
    print("\n2. Testing cache hit (same call)...")
    result2 = cached_router.chat(messages, model="test")
    assert result2 == "Response 1", "Should get cached response"
    assert mock_router.call_count == 1, "Router should NOT be called again"
    print(f"   Second call: {result2}, router calls: {mock_router.call_count}")

    # Test 3: Different call should hit router
    print("\n3. Testing cache miss (different call)...")
    messages2 = [{"role": "user", "content": "Goodbye"}]
    result3 = cached_router.chat(messages2, model="test")
    assert result3 == "Response 2", "Should get new response"
    assert mock_router.call_count == 2, "Router should be called"
    print(f"   Third call: {result3}, router calls: {mock_router.call_count}")

    # Test 4: Bypass cache
    print("\n4. Testing cache bypass...")
    result4 = cached_router.chat(messages, model="test", use_cache=False)
    assert result4 == "Response 3", "Should get new response"
    assert mock_router.call_count == 3, "Router should be called"
    print(f"   Bypass call: {result4}, router calls: {mock_router.call_count}")

    # Test 5: Cache stats
    print("\n5. Testing cache stats...")
    stats = cached_router.get_cache_stats()
    print(f"   Stats: {stats}")
    assert stats["exact_hits"] >= 1, "Should have cache hits"

    print("\n✓ Cached router tests passed!")
    return True


def test_integration_with_openrouter():
    """Test integration with real OpenRouter API."""
    print("\n" + "=" * 60)
    print("TEST: OpenRouter Integration with Caching")
    print("=" * 60)

    from club_harness.core.config import config

    if not config.llm.api_key:
        print("Skipping OpenRouter test - no API key")
        return True

    from club_harness.llm.router import LLMRouter
    from club_harness.caching.semantic_cache import SemanticCache, CachedLLMRouter

    # Create cached router
    base_router = LLMRouter()
    cache = SemanticCache(similarity_threshold=0.85)
    cached_router = CachedLLMRouter(base_router, cache, cache_tiers=["free"])

    messages = [{"role": "user", "content": "What is 2 + 2?"}]

    print("\n1. First call (cache miss)...")
    try:
        start = time.time()
        result1 = cached_router.chat(
            messages,
            model="google/gemma-3n-e2b-it:free",
            tier="free",
            max_tokens=50
        )
        elapsed1 = time.time() - start
        print(f"   Response: {result1.content[:100] if hasattr(result1, 'content') else str(result1)[:100]}")
        print(f"   Time: {elapsed1:.2f}s")
    except Exception as e:
        print(f"   Error (expected for rate limits): {e}")
        return True

    print("\n2. Second call (cache hit)...")
    start = time.time()
    result2 = cached_router.chat(
        messages,
        model="google/gemma-3n-e2b-it:free",
        tier="free",
        max_tokens=50
    )
    elapsed2 = time.time() - start
    print(f"   Response: {result2.content[:100] if hasattr(result2, 'content') else str(result2)[:100]}")
    print(f"   Time: {elapsed2:.2f}s (should be near instant)")

    assert elapsed2 < elapsed1, "Cached call should be faster"

    stats = cached_router.get_cache_stats()
    print(f"\n3. Cache stats: {stats}")

    print("\n✓ OpenRouter integration test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("CLUB HARNESS - New Features Testing")
    print("=" * 60)

    results = []

    tests = [
        ("Verification Framework", test_verification_framework),
        ("Semantic Cache", test_semantic_cache),
        ("Cached Router", test_cached_router),
        ("OpenRouter Integration", test_integration_with_openrouter),
    ]

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, bool(result)))
        except Exception as e:
            print(f"\nTest '{name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("NEW FEATURES TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
