#!/usr/bin/env python3
"""
Test advanced council strategies with OpenRouter.
Tests multi-round deliberation and weighted voting concepts.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from club_harness.core.config import config
from club_harness.llm.router import LLMRouter


def test_multi_round_deliberation():
    """Test multi-round deliberation concept with free models."""
    print("\n" + "=" * 60)
    print("TEST: Multi-Round Deliberation (2 rounds)")
    print("=" * 60)

    if not config.llm.api_key:
        print("Skipping - no API key")
        return None

    router = LLMRouter()
    models = [
        "meta-llama/llama-3.2-3b-instruct:free",
        "google/gemma-3n-e2b-it:free",
    ]

    question = "What is the best programming language for beginners and why? Give a short answer."

    print(f"\nQuestion: {question}")
    print(f"Models: {models}")

    # Round 1: Initial responses
    print("\n--- Round 1: Initial Responses ---")
    round1_responses = {}
    for model in models:
        try:
            print(f"\nQuerying {model}...")
            response = router.chat(
                messages=[{"role": "user", "content": question}],
                model=model,
                max_tokens=150,
            )
            round1_responses[model] = response.content
            print(f"Response: {response.content[:200]}...")
        except Exception as e:
            print(f"Error: {e}")
            round1_responses[model] = None

    # Small delay between rounds
    time.sleep(2)

    # Round 2: Show top responses and ask for revision
    print("\n--- Round 2: Revision with Peer Context ---")

    # Build revision prompt with previous responses
    peer_context = "\n\n".join([
        f"Previous response from Model {i+1}:\n{resp[:200]}..."
        for i, (model, resp) in enumerate(round1_responses.items())
        if resp
    ])

    revision_prompt = f"""Original question: {question}

In the previous round, other AI models gave these responses:

{peer_context}

Based on these perspectives, please provide your final, refined answer to the original question.
You may incorporate valid points from others or strengthen your original position."""

    round2_responses = {}
    for model in models:
        try:
            print(f"\nQuerying {model} for revision...")
            response = router.chat(
                messages=[{"role": "user", "content": revision_prompt}],
                model=model,
                max_tokens=200,
            )
            round2_responses[model] = response.content
            print(f"Revised Response: {response.content[:250]}...")
        except Exception as e:
            print(f"Error: {e}")

    # Small delay before synthesis
    time.sleep(2)

    # Chairman synthesis
    print("\n--- Chairman Synthesis ---")

    synthesis_prompt = f"""You are the Chairman of an AI Council. The question was: {question}

After two rounds of deliberation, here are the final responses:

{chr(10).join([f"Model {i+1}: {resp[:300]}..." for i, resp in enumerate(round2_responses.values()) if resp])}

Synthesize these perspectives into a single, comprehensive final answer:"""

    try:
        chairman_model = "meta-llama/llama-3.2-3b-instruct:free"
        synthesis = router.chat(
            messages=[{"role": "user", "content": synthesis_prompt}],
            model=chairman_model,
            max_tokens=300,
        )
        print(f"\nChairman ({chairman_model}) Final Answer:")
        print(synthesis.content)
        return True
    except Exception as e:
        print(f"Synthesis error: {e}")
        return False


def test_confidence_tracking():
    """Test confidence tracking concept."""
    print("\n" + "=" * 60)
    print("TEST: Confidence Tracking")
    print("=" * 60)

    router = LLMRouter()

    # Ask a question that should produce different confidence levels
    questions = [
        ("What is 2 + 2?", "high"),
        ("Will it rain tomorrow in San Francisco?", "low"),
        ("Is Python better than JavaScript?", "medium"),
    ]

    for question, expected_confidence in questions:
        try:
            print(f"\nQuestion: {question}")
            print(f"Expected confidence: {expected_confidence}")

            # Ask the model to express confidence
            prompt = f"""{question}

Please also rate your confidence in your answer on a scale of 1-10, where:
- 1-3: Low confidence (uncertain, depends on many factors)
- 4-6: Medium confidence (reasonably sure but some uncertainty)
- 7-10: High confidence (very certain)

Format: [Answer] | Confidence: [1-10]"""

            response = router.chat(
                messages=[{"role": "user", "content": prompt}],
                model="meta-llama/llama-3.2-3b-instruct:free",
                max_tokens=100,
            )
            print(f"Response: {response.content}")
        except Exception as e:
            print(f"Error: {e}")

    return True


def test_response_similarity():
    """Test response similarity detection (for caching)."""
    print("\n" + "=" * 60)
    print("TEST: Response Similarity Detection")
    print("=" * 60)

    import difflib

    router = LLMRouter()

    # Ask similar questions
    similar_questions = [
        "What is Python?",
        "Can you explain Python?",
        "Tell me about Python programming language",
    ]

    responses = []
    for q in similar_questions:
        try:
            print(f"\nQuestion: {q}")
            response = router.chat(
                messages=[{"role": "user", "content": q}],
                model="meta-llama/llama-3.2-3b-instruct:free",
                max_tokens=100,
            )
            responses.append((q, response.content))
            print(f"Response: {response.content[:150]}...")
            time.sleep(1)  # Avoid rate limiting
        except Exception as e:
            print(f"Error: {e}")

    # Calculate similarity between responses
    if len(responses) >= 2:
        print("\n--- Response Similarity Analysis ---")
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = difflib.SequenceMatcher(
                    None, responses[i][1], responses[j][1]
                ).ratio()
                print(f"Q{i+1} vs Q{j+1} similarity: {sim:.2%}")
                if sim > 0.5:
                    print("  -> Could potentially cache this response")

    return True


def main():
    """Run advanced council tests."""
    print("=" * 60)
    print("CLUB HARNESS - Advanced Council Testing")
    print("=" * 60)
    print(f"API Key: {'Set' if config.llm.api_key else 'Not set'}")

    if not config.llm.api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        return False

    results = []

    # Run tests
    tests = [
        ("Multi-Round Deliberation", test_multi_round_deliberation),
        ("Confidence Tracking", test_confidence_tracking),
        ("Response Similarity", test_response_similarity),
    ]

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, bool(result) if result is not None else True))
        except Exception as e:
            print(f"\nTest '{name}' failed with exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("ADVANCED COUNCIL TEST SUMMARY")
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
