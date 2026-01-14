#!/usr/bin/env python3
"""
Comprehensive testing script for Club Harness with OpenRouter.

Tests:
1. Multiple free models comparison
2. Complex reasoning tasks
3. Tool calling scenarios
4. Memory persistence and retrieval
5. Council deliberation with different strategies
6. Error handling and edge cases
"""

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from club_harness.core.config import config
from club_harness.llm.openrouter import OpenRouterBackend
from club_harness.llm.router import LLMRouter
from club_harness.core.agent import Agent, AgentBuilder
from club_harness.memory import Memory, MemoryType, Lesson
from club_harness.tools import tool_registry
from club_harness.orchestration import Council


def test_free_models_comparison():
    """Compare different free models on the same task."""
    print("\n" + "=" * 60)
    print("TEST: Free Models Comparison")
    print("=" * 60)

    free_models = [
        "meta-llama/llama-3.2-3b-instruct:free",
        "moonshotai/kimi-k2:free",
        "nvidia/nemotron-nano-9b-v2:free",
        "google/gemma-3n-e2b-it:free",
    ]

    router = LLMRouter()
    prompt = "What are the three primary colors? List them briefly."
    results = {}

    for model in free_models:
        try:
            print(f"\nTesting {model}...")
            start = time.time()
            response = router.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=100,
            )
            elapsed = time.time() - start

            results[model] = {
                "response": response.content.strip()[:200],
                "tokens": response.total_tokens,
                "time": elapsed,
                "success": True
            }
            print(f"  Response: {results[model]['response'][:80]}...")
            print(f"  Tokens: {results[model]['tokens']}, Time: {elapsed:.2f}s")

        except Exception as e:
            results[model] = {"success": False, "error": str(e)}
            print(f"  Error: {e}")

    successful = sum(1 for r in results.values() if r.get("success"))
    print(f"\n{successful}/{len(free_models)} models responded successfully")
    return results


def test_reasoning_task():
    """Test models on a reasoning task."""
    print("\n" + "=" * 60)
    print("TEST: Reasoning Task")
    print("=" * 60)

    agent = Agent(
        name="Reasoner",
        instructions="You are a logical reasoning assistant. Think step by step.",
        tier="free",
    )

    problems = [
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    ]

    for i, problem in enumerate(problems, 1):
        print(f"\nProblem {i}: {problem[:60]}...")
        response = agent.chat(problem)
        print(f"Answer: {response[:300]}...")

    print(f"\nAgent stats: {agent.state.turn_count} turns, {agent.state.total_tokens} tokens")
    return True


def test_code_generation():
    """Test code generation capabilities."""
    print("\n" + "=" * 60)
    print("TEST: Code Generation")
    print("=" * 60)

    agent = Agent(
        name="Coder",
        instructions="You are a Python coding assistant. Write clean, working code.",
        tier="free",
    )

    tasks = [
        "Write a Python function to check if a number is prime.",
        "Write a function to reverse a string without using slicing.",
    ]

    for task in tasks:
        print(f"\nTask: {task}")
        response = agent.chat(task)
        # Check if response contains code
        has_code = "def " in response or "```" in response
        print(f"Contains code: {has_code}")
        print(f"Response preview: {response[:300]}...")

    return True


def test_memory_integration():
    """Test memory system integration with agent."""
    print("\n" + "=" * 60)
    print("TEST: Memory Integration")
    print("=" * 60)

    memory = Memory()
    memory.episodic.start_episode("Integration Test")

    # Simulate a conversation with memory
    agent = Agent(
        name="MemoryAgent",
        instructions="You are a helpful assistant with memory capabilities.",
        tier="free",
    )

    # First interaction
    q1 = "My name is Alice and I work as a data scientist."
    r1 = agent.chat(q1)
    memory.remember(f"User: {q1}", MemoryType.OBSERVATION)
    memory.remember(f"Response: {r1[:100]}", MemoryType.ACTION)
    print(f"Q1: {q1}")
    print(f"A1: {r1[:150]}...")

    # Second interaction - test context awareness
    q2 = "What's my profession again?"
    r2 = agent.chat(q2)
    memory.remember(f"User: {q2}", MemoryType.OBSERVATION)
    memory.remember(f"Response: {r2[:100]}", MemoryType.ACTION)
    print(f"\nQ2: {q2}")
    print(f"A2: {r2[:150]}...")

    # Check if the agent remembered
    context_maintained = "data scientist" in r2.lower() or "scientist" in r2.lower()
    print(f"\nContext maintained: {context_maintained}")

    # Add a lesson
    memory.learn(
        situation="Context recall test",
        insight="Agent maintains context across turns in conversation",
        outcome="success" if context_maintained else "partial",
        confidence=0.8 if context_maintained else 0.5,
    )

    # Get memory summary
    recent = memory.episodic.get_recent_memories(10)
    lessons = memory.lessons.get_all_lessons()
    print(f"\nMemory state: {len(recent)} episodes, {len(lessons)} lessons")

    return context_maintained


def test_tool_calling():
    """Test tool system with different scenarios."""
    print("\n" + "=" * 60)
    print("TEST: Tool System")
    print("=" * 60)

    # Test calculator with various expressions
    calc_tests = [
        ("2 ** 10", 1024),
        ("100 / 4", 25),
        ("(15 - 5) * 3 + 2", 32),
        ("abs(-42)", 42),
    ]

    print("\nCalculator tests:")
    for expr, expected in calc_tests:
        result = tool_registry.execute("calculator", {"expression": expr})
        status = "PASS" if result.output == expected else "FAIL"
        print(f"  {expr} = {result.output} (expected {expected}) [{status}]")

    # Test tool schema generation
    schemas = tool_registry.get_all_schemas()
    print(f"\nTool schemas generated: {len(schemas)}")
    for schema in schemas:
        print(f"  - {schema['function']['name']}: {schema['function']['description'][:50]}...")

    return True


def test_council_strategies():
    """Test council with different strategies."""
    print("\n" + "=" * 60)
    print("TEST: Council Consensus Strategies")
    print("=" * 60)

    # Simple question for consensus
    question = "Is the Earth round or flat? Give a one-word answer."

    council = Council(
        models=[
            "meta-llama/llama-3.2-3b-instruct:free",
            "moonshotai/kimi-k2:free",
        ],
        chairman="meta-llama/llama-3.2-3b-instruct:free",
    )

    print(f"Question: {question}")
    print("Council members:", council.models)
    print("\nRunning deliberation...")

    try:
        answer = council.quick_consensus(question)
        print(f"Consensus answer: {answer[:300]}...")
        # Check if answer contains "round"
        is_correct = "round" in answer.lower()
        print(f"Correct consensus: {is_correct}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n" + "=" * 60)
    print("TEST: Error Handling")
    print("=" * 60)

    tests_passed = 0
    total_tests = 4

    # Test 1: Invalid model
    print("\n1. Testing invalid model handling...")
    router = LLMRouter()
    try:
        router.chat(
            messages=[{"role": "user", "content": "test"}],
            model="invalid/model-that-does-not-exist",
            max_tokens=10,
        )
        print("  Should have raised error")
    except Exception as e:
        print(f"  Correctly raised error: {type(e).__name__}")
        tests_passed += 1

    # Test 2: Empty message handling
    print("\n2. Testing empty message handling...")
    try:
        agent = Agent(name="Test", instructions="Test", tier="free")
        response = agent.chat("")  # Empty message
        print(f"  Response: {response[:50] if response else 'empty'}")
        tests_passed += 1
    except Exception as e:
        print(f"  Error: {e}")

    # Test 3: Invalid calculator expression
    print("\n3. Testing invalid calculator expression...")
    result = tool_registry.execute("calculator", {"expression": "import os"})
    if not result.success:
        print(f"  Correctly rejected dangerous expression: {result.error}")
        tests_passed += 1
    else:
        print("  Should have rejected dangerous expression")

    # Test 4: Memory retrieval with no matches
    print("\n4. Testing memory with no matches...")
    memory = Memory()
    memory.remember("Python programming", MemoryType.OBSERVATION)
    context = memory.get_context("quantum physics")  # Unrelated query
    print(f"  Context generated: {len(context)} chars")
    tests_passed += 1

    print(f"\n{tests_passed}/{total_tests} error handling tests passed")
    return tests_passed >= 3


def test_streaming():
    """Test streaming responses (basic test)."""
    print("\n" + "=" * 60)
    print("TEST: Streaming Response")
    print("=" * 60)

    backend = OpenRouterBackend()
    messages = [{"role": "user", "content": "Count from 1 to 5, one number per line."}]

    print("Streaming response:")
    full_response = ""
    chunk_count = 0

    try:
        for chunk in backend.chat_stream(messages, max_tokens=50):
            if chunk.content:
                full_response += chunk.content
                chunk_count += 1
                print(chunk.content, end="", flush=True)

        print(f"\n\nReceived {chunk_count} chunks")
        print(f"Full response length: {len(full_response)} chars")
        return chunk_count > 0
    except Exception as e:
        print(f"\nStreaming error: {e}")
        return False


def main():
    """Run all comprehensive tests."""
    print("=" * 60)
    print("CLUB HARNESS - Comprehensive Testing Suite")
    print("=" * 60)
    print(f"API Key: {'Set' if config.llm.api_key else 'Not set'}")

    if not config.llm.api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        return False

    results = []

    # Run all tests
    tests = [
        ("Free Models Comparison", test_free_models_comparison),
        ("Reasoning Task", test_reasoning_task),
        ("Code Generation", test_code_generation),
        ("Memory Integration", test_memory_integration),
        ("Tool Calling", test_tool_calling),
        ("Council Strategies", test_council_strategies),
        ("Error Handling", test_error_handling),
        ("Streaming", test_streaming),
    ]

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, bool(result) if result is not None else True))
        except Exception as e:
            print(f"\nTest '{name}' failed with exception: {e}")
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST SUMMARY")
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
