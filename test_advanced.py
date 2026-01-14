#!/usr/bin/env python3
"""
Advanced test script for Club Harness features.

Tests:
1. Memory system (episodic + lessons)
2. Council consensus
3. Tool system
4. Integrated agent with memory
"""

import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from club_harness.memory import Memory, MemoryType, Lesson
from club_harness.tools import tool_registry, ToolCall
from club_harness.orchestration import Council
from club_harness.core.agent import Agent, AgentBuilder
from club_harness.core.config import config


def test_memory_system():
    """Test the memory system."""
    print("\n=== Test 1: Memory System ===")

    try:
        memory = Memory()

        # Test episodic memory
        memory.episodic.start_episode("Test Session")
        memory.remember("User asked about Python", MemoryType.OBSERVATION)
        memory.remember("I explained list comprehensions", MemoryType.ACTION)
        memory.remember("This was a good teaching moment", MemoryType.THOUGHT, importance=0.8)

        # Get recent memories
        recent = memory.episodic.get_recent_memories(3)
        print(f"Recent memories: {len(recent)}")
        for m in recent:
            print(f"  [{m.memory_type.value}] {m.content}")

        # Test lesson learning
        lesson = memory.learn(
            situation="Teaching programming concepts",
            insight="Use concrete examples before abstract explanations",
            outcome="success",
            confidence=0.7,
        )
        print(f"\nLearned lesson: {lesson.insight}")
        print(f"Confidence: {lesson.confidence}")

        # Test context generation
        context = memory.get_context("programming question")
        print(f"\nGenerated context length: {len(context)} chars")

        # Test working memory
        memory.set_working("current_topic", "Python")
        assert memory.get_working("current_topic") == "Python"
        print("Working memory: OK")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_system():
    """Test the tool system."""
    print("\n=== Test 2: Tool System ===")

    try:
        # List available tools
        tools = tool_registry.list_tools()
        print(f"Available tools: {tools}")

        # Test calculator tool
        result = tool_registry.execute("calculator", {"expression": "2 + 2 * 3"})
        print(f"\nCalculator 2+2*3 = {result.output}")
        assert result.success
        assert result.output == 8

        # Test with complex expression
        result = tool_registry.execute("calculator", {"expression": "(10 + 5) * 2"})
        print(f"Calculator (10+5)*2 = {result.output}")
        assert result.output == 30

        # Test tool schemas
        schemas = tool_registry.get_all_schemas()
        print(f"\nGenerated {len(schemas)} tool schemas for LLM")

        # Test web search (simulated)
        result = tool_registry.execute("web_search", {"query": "Python tutorials"})
        print(f"Web search returned {len(result.output['results'])} results")

        # Test shell tool requires confirmation
        result = tool_registry.execute("shell", {"command": "echo hello"})
        assert not result.success  # Should fail without confirmation
        print("Shell tool correctly requires confirmation")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_council_basic():
    """Test council with simple synchronous call."""
    print("\n=== Test 3: Council Consensus ===")

    if not config.llm.api_key:
        print("Skipping - no API key")
        return True

    try:
        # Create council with free models
        council = Council(
            models=[
                "meta-llama/llama-3.2-3b-instruct:free",
                "qwen/qwen3-coder:free",
            ],
            chairman="meta-llama/llama-3.2-3b-instruct:free",
        )

        print("Council models:", council.models)
        print("Chairman:", council.chairman)

        # Test quick consensus
        print("\nRunning council deliberation (this may take a moment)...")
        answer = council.quick_consensus("What is 15 + 27? Just give the number.")
        print(f"Council answer: {answer[:200]}...")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_with_memory():
    """Test agent with integrated memory."""
    print("\n=== Test 4: Agent with Memory ===")

    if not config.llm.api_key:
        print("Skipping - no API key")
        return True

    try:
        # Create agent with memory
        memory = Memory()
        memory.episodic.start_episode("Math tutoring session")

        agent = (
            AgentBuilder("MathTutor")
            .with_instructions("You are a math tutor. Be concise. Show your work.")
            .with_persona(style="patient", expertise="mathematics")
            .with_tier("free")
            .build()
        )

        # First interaction
        print("Turn 1: Asking about addition...")
        r1 = agent.chat("What is 5 + 3?")
        memory.remember(f"User asked: 5 + 3. I answered: {r1[:50]}", MemoryType.CONVERSATION)
        print(f"Response: {r1}")

        # Second interaction (context-aware)
        print("\nTurn 2: Follow-up question...")
        r2 = agent.chat("Now multiply that result by 4")
        memory.remember(f"Follow-up multiply. I answered: {r2[:50]}", MemoryType.CONVERSATION)
        print(f"Response: {r2}")

        # Extract lesson
        memory.learn(
            situation="Multi-step math problems",
            insight="Break down into steps and reference previous results",
            outcome="success",
        )

        # Check agent state
        print(f"\nAgent state:")
        print(f"  Turns: {agent.state.turn_count}")
        print(f"  Tokens: {agent.state.total_tokens}")
        print(f"  Messages: {len(agent.state.messages)}")

        # Check memory
        print(f"\nMemory state:")
        print(f"  Episodic entries: {len(memory.episodic.get_recent_memories(100))}")
        print(f"  Lessons: {len(memory.lessons.get_all_lessons())}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lesson_decay():
    """Test lesson confidence decay over time."""
    print("\n=== Test 5: Lesson Decay ===")

    try:
        from datetime import datetime, timedelta

        lesson = Lesson(
            id="test",
            situation="Testing",
            insight="Test insight",
            outcome="success",
            confidence=0.8,
        )

        # Simulate age
        lesson.created_at = datetime.now() - timedelta(days=7)

        decayed = lesson.decay(half_life_days=14.0)
        print(f"Original confidence: {lesson.confidence}")
        print(f"After 7 days (14-day half-life): {decayed:.3f}")

        # Should be approximately 0.8 * 0.7 ≈ 0.56
        assert 0.5 < decayed < 0.7
        print("Decay calculation: OK")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("CLUB HARNESS - Advanced Features Test")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Memory System", test_memory_system()))
    results.append(("Tool System", test_tool_system()))
    results.append(("Lesson Decay", test_lesson_decay()))
    results.append(("Agent with Memory", test_agent_with_memory()))
    results.append(("Council Consensus", test_council_basic()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
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
