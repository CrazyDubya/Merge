#!/usr/bin/env python3
"""
Test script for Club Harness OpenRouter integration.

Tests:
1. Basic LLM connection
2. Agent creation and chat
3. Model tier switching
4. Multi-turn conversation
"""

import os
import sys

# Add club_harness to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from club_harness.core.config import config
from club_harness.llm.openrouter import OpenRouterBackend
from club_harness.llm.router import LLMRouter
from club_harness.core.agent import Agent, AgentBuilder


def test_config():
    """Test configuration loading."""
    print("\n=== Test 1: Configuration ===")
    print(f"Provider: {config.llm.provider}")
    print(f"Default Model: {config.llm.model}")
    print(f"API Key Set: {'Yes' if config.llm.api_key else 'No'}")
    print(f"Free Models: {config.model_tiers['free']}")
    return config.llm.api_key is not None


def test_openrouter_backend():
    """Test direct OpenRouter API connection."""
    print("\n=== Test 2: OpenRouter Backend ===")

    try:
        backend = OpenRouterBackend()

        messages = [
            {"role": "user", "content": "Say 'Hello from Club Harness!' and nothing else."}
        ]

        print(f"Testing with model: {backend.default_model}")
        response = backend.chat(messages, max_tokens=50)

        print(f"Response: {response.content}")
        print(f"Model: {response.model}")
        print(f"Tokens: {response.usage}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_llm_router():
    """Test LLM router with tier selection."""
    print("\n=== Test 3: LLM Router ===")

    try:
        router = LLMRouter()

        # Test free tier
        print("Testing free tier...")
        response = router.chat(
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            tier="free",
            max_tokens=20,
        )
        print(f"Free tier response: {response.content}")
        print(f"Model used: {response.model}")
        print(f"Tokens: {response.total_tokens}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_agent():
    """Test Agent creation and chat."""
    print("\n=== Test 4: Agent ===")

    try:
        agent = (
            AgentBuilder("TestAgent")
            .with_instructions("You are a helpful assistant. Be concise.")
            .with_persona(style="friendly", expertise="general")
            .with_tier("free")
            .build()
        )

        print(f"Created agent: {agent.name}")
        print(f"Persona: {agent.persona}")

        response = agent.chat("What's the capital of France? One word answer.")
        print(f"Agent response: {response}")
        print(f"Turns used: {agent.state.turn_count}")
        print(f"Total tokens: {agent.state.total_tokens}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_multi_turn():
    """Test multi-turn conversation."""
    print("\n=== Test 5: Multi-turn Conversation ===")

    try:
        agent = Agent(
            name="ConversationAgent",
            instructions="You are a math tutor. Be very concise.",
            tier="free",
        )

        # Turn 1
        r1 = agent.chat("What is 5 + 3?")
        print(f"Turn 1: {r1}")

        # Turn 2 (should have context)
        r2 = agent.chat("Now multiply that by 2")
        print(f"Turn 2: {r2}")

        print(f"Total turns: {agent.state.turn_count}")
        print(f"Total tokens: {agent.state.total_tokens}")
        print(f"Message history: {len(agent.state.messages)} messages")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_different_models():
    """Test different free models."""
    print("\n=== Test 6: Different Free Models ===")

    # Currently available free models (Jan 2026)
    free_models = [
        "meta-llama/llama-3.2-3b-instruct:free",
        "qwen/qwen3-coder:free",
    ]

    router = LLMRouter()

    for model in free_models:
        try:
            print(f"\nTesting {model}...")
            response = router.chat(
                messages=[{"role": "user", "content": "Hello! Say hi back in 5 words or less."}],
                model=model,
                max_tokens=30,
            )
            print(f"Response: {response.content.strip()}")
        except Exception as e:
            print(f"Error with {model}: {e}")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("CLUB HARNESS - OpenRouter Integration Test")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Config", test_config()))

    if results[0][1]:  # Only continue if API key is set
        results.append(("OpenRouter Backend", test_openrouter_backend()))
        results.append(("LLM Router", test_llm_router()))
        results.append(("Agent", test_agent()))
        results.append(("Multi-turn", test_multi_turn()))
        results.append(("Different Models", test_different_models()))
    else:
        print("\nSkipping tests - OPENROUTER_API_KEY not set")

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
