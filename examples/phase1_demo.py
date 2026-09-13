"""Phase 1 Features Demo - LLM Providers, Storage, and Monitoring.

This example demonstrates the new Phase 1 capabilities:
1. OpenAI provider integration
2. Anthropic provider integration
3. Google Gemini provider integration
4. Persistent storage with PostgreSQL
5. Rate limiting and quota management
6. Prometheus metrics

Usage:
    Set environment variables:
    - OPENAI_API_KEY (optional)
    - ANTHROPIC_API_KEY (optional)
    - GOOGLE_API_KEY (optional)
    - DATABASE_URL (optional, for PostgreSQL)

    Then run:
    python examples/phase1_demo.py
"""

import asyncio
import os
from typing import Optional

# Village core imports
from village import Village, Villager

# Storage imports
from village.storage.memory import InMemoryStorage

# Rate limiting imports
from village.utils.rate_limiter import RateLimiter, QuotaManager


async def demo_openai_provider() -> None:
    """Demonstrate OpenAI provider usage."""
    print("\n" + "="*60)
    print("DEMO 1: OpenAI Provider")
    print("="*60)

    try:
        from village.llm.openai import OpenAIProvider

        # Initialize provider
        provider = OpenAIProvider(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo"
        )

        # Get model info
        info = provider.get_model_info()
        print(f"\nModel Info:")
        print(f"  Provider: {info['provider']}")
        print(f"  Model: {info['model']}")
        print(f"  Context Window: {info['context_window']} tokens")

        # Create villager with OpenAI
        analyst = Villager("analyst", provider, role="analyst")

        # Process a task
        print("\nProcessing task with OpenAI...")
        result = await analyst.process_task(
            "Analyze the benefits of multi-agent AI systems in 3 points"
        )
        print(f"Result: {result[:200]}...")

    except ImportError:
        print("OpenAI package not installed. Skipping OpenAI demo.")
        print("Install with: pip install openai>=1.0.0")
    except Exception as e:
        print(f"Error: {e}")


async def demo_anthropic_provider() -> None:
    """Demonstrate Anthropic provider usage."""
    print("\n" + "="*60)
    print("DEMO 2: Anthropic Claude Provider")
    print("="*60)

    try:
        from village.llm.anthropic import AnthropicProvider

        # Initialize provider
        provider = AnthropicProvider(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-sonnet-20240229"
        )

        # Get model info
        info = provider.get_model_info()
        print(f"\nModel Info:")
        print(f"  Provider: {info['provider']}")
        print(f"  Model: {info['model']}")
        print(f"  Context Window: {info['context_window']} tokens")
        print(f"  Supports Vision: {info['supports_vision']}")

        # Create villager with Anthropic
        researcher = Villager("researcher", provider, role="researcher")

        # Process a task
        print("\nProcessing task with Claude...")
        result = await researcher.process_task(
            "Research the current state of multi-agent AI systems"
        )
        print(f"Result: {result[:200]}...")

    except ImportError:
        print("Anthropic package not installed. Skipping Anthropic demo.")
        print("Install with: pip install anthropic>=0.7.0")
    except Exception as e:
        print(f"Error: {e}")


async def demo_google_provider() -> None:
    """Demonstrate Google Gemini provider usage."""
    print("\n" + "="*60)
    print("DEMO 3: Google Gemini Provider")
    print("="*60)

    try:
        from village.llm.google import GoogleProvider

        # Initialize provider
        provider = GoogleProvider(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model="gemini-pro"
        )

        # Get model info
        info = provider.get_model_info()
        print(f"\nModel Info:")
        print(f"  Provider: {info['provider']}")
        print(f"  Model: {info['model']}")
        print(f"  Context Window: {info['context_window']} tokens")

        # Create villager with Google
        writer = Villager("writer", provider, role="writer")

        # Process a task
        print("\nProcessing task with Gemini...")
        result = await writer.process_task(
            "Write a brief overview of the Village framework"
        )
        print(f"Result: {result[:200]}...")

    except ImportError:
        print("Google GenerativeAI package not installed. Skipping Google demo.")
        print("Install with: pip install google-generativeai>=0.3.0")
    except Exception as e:
        print(f"Error: {e}")


async def demo_storage() -> None:
    """Demonstrate storage capabilities."""
    print("\n" + "="*60)
    print("DEMO 4: Persistent Storage")
    print("="*60)

    # Use in-memory storage for demo
    storage = InMemoryStorage()

    # Save village data
    village_data = {
        "name": "AI Research Village",
        "villagers": ["analyst", "researcher", "writer"],
        "created": "2025-01-01"
    }

    print("\nSaving village data...")
    await storage.save_village("village1", village_data)

    # Load village data
    loaded = await storage.load_village("village1")
    print(f"Loaded village: {loaded['name']}")
    print(f"Villagers: {', '.join(loaded['villagers'])}")

    # Save memory entries
    print("\nSaving memory entries...")
    await storage.save_memory("analyst", "expertise", "data analysis")
    await storage.save_memory("analyst", "status", "active")

    # Load memory
    expertise = await storage.load_memory("analyst", "expertise")
    print(f"Analyst expertise: {expertise}")

    # List memory keys
    keys = await storage.list_memory_keys("analyst")
    print(f"Analyst memory keys: {keys}")

    # Save task history
    print("\nSaving task history...")
    await storage.save_task_history(
        "village1",
        "Analyze AI trends",
        "Completed analysis of multi-agent systems"
    )

    # Load task history
    history = await storage.load_task_history("village1", limit=10)
    print(f"Task history entries: {len(history)}")
    if history:
        print(f"Latest task: {history[0]['task']}")

    # Health check
    healthy = await storage.health_check()
    print(f"\nStorage health: {'OK' if healthy else 'FAILED'}")


async def demo_rate_limiting() -> None:
    """Demonstrate rate limiting and quota management."""
    print("\n" + "="*60)
    print("DEMO 5: Rate Limiting & Quota Management")
    print("="*60)

    # Create rate limiter
    limiter = RateLimiter(requests_per_minute=10, burst_size=5)

    print("\nTesting rate limiter...")
    print(f"Tokens available: {limiter.get_tokens_available():.2f}")

    # Acquire tokens
    print("Acquiring 3 tokens...")
    wait_time = await limiter.acquire(3)
    print(f"Acquired in {wait_time:.3f}s")
    print(f"Tokens remaining: {limiter.get_tokens_available():.2f}")

    # Try to acquire without waiting
    print("\nTrying non-blocking acquisition...")
    success = await limiter.try_acquire(1)
    print(f"Success: {success}")
    print(f"Tokens remaining: {limiter.get_tokens_available():.2f}")

    # Quota manager
    print("\n" + "-"*60)
    print("Testing quota manager...")

    quota = QuotaManager(
        hourly_quota=100,
        daily_quota=1000,
        monthly_quota=10000
    )

    # Check quota
    print("\nChecking quota availability...")
    can_use = await quota.check_quota(10)
    print(f"Can use 10 requests: {can_use}")

    # Use quota
    print("Using 10 requests...")
    used = await quota.use_quota(10)
    print(f"Quota used successfully: {used}")

    # Get usage stats
    stats = await quota.get_usage_stats()
    print("\nQuota usage stats:")
    for period, data in stats.items():
        print(f"  {period.capitalize()}:")
        print(f"    Used: {data['used']}")
        print(f"    Quota: {data['quota']}")
        print(f"    Remaining: {data['remaining']}")


async def demo_metrics() -> None:
    """Demonstrate Prometheus metrics."""
    print("\n" + "="*60)
    print("DEMO 6: Prometheus Metrics")
    print("="*60)

    try:
        from village.utils.metrics import initialize_metrics, metrics_available

        if not metrics_available():
            print("Prometheus client not installed.")
            print("Install with: pip install prometheus-client>=0.17.0")
            return

        # Initialize metrics
        metrics = initialize_metrics()
        print("\nMetrics initialized successfully!")

        # Record some metrics
        print("\nRecording sample metrics...")

        # LLM request
        metrics.llm_requests_total.labels(
            provider="openai",
            model="gpt-3.5-turbo",
            method="chat",
            status="success"
        ).inc()

        # Villager task
        metrics.villager_tasks_total.labels(
            villager="analyst",
            village="ai_research",
            status="completed"
        ).inc()

        # Set gauges
        metrics.villages_total.set(1)
        metrics.villagers_active.labels(village="ai_research").set(3)

        # Export metrics
        print("\nExporting metrics in Prometheus format...")
        output = metrics.export_metrics()
        lines = output.decode('utf-8').split('\n')

        # Show sample metrics
        print("\nSample metrics (first 20 lines):")
        for line in lines[:20]:
            if line and not line.startswith('#'):
                print(f"  {line}")

    except Exception as e:
        print(f"Error: {e}")


async def demo_multi_provider_village() -> None:
    """Demonstrate a village using multiple providers."""
    print("\n" + "="*60)
    print("DEMO 7: Multi-Provider Village Collaboration")
    print("="*60)

    try:
        # This demo requires at least one provider to be available
        provider = None
        provider_name = "None"

        # Try to initialize a provider
        try:
            from village.llm.openai import OpenAIProvider
            if os.getenv("OPENAI_API_KEY"):
                provider = OpenAIProvider()
                provider_name = "OpenAI"
        except (ImportError, Exception):
            pass

        if not provider:
            try:
                from village.llm.anthropic import AnthropicProvider
                if os.getenv("ANTHROPIC_API_KEY"):
                    provider = AnthropicProvider()
                    provider_name = "Anthropic"
            except (ImportError, Exception):
                pass

        if not provider:
            print("No LLM provider available. Set API keys to run this demo.")
            return

        print(f"\nUsing {provider_name} provider")

        # Create village
        village = Village("Multi-Agent Research Team")

        # Create villagers with different roles
        analyst = Villager("analyst", provider, role="analyst")
        researcher = Villager("researcher", provider, role="researcher")
        writer = Villager("writer", provider, role="writer")

        # Add villagers to village
        village.add_villager(analyst)
        village.add_villager(researcher)
        village.add_villager(writer)

        print(f"\nVillage created with {len(village)} villagers:")
        for name in village.list_villagers():
            print(f"  - {name}")

        # Collaborate on a task
        print("\nCollaborating on task...")
        result = await village.collaborate(
            "Briefly explain the concept of multi-agent AI systems"
        )

        print(f"\nCollaboration result preview:")
        print(f"{result[:300]}...")

        # Show task history
        history = village.get_task_history()
        print(f"\nTask history: {len(history)} tasks completed")

    except Exception as e:
        print(f"Error: {e}")


async def main() -> None:
    """Run all demos."""
    print("\n" + "#"*60)
    print("# Village Framework - Phase 1 Feature Demonstrations")
    print("#"*60)
    print("\nThis demo showcases the new Phase 1 capabilities:")
    print("  1. OpenAI GPT provider")
    print("  2. Anthropic Claude provider")
    print("  3. Google Gemini provider")
    print("  4. Persistent storage (PostgreSQL + In-Memory)")
    print("  5. Rate limiting and quota management")
    print("  6. Prometheus metrics and monitoring")
    print("  7. Multi-provider village collaboration")

    # Run individual demos
    await demo_openai_provider()
    await demo_anthropic_provider()
    await demo_google_provider()
    await demo_storage()
    await demo_rate_limiting()
    await demo_metrics()
    await demo_multi_provider_village()

    print("\n" + "#"*60)
    print("# Demo Complete!")
    print("#"*60)
    print("\nNext steps:")
    print("  1. Set up your API keys for the providers you want to use")
    print("  2. Install optional dependencies: pip install -e .[all]")
    print("  3. Configure PostgreSQL for persistent storage (optional)")
    print("  4. Set up Prometheus to scrape metrics (optional)")
    print("  5. See documentation for more advanced usage")


if __name__ == "__main__":
    asyncio.run(main())
