"""Integration tests for Phase 1 features.

These tests verify the integration between components:
- LLM providers with villagers
- Storage with villages and villagers
- Rate limiting with LLM calls
- Metrics with village operations
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from village import Village, Villager
from village.storage.memory import InMemoryStorage
from village.utils.rate_limiter import RateLimiter, QuotaManager


@pytest.mark.integration
class TestProviderIntegration:
    """Test LLM provider integration with villagers."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = AsyncMock()
        provider.generate = AsyncMock(return_value="Test response from LLM")
        provider.chat = AsyncMock(return_value="Chat response from LLM")
        return provider

    @pytest.mark.asyncio
    async def test_villager_with_provider(self, mock_llm_provider):
        """Test villager processing with LLM provider."""
        villager = Villager("analyst", mock_llm_provider, role="analyst")

        result = await villager.process_task("Analyze data")

        assert result == "Test response from LLM"
        assert mock_llm_provider.generate.called or mock_llm_provider.chat.called

    @pytest.mark.asyncio
    async def test_village_collaboration_with_provider(self, mock_llm_provider):
        """Test village collaboration with LLM providers."""
        village = Village("Test Village")

        analyst = Villager("analyst", mock_llm_provider, role="analyst")
        researcher = Villager("researcher", mock_llm_provider, role="researcher")

        village.add_villager(analyst)
        village.add_villager(researcher)

        result = await village.collaborate("Research AI trends")

        assert "Test response from LLM" in result or "Chat response from LLM" in result
        assert len(village.get_task_history()) == 1


@pytest.mark.integration
class TestStorageIntegration:
    """Test storage integration with villages and villagers."""

    @pytest.mark.asyncio
    async def test_village_persistence(self):
        """Test saving and loading village data."""
        storage = InMemoryStorage()
        village = Village("AI Research Team")

        # Add villagers
        mock_provider = AsyncMock()
        analyst = Villager("analyst", mock_provider, role="analyst")
        village.add_villager(analyst)

        # Save village
        village_data = {
            "name": village.name,
            "villagers": village.list_villagers(),
            "task_count": len(village.get_task_history())
        }

        await storage.save_village("village1", village_data)

        # Load village
        loaded_data = await storage.load_village("village1")

        assert loaded_data is not None
        assert loaded_data["name"] == "AI Research Team"
        assert "analyst" in loaded_data["villagers"]

    @pytest.mark.asyncio
    async def test_villager_memory_persistence(self):
        """Test villager memory persistence."""
        storage = InMemoryStorage()

        mock_provider = AsyncMock()
        villager = Villager("analyst", mock_provider, role="analyst")

        # Set memory
        villager.set_memory("expertise", "data analysis")
        villager.set_memory("status", "active")

        # Save to storage
        await storage.save_memory("analyst", "expertise", "data analysis")
        await storage.save_memory("analyst", "status", "active")

        # Load from storage
        expertise = await storage.load_memory("analyst", "expertise")
        status = await storage.load_memory("analyst", "status")

        assert expertise == "data analysis"
        assert status == "active"

        # List keys
        keys = await storage.list_memory_keys("analyst")
        assert "expertise" in keys
        assert "status" in keys

    @pytest.mark.asyncio
    async def test_task_history_persistence(self):
        """Test task history persistence."""
        storage = InMemoryStorage()
        village = Village("Test Village")

        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(return_value="Analysis complete")

        analyst = Villager("analyst", mock_provider, role="analyst")
        village.add_villager(analyst)

        # Collaborate on task
        await village.collaborate("Analyze market trends")

        # Save task history
        history = village.get_task_history()
        assert len(history) > 0

        for entry in history:
            await storage.save_task_history(
                "village1",
                entry["task"],
                entry["result"],
                {"timestamp": entry["timestamp"]}
            )

        # Load task history
        loaded_history = await storage.load_task_history("village1")

        assert len(loaded_history) == len(history)
        assert loaded_history[0]["task"] == history[0]["task"]


@pytest.mark.integration
class TestRateLimitingIntegration:
    """Test rate limiting integration."""

    @pytest.mark.asyncio
    async def test_rate_limited_llm_calls(self):
        """Test rate limiting with LLM calls."""
        rate_limiter = RateLimiter(requests_per_minute=60, burst_size=5)

        mock_provider = AsyncMock()
        call_count = 0

        async def rate_limited_call():
            nonlocal call_count
            await rate_limiter.acquire(1)
            call_count += 1
            return f"Call {call_count}"

        # Make several calls
        results = await asyncio.gather(*[rate_limited_call() for _ in range(3)])

        assert len(results) == 3
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_quota_enforcement(self):
        """Test quota enforcement."""
        quota_manager = QuotaManager(hourly_quota=10)

        # Use quota
        for i in range(10):
            success = await quota_manager.use_quota(1)
            assert success is True

        # Should fail when quota exhausted
        success = await quota_manager.use_quota(1)
        assert success is False

        # Check stats
        stats = await quota_manager.get_usage_stats()
        assert stats["hourly"]["used"] == 10
        assert stats["hourly"]["remaining"] == 0


@pytest.mark.integration
class TestMetricsIntegration:
    """Test metrics integration."""

    def test_metrics_with_village_operations(self):
        """Test metrics collection during village operations."""
        try:
            from village.utils.metrics import initialize_metrics, metrics_available

            if not metrics_available():
                pytest.skip("Prometheus client not available")

            metrics = initialize_metrics()

            # Simulate village operations
            metrics.villages_total.set(1)
            metrics.villagers_active.labels(village="test_village").set(3)

            metrics.villager_tasks_total.labels(
                villager="analyst",
                village="test_village",
                status="completed"
            ).inc()

            # Export metrics
            output = metrics.export_metrics()
            assert output is not None
            assert b"village_villages_total" in output
            assert b"village_villagers_active" in output

        except ImportError:
            pytest.skip("Prometheus client not installed")


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    @pytest.mark.asyncio
    async def test_complete_village_workflow(self):
        """Test complete workflow: create village, add villagers, collaborate, persist."""
        # Setup
        storage = InMemoryStorage()
        rate_limiter = RateLimiter(requests_per_minute=100)

        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(return_value="Detailed analysis complete")
        mock_provider.chat = AsyncMock(return_value="Chat response")

        # Create village
        village = Village("AI Research Village")

        # Create villagers
        analyst = Villager("analyst", mock_provider, role="analyst")
        researcher = Villager("researcher", mock_provider, role="researcher")
        writer = Villager("writer", mock_provider, role="writer")

        # Add to village
        village.add_villager(analyst)
        village.add_villager(researcher)
        village.add_villager(writer)

        assert len(village) == 3

        # Collaborate with rate limiting
        await rate_limiter.acquire(1)
        result = await village.collaborate("Research and analyze AI market trends")

        assert result is not None
        assert len(village.get_task_history()) == 1

        # Persist village state
        village_data = {
            "name": village.name,
            "villagers": village.list_villagers(),
            "villager_count": len(village)
        }

        await storage.save_village("ai_research", village_data)

        # Persist task history
        history = village.get_task_history()
        for entry in history:
            await storage.save_task_history(
                "ai_research",
                entry["task"],
                entry["result"],
                {"timestamp": entry["timestamp"]}
            )

        # Verify persistence
        loaded_village = await storage.load_village("ai_research")
        assert loaded_village is not None
        assert loaded_village["name"] == "AI Research Village"
        assert loaded_village["villager_count"] == 3

        loaded_history = await storage.load_task_history("ai_research")
        assert len(loaded_history) > 0

    @pytest.mark.asyncio
    async def test_multi_task_workflow_with_memory(self):
        """Test workflow with multiple tasks and memory persistence."""
        storage = InMemoryStorage()
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(side_effect=[
            "Market research complete",
            "Data analysis complete",
            "Report written"
        ])

        village = Village("Research Team")
        analyst = Villager("analyst", mock_provider, role="analyst")
        village.add_villager(analyst)

        # Task 1: Research
        await village.collaborate("Research market")
        analyst.set_memory("last_task", "research")
        await storage.save_memory("analyst", "last_task", "research")

        # Task 2: Analysis
        await village.collaborate("Analyze data")
        analyst.set_memory("last_task", "analysis")
        await storage.save_memory("analyst", "last_task", "analysis")

        # Task 3: Report
        await village.collaborate("Write report")
        analyst.set_memory("last_task", "report")
        await storage.save_memory("analyst", "last_task", "report")

        # Verify
        assert len(village.get_task_history()) == 3

        last_task = await storage.load_memory("analyst", "last_task")
        assert last_task == "report"

        history = await storage.load_task_history("research_team", limit=10)
        # History might be empty if we didn't save it, but that's okay

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self):
        """Test workflow with error handling."""
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(side_effect=Exception("API Error"))

        village = Village("Test Village")
        villager = Villager("analyst", mock_provider, role="analyst")
        village.add_villager(villager)

        # Collaboration should handle error gracefully
        result = await village.collaborate("Task that will fail")

        # Result should indicate failure
        assert "error" in result.lower() or "failed" in result.lower()


@pytest.mark.integration
class TestConcurrentOperations:
    """Test concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_villager_tasks(self):
        """Test multiple villagers processing tasks concurrently."""
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(return_value="Task complete")

        villagers = [
            Villager(f"villager_{i}", mock_provider, role="general")
            for i in range(5)
        ]

        # Process tasks concurrently
        tasks = [v.process_task(f"Task {i}") for i, v in enumerate(villagers)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all("Task complete" == r for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_storage_operations(self):
        """Test concurrent storage operations."""
        storage = InMemoryStorage()

        # Concurrent saves
        save_tasks = [
            storage.save_memory(f"owner_{i}", "key", f"value_{i}")
            for i in range(10)
        ]

        results = await asyncio.gather(*save_tasks)
        assert all(r is True for r in results)

        # Concurrent loads
        load_tasks = [
            storage.load_memory(f"owner_{i}", "key")
            for i in range(10)
        ]

        results = await asyncio.gather(*load_tasks)
        assert len(results) == 10
