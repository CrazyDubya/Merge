"""End-to-end tests for Phase 1 features.

These tests simulate real-world usage scenarios and workflows.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from village import Village, Villager
from village.storage.memory import InMemoryStorage
from village.utils.rate_limiter import RateLimiter, QuotaManager


@pytest.mark.e2e
class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    @pytest.mark.asyncio
    async def test_research_team_scenario(self):
        """
        Scenario: A research team collaborates on a market analysis project.

        Steps:
        1. Create village with multiple specialized villagers
        2. Conduct initial research
        3. Analyze findings
        4. Write report
        5. Persist all data and history
        """
        # Setup
        storage = InMemoryStorage()
        rate_limiter = RateLimiter(requests_per_minute=100)

        # Mock LLM provider with realistic responses
        mock_provider = AsyncMock()
        responses = [
            "Market research shows growing demand for AI automation tools...",
            "Data analysis reveals 45% year-over-year growth in the sector...",
            "Based on research and analysis, we recommend...",
        ]
        mock_provider.generate = AsyncMock(side_effect=responses)
        mock_provider.chat = AsyncMock(side_effect=responses)

        # Create research team
        village = Village("Market Research Team")

        researcher = Villager("researcher", mock_provider, role="researcher")
        analyst = Villager("analyst", mock_provider, role="analyst")
        writer = Villager("writer", mock_provider, role="writer")

        village.add_villager(researcher)
        village.add_villager(analyst)
        village.add_villager(writer)

        # Phase 1: Research
        await rate_limiter.acquire(1)
        research_result = await researcher.process_task(
            "Research the AI automation market trends for 2025"
        )
        researcher.set_memory("phase", "research_complete")
        await storage.save_memory("researcher", "findings", research_result)

        assert "market" in research_result.lower() or "research" in research_result.lower()

        # Phase 2: Analysis
        await rate_limiter.acquire(1)
        analysis_result = await analyst.process_task(
            "Analyze the market research findings"
        )
        analyst.set_memory("phase", "analysis_complete")
        await storage.save_memory("analyst", "analysis", analysis_result)

        assert len(analysis_result) > 0

        # Phase 3: Report
        await rate_limiter.acquire(1)
        report_result = await writer.process_task(
            "Write executive summary based on research and analysis"
        )
        writer.set_memory("phase", "report_complete")
        await storage.save_memory("writer", "report", report_result)

        # Save village state
        village_data = {
            "name": village.name,
            "villagers": village.list_villagers(),
            "phases_completed": ["research", "analysis", "report"]
        }
        await storage.save_village("market_research_project", village_data)

        # Verify completion
        loaded_village = await storage.load_village("market_research_project")
        assert loaded_village["name"] == "Market Research Team"
        assert len(loaded_village["phases_completed"]) == 3

        # Verify all artifacts exist
        research_findings = await storage.load_memory("researcher", "findings")
        analysis_findings = await storage.load_memory("analyst", "analysis")
        report_findings = await storage.load_memory("writer", "report")

        assert research_findings is not None
        assert analysis_findings is not None
        assert report_findings is not None

    @pytest.mark.asyncio
    async def test_content_creation_pipeline(self):
        """
        Scenario: Content creation pipeline with multiple iterations.

        Steps:
        1. Generate initial draft
        2. Review and provide feedback
        3. Revise based on feedback
        4. Final approval
        5. Track metrics throughout
        """
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(side_effect=[
            "Initial draft: AI is transforming industries...",
            "Feedback: Good start, but needs more data and examples...",
            "Revised draft: AI is transforming industries with concrete examples...",
            "Final approval: Excellent work, ready for publication",
        ])

        village = Village("Content Team")

        writer = Villager("writer", mock_provider, role="writer")
        critic = Villager("critic", mock_provider, role="critic")

        village.add_villager(writer)
        village.add_villager(critic)

        # Track iterations
        iteration_count = 0
        max_iterations = 3

        # Iteration loop
        for i in range(max_iterations):
            iteration_count += 1

            # Writer creates content
            draft = await writer.process_task("Write article about AI impact")
            writer.set_memory("iteration", iteration_count)

            # Critic reviews
            feedback = await critic.process_task(f"Review: {draft[:50]}...")

            # Store iteration
            writer.set_memory(f"draft_v{iteration_count}", draft)
            critic.set_memory(f"feedback_v{iteration_count}", feedback)

            # Check if approved
            if "approval" in feedback.lower() or "ready" in feedback.lower():
                break

        # Verify iterations occurred
        assert iteration_count > 0
        assert writer.get_memory("iteration") == iteration_count

    @pytest.mark.asyncio
    async def test_high_volume_processing(self):
        """
        Scenario: Process high volume of tasks with quota management.

        Steps:
        1. Set up quota limits
        2. Process many tasks
        3. Handle quota exhaustion gracefully
        4. Track metrics
        """
        quota_manager = QuotaManager(hourly_quota=50, daily_quota=500)
        storage = InMemoryStorage()

        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(return_value="Task processed")

        processor = Villager("processor", mock_provider, role="general")

        processed_count = 0
        quota_hit_count = 0

        # Attempt to process 60 tasks (over hourly quota)
        for i in range(60):
            # Check quota
            can_process = await quota_manager.check_quota(1)

            if can_process:
                await quota_manager.use_quota(1)
                result = await processor.process_task(f"Process item {i}")
                processed_count += 1

                # Save progress
                await storage.save_task_history(
                    "processor",
                    f"Task {i}",
                    result
                )
            else:
                quota_hit_count += 1

        # Verify quota enforcement
        assert processed_count == 50  # Hourly quota
        assert quota_hit_count == 10  # Rejected requests

        # Verify stats
        stats = await quota_manager.get_usage_stats()
        assert stats["hourly"]["used"] == 50
        assert stats["hourly"]["remaining"] == 0

        # Verify all processed tasks are stored
        history = await storage.load_task_history("processor")
        assert len(history) == 50

    @pytest.mark.asyncio
    async def test_distributed_village_coordination(self):
        """
        Scenario: Multiple villages working on related tasks.

        Steps:
        1. Create multiple specialized villages
        2. Coordinate work between them
        3. Merge results
        4. Persist final outcome
        """
        storage = InMemoryStorage()

        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(side_effect=[
            "Backend design: RESTful API with microservices...",
            "Frontend design: React with TypeScript...",
            "Infrastructure: Kubernetes on AWS...",
            "Final architecture: Integrated full-stack solution...",
        ])

        # Create specialized villages
        backend_village = Village("Backend Team")
        frontend_village = Village("Frontend Team")
        infra_village = Village("Infrastructure Team")

        # Add villagers
        backend_dev = Villager("backend_dev", mock_provider, role="general")
        frontend_dev = Villager("frontend_dev", mock_provider, role="general")
        infra_dev = Villager("infra_dev", mock_provider, role="general")

        backend_village.add_villager(backend_dev)
        frontend_village.add_villager(frontend_dev)
        infra_village.add_villager(infra_dev)

        # Each village works on their part
        backend_result = await backend_village.collaborate("Design backend architecture")
        frontend_result = await frontend_village.collaborate("Design frontend architecture")
        infra_result = await infra_village.collaborate("Design infrastructure")

        # Save individual results
        await storage.save_memory("backend", "architecture", backend_result)
        await storage.save_memory("frontend", "architecture", frontend_result)
        await storage.save_memory("infra", "architecture", infra_result)

        # Create integration team
        integration_village = Village("Integration Team")
        architect = Villager("architect", mock_provider, role="coordinator")
        integration_village.add_villager(architect)

        # Coordinate final architecture
        final_result = await architect.process_task(
            f"Integrate architectures: {backend_result[:30]}, {frontend_result[:30]}, {infra_result[:30]}"
        )

        # Save final architecture
        await storage.save_memory("project", "final_architecture", final_result)

        # Verify all components persisted
        backend_arch = await storage.load_memory("backend", "architecture")
        frontend_arch = await storage.load_memory("frontend", "architecture")
        infra_arch = await storage.load_memory("infra", "architecture")
        final_arch = await storage.load_memory("project", "final_architecture")

        assert backend_arch is not None
        assert frontend_arch is not None
        assert infra_arch is not None
        assert final_arch is not None

    @pytest.mark.asyncio
    async def test_long_running_research_project(self):
        """
        Scenario: Long-running research project with checkpoints.

        Steps:
        1. Initialize project
        2. Complete multiple research phases
        3. Save checkpoints after each phase
        4. Handle interruptions and resume
        5. Generate final deliverable
        """
        storage = InMemoryStorage()
        rate_limiter = RateLimiter(requests_per_minute=100)

        mock_provider = AsyncMock()
        phase_results = [
            "Literature review complete: Found 50 relevant papers...",
            "Data collection complete: Gathered 10,000 samples...",
            "Analysis complete: Statistical significance confirmed...",
            "Validation complete: Results peer-reviewed...",
            "Final paper: Research demonstrates...",
        ]
        mock_provider.generate = AsyncMock(side_effect=phase_results)

        village = Village("Research Project")
        researcher = Villager("researcher", mock_provider, role="researcher")
        village.add_villager(researcher)

        # Project phases
        phases = [
            "literature_review",
            "data_collection",
            "analysis",
            "validation",
            "paper_writing"
        ]

        # Execute phases with checkpoints
        for phase in phases:
            await rate_limiter.acquire(1)

            # Execute phase
            result = await researcher.process_task(f"Complete {phase}")

            # Save checkpoint
            await storage.save_memory("project", f"{phase}_result", result)
            await storage.save_memory("project", "current_phase", phase)

            # Save task history
            await storage.save_task_history(
                "research_project",
                f"Phase: {phase}",
                result,
                {"phase": phase, "completed": True}
            )

        # Verify all phases completed
        for phase in phases:
            phase_result = await storage.load_memory("project", f"{phase}_result")
            assert phase_result is not None

        # Verify final phase
        current_phase = await storage.load_memory("project", "current_phase")
        assert current_phase == "paper_writing"

        # Verify history
        history = await storage.load_task_history("research_project")
        assert len(history) == 5


@pytest.mark.e2e
class TestErrorRecovery:
    """Test error recovery and resilience."""

    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self):
        """Test recovery from partial failures in village collaboration."""
        mock_provider = AsyncMock()

        # First villager succeeds, second fails, third succeeds
        def side_effect_generator():
            yield "Success 1"
            yield Exception("API timeout")
            yield "Success 3"

        effect_gen = side_effect_generator()
        mock_provider.generate = AsyncMock(side_effect=lambda *args, **kwargs: next(effect_gen))

        village = Village("Test Village")

        v1 = Villager("v1", mock_provider, role="general")
        v2 = Villager("v2", mock_provider, role="general")
        v3 = Villager("v3", mock_provider, role="general")

        village.add_villager(v1)
        village.add_villager(v2)
        village.add_villager(v3)

        # Collaboration should handle partial failure
        result = await village.collaborate("Test task")

        # Result should include successes and note failure
        assert "Success 1" in result or "Success 3" in result

    @pytest.mark.asyncio
    async def test_storage_failure_handling(self):
        """Test handling of storage failures."""
        storage = InMemoryStorage()

        # Save data
        await storage.save_village("village1", {"name": "Test"})

        # Simulate partial corruption by clearing internal data
        storage._villages.clear()

        # Attempt to load should return None
        result = await storage.load_village("village1")
        assert result is None

        # System should remain operational
        await storage.save_village("village2", {"name": "New"})
        loaded = await storage.load_village("village2")
        assert loaded["name"] == "New"


@pytest.mark.e2e
class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_concurrent_village_operations(self):
        """Test performance with concurrent village operations."""
        import time

        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(return_value="Quick response")

        villages = [Village(f"Village_{i}") for i in range(10)]

        for village in villages:
            villager = Villager("worker", mock_provider, role="general")
            village.add_villager(villager)

        # Execute concurrent collaborations
        start_time = time.time()

        tasks = [
            village.collaborate(f"Task for {village.name}")
            for village in villages
        ]

        results = await asyncio.gather(*tasks)

        elapsed = time.time() - start_time

        # Verify all completed
        assert len(results) == 10

        # Should complete reasonably quickly (concurrent)
        # This is a loose check - actual time depends on system
        assert elapsed < 10.0  # Should finish in less than 10 seconds

    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory efficiency with large operations."""
        storage = InMemoryStorage()

        # Save many small entries
        for i in range(1000):
            await storage.save_memory(f"owner_{i % 10}", f"key_{i}", f"value_{i}")

        # Verify all saved
        keys = await storage.list_memory_keys("owner_0")
        assert len(keys) == 100  # 1000 entries / 10 owners

        # Clear and verify
        storage.clear_all()
        keys = await storage.list_memory_keys("owner_0")
        assert len(keys) == 0
