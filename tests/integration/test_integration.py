"""Integration tests for Village framework."""

import pytest
from unittest.mock import AsyncMock

from village import Village, Villager


class MockLLMProvider:
    """Mock LLM provider for integration testing."""
    
    def __init__(self, name: str = "Mock"):
        self.name = name
        self.call_count = 0
        
    async def generate(self, prompt: str) -> str:
        """Generate a mock response."""
        self.call_count += 1
        return f"[{self.name}] Response {self.call_count}: Generated from prompt length {len(prompt)}"


@pytest.mark.integration
class TestVillageIntegration:
    """Integration tests for Village components working together."""
    
    @pytest.mark.asyncio
    async def test_full_village_workflow(self):
        """Test complete village workflow from creation to collaboration."""
        # Create village
        village = Village("Integration Test Village")
        
        # Create providers
        provider1 = MockLLMProvider("Provider1")
        provider2 = MockLLMProvider("Provider2")
        
        # Create villagers
        analyst = Villager("analyst", provider1, role="analyst")
        researcher = Villager("researcher", provider2, role="researcher")
        coordinator = Villager("coordinator", provider1, role="coordinator")
        
        # Add villagers to village
        village.add_villager(analyst)
        village.add_villager(researcher)
        village.add_villager(coordinator)
        
        assert len(village) == 3
        
        # Test individual task processing
        task1 = "Analyze market data"
        result1 = await analyst.process_task(task1)
        assert "Provider1" in result1
        assert "Generated from prompt length" in result1  # Should include prompt length info
        
        # Test villager communication
        analyst.communicate_with(researcher, "Found interesting trends")
        message = researcher.get_memory("received_from_analyst")
        assert message == "Found interesting trends"
        
        # Test collaborative task
        collab_task = "Create comprehensive market report"
        collab_result = await village.collaborate(collab_task)
        
        # Verify all villagers participated
        assert "analyst:" in collab_result
        assert "researcher:" in collab_result
        assert "coordinator:" in collab_result
        
        # Verify providers were called
        assert provider1.call_count >= 2  # analyst + coordinator
        assert provider2.call_count >= 1  # researcher
        
        # Check task history
        history = village.get_task_history()
        assert len(history) == 1
        assert history[0]["task"] == collab_task
        
    @pytest.mark.asyncio
    async def test_villager_memory_persistence(self):
        """Test that villager memory persists across tasks."""
        provider = MockLLMProvider()
        villager = Villager("memory_test", provider)
        
        # Store initial memory
        villager.set_memory("project", "AI Research")
        villager.set_memory("findings", ["trend1", "trend2"])
        
        # Process multiple tasks
        await villager.process_task("Task 1")
        await villager.process_task("Task 2")
        
        # Memory should persist
        assert villager.get_memory("project") == "AI Research"
        assert villager.get_memory("findings") == ["trend1", "trend2"]
        
        # History should include both tasks
        history = villager.get_conversation_history()
        assert len(history) == 2
        assert history[0]["task"] == "Task 1"
        assert history[1]["task"] == "Task 2"
        
    @pytest.mark.asyncio
    async def test_error_handling_in_collaboration(self):
        """Test error handling during collaborative tasks."""
        village = Village("Error Test Village")
        
        # Create provider that will raise error
        error_provider = AsyncMock()
        error_provider.generate.side_effect = Exception("LLM Error")
        
        normal_provider = MockLLMProvider()
        
        # Add villagers with different providers
        error_villager = Villager("error_villager", error_provider)
        normal_villager = Villager("normal_villager", normal_provider)
        
        village.add_villager(error_villager)
        village.add_villager(normal_villager)
        
        # Collaborate despite one villager failing
        result = await village.collaborate("Test task")
        
        # Should include error message and normal response
        assert "Error - Error processing task: LLM Error" in result
        assert "normal_villager:" in result
        assert "Mock" in result  # From normal provider
        
    @pytest.mark.asyncio
    async def test_village_scaling(self):
        """Test village performance with multiple villagers."""
        village = Village("Scale Test Village")
        providers = [MockLLMProvider(f"Provider{i}") for i in range(5)]
        
        # Add multiple villagers
        for i, provider in enumerate(providers):
            villager = Villager(f"villager_{i}", provider, role=f"role_{i}")
            village.add_villager(villager)
            
        assert len(village) == 5
        
        # Test collaborative task with all villagers
        task = "Scale test task"
        result = await village.collaborate(task)
        
        # Verify all villagers participated
        for i in range(5):
            assert f"villager_{i}:" in result
            
        # Verify all providers were called
        for provider in providers:
            assert provider.call_count == 1