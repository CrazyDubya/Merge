"""Unit tests for Villager class."""

import pytest
from unittest.mock import AsyncMock, Mock

from village.core.villager import Villager
from village.exceptions import VillagerError


class TestVillager:
    """Test cases for the Villager class."""
    
    def test_villager_creation_default(self):
        """Test villager creation with default parameters."""
        villager = Villager("test")
        assert villager.name == "test"
        assert villager.role == "general"
        assert villager.village is None
        assert len(villager._memory) == 0
        
    def test_villager_creation_with_params(self, mock_llm_provider):
        """Test villager creation with custom parameters."""
        custom_prompt = "You are a specialist."
        villager = Villager(
            "specialist", 
            mock_llm_provider, 
            role="expert",
            system_prompt=custom_prompt
        )
        
        assert villager.name == "specialist"
        assert villager.role == "expert"
        assert villager.llm_provider == mock_llm_provider
        assert villager.system_prompt == custom_prompt
        
    def test_default_system_prompts(self):
        """Test default system prompts for different roles."""
        analyst = Villager("analyst", role="analyst")
        assert "data analyst" in analyst.system_prompt.lower()
        
        researcher = Villager("researcher", role="researcher")
        assert "researcher" in researcher.system_prompt.lower()
        
        general = Villager("general", role="general")
        assert "helpful assistant" in general.system_prompt.lower()
        
    @pytest.mark.asyncio
    async def test_process_task_success(self, villager):
        """Test successful task processing."""
        task = "Analyze this data"
        result = await villager.process_task(task)
        
        assert result == "Mock LLM response"
        assert len(villager.get_conversation_history()) == 1
        
        history = villager.get_conversation_history()[0]
        assert history["task"] == task
        assert history["response"] == result
        
    @pytest.mark.asyncio
    async def test_process_task_no_provider_raises_error(self):
        """Test processing task without LLM provider raises error."""
        villager = Villager("test")
        
        with pytest.raises(VillagerError, match="no LLM provider"):
            await villager.process_task("test task")
            
    @pytest.mark.asyncio 
    async def test_process_task_with_history_context(self, villager):
        """Test task processing includes conversation history."""
        # Add some history
        villager._conversation_history = [
            {"task": "Previous task", "response": "Previous response", "timestamp": 123}
        ]
        
        task = "New task"
        await villager.process_task(task)
        
        # Verify LLM was called with context
        villager.llm_provider.generate.assert_called_once()
        call_args = villager.llm_provider.generate.call_args[0][0]
        assert "Previous task" in call_args
        assert "New task" in call_args
        
    def test_memory_operations(self, villager):
        """Test memory storage and retrieval."""
        # Test setting and getting memory
        villager.set_memory("key1", "value1")
        villager.set_memory("key2", {"nested": "data"})
        
        assert villager.get_memory("key1") == "value1"
        assert villager.get_memory("key2") == {"nested": "data"}
        assert villager.get_memory("nonexistent") is None
        assert villager.get_memory("nonexistent", "default") == "default"
        
    def test_clear_memory(self, villager):
        """Test clearing memory."""
        villager.set_memory("test", "data")
        villager.clear_memory()
        assert len(villager._memory) == 0
        
    def test_clear_history(self, villager):
        """Test clearing conversation history."""
        villager._conversation_history = [{"test": "data"}]
        villager.clear_history()
        assert len(villager.get_conversation_history()) == 0
        
    def test_communicate_with(self, mock_llm_provider):
        """Test communication between villagers."""
        villager1 = Villager("villager1", mock_llm_provider)
        villager2 = Villager("villager2", mock_llm_provider)
        
        message = "Hello from villager1"
        villager1.communicate_with(villager2, message)
        
        # Check message is stored in both villagers' memory
        assert villager1.get_memory("sent_to_villager2") == message
        assert villager2.get_memory("received_from_villager1") == message
        
    @pytest.mark.asyncio
    async def test_call_llm_with_generate_method(self, villager):
        """Test LLM calling with generate method."""
        result = await villager._call_llm("test prompt")
        assert result == "Mock LLM response"
        villager.llm_provider.generate.assert_called_once_with("test prompt")
        
    @pytest.mark.asyncio
    async def test_call_llm_with_chat_method(self, mock_llm_provider):
        """Test LLM calling with chat method."""
        # Mock provider with only chat method
        mock_llm_provider.generate = None
        villager = Villager("test", mock_llm_provider)
        
        result = await villager._call_llm("test prompt")
        assert result == "Mock chat response"
        
    @pytest.mark.asyncio
    async def test_call_llm_fallback(self):
        """Test LLM calling fallback for unsupported providers."""
        mock_provider = Mock()
        # Remove async methods
        del mock_provider.generate
        del mock_provider.chat
        
        villager = Villager("test", mock_provider)
        result = await villager._call_llm("test prompt")
        assert str(mock_provider) in result
        
    def test_villager_repr(self, villager):
        """Test villager string representation."""
        repr_str = repr(villager)
        assert "Villager(name='test_villager', role='analyst')" == repr_str