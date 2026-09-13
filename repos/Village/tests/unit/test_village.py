"""Unit tests for Village class."""

import pytest
from unittest.mock import AsyncMock

from village.core.village import Village
from village.core.villager import Villager
from village.exceptions import VillageError


class TestVillage:
    """Test cases for the Village class."""
    
    def test_village_creation(self):
        """Test village can be created with default name."""
        village = Village()
        assert village.name == "default"
        assert len(village.villagers) == 0
        
    def test_village_creation_with_name(self):
        """Test village can be created with custom name."""
        village = Village("custom_village")
        assert village.name == "custom_village"
        
    def test_add_villager(self, village, villager):
        """Test adding a villager to the village."""
        village.add_villager(villager)
        assert len(village) == 1
        assert villager.name in village.villagers
        assert village.get_villager(villager.name) == villager
        assert villager.village == village
        
    def test_add_duplicate_villager_raises_error(self, village, mock_llm_provider):
        """Test adding villager with duplicate name raises error."""
        villager1 = Villager("test", mock_llm_provider)
        villager2 = Villager("test", mock_llm_provider)
        
        village.add_villager(villager1)
        
        with pytest.raises(VillageError, match="already exists"):
            village.add_villager(villager2)
            
    def test_remove_villager(self, village, villager):
        """Test removing a villager from the village."""
        village.add_villager(villager)
        removed = village.remove_villager(villager.name)
        
        assert removed == villager
        assert len(village) == 0
        assert villager.village is None
        
    def test_remove_nonexistent_villager(self, village):
        """Test removing non-existent villager returns None."""
        result = village.remove_villager("nonexistent")
        assert result is None
        
    def test_get_villager(self, village, villager):
        """Test getting villager by name."""
        village.add_villager(villager)
        retrieved = village.get_villager(villager.name)
        assert retrieved == villager
        
    def test_get_nonexistent_villager(self, village):
        """Test getting non-existent villager returns None."""
        result = village.get_villager("nonexistent")
        assert result is None
        
    def test_list_villagers(self, village_with_villagers):
        """Test listing all villager names."""
        names = village_with_villagers.list_villagers()
        assert "analyst" in names
        assert "researcher" in names
        assert len(names) == 2
        
    @pytest.mark.asyncio
    async def test_collaborate_success(self, village_with_villagers):
        """Test successful collaboration."""
        task = "Analyze market trends"
        result = await village_with_villagers.collaborate(task)
        
        assert "analyst: Mock LLM response" in result
        assert "researcher: Mock LLM response" in result
        
        # Check task history
        history = village_with_villagers.get_task_history()
        assert len(history) == 1
        assert history[0]["task"] == task
        
    @pytest.mark.asyncio
    async def test_collaborate_empty_village_raises_error(self, village):
        """Test collaboration with empty village raises error."""
        with pytest.raises(VillageError, match="No villagers available"):
            await village.collaborate("test task")
            
    def test_clear_history(self, village_with_villagers):
        """Test clearing task history."""
        # Add some history first
        village_with_villagers._task_history.append({"test": "data"})
        
        village_with_villagers.clear_history()
        assert len(village_with_villagers.get_task_history()) == 0
        
    def test_village_length(self, village_with_villagers):
        """Test village length returns number of villagers."""
        assert len(village_with_villagers) == 2
        
    def test_village_repr(self, village):
        """Test village string representation."""
        repr_str = repr(village)
        assert "Village(name='test_village', villagers=0)" == repr_str