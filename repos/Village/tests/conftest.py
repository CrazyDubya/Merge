"""Test configuration and fixtures."""

import pytest
from unittest.mock import AsyncMock, Mock

from village.core.village import Village
from village.core.villager import Villager


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    provider = Mock()
    provider.generate = AsyncMock(return_value="Mock LLM response")
    provider.chat = AsyncMock(return_value="Mock chat response")
    return provider


@pytest.fixture
def village():
    """Create a test village."""
    return Village("test_village")


@pytest.fixture
def villager(mock_llm_provider):
    """Create a test villager."""
    return Villager("test_villager", mock_llm_provider, role="analyst")


@pytest.fixture
def village_with_villagers(village, mock_llm_provider):
    """Create a village with multiple villagers."""
    villager1 = Villager("analyst", mock_llm_provider, role="analyst")
    villager2 = Villager("researcher", mock_llm_provider, role="researcher")
    
    village.add_villager(villager1)
    village.add_villager(villager2)
    
    return village