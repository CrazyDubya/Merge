"""Village - A LLM Interaction Architecture Framework.

This package provides a modular architecture for creating and managing
LLM-powered agents (villagers) that can collaborate to solve complex tasks.
"""

__version__ = "0.1.0"
__author__ = "CrazyDubya"
__email__ = "support@village-ai.com"

from village.core.village import Village
from village.core.villager import Villager
from village.exceptions import VillageError, VillagerError

__all__ = [
    "Village",
    "Villager", 
    "VillageError",
    "VillagerError",
]