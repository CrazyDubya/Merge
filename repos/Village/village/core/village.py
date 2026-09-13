"""Village orchestrator - manages villagers and their interactions."""

import asyncio
from typing import Any, Dict, List, Optional

from village.exceptions import VillageError


class Village:
    """Central orchestrator for managing villager interactions.
    
    The Village class manages a collection of villagers, facilitates their
    communication, and coordinates collaborative tasks.
    """
    
    def __init__(self, name: str = "default") -> None:
        """Initialize a new village.
        
        Args:
            name: The name of the village
        """
        self.name = name
        self.villagers: Dict[str, Any] = {}
        self._task_history: List[Dict[str, Any]] = []
        
    def add_villager(self, villager: Any) -> None:
        """Add a villager to the village.
        
        Args:
            villager: The villager to add
            
        Raises:
            VillageError: If villager name already exists
        """
        if villager.name in self.villagers:
            raise VillageError(f"Villager '{villager.name}' already exists")
        
        self.villagers[villager.name] = villager
        villager.village = self
        
    def remove_villager(self, name: str) -> Optional[Any]:
        """Remove a villager from the village.
        
        Args:
            name: The name of the villager to remove
            
        Returns:
            The removed villager, or None if not found
        """
        villager = self.villagers.pop(name, None)
        if villager:
            villager.village = None
        return villager
        
    def get_villager(self, name: str) -> Optional[Any]:
        """Get a villager by name.
        
        Args:
            name: The name of the villager
            
        Returns:
            The villager if found, None otherwise
        """
        return self.villagers.get(name)
        
    def list_villagers(self) -> List[str]:
        """Get a list of all villager names.
        
        Returns:
            List of villager names
        """
        return list(self.villagers.keys())
        
    async def collaborate(self, task: str) -> str:
        """Execute a collaborative task across all villagers.
        
        Args:
            task: The task description to be executed
            
        Returns:
            The collaborative result as a string
            
        Raises:
            VillageError: If no villagers are available
        """
        if not self.villagers:
            raise VillageError("No villagers available for collaboration")
            
        # Record the task
        task_record = {
            "task": task,
            "villagers": list(self.villagers.keys()),
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Execute task across all villagers
        results = []
        for villager in self.villagers.values():
            try:
                result = await villager.process_task(task)
                results.append(f"{villager.name}: {result}")
            except Exception as e:
                results.append(f"{villager.name}: Error - {str(e)}")
                
        # Combine results
        collaborative_result = "\n".join(results)
        task_record["result"] = collaborative_result
        self._task_history.append(task_record)
        
        return collaborative_result
        
    def get_task_history(self) -> List[Dict[str, Any]]:
        """Get the history of executed tasks.
        
        Returns:
            List of task execution records
        """
        return self._task_history.copy()
        
    def clear_history(self) -> None:
        """Clear the task execution history."""
        self._task_history.clear()
        
    def __len__(self) -> int:
        """Return the number of villagers in the village."""
        return len(self.villagers)
        
    def __repr__(self) -> str:
        """Return string representation of the village."""
        return f"Village(name='{self.name}', villagers={len(self.villagers)})"