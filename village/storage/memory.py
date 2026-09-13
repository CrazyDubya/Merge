"""In-memory storage implementation."""

from typing import Any, Dict, List, Optional
from datetime import datetime
import copy

from village.storage.base import BaseStorage


class InMemoryStorage(BaseStorage):
    """In-memory storage backend.

    Warning: Data is not persisted and will be lost when the process ends.
    Use this for development, testing, or when persistence is not required.
    """

    def __init__(self) -> None:
        """Initialize in-memory storage."""
        self._villages: Dict[str, Dict[str, Any]] = {}
        self._villagers: Dict[str, Dict[str, Any]] = {}
        self._memory: Dict[str, Dict[str, Any]] = {}
        self._task_history: Dict[str, List[Dict[str, Any]]] = {}

    async def save_village(self, village_id: str, data: Dict[str, Any]) -> bool:
        """Save village data."""
        self._villages[village_id] = copy.deepcopy(data)
        return True

    async def load_village(self, village_id: str) -> Optional[Dict[str, Any]]:
        """Load village data."""
        data = self._villages.get(village_id)
        return copy.deepcopy(data) if data else None

    async def delete_village(self, village_id: str) -> bool:
        """Delete village data."""
        if village_id in self._villages:
            del self._villages[village_id]
            # Also clean up related data
            if village_id in self._task_history:
                del self._task_history[village_id]
            return True
        return False

    async def list_villages(self) -> List[str]:
        """List all village IDs."""
        return list(self._villages.keys())

    async def save_villager(self, villager_id: str, data: Dict[str, Any]) -> bool:
        """Save villager data."""
        self._villagers[villager_id] = copy.deepcopy(data)
        return True

    async def load_villager(self, villager_id: str) -> Optional[Dict[str, Any]]:
        """Load villager data."""
        data = self._villagers.get(villager_id)
        return copy.deepcopy(data) if data else None

    async def save_memory(
        self,
        owner_id: str,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save memory entry."""
        if owner_id not in self._memory:
            self._memory[owner_id] = {}

        self._memory[owner_id][key] = {
            "value": copy.deepcopy(value),
            "metadata": copy.deepcopy(metadata) if metadata else {},
            "updated_at": datetime.utcnow().isoformat()
        }
        return True

    async def load_memory(self, owner_id: str, key: str) -> Optional[Any]:
        """Load memory entry."""
        if owner_id in self._memory and key in self._memory[owner_id]:
            entry = self._memory[owner_id][key]
            return copy.deepcopy(entry["value"])
        return None

    async def list_memory_keys(self, owner_id: str) -> List[str]:
        """List all memory keys for an owner."""
        if owner_id in self._memory:
            return list(self._memory[owner_id].keys())
        return []

    async def save_task_history(
        self,
        village_id: str,
        task: str,
        result: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save task execution history."""
        if village_id not in self._task_history:
            self._task_history[village_id] = []

        entry = {
            "task": task,
            "result": result,
            "metadata": copy.deepcopy(metadata) if metadata else {},
            "timestamp": datetime.utcnow().isoformat()
        }

        self._task_history[village_id].append(entry)
        return True

    async def load_task_history(
        self,
        village_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load task execution history."""
        if village_id not in self._task_history:
            return []

        history = self._task_history[village_id]
        if limit:
            history = history[-limit:]

        return copy.deepcopy(history)

    async def clear_task_history(self, village_id: str) -> bool:
        """Clear task execution history for a village."""
        if village_id in self._task_history:
            self._task_history[village_id] = []
        return True

    async def health_check(self) -> bool:
        """Check if storage backend is healthy."""
        # In-memory storage is always healthy if initialized
        return True

    def clear_all(self) -> None:
        """Clear all data (useful for testing)."""
        self._villages.clear()
        self._villagers.clear()
        self._memory.clear()
        self._task_history.clear()
