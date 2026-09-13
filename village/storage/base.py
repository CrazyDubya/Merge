"""Base storage interface for Village framework."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime


class BaseStorage(ABC):
    """Abstract base class for storage backends.

    This defines the interface that all storage implementations must follow.
    """

    @abstractmethod
    async def save_village(self, village_id: str, data: Dict[str, Any]) -> bool:
        """Save village data.

        Args:
            village_id: Unique village identifier
            data: Village data to save

        Returns:
            True if successful

        Raises:
            StorageError: If save operation fails
        """
        pass

    @abstractmethod
    async def load_village(self, village_id: str) -> Optional[Dict[str, Any]]:
        """Load village data.

        Args:
            village_id: Unique village identifier

        Returns:
            Village data dictionary or None if not found

        Raises:
            StorageError: If load operation fails
        """
        pass

    @abstractmethod
    async def delete_village(self, village_id: str) -> bool:
        """Delete village data.

        Args:
            village_id: Unique village identifier

        Returns:
            True if successful

        Raises:
            StorageError: If delete operation fails
        """
        pass

    @abstractmethod
    async def list_villages(self) -> List[str]:
        """List all village IDs.

        Returns:
            List of village IDs

        Raises:
            StorageError: If list operation fails
        """
        pass

    @abstractmethod
    async def save_villager(self, villager_id: str, data: Dict[str, Any]) -> bool:
        """Save villager data.

        Args:
            villager_id: Unique villager identifier
            data: Villager data to save

        Returns:
            True if successful

        Raises:
            StorageError: If save operation fails
        """
        pass

    @abstractmethod
    async def load_villager(self, villager_id: str) -> Optional[Dict[str, Any]]:
        """Load villager data.

        Args:
            villager_id: Unique villager identifier

        Returns:
            Villager data dictionary or None if not found

        Raises:
            StorageError: If load operation fails
        """
        pass

    @abstractmethod
    async def save_memory(
        self,
        owner_id: str,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save memory entry.

        Args:
            owner_id: ID of the village or villager owning this memory
            key: Memory key
            value: Memory value
            metadata: Optional metadata for the memory entry

        Returns:
            True if successful

        Raises:
            StorageError: If save operation fails
        """
        pass

    @abstractmethod
    async def load_memory(
        self,
        owner_id: str,
        key: str
    ) -> Optional[Any]:
        """Load memory entry.

        Args:
            owner_id: ID of the village or villager owning this memory
            key: Memory key

        Returns:
            Memory value or None if not found

        Raises:
            StorageError: If load operation fails
        """
        pass

    @abstractmethod
    async def list_memory_keys(self, owner_id: str) -> List[str]:
        """List all memory keys for an owner.

        Args:
            owner_id: ID of the village or villager

        Returns:
            List of memory keys

        Raises:
            StorageError: If list operation fails
        """
        pass

    @abstractmethod
    async def save_task_history(
        self,
        village_id: str,
        task: str,
        result: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save task execution history.

        Args:
            village_id: Village identifier
            task: Task description
            result: Task result
            metadata: Optional metadata (timestamp, participants, etc.)

        Returns:
            True if successful

        Raises:
            StorageError: If save operation fails
        """
        pass

    @abstractmethod
    async def load_task_history(
        self,
        village_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load task execution history.

        Args:
            village_id: Village identifier
            limit: Optional limit on number of records to return

        Returns:
            List of task history records

        Raises:
            StorageError: If load operation fails
        """
        pass

    @abstractmethod
    async def clear_task_history(self, village_id: str) -> bool:
        """Clear task execution history for a village.

        Args:
            village_id: Village identifier

        Returns:
            True if successful

        Raises:
            StorageError: If clear operation fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if storage backend is healthy.

        Returns:
            True if storage is accessible and operational

        Raises:
            StorageError: If health check fails
        """
        pass
