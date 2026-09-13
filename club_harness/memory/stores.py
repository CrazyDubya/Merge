"""Storage backends ported from Village.

Combines Village's storage/base.py (BaseStorage), storage/memory.py
(InMemoryStorage) and storage/postgres.py (PostgresStorage). The postgres
backend imports asyncpg lazily; the in-memory backend has no extra deps.

Ported from Village (repos/Village) via git subtree merge; see docs/PROVENANCE.md.

Five-store taxonomy (from CascadeProjects, design note only):
    task        - working memory for the current task
    recent      - short-term conversational context
    acquired    - facts learned during the session
    long-term   - durable knowledge across sessions
    speculative - hypotheses / what-if branches not yet verified
These map onto BaseStorage implementations; see docs/memory-taxonomy.md.
"""


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


from typing import Any, Dict, List, Optional
from datetime import datetime
import copy



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


import json
import os
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

from club_harness.core.errors import MemoryError as VillageMemoryError


class PostgreSQLStorage(BaseStorage):
    """PostgreSQL storage backend for persistent data.

    Attributes:
        connection_string: PostgreSQL connection string
        pool: Connection pool for database operations
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        min_pool_size: int = 5,
        max_pool_size: int = 20,
        **kwargs: Any
    ) -> None:
        """Initialize PostgreSQL storage.

        Args:
            connection_string: PostgreSQL connection string (defaults to DATABASE_URL env var)
            min_pool_size: Minimum connection pool size
            max_pool_size: Maximum connection pool size
            **kwargs: Additional connection parameters

        Raises:
            VillageMemoryError: If asyncpg is not installed or connection fails
        """
        if not ASYNCPG_AVAILABLE:
            raise VillageMemoryError(
                "asyncpg package not installed. Install with: pip install asyncpg>=0.28.0"
            )

        self.connection_string = connection_string or os.getenv("DATABASE_URL")
        if not self.connection_string:
            raise VillageMemoryError(
                "PostgreSQL connection string required. Set DATABASE_URL environment variable "
                "or pass connection_string parameter."
            )

        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.pool_kwargs = kwargs
        self._pool: Optional[asyncpg.Pool] = None

    async def initialize(self) -> None:
        """Initialize database connection pool and create tables.

        Raises:
            VillageMemoryError: If initialization fails
        """
        try:
            # Create connection pool
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
                **self.pool_kwargs
            )

            # Create tables
            await self._create_tables()

        except Exception as e:
            raise VillageMemoryError(f"Failed to initialize PostgreSQL storage: {str(e)}") from e

    async def close(self) -> None:
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def _create_tables(self) -> None:
        """Create required database tables."""
        if not self._pool:
            raise VillageMemoryError("Storage not initialized. Call initialize() first.")

        async with self._pool.acquire() as conn:
            # Villages table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS villages (
                    id TEXT PRIMARY KEY,
                    data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Villagers table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS villagers (
                    id TEXT PRIMARY KEY,
                    data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Memory table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    owner_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value JSONB NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (owner_id, key)
                )
            """)

            # Create index for memory lookups
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_owner
                ON memory(owner_id)
            """)

            # Task history table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS task_history (
                    id SERIAL PRIMARY KEY,
                    village_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    result TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Create index for task history lookups
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_history_village
                ON task_history(village_id, created_at DESC)
            """)

    async def save_village(self, village_id: str, data: Dict[str, Any]) -> bool:
        """Save village data."""
        if not self._pool:
            raise VillageMemoryError("Storage not initialized. Call initialize() first.")

        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO villages (id, data, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (id)
                DO UPDATE SET data = $2, updated_at = NOW()
            """, village_id, json.dumps(data))

        return True

    async def load_village(self, village_id: str) -> Optional[Dict[str, Any]]:
        """Load village data."""
        if not self._pool:
            raise VillageMemoryError("Storage not initialized. Call initialize() first.")

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM villages WHERE id = $1",
                village_id
            )

        return json.loads(row["data"]) if row else None

    async def delete_village(self, village_id: str) -> bool:
        """Delete village data."""
        if not self._pool:
            raise VillageMemoryError("Storage not initialized. Call initialize() first.")

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM villages WHERE id = $1",
                village_id
            )

        return result != "DELETE 0"

    async def list_villages(self) -> List[str]:
        """List all village IDs."""
        if not self._pool:
            raise VillageMemoryError("Storage not initialized. Call initialize() first.")

        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT id FROM villages ORDER BY created_at")

        return [row["id"] for row in rows]

    async def save_villager(self, villager_id: str, data: Dict[str, Any]) -> bool:
        """Save villager data."""
        if not self._pool:
            raise VillageMemoryError("Storage not initialized. Call initialize() first.")

        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO villagers (id, data, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (id)
                DO UPDATE SET data = $2, updated_at = NOW()
            """, villager_id, json.dumps(data))

        return True

    async def load_villager(self, villager_id: str) -> Optional[Dict[str, Any]]:
        """Load villager data."""
        if not self._pool:
            raise VillageMemoryError("Storage not initialized. Call initialize() first.")

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM villagers WHERE id = $1",
                villager_id
            )

        return json.loads(row["data"]) if row else None

    async def save_memory(
        self,
        owner_id: str,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save memory entry."""
        if not self._pool:
            raise VillageMemoryError("Storage not initialized. Call initialize() first.")

        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO memory (owner_id, key, value, metadata, updated_at)
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (owner_id, key)
                DO UPDATE SET value = $3, metadata = $4, updated_at = NOW()
            """, owner_id, key, json.dumps(value), json.dumps(metadata or {}))

        return True

    async def load_memory(self, owner_id: str, key: str) -> Optional[Any]:
        """Load memory entry."""
        if not self._pool:
            raise VillageMemoryError("Storage not initialized. Call initialize() first.")

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value FROM memory WHERE owner_id = $1 AND key = $2",
                owner_id, key
            )

        return json.loads(row["value"]) if row else None

    async def list_memory_keys(self, owner_id: str) -> List[str]:
        """List all memory keys for an owner."""
        if not self._pool:
            raise VillageMemoryError("Storage not initialized. Call initialize() first.")

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key FROM memory WHERE owner_id = $1 ORDER BY updated_at DESC",
                owner_id
            )

        return [row["key"] for row in rows]

    async def save_task_history(
        self,
        village_id: str,
        task: str,
        result: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save task execution history."""
        if not self._pool:
            raise VillageMemoryError("Storage not initialized. Call initialize() first.")

        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO task_history (village_id, task, result, metadata)
                VALUES ($1, $2, $3, $4)
            """, village_id, task, result, json.dumps(metadata or {}))

        return True

    async def load_task_history(
        self,
        village_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load task execution history."""
        if not self._pool:
            raise VillageMemoryError("Storage not initialized. Call initialize() first.")

        async with self._pool.acquire() as conn:
            if limit:
                rows = await conn.fetch("""
                    SELECT task, result, metadata, created_at
                    FROM task_history
                    WHERE village_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                """, village_id, limit)
            else:
                rows = await conn.fetch("""
                    SELECT task, result, metadata, created_at
                    FROM task_history
                    WHERE village_id = $1
                    ORDER BY created_at DESC
                """, village_id)

        return [
            {
                "task": row["task"],
                "result": row["result"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "timestamp": row["created_at"].isoformat()
            }
            for row in rows
        ]

    async def clear_task_history(self, village_id: str) -> bool:
        """Clear task execution history for a village."""
        if not self._pool:
            raise VillageMemoryError("Storage not initialized. Call initialize() first.")

        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM task_history WHERE village_id = $1",
                village_id
            )

        return True

    async def health_check(self) -> bool:
        """Check if storage backend is healthy."""
        if not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False
