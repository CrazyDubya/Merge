"""PostgreSQL storage implementation."""

import json
import os
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

from village.storage.base import BaseStorage
from village.exceptions import MemoryError as VillageMemoryError


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
