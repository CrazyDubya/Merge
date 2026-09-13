"""Storage abstractions for persistent data."""

from village.storage.base import BaseStorage
from village.storage.memory import InMemoryStorage

__all__ = ["BaseStorage", "InMemoryStorage"]

try:
    from village.storage.postgres import PostgreSQLStorage
    __all__.append("PostgreSQLStorage")
except ImportError:
    pass
