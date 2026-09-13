"""Custom exceptions for the Village framework."""


class VillageError(Exception):
    """Base exception for all village-related errors."""
    pass


class VillagerError(VillageError):
    """Exception raised for villager-specific errors."""
    pass


class LLMProviderError(VillageError):
    """Exception raised for LLM provider-related errors."""
    pass


class CommunicationError(VillageError):
    """Exception raised for inter-villager communication errors."""
    pass


class MemoryError(VillageError):
    """Exception raised for memory management errors."""
    pass


class ConfigurationError(VillageError):
    """Exception raised for configuration-related errors."""
    pass


class SecurityError(VillageError):
    """Exception raised for security-related issues."""
    pass