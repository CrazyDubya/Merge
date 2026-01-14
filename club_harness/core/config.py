"""
Configuration management for Club Harness.

Inspired by:
- TinyTroupe: ConfigManager with decorator defaults
- 12-factor: Environment-based configuration
"""

import os
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "openrouter"
    model: str = "meta-llama/llama-3.2-3b-instruct:free"
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class AgentConfig:
    """Agent behavior configuration."""
    max_turns: int = 20
    max_tool_calls_per_turn: int = 5
    context_window_limit: int = 32000
    context_utilization_target: float = 0.4  # 12-factor: stay under 40%
    enable_memory_consolidation: bool = True
    enable_self_evaluation: bool = False


@dataclass
class ExecutionConfig:
    """Execution environment configuration."""
    sandbox_enabled: bool = False
    sandbox_type: str = "process"  # process, container, vm
    timeout_seconds: int = 120
    confirm_dangerous_tools: bool = True


@dataclass
class Config:
    """
    Central configuration for Club Harness.

    Supports:
    - Environment variable overrides
    - Programmatic updates
    - Decorator-based defaults (TinyTroupe pattern)
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Model tiers for cost-aware routing (hivey-inspired)
    # Updated Jan 2026 with currently available free models
    model_tiers: Dict[str, List[str]] = field(default_factory=lambda: {
        "free": [
            "meta-llama/llama-3.2-3b-instruct:free",
            "qwen/qwen3-coder:free",
            "google/gemma-3n-e2b-it:free",
            "moonshotai/kimi-k2:free",
            "nvidia/nemotron-nano-9b-v2:free",
        ],
        "cheap": [
            "google/gemini-2.0-flash-001",
            "anthropic/claude-3-haiku",
            "openai/gpt-4o-mini",
        ],
        "standard": [
            "google/gemini-2.0-pro",
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
        ],
        "reasoning": [
            "openai/o1-mini",
            "deepseek/deepseek-r1",
        ],
        "advanced": [
            "anthropic/claude-3-opus",
            "openai/gpt-4-turbo",
        ],
    })

    def __post_init__(self):
        """Load from environment variables."""
        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # LLM settings
        if api_key := os.getenv("OPENROUTER_API_KEY"):
            self.llm.api_key = api_key
            self.llm.provider = "openrouter"
            self.llm.base_url = "https://openrouter.ai/api/v1"

        if model := os.getenv("CLUB_MODEL"):
            self.llm.model = model

        if temp := os.getenv("CLUB_TEMPERATURE"):
            self.llm.temperature = float(temp)

        # Agent settings
        if max_turns := os.getenv("CLUB_MAX_TURNS"):
            self.agent.max_turns = int(max_turns)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by dot-notation key."""
        parts = key.split(".")
        obj = self
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return default
        return obj

    def update(self, key: str, value: Any) -> None:
        """Update a config value by dot-notation key."""
        parts = key.split(".")
        obj = self
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def get_model_for_tier(self, tier: str) -> str:
        """Get a model from a specific tier."""
        models = self.model_tiers.get(tier, self.model_tiers["free"])
        return models[0] if models else self.llm.model


def config_defaults(**defaults: str) -> Callable:
    """
    Decorator for applying config defaults to function parameters.

    Inspired by TinyTroupe's config_manager.config_defaults.

    Usage:
        @config_defaults(model="llm.model", temperature="llm.temperature")
        def my_function(prompt, model=None, temperature=None):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for param, config_key in defaults.items():
                if kwargs.get(param) is None:
                    kwargs[param] = config.get(config_key)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Global config instance
config = Config()
