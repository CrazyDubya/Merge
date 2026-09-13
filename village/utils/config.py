"""Configuration management for the Village framework."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union

from village.exceptions import ConfigurationError


class Config:
    """Configuration manager for Village framework.
    
    Handles loading and accessing configuration from various sources:
    - Default configuration files
    - Environment variables
    - Runtime overrides
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self._config: Dict[str, Any] = {}
        self._load_default_config()
        
        if config_path:
            self._load_config_file(config_path)
            
        self._load_environment_variables()
        
    def _load_default_config(self) -> None:
        """Load default configuration."""
        default_config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
        if default_config_path.exists():
            self._load_config_file(default_config_path)
            
    def _load_config_file(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Raises:
            ConfigurationError: If file cannot be loaded
        """
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self._merge_config(file_config)
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {config_path}: {e}")
            
    def _load_environment_variables(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            'VILLAGE_NAME': ['village', 'name'],
            'VILLAGE_MAX_VILLAGERS': ['village', 'max_villagers'],
            'LLM_DEFAULT_PROVIDER': ['llm', 'default_provider'],
            'LLM_TIMEOUT': ['llm', 'timeout'],
            'LLM_MAX_TOKENS': ['llm', 'generation', 'max_tokens'],
            'LLM_TEMPERATURE': ['llm', 'generation', 'temperature'],
            'MEMORY_MAX_HISTORY': ['memory', 'max_history_length'],
            'LOGGING_LEVEL': ['logging', 'level'],
            'DEBUG_MODE': ['development', 'debug_mode'],
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(config_path, self._parse_env_value(value))
                
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
            
        # Try to parse as boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
            
        # Return as string
        return value
        
    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """Merge new configuration with existing configuration."""
        self._deep_merge(self._config, new_config)
        
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Deep merge two dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
                
    def _set_nested_value(self, path: list, value: Any) -> None:
        """Set a nested configuration value."""
        current = self._config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key.
        
        Args:
            key: Dot-separated configuration key (e.g., 'village.name')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        current = self._config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
            
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dot-separated key.
        
        Args:
            key: Dot-separated configuration key
            value: Value to set
        """
        keys = key.split('.')
        self._set_nested_value(keys, value)
        
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Configuration section as dictionary
        """
        return self.get(section, {})
        
    def to_dict(self) -> Dict[str, Any]:
        """Get entire configuration as dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()
        
    def validate(self) -> bool:
        """Validate configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        required_sections = ['village', 'llm', 'memory', 'logging']
        
        for section in required_sections:
            if section not in self._config:
                raise ConfigurationError(f"Missing required configuration section: {section}")
                
        # Validate specific values
        max_villagers = self.get('village.max_villagers', 0)
        if not isinstance(max_villagers, int) or max_villagers <= 0:
            raise ConfigurationError("village.max_villagers must be a positive integer")
            
        timeout = self.get('llm.timeout', 0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ConfigurationError("llm.timeout must be a positive number")
            
        return True


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config