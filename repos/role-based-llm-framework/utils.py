# utils.py
import logging
import yaml
import os
import re
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_configuration() -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable substitution."""
    config_path = Path("config/config.yaml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r") as file:
            config_content = file.read()
            
        # Simple environment variable substitution
        config_content = substitute_env_variables(config_content)
        config = yaml.safe_load(config_content)
        
        # Validate required API keys
        validate_api_keys(config)
        
        return config
    except Exception as e:
        raise Exception(f"Error loading configuration: {str(e)}")

def substitute_env_variables(content: str) -> str:
    """Replace ${VAR} and ${VAR:-default} patterns with environment variables."""
    
    def replace_var(match):
        var_expr = match.group(1)
        if ":-" in var_expr:
            var_name, default_value = var_expr.split(":-", 1)
            return os.getenv(var_name, default_value)
        else:
            return os.getenv(var_expr, "")
    
    # Replace ${VAR} and ${VAR:-default} patterns
    pattern = r'\$\{([^}]+)\}'
    return re.sub(pattern, replace_var, content)

def validate_api_keys(config: Dict[str, Any]) -> None:
    """Validate that at least one API key is configured."""
    api_keys = [
        config.get('anthropic_api_key'),
        config.get('openai_api_key'),
        config.get('deepseek_api_key')
    ]
    
    if not any(key and key.strip() and key != "your_anthropic_api_key_here" for key in api_keys):
        raise ValueError("At least one valid API key must be configured")

def setup_logging(config: Dict[str, Any] = None) -> logging.Logger:
    """Setup structured logging configuration."""
    if config is None:
        config = {}
    
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_file = config.get('logging', {}).get('file', 'logs/app.log')
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def log_message(message: str, level: str = 'INFO') -> None:
    """Log a message with specified level."""
    logger = logging.getLogger(__name__)
    getattr(logger, level.lower())(message)

def handle_error(error: Exception, context: str = "") -> None:
    """Handle errors with proper logging and context."""
    logger = logging.getLogger(__name__)
    error_msg = f"An error occurred"
    if context:
        error_msg += f" in {context}"
    error_msg += f": {error}"
    logger.error(error_msg, exc_info=True)
