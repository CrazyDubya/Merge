"""Structured logging configuration for the Village framework."""

import sys
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.types import FilteringBoundLogger

from village.utils.config import get_config


def configure_logging(
    level: Optional[str] = None,
    format_type: Optional[str] = None,
    output: Optional[str] = None,
    file_path: Optional[str] = None
) -> None:
    """Configure structured logging for the Village framework.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Log format ('structured' or 'simple')
        output: Output destination ('console', 'file', or 'both')
        file_path: Path to log file (if output includes 'file')
    """
    config = get_config()
    
    # Use provided values or fall back to configuration
    level = level or config.get('logging.level', 'INFO')
    format_type = format_type or config.get('logging.format', 'structured')
    output = output or config.get('logging.output', 'console')
    file_path = file_path or config.get('logging.file_path', './logs/village.log')
    
    # Set logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]
    
    if format_type == 'structured':
        processors.extend([
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ])
    else:
        processors.extend([
            structlog.processors.TimeStamper(fmt="[%Y-%m-%d %H:%M:%S]"),
            structlog.dev.ConsoleRenderer(colors=True)
        ])
    
    # Configure handlers
    handlers = []
    
    if output in ('console', 'both'):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        handlers.append(console_handler)
    
    if output in ('file', 'both'):
        # Ensure log directory exists
        log_file = Path(file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout if not handlers else None,
        level=log_level,
        handlers=handlers if handlers else None,
    )


def get_logger(name: str) -> FilteringBoundLogger:
    """Get a logger instance for the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return structlog.get_logger(name)


def log_villager_action(
    logger: FilteringBoundLogger,
    villager_name: str,
    action: str,
    details: Optional[Dict[str, Any]] = None,
    level: str = 'info'
) -> None:
    """Log a villager action with structured data.
    
    Args:
        logger: Logger instance
        villager_name: Name of the villager
        action: Action being performed
        details: Additional details to log
        level: Log level
    """
    log_data = {
        'villager': villager_name,
        'action': action,
        'component': 'villager'
    }
    
    if details:
        log_data.update(details)
    
    getattr(logger, level)("Villager action", **log_data)


def log_village_event(
    logger: FilteringBoundLogger,
    village_name: str,
    event: str,
    details: Optional[Dict[str, Any]] = None,
    level: str = 'info'
) -> None:
    """Log a village event with structured data.
    
    Args:
        logger: Logger instance
        village_name: Name of the village
        event: Event description
        details: Additional details to log
        level: Log level
    """
    log_data = {
        'village': village_name,
        'event': event,
        'component': 'village'
    }
    
    if details:
        log_data.update(details)
    
    getattr(logger, level)("Village event", **log_data)


def log_llm_interaction(
    logger: FilteringBoundLogger,
    provider: str,
    action: str,
    prompt_length: Optional[int] = None,
    response_length: Optional[int] = None,
    duration: Optional[float] = None,
    error: Optional[str] = None,
    level: str = 'info'
) -> None:
    """Log an LLM interaction with structured data.
    
    Args:
        logger: Logger instance
        provider: LLM provider name
        action: Action performed (generate, chat, etc.)
        prompt_length: Length of the prompt in characters
        response_length: Length of the response in characters
        duration: Duration of the interaction in seconds
        error: Error message if interaction failed
        level: Log level
    """
    log_data = {
        'provider': provider,
        'action': action,
        'component': 'llm'
    }
    
    if prompt_length is not None:
        log_data['prompt_length'] = prompt_length
    if response_length is not None:
        log_data['response_length'] = response_length
    if duration is not None:
        log_data['duration_seconds'] = duration
    if error:
        log_data['error'] = error
        level = 'error'
    
    getattr(logger, level)("LLM interaction", **log_data)


def log_performance_metrics(
    logger: FilteringBoundLogger,
    component: str,
    metrics: Dict[str, Any],
    level: str = 'info'
) -> None:
    """Log performance metrics with structured data.
    
    Args:
        logger: Logger instance
        component: Component name (village, villager, llm, etc.)
        metrics: Performance metrics dictionary
        level: Log level
    """
    log_data = {
        'component': component,
        'metrics': metrics,
        'type': 'performance'
    }
    
    getattr(logger, level)("Performance metrics", **log_data)


def log_security_event(
    logger: FilteringBoundLogger,
    event_type: str,
    severity: str,
    details: Dict[str, Any],
    level: str = 'warning'
) -> None:
    """Log a security event with structured data.
    
    Args:
        logger: Logger instance
        event_type: Type of security event
        severity: Severity level (low, medium, high, critical)
        details: Event details
        level: Log level
    """
    log_data = {
        'event_type': event_type,
        'severity': severity,
        'component': 'security',
        **details
    }
    
    getattr(logger, level)("Security event", **log_data)


# Initialize logging on module import
try:
    configure_logging()
except Exception:
    # Fallback to basic logging if configuration fails
    logging.basicConfig(level=logging.INFO)