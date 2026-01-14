"""
Error handling utilities for Club Harness.

Adapted from hivey's error handling patterns.
Provides standardized error handling, custom exceptions, and decorators.
"""

import functools
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)


@dataclass
class ErrorDetails:
    """Contains detailed information about an error."""

    error_type: str
    message: str
    details: Optional[str] = None
    traceback_str: Optional[str] = None
    context: Optional[dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "details": self.details,
            "traceback": self.traceback_str,
            "context": self.context,
        }


class ClubHarnessError(Exception):
    """Base exception for all Club Harness errors."""

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        context: Optional[dict] = None,
    ):
        self.message = message
        self.details = details
        self.context = context or {}
        super().__init__(message)

    def to_dict(self) -> dict:
        """Convert exception to dictionary format."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "context": self.context,
        }

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class ConfigurationError(ClubHarnessError):
    """Raised when there's a configuration-related error."""
    pass


class LLMError(ClubHarnessError):
    """Raised when there's an LLM-related error."""
    pass


class RateLimitError(LLMError):
    """Raised when rate limited by the LLM provider."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ModelNotAvailableError(LLMError):
    """Raised when a requested model is not available."""
    pass


class ToolError(ClubHarnessError):
    """Raised when there's a tool execution error."""
    pass


class ValidationError(ClubHarnessError):
    """Raised when input validation fails."""
    pass


class MemoryError(ClubHarnessError):
    """Raised when there's a memory system error."""
    pass


class PlanningError(ClubHarnessError):
    """Raised when planning fails."""
    pass


class LoopDetectedError(ClubHarnessError):
    """Raised when an agent loop is detected."""

    def __init__(
        self,
        message: str = "Agent loop detected",
        loop_type: str = "",
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.loop_type = loop_type


def safe_execute(
    operation: Callable,
    *args,
    error_message: str = "Operation failed",
    return_on_error: Any = None,
    reraise: bool = False,
    log_error: bool = True,
    **kwargs,
) -> Any:
    """
    Safely execute an operation with standardized error handling.

    Args:
        operation: The function/method to execute
        *args: Positional arguments for the operation
        error_message: Custom error message prefix
        return_on_error: Value to return if operation fails
        reraise: Whether to re-raise the exception after logging
        log_error: Whether to log the error
        **kwargs: Keyword arguments for the operation

    Returns:
        Operation result or return_on_error value
    """
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        if log_error:
            error_details = ErrorDetails(
                error_type=type(e).__name__,
                message=str(e),
                details=error_message,
                traceback_str=traceback.format_exc(),
                context={"args": str(args)[:200], "kwargs": str(kwargs)[:200]},
            )
            logger.error(
                f"{error_message}: {error_details.error_type} - {error_details.message}"
            )

        if reraise:
            raise

        return return_on_error


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
):
    """
    Decorator to retry a function on failure with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by on each retry
        exceptions: Tuple of exception types to catch and retry on
        on_retry: Optional callback called on each retry (attempt, exception)

    Example:
        @retry_on_failure(max_attempts=3, delay=2.0)
        def fetch_data():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} "
                            f"attempts: {e}"
                        )
                        raise

                    logger.warning(
                        f"Function {func.__name__} failed on attempt "
                        f"{attempt + 1}/{max_attempts}: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )

                    if on_retry:
                        on_retry(attempt + 1, e)

                    time.sleep(current_delay)
                    current_delay *= backoff_factor

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def validate_input(
    validator: Callable[[Any], bool],
    error_message: str = "Input validation failed",
):
    """
    Decorator to validate function inputs.

    Args:
        validator: Function that takes the input and returns True if valid
        error_message: Error message to use if validation fails

    Example:
        @validate_input(lambda x: x is not None, "Input cannot be None")
        def process(data):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i, arg in enumerate(args):
                if not validator(arg):
                    raise ValidationError(
                        error_message,
                        details=f"Invalid argument at position {i}",
                        context={"function": func.__name__},
                    )

            for key, value in kwargs.items():
                if not validator(value):
                    raise ValidationError(
                        error_message,
                        details=f"Invalid keyword argument: {key}",
                        context={"function": func.__name__},
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def handle_llm_errors(func: Callable) -> Callable:
    """
    Decorator to standardize LLM error handling.

    Converts common HTTP errors to LLMError subtypes.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()

            # Check for rate limiting
            if "429" in error_str or "rate" in error_str or "too many" in error_str:
                raise RateLimitError(
                    f"Rate limit in {func.__name__}",
                    details=str(e),
                    context={"function": func.__name__},
                )

            # Check for model not found
            if "404" in error_str or "not found" in error_str:
                raise ModelNotAvailableError(
                    f"Model not available in {func.__name__}",
                    details=str(e),
                    context={"function": func.__name__},
                )

            # Generic LLM error
            raise LLMError(
                f"LLM error in {func.__name__}",
                details=str(e),
                context={"function": func.__name__},
            )

    return wrapper


# Type alias for error handlers
ErrorHandler = Callable[[Exception], Any]


class ErrorBoundary:
    """
    Context manager for error boundaries with recovery.

    Example:
        with ErrorBoundary(fallback=default_value, on_error=log_error):
            risky_operation()
    """

    def __init__(
        self,
        fallback: Any = None,
        on_error: Optional[ErrorHandler] = None,
        reraise: bool = False,
    ):
        self.fallback = fallback
        self.on_error = on_error
        self.reraise = reraise
        self.exception: Optional[Exception] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.exception = exc_val

            if self.on_error:
                self.on_error(exc_val)

            if self.reraise:
                return False  # Re-raise

            return True  # Suppress

        return False
