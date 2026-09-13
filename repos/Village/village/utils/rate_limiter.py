"""Rate limiting utilities for API calls and resource management."""

import time
import asyncio
from typing import Dict, Optional
from collections import deque
from datetime import datetime, timedelta


class RateLimiter:
    """Rate limiter using token bucket algorithm.

    Attributes:
        requests_per_minute: Maximum requests allowed per minute
        burst_size: Maximum burst size (default: requests_per_minute)
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: Optional[int] = None
    ) -> None:
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            burst_size: Maximum burst size (defaults to requests_per_minute)
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or requests_per_minute

        # Token bucket parameters
        self.tokens = float(self.burst_size)
        self.max_tokens = float(self.burst_size)
        self.refill_rate = requests_per_minute / 60.0  # Tokens per second
        self.last_refill = time.monotonic()

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens from the bucket, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Time waited in seconds

        Raises:
            ValueError: If requested tokens exceed burst size
        """
        if tokens > self.burst_size:
            raise ValueError(
                f"Requested tokens ({tokens}) exceeds burst size ({self.burst_size})"
            )

        async with self._lock:
            wait_time = 0.0

            while True:
                # Refill tokens based on elapsed time
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(
                    self.max_tokens,
                    self.tokens + (elapsed * self.refill_rate)
                )
                self.last_refill = now

                # Check if we have enough tokens
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return wait_time

                # Calculate wait time for required tokens
                tokens_needed = tokens - self.tokens
                sleep_time = tokens_needed / self.refill_rate

                # Wait for tokens to be available
                await asyncio.sleep(sleep_time)
                wait_time += sleep_time

    async def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        async with self._lock:
            # Refill tokens
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.max_tokens,
                self.tokens + (elapsed * self.refill_rate)
            )
            self.last_refill = now

            # Try to acquire
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    def get_tokens_available(self) -> float:
        """Get current number of available tokens.

        Returns:
            Number of available tokens
        """
        now = time.monotonic()
        elapsed = now - self.last_refill
        return min(
            self.max_tokens,
            self.tokens + (elapsed * self.refill_rate)
        )


class QuotaManager:
    """Quota manager for tracking resource usage.

    Tracks usage across different time windows (hour, day, month).
    """

    def __init__(
        self,
        hourly_quota: Optional[int] = None,
        daily_quota: Optional[int] = None,
        monthly_quota: Optional[int] = None
    ) -> None:
        """Initialize quota manager.

        Args:
            hourly_quota: Maximum requests per hour
            daily_quota: Maximum requests per day
            monthly_quota: Maximum requests per month
        """
        self.hourly_quota = hourly_quota
        self.daily_quota = daily_quota
        self.monthly_quota = monthly_quota

        # Track usage with timestamps
        self._hourly_usage: deque = deque()
        self._daily_usage: deque = deque()
        self._monthly_usage: deque = deque()

        self._lock = asyncio.Lock()

    async def check_quota(self, amount: int = 1) -> bool:
        """Check if quota allows the requested amount.

        Args:
            amount: Amount to check

        Returns:
            True if quota allows the amount
        """
        async with self._lock:
            now = datetime.utcnow()

            # Clean old entries and check quotas
            if self.hourly_quota:
                self._clean_old_entries(self._hourly_usage, now, hours=1)
                if len(self._hourly_usage) + amount > self.hourly_quota:
                    return False

            if self.daily_quota:
                self._clean_old_entries(self._daily_usage, now, days=1)
                if len(self._daily_usage) + amount > self.daily_quota:
                    return False

            if self.monthly_quota:
                self._clean_old_entries(self._monthly_usage, now, days=30)
                if len(self._monthly_usage) + amount > self.monthly_quota:
                    return False

            return True

    async def use_quota(self, amount: int = 1) -> bool:
        """Use quota if available.

        Args:
            amount: Amount to use

        Returns:
            True if quota was available and used

        Raises:
            ValueError: If amount is negative or zero
        """
        if amount <= 0:
            raise ValueError("Amount must be positive")

        async with self._lock:
            now = datetime.utcnow()

            # Check if quota is available
            if not await self.check_quota(amount):
                return False

            # Record usage
            for _ in range(amount):
                if self.hourly_quota:
                    self._hourly_usage.append(now)
                if self.daily_quota:
                    self._daily_usage.append(now)
                if self.monthly_quota:
                    self._monthly_usage.append(now)

            return True

    def _clean_old_entries(
        self,
        usage_deque: deque,
        now: datetime,
        **kwargs: int
    ) -> None:
        """Remove old entries from usage tracking.

        Args:
            usage_deque: Deque to clean
            now: Current timestamp
            **kwargs: Timedelta parameters (hours, days, etc.)
        """
        cutoff = now - timedelta(**kwargs)
        while usage_deque and usage_deque[0] < cutoff:
            usage_deque.popleft()

    async def get_usage_stats(self) -> Dict[str, Dict[str, int]]:
        """Get current usage statistics.

        Returns:
            Dictionary with usage stats for each period
        """
        async with self._lock:
            now = datetime.utcnow()

            # Clean old entries
            if self.hourly_quota:
                self._clean_old_entries(self._hourly_usage, now, hours=1)
            if self.daily_quota:
                self._clean_old_entries(self._daily_usage, now, days=1)
            if self.monthly_quota:
                self._clean_old_entries(self._monthly_usage, now, days=30)

            return {
                "hourly": {
                    "used": len(self._hourly_usage),
                    "quota": self.hourly_quota or 0,
                    "remaining": (self.hourly_quota or 0) - len(self._hourly_usage)
                },
                "daily": {
                    "used": len(self._daily_usage),
                    "quota": self.daily_quota or 0,
                    "remaining": (self.daily_quota or 0) - len(self._daily_usage)
                },
                "monthly": {
                    "used": len(self._monthly_usage),
                    "quota": self.monthly_quota or 0,
                    "remaining": (self.monthly_quota or 0) - len(self._monthly_usage)
                }
            }

    async def reset(self) -> None:
        """Reset all quota usage."""
        async with self._lock:
            self._hourly_usage.clear()
            self._daily_usage.clear()
            self._monthly_usage.clear()
