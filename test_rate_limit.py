"""Tests for the ported Village rate limiter and its Agent wiring."""

import asyncio
import pytest

from club_harness.core.rate_limit import RateLimiter, QuotaManager


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_immediate(self):
        limiter = RateLimiter(requests_per_minute=60)
        wait = await limiter.acquire(1)
        assert wait >= 0

    @pytest.mark.asyncio
    async def test_try_acquire(self):
        limiter = RateLimiter(requests_per_minute=60, burst_size=2)
        assert await limiter.try_acquire(1) is True
        assert await limiter.try_acquire(1) is True
        assert await limiter.try_acquire(1) is False

    def test_acquire_nowait_sync(self):
        limiter = RateLimiter(requests_per_minute=60, burst_size=2)
        assert limiter.acquire_nowait(1) is True
        assert limiter.acquire_nowait(1) is True
        assert limiter.acquire_nowait(1) is False

    def test_acquire_nowait_refills(self):
        limiter = RateLimiter(requests_per_minute=6000, burst_size=1)
        assert limiter.acquire_nowait(1) is True
        assert limiter.acquire_nowait(1) is False
        # refill_rate = 100/s; sleep 50ms -> ~5 tokens, capped at burst 1
        import time
        time.sleep(0.05)
        assert limiter.acquire_nowait(1) is True

    def test_tokens_exceeding_burst_raises(self):
        import asyncio as _a

        async def go():
            limiter = RateLimiter(requests_per_minute=60, burst_size=2)
            with pytest.raises(ValueError):
                await limiter.acquire(10)

        _a.run(go())


class TestQuotaManager:
    @pytest.mark.asyncio
    async def test_check_and_use_quota(self):
        qm = QuotaManager(hourly_quota=2)
        assert await qm.check_quota(1) is True
        assert await qm.use_quota(1) is True
        assert await qm.use_quota(1) is True
        assert await qm.use_quota(1) is False


class TestAgentRateLimitWiring:
    def test_chat_without_limiter_works(self):
        from unittest.mock import MagicMock, patch
        from club_harness.core.agent import Agent

        agent = Agent.__new__(Agent)
        agent.name = "t"
        agent.rate_limiter = None
        with patch.object(Agent, "step") as step:
            step.return_value = MagicMock(content="ok")
            assert agent.chat("hi") == "ok"

    def test_chat_with_exhausted_limiter_raises(self):
        from unittest.mock import MagicMock, patch
        from club_harness.core.agent import Agent
        from club_harness.core.errors import RateLimitError

        limiter = RateLimiter(requests_per_minute=60, burst_size=1)
        assert limiter.acquire_nowait(1) is True  # exhaust

        agent = Agent.__new__(Agent)
        agent.name = "t"
        agent.rate_limiter = limiter
        with patch.object(Agent, "step") as step:
            with pytest.raises(RateLimitError):
                agent.chat("hi")
            step.assert_not_called()

    def test_chat_with_available_limiter_passes(self):
        from unittest.mock import MagicMock, patch
        from club_harness.core.agent import Agent

        limiter = RateLimiter(requests_per_minute=60, burst_size=10)
        agent = Agent.__new__(Agent)
        agent.name = "t"
        agent.rate_limiter = limiter
        with patch.object(Agent, "step") as step:
            step.return_value = MagicMock(content="ok")
            assert agent.chat("hi") == "ok"
