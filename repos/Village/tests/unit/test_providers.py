"""Unit tests for LLM providers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from village.exceptions import LLMProviderError


@pytest.mark.asyncio
class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    @pytest.fixture
    def mock_openai_response(self):
        """Create a mock OpenAI response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        return mock_response

    @pytest.mark.skipif(True, reason="OpenAI package optional")
    async def test_openai_generate(self, mock_openai_response):
        """Test OpenAI text generation."""
        with patch('village.llm.openai.OPENAI_AVAILABLE', True):
            with patch('village.llm.openai.AsyncOpenAI') as mock_client_class:
                from village.llm.openai import OpenAIProvider

                # Setup mock
                mock_client = AsyncMock()
                mock_client.chat.completions.create = AsyncMock(
                    return_value=mock_openai_response
                )
                mock_client_class.return_value = mock_client

                # Create provider
                provider = OpenAIProvider(api_key="test_key")

                # Test generation
                result = await provider.generate("Test prompt")
                assert result == "Test response"

    @pytest.mark.skipif(True, reason="OpenAI package optional")
    async def test_openai_chat(self, mock_openai_response):
        """Test OpenAI chat completion."""
        with patch('village.llm.openai.OPENAI_AVAILABLE', True):
            with patch('village.llm.openai.AsyncOpenAI') as mock_client_class:
                from village.llm.openai import OpenAIProvider

                mock_client = AsyncMock()
                mock_client.chat.completions.create = AsyncMock(
                    return_value=mock_openai_response
                )
                mock_client_class.return_value = mock_client

                provider = OpenAIProvider(api_key="test_key")

                messages = [
                    {"role": "user", "content": "Hello"}
                ]
                result = await provider.chat(messages)
                assert result == "Test response"

    def test_openai_provider_requires_api_key(self):
        """Test that OpenAI provider requires API key."""
        with patch('village.llm.openai.OPENAI_AVAILABLE', True):
            with patch('os.getenv', return_value=None):
                from village.llm.openai import OpenAIProvider

                with pytest.raises(LLMProviderError, match="API key required"):
                    OpenAIProvider()


@pytest.mark.asyncio
class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    @pytest.fixture
    def mock_anthropic_response(self):
        """Create a mock Anthropic response."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Test response"
        return mock_response

    @pytest.mark.skipif(True, reason="Anthropic package optional")
    async def test_anthropic_generate(self, mock_anthropic_response):
        """Test Anthropic text generation."""
        with patch('village.llm.anthropic.ANTHROPIC_AVAILABLE', True):
            with patch('village.llm.anthropic.AsyncAnthropic') as mock_client_class:
                from village.llm.anthropic import AnthropicProvider

                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(
                    return_value=mock_anthropic_response
                )
                mock_client_class.return_value = mock_client

                provider = AnthropicProvider(api_key="test_key")

                result = await provider.generate("Test prompt")
                assert result == "Test response"

    @pytest.mark.skipif(True, reason="Anthropic package optional")
    async def test_anthropic_chat(self, mock_anthropic_response):
        """Test Anthropic chat completion."""
        with patch('village.llm.anthropic.ANTHROPIC_AVAILABLE', True):
            with patch('village.llm.anthropic.AsyncAnthropic') as mock_client_class:
                from village.llm.anthropic import AnthropicProvider

                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(
                    return_value=mock_anthropic_response
                )
                mock_client_class.return_value = mock_client

                provider = AnthropicProvider(api_key="test_key")

                messages = [
                    {"role": "user", "content": "Hello"}
                ]
                result = await provider.chat(messages)
                assert result == "Test response"

    def test_anthropic_provider_requires_api_key(self):
        """Test that Anthropic provider requires API key."""
        with patch('village.llm.anthropic.ANTHROPIC_AVAILABLE', True):
            with patch('os.getenv', return_value=None):
                from village.llm.anthropic import AnthropicProvider

                with pytest.raises(LLMProviderError, match="API key required"):
                    AnthropicProvider()


@pytest.mark.asyncio
class TestGoogleProvider:
    """Tests for Google provider."""

    @pytest.fixture
    def mock_google_response(self):
        """Create a mock Google response."""
        mock_response = MagicMock()
        mock_response.text = "Test response"
        return mock_response

    @pytest.mark.skipif(True, reason="Google package optional")
    async def test_google_generate(self, mock_google_response):
        """Test Google text generation."""
        with patch('village.llm.google.GOOGLE_AVAILABLE', True):
            with patch('village.llm.google.genai') as mock_genai:
                from village.llm.google import GoogleProvider

                mock_genai.configure = MagicMock()
                mock_model = AsyncMock()
                mock_model.generate_content_async = AsyncMock(
                    return_value=mock_google_response
                )
                mock_genai.GenerativeModel.return_value = mock_model

                provider = GoogleProvider(api_key="test_key")

                result = await provider.generate("Test prompt")
                assert result == "Test response"

    def test_google_provider_requires_api_key(self):
        """Test that Google provider requires API key."""
        with patch('village.llm.google.GOOGLE_AVAILABLE', True):
            with patch('os.getenv', return_value=None):
                from village.llm.google import GoogleProvider

                with pytest.raises(LLMProviderError, match="API key required"):
                    GoogleProvider()


@pytest.mark.asyncio
class TestRateLimiter:
    """Tests for rate limiter."""

    async def test_rate_limiter_acquire(self):
        """Test rate limiter token acquisition."""
        from village.utils.rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=60)

        # Should acquire immediately
        wait_time = await limiter.acquire(1)
        assert wait_time >= 0

    async def test_rate_limiter_try_acquire(self):
        """Test non-blocking token acquisition."""
        from village.utils.rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=60, burst_size=2)

        # Should succeed
        assert await limiter.try_acquire(1) is True
        assert await limiter.try_acquire(1) is True

        # Should fail (burst exhausted)
        assert await limiter.try_acquire(1) is False

    async def test_quota_manager_check_quota(self):
        """Test quota checking."""
        from village.utils.rate_limiter import QuotaManager

        quota = QuotaManager(hourly_quota=10)

        # Should have quota
        assert await quota.check_quota(5) is True

        # Use quota
        assert await quota.use_quota(5) is True

        # Should still have quota
        assert await quota.check_quota(5) is True

        # Use remaining quota
        assert await quota.use_quota(5) is True

        # Should not have quota
        assert await quota.check_quota(1) is False


@pytest.mark.asyncio
class TestStorage:
    """Tests for storage backends."""

    async def test_memory_storage(self):
        """Test in-memory storage."""
        from village.storage.memory import InMemoryStorage

        storage = InMemoryStorage()

        # Test village operations
        village_data = {"name": "Test Village", "villagers": []}
        assert await storage.save_village("village1", village_data) is True

        loaded = await storage.load_village("village1")
        assert loaded == village_data

        # Test memory operations
        assert await storage.save_memory("owner1", "key1", "value1") is True

        value = await storage.load_memory("owner1", "key1")
        assert value == "value1"

        keys = await storage.list_memory_keys("owner1")
        assert "key1" in keys

        # Test task history
        assert await storage.save_task_history(
            "village1",
            "Test task",
            "Test result"
        ) is True

        history = await storage.load_task_history("village1")
        assert len(history) == 1
        assert history[0]["task"] == "Test task"

        # Test health check
        assert await storage.health_check() is True


@pytest.mark.asyncio
class TestMetrics:
    """Tests for Prometheus metrics."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        from village.utils.metrics import metrics_available, initialize_metrics

        if not metrics_available():
            pytest.skip("Prometheus client not available")

        metrics = initialize_metrics()
        assert metrics is not None

        # Test metric export
        output = metrics.export_metrics()
        assert output is not None
        assert len(output) > 0
