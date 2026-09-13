"""OpenAI LLM Provider Implementation.

This module provides integration with OpenAI's GPT models including GPT-3.5 and GPT-4.
"""

import os
from typing import Any, Dict, List, Optional

try:
    from openai import AsyncOpenAI, OpenAI
    from openai.types.chat import ChatCompletion
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from village.llm.base import BaseLLMProvider
from village.exceptions import LLMProviderError


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM Provider supporting GPT-3.5, GPT-4, and other OpenAI models.

    Attributes:
        api_key: OpenAI API key
        model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
        organization: Optional organization ID
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        organization: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 60,
        **kwargs: Any
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name to use
            organization: Optional organization ID
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options

        Raises:
            LLMProviderError: If OpenAI package is not installed
        """
        if not OPENAI_AVAILABLE:
            raise LLMProviderError(
                "OpenAI package not installed. Install with: pip install openai>=1.0.0"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMProviderError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.organization = organization
        self.max_retries = max_retries
        self.timeout = timeout
        self.config = kwargs

        # Initialize sync and async clients
        self._client = OpenAI(
            api_key=self.api_key,
            organization=self.organization,
            max_retries=self.max_retries,
            timeout=self.timeout
        )
        self._async_client = AsyncOpenAI(
            api_key=self.api_key,
            organization=self.organization,
            max_retries=self.max_retries,
            timeout=self.timeout
        )

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> str:
        """Generate text completion from a prompt.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            **kwargs: Additional generation parameters

        Returns:
            Generated text completion

        Raises:
            LLMProviderError: If generation fails
        """
        try:
            # Use chat completion API for better results
            messages = [{"role": "user", "content": prompt}]
            response = await self._async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            raise LLMProviderError(f"OpenAI generation failed: {str(e)}") from e

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> str:
        """Generate chat completion from messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            **kwargs: Additional generation parameters

        Returns:
            Generated chat response

        Raises:
            LLMProviderError: If chat completion fails
        """
        try:
            response = await self._async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            raise LLMProviderError(f"OpenAI chat completion failed: {str(e)}") from e

    def validate_config(self) -> bool:
        """Validate provider configuration.

        Returns:
            True if configuration is valid

        Raises:
            LLMProviderError: If configuration is invalid
        """
        if not self.api_key:
            raise LLMProviderError("OpenAI API key is required")

        if not self.model:
            raise LLMProviderError("Model name is required")

        # Test API connection
        try:
            self._client.models.retrieve(self.model)
            return True
        except Exception as e:
            raise LLMProviderError(
                f"Failed to validate OpenAI configuration: {str(e)}"
            ) from e

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary containing model information
        """
        try:
            model_data = self._client.models.retrieve(self.model)
            return {
                "id": model_data.id,
                "provider": "openai",
                "model": self.model,
                "created": model_data.created,
                "owned_by": model_data.owned_by,
                "context_window": self._get_context_window(),
                "supports_functions": self._supports_function_calling(),
            }
        except Exception as e:
            return {
                "id": self.model,
                "provider": "openai",
                "model": self.model,
                "error": str(e)
            }

    def _get_context_window(self) -> int:
        """Get context window size for the current model.

        Returns:
            Context window size in tokens
        """
        context_windows = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4-turbo-preview": 128000,
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-16k": 16385,
        }

        # Check for exact match or partial match
        for model_name, window in context_windows.items():
            if self.model.startswith(model_name):
                return window

        return 4096  # Default fallback

    def _supports_function_calling(self) -> bool:
        """Check if model supports function calling.

        Returns:
            True if function calling is supported
        """
        function_calling_models = [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ]

        return any(self.model.startswith(m) for m in function_calling_models)

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> Any:
        """Stream text generation from a prompt.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            **kwargs: Additional generation parameters

        Yields:
            Text chunks as they are generated

        Raises:
            LLMProviderError: If streaming fails
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            stream = await self._async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise LLMProviderError(f"OpenAI streaming failed: {str(e)}") from e
