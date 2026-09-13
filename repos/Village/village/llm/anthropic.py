"""Anthropic LLM Provider Implementation.

This module provides integration with Anthropic's Claude models.
"""

import os
from typing import Any, Dict, List, Optional

try:
    from anthropic import AsyncAnthropic, Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from village.llm.base import BaseLLMProvider
from village.exceptions import LLMProviderError


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM Provider supporting Claude models.

    Attributes:
        api_key: Anthropic API key
        model: Model name (e.g., 'claude-3-opus-20240229', 'claude-3-sonnet-20240229')
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-sonnet-20240229",
        max_retries: int = 3,
        timeout: int = 60,
        **kwargs: Any
    ) -> None:
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model name to use
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options

        Raises:
            LLMProviderError: If Anthropic package is not installed
        """
        if not ANTHROPIC_AVAILABLE:
            raise LLMProviderError(
                "Anthropic package not installed. Install with: pip install anthropic>=0.7.0"
            )

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMProviderError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.config = kwargs

        # Initialize sync and async clients
        self._client = Anthropic(
            api_key=self.api_key,
            max_retries=self.max_retries,
            timeout=self.timeout
        )
        self._async_client = AsyncAnthropic(
            api_key=self.api_key,
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
            max_tokens: Maximum tokens to generate (default: 1024)
            temperature: Sampling temperature (0-1)
            **kwargs: Additional generation parameters

        Returns:
            Generated text completion

        Raises:
            LLMProviderError: If generation fails
        """
        if max_tokens is None:
            max_tokens = 1024

        try:
            # Convert to messages format for Claude
            messages = [{"role": "user", "content": prompt}]

            response = await self._async_client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            # Extract text from response
            return response.content[0].text if response.content else ""

        except Exception as e:
            raise LLMProviderError(f"Anthropic generation failed: {str(e)}") from e

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
            max_tokens: Maximum tokens to generate (default: 1024)
            temperature: Sampling temperature (0-1)
            **kwargs: Additional generation parameters

        Returns:
            Generated chat response

        Raises:
            LLMProviderError: If chat completion fails
        """
        if max_tokens is None:
            max_tokens = 1024

        try:
            # Convert messages to Anthropic format
            # Anthropic requires alternating user/assistant messages
            anthropic_messages = self._convert_messages(messages)

            # Extract system prompt if present
            system_prompt = kwargs.pop("system", None)
            if not system_prompt:
                # Check if first message is a system message
                if anthropic_messages and anthropic_messages[0].get("role") == "system":
                    system_msg = anthropic_messages.pop(0)
                    system_prompt = system_msg["content"]

            # Build request parameters
            request_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }

            if system_prompt:
                request_params["system"] = system_prompt

            response = await self._async_client.messages.create(**request_params)

            # Extract text from response
            return response.content[0].text if response.content else ""

        except Exception as e:
            raise LLMProviderError(f"Anthropic chat completion failed: {str(e)}") from e

    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert messages to Anthropic format.

        Anthropic requires alternating user/assistant messages and handles
        system prompts separately.

        Args:
            messages: List of message dictionaries

        Returns:
            Converted messages list
        """
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Keep system messages for extraction
            if role == "system":
                converted.append({"role": "system", "content": content})
            # Convert other roles
            elif role in ["user", "assistant"]:
                converted.append({"role": role, "content": content})
            else:
                # Default unknown roles to user
                converted.append({"role": "user", "content": content})

        return converted

    def validate_config(self) -> bool:
        """Validate provider configuration.

        Returns:
            True if configuration is valid

        Raises:
            LLMProviderError: If configuration is invalid
        """
        if not self.api_key:
            raise LLMProviderError("Anthropic API key is required")

        if not self.model:
            raise LLMProviderError("Model name is required")

        # Validate model name format
        valid_prefixes = ["claude-3", "claude-2", "claude-instant"]
        if not any(self.model.startswith(prefix) for prefix in valid_prefixes):
            raise LLMProviderError(
                f"Invalid model name: {self.model}. Must start with one of {valid_prefixes}"
            )

        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary containing model information
        """
        return {
            "id": self.model,
            "provider": "anthropic",
            "model": self.model,
            "context_window": self._get_context_window(),
            "supports_vision": self._supports_vision(),
            "model_family": self._get_model_family(),
        }

    def _get_context_window(self) -> int:
        """Get context window size for the current model.

        Returns:
            Context window size in tokens
        """
        if "claude-3" in self.model:
            return 200000  # Claude 3 models have 200K context
        elif "claude-2" in self.model:
            return 100000  # Claude 2 has 100K context
        elif "claude-instant" in self.model:
            return 100000  # Claude Instant has 100K context
        return 100000  # Default

    def _supports_vision(self) -> bool:
        """Check if model supports vision capabilities.

        Returns:
            True if vision is supported
        """
        # Claude 3 models support vision
        return "claude-3" in self.model

    def _get_model_family(self) -> str:
        """Get the model family name.

        Returns:
            Model family identifier
        """
        if "opus" in self.model:
            return "claude-3-opus"
        elif "sonnet" in self.model:
            return "claude-3-sonnet"
        elif "haiku" in self.model:
            return "claude-3-haiku"
        elif "claude-2" in self.model:
            return "claude-2"
        elif "claude-instant" in self.model:
            return "claude-instant"
        return "unknown"

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
            max_tokens: Maximum tokens to generate (default: 1024)
            temperature: Sampling temperature (0-1)
            **kwargs: Additional generation parameters

        Yields:
            Text chunks as they are generated

        Raises:
            LLMProviderError: If streaming fails
        """
        if max_tokens is None:
            max_tokens = 1024

        try:
            messages = [{"role": "user", "content": prompt}]

            async with self._async_client.messages.stream(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            raise LLMProviderError(f"Anthropic streaming failed: {str(e)}") from e
