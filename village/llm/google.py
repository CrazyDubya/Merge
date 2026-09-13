"""Google Gemini LLM Provider Implementation.

This module provides integration with Google's Gemini models.
"""

import os
from typing import Any, Dict, List, Optional

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerateContentResponse
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

from village.llm.base import BaseLLMProvider
from village.exceptions import LLMProviderError


class GoogleProvider(BaseLLMProvider):
    """Google LLM Provider supporting Gemini models.

    Attributes:
        api_key: Google API key
        model: Model name (e.g., 'gemini-pro', 'gemini-pro-vision')
        safety_settings: Safety filter settings
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-pro",
        safety_settings: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize Google provider.

        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model: Model name to use
            safety_settings: Optional safety filter configuration
            **kwargs: Additional configuration options

        Raises:
            LLMProviderError: If Google package is not installed
        """
        if not GOOGLE_AVAILABLE:
            raise LLMProviderError(
                "Google GenerativeAI package not installed. "
                "Install with: pip install google-generativeai>=0.3.0"
            )

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise LLMProviderError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model_name = model
        self.safety_settings = safety_settings or {}
        self.config = kwargs

        # Configure the API
        genai.configure(api_key=self.api_key)

        # Initialize the model
        try:
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings=self.safety_settings
            )
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Google model: {str(e)}") from e

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
            temperature: Sampling temperature (0-1)
            **kwargs: Additional generation parameters

        Returns:
            Generated text completion

        Raises:
            LLMProviderError: If generation fails
        """
        try:
            # Build generation config
            generation_config = {
                "temperature": temperature,
            }
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens

            # Merge with any additional kwargs
            generation_config.update(kwargs)

            # Generate content asynchronously
            response = await self._model.generate_content_async(
                prompt,
                generation_config=generation_config
            )

            return response.text if response.text else ""

        except Exception as e:
            raise LLMProviderError(f"Google generation failed: {str(e)}") from e

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
            temperature: Sampling temperature (0-1)
            **kwargs: Additional generation parameters

        Returns:
            Generated chat response

        Raises:
            LLMProviderError: If chat completion fails
        """
        try:
            # Build generation config
            generation_config = {
                "temperature": temperature,
            }
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens

            generation_config.update(kwargs)

            # Convert messages to Gemini format
            gemini_messages = self._convert_messages(messages)

            # Start chat session
            chat = self._model.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])

            # Send the last message
            last_message = gemini_messages[-1] if gemini_messages else {"parts": [""]}
            response = await chat.send_message_async(
                last_message["parts"],
                generation_config=generation_config
            )

            return response.text if response.text else ""

        except Exception as e:
            raise LLMProviderError(f"Google chat completion failed: {str(e)}") from e

    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Convert messages to Gemini format.

        Args:
            messages: List of message dictionaries

        Returns:
            Converted messages list in Gemini format
        """
        gemini_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Convert role to Gemini format
            if role == "assistant":
                gemini_role = "model"
            elif role == "system":
                # System messages can be prepended to user messages
                gemini_role = "user"
            else:
                gemini_role = "user"

            gemini_messages.append({
                "role": gemini_role,
                "parts": [content]
            })

        return gemini_messages

    def validate_config(self) -> bool:
        """Validate provider configuration.

        Returns:
            True if configuration is valid

        Raises:
            LLMProviderError: If configuration is invalid
        """
        if not self.api_key:
            raise LLMProviderError("Google API key is required")

        if not self.model_name:
            raise LLMProviderError("Model name is required")

        # Validate model name
        valid_models = [
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]

        if not any(self.model_name.startswith(m) for m in valid_models):
            raise LLMProviderError(
                f"Invalid model name: {self.model_name}. "
                f"Must be one of {valid_models}"
            )

        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary containing model information
        """
        try:
            # Get model information from API
            model_info = genai.get_model(f"models/{self.model_name}")

            return {
                "id": self.model_name,
                "provider": "google",
                "model": self.model_name,
                "display_name": model_info.display_name,
                "description": model_info.description,
                "input_token_limit": model_info.input_token_limit,
                "output_token_limit": model_info.output_token_limit,
                "supported_generation_methods": model_info.supported_generation_methods,
                "context_window": self._get_context_window(),
                "supports_vision": self._supports_vision(),
            }
        except Exception as e:
            return {
                "id": self.model_name,
                "provider": "google",
                "model": self.model_name,
                "context_window": self._get_context_window(),
                "supports_vision": self._supports_vision(),
                "error": str(e)
            }

    def _get_context_window(self) -> int:
        """Get context window size for the current model.

        Returns:
            Context window size in tokens
        """
        context_windows = {
            "gemini-1.5-pro": 1000000,  # 1M tokens
            "gemini-1.5-flash": 1000000,  # 1M tokens
            "gemini-pro": 32760,
            "gemini-pro-vision": 16384,
        }

        for model_name, window in context_windows.items():
            if self.model_name.startswith(model_name):
                return window

        return 32760  # Default

    def _supports_vision(self) -> bool:
        """Check if model supports vision capabilities.

        Returns:
            True if vision is supported
        """
        vision_models = ["gemini-pro-vision", "gemini-1.5"]
        return any(self.model_name.startswith(m) for m in vision_models)

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
            temperature: Sampling temperature (0-1)
            **kwargs: Additional generation parameters

        Yields:
            Text chunks as they are generated

        Raises:
            LLMProviderError: If streaming fails
        """
        try:
            # Build generation config
            generation_config = {
                "temperature": temperature,
            }
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens

            generation_config.update(kwargs)

            # Stream content
            response = await self._model.generate_content_async(
                prompt,
                generation_config=generation_config,
                stream=True
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            raise LLMProviderError(f"Google streaming failed: {str(e)}") from e
