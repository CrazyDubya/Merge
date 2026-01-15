"""
OpenRouter backend for Club Harness.

Provides access to 400+ models through a unified API.
Inspired by:
- llm-council: Async model queries
- hivey: Cost-aware routing
- qwen-code: Streaming support
"""

import json
import os
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx


@dataclass
class OpenRouterResponse:
    """Response from OpenRouter API."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    tool_calls: Optional[List[Dict[str, Any]]] = None


class OpenRouterBackend:
    """
    OpenRouter API backend.

    Features:
    - Async/sync support
    - Streaming responses
    - Tool calling
    - Retry with exponential backoff
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "meta-llama/llama-3.1-8b-instruct",  # Reliable cheap model Jan 2026
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.default_model = default_model
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. "
                "Set OPENROUTER_API_KEY environment variable."
            )

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/CrazyDubya/Merge",
            "X-Title": "Club Harness",
        }

    def _build_request(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Build API request payload."""
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if tools:
            payload["tools"] = tools

        return payload

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> OpenRouterResponse:
        """
        Send a chat completion request (synchronous).

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier (defaults to config)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Optional tool definitions for function calling

        Returns:
            OpenRouterResponse with content and metadata
        """
        payload = self._build_request(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            stream=False,
        )

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        return self._parse_response(data)

    async def chat_async(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> OpenRouterResponse:
        """Send a chat completion request (asynchronous)."""
        payload = self._build_request(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            stream=False,
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        return self._parse_response(data)

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        Stream chat completion tokens (synchronous generator).

        Yields OpenRouterResponse objects with partial content.
        """
        payload = self._build_request(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{self.BASE_URL}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if content := chunk.get("choices", [{}])[0].get(
                                "delta", {}
                            ).get("content"):
                                yield OpenRouterResponse(
                                    content=content,
                                    model=chunk.get("model", self.default_model),
                                    usage={},
                                    finish_reason="",
                                )
                        except json.JSONDecodeError:
                            continue

    async def chat_stream_async(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[OpenRouterResponse, None]:
        """Stream chat completion tokens (asynchronous generator)."""
        payload = self._build_request(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.BASE_URL}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if content := chunk.get("choices", [{}])[0].get(
                                "delta", {}
                            ).get("content"):
                                yield OpenRouterResponse(
                                    content=content,
                                    model=chunk.get("model", self.default_model),
                                    usage={},
                                    finish_reason="",
                                )
                        except json.JSONDecodeError:
                            continue

    def _parse_response(self, data: Dict[str, Any]) -> OpenRouterResponse:
        """Parse API response into OpenRouterResponse."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        return OpenRouterResponse(
            content=message.get("content", ""),
            model=data.get("model", self.default_model),
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", "unknown"),
            tool_calls=message.get("tool_calls"),
        )

    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{self.BASE_URL}/models",
                headers=self._get_headers(),
            )
            response.raise_for_status()
            return response.json().get("data", [])
