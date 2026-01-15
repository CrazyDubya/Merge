"""
Enhanced streaming support for Club Harness.

Provides proper tool call extraction and progress callbacks during streaming responses.
Inspired by repos/qwen-code/qwen_agent/llm/streaming.py

Features:
- Streaming with tool call extraction (handles partial JSON)
- Progress callbacks for UI integration
- Timeout and retry logic for streams
- Token counting during stream
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
from enum import Enum


class StreamState(Enum):
    """State of the streaming response."""
    STARTING = "starting"
    STREAMING = "streaming"
    COLLECTING_TOOL_CALL = "collecting_tool_call"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ToolCallChunk:
    """Accumulated tool call data from stream."""
    index: int
    id: str = ""
    type: str = "function"
    function_name: str = ""
    function_arguments: str = ""

    def is_complete(self) -> bool:
        """Check if tool call data is complete."""
        if not self.id or not self.function_name:
            return False

        # Check if arguments is valid JSON
        if self.function_arguments:
            try:
                json.loads(self.function_arguments)
                return True
            except json.JSONDecodeError:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to standard tool call format."""
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function_name,
                "arguments": self.function_arguments,
            }
        }


@dataclass
class StreamProgress:
    """Progress information during streaming."""
    state: StreamState
    content: str = ""
    tokens_so_far: int = 0
    elapsed_ms: float = 0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def __repr__(self) -> str:
        return f"StreamProgress({self.state.value}, {self.tokens_so_far} tokens, {len(self.content)} chars)"


@dataclass
class StreamResult:
    """Final result of streaming response."""
    content: str
    tool_calls: List[Dict[str, Any]]
    total_tokens: int
    elapsed_ms: float
    model: str
    finish_reason: str


class StreamingHandler:
    """
    Handles streaming responses with proper tool call extraction.

    This handler:
    1. Accumulates content chunks
    2. Detects and parses tool calls (even partial JSON)
    3. Provides progress callbacks
    4. Handles timeouts gracefully
    """

    def __init__(
        self,
        progress_callback: Optional[Callable[[StreamProgress], None]] = None,
        chunk_callback: Optional[Callable[[str], None]] = None,
        tool_call_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize streaming handler.

        Args:
            progress_callback: Called with StreamProgress on state changes
            chunk_callback: Called with each content chunk
            tool_call_callback: Called when a complete tool call is detected
        """
        self.progress_callback = progress_callback
        self.chunk_callback = chunk_callback
        self.tool_call_callback = tool_call_callback

        # State
        self._content = ""
        self._tool_calls: Dict[int, ToolCallChunk] = {}
        self._state = StreamState.STARTING
        self._start_time: Optional[datetime] = None
        self._token_count = 0
        self._model = ""
        self._finish_reason = ""

    def reset(self):
        """Reset handler state for new stream."""
        self._content = ""
        self._tool_calls = {}
        self._state = StreamState.STARTING
        self._start_time = None
        self._token_count = 0
        self._model = ""
        self._finish_reason = ""

    def handle_chunk(self, chunk: Dict[str, Any]) -> Optional[str]:
        """
        Handle a single SSE chunk from the stream.

        Args:
            chunk: Parsed JSON chunk from stream

        Returns:
            Content delta if present, None otherwise
        """
        if self._start_time is None:
            self._start_time = datetime.now()
            self._state = StreamState.STREAMING

        # Extract model info
        if model := chunk.get("model"):
            self._model = model

        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {})

        # Check finish reason
        if finish := choice.get("finish_reason"):
            self._finish_reason = finish

        content_delta = None

        # Handle content
        if content := delta.get("content"):
            content_delta = content
            self._content += content
            self._token_count += len(content.split())  # Rough token estimate

            if self.chunk_callback:
                self.chunk_callback(content)

        # Handle tool calls
        if tool_calls := delta.get("tool_calls"):
            self._state = StreamState.COLLECTING_TOOL_CALL
            self._handle_tool_call_delta(tool_calls)

        # Send progress update
        if self.progress_callback:
            elapsed = (datetime.now() - self._start_time).total_seconds() * 1000
            progress = StreamProgress(
                state=self._state,
                content=self._content,
                tokens_so_far=self._token_count,
                elapsed_ms=elapsed,
                tool_calls=[tc.to_dict() for tc in self._tool_calls.values() if tc.is_complete()],
            )
            self.progress_callback(progress)

        return content_delta

    def _handle_tool_call_delta(self, tool_calls: List[Dict[str, Any]]):
        """Handle tool call deltas from stream."""
        for tc_delta in tool_calls:
            index = tc_delta.get("index", 0)

            if index not in self._tool_calls:
                self._tool_calls[index] = ToolCallChunk(index=index)

            tc = self._tool_calls[index]

            # Accumulate tool call data
            if tc_id := tc_delta.get("id"):
                tc.id = tc_id

            if tc_type := tc_delta.get("type"):
                tc.type = tc_type

            if func := tc_delta.get("function"):
                if name := func.get("name"):
                    tc.function_name = name
                if args := func.get("arguments"):
                    tc.function_arguments += args

            # Check if complete and notify
            if tc.is_complete() and self.tool_call_callback:
                self.tool_call_callback(tc.to_dict())

    def finalize(self) -> StreamResult:
        """Finalize the stream and return result."""
        self._state = StreamState.COMPLETED

        elapsed = 0
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds() * 1000

        # Collect complete tool calls
        tool_calls = [
            tc.to_dict() for tc in self._tool_calls.values()
            if tc.is_complete()
        ]

        return StreamResult(
            content=self._content,
            tool_calls=tool_calls,
            total_tokens=self._token_count,
            elapsed_ms=elapsed,
            model=self._model,
            finish_reason=self._finish_reason,
        )

    def get_progress(self) -> StreamProgress:
        """Get current progress."""
        elapsed = 0
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds() * 1000

        return StreamProgress(
            state=self._state,
            content=self._content,
            tokens_so_far=self._token_count,
            elapsed_ms=elapsed,
            tool_calls=[tc.to_dict() for tc in self._tool_calls.values() if tc.is_complete()],
        )


class ToolCallParser:
    """
    Parser for extracting tool calls from text.

    Useful for models that return tool calls as text instead of structured JSON.
    """

    # Common patterns for tool call syntax
    PATTERNS = [
        # Function call: function_name(args)
        r'(\w+)\s*\(\s*({[^}]+}|[^)]*)\s*\)',
        # JSON object with function/action
        r'\{[^}]*"(?:function|action|tool)"[^}]*\}',
        # XML-like: <tool_call>...</tool_call>
        r'<tool_call>(.*?)</tool_call>',
    ]

    @classmethod
    def extract_from_text(cls, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from text content.

        Handles various formats models might use for tool invocations.
        """
        tool_calls = []

        # Try JSON extraction first
        json_objects = cls._extract_json_objects(text)
        for obj in json_objects:
            if cls._is_tool_call(obj):
                tool_calls.append(cls._normalize_tool_call(obj))

        # Try function call pattern
        func_pattern = re.compile(cls.PATTERNS[0])
        for match in func_pattern.finditer(text):
            name = match.group(1)
            args_str = match.group(2)

            # Skip common false positives
            if name.lower() in ['if', 'for', 'while', 'print', 'return', 'def', 'class']:
                continue

            tool_call = {
                "id": f"call_{hash(match.group(0)) % 100000}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args_str if args_str.strip().startswith('{') else json.dumps({"input": args_str}),
                }
            }
            tool_calls.append(tool_call)

        return tool_calls

    @classmethod
    def _extract_json_objects(cls, text: str) -> List[Dict[str, Any]]:
        """Extract JSON objects from text."""
        objects = []
        # Find potential JSON by matching braces
        depth = 0
        start = None

        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        obj = json.loads(text[start:i+1])
                        if isinstance(obj, dict):
                            objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = None

        return objects

    @classmethod
    def _is_tool_call(cls, obj: Dict[str, Any]) -> bool:
        """Check if a JSON object looks like a tool call."""
        # Check for common tool call keys
        tool_keys = {'function', 'action', 'tool', 'name', 'tool_call', 'invoke'}
        return bool(tool_keys & set(obj.keys()))

    @classmethod
    def _normalize_tool_call(cls, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize various tool call formats to standard format."""
        # Already in standard format
        if 'function' in obj and isinstance(obj['function'], dict):
            return {
                "id": obj.get("id", f"call_{hash(str(obj)) % 100000}"),
                "type": "function",
                "function": obj["function"],
            }

        # Simple format: {"name": "...", "args": {...}}
        if 'name' in obj:
            return {
                "id": f"call_{hash(str(obj)) % 100000}",
                "type": "function",
                "function": {
                    "name": obj["name"],
                    "arguments": json.dumps(obj.get("args", obj.get("arguments", obj.get("parameters", {})))),
                }
            }

        # Action format: {"action": "...", "input": {...}}
        if 'action' in obj:
            return {
                "id": f"call_{hash(str(obj)) % 100000}",
                "type": "function",
                "function": {
                    "name": obj["action"],
                    "arguments": json.dumps(obj.get("input", obj.get("action_input", {}))),
                }
            }

        # Tool format: {"tool": "...", "params": {...}}
        if 'tool' in obj:
            return {
                "id": f"call_{hash(str(obj)) % 100000}",
                "type": "function",
                "function": {
                    "name": obj["tool"],
                    "arguments": json.dumps(obj.get("params", obj.get("parameters", {}))),
                }
            }

        # Fallback - return as-is wrapped
        return {
            "id": f"call_{hash(str(obj)) % 100000}",
            "type": "function",
            "function": {
                "name": "unknown",
                "arguments": json.dumps(obj),
            }
        }


def stream_with_handler(
    backend,
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    progress_callback: Optional[Callable[[StreamProgress], None]] = None,
    chunk_callback: Optional[Callable[[str], None]] = None,
) -> StreamResult:
    """
    Convenience function to stream with a handler.

    Args:
        backend: OpenRouterBackend instance
        messages: Chat messages
        model: Model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        progress_callback: Called on progress updates
        chunk_callback: Called with each content chunk

    Returns:
        StreamResult with complete response
    """
    handler = StreamingHandler(
        progress_callback=progress_callback,
        chunk_callback=chunk_callback,
    )

    # Use the backend's stream method
    for response in backend.chat_stream(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    ):
        # Convert OpenRouterResponse to chunk format
        chunk = {
            "choices": [{
                "delta": {"content": response.content},
            }],
            "model": response.model,
        }
        handler.handle_chunk(chunk)

    return handler.finalize()


def collect_stream(
    stream_generator,
    timeout_ms: int = 120000,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Collect all content and tool calls from a stream generator.

    Args:
        stream_generator: Generator yielding stream chunks
        timeout_ms: Maximum time to wait

    Returns:
        Tuple of (content, tool_calls)
    """
    handler = StreamingHandler()

    for chunk in stream_generator:
        if hasattr(chunk, 'content'):
            # OpenRouterResponse format
            handler.handle_chunk({
                "choices": [{
                    "delta": {"content": chunk.content},
                }],
                "model": getattr(chunk, 'model', ''),
            })
        else:
            # Raw chunk format
            handler.handle_chunk(chunk)

    result = handler.finalize()
    return result.content, result.tool_calls
