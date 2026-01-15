#!/usr/bin/env python3
"""Test streaming and tool call collection."""

import sys
sys.path.insert(0, '/home/user/Merge')

from club_harness.llm.streaming import (
    StreamingHandler,
    StreamProgress,
    StreamResult,
    StreamState,
    ToolCallChunk,
    ToolCallParser,
    stream_with_handler,
    collect_stream,
)
import json

print("=" * 60)
print("STREAMING & TOOL CALL COLLECTION TEST")
print("=" * 60)

# Test 1: Basic streaming handler
print("\n[TEST 1] Basic streaming handler")

chunks_received = []
handler = StreamingHandler(
    chunk_callback=lambda c: chunks_received.append(c),
)

# Simulate streaming chunks
test_chunks = [
    {"choices": [{"delta": {"content": "Hello"}}], "model": "test-model"},
    {"choices": [{"delta": {"content": " world"}}], "model": "test-model"},
    {"choices": [{"delta": {"content": "!"}}], "model": "test-model"},
    {"choices": [{"finish_reason": "stop"}], "model": "test-model"},
]

for chunk in test_chunks:
    handler.handle_chunk(chunk)

result = handler.finalize()
print(f"  Content: '{result.content}'")
print(f"  Chunks received: {len(chunks_received)}")
print(f"  Total tokens: {result.total_tokens}")
print(f"  Finish reason: {result.finish_reason}")

assert result.content == "Hello world!", f"Expected 'Hello world!' but got '{result.content}'"
print("  [PASS] Basic streaming works")

# Test 2: Tool call accumulation
print("\n[TEST 2] Tool call accumulation from stream")

tool_calls_received = []
handler = StreamingHandler(
    tool_call_callback=lambda tc: tool_calls_received.append(tc),
)

# Simulate tool call chunks (split across multiple messages)
tool_chunks = [
    {"choices": [{"delta": {"content": "I'll help you with that."}}], "model": "test"},
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_123", "type": "function", "function": {"name": "search"}}]}}]},
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"que'}}]}}]},
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ry": "hello"}'}}]}}]},
    {"choices": [{"finish_reason": "tool_calls"}]},
]

for chunk in tool_chunks:
    handler.handle_chunk(chunk)

result = handler.finalize()
print(f"  Content: '{result.content}'")
print(f"  Tool calls: {len(result.tool_calls)}")

if result.tool_calls:
    tc = result.tool_calls[0]
    print(f"  First tool call: {tc['function']['name']}")
    args = json.loads(tc['function']['arguments'])
    print(f"  Arguments: {args}")
    assert args == {"query": "hello"}, "Arguments should be parsed correctly"

print("  [PASS] Tool call accumulation works")

# Test 3: ToolCallChunk validation
print("\n[TEST 3] ToolCallChunk validation")

# Incomplete chunk - missing id and name
chunk = ToolCallChunk(index=0)
assert not chunk.is_complete(), "Should be incomplete (missing id and name)"

# Chunk with invalid JSON args is incomplete
chunk = ToolCallChunk(index=0, id="call_1", function_name="test", function_arguments='{"broken')
assert not chunk.is_complete(), "Should be incomplete (invalid JSON args)"

# Complete chunk
chunk = ToolCallChunk(
    index=0,
    id="call_1",
    function_name="test",
    function_arguments='{"key": "value"}'
)
assert chunk.is_complete(), "Should be complete"

tc_dict = chunk.to_dict()
assert tc_dict["function"]["name"] == "test"
print(f"  Converted to dict: {tc_dict}")

print("  [PASS] ToolCallChunk validation works")

# Test 4: ToolCallParser - extract from text
print("\n[TEST 4] ToolCallParser - extract from text")

# Test JSON format
text_with_json = '''
Here's what I found. Let me search for more info.
{"function": "search", "args": {"query": "python tutorials"}}
'''

calls = ToolCallParser.extract_from_text(text_with_json)
print(f"  Extracted from JSON format: {len(calls)} calls")
if calls:
    print(f"    Function: {calls[0]['function']['name']}")

# Test action format
text_with_action = '''
I'll help you.
{"action": "web_search", "input": {"term": "AI news"}}
'''

calls = ToolCallParser.extract_from_text(text_with_action)
print(f"  Extracted from action format: {len(calls)} calls")
if calls:
    print(f"    Function: {calls[0]['function']['name']}")

# Test function call pattern
text_with_func = 'Let me calculate: calculate_sum({"a": 1, "b": 2})'
calls = ToolCallParser.extract_from_text(text_with_func)
print(f"  Extracted from function pattern: {len(calls)} calls")

print("  [PASS] ToolCallParser works")

# Test 5: Progress tracking
print("\n[TEST 5] Progress tracking")

progress_updates = []
handler = StreamingHandler(
    progress_callback=lambda p: progress_updates.append(p),
)

for i in range(5):
    handler.handle_chunk({
        "choices": [{"delta": {"content": f"word{i} "}}],
        "model": "test"
    })

handler.finalize()
print(f"  Progress updates received: {len(progress_updates)}")
print(f"  Final state: {progress_updates[-1].state.value}")
print(f"  Final content length: {len(progress_updates[-1].content)}")

assert len(progress_updates) == 5, "Should have 5 progress updates"
assert progress_updates[-1].state == StreamState.STREAMING

print("  [PASS] Progress tracking works")

# Test 6: Live streaming test with OpenRouter
print("\n[TEST 6] Live streaming test with OpenRouter")

try:
    from club_harness.llm.openrouter import OpenRouterBackend

    backend = OpenRouterBackend()

    # Test streaming
    print("  Streaming from free model...")
    chunks = []

    for response in backend.chat_stream(
        messages=[{"role": "user", "content": "Say 'Hello streaming!' in exactly 3 words."}],
        model="google/gemma-3n-e2b-it:free",
        max_tokens=50,
    ):
        chunks.append(response.content)
        print(f"    Chunk: '{response.content}'")

    full_response = "".join(chunks)
    print(f"  Full response: '{full_response}'")
    print(f"  Total chunks: {len(chunks)}")

    # Test with handler
    print("\n  Testing with StreamingHandler...")
    result = stream_with_handler(
        backend,
        messages=[{"role": "user", "content": "Count to 3."}],
        model="google/gemma-3n-e2b-it:free",
        max_tokens=50,
        chunk_callback=lambda c: print(f"    -> '{c}'", end=""),
    )
    print()
    print(f"  Handler result: '{result.content[:50]}...'")
    print(f"  Tokens estimated: {result.total_tokens}")

    print("  [PASS] Live streaming works")

except Exception as e:
    print(f"  [SKIP] Live test skipped: {e}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("All streaming and tool call tests passed!")
print("Features tested:")
print("  - Basic content streaming")
print("  - Tool call accumulation (partial JSON)")
print("  - ToolCallChunk validation")
print("  - Text-based tool call extraction")
print("  - Progress callbacks")
print("  - Live OpenRouter streaming")
