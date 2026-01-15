#!/usr/bin/env python3
"""Test training data generation."""

import sys
sys.path.insert(0, '/home/user/Merge')

from club_harness.training import (
    TrainingDataGenerator,
    TrainingExample,
    TrainingFormat,
    ConversationFilter,
    QualityFilter,
    DiversityScorer,
)
import json

print("=" * 60)
print("TRAINING DATA GENERATOR TEST")
print("=" * 60)

# Test 1: Basic example creation
print("\n[TEST 1] Basic example creation and format conversion")

example = TrainingExample(
    example_id="test-001",
    messages=[
        {"role": "user", "content": "Write a function to add two numbers"},
        {"role": "assistant", "content": "def add(a, b):\n    return a + b"},
    ],
    system_prompt="You are a helpful coding assistant.",
    task_type="coding",
    quality_score=0.9,
)

openai_fmt = example.to_openai_format()
print(f"  OpenAI format: {json.dumps(openai_fmt)[:80]}...")

anthropic_fmt = example.to_anthropic_format()
print(f"  Anthropic format keys: {list(anthropic_fmt.keys())}")

alpaca_fmt = example.to_alpaca_format()
print(f"  Alpaca instruction: {alpaca_fmt['instruction'][:50]}...")

sharegpt_fmt = example.to_sharegpt_format()
print(f"  ShareGPT conversations: {len(sharegpt_fmt['conversations'])} turns")

print("  [PASS] Format conversion works")

# Test 2: Conversation filtering
print("\n[TEST 2] Conversation filtering")

filter = ConversationFilter(
    min_turns=1,
    max_turns=10,
    min_message_length=5,
    banned_patterns=["secret", "password"],
)

# Should pass
messages = [
    {"role": "user", "content": "Hello there!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
]
passed, reason = filter.filter(messages)
print(f"  Valid conversation: {passed} ({reason})")
assert passed, "Should pass"

# Should fail - too short
messages = [{"role": "user", "content": "Hi"}]
passed, reason = filter.filter(messages)
print(f"  Short message: {passed} ({reason})")
assert not passed, "Should fail"

# Should fail - banned pattern
messages = [
    {"role": "user", "content": "What's the secret password?"},
    {"role": "assistant", "content": "I cannot share that."},
]
passed, reason = filter.filter(messages)
print(f"  Banned pattern: {passed} ({reason})")
assert not passed, "Should fail"

print("  [PASS] Filtering works")

# Test 3: Quality filtering
print("\n[TEST 3] Quality filtering")

quality_filter = QualityFilter(min_quality_score=0.6, require_success=True)

passed, reason = quality_filter.filter(0.8, completed=True, success=True)
print(f"  High quality success: {passed}")
assert passed, "Should pass"

passed, reason = quality_filter.filter(0.4, completed=True, success=True)
print(f"  Low quality: {passed} ({reason})")
assert not passed, "Should fail"

passed, reason = quality_filter.filter(0.8, completed=True, success=False)
print(f"  High quality failure: {passed} ({reason})")
assert not passed, "Should fail"

print("  [PASS] Quality filtering works")

# Test 4: Diversity scoring
print("\n[TEST 4] Diversity scoring and deduplication")

scorer = DiversityScorer()

ex1 = TrainingExample(
    example_id="div-001",
    messages=[{"role": "user", "content": "Write a sorting algorithm"}],
    task_type="coding",
)

score1 = scorer.score(ex1)
print(f"  First example score: {score1:.2f}")
assert score1 > 0.5, "First should have high score"

scorer.add(ex1)

# Duplicate should score 0
score2 = scorer.score(ex1)
print(f"  Duplicate score: {score2:.2f}")
assert score2 == 0.0, "Duplicate should score 0"

# Similar should score lower
ex2 = TrainingExample(
    example_id="div-002",
    messages=[{"role": "user", "content": "Write a different sorting algorithm"}],
    task_type="coding",
)
score3 = scorer.score(ex2)
print(f"  Similar example score: {score3:.2f}")

print("  [PASS] Diversity scoring works")

# Test 5: Full generator pipeline
print("\n[TEST 5] Full generator pipeline")

generator = TrainingDataGenerator(
    output_path="/tmp/test-training-data",
    conversation_filter=ConversationFilter(min_turns=1),
    quality_filter=QualityFilter(min_quality_score=0.5),
)

# Add from trace
trace = {
    "trace_id": "trace-001",
    "original_task": "Implement a binary search function",
    "task_type": "coding",
    "turns": [
        {"input": "Implement binary search", "output": "Here's the implementation:"},
    ],
    "final_output": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
    "completed": True,
    "success": True,
}

evaluation = {
    "evaluation_id": "eval-001",
    "overall_score": 0.85,
    "scores": {"task_success": 0.9, "efficiency": 0.8},
}

example = generator.add_from_trace(trace, evaluation, system_prompt="You are a coding assistant.")
print(f"  Added from trace: {example.example_id if example else 'filtered'}")
assert example is not None, "Should be accepted"

# Add more examples
for i in range(5):
    generator.add_from_conversation(
        messages=[
            {"role": "user", "content": f"Question {i}: How do I use Python lists?"},
            {"role": "assistant", "content": f"Answer {i}: Lists in Python are flexible arrays..."},
        ],
        task_type="explanation",
        quality_score=0.7,
    )

stats = generator.get_stats()
print(f"  Total processed: {stats['total_processed']}")
print(f"  Accepted: {stats['accepted']}")
print(f"  Acceptance rate: {stats['acceptance_rate']:.1%}")

print("  [PASS] Generator pipeline works")

# Test 6: Export functionality
print("\n[TEST 6] Export to multiple formats")

# Export to OpenAI format
openai_path = generator.export(TrainingFormat.OPENAI, "test_openai.jsonl")
print(f"  Exported OpenAI format: {openai_path}")

# Export to Alpaca format
alpaca_path = generator.export(TrainingFormat.ALPACA, "test_alpaca.jsonl")
print(f"  Exported Alpaca format: {alpaca_path}")

# Read and verify
with open(openai_path) as f:
    lines = f.readlines()
    print(f"  OpenAI file has {len(lines)} examples")
    first = json.loads(lines[0])
    print(f"  First example has {len(first['messages'])} messages")

print("  [PASS] Export works")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
stats = generator.get_stats()
print(f"Total examples: {stats['total_examples']}")
print(f"Average quality: {stats['average_quality']:.2f}")
print(f"Task types: {stats['task_types']}")
print(f"Acceptance rate: {stats['acceptance_rate']:.1%}")

print("\nAll training data generator tests passed!")
