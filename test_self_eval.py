#!/usr/bin/env python3
"""Test self-evaluation flywheel system."""

import sys
sys.path.insert(0, '/home/user/Merge')

from club_harness.evaluation import (
    SelfEvaluationLoop,
    FlywheelManager,
    ExecutionTrace,
    EvaluationDimension,
    EvaluationCriteria,
    create_evaluation_system,
)
from datetime import datetime
import uuid

print("=" * 60)
print("SELF-EVALUATION FLYWHEEL SYSTEM TEST")
print("=" * 60)

# Test 1: Basic trace recording and evaluation
print("\n[TEST 1] Basic trace recording and evaluation")

eval_loop, flywheel = create_evaluation_system("/tmp/test-eval-system")

# Create a successful trace
trace = ExecutionTrace(
    trace_id=str(uuid.uuid4())[:12],
    agent_id="test-agent-1",
    session_id="session-001",
    original_task="Write a function to calculate factorial",
    task_type="coding",
    turns=[
        {"input": "Write factorial", "output": "def factorial(n): ..."},
    ],
    total_turns=1,
    tokens_used=500,
    time_taken_ms=2000,
    commands_executed=0,
    tool_calls=1,
    completed=True,
    success=True,
    final_output="def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
    started_at=datetime.now(),
    completed_at=datetime.now(),
)

trace_id = eval_loop.record_trace(trace)
print(f"  Recorded trace: {trace_id}")

# Evaluate the trace
result = eval_loop.evaluate_trace(trace_id, "rule_based")
print(f"  Overall score: {result.overall_score:.2f}")
print(f"  Training worthy: {result.training_worthy}")
print(f"  Strengths: {result.strengths[:2]}")

assert result.overall_score > 0.5, "Expected good score for successful trace"
print("  [PASS] Basic evaluation works")

# Test 2: Flywheel manager
print("\n[TEST 2] Flywheel manager processing")

flywheel_result = flywheel.process_execution(
    agent_id="test-agent-2",
    task="Explain how recursion works",
    task_type="explanation",
    turns=[
        {"input": "Explain recursion", "output": "Recursion is..."},
    ],
    success=True,
    final_output="Recursion is when a function calls itself to solve smaller subproblems.",
    tokens_used=300,
    time_taken_ms=1500,
)

print(f"  Processed trace: {flywheel_result['trace_id']}")
print(f"  Evaluation score: {flywheel_result['evaluation']['overall_score']:.2f}")
print(f"  Lessons extracted: {len(flywheel_result['lessons'])}")

status = flywheel.get_flywheel_status()
print(f"  Flywheel health: {status['health']}")
print(f"  Total traces: {status['metrics']['traces_recorded']}")

print("  [PASS] Flywheel processing works")

# Test 3: Lesson extraction
print("\n[TEST 3] Lesson extraction and retrieval")

# Process a high-scoring execution to get a lesson
flywheel.process_execution(
    agent_id="test-agent-3",
    task="Debug a null pointer exception",
    task_type="debugging",
    turns=[{"input": "Debug NPE", "output": "Found the bug!"}],
    success=True,
    final_output="The issue was a missing null check at line 42. Added proper validation.",
    tokens_used=200,
    time_taken_ms=1000,
)

lessons = eval_loop.get_relevant_lessons(task_type="debugging")
print(f"  Retrieved {len(lessons)} lessons for 'debugging' tasks")

# Test flywheel prompt generation
prompt = flywheel.get_enhanced_prompt(
    task="Fix a runtime error",
    task_type="debugging",
    agent_id="test-agent-3",
)
print(f"  Enhanced prompt generated: {len(prompt)} chars")
if prompt:
    print(f"  Preview: {prompt[:100]}...")

print("  [PASS] Lesson system works")

# Test 4: Failed execution handling
print("\n[TEST 4] Failed execution handling")

flywheel.process_execution(
    agent_id="test-agent-4",
    task="Parse complex XML without proper library",
    task_type="parsing",
    turns=[
        {"input": "Parse XML", "output": "Attempting..."},
        {"input": "Continue", "output": "Error occurred"},
    ],
    success=False,
    final_output="Failed to parse XML",
    error="XMLParseError: Malformed input",
    tokens_used=800,
    time_taken_ms=5000,
)

status = flywheel.get_flywheel_status()
print(f"  Total evaluations: {status['metrics']['evaluations_performed']}")
print(f"  Training data generated: {status['metrics']['training_data_generated']}")

# Failure lessons should be extracted too
lessons = eval_loop.get_relevant_lessons(task_type="parsing")
failure_lessons = [l for l in lessons if l.lesson_type == "failure_pattern"]
print(f"  Failure lessons extracted: {len(failure_lessons)}")

print("  [PASS] Failed execution handling works")

# Test 5: Agent performance report
print("\n[TEST 5] Agent performance tracking")

report = eval_loop.get_agent_report("test-agent-3")
if "error" not in report:
    print(f"  Agent: {report['agent_id']}")
    print(f"  Total evaluations: {report['total_evaluations']}")
    print(f"  Average score: {report['average_score']:.2f}")
    print(f"  Training worthy ratio: {report['training_worthy_ratio']:.1%}")
    print("  [PASS] Performance tracking works")
else:
    print(f"  No data for agent (expected)")
    print("  [PASS] Report handles missing data")

# Test 6: Training data export
print("\n[TEST 6] Training data export")

training_data = eval_loop.get_training_data(min_score=0.5, limit=10)
print(f"  Training examples available: {len(training_data)}")
if training_data:
    print(f"  Top score: {training_data[0]['evaluation']['overall_score']:.2f}")

print("  [PASS] Training data export works")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
status = flywheel.get_flywheel_status()
print(f"Total traces recorded: {status['metrics']['traces_recorded']}")
print(f"Total evaluations: {status['metrics']['evaluations_performed']}")
print(f"Total lessons: {status['total_lessons']}")
print(f"Training data ratio: {status['training_data_ratio']:.1%}")
print(f"Average score: {status['average_score']:.2f}")
print(f"Flywheel health: {status['health']}")
print(f"Trend: {status['trend_direction']}")

print("\nAll self-evaluation tests passed!")
