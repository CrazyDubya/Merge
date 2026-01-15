"""
Evaluation module for Club Harness.

Provides self-evaluation, training data generation, and flywheel learning.
"""

from .self_eval import (
    EvaluationDimension,
    EvaluationLevel,
    EvaluationCriteria,
    EvaluationResult,
    ExecutionTrace,
    SelfEvaluator,
    RuleBasedEvaluator,
    LLMEvaluator,
    SelfEvaluationLoop,
    FlywheelManager,
    create_evaluation_system,
)

__all__ = [
    "EvaluationDimension",
    "EvaluationLevel",
    "EvaluationCriteria",
    "EvaluationResult",
    "ExecutionTrace",
    "SelfEvaluator",
    "RuleBasedEvaluator",
    "LLMEvaluator",
    "SelfEvaluationLoop",
    "FlywheelManager",
    "create_evaluation_system",
]
