"""Core agent types and state management."""

from .types import (
    AgentState,
    Confidence,
    ConfidenceSource,
    Fact,
    Goal,
    Message,
    ToolCall,
    ToolResult,
)
from .config import Config, config
from .agent import Agent
from .loop_detection import LoopDetector, LoopDetectionResult, detect_loop

__all__ = [
    "AgentState",
    "Confidence",
    "ConfidenceSource",
    "Fact",
    "Goal",
    "Message",
    "ToolCall",
    "ToolResult",
    "Config",
    "config",
    "Agent",
    "LoopDetector",
    "LoopDetectionResult",
    "detect_loop",
]
