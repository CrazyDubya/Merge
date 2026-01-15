"""
Club Harness - Unified Agent Orchestration Framework

A consolidation of best practices from:
- TinyTroupe (persona simulation, memory)
- llm-council (multi-LLM consensus)
- LisaSimpson (deliberative planning)
- Agentic-Hub (agent orchestration)
- 12-factor-agents (production principles)
- hivey (swarm intelligence)
- qwen-code (coding agent architecture)

New Features (Jan 2026):
- Self-Evaluation Flywheel System (continuous improvement)
- Training Data Generation (multi-format export)
- Streaming & Tool Call Collection (enhanced streaming)
- Knowledge Base with Semantic Search (RAG support)
- Demographic Persona Generation (Big Five traits)
"""

__version__ = "0.2.0"

# Core components
from .core.agent import Agent, AgentBuilder, AgentResult
from .core.types import Message, AgentState, Goal, Confidence
from .core.config import config

# Memory system
from .memory.memory import Memory, EpisodicMemory, LessonMemory

# Planning
from .planning.goap import Planner as GOAPPlanner, WorldState, Action as GOAPAction

# Verification
from .verification.checks import (
    Check,
    CheckResult,
    VerificationPlan,
    PredicateCheck,
    FactCheck,
    ConfidenceCheck,
)

# Orchestration
from .orchestration.council import Council, CouncilResponse

# Caching
from .caching.semantic_cache import SemanticCache

# New feature modules
from .evaluation import (
    SelfEvaluationLoop,
    FlywheelManager,
    ExecutionTrace,
    EvaluationResult,
    create_evaluation_system,
)

from .training import (
    TrainingDataGenerator,
    TrainingExample,
    TrainingFormat,
)

from .knowledge import (
    SemanticKnowledgeBase,
    RAGHelper,
    Document,
    SearchResult,
)

from .personas import (
    Persona,
    PersonaGenerator,
    PersonaPresets,
    BigFiveTraits,
    Demographics,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "Agent",
    "AgentBuilder",
    "AgentResult",
    "Message",
    "AgentState",
    "Goal",
    "Confidence",
    "config",
    # Memory
    "Memory",
    "EpisodicMemory",
    "LessonMemory",
    # Planning
    "GOAPPlanner",
    "WorldState",
    "GOAPAction",
    # Verification
    "Check",
    "CheckResult",
    "VerificationPlan",
    "PredicateCheck",
    "FactCheck",
    "ConfidenceCheck",
    # Orchestration
    "Council",
    "CouncilResponse",
    # Caching
    "SemanticCache",
    # Evaluation (NEW)
    "SelfEvaluationLoop",
    "FlywheelManager",
    "ExecutionTrace",
    "EvaluationResult",
    "create_evaluation_system",
    # Training (NEW)
    "TrainingDataGenerator",
    "TrainingExample",
    "TrainingFormat",
    # Knowledge (NEW)
    "SemanticKnowledgeBase",
    "RAGHelper",
    "Document",
    "SearchResult",
    # Personas (NEW)
    "Persona",
    "PersonaGenerator",
    "PersonaPresets",
    "BigFiveTraits",
    "Demographics",
]
