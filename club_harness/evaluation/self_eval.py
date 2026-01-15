"""
Self-Evaluation Flywheel System for Club Harness.

This module implements a comprehensive self-evaluation system that enables agents to:
1. Evaluate their own performance against multiple criteria
2. Learn from successes and failures
3. Generate training data from execution traces
4. Create a flywheel effect where improvements compound over time

Adapted from repos/Agentic-Hub/harness/evaluation/self_eval.py
"""

import json
import uuid
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple


class EvaluationDimension(Enum):
    """Dimensions along which agent performance is evaluated."""
    # Task completion
    TASK_SUCCESS = "task_success"
    TASK_COMPLETENESS = "task_completeness"

    # Quality metrics
    CODE_QUALITY = "code_quality"
    REASONING_QUALITY = "reasoning_quality"
    OUTPUT_QUALITY = "output_quality"

    # Efficiency metrics
    EFFICIENCY = "efficiency"
    TOKEN_EFFICIENCY = "token_efficiency"
    TIME_EFFICIENCY = "time_efficiency"

    # Safety and alignment
    SAFETY = "safety"
    ALIGNMENT = "alignment"

    # Communication
    CLARITY = "clarity"
    HELPFULNESS = "helpfulness"

    # Tool use
    TOOL_SELECTION = "tool_selection"
    TOOL_USAGE = "tool_usage"


class EvaluationLevel(Enum):
    """Granularity levels for evaluation."""
    TURN = "turn"
    TASK = "task"
    SESSION = "session"
    AGENT = "agent"


@dataclass
class EvaluationCriteria:
    """Defines criteria for a specific evaluation."""
    dimension: EvaluationDimension
    weight: float = 1.0
    threshold: float = 0.5
    description: str = ""
    custom_evaluator: Optional[Callable] = None


@dataclass
class EvaluationResult:
    """Result of a single evaluation."""
    evaluation_id: str
    timestamp: datetime
    agent_id: str
    level: EvaluationLevel
    target_id: str

    scores: Dict[EvaluationDimension, float]
    overall_score: float

    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]

    training_worthy: bool
    training_category: Optional[str]

    evaluator: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "level": self.level.value,
            "target_id": self.target_id,
            "scores": {k.value: v for k, v in self.scores.items()},
            "overall_score": self.overall_score,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "suggestions": self.suggestions,
            "training_worthy": self.training_worthy,
            "training_category": self.training_category,
            "evaluator": self.evaluator,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        return cls(
            evaluation_id=data["evaluation_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            agent_id=data["agent_id"],
            level=EvaluationLevel(data["level"]),
            target_id=data["target_id"],
            scores={EvaluationDimension(k): v for k, v in data["scores"].items()},
            overall_score=data["overall_score"],
            strengths=data["strengths"],
            weaknesses=data["weaknesses"],
            suggestions=data["suggestions"],
            training_worthy=data["training_worthy"],
            training_category=data.get("training_category"),
            evaluator=data["evaluator"],
            confidence=data["confidence"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExecutionTrace:
    """Complete trace of an agent's execution for evaluation."""
    trace_id: str
    agent_id: str
    session_id: str

    original_task: str
    task_type: str
    constraints: List[str] = field(default_factory=list)

    turns: List[Dict[str, Any]] = field(default_factory=list)
    total_turns: int = 0

    tokens_used: int = 0
    time_taken_ms: float = 0
    commands_executed: int = 0
    tool_calls: int = 0

    completed: bool = False
    success: bool = False
    final_output: str = ""
    error: Optional[str] = None

    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "original_task": self.original_task,
            "task_type": self.task_type,
            "constraints": self.constraints,
            "turns": self.turns,
            "total_turns": self.total_turns,
            "tokens_used": self.tokens_used,
            "time_taken_ms": self.time_taken_ms,
            "commands_executed": self.commands_executed,
            "tool_calls": self.tool_calls,
            "completed": self.completed,
            "success": self.success,
            "final_output": self.final_output[:1000] if self.final_output else "",
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionTrace":
        return cls(
            trace_id=data["trace_id"],
            agent_id=data["agent_id"],
            session_id=data["session_id"],
            original_task=data["original_task"],
            task_type=data["task_type"],
            constraints=data.get("constraints", []),
            turns=data.get("turns", []),
            total_turns=data.get("total_turns", 0),
            tokens_used=data.get("tokens_used", 0),
            time_taken_ms=data.get("time_taken_ms", 0),
            commands_executed=data.get("commands_executed", 0),
            tool_calls=data.get("tool_calls", 0),
            completed=data.get("completed", False),
            success=data.get("success", False),
            final_output=data.get("final_output", ""),
            error=data.get("error"),
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
        )


@dataclass
class LearnedLesson:
    """A lesson learned from evaluation."""
    lesson_id: str
    created_at: datetime

    source_trace_id: str
    source_evaluation_id: str

    lesson_type: str  # success_pattern, failure_pattern, optimization
    category: str
    description: str

    applies_when: List[str]
    task_types: List[str]

    do_this: List[str]
    avoid_this: List[str]

    confidence: float
    times_applied: int = 0
    success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "created_at": self.created_at.isoformat(),
            "source_trace_id": self.source_trace_id,
            "source_evaluation_id": self.source_evaluation_id,
            "lesson_type": self.lesson_type,
            "category": self.category,
            "description": self.description,
            "applies_when": self.applies_when,
            "task_types": self.task_types,
            "do_this": self.do_this,
            "avoid_this": self.avoid_this,
            "confidence": self.confidence,
            "times_applied": self.times_applied,
            "success_rate": self.success_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnedLesson":
        return cls(
            lesson_id=data["lesson_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            source_trace_id=data["source_trace_id"],
            source_evaluation_id=data["source_evaluation_id"],
            lesson_type=data["lesson_type"],
            category=data["category"],
            description=data["description"],
            applies_when=data["applies_when"],
            task_types=data["task_types"],
            do_this=data["do_this"],
            avoid_this=data["avoid_this"],
            confidence=data["confidence"],
            times_applied=data.get("times_applied", 0),
            success_rate=data.get("success_rate", 0.0),
        )


class SelfEvaluator(ABC):
    """Abstract base class for evaluators."""

    @abstractmethod
    def evaluate(
        self,
        trace: ExecutionTrace,
        criteria: List[EvaluationCriteria]
    ) -> EvaluationResult:
        """Evaluate an execution trace."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get evaluator name."""
        pass


class RuleBasedEvaluator(SelfEvaluator):
    """Rule-based evaluation using heuristics."""

    def __init__(self):
        self.rules: Dict[EvaluationDimension, List[Callable]] = {}
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default evaluation rules."""
        self.rules[EvaluationDimension.TASK_SUCCESS] = [
            lambda t: 1.0 if t.success else 0.0,
            lambda t: 1.0 if t.completed else 0.3,
            lambda t: 0.0 if t.error else 1.0,
        ]

        self.rules[EvaluationDimension.TASK_COMPLETENESS] = [
            lambda t: 1.0 if t.completed and t.success else 0.5 if t.completed else 0.2,
            lambda t: min(1.0, len(t.final_output) / 100) if t.final_output else 0.3,
        ]

        self.rules[EvaluationDimension.EFFICIENCY] = [
            lambda t: min(1.0, 5.0 / max(1, t.total_turns)),
            lambda t: min(1.0, 10000 / max(1, t.tokens_used)) if t.tokens_used > 0 else 0.5,
        ]

        self.rules[EvaluationDimension.TOKEN_EFFICIENCY] = [
            lambda t: min(1.0, 5000 / max(1, t.tokens_used)) if t.tokens_used > 0 else 0.5,
        ]

        self.rules[EvaluationDimension.TIME_EFFICIENCY] = [
            lambda t: min(1.0, 30000 / max(1, t.time_taken_ms)) if t.time_taken_ms > 0 else 0.5,
        ]

        self.rules[EvaluationDimension.TOOL_USAGE] = [
            lambda t: min(1.0, (t.commands_executed + t.tool_calls) / max(1, t.total_turns)),
            lambda t: 1.0 if t.tool_calls > 0 or t.commands_executed > 0 else 0.5,
        ]

        self.rules[EvaluationDimension.SAFETY] = [
            lambda t: 0.0 if t.error and "unsafe" in str(t.error).lower() else 1.0,
            lambda t: 1.0,  # Default safe
        ]

        self.rules[EvaluationDimension.OUTPUT_QUALITY] = [
            lambda t: 0.8 if t.final_output and len(t.final_output) > 50 else 0.4,
        ]

    def evaluate(
        self,
        trace: ExecutionTrace,
        criteria: List[EvaluationCriteria]
    ) -> EvaluationResult:
        scores = {}

        for criterion in criteria:
            dim = criterion.dimension
            if dim in self.rules:
                rule_scores = [rule(trace) for rule in self.rules[dim]]
                scores[dim] = sum(rule_scores) / len(rule_scores)
            elif criterion.custom_evaluator:
                scores[dim] = criterion.custom_evaluator(trace)
            else:
                scores[dim] = 0.5

        # Calculate weighted overall score
        total_weight = sum(c.weight for c in criteria)
        overall_score = sum(
            scores.get(c.dimension, 0.5) * c.weight / total_weight
            for c in criteria
        ) if total_weight > 0 else 0.5

        # Generate feedback
        strengths = []
        weaknesses = []
        suggestions = []

        for dim, score in scores.items():
            if score >= 0.8:
                strengths.append(f"Strong {dim.value}: {score:.2f}")
            elif score <= 0.4:
                weaknesses.append(f"Weak {dim.value}: {score:.2f}")
                suggestions.append(f"Improve {dim.value}")

        training_worthy = overall_score >= 0.7 and trace.success

        return EvaluationResult(
            evaluation_id=str(uuid.uuid4())[:12],
            timestamp=datetime.now(),
            agent_id=trace.agent_id,
            level=EvaluationLevel.TASK,
            target_id=trace.trace_id,
            scores=scores,
            overall_score=overall_score,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            training_worthy=training_worthy,
            training_category="success" if trace.success else "failure",
            evaluator=self.get_name(),
            confidence=0.7,
        )

    def get_name(self) -> str:
        return "rule_based"


class LLMEvaluator(SelfEvaluator):
    """LLM-based evaluation using model self-critique."""

    def __init__(self, llm_router=None):
        self.llm_router = llm_router

    def evaluate(
        self,
        trace: ExecutionTrace,
        criteria: List[EvaluationCriteria]
    ) -> EvaluationResult:
        """Use LLM to evaluate the trace."""
        if not self.llm_router:
            return self._fallback_evaluation(trace, criteria)

        prompt = self._build_evaluation_prompt(trace, criteria)

        try:
            response = self.llm_router.chat(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                tier="free",
            )
            return self._parse_llm_evaluation(response.content, trace, criteria)
        except Exception as e:
            return self._fallback_evaluation(trace, criteria)

    def _build_evaluation_prompt(
        self,
        trace: ExecutionTrace,
        criteria: List[EvaluationCriteria]
    ) -> str:
        criteria_list = "\n".join([
            f"- {c.dimension.value}: {c.description or c.dimension.value}"
            for c in criteria
        ])

        return f"""Evaluate this agent execution. Respond with JSON only.

TASK: {trace.original_task}

EXECUTION:
- Turns: {trace.total_turns}
- Commands: {trace.commands_executed}
- Tool calls: {trace.tool_calls}
- Completed: {trace.completed}
- Success: {trace.success}
- Error: {trace.error or 'None'}

OUTPUT (truncated):
{trace.final_output[:500] if trace.final_output else 'None'}

CRITERIA:
{criteria_list}

Respond with this exact JSON format:
{{"scores": {{"task_success": 0.8, "efficiency": 0.7}}, "overall_score": 0.75, "strengths": ["good"], "weaknesses": ["slow"], "suggestions": ["faster"], "training_worthy": true, "confidence": 0.8}}"""

    def _parse_llm_evaluation(
        self,
        content: str,
        trace: ExecutionTrace,
        criteria: List[EvaluationCriteria]
    ) -> EvaluationResult:
        """Parse LLM response into evaluation result."""
        import re

        try:
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found")

            scores = {}
            for c in criteria:
                dim_name = c.dimension.value
                if dim_name in data.get("scores", {}):
                    scores[c.dimension] = float(data["scores"][dim_name])
                else:
                    scores[c.dimension] = 0.5

            return EvaluationResult(
                evaluation_id=str(uuid.uuid4())[:12],
                timestamp=datetime.now(),
                agent_id=trace.agent_id,
                level=EvaluationLevel.TASK,
                target_id=trace.trace_id,
                scores=scores,
                overall_score=float(data.get("overall_score", 0.5)),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                suggestions=data.get("suggestions", []),
                training_worthy=data.get("training_worthy", False),
                training_category="llm_evaluated",
                evaluator=self.get_name(),
                confidence=float(data.get("confidence", 0.8)),
            )
        except Exception:
            return self._fallback_evaluation(trace, criteria)

    def _fallback_evaluation(
        self,
        trace: ExecutionTrace,
        criteria: List[EvaluationCriteria]
    ) -> EvaluationResult:
        """Fallback to rule-based when LLM unavailable."""
        return RuleBasedEvaluator().evaluate(trace, criteria)

    def get_name(self) -> str:
        return "llm_evaluator"


class SelfEvaluationLoop:
    """
    The main self-evaluation loop that drives continuous improvement.

    Implements a flywheel effect:
    1. Execute tasks
    2. Evaluate performance
    3. Extract lessons
    4. Apply lessons to future tasks
    5. Generate training data
    6. Repeat
    """

    def __init__(
        self,
        base_path: str = "/tmp/club-harness-eval",
        llm_router=None,
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.evaluators: Dict[str, SelfEvaluator] = {
            "rule_based": RuleBasedEvaluator(),
            "llm": LLMEvaluator(llm_router),
        }

        self.default_criteria = [
            EvaluationCriteria(EvaluationDimension.TASK_SUCCESS, weight=2.0),
            EvaluationCriteria(EvaluationDimension.TASK_COMPLETENESS, weight=1.5),
            EvaluationCriteria(EvaluationDimension.EFFICIENCY, weight=1.0),
            EvaluationCriteria(EvaluationDimension.TOOL_USAGE, weight=1.0),
            EvaluationCriteria(EvaluationDimension.OUTPUT_QUALITY, weight=1.5),
            EvaluationCriteria(EvaluationDimension.SAFETY, weight=2.0),
        ]

        # In-memory storage
        self.traces: Dict[str, ExecutionTrace] = {}
        self.evaluations: Dict[str, EvaluationResult] = {}
        self.lessons: Dict[str, LearnedLesson] = {}

        # Indexes
        self.lessons_by_category: Dict[str, List[str]] = {}
        self.lessons_by_task_type: Dict[str, List[str]] = {}

        # Performance tracking
        self.agent_performance: Dict[str, Dict[str, Any]] = {}

        self._lock = threading.Lock()

        # Load persisted data
        self._load_data()

    def _load_data(self):
        """Load persisted data from disk."""
        traces_file = self.base_path / "traces.json"
        if traces_file.exists():
            try:
                with open(traces_file) as f:
                    data = json.load(f)
                    for trace_data in data:
                        trace = ExecutionTrace.from_dict(trace_data)
                        self.traces[trace.trace_id] = trace
            except Exception:
                pass

        lessons_file = self.base_path / "lessons.json"
        if lessons_file.exists():
            try:
                with open(lessons_file) as f:
                    data = json.load(f)
                    for lesson_data in data:
                        lesson = LearnedLesson.from_dict(lesson_data)
                        self.lessons[lesson.lesson_id] = lesson
                        self._index_lesson(lesson)
            except Exception:
                pass

    def _save_data(self):
        """Persist data to disk."""
        try:
            traces_file = self.base_path / "traces.json"
            with open(traces_file, 'w') as f:
                json.dump([t.to_dict() for t in self.traces.values()], f)

            lessons_file = self.base_path / "lessons.json"
            with open(lessons_file, 'w') as f:
                json.dump([l.to_dict() for l in self.lessons.values()], f)
        except Exception:
            pass

    def _index_lesson(self, lesson: LearnedLesson):
        """Index a lesson for efficient retrieval."""
        if lesson.category not in self.lessons_by_category:
            self.lessons_by_category[lesson.category] = []
        if lesson.lesson_id not in self.lessons_by_category[lesson.category]:
            self.lessons_by_category[lesson.category].append(lesson.lesson_id)

        for task_type in lesson.task_types:
            if task_type not in self.lessons_by_task_type:
                self.lessons_by_task_type[task_type] = []
            if lesson.lesson_id not in self.lessons_by_task_type[task_type]:
                self.lessons_by_task_type[task_type].append(lesson.lesson_id)

    def record_trace(self, trace: ExecutionTrace) -> str:
        """Record an execution trace for evaluation."""
        with self._lock:
            self.traces[trace.trace_id] = trace
            self._save_data()
        return trace.trace_id

    def create_trace(
        self,
        agent_id: str,
        task: str,
        task_type: str = "general",
        session_id: Optional[str] = None,
    ) -> ExecutionTrace:
        """Create a new execution trace."""
        return ExecutionTrace(
            trace_id=str(uuid.uuid4())[:12],
            agent_id=agent_id,
            session_id=session_id or str(uuid.uuid4())[:8],
            original_task=task,
            task_type=task_type,
            started_at=datetime.now(),
        )

    def evaluate_trace(
        self,
        trace_id: str,
        evaluator_name: str = "rule_based",
        criteria: Optional[List[EvaluationCriteria]] = None,
    ) -> EvaluationResult:
        """Evaluate a recorded trace."""
        trace = self.traces.get(trace_id)
        if not trace:
            raise ValueError(f"Trace not found: {trace_id}")

        evaluator = self.evaluators.get(evaluator_name, self.evaluators["rule_based"])
        criteria = criteria or self.default_criteria
        result = evaluator.evaluate(trace, criteria)

        with self._lock:
            self.evaluations[result.evaluation_id] = result
            self._update_performance(result)

        return result

    def _update_performance(self, result: EvaluationResult):
        """Update agent performance tracking."""
        agent_id = result.agent_id
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = {
                "total_evaluations": 0,
                "average_score": 0.0,
                "score_history": [],
                "dimension_averages": {},
                "improvement_trend": 0.0,
                "training_worthy_count": 0,
            }

        perf = self.agent_performance[agent_id]
        perf["total_evaluations"] += 1
        perf["score_history"].append(result.overall_score)

        if result.training_worthy:
            perf["training_worthy_count"] += 1

        n = perf["total_evaluations"]
        perf["average_score"] = (
            (perf["average_score"] * (n - 1) + result.overall_score) / n
        )

        for dim, score in result.scores.items():
            dim_key = dim.value
            if dim_key not in perf["dimension_averages"]:
                perf["dimension_averages"][dim_key] = {"sum": 0, "count": 0}
            perf["dimension_averages"][dim_key]["sum"] += score
            perf["dimension_averages"][dim_key]["count"] += 1

        # Calculate improvement trend
        history = perf["score_history"]
        if len(history) >= 20:
            recent = sum(history[-10:]) / 10
            previous = sum(history[-20:-10]) / 10
            perf["improvement_trend"] = recent - previous

    def extract_lesson(
        self,
        evaluation_id: str,
        lesson_type: str = "auto",
    ) -> Optional[LearnedLesson]:
        """Extract a lesson from an evaluation."""
        result = self.evaluations.get(evaluation_id)
        if not result:
            return None

        trace = self.traces.get(result.target_id)
        if not trace:
            return None

        if lesson_type == "auto":
            if result.overall_score >= 0.8:
                lesson_type = "success_pattern"
            elif result.overall_score <= 0.4:
                lesson_type = "failure_pattern"
            else:
                lesson_type = "optimization"

        lesson = LearnedLesson(
            lesson_id=str(uuid.uuid4())[:12],
            created_at=datetime.now(),
            source_trace_id=trace.trace_id,
            source_evaluation_id=evaluation_id,
            lesson_type=lesson_type,
            category=trace.task_type or "general",
            description=self._generate_lesson_description(result, trace),
            applies_when=[trace.task_type] if trace.task_type else ["general"],
            task_types=[trace.task_type] if trace.task_type else [],
            do_this=result.strengths if lesson_type == "success_pattern" else result.suggestions,
            avoid_this=result.weaknesses if lesson_type == "failure_pattern" else [],
            confidence=result.confidence * result.overall_score,
        )

        with self._lock:
            self.lessons[lesson.lesson_id] = lesson
            self._index_lesson(lesson)
            self._save_data()

        return lesson

    def _generate_lesson_description(
        self,
        result: EvaluationResult,
        trace: ExecutionTrace
    ) -> str:
        """Generate a lesson description from evaluation."""
        if result.overall_score >= 0.8:
            return f"Successful approach to {trace.task_type or 'task'}: {trace.original_task[:80]}"
        elif result.overall_score <= 0.4:
            weaknesses = "; ".join(result.weaknesses[:2]) if result.weaknesses else "unknown issues"
            return f"Failed approach: Avoid {weaknesses}"
        else:
            suggestions = "; ".join(result.suggestions[:2]) if result.suggestions else "optimization needed"
            return f"Partial success: {suggestions}"

    def get_relevant_lessons(
        self,
        task_type: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List[LearnedLesson]:
        """Get lessons relevant to a task."""
        lesson_ids = set()

        if task_type and task_type in self.lessons_by_task_type:
            lesson_ids.update(self.lessons_by_task_type[task_type])

        if category and category in self.lessons_by_category:
            lesson_ids.update(self.lessons_by_category[category])

        if not lesson_ids:
            lessons = sorted(
                self.lessons.values(),
                key=lambda l: l.confidence * (l.success_rate + 0.1),
                reverse=True,
            )
            return lessons[:limit]

        lessons = [self.lessons[lid] for lid in lesson_ids if lid in self.lessons]
        lessons.sort(key=lambda l: l.confidence * (l.success_rate + 0.1), reverse=True)
        return lessons[:limit]

    def apply_lesson(self, lesson_id: str, success: bool):
        """Record that a lesson was applied with outcome."""
        if lesson_id in self.lessons:
            lesson = self.lessons[lesson_id]
            lesson.times_applied += 1
            alpha = 0.3
            lesson.success_rate = (
                alpha * (1.0 if success else 0.0) +
                (1 - alpha) * lesson.success_rate
            )
            self._save_data()

    def get_flywheel_prompt(
        self,
        task: str,
        task_type: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> str:
        """
        Generate a prompt addendum with learned lessons.
        This is the flywheel effect - past learnings improve future performance.
        """
        sections = []

        lessons = self.get_relevant_lessons(task_type=task_type, limit=5)

        if lessons:
            success_lessons = [l for l in lessons if l.lesson_type == "success_pattern"]
            failure_lessons = [l for l in lessons if l.lesson_type == "failure_pattern"]

            if success_lessons:
                sections.append("# Learned Success Patterns")
                for l in success_lessons[:3]:
                    sections.append(f"- {l.description}")
                    if l.do_this:
                        sections.append(f"  DO: {'; '.join(l.do_this[:2])}")

            if failure_lessons:
                sections.append("\n# Known Failure Patterns to Avoid")
                for l in failure_lessons[:3]:
                    sections.append(f"- {l.description}")
                    if l.avoid_this:
                        sections.append(f"  AVOID: {'; '.join(l.avoid_this[:2])}")

        if agent_id and agent_id in self.agent_performance:
            perf = self.agent_performance[agent_id]
            sections.append(f"\n# Performance Context")
            sections.append(f"- Average score: {perf['average_score']:.2f}")
            if perf['improvement_trend'] != 0:
                sections.append(f"- Trend: {perf['improvement_trend']:+.2f}")

            weak_dims = sorted(
                perf["dimension_averages"].items(),
                key=lambda x: x[1]["sum"] / max(1, x[1]["count"])
            )[:2]
            if weak_dims:
                sections.append(f"- Focus areas: {', '.join(d[0] for d in weak_dims)}")

        return "\n".join(sections) if sections else ""

    def get_agent_report(self, agent_id: str) -> Dict[str, Any]:
        """Get performance report for an agent."""
        if agent_id not in self.agent_performance:
            return {"error": "No performance data"}

        perf = self.agent_performance[agent_id]

        dim_avgs = {
            k: v["sum"] / max(1, v["count"])
            for k, v in perf["dimension_averages"].items()
        }

        return {
            "agent_id": agent_id,
            "total_evaluations": perf["total_evaluations"],
            "average_score": perf["average_score"],
            "improvement_trend": perf["improvement_trend"],
            "training_worthy_ratio": perf["training_worthy_count"] / max(1, perf["total_evaluations"]),
            "dimension_averages": dim_avgs,
            "strongest_dimensions": sorted(dim_avgs.items(), key=lambda x: x[1], reverse=True)[:3],
            "weakest_dimensions": sorted(dim_avgs.items(), key=lambda x: x[1])[:3],
            "recent_scores": perf["score_history"][-10:],
        }

    def get_training_data(
        self,
        min_score: float = 0.7,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get training-worthy traces for model improvement."""
        training_data = []

        for eval_result in self.evaluations.values():
            if eval_result.training_worthy and eval_result.overall_score >= min_score:
                trace = self.traces.get(eval_result.target_id)
                if trace:
                    training_data.append({
                        "trace": trace.to_dict(),
                        "evaluation": eval_result.to_dict(),
                        "type": "successful_execution",
                    })

        training_data.sort(key=lambda x: x["evaluation"]["overall_score"], reverse=True)
        return training_data[:limit]


class FlywheelManager:
    """
    Manages the flywheel effect across multiple agents.

    The flywheel:
    1. Agents execute tasks
    2. Self-evaluation scores performance
    3. High-scoring traces become training data
    4. Lessons are extracted from all evaluations
    5. Future tasks benefit from lessons
    6. Better performance leads to better training data
    7. Cycle continues with compounding improvements
    """

    def __init__(
        self,
        eval_loop: SelfEvaluationLoop,
        auto_evaluate: bool = True,
        auto_extract_lessons: bool = True,
        min_score_for_lesson: float = 0.3,
    ):
        self.eval_loop = eval_loop
        self.auto_evaluate = auto_evaluate
        self.auto_extract_lessons = auto_extract_lessons
        self.min_score_for_lesson = min_score_for_lesson

        self.metrics = {
            "traces_recorded": 0,
            "evaluations_performed": 0,
            "lessons_extracted": 0,
            "lessons_applied": 0,
            "training_data_generated": 0,
            "average_score_trend": [],
        }

    def process_execution(
        self,
        agent_id: str,
        task: str,
        task_type: str,
        turns: List[Dict[str, Any]],
        success: bool,
        final_output: str,
        session_id: Optional[str] = None,
        error: Optional[str] = None,
        tokens_used: int = 0,
        time_taken_ms: float = 0,
    ) -> Dict[str, Any]:
        """
        Process a complete execution through the flywheel.
        Returns evaluation result and any extracted lessons.
        """
        trace = ExecutionTrace(
            trace_id=str(uuid.uuid4())[:12],
            agent_id=agent_id,
            session_id=session_id or str(uuid.uuid4())[:8],
            original_task=task,
            task_type=task_type,
            turns=turns,
            total_turns=len(turns),
            tokens_used=tokens_used,
            time_taken_ms=time_taken_ms,
            commands_executed=sum(1 for t in turns if t.get("commands")),
            tool_calls=sum(1 for t in turns if t.get("tool_calls")),
            completed=True,
            success=success,
            final_output=final_output,
            error=error,
            started_at=datetime.now() - timedelta(milliseconds=time_taken_ms),
            completed_at=datetime.now(),
        )

        self.eval_loop.record_trace(trace)
        self.metrics["traces_recorded"] += 1

        result = {
            "trace_id": trace.trace_id,
            "evaluation": None,
            "lessons": [],
        }

        if self.auto_evaluate:
            evaluation = self.eval_loop.evaluate_trace(trace.trace_id)
            self.metrics["evaluations_performed"] += 1
            result["evaluation"] = evaluation.to_dict()

            self.metrics["average_score_trend"].append(evaluation.overall_score)
            if len(self.metrics["average_score_trend"]) > 100:
                self.metrics["average_score_trend"] = self.metrics["average_score_trend"][-100:]

            if evaluation.training_worthy:
                self.metrics["training_data_generated"] += 1

            if self.auto_extract_lessons:
                if evaluation.overall_score >= 0.7 or evaluation.overall_score <= self.min_score_for_lesson:
                    lesson = self.eval_loop.extract_lesson(evaluation.evaluation_id)
                    if lesson:
                        self.metrics["lessons_extracted"] += 1
                        result["lessons"].append(lesson.to_dict())

        return result

    def get_enhanced_prompt(
        self,
        task: str,
        task_type: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> str:
        """Get an enhanced prompt that includes flywheel learnings."""
        return self.eval_loop.get_flywheel_prompt(task, task_type, agent_id)

    def record_lesson_application(self, lesson_id: str, success: bool):
        """Record that a lesson was applied."""
        self.eval_loop.apply_lesson(lesson_id, success)
        self.metrics["lessons_applied"] += 1

    def get_flywheel_status(self) -> Dict[str, Any]:
        """Get current flywheel status and health."""
        avg_score = 0.0
        trend = self.metrics["average_score_trend"]

        if trend:
            avg_score = sum(trend) / len(trend)

            if len(trend) >= 20:
                recent = sum(trend[-10:]) / 10
                previous = sum(trend[-20:-10]) / 10
                trend_direction = "improving" if recent > previous else "declining" if recent < previous else "stable"
            else:
                trend_direction = "insufficient_data"
        else:
            trend_direction = "no_data"

        return {
            "metrics": self.metrics,
            "average_score": avg_score,
            "trend_direction": trend_direction,
            "total_lessons": len(self.eval_loop.lessons),
            "training_data_ratio": (
                self.metrics["training_data_generated"] /
                max(1, self.metrics["evaluations_performed"])
            ),
            "health": "healthy" if avg_score >= 0.6 else "needs_attention" if avg_score > 0 else "no_data",
        }


def create_evaluation_system(
    base_path: str = "/tmp/club-harness-eval",
    llm_router=None,
) -> Tuple[SelfEvaluationLoop, FlywheelManager]:
    """Create and configure evaluation system."""
    eval_loop = SelfEvaluationLoop(base_path, llm_router)
    flywheel = FlywheelManager(eval_loop)
    return eval_loop, flywheel
