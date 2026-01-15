"""
Training Data Generator for Club Harness.

Automatically generates high-quality training data from agent interactions
for fine-tuning or RLHF.

Features:
- Conversation filtering (quality thresholds)
- Format conversion (OpenAI, Anthropic, custom)
- Deduplication and diversity scoring
- Integration with self-evaluation for auto-labeling
"""

import json
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid


class TrainingFormat(Enum):
    """Supported training data formats."""
    OPENAI = "openai"  # OpenAI fine-tuning format
    ANTHROPIC = "anthropic"  # Anthropic format
    ALPACA = "alpaca"  # Alpaca/Stanford format
    SHAREGPT = "sharegpt"  # ShareGPT format
    JSONL = "jsonl"  # Raw JSONL with messages
    CUSTOM = "custom"


@dataclass
class TrainingExample:
    """A single training example."""
    example_id: str
    messages: List[Dict[str, str]]  # role, content
    system_prompt: Optional[str] = None

    # Metadata
    source: str = "club_harness"
    task_type: str = "general"
    quality_score: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)

    # Labels
    labels: List[str] = field(default_factory=list)
    evaluation_id: Optional[str] = None

    # For deduplication
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute content hash for deduplication."""
        content = json.dumps(self.messages, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI fine-tuning format."""
        formatted_messages = []

        if self.system_prompt:
            formatted_messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        formatted_messages.extend(self.messages)

        return {"messages": formatted_messages}

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic format."""
        # Anthropic uses Human/Assistant structure
        conversation = []

        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                conversation.append(f"\n\nHuman: {content}")
            elif role == "assistant":
                conversation.append(f"\n\nAssistant: {content}")

        return {
            "prompt": "".join(conversation[:-1]) + "\n\nAssistant:",
            "completion": self.messages[-1]["content"] if self.messages[-1]["role"] == "assistant" else "",
            "system": self.system_prompt or "",
        }

    def to_alpaca_format(self) -> Dict[str, Any]:
        """Convert to Alpaca format."""
        instruction = ""
        input_text = ""
        output = ""

        for msg in self.messages:
            if msg["role"] == "user":
                if not instruction:
                    instruction = msg["content"]
                else:
                    input_text = msg["content"]
            elif msg["role"] == "assistant":
                output = msg["content"]

        return {
            "instruction": instruction,
            "input": input_text,
            "output": output,
        }

    def to_sharegpt_format(self) -> Dict[str, Any]:
        """Convert to ShareGPT format."""
        conversations = []

        for msg in self.messages:
            role_map = {"user": "human", "assistant": "gpt", "system": "system"}
            conversations.append({
                "from": role_map.get(msg["role"], msg["role"]),
                "value": msg["content"]
            })

        return {
            "id": self.example_id,
            "conversations": conversations,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to generic dictionary."""
        return {
            "example_id": self.example_id,
            "messages": self.messages,
            "system_prompt": self.system_prompt,
            "source": self.source,
            "task_type": self.task_type,
            "quality_score": self.quality_score,
            "created_at": self.created_at.isoformat(),
            "labels": self.labels,
            "evaluation_id": self.evaluation_id,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingExample":
        return cls(
            example_id=data["example_id"],
            messages=data["messages"],
            system_prompt=data.get("system_prompt"),
            source=data.get("source", "club_harness"),
            task_type=data.get("task_type", "general"),
            quality_score=data.get("quality_score", 0.5),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            labels=data.get("labels", []),
            evaluation_id=data.get("evaluation_id"),
            content_hash=data.get("content_hash", ""),
        )


class ConversationFilter:
    """Filter conversations based on various criteria."""

    def __init__(
        self,
        min_turns: int = 1,
        max_turns: int = 20,
        min_message_length: int = 10,
        max_message_length: int = 10000,
        required_roles: Optional[List[str]] = None,
        banned_patterns: Optional[List[str]] = None,
    ):
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.min_message_length = min_message_length
        self.max_message_length = max_message_length
        self.required_roles = required_roles or ["user", "assistant"]
        self.banned_patterns = [re.compile(p, re.IGNORECASE) for p in (banned_patterns or [])]

    def filter(self, messages: List[Dict[str, str]]) -> Tuple[bool, str]:
        """
        Check if messages pass the filter.
        Returns (passed, reason).
        """
        if len(messages) < self.min_turns:
            return False, f"Too few turns: {len(messages)} < {self.min_turns}"

        if len(messages) > self.max_turns:
            return False, f"Too many turns: {len(messages)} > {self.max_turns}"

        # Check required roles
        roles_present = set(m["role"] for m in messages)
        for role in self.required_roles:
            if role not in roles_present:
                return False, f"Missing required role: {role}"

        # Check message lengths
        for msg in messages:
            content = msg.get("content", "")
            if len(content) < self.min_message_length:
                return False, f"Message too short: {len(content)} chars"
            if len(content) > self.max_message_length:
                return False, f"Message too long: {len(content)} chars"

        # Check banned patterns
        full_text = " ".join(m.get("content", "") for m in messages)
        for pattern in self.banned_patterns:
            if pattern.search(full_text):
                return False, f"Contains banned pattern: {pattern.pattern}"

        return True, "passed"


class QualityFilter:
    """Filter based on quality metrics."""

    def __init__(
        self,
        min_quality_score: float = 0.5,
        require_complete: bool = True,
        require_success: bool = False,
    ):
        self.min_quality_score = min_quality_score
        self.require_complete = require_complete
        self.require_success = require_success

    def filter(
        self,
        quality_score: float,
        completed: bool = True,
        success: bool = True,
    ) -> Tuple[bool, str]:
        """Check if quality passes."""
        if quality_score < self.min_quality_score:
            return False, f"Quality too low: {quality_score:.2f} < {self.min_quality_score}"

        if self.require_complete and not completed:
            return False, "Incomplete execution"

        if self.require_success and not success:
            return False, "Unsuccessful execution"

        return True, "passed"


class DiversityScorer:
    """Score diversity of training examples."""

    def __init__(self):
        self.seen_hashes: Set[str] = set()
        self.task_type_counts: Dict[str, int] = {}
        self.ngram_counts: Dict[str, int] = {}

    def score(self, example: TrainingExample) -> float:
        """
        Score the diversity contribution of an example.
        Higher score = more diverse/novel.
        """
        score = 1.0

        # Penalize duplicates
        if example.content_hash in self.seen_hashes:
            return 0.0

        # Penalize over-represented task types
        task_count = self.task_type_counts.get(example.task_type, 0)
        if task_count > 10:
            score *= 10 / task_count

        # Check n-gram novelty
        content = " ".join(m.get("content", "") for m in example.messages)
        words = content.lower().split()

        if len(words) >= 3:
            novel_ngrams = 0
            total_ngrams = 0

            for i in range(len(words) - 2):
                ngram = " ".join(words[i:i+3])
                total_ngrams += 1
                if ngram not in self.ngram_counts:
                    novel_ngrams += 1

            if total_ngrams > 0:
                novelty = novel_ngrams / total_ngrams
                score *= (0.5 + 0.5 * novelty)  # At least 50% of original score

        return min(1.0, score)

    def add(self, example: TrainingExample):
        """Add example to diversity tracking."""
        self.seen_hashes.add(example.content_hash)

        self.task_type_counts[example.task_type] = \
            self.task_type_counts.get(example.task_type, 0) + 1

        content = " ".join(m.get("content", "") for m in example.messages)
        words = content.lower().split()

        if len(words) >= 3:
            for i in range(len(words) - 2):
                ngram = " ".join(words[i:i+3])
                self.ngram_counts[ngram] = self.ngram_counts.get(ngram, 0) + 1

    def is_duplicate(self, example: TrainingExample) -> bool:
        """Check if example is a duplicate."""
        return example.content_hash in self.seen_hashes


class TrainingDataGenerator:
    """
    Main training data generator.

    Generates high-quality training data from agent interactions,
    with filtering, deduplication, and format conversion.
    """

    def __init__(
        self,
        output_path: str = "/tmp/training-data",
        conversation_filter: Optional[ConversationFilter] = None,
        quality_filter: Optional[QualityFilter] = None,
        enable_deduplication: bool = True,
        enable_diversity_scoring: bool = True,
    ):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.conversation_filter = conversation_filter or ConversationFilter()
        self.quality_filter = quality_filter or QualityFilter()
        self.enable_deduplication = enable_deduplication
        self.enable_diversity_scoring = enable_diversity_scoring

        self.diversity_scorer = DiversityScorer() if enable_diversity_scoring else None
        self.examples: List[TrainingExample] = []

        # Stats
        self.stats = {
            "total_processed": 0,
            "filtered_by_conversation": 0,
            "filtered_by_quality": 0,
            "filtered_by_duplicate": 0,
            "accepted": 0,
        }

    def add_from_trace(
        self,
        trace_dict: Dict[str, Any],
        evaluation_dict: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> Optional[TrainingExample]:
        """
        Add training example from an execution trace.

        Args:
            trace_dict: ExecutionTrace as dictionary
            evaluation_dict: Optional EvaluationResult as dictionary
            system_prompt: Optional system prompt to include

        Returns:
            TrainingExample if accepted, None if filtered
        """
        self.stats["total_processed"] += 1

        # Extract messages from trace
        messages = self._extract_messages(trace_dict)

        # Apply conversation filter
        passed, reason = self.conversation_filter.filter(messages)
        if not passed:
            self.stats["filtered_by_conversation"] += 1
            return None

        # Apply quality filter
        quality_score = evaluation_dict.get("overall_score", 0.5) if evaluation_dict else 0.5
        completed = trace_dict.get("completed", True)
        success = trace_dict.get("success", True)

        passed, reason = self.quality_filter.filter(quality_score, completed, success)
        if not passed:
            self.stats["filtered_by_quality"] += 1
            return None

        # Create example
        example = TrainingExample(
            example_id=str(uuid.uuid4())[:12],
            messages=messages,
            system_prompt=system_prompt,
            task_type=trace_dict.get("task_type", "general"),
            quality_score=quality_score,
            labels=self._generate_labels(trace_dict, evaluation_dict),
            evaluation_id=evaluation_dict.get("evaluation_id") if evaluation_dict else None,
        )

        # Check for duplicates
        if self.enable_deduplication and self.diversity_scorer:
            if self.diversity_scorer.is_duplicate(example):
                self.stats["filtered_by_duplicate"] += 1
                return None

        # Add diversity score
        if self.diversity_scorer:
            example.quality_score *= self.diversity_scorer.score(example)
            self.diversity_scorer.add(example)

        self.examples.append(example)
        self.stats["accepted"] += 1

        return example

    def add_from_conversation(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        task_type: str = "general",
        quality_score: float = 0.5,
        labels: Optional[List[str]] = None,
    ) -> Optional[TrainingExample]:
        """Add training example from raw conversation."""
        self.stats["total_processed"] += 1

        # Apply conversation filter
        passed, reason = self.conversation_filter.filter(messages)
        if not passed:
            self.stats["filtered_by_conversation"] += 1
            return None

        # Apply quality filter
        passed, reason = self.quality_filter.filter(quality_score)
        if not passed:
            self.stats["filtered_by_quality"] += 1
            return None

        example = TrainingExample(
            example_id=str(uuid.uuid4())[:12],
            messages=messages,
            system_prompt=system_prompt,
            task_type=task_type,
            quality_score=quality_score,
            labels=labels or [],
        )

        if self.enable_deduplication and self.diversity_scorer:
            if self.diversity_scorer.is_duplicate(example):
                self.stats["filtered_by_duplicate"] += 1
                return None

        if self.diversity_scorer:
            example.quality_score *= self.diversity_scorer.score(example)
            self.diversity_scorer.add(example)

        self.examples.append(example)
        self.stats["accepted"] += 1

        return example

    def _extract_messages(self, trace_dict: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract messages from trace."""
        messages = []

        # Add original task as first user message
        if trace_dict.get("original_task"):
            messages.append({
                "role": "user",
                "content": trace_dict["original_task"]
            })

        # Add turns
        for turn in trace_dict.get("turns", []):
            if turn.get("input"):
                messages.append({"role": "user", "content": turn["input"]})
            if turn.get("output"):
                messages.append({"role": "assistant", "content": turn["output"]})

        # Add final output if different
        if trace_dict.get("final_output") and (not messages or messages[-1].get("content") != trace_dict["final_output"]):
            messages.append({
                "role": "assistant",
                "content": trace_dict["final_output"]
            })

        return messages

    def _generate_labels(
        self,
        trace_dict: Dict[str, Any],
        evaluation_dict: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Generate labels for the example."""
        labels = []

        # Task type label
        if trace_dict.get("task_type"):
            labels.append(f"task:{trace_dict['task_type']}")

        # Success/failure label
        if trace_dict.get("success"):
            labels.append("outcome:success")
        else:
            labels.append("outcome:failure")

        # Quality tier
        if evaluation_dict:
            score = evaluation_dict.get("overall_score", 0)
            if score >= 0.8:
                labels.append("quality:high")
            elif score >= 0.6:
                labels.append("quality:medium")
            else:
                labels.append("quality:low")

            # Add dimension labels
            for dim, dim_score in evaluation_dict.get("scores", {}).items():
                if dim_score >= 0.8:
                    labels.append(f"strong:{dim}")
                elif dim_score <= 0.4:
                    labels.append(f"weak:{dim}")

        return labels

    def export(
        self,
        format: TrainingFormat = TrainingFormat.OPENAI,
        filename: Optional[str] = None,
        min_quality: float = 0.0,
    ) -> str:
        """
        Export training data to file.

        Args:
            format: Output format
            filename: Output filename (auto-generated if None)
            min_quality: Minimum quality score to include

        Returns:
            Path to exported file
        """
        # Filter by quality
        examples = [e for e in self.examples if e.quality_score >= min_quality]

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{format.value}_{timestamp}.jsonl"

        output_file = self.output_path / filename

        with open(output_file, 'w') as f:
            for example in examples:
                if format == TrainingFormat.OPENAI:
                    data = example.to_openai_format()
                elif format == TrainingFormat.ANTHROPIC:
                    data = example.to_anthropic_format()
                elif format == TrainingFormat.ALPACA:
                    data = example.to_alpaca_format()
                elif format == TrainingFormat.SHAREGPT:
                    data = example.to_sharegpt_format()
                else:
                    data = example.to_dict()

                f.write(json.dumps(data) + "\n")

        return str(output_file)

    def export_all_formats(self, min_quality: float = 0.0) -> Dict[str, str]:
        """Export to all supported formats."""
        paths = {}

        for format in [TrainingFormat.OPENAI, TrainingFormat.ANTHROPIC,
                       TrainingFormat.ALPACA, TrainingFormat.SHAREGPT]:
            paths[format.value] = self.export(format, min_quality=min_quality)

        return paths

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            **self.stats,
            "total_examples": len(self.examples),
            "average_quality": sum(e.quality_score for e in self.examples) / len(self.examples) if self.examples else 0,
            "task_types": list(set(e.task_type for e in self.examples)),
            "acceptance_rate": self.stats["accepted"] / max(1, self.stats["total_processed"]),
        }

    def clear(self):
        """Clear all examples and reset stats."""
        self.examples.clear()
        self.diversity_scorer = DiversityScorer() if self.enable_diversity_scoring else None
        self.stats = {
            "total_processed": 0,
            "filtered_by_conversation": 0,
            "filtered_by_quality": 0,
            "filtered_by_duplicate": 0,
            "accepted": 0,
        }


def create_training_generator(
    output_path: str = "/tmp/training-data",
    min_quality: float = 0.5,
    min_turns: int = 1,
    max_turns: int = 20,
) -> TrainingDataGenerator:
    """Convenience function to create a configured generator."""
    return TrainingDataGenerator(
        output_path=output_path,
        conversation_filter=ConversationFilter(min_turns=min_turns, max_turns=max_turns),
        quality_filter=QualityFilter(min_quality_score=min_quality),
    )
