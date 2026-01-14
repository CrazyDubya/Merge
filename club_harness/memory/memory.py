"""
Memory systems for Club Harness.

Combines patterns from:
- TinyTroupe: Episodic memory with consolidation, semantic memory with embeddings
- LisaSimpson: Lesson extraction with confidence decay
- 12-factor: Event-sourced state management
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import json


class MemoryType(Enum):
    """Types of memory entries."""
    OBSERVATION = "observation"
    ACTION = "action"
    THOUGHT = "thought"
    LESSON = "lesson"
    FACT = "fact"
    CONVERSATION = "conversation"


@dataclass
class MemoryEntry:
    """Single memory entry with metadata."""
    content: str
    memory_type: MemoryType
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5  # 0.0 to 1.0
    source: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "type": self.memory_type.value,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "source": self.source,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            content=data["content"],
            memory_type=MemoryType(data["type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            importance=data.get("importance", 0.5),
            source=data.get("source"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Episode:
    """
    An episode is a sequence of related memories.

    Inspired by TinyTroupe's episodic memory consolidation.
    """
    id: str
    title: str
    entries: List[MemoryEntry] = field(default_factory=list)
    summary: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    consolidated: bool = False
    lessons_extracted: List[str] = field(default_factory=list)

    def add_entry(self, entry: MemoryEntry) -> None:
        """Add an entry to this episode."""
        self.entries.append(entry)
        self.end_time = entry.timestamp

    def duration_minutes(self) -> float:
        """Get episode duration in minutes."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 60
        return 0.0


@dataclass
class Lesson:
    """
    A lesson learned from experience.

    Inspired by LisaSimpson's lesson extraction system.
    """
    id: str
    situation: str  # When this lesson applies
    insight: str    # What was learned
    outcome: str    # success, failure, partial
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)
    application_count: int = 0
    last_applied: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

    def apply(self) -> None:
        """Record that this lesson was applied."""
        self.application_count += 1
        self.last_applied = datetime.now()

    def reinforce(self, success: bool) -> None:
        """Reinforce or weaken the lesson based on outcome."""
        if success:
            self.confidence = min(1.0, self.confidence + 0.1)
        else:
            self.confidence = max(0.0, self.confidence - 0.15)

    def is_stale(self, max_age_days: int = 30) -> bool:
        """Check if lesson is stale and should be reviewed."""
        age = (datetime.now() - self.created_at).days
        return age > max_age_days and self.application_count < 3

    def decay(self, half_life_days: float = 14.0) -> float:
        """Apply time-based decay to confidence."""
        age_days = (datetime.now() - self.created_at).total_seconds() / 86400
        decay_factor = 0.5 ** (age_days / half_life_days)
        return self.confidence * decay_factor


class EpisodicMemory:
    """
    Episodic memory with automatic consolidation.

    Features:
    - Fixed prefix (always keep recent N memories)
    - Lookback window for context building
    - Automatic consolidation of old episodes
    - Bounded growth
    """

    def __init__(
        self,
        fixed_prefix_length: int = 20,
        lookback_length: int = 20,
        max_episodes: int = 100,
        min_episode_length: int = 5,
        max_episode_length: int = 50,
    ):
        self.fixed_prefix_length = fixed_prefix_length
        self.lookback_length = lookback_length
        self.max_episodes = max_episodes
        self.min_episode_length = min_episode_length
        self.max_episode_length = max_episode_length

        self.current_episode: Optional[Episode] = None
        self.episodes: List[Episode] = []
        self._entry_count = 0

    def start_episode(self, title: str) -> Episode:
        """Start a new episode."""
        # Finish current episode if exists
        if self.current_episode:
            self.finish_episode()

        episode_id = f"ep_{self._entry_count}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.current_episode = Episode(id=episode_id, title=title)
        return self.current_episode

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.OBSERVATION,
        importance: float = 0.5,
        **kwargs,
    ) -> MemoryEntry:
        """Add a memory to the current episode."""
        if not self.current_episode:
            self.start_episode("Auto-started episode")

        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            importance=importance,
            **kwargs,
        )
        self.current_episode.add_entry(entry)
        self._entry_count += 1

        # Auto-finish if episode is too long
        if len(self.current_episode.entries) >= self.max_episode_length:
            self.finish_episode()

        return entry

    def finish_episode(self, summary: Optional[str] = None) -> Optional[Episode]:
        """Finish the current episode."""
        if not self.current_episode:
            return None

        episode = self.current_episode
        episode.end_time = datetime.now()

        if summary:
            episode.summary = summary

        # Only save if episode has enough content
        if len(episode.entries) >= self.min_episode_length:
            self.episodes.append(episode)
            self._prune_old_episodes()

        self.current_episode = None
        return episode

    def _prune_old_episodes(self) -> None:
        """Remove old episodes to maintain bounded growth."""
        while len(self.episodes) > self.max_episodes:
            # Remove oldest non-consolidated episode
            for i, ep in enumerate(self.episodes):
                if ep.consolidated:
                    self.episodes.pop(i)
                    break
            else:
                # All consolidated, remove oldest
                self.episodes.pop(0)

    def get_recent_memories(self, n: Optional[int] = None) -> List[MemoryEntry]:
        """Get the most recent memories across all episodes."""
        n = n or self.lookback_length
        all_entries = []

        # Current episode entries
        if self.current_episode:
            all_entries.extend(self.current_episode.entries)

        # Past episode entries (most recent first)
        for episode in reversed(self.episodes):
            all_entries.extend(reversed(episode.entries))

        # Return most recent n
        return list(reversed(all_entries[:n]))

    def search_memories(
        self,
        query: str,
        limit: int = 10,
        memory_types: Optional[List[MemoryType]] = None,
    ) -> List[MemoryEntry]:
        """
        Simple keyword search through memories.

        For production, integrate with semantic search.
        """
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for episode in self.episodes:
            for entry in episode.entries:
                if memory_types and entry.memory_type not in memory_types:
                    continue

                content_lower = entry.content.lower()
                # Simple relevance score: count matching words
                content_words = set(content_lower.split())
                overlap = len(query_words & content_words)

                if overlap > 0 or query_lower in content_lower:
                    results.append((entry, overlap + (1 if query_lower in content_lower else 0)))

        # Sort by relevance and return top N
        results.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in results[:limit]]

    def to_context_string(self, max_entries: int = 20) -> str:
        """Format recent memories as context for LLM."""
        memories = self.get_recent_memories(max_entries)
        if not memories:
            return "No previous memories."

        lines = ["Recent memories:"]
        for entry in memories:
            time_str = entry.timestamp.strftime("%H:%M")
            lines.append(f"  [{time_str}] ({entry.memory_type.value}) {entry.content}")

        return "\n".join(lines)


class LessonMemory:
    """
    Long-term lesson storage with retrieval and reinforcement.

    Inspired by LisaSimpson's learning system.
    """

    def __init__(self):
        self.lessons: Dict[str, Lesson] = {}
        self._lesson_count = 0

    def add_lesson(
        self,
        situation: str,
        insight: str,
        outcome: str = "success",
        confidence: float = 0.5,
        evidence: Optional[List[str]] = None,
    ) -> Lesson:
        """Add a new lesson."""
        # Check for contradicting lessons
        contradicting = self.find_contradicting(situation, insight)
        if contradicting:
            # Higher confidence wins
            if confidence <= contradicting.confidence:
                contradicting.reinforce(False)
                return contradicting

        lesson_id = f"lesson_{self._lesson_count}"
        self._lesson_count += 1

        lesson = Lesson(
            id=lesson_id,
            situation=situation,
            insight=insight,
            outcome=outcome,
            confidence=confidence,
            evidence=evidence or [],
        )
        self.lessons[lesson_id] = lesson
        return lesson

    def find_contradicting(self, situation: str, insight: str) -> Optional[Lesson]:
        """Find lessons that might contradict a new insight."""
        situation_lower = situation.lower()

        for lesson in self.lessons.values():
            # Check if same situation
            if self._similarity(situation_lower, lesson.situation.lower()) > 0.7:
                # Check if contradicting insight (simplified)
                if "not" in insight.lower() != "not" in lesson.insight.lower():
                    return lesson
        return None

    def _similarity(self, a: str, b: str) -> float:
        """Simple word overlap similarity."""
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        overlap = len(words_a & words_b)
        return overlap / max(len(words_a), len(words_b))

    def get_relevant_lessons(
        self,
        situation: str,
        min_confidence: float = 0.3,
        limit: int = 5,
    ) -> List[Lesson]:
        """Get lessons relevant to a situation."""
        situation_lower = situation.lower()
        relevant = []

        for lesson in self.lessons.values():
            # Apply decay
            decayed_confidence = lesson.decay()
            if decayed_confidence < min_confidence:
                continue

            # Check relevance
            sim = self._similarity(situation_lower, lesson.situation.lower())
            if sim > 0.3:
                relevant.append((lesson, sim * decayed_confidence))

        # Sort by combined score
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [lesson for lesson, _ in relevant[:limit]]

    def apply_lesson(self, lesson_id: str, success: bool) -> None:
        """Record lesson application and update confidence."""
        if lesson_id in self.lessons:
            lesson = self.lessons[lesson_id]
            lesson.apply()
            lesson.reinforce(success)

    def get_all_lessons(self, include_stale: bool = False) -> List[Lesson]:
        """Get all lessons, optionally filtering stale ones."""
        if include_stale:
            return list(self.lessons.values())
        return [l for l in self.lessons.values() if not l.is_stale()]

    def to_context_string(self, situation: str, max_lessons: int = 5) -> str:
        """Format relevant lessons as context for LLM."""
        lessons = self.get_relevant_lessons(situation, limit=max_lessons)
        if not lessons:
            return ""

        lines = ["Relevant lessons from past experience:"]
        for lesson in lessons:
            conf = f"{lesson.decay():.0%}"
            lines.append(f"  - [{conf}] When: {lesson.situation}")
            lines.append(f"    Learned: {lesson.insight}")

        return "\n".join(lines)


class Memory:
    """
    Unified memory system combining episodic and lesson memory.

    Provides a single interface for all memory operations.
    """

    def __init__(self):
        self.episodic = EpisodicMemory()
        self.lessons = LessonMemory()
        self.working_memory: Dict[str, Any] = {}

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.OBSERVATION,
        importance: float = 0.5,
    ) -> MemoryEntry:
        """Add something to episodic memory."""
        return self.episodic.add_memory(content, memory_type, importance)

    def learn(
        self,
        situation: str,
        insight: str,
        outcome: str = "success",
        confidence: float = 0.5,
    ) -> Lesson:
        """Extract a lesson from experience."""
        return self.lessons.add_lesson(situation, insight, outcome, confidence)

    def recall(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search for relevant memories."""
        return self.episodic.search_memories(query, limit)

    def get_context(self, situation: str = "", max_memories: int = 15) -> str:
        """Get combined context from all memory systems."""
        parts = []

        # Recent episodic memories
        episodic_context = self.episodic.to_context_string(max_memories)
        if episodic_context:
            parts.append(episodic_context)

        # Relevant lessons
        if situation:
            lesson_context = self.lessons.to_context_string(situation)
            if lesson_context:
                parts.append(lesson_context)

        # Working memory
        if self.working_memory:
            wm_lines = ["Current working memory:"]
            for key, value in self.working_memory.items():
                wm_lines.append(f"  {key}: {value}")
            parts.append("\n".join(wm_lines))

        return "\n\n".join(parts) if parts else ""

    def set_working(self, key: str, value: Any) -> None:
        """Set a working memory value."""
        self.working_memory[key] = value

    def get_working(self, key: str, default: Any = None) -> Any:
        """Get a working memory value."""
        return self.working_memory.get(key, default)

    def clear_working(self) -> None:
        """Clear working memory."""
        self.working_memory.clear()
