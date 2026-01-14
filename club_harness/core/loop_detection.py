"""
Loop detection utilities for Club Harness agents.

Adapted from TinyTroupe's AdvancedLoopDetector.
Detects various types of agent loops to prevent infinite execution.
"""

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import difflib

logger = logging.getLogger(__name__)


@dataclass
class LoopDetectionResult:
    """Result of loop detection check."""
    is_loop: bool
    loop_type: str = ""
    problematic_actions: List[Dict] = field(default_factory=list)
    message: str = ""

    def __bool__(self):
        return self.is_loop


class LoopDetector:
    """
    Advanced loop detection system for agents.

    Detects various types of loops:
    1. Identical action loops (exact repetition)
    2. Similar action loops (slight variations)
    3. Alternating patterns (A-B-A-B...)
    4. Complex patterns (A-B-C-A-B-C...)
    5. Response-based loops (similar outputs)

    From TinyTroupe's AdvancedLoopDetector.
    """

    def __init__(
        self,
        max_history: int = 50,
        similarity_threshold: float = 0.85,
        min_repetitions: int = 3,
    ):
        """
        Initialize loop detector.

        Args:
            max_history: Maximum number of actions to track
            similarity_threshold: Threshold for considering actions similar (0-1)
            min_repetitions: Minimum repetitions to consider a loop
        """
        self.max_history = max_history
        self.similarity_threshold = similarity_threshold
        self.min_repetitions = min_repetitions
        self.action_history: deque = deque(maxlen=max_history)
        self.response_history: deque = deque(maxlen=max_history)

    def add_action(self, action: Dict[str, Any], response: Optional[str] = None) -> None:
        """Add an action to the history."""
        self.action_history.append(action)
        if response:
            self.response_history.append(response)

    def add_response(self, response: str) -> None:
        """Add a response to check for repetitive outputs."""
        self.response_history.append(response)

    def check(self) -> LoopDetectionResult:
        """
        Check if the agent is in a loop.

        Returns:
            LoopDetectionResult with loop information
        """
        if len(self.action_history) < self.min_repetitions:
            return LoopDetectionResult(is_loop=False)

        actions = list(self.action_history)

        # 1. Check for identical action loops
        result = self._detect_identical_loops(actions)
        if result.is_loop:
            return result

        # 2. Check for similar action loops
        result = self._detect_similar_loops(actions)
        if result.is_loop:
            return result

        # 3. Check for alternating patterns
        result = self._detect_alternating_patterns(actions)
        if result.is_loop:
            return result

        # 4. Check for complex patterns
        result = self._detect_complex_patterns(actions)
        if result.is_loop:
            return result

        # 5. Check for repetitive responses
        if len(self.response_history) >= self.min_repetitions:
            result = self._detect_response_loops()
            if result.is_loop:
                return result

        return LoopDetectionResult(is_loop=False)

    def _detect_identical_loops(self, actions: List[Dict]) -> LoopDetectionResult:
        """Detect exact identical action repetitions."""
        if len(actions) < self.min_repetitions:
            return LoopDetectionResult(is_loop=False)

        # Check for consecutive identical actions
        for i in range(len(actions) - self.min_repetitions + 1):
            action_str = json.dumps(actions[i], sort_keys=True)
            consecutive_count = 1

            for j in range(i + 1, len(actions)):
                if json.dumps(actions[j], sort_keys=True) == action_str:
                    consecutive_count += 1
                else:
                    break

            if consecutive_count >= self.min_repetitions:
                return LoopDetectionResult(
                    is_loop=True,
                    loop_type="identical_repetition",
                    problematic_actions=actions[i:i + consecutive_count],
                    message=f"Detected {consecutive_count} identical consecutive actions",
                )

        return LoopDetectionResult(is_loop=False)

    def _detect_similar_loops(self, actions: List[Dict]) -> LoopDetectionResult:
        """Detect similar (but not identical) action repetitions."""
        if len(actions) < self.min_repetitions:
            return LoopDetectionResult(is_loop=False)

        # Look for patterns of similar actions
        for window_size in range(1, min(6, len(actions) // self.min_repetitions + 1)):
            for start in range(len(actions) - window_size * self.min_repetitions + 1):
                pattern = actions[start:start + window_size]
                repetitions = []

                current_pos = start
                while current_pos + window_size <= len(actions):
                    candidate = actions[current_pos:current_pos + window_size]
                    if self._patterns_similar(pattern, candidate):
                        repetitions.append(candidate)
                        current_pos += window_size
                    else:
                        break

                if len(repetitions) >= self.min_repetitions:
                    flat_actions = [a for rep in repetitions for a in rep]
                    return LoopDetectionResult(
                        is_loop=True,
                        loop_type="similar_repetition",
                        problematic_actions=flat_actions,
                        message=f"Detected {len(repetitions)} similar action patterns",
                    )

        return LoopDetectionResult(is_loop=False)

    def _detect_alternating_patterns(self, actions: List[Dict]) -> LoopDetectionResult:
        """Detect alternating patterns like A-B-A-B-A-B."""
        min_cycles = self.min_repetitions
        if len(actions) < min_cycles * 2:
            return LoopDetectionResult(is_loop=False)

        # Check for 2-action alternating patterns
        for i in range(len(actions) - min_cycles * 2 + 1):
            action_a = actions[i]
            action_b = actions[i + 1]

            # Check if pattern continues
            is_alternating = True
            for j in range(2, min_cycles * 2):
                expected = action_a if j % 2 == 0 else action_b
                actual = actions[i + j]

                if not self._actions_similar(actual, expected):
                    is_alternating = False
                    break

            if is_alternating:
                return LoopDetectionResult(
                    is_loop=True,
                    loop_type="alternating_pattern",
                    problematic_actions=actions[i:i + min_cycles * 2],
                    message=f"Detected alternating A-B pattern ({min_cycles} cycles)",
                )

        return LoopDetectionResult(is_loop=False)

    def _detect_complex_patterns(self, actions: List[Dict]) -> LoopDetectionResult:
        """Detect complex repeating patterns like A-B-C-A-B-C."""
        if len(actions) < 6:
            return LoopDetectionResult(is_loop=False)

        # Check for patterns of length 3-8
        for pattern_length in range(3, min(9, len(actions) // 2 + 1)):
            for start in range(len(actions) - pattern_length * 2 + 1):
                pattern = actions[start:start + pattern_length]
                next_pattern = actions[start + pattern_length:start + pattern_length * 2]

                if self._patterns_similar(pattern, next_pattern):
                    # Check if pattern repeats once more
                    if start + pattern_length * 3 <= len(actions):
                        third_pattern = actions[start + pattern_length * 2:start + pattern_length * 3]
                        if self._patterns_similar(pattern, third_pattern):
                            return LoopDetectionResult(
                                is_loop=True,
                                loop_type="complex_pattern",
                                problematic_actions=actions[start:start + pattern_length * 3],
                                message=f"Detected repeating pattern of length {pattern_length}",
                            )

        return LoopDetectionResult(is_loop=False)

    def _detect_response_loops(self) -> LoopDetectionResult:
        """Detect repetitive responses."""
        responses = list(self.response_history)
        if len(responses) < self.min_repetitions:
            return LoopDetectionResult(is_loop=False)

        # Check last N responses for similarity
        recent = responses[-self.min_repetitions:]
        base_response = recent[0]

        similar_count = 1
        for response in recent[1:]:
            similarity = difflib.SequenceMatcher(None, base_response, response).ratio()
            if similarity >= self.similarity_threshold:
                similar_count += 1

        if similar_count >= self.min_repetitions:
            return LoopDetectionResult(
                is_loop=True,
                loop_type="response_repetition",
                message=f"Detected {similar_count} similar consecutive responses",
            )

        return LoopDetectionResult(is_loop=False)

    def _patterns_similar(self, pattern1: List[Dict], pattern2: List[Dict]) -> bool:
        """Check if two patterns are similar."""
        if len(pattern1) != len(pattern2):
            return False

        for a1, a2 in zip(pattern1, pattern2):
            if not self._actions_similar(a1, a2):
                return False
        return True

    def _actions_similar(self, action1: Dict, action2: Dict) -> bool:
        """Check if two actions are similar based on similarity threshold."""
        str1 = json.dumps(action1, sort_keys=True)
        str2 = json.dumps(action2, sort_keys=True)
        similarity = difflib.SequenceMatcher(None, str1, str2).ratio()
        return similarity >= self.similarity_threshold

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the history."""
        return {
            "total_actions": len(self.action_history),
            "total_responses": len(self.response_history),
            "unique_action_types": len(
                set(a.get("type", "unknown") for a in self.action_history)
            ),
        }

    def reset(self) -> None:
        """Reset the loop detector."""
        self.action_history.clear()
        self.response_history.clear()


# Convenience function for quick loop checks
def detect_loop(
    actions: List[Dict],
    similarity_threshold: float = 0.85,
    min_repetitions: int = 3,
) -> LoopDetectionResult:
    """
    Quick function to detect loops in a list of actions.

    Args:
        actions: List of action dictionaries
        similarity_threshold: Threshold for considering actions similar
        min_repetitions: Minimum repetitions to consider a loop

    Returns:
        LoopDetectionResult
    """
    detector = LoopDetector(
        similarity_threshold=similarity_threshold,
        min_repetitions=min_repetitions,
    )
    for action in actions:
        detector.add_action(action)
    return detector.check()
