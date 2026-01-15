"""
Verification system for Club Harness.

Adapted from LisaSimpson's verification.py.

Provides multi-level verification against specifications.
Key differentiator from simple string matching - we verify semantic properties.
"""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class CheckResult:
    """Result of running a verification check."""
    passed: bool
    confidence: float  # 0.0 to 1.0
    message: str
    check_name: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"CheckResult({status}, {self.confidence:.0%}, {self.message!r})"

    def __bool__(self) -> bool:
        return self.passed


class Check(ABC):
    """
    Base class for verification checks.

    Checks are the atomic units of verification. They can be:
    - Predicate checks (arbitrary conditions)
    - Fact checks (verify facts in world state)
    - Format checks (verify output structure)
    - Confidence checks (verify confidence levels)
    """

    @property
    def name(self) -> str:
        """Human-readable name for the check."""
        return self.__class__.__name__

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> CheckResult:
        """
        Run the check against the given context.

        Args:
            context: Context dictionary with relevant data

        Returns:
            CheckResult with pass/fail status and confidence
        """
        pass

    async def run_async(self, context: Dict[str, Any]) -> CheckResult:
        """Async version of run (defaults to sync)."""
        return self.run(context)


class PredicateCheck(Check):
    """
    Checks an arbitrary predicate.

    Useful for custom verification logic.
    """

    def __init__(
        self,
        predicate: Callable[[Dict[str, Any]], bool],
        description: str,
        confidence_if_true: float = 0.9,
        confidence_if_false: float = 0.1,
    ):
        """
        Initialize the predicate check.

        Args:
            predicate: Function that returns True if check passes
            description: Human-readable description
            confidence_if_true: Confidence when predicate is True
            confidence_if_false: Confidence when predicate is False
        """
        self.predicate = predicate
        self.description = description
        self.confidence_if_true = confidence_if_true
        self.confidence_if_false = confidence_if_false

    @property
    def name(self) -> str:
        return f"PredicateCheck({self.description[:30]})"

    def run(self, context: Dict[str, Any]) -> CheckResult:
        """Run the predicate check."""
        try:
            result = self.predicate(context)
            return CheckResult(
                passed=result,
                confidence=self.confidence_if_true if result else self.confidence_if_false,
                message=self.description + (" satisfied" if result else " not satisfied"),
                check_name=self.name,
            )
        except Exception as e:
            return CheckResult(
                passed=False,
                confidence=0.0,
                message=f"Predicate check failed: {e}",
                check_name=self.name,
                details={"error": str(e)},
            )


class FactCheck(Check):
    """
    Checks that specific facts exist in the context.

    Verifies that expected facts are present.
    """

    def __init__(
        self,
        required_facts: List[str],
        context_key: str = "facts",
    ):
        """
        Initialize the fact check.

        Args:
            required_facts: Facts that must be present
            context_key: Key in context where facts are stored
        """
        self.required_facts = required_facts
        self.context_key = context_key

    @property
    def name(self) -> str:
        return f"FactCheck({len(self.required_facts)} facts)"

    def run(self, context: Dict[str, Any]) -> CheckResult:
        """Check that all required facts are present."""
        facts = context.get(self.context_key, set())

        if isinstance(facts, list):
            facts = set(facts)

        missing = [f for f in self.required_facts if f not in facts]

        if missing:
            return CheckResult(
                passed=False,
                confidence=0.1,
                message=f"Missing facts: {missing}",
                check_name=self.name,
                details={"missing": missing, "required": self.required_facts},
            )

        return CheckResult(
            passed=True,
            confidence=0.95,
            message="All required facts present",
            check_name=self.name,
            details={"checked": self.required_facts},
        )


class OutputFormatCheck(Check):
    """
    Checks that output matches expected format.

    Supports:
    - JSON validation
    - Regex matching
    - Key presence
    - Type checking
    """

    def __init__(
        self,
        output_key: str = "output",
        expected_type: Optional[type] = None,
        required_keys: Optional[List[str]] = None,
        regex_pattern: Optional[str] = None,
        is_valid_json: bool = False,
    ):
        """
        Initialize the format check.

        Args:
            output_key: Key in context containing output
            expected_type: Expected Python type
            required_keys: Required keys if output is dict
            regex_pattern: Regex pattern to match
            is_valid_json: Whether output should be valid JSON
        """
        self.output_key = output_key
        self.expected_type = expected_type
        self.required_keys = required_keys
        self.regex_pattern = regex_pattern
        self.is_valid_json = is_valid_json

    @property
    def name(self) -> str:
        return "OutputFormatCheck"

    def run(self, context: Dict[str, Any]) -> CheckResult:
        """Check output format."""
        output = context.get(self.output_key)

        if output is None:
            return CheckResult(
                passed=False,
                confidence=0.0,
                message=f"No output found at key '{self.output_key}'",
                check_name=self.name,
            )

        # Check JSON validity
        if self.is_valid_json:
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except json.JSONDecodeError as e:
                    return CheckResult(
                        passed=False,
                        confidence=0.1,
                        message=f"Invalid JSON: {e}",
                        check_name=self.name,
                        details={"error": str(e)},
                    )

        # Check type
        if self.expected_type and not isinstance(output, self.expected_type):
            return CheckResult(
                passed=False,
                confidence=0.1,
                message=f"Expected {self.expected_type.__name__}, got {type(output).__name__}",
                check_name=self.name,
            )

        # Check required keys
        if self.required_keys and isinstance(output, dict):
            missing = [k for k in self.required_keys if k not in output]
            if missing:
                return CheckResult(
                    passed=False,
                    confidence=0.2,
                    message=f"Missing keys: {missing}",
                    check_name=self.name,
                    details={"missing": missing},
                )

        # Check regex
        if self.regex_pattern:
            text = str(output)
            if not re.search(self.regex_pattern, text):
                return CheckResult(
                    passed=False,
                    confidence=0.2,
                    message=f"Output doesn't match pattern: {self.regex_pattern}",
                    check_name=self.name,
                )

        return CheckResult(
            passed=True,
            confidence=0.9,
            message="Output format valid",
            check_name=self.name,
        )


class ConfidenceCheck(Check):
    """
    Checks that confidence meets a threshold.
    """

    def __init__(
        self,
        confidence_key: str = "confidence",
        min_confidence: float = 0.7,
    ):
        """
        Initialize confidence check.

        Args:
            confidence_key: Key in context containing confidence
            min_confidence: Minimum required confidence
        """
        self.confidence_key = confidence_key
        self.min_confidence = min_confidence

    @property
    def name(self) -> str:
        return f"ConfidenceCheck(>={self.min_confidence:.0%})"

    def run(self, context: Dict[str, Any]) -> CheckResult:
        """Check confidence level."""
        confidence = context.get(self.confidence_key)

        if confidence is None:
            return CheckResult(
                passed=False,
                confidence=0.0,
                message=f"No confidence found at key '{self.confidence_key}'",
                check_name=self.name,
            )

        # Handle Confidence objects
        if hasattr(confidence, 'value'):
            confidence = confidence.value

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            return CheckResult(
                passed=False,
                confidence=0.0,
                message=f"Invalid confidence value: {confidence}",
                check_name=self.name,
            )

        passed = confidence >= self.min_confidence

        return CheckResult(
            passed=passed,
            confidence=confidence,
            message=f"Confidence {confidence:.0%} {'meets' if passed else 'below'} threshold {self.min_confidence:.0%}",
            check_name=self.name,
            details={"confidence": confidence, "threshold": self.min_confidence},
        )


class CompositeCheck(Check):
    """
    Combines multiple checks with AND/OR logic.
    """

    def __init__(
        self,
        checks: List[Check],
        require_all: bool = True,
        name: Optional[str] = None,
    ):
        """
        Initialize composite check.

        Args:
            checks: List of checks to combine
            require_all: True for AND, False for OR
            name: Optional custom name
        """
        self.checks = checks
        self.require_all = require_all
        self._name = name

    @property
    def name(self) -> str:
        if self._name:
            return self._name
        op = "AND" if self.require_all else "OR"
        return f"CompositeCheck({op}, {len(self.checks)} checks)"

    def run(self, context: Dict[str, Any]) -> CheckResult:
        """Run all checks and combine results."""
        results = [check.run(context) for check in self.checks]

        passed_count = sum(1 for r in results if r.passed)
        total = len(results)

        if self.require_all:
            passed = passed_count == total
            message = f"{passed_count}/{total} checks passed (all required)"
        else:
            passed = passed_count > 0
            message = f"{passed_count}/{total} checks passed (any required)"

        # Aggregate confidence
        if results:
            if self.require_all:
                confidence = min(r.confidence for r in results)
            else:
                confidence = max(r.confidence for r in results)
        else:
            confidence = 1.0 if passed else 0.0

        return CheckResult(
            passed=passed,
            confidence=confidence,
            message=message,
            check_name=self.name,
            details={
                "check_results": [
                    {"name": r.check_name, "passed": r.passed, "confidence": r.confidence}
                    for r in results
                ]
            },
        )


@dataclass
class VerificationResult:
    """Result of running a full verification plan."""
    satisfied: bool
    confidence: float
    check_results: List[CheckResult]

    @property
    def all_passed(self) -> bool:
        """Check if all individual checks passed."""
        return all(r.passed for r in self.check_results)

    @property
    def failures(self) -> List[CheckResult]:
        """Get failed checks."""
        return [r for r in self.check_results if not r.passed]

    def summary(self) -> str:
        """Generate a human-readable summary."""
        passed = sum(1 for r in self.check_results if r.passed)
        total = len(self.check_results)
        return f"{passed}/{total} checks passed, confidence: {self.confidence:.0%}"

    def __bool__(self) -> bool:
        return self.satisfied


@dataclass
class VerificationPlan:
    """
    A plan for verifying that a goal has been achieved.

    Combines multiple checks and requires a minimum confidence level.
    """
    checks: List[Check] = field(default_factory=list)
    required_confidence: float = 0.8
    require_all_pass: bool = True

    def add_check(self, check: Check) -> VerificationPlan:
        """Add a check to the plan (returns self for chaining)."""
        self.checks.append(check)
        return self

    def verify(self, context: Dict[str, Any]) -> VerificationResult:
        """
        Execute all checks and aggregate results.

        Args:
            context: Context dictionary with data to verify

        Returns:
            VerificationResult with aggregated outcomes
        """
        if not self.checks:
            return VerificationResult(
                satisfied=True,
                confidence=1.0,
                check_results=[],
            )

        # Run all checks
        results = [check.run(context) for check in self.checks]

        # Count passes and failures
        failures = [r for r in results if not r.passed]
        all_passed = len(failures) == 0

        # Compute aggregate confidence
        if results:
            min_confidence = min(r.confidence for r in results)
        else:
            min_confidence = 1.0

        # Determine if verification is satisfied
        if self.require_all_pass:
            satisfied = all_passed and min_confidence >= self.required_confidence
        else:
            passed_ratio = (len(results) - len(failures)) / len(results)
            satisfied = passed_ratio >= 0.8 and min_confidence >= self.required_confidence

        return VerificationResult(
            satisfied=satisfied,
            confidence=min_confidence,
            check_results=results,
        )

    async def verify_async(self, context: Dict[str, Any]) -> VerificationResult:
        """Async version of verify."""
        if not self.checks:
            return VerificationResult(
                satisfied=True,
                confidence=1.0,
                check_results=[],
            )

        # Run all checks concurrently
        results = await asyncio.gather(*[check.run_async(context) for check in self.checks])

        failures = [r for r in results if not r.passed]
        all_passed = len(failures) == 0

        if results:
            min_confidence = min(r.confidence for r in results)
        else:
            min_confidence = 1.0

        if self.require_all_pass:
            satisfied = all_passed and min_confidence >= self.required_confidence
        else:
            passed_ratio = (len(results) - len(failures)) / len(results)
            satisfied = passed_ratio >= 0.8 and min_confidence >= self.required_confidence

        return VerificationResult(
            satisfied=satisfied,
            confidence=min_confidence,
            check_results=list(results),
        )

    def __repr__(self) -> str:
        check_names = [c.name for c in self.checks]
        return f"VerificationPlan({check_names}, required={self.required_confidence})"
