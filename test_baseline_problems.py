#!/usr/bin/env python3
"""
Baseline Problem Set for Club Harness
Tests across 5 categories: Technical, Reasoning, Analytical, Creative, Factual

This provides a standardized evaluation of LLM capabilities.
"""

import os
import sys
import time
import json
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from club_harness.core.config import config
from club_harness.llm.router import LLMRouter
from club_harness.orchestration.council import Council


@dataclass
class TestProblem:
    """A test problem with expected characteristics."""
    id: str
    category: str
    difficulty: str  # easy, medium, hard
    question: str
    validator: callable  # Function to validate response
    expected_keywords: List[str] = field(default_factory=list)
    time_limit_seconds: float = 60.0


@dataclass
class TestResult:
    """Result of a test problem."""
    problem_id: str
    category: str
    difficulty: str
    passed: bool
    response: str
    time_taken: float
    validation_details: str
    error: str = None


# === BASELINE PROBLEM SET ===

BASELINE_PROBLEMS = [
    # --- FACTUAL (easy baselines) ---
    TestProblem(
        id="factual_1",
        category="factual",
        difficulty="easy",
        question="What is the capital of Japan?",
        validator=lambda r: "tokyo" in r.lower(),
        expected_keywords=["Tokyo"],
    ),
    TestProblem(
        id="factual_2",
        category="factual",
        difficulty="medium",
        question="Name three programming languages that were created before 1980.",
        validator=lambda r: sum(1 for lang in ["fortran", "cobol", "lisp", "basic", "c", "pascal", "algol"]
                               if lang in r.lower()) >= 2,
        expected_keywords=["FORTRAN", "COBOL", "LISP", "C"],
    ),

    # --- REASONING (math and logic) ---
    TestProblem(
        id="reasoning_1",
        category="reasoning",
        difficulty="easy",
        question="If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?",
        validator=lambda r: "60" in r,
        expected_keywords=["60", "mph"],
    ),
    TestProblem(
        id="reasoning_2",
        category="reasoning",
        difficulty="medium",
        question="A farmer has chickens and rabbits. There are 35 heads and 94 legs in total. How many chickens are there?",
        validator=lambda r: "23" in r,
        expected_keywords=["23", "chickens"],
    ),
    TestProblem(
        id="reasoning_3",
        category="reasoning",
        difficulty="hard",
        question="""Three people (A, B, C) need to cross a river. The boat holds 2 people.
A takes 1 minute, B takes 2 minutes, C takes 5 minutes.
When two cross together, they go at the slower person's speed.
Someone must bring the boat back. What's the minimum time for all to cross?""",
        validator=lambda r: any(x in r for x in ["9 minute", "9 min", "nine minute"]),
        expected_keywords=["9", "minutes"],
    ),

    # --- TECHNICAL (code and programming) ---
    TestProblem(
        id="technical_1",
        category="technical",
        difficulty="easy",
        question="Write a Python function that returns True if a string is a palindrome, False otherwise.",
        validator=lambda r: "def" in r and ("[::-1]" in r or "reversed" in r or "for" in r),
        expected_keywords=["def", "return", "True", "False"],
    ),
    TestProblem(
        id="technical_2",
        category="technical",
        difficulty="medium",
        question="What is the time complexity of binary search and why?",
        validator=lambda r: ("log" in r.lower() or "o(log n)" in r.lower()) and
                          ("halv" in r.lower() or "divid" in r.lower() or "half" in r.lower()),
        expected_keywords=["O(log n)", "logarithmic", "divide"],
    ),
    TestProblem(
        id="technical_3",
        category="technical",
        difficulty="hard",
        question="""Explain the difference between a mutex and a semaphore.
When would you use each one? Give a practical example.""",
        validator=lambda r: ("mutex" in r.lower() and "semaphore" in r.lower() and
                           ("binary" in r.lower() or "lock" in r.lower() or "count" in r.lower())),
        expected_keywords=["mutex", "semaphore", "lock", "thread"],
    ),

    # --- ANALYTICAL (comparison and evaluation) ---
    TestProblem(
        id="analytical_1",
        category="analytical",
        difficulty="easy",
        question="Compare arrays and linked lists. Give one advantage of each.",
        validator=lambda r: (("array" in r.lower() and "linked" in r.lower()) and
                           ("access" in r.lower() or "insert" in r.lower() or "memory" in r.lower())),
        expected_keywords=["array", "linked list", "access", "insertion"],
    ),
    TestProblem(
        id="analytical_2",
        category="analytical",
        difficulty="medium",
        question="""Analyze the trade-offs between SQL and NoSQL databases.
Consider: scalability, consistency, query flexibility, and use cases.""",
        validator=lambda r: (sum(1 for kw in ["sql", "nosql", "scal", "consist", "schema", "relational"]
                               if kw in r.lower()) >= 4),
        expected_keywords=["SQL", "NoSQL", "scalability", "consistency", "schema"],
    ),

    # --- CREATIVE (open-ended) ---
    TestProblem(
        id="creative_1",
        category="creative",
        difficulty="easy",
        question="Write a haiku about programming.",
        validator=lambda r: len(r.split()) >= 8 and len(r.split('\n')) >= 1,
        expected_keywords=[],  # Creative content varies
    ),
    TestProblem(
        id="creative_2",
        category="creative",
        difficulty="medium",
        question="""Propose three innovative features for a smart home system
that don't exist yet. Explain each briefly.""",
        validator=lambda r: len(r) > 200 and sum(1 for c in r if c.isdigit() or c == '.') >= 2,
        expected_keywords=["1", "2", "3", "feature"],
    ),
]


class BaselineEvaluator:
    """Evaluates LLM performance on baseline problems."""

    def __init__(self, model: str = None, use_council: bool = False):
        self.router = LLMRouter()
        self.model = model or "google/gemma-3n-e2b-it:free"
        self.use_council = use_council
        self.council = None
        if use_council:
            self.council = Council(
                models=["google/gemma-3n-e2b-it:free", "nvidia/nemotron-nano-9b-v2:free"],
                strategy="simple_ranking"
            )
        self.results: List[TestResult] = []

    def run_problem(self, problem: TestProblem) -> TestResult:
        """Run a single test problem."""
        print(f"\n  [{problem.id}] {problem.category.upper()} ({problem.difficulty})")
        print(f"  Q: {problem.question[:80]}...")

        start_time = time.time()
        error = None
        response = ""

        try:
            if self.use_council and self.council:
                # Use council for consensus
                response = self.council.quick_consensus(problem.question)
            else:
                # Single model
                result = self.router.chat(
                    messages=[{"role": "user", "content": problem.question}],
                    model=self.model,
                    max_tokens=500,
                )
                response = result.content if hasattr(result, 'content') else str(result)

            elapsed = time.time() - start_time

            # Validate response
            passed = problem.validator(response)
            validation_details = "Passed validation" if passed else "Failed validation check"

            # Check expected keywords
            if problem.expected_keywords:
                found = [kw for kw in problem.expected_keywords if kw.lower() in response.lower()]
                validation_details += f" | Keywords found: {found}"

        except Exception as e:
            elapsed = time.time() - start_time
            error = str(e)
            passed = False
            validation_details = f"Error: {error}"

        result = TestResult(
            problem_id=problem.id,
            category=problem.category,
            difficulty=problem.difficulty,
            passed=passed,
            response=response[:500] if response else "",
            time_taken=elapsed,
            validation_details=validation_details,
            error=error,
        )

        self.results.append(result)

        status = "PASS" if passed else "FAIL"
        print(f"  A: {response[:100]}..." if response else f"  Error: {error}")
        print(f"  Result: {status} ({elapsed:.2f}s)")

        return result

    def run_all(self, problems: List[TestProblem] = None) -> Dict[str, Any]:
        """Run all problems and return summary."""
        problems = problems or BASELINE_PROBLEMS

        print("\n" + "=" * 70)
        print("BASELINE PROBLEM SET EVALUATION")
        print("=" * 70)
        print(f"Model: {self.model}")
        print(f"Council Mode: {self.use_council}")
        print(f"Total Problems: {len(problems)}")

        for problem in problems:
            try:
                self.run_problem(problem)
                time.sleep(1)  # Rate limiting
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as e:
                print(f"\n  ERROR on {problem.id}: {e}")

        return self.get_summary()

    def run_category(self, category: str) -> Dict[str, Any]:
        """Run problems in a specific category."""
        filtered = [p for p in BASELINE_PROBLEMS if p.category == category]
        return self.run_all(filtered)

    def get_summary(self) -> Dict[str, Any]:
        """Generate evaluation summary."""
        if not self.results:
            return {"error": "No results"}

        # Overall stats
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        # By category
        by_category = {}
        for r in self.results:
            if r.category not in by_category:
                by_category[r.category] = {"total": 0, "passed": 0}
            by_category[r.category]["total"] += 1
            if r.passed:
                by_category[r.category]["passed"] += 1

        # By difficulty
        by_difficulty = {}
        for r in self.results:
            if r.difficulty not in by_difficulty:
                by_difficulty[r.difficulty] = {"total": 0, "passed": 0}
            by_difficulty[r.difficulty]["total"] += 1
            if r.passed:
                by_difficulty[r.difficulty]["passed"] += 1

        # Timing
        avg_time = sum(r.time_taken for r in self.results) / total if total else 0
        total_time = sum(r.time_taken for r in self.results)

        summary = {
            "overall": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": f"{100*passed/total:.1f}%" if total else "N/A",
            },
            "by_category": {
                cat: {
                    "total": stats["total"],
                    "passed": stats["passed"],
                    "rate": f"{100*stats['passed']/stats['total']:.1f}%" if stats["total"] else "N/A"
                }
                for cat, stats in by_category.items()
            },
            "by_difficulty": {
                diff: {
                    "total": stats["total"],
                    "passed": stats["passed"],
                    "rate": f"{100*stats['passed']/stats['total']:.1f}%" if stats["total"] else "N/A"
                }
                for diff, stats in by_difficulty.items()
            },
            "timing": {
                "average_seconds": round(avg_time, 2),
                "total_seconds": round(total_time, 2),
            },
            "model": self.model,
            "council_mode": self.use_council,
        }

        return summary

    def print_report(self):
        """Print detailed evaluation report."""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print("EVALUATION REPORT")
        print("=" * 70)

        # Overall
        overall = summary["overall"]
        print(f"\nOVERALL: {overall['passed']}/{overall['total']} ({overall['pass_rate']})")

        # By category
        print("\nBY CATEGORY:")
        for cat, stats in summary["by_category"].items():
            print(f"  {cat.upper():12s}: {stats['passed']}/{stats['total']} ({stats['rate']})")

        # By difficulty
        print("\nBY DIFFICULTY:")
        for diff, stats in summary["by_difficulty"].items():
            print(f"  {diff.upper():12s}: {stats['passed']}/{stats['total']} ({stats['rate']})")

        # Timing
        timing = summary["timing"]
        print(f"\nTIMING:")
        print(f"  Average: {timing['average_seconds']:.2f}s per problem")
        print(f"  Total:   {timing['total_seconds']:.2f}s")

        # Failed problems
        failed = [r for r in self.results if not r.passed]
        if failed:
            print(f"\nFAILED PROBLEMS ({len(failed)}):")
            for r in failed:
                print(f"  - {r.problem_id}: {r.validation_details}")

        print("\n" + "=" * 70)


def run_quick_baseline():
    """Run a quick baseline test (subset of problems)."""
    quick_problems = [p for p in BASELINE_PROBLEMS if p.difficulty == "easy"]

    print("\n" + "=" * 70)
    print("QUICK BASELINE TEST (Easy problems only)")
    print("=" * 70)

    if not config.llm.api_key:
        print("Skipping - no API key")
        return None

    evaluator = BaselineEvaluator()
    evaluator.run_all(quick_problems)
    evaluator.print_report()

    return evaluator.get_summary()


def run_full_baseline():
    """Run full baseline evaluation."""
    print("\n" + "=" * 70)
    print("FULL BASELINE TEST")
    print("=" * 70)

    if not config.llm.api_key:
        print("Skipping - no API key")
        return None

    evaluator = BaselineEvaluator()
    evaluator.run_all()
    evaluator.print_report()

    return evaluator.get_summary()


def run_council_baseline():
    """Run baseline with council consensus."""
    quick_problems = [p for p in BASELINE_PROBLEMS if p.difficulty in ["easy", "medium"]][:6]

    print("\n" + "=" * 70)
    print("COUNCIL BASELINE TEST")
    print("=" * 70)

    if not config.llm.api_key:
        print("Skipping - no API key")
        return None

    evaluator = BaselineEvaluator(use_council=True)
    evaluator.run_all(quick_problems)
    evaluator.print_report()

    return evaluator.get_summary()


def compare_models():
    """Compare multiple models on the same problems."""
    quick_problems = [p for p in BASELINE_PROBLEMS if p.difficulty == "easy"]

    models = [
        "google/gemma-3n-e2b-it:free",
        "nvidia/nemotron-nano-9b-v2:free",
    ]

    print("\n" + "=" * 70)
    print("MODEL COMPARISON TEST")
    print("=" * 70)

    if not config.llm.api_key:
        print("Skipping - no API key")
        return None

    results = {}
    for model in models:
        print(f"\n--- Testing {model} ---")
        evaluator = BaselineEvaluator(model=model)
        evaluator.run_all(quick_problems)
        results[model] = evaluator.get_summary()

    # Print comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    for model, summary in results.items():
        overall = summary["overall"]
        print(f"{model}: {overall['passed']}/{overall['total']} ({overall['pass_rate']})")

    return results


def main():
    """Run baseline evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline problem set evaluation")
    parser.add_argument("--quick", action="store_true", help="Run quick test (easy only)")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    parser.add_argument("--council", action="store_true", help="Run with council consensus")
    parser.add_argument("--compare", action="store_true", help="Compare multiple models")
    parser.add_argument("--category", type=str, help="Test specific category")
    args = parser.parse_args()

    if args.full:
        run_full_baseline()
    elif args.council:
        run_council_baseline()
    elif args.compare:
        compare_models()
    elif args.category:
        if not config.llm.api_key:
            print("No API key")
            return
        evaluator = BaselineEvaluator()
        evaluator.run_category(args.category)
        evaluator.print_report()
    else:
        # Default: quick test
        run_quick_baseline()


if __name__ == "__main__":
    main()
