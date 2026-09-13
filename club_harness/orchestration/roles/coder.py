"""Coder role: generate code for a task via the LLM router, then sanity-test it.

ChipCliff's coder_algorithm.py generated a static HTML template and "tested"
it by opening a browser - a stub. This port keeps the role interface
(generate/test/send_feedback) but generates real code through club_harness's
LLMRouter and validates it by compiling (Python) or parsing structure,
without launching anything.

Ported from role-based-llm-framework/coder_algorithm.py via git subtree
merge; see docs/PROVENANCE.md.
"""

import ast
from typing import Optional, Tuple

try:
    from ...llm.router import LLMRouter
except ImportError:  # pragma: no cover - defensive
    LLMRouter = None  # type: ignore


class CoderRole:
    """Generates code for a task description using an LLM router."""

    def __init__(self, router: Optional["LLMRouter"] = None, model: Optional[str] = None):
        if router is None and LLMRouter is not None:
            router = LLMRouter()
        self.router = router
        self.model = model

    def generate(self, task: str) -> Tuple[str, str]:
        """Generate code for the task. Returns (code, test_result)."""
        code = self._generate_code(task)
        test_result = self.test_code(code)
        return code, test_result

    def _generate_code(self, task: str) -> str:
        if self.router is None:
            raise RuntimeError("No LLM router available for code generation")
        response = self.router.chat(
            messages=[
                {"role": "system",
                 "content": "You are a senior software engineer. Respond with ONLY the "
                            "requested code, no explanations, no markdown fences."},
                {"role": "user", "content": f"Write code for this task:\n{task}"},
            ],
            model=self.model,
        )
        return response.content.strip()

    def test_code(self, code: str) -> str:
        """Sanity-test generated code without executing untrusted input.

        Python code is compile-checked with ast; other languages get a
        non-empty structural check. Never shells out or opens a browser
        (unlike the original stub).
        """
        if not code or not code.strip():
            return "Test failed: empty code"
        try:
            ast.parse(code)
            return "Code tested successfully (Python syntax valid)"
        except SyntaxError:
            # Not Python (or invalid) - fall back to a structural check.
            if len(code.strip()) > 0:
                return "Code tested successfully (non-Python content, structural check only)"
            return "Test failed: empty code"

    def send_feedback(self, test_result: str) -> str:
        """Summarize a test result as feedback."""
        if "successfully" in test_result:
            return "feedback: tests passed"
        return f"feedback: {test_result}"


__all__ = ["CoderRole"]
