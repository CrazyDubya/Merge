"""Executor: sandboxed tool runner with evidence bundles.

Every tool call produces an EvidenceBundle (inputs, outputs, timestamps,
tool identity, cost, errors, traces) — even on failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mad.models import (
    AutonomyLevel,
    EvidenceBundle,
    Plan,
    PlanStep,
    Result,
    ResultStatus,
    Stakes,
    _new_id,
    _now,
    _hash_dict,
)


class ToolError(Exception):
    pass


class ToolNotRegistered(ToolError):
    pass


class ToolNotPermitted(ToolError):
    pass


@dataclass
class ToolResult:
    success: bool = True
    output: Any = None
    error: str = ""
    cost: float = 0.0


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

def noop_tool(params: dict[str, Any]) -> ToolResult:
    """No-op tool for testing. Always succeeds."""
    return ToolResult(success=True, output={"message": "noop executed"}, cost=0.0)


def read_file_tool(params: dict[str, Any]) -> ToolResult:
    """Read-only filesystem tool. Reads file contents."""
    path_str = params.get("path", "")
    if not path_str:
        return ToolResult(success=False, error="No path provided", cost=0.1)
    path = Path(path_str)
    if not path.exists():
        return ToolResult(success=False, error=f"File not found: {path}", cost=0.1)
    if not path.is_file():
        return ToolResult(success=False, error=f"Not a file: {path}", cost=0.1)
    try:
        content = path.read_text(errors="replace")
        return ToolResult(success=True, output={"content": content, "path": str(path)}, cost=0.2)
    except Exception as e:
        return ToolResult(success=False, error=str(e), cost=0.1)


def list_dir_tool(params: dict[str, Any]) -> ToolResult:
    """Read-only filesystem tool. Lists directory contents."""
    path_str = params.get("path", ".")
    path = Path(path_str)
    if not path.exists():
        return ToolResult(success=False, error=f"Directory not found: {path}", cost=0.1)
    if not path.is_dir():
        return ToolResult(success=False, error=f"Not a directory: {path}", cost=0.1)
    try:
        entries = sorted(str(e.name) for e in path.iterdir())
        return ToolResult(success=True, output={"entries": entries, "path": str(path)}, cost=0.1)
    except Exception as e:
        return ToolResult(success=False, error=str(e), cost=0.1)


def write_file_tool(params: dict[str, Any]) -> ToolResult:
    """Write content to a file. Requires CONTROLLED_WRITE or higher autonomy."""
    path_str = params.get("path", "")
    content = params.get("content", "")
    if not path_str:
        return ToolResult(success=False, error="No path provided", cost=0.1)
    path = Path(path_str)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return ToolResult(
            success=True,
            output={"path": str(path), "bytes_written": len(content)},
            cost=0.5,
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e), cost=0.1)


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

BUILTIN_TOOLS: dict[str, Callable[[dict[str, Any]], ToolResult]] = {
    "noop": noop_tool,
    "read_file": read_file_tool,
    "list_dir": list_dir_tool,
    "write_file": write_file_tool,
}


class Executor:
    """Sandboxed tool executor that produces evidence bundles for every call."""

    _MAX_EVIDENCE = 10000

    def __init__(self, stakes: Stakes) -> None:
        self.stakes = stakes
        self._tools: dict[str, Callable[[dict[str, Any]], ToolResult]] = dict(BUILTIN_TOOLS)
        self._evidence: list[EvidenceBundle] = []

    def register_tool(self, name: str, fn: Callable[[dict[str, Any]], ToolResult]) -> None:
        self._tools[name] = fn

    @property
    def evidence_log(self) -> list[EvidenceBundle]:
        return list(self._evidence)

    def drain_evidence(self) -> list[EvidenceBundle]:
        """Return and clear all accumulated evidence bundles."""
        drained = self._evidence
        self._evidence = []
        return drained

    def execute_plan(
        self, plan: Plan, abort_on_error: bool = True,
    ) -> tuple[Result, list[EvidenceBundle]]:
        """Execute all steps in a plan, collecting evidence bundles.

        Returns (Result, list of EvidenceBundle produced during this execution).
        """
        assert plan is not None, "plan must not be None"
        assert plan.steps, "plan must have at least one step"
        artifacts: list[dict[str, Any]] = []
        errors: list[str] = []
        evidence_refs: list[str] = []
        total_cost = 0.0
        side_effects: list[str] = []
        execution_bundles: list[EvidenceBundle] = []

        for step in plan.steps:
            bundle = self._execute_step(step)
            self._evidence.append(bundle)
            execution_bundles.append(bundle)
            evidence_refs.append(bundle.bundle_id)
            total_cost += bundle.cost

            if bundle.errors:
                errors.extend(bundle.errors)
                if abort_on_error:
                    return Result(
                        status=ResultStatus.ABORTED,
                        artifacts=artifacts,
                        costs=total_cost,
                        errors=errors,
                        evidence_refs=evidence_refs,
                        side_effects=side_effects,
                    ), execution_bundles
            else:
                artifacts.append(bundle.outputs)
                if step.tool not in ("noop",):
                    side_effects.append(f"tool:{step.tool}")

        # Deduct from budget
        self.stakes.deduct(total_cost)

        # Trim evidence log to prevent unbounded growth
        if len(self._evidence) > self._MAX_EVIDENCE:
            self._evidence = self._evidence[-self._MAX_EVIDENCE:]

        status = ResultStatus.SUCCESS if not errors else ResultStatus.PARTIAL
        return Result(
            status=status,
            artifacts=artifacts,
            costs=total_cost,
            errors=errors,
            evidence_refs=evidence_refs,
            side_effects=side_effects,
        ), execution_bundles

    def _execute_step(self, step: PlanStep) -> EvidenceBundle:
        """Execute a single step and produce an evidence bundle."""
        assert step.tool, "step must have a tool name"
        start_ts = _now()
        bundle = EvidenceBundle(
            tool_id=step.tool,
            inputs={"action": step.action, "params": step.params, "description": step.description},
            operator=self.stakes.to_dict().get("autonomy_level", 1),
            sandbox=True,
        )

        if step.tool not in self._tools:
            bundle.errors.append(f"Tool not registered: {step.tool}")
            bundle.timestamps = {"start": start_ts, "end": _now()}
            bundle.hashes = {"input": _hash_dict(bundle.inputs)}
            return bundle

        try:
            tool_fn = self._tools[step.tool]
            result = tool_fn(step.params)
            assert isinstance(result, ToolResult), (
                f"Tool {step.tool} must return ToolResult, got {type(result).__name__}"
            )
            bundle.outputs = result.output if result.output else {}
            bundle.cost = result.cost
            if not result.success:
                bundle.errors.append(result.error)
        except AssertionError:
            raise
        except Exception as e:
            bundle.errors.append(f"Tool execution error: {e}")
            bundle.cost = 0.1

        end_ts = _now()
        bundle.timestamps = {"start": start_ts, "end": end_ts}
        bundle.hashes = {
            "input": _hash_dict(bundle.inputs),
            "output": _hash_dict(bundle.outputs),
        }
        assert bundle.timestamps, "bundle must have timestamps after execution"
        assert bundle.hashes, "bundle must have hashes after execution"
        return bundle
