"""Tests for executor: evidence bundles, sandboxing, tool execution."""

import pytest

from mad.executor import Executor, ToolResult, noop_tool
from mad.models import (
    EvidenceBundle,
    Plan,
    PlanStep,
    ResultStatus,
    Stakes,
)


class TestEvidenceBundles:
    """Core invariant: every tool call emits EvidenceBundle, even on error."""

    def test_noop_produces_evidence(self):
        executor = Executor(Stakes())
        plan = Plan(
            steps=[PlanStep(action="test", tool="noop", params={}, description="test noop")],
            required_tools=["noop"],
        )
        result, _ = executor.execute_plan(plan)
        assert len(executor.evidence_log) == 1
        bundle = executor.evidence_log[0]
        assert bundle.tool_id == "noop"
        assert bundle.timestamps.get("start") is not None
        assert bundle.timestamps.get("end") is not None
        assert "input" in bundle.hashes

    def test_error_still_produces_evidence(self):
        executor = Executor(Stakes())
        plan = Plan(
            steps=[PlanStep(action="test", tool="nonexistent_tool", params={}, description="bad tool")],
            required_tools=["nonexistent_tool"],
        )
        result, _ = executor.execute_plan(plan)
        assert len(executor.evidence_log) == 1
        bundle = executor.evidence_log[0]
        assert len(bundle.errors) > 0
        assert bundle.tool_id == "nonexistent_tool"
        assert bundle.timestamps.get("start") is not None

    def test_multi_step_all_produce_evidence(self):
        executor = Executor(Stakes())
        plan = Plan(
            steps=[
                PlanStep(action="a", tool="noop", params={}, description=""),
                PlanStep(action="b", tool="noop", params={}, description=""),
                PlanStep(action="c", tool="noop", params={}, description=""),
            ],
            required_tools=["noop"],
        )
        result, _ = executor.execute_plan(plan)
        assert len(executor.evidence_log) == 3
        for bundle in executor.evidence_log:
            assert bundle.bundle_id != ""
            assert bundle.timestamps != {}

    def test_evidence_has_input_hash(self):
        executor = Executor(Stakes())
        plan = Plan(
            steps=[PlanStep(action="test", tool="noop", params={"key": "val"}, description="")],
            required_tools=["noop"],
        )
        executor.execute_plan(plan)
        bundle = executor.evidence_log[0]
        assert "input" in bundle.hashes
        assert len(bundle.hashes["input"]) == 64  # SHA256

    def test_evidence_has_output_hash(self):
        executor = Executor(Stakes())
        plan = Plan(
            steps=[PlanStep(action="test", tool="noop", params={}, description="")],
            required_tools=["noop"],
        )
        executor.execute_plan(plan)
        bundle = executor.evidence_log[0]
        assert "output" in bundle.hashes


class TestExecution:
    def test_successful_execution(self):
        executor = Executor(Stakes())
        plan = Plan(
            steps=[PlanStep(action="test", tool="noop", params={}, description="")],
            required_tools=["noop"],
        )
        result, _ = executor.execute_plan(plan)
        assert result.status == ResultStatus.SUCCESS
        assert len(result.artifacts) == 1
        assert len(result.errors) == 0

    def test_abort_on_error(self):
        executor = Executor(Stakes())
        plan = Plan(
            steps=[
                PlanStep(action="a", tool="nonexistent", params={}, description=""),
                PlanStep(action="b", tool="noop", params={}, description=""),
            ],
            required_tools=["nonexistent", "noop"],
        )
        result, _ = executor.execute_plan(plan, abort_on_error=True)
        assert result.status == ResultStatus.ABORTED
        assert len(result.errors) > 0
        # Second step should not have been executed
        assert len(executor.evidence_log) == 1

    def test_continue_on_error(self):
        executor = Executor(Stakes())
        plan = Plan(
            steps=[
                PlanStep(action="a", tool="nonexistent", params={}, description=""),
                PlanStep(action="b", tool="noop", params={}, description=""),
            ],
            required_tools=["nonexistent", "noop"],
        )
        result, _ = executor.execute_plan(plan, abort_on_error=False)
        assert result.status == ResultStatus.PARTIAL
        assert len(executor.evidence_log) == 2


class TestToolRegistration:
    def test_register_custom_tool(self):
        executor = Executor(Stakes())

        def custom_tool(params):
            return ToolResult(success=True, output={"custom": True}, cost=0.5)

        executor.register_tool("custom", custom_tool)
        plan = Plan(
            steps=[PlanStep(action="test", tool="custom", params={}, description="")],
            required_tools=["custom"],
        )
        result, _ = executor.execute_plan(plan)
        assert result.status == ResultStatus.SUCCESS
        assert result.artifacts[0]["custom"] is True


class TestCostTracking:
    def test_costs_tracked(self):
        stakes = Stakes(resource_budget=100.0)
        executor = Executor(stakes)
        plan = Plan(
            steps=[
                PlanStep(action="a", tool="noop", params={}, description=""),
                PlanStep(action="b", tool="noop", params={}, description=""),
            ],
            required_tools=["noop"],
        )
        result, _ = executor.execute_plan(plan)
        assert result.costs >= 0
        assert result.evidence_refs != []
