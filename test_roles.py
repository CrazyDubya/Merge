"""Tests for the ported ChipCliff roles (orchestration/roles/).

Covers the task classifier (heuristic path - no torch needed), the
ProjectManager dispatch, and the coder/researcher role interfaces.
"""

import pytest
from unittest.mock import MagicMock

from club_harness.orchestration.roles import (
    TaskClassifier,
    classify_task_heuristic,
    ProjectManager,
    CoderRole,
    ResearcherRole,
)


class TestHeuristicClassifier:
    def test_coding_task(self):
        assert classify_task_heuristic("Write a Python function to sort a list") == "coding"

    def test_research_task(self):
        assert classify_task_heuristic("Research the history of the transistor") == "research"

    def test_unknown_task(self):
        assert classify_task_heuristic("blorpt zzz qqq") is None

    def test_classifier_falls_back_without_torch(self):
        clf = TaskClassifier(use_torch=False)
        assert clf.classify_task("Debug this API endpoint") == "coding"
        assert clf.classify_task("Summarize recent AI news") == "research"


class TestProjectManager:
    def _pm(self, category):
        pm = ProjectManager()
        pm.classifier = MagicMock()
        pm.classifier.classify_task.return_value = category
        pm.coder = MagicMock()
        pm.coder.generate.return_value = ("code", "ok")
        pm.researcher = MagicMock()
        pm.researcher.research.return_value = "findings"
        return pm

    def test_assign_coding_task(self):
        pm = self._pm("coding")
        task_id = pm.assign_task("write a function")
        assert task_id is not None
        pm.coder.generate.assert_called_once()
        assert pm.get_status(task_id)["status"] == "completed"

    def test_assign_research_task(self):
        pm = self._pm("research")
        task_id = pm.assign_task("what is fusion")
        assert task_id is not None
        pm.researcher.research.assert_called_once()

    def test_assign_unknown_category_fails(self):
        pm = self._pm("unknown")
        assert pm.assign_task("do something") is None

    def test_update_status(self):
        pm = self._pm("coding")
        task_id = pm.assign_task("write a function")
        pm.update_status(task_id, "In Progress")
        assert pm.get_status(task_id)["status"] == "In Progress"

    def test_update_status_unknown_task_raises(self):
        pm = self._pm("coding")
        with pytest.raises(KeyError):
            pm.update_status("nope", "done")


class TestCoderRole:
    def test_generate_uses_router(self):
        router = MagicMock()
        router.chat.return_value = MagicMock(content="print('hi')")
        coder = CoderRole(router=router)
        code, test_result = coder.generate("print hi")
        assert code == "print('hi')"
        assert "successfully" in test_result
        router.chat.assert_called_once()

    def test_test_code_valid_python(self):
        coder = CoderRole(router=MagicMock())
        assert "successfully" in coder.test_code("x = 1\nprint(x)")

    def test_test_code_empty(self):
        coder = CoderRole(router=MagicMock())
        assert "failed" in coder.test_code("   ").lower()

    def test_generate_no_router_raises(self):
        coder = CoderRole(router=None)
        coder.router = None
        with pytest.raises(RuntimeError):
            coder.generate("task")

    def test_send_feedback(self):
        coder = CoderRole(router=MagicMock())
        assert "passed" in coder.send_feedback("Code tested successfully")
        assert "Test failed" in coder.send_feedback("Test failed: boom")


class TestResearcherRole:
    def test_summarize_results(self):
        r = ResearcherRole(router=None)
        data = [{"title": "T", "description": "D"}]
        summary = r.summarize_results(data)
        assert "T" in summary and "D" in summary

    def test_summarize_empty(self):
        r = ResearcherRole(router=None)
        assert r.summarize_results([]) == "No results found."

    def test_enhance_queries_no_router(self):
        r = ResearcherRole(router=None)
        base = ["a best practices"]
        assert r.enhance_queries("task", base) == base

    def test_enhance_queries_uses_router(self):
        router = MagicMock()
        router.chat.return_value = MagicMock(content="query one\nquery two")
        r = ResearcherRole(router=router)
        queries = r.enhance_queries("task", ["base q"])
        assert "base q" in queries
        assert "query one" in queries

    def test_fetch_data_without_requests(self):
        # requests IS installed here; just verify it returns a list type
        r = ResearcherRole(router=None)
        result = r.fetch_data([])
        assert result == []
