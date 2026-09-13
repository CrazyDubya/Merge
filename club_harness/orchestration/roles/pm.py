"""Project Manager role: classify a task, assign it to a role, track status.

Ports ChipCliff's pm_algorithm.py assign_task/update_status flow onto
club_harness roles. Task state is kept in-memory (ChipCliff used XML files
via xml_utils; the XML persistence stays with the dashboard app in
apps/dashboard/).

Ported from role-based-llm-framework/pm_algorithm.py via git subtree merge;
see docs/PROVENANCE.md.
"""

import uuid
from typing import Dict, Optional

from .classifier import TaskClassifier
from .coder import CoderRole
from .researcher import ResearcherRole


class ProjectManager:
    """Routes tasks to the Coder or Researcher role and tracks their status."""

    def __init__(
        self,
        classifier: Optional[TaskClassifier] = None,
        coder: Optional[CoderRole] = None,
        researcher: Optional[ResearcherRole] = None,
    ) -> None:
        self.classifier = classifier or TaskClassifier()
        self.coder = coder or CoderRole()
        self.researcher = researcher or ResearcherRole()
        self.tasks: Dict[str, Dict[str, str]] = {}

    def classify_task(self, task: str) -> Optional[str]:
        """Classify a task as 'coding' or 'research'."""
        return self.classifier.classify_task(task)

    def assign_task(self, task: str, category: Optional[str] = None) -> Optional[str]:
        """Assign a task to the right role. Returns a task id, or None."""
        task_id = str(uuid.uuid4())
        category = category or self.classify_task(task)
        self.tasks[task_id] = {"task": task, "category": category or "unknown",
                               "status": "assigned"}
        try:
            if category == "coding":
                result = self.coder.generate(task)
            elif category == "research":
                result = self.researcher.research(task)
            else:
                raise ValueError(f"Unknown category: {category}")
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["result"] = result
            return task_id
        except Exception as e:
            self.tasks[task_id]["status"] = f"failed: {e}"
            return None

    def update_status(self, task_id: str, status: str) -> None:
        """Update a task's status."""
        if task_id not in self.tasks:
            raise KeyError(f"Unknown task id: {task_id}")
        self.tasks[task_id]["status"] = status

    def get_status(self, task_id: str) -> Optional[Dict[str, str]]:
        """Get a task's record."""
        return self.tasks.get(task_id)


__all__ = ["ProjectManager"]
