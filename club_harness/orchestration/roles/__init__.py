"""Role-based agent orchestration ported from ChipCliff.

The role-based-llm-framework (ChipCliff) contributed a PM/coder/researcher
dispatch pattern: a Project Manager classifies an incoming task, then routes
it to a Coder role or a Researcher role. This package ports that pattern onto
club_harness's own LLMRouter instead of ChipCliff's thin call_openai shim.

Ported from role-based-llm-framework via git subtree merge;
see docs/PROVENANCE.md.

Layout:
    classifier.py - TaskClassifier: torch/transformers when installed,
                    keyword-heuristic fallback otherwise.
    pm.py         - ProjectManager: classify -> assign -> track.
    coder.py      - CoderRole: generate code via the LLM router, sanity-test it.
    researcher.py - ResearcherRole: build queries, fetch sources, summarize.
"""

from .classifier import TaskClassifier, classify_task_heuristic
from .pm import ProjectManager
from .coder import CoderRole
from .researcher import ResearcherRole

__all__ = [
    "TaskClassifier",
    "classify_task_heuristic",
    "ProjectManager",
    "CoderRole",
    "ResearcherRole",
]
