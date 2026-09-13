"""Village-style multi-agent collaboration, adapted to club_harness Agents.

Ported from Village's village/core/village.py collaborate() loop via git
subtree merge; see docs/PROVENANCE.md. The original orchestrated Village's
own Villager objects (async process_task); this adaptation orchestrates
club_harness.core.Agent objects through their chat() interface, with an
async variant for event-loop callers.
"""

import asyncio
from typing import Any, Dict, List, Optional

from ..core.agent import Agent
from ..core.errors import ClubHarnessError


class VillageTeam:
    """A named team of Agents that collaborate on tasks.

    Mirrors Village.collaborate(): every member works the task, results are
    combined, and a task history is kept.
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self.members: Dict[str, Agent] = {}
        self._task_history: List[Dict[str, Any]] = []

    def add_member(self, agent: Agent) -> None:
        """Add an Agent to the team."""
        if agent.name in self.members:
            raise ClubHarnessError(f"Member '{agent.name}' already in team")
        self.members[agent.name] = agent

    def remove_member(self, name: str) -> Optional[Agent]:
        """Remove a member by name."""
        return self.members.pop(name, None)

    def get_member(self, name: str) -> Optional[Agent]:
        """Get a member by name."""
        return self.members.get(name)

    def collaborate(self, task: str) -> str:
        """Run every member on the task and combine results.

        Args:
            task: The task description.

        Returns:
            Combined per-member results as a string.

        Raises:
            ClubHarnessError: If the team has no members.
        """
        if not self.members:
            raise ClubHarnessError("No members available for collaboration")

        task_record: Dict[str, Any] = {
            "task": task,
            "members": list(self.members.keys()),
        }

        results = []
        for member in self.members.values():
            try:
                result = member.chat(task)
                results.append(f"{member.name}: {result}")
            except Exception as e:
                results.append(f"{member.name}: Error - {str(e)}")

        combined = "\n".join(results)
        task_record["result"] = combined
        self._task_history.append(task_record)
        return combined

    async def collaborate_async(self, task: str) -> str:
        """Async variant: runs members concurrently in threads."""
        if not self.members:
            raise ClubHarnessError("No members available for collaboration")

        async def run_member(member: Agent) -> str:
            try:
                result = await asyncio.to_thread(member.chat, task)
                return f"{member.name}: {result}"
            except Exception as e:
                return f"{member.name}: Error - {str(e)}"

        results = await asyncio.gather(
            *[run_member(m) for m in self.members.values()]
        )
        combined = "\n".join(results)
        self._task_history.append(
            {"task": task, "members": list(self.members.keys()), "result": combined}
        )
        return combined

    def get_task_history(self) -> List[Dict[str, Any]]:
        """Return the history of collaborated tasks."""
        return list(self._task_history)


__all__ = ["VillageTeam"]
