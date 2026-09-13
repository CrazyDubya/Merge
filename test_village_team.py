"""Tests for VillageTeam (Village collaborate() loop adapted to club_harness).

Ported concept from Village tests/unit/test_village.py; rewritten against
club_harness.orchestration.village.VillageTeam and club_harness Agents.
"""

import pytest
from unittest.mock import MagicMock

from club_harness.orchestration.village import VillageTeam
from club_harness.core.errors import ClubHarnessError


def make_agent(name, reply):
    agent = MagicMock()
    agent.name = name
    agent.chat.return_value = reply
    return agent


class TestVillageTeam:
    def test_add_and_get_member(self):
        team = VillageTeam("test")
        agent = make_agent("alice", "hi")
        team.add_member(agent)
        assert team.get_member("alice") is agent

    def test_add_duplicate_member_raises(self):
        team = VillageTeam("test")
        team.add_member(make_agent("alice", "hi"))
        with pytest.raises(ClubHarnessError):
            team.add_member(make_agent("alice", "hi again"))

    def test_remove_member(self):
        team = VillageTeam("test")
        agent = make_agent("alice", "hi")
        team.add_member(agent)
        assert team.remove_member("alice") is agent
        assert team.get_member("alice") is None

    def test_collaborate_combines_results(self):
        team = VillageTeam("test")
        team.add_member(make_agent("alice", "result A"))
        team.add_member(make_agent("bob", "result B"))
        combined = team.collaborate("do the thing")
        assert "alice: result A" in combined
        assert "bob: result B" in combined

    def test_collaborate_records_history(self):
        team = VillageTeam("test")
        team.add_member(make_agent("alice", "done"))
        team.collaborate("task 1")
        history = team.get_task_history()
        assert len(history) == 1
        assert history[0]["task"] == "task 1"
        assert "alice" in history[0]["members"]

    def test_collaborate_no_members_raises(self):
        team = VillageTeam("empty")
        with pytest.raises(ClubHarnessError):
            team.collaborate("task")

    def test_collaborate_survives_member_error(self):
        team = VillageTeam("test")
        bad = make_agent("bad", "")
        bad.chat.side_effect = RuntimeError("boom")
        team.add_member(bad)
        team.add_member(make_agent("good", "fine"))
        combined = team.collaborate("task")
        assert "bad: Error - boom" in combined
        assert "good: fine" in combined

    @pytest.mark.asyncio
    async def test_collaborate_async(self):
        team = VillageTeam("test")
        team.add_member(make_agent("alice", "async A"))
        team.add_member(make_agent("bob", "async B"))
        combined = await team.collaborate_async("task")
        assert "alice: async A" in combined
        assert "bob: async B" in combined
