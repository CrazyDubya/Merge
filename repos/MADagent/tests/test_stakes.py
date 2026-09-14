"""Tests for stakes system: continuity cost, budgets, reputation."""

import pytest

from mad.models import AutonomyLevel, Stakes


class TestContinuityCost:
    def test_initial_cost_is_zero(self):
        stakes = Stakes()
        assert stakes.continuity_cost == 0.0

    def test_increment_continuity_cost(self):
        stakes = Stakes()
        stakes.increment_continuity_cost(1.0)
        assert stakes.continuity_cost == 1.0
        stakes.increment_continuity_cost(0.5)
        assert stakes.continuity_cost == 1.5

    def test_default_increment(self):
        stakes = Stakes()
        stakes.increment_continuity_cost()
        assert stakes.continuity_cost == 1.0


class TestReputation:
    def test_initial_reputation(self):
        stakes = Stakes()
        assert stakes.reputation_value == 1.0

    def test_penalize_reputation(self):
        stakes = Stakes()
        stakes.penalize_reputation(0.3)
        assert stakes.reputation_value == pytest.approx(0.7)

    def test_reputation_floor(self):
        stakes = Stakes()
        stakes.penalize_reputation(5.0)
        assert stakes.reputation_value == 0.0

    def test_reward_reputation(self):
        stakes = Stakes()
        stakes.reward_reputation(0.5)
        assert stakes.reputation_value == 1.5

    def test_reputation_ceiling(self):
        stakes = Stakes()
        stakes.reward_reputation(5.0)
        assert stakes.reputation_value == 2.0


class TestResourceBudget:
    def test_can_afford(self):
        stakes = Stakes(resource_budget=100.0)
        assert stakes.can_afford(50.0) is True
        assert stakes.can_afford(100.0) is True
        assert stakes.can_afford(100.1) is False

    def test_deduct(self):
        stakes = Stakes(resource_budget=100.0)
        stakes.deduct(30.0)
        assert stakes.resource_budget == pytest.approx(70.0)
        assert stakes.daily_tool_calls == 1

    def test_budget_exceeded(self):
        stakes = Stakes(daily_tool_limit=3)
        assert stakes.budget_exceeded() is False
        stakes.deduct(0)
        stakes.deduct(0)
        stakes.deduct(0)
        assert stakes.budget_exceeded() is True


class TestAutonomyLevel:
    def test_autonomy_ordering(self):
        assert AutonomyLevel.OBSERVE_ONLY < AutonomyLevel.READ_ONLY
        assert AutonomyLevel.READ_ONLY < AutonomyLevel.CONTROLLED_WRITE
        assert AutonomyLevel.CONTROLLED_WRITE < AutonomyLevel.FULL


class TestStakesSerialization:
    def test_to_dict(self):
        stakes = Stakes(
            continuity_cost=1.5,
            reputation_value=0.8,
            resource_budget=500.0,
            autonomy_level=AutonomyLevel.READ_ONLY,
        )
        d = stakes.to_dict()
        assert d["continuity_cost"] == 1.5
        assert d["reputation_value"] == 0.8
        assert d["resource_budget"] == 500.0
        assert d["autonomy_level"] == 1

    def test_from_dict(self):
        d = {
            "continuity_cost": 2.0,
            "reputation_value": 0.5,
            "resource_budget": 250.0,
            "autonomy_level": 2,
        }
        stakes = Stakes.from_dict(d)
        assert stakes.continuity_cost == 2.0
        assert stakes.reputation_value == 0.5
        assert stakes.autonomy_level == AutonomyLevel.CONTROLLED_WRITE
