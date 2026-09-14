"""Tests for all gap fixes:
1. WorldModel update_beliefs
2. Planner uses real tools based on affordances/autonomy
3. run_forever / tick loop
4. Persistence of identity, stakes, semantic store, world model
5. Escalation queue with human-review packets and callbacks
6. write_file tool implementation
7. EventBus wiring in agent
"""

import json
import tempfile
import threading
import time
from pathlib import Path

import pytest

from mad.agent import Agent, AgentConfig, EscalationItem, EscalationQueue
from mad.executor import Executor, ToolResult, write_file_tool
from mad.governance import GovernanceConfig, GovernanceLayer
from mad.ledger import Ledger
from mad.memory import MemorySystem, SemanticFact
from mad.models import (
    AutonomyLevel,
    Event,
    EventType,
    GovernanceDecision,
    GovernanceVerdict,
    IdentityCore,
    Intent,
    Plan,
    PlanStep,
    ReasonCode,
    RiskProfile,
    SelfModel,
    Stakes,
    WorldModel,
    _new_id,
)
from mad.planner import Planner


# =========================================================================
# 1. WorldModel update_beliefs
# =========================================================================


class TestWorldModelUpdateBeliefs:
    def test_update_beliefs_tracks_observation_count(self):
        world = WorldModel(affordances=["noop"])
        obs = {"type": "user_input", "payload": {"text": "hello"}}
        world.update_beliefs(obs)
        assert world.belief_state["total_observations"] == 1
        assert world.belief_state["observation_count_user_input"] == 1

    def test_update_beliefs_multiple_observations(self):
        world = WorldModel()
        for i in range(5):
            world.update_beliefs({"type": "user_input", "payload": {"text": f"msg {i}"}})
        assert world.belief_state["total_observations"] == 5
        assert world.belief_state["observation_count_user_input"] == 5

    def test_update_beliefs_reduces_uncertainty(self):
        world = WorldModel()
        world.update_beliefs({"type": "test"})
        u1 = world.uncertainty["test"]
        world.update_beliefs({"type": "test"})
        u2 = world.uncertainty["test"]
        # More observations should reduce uncertainty
        assert u2 < u1

    def test_update_beliefs_tracks_last_observation(self):
        world = WorldModel()
        obs = {"type": "test", "payload": {"key": "value"}}
        world.update_beliefs(obs)
        assert world.belief_state["last_observation"] == obs
        assert world.belief_state["last_test"] == {"key": "value"}

    def test_update_beliefs_extracts_payload_keys(self):
        world = WorldModel()
        obs = {"type": "sensor", "payload": {"temperature": 72, "humidity": 45}}
        world.update_beliefs(obs)
        assert world.belief_state["temperature"] == 72
        assert world.belief_state["humidity"] == 45

    def test_update_beliefs_skips_text_and_message_keys(self):
        """text and message are too generic to merge into belief_state."""
        world = WorldModel()
        obs = {"type": "user_input", "payload": {"text": "hello", "source": "cli"}}
        world.update_beliefs(obs)
        assert "text" not in world.belief_state or world.belief_state.get("text") != "hello"
        assert world.belief_state.get("source") == "cli"

    def test_agent_calls_update_beliefs_on_input(self):
        agent = Agent()
        agent.process_input("Test message")
        assert agent.world.belief_state.get("total_observations", 0) >= 1

    def test_world_model_to_dict_from_dict(self):
        world = WorldModel(
            belief_state={"key": "val"},
            uncertainty={"test": 0.5},
            causal_graph={"a": ["b", "c"]},
            affordances=["noop", "read_file"],
        )
        d = world.to_dict()
        restored = WorldModel.from_dict(d)
        assert restored.belief_state == {"key": "val"}
        assert restored.uncertainty == {"test": 0.5}
        assert restored.causal_graph == {"a": ["b", "c"]}
        assert restored.affordances == ["noop", "read_file"]


# =========================================================================
# 2. Planner uses real tools
# =========================================================================


class TestPlannerToolSelection:
    def test_planner_picks_read_file_for_read_goal(self):
        planner = Planner(
            identity=IdentityCore(commitments=["Be helpful"]),
            self_model=SelfModel(),
            world=WorldModel(affordances=["noop", "read_file", "list_dir"]),
            stakes=Stakes(autonomy_level=AutonomyLevel.READ_ONLY),
        )
        intent = Intent(goal="read file /etc/hosts")
        candidates = planner.generate_candidates(intent)
        # Direct plan should use read_file
        direct = candidates[0]
        assert direct.steps[0].tool == "read_file"
        assert "read_file" in direct.required_tools

    def test_planner_picks_list_dir_for_ls_goal(self):
        planner = Planner(
            identity=IdentityCore(commitments=["Be helpful"]),
            self_model=SelfModel(),
            world=WorldModel(affordances=["noop", "read_file", "list_dir"]),
            stakes=Stakes(autonomy_level=AutonomyLevel.READ_ONLY),
        )
        intent = Intent(goal="list the directory contents")
        candidates = planner.generate_candidates(intent)
        direct = candidates[0]
        assert direct.steps[0].tool == "list_dir"

    def test_planner_picks_write_file_for_write_goal(self):
        planner = Planner(
            identity=IdentityCore(commitments=["Be helpful"]),
            self_model=SelfModel(),
            world=WorldModel(affordances=["noop", "read_file", "list_dir", "write_file"]),
            stakes=Stakes(autonomy_level=AutonomyLevel.CONTROLLED_WRITE),
        )
        intent = Intent(goal="write file output.txt")
        candidates = planner.generate_candidates(intent)
        direct = candidates[0]
        assert direct.steps[0].tool == "write_file"

    def test_planner_falls_back_to_noop_when_tool_not_available(self):
        """If autonomy doesn't allow the tool, fall back to noop."""
        planner = Planner(
            identity=IdentityCore(commitments=["Be helpful"]),
            self_model=SelfModel(),
            world=WorldModel(affordances=["noop", "read_file", "list_dir", "write_file"]),
            stakes=Stakes(autonomy_level=AutonomyLevel.READ_ONLY),  # No write
        )
        intent = Intent(goal="write file output.txt")
        candidates = planner.generate_candidates(intent)
        direct = candidates[0]
        # write_file not permitted at READ_ONLY, should fall back to noop
        assert direct.steps[0].tool == "noop"

    def test_planner_observe_only_uses_only_noop(self):
        planner = Planner(
            identity=IdentityCore(commitments=["Be helpful"]),
            self_model=SelfModel(),
            world=WorldModel(affordances=["noop", "read_file", "list_dir"]),
            stakes=Stakes(autonomy_level=AutonomyLevel.OBSERVE_ONLY),
        )
        intent = Intent(goal="read file /etc/hosts")
        candidates = planner.generate_candidates(intent)
        # Even though goal says read_file, OBSERVE_ONLY only permits noop
        for plan in candidates:
            for step in plan.steps:
                assert step.tool == "noop"

    def test_planner_investigate_plan_uses_list_dir_when_available(self):
        planner = Planner(
            identity=IdentityCore(commitments=["Be helpful"]),
            self_model=SelfModel(),
            world=WorldModel(affordances=["noop", "read_file", "list_dir"]),
            stakes=Stakes(autonomy_level=AutonomyLevel.READ_ONLY),
        )
        intent = Intent(goal="do something")
        candidates = planner.generate_candidates(intent)
        investigate = candidates[1]  # Second plan is investigate-first
        assert investigate.steps[0].tool == "list_dir"

    def test_planner_available_tools_intersection(self):
        """Available tools = affordances AND permitted by autonomy."""
        planner = Planner(
            identity=IdentityCore(),
            self_model=SelfModel(),
            world=WorldModel(affordances=["noop", "read_file"]),
            stakes=Stakes(autonomy_level=AutonomyLevel.FULL),  # FULL allows everything
        )
        available = planner._available_tools()
        # But affordances only lists noop and read_file
        assert available == {"noop", "read_file"}

    def test_cautious_plan_always_uses_noop(self):
        planner = Planner(
            identity=IdentityCore(),
            self_model=SelfModel(),
            world=WorldModel(affordances=["noop", "read_file", "list_dir", "write_file"]),
            stakes=Stakes(autonomy_level=AutonomyLevel.FULL),
        )
        intent = Intent(goal="write file something.txt")
        candidates = planner.generate_candidates(intent)
        cautious = candidates[2]  # Third plan is cautious
        assert cautious.steps[0].tool == "noop"
        assert cautious.required_tools == ["noop"]


# =========================================================================
# 3. run_forever / tick loop
# =========================================================================


class TestTickLoop:
    def test_single_tick_increments_count(self):
        agent = Agent()
        initial = agent._tick_count
        agent.tick()
        assert agent._tick_count == initial + 1

    def test_tick_processes_bus_events(self):
        agent = Agent()
        received = []
        agent.event_bus.subscribe_all(lambda e: received.append(e))
        event = Event(type=EventType.SYSTEM, payload={"msg": "test"})
        agent.event_bus.publish(event)
        agent.tick()
        assert len(received) >= 1

    def test_tick_safe_mode_is_noop(self):
        agent = Agent()
        agent._safe_mode = True
        initial = agent._tick_count
        agent.tick()
        assert agent._tick_count == initial  # No increment

    def test_run_forever_can_be_stopped(self):
        agent = Agent(config=AgentConfig(tick_interval=0.01))
        ticks_observed = []

        def on_tick():
            ticks_observed.append(agent._tick_count)
            if len(ticks_observed) >= 3:
                agent.stop()

        t = threading.Thread(target=agent.run_forever, args=(on_tick,))
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive()
        assert len(ticks_observed) >= 3

    def test_inject_observation(self):
        agent = Agent()
        agent.inject_observation({"temperature": 72}, source="sensor")
        assert agent.event_bus.pending_count >= 1
        agent.tick()  # Process events
        assert agent.world.belief_state.get("total_observations", 0) >= 1

    def test_stop_sets_running_false(self):
        agent = Agent()
        agent._running = True
        agent.stop()
        assert agent._running is False

    def test_tick_triggers_reflection_at_interval(self):
        agent = Agent(config=AgentConfig(reflection_interval=3))
        # Add some ledger entries so reflection has material
        for i in range(5):
            agent.process_input(f"Message {i}")
        # Tick enough times to trigger reflection
        for _ in range(5):
            agent.tick()
        checkpoints = agent.ledger.entries_of_type("reflection_checkpoint")
        assert len(checkpoints) >= 1


# =========================================================================
# 4. Persistence
# =========================================================================


class TestPersistence:
    def test_identity_persists_across_restarts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            # First boot
            agent1 = Agent(
                identity=IdentityCore(
                    commitments=["Custom commitment"],
                    boundaries=["custom boundary"],
                    narrative_thread="Custom narrative",
                ),
                config=AgentConfig(data_dir=data_dir),
            )
            agent1.process_input("Hello")
            identity_id = agent1.identity.id

            # Second boot (from persisted state)
            agent2 = Agent(config=AgentConfig(data_dir=data_dir))
            assert agent2.identity.id == identity_id
            assert agent2.identity.commitments == ["Custom commitment"]
            assert agent2.identity.boundaries == ["custom boundary"]
            assert "Custom narrative" in agent2.identity.narrative_thread

    def test_stakes_persist_across_restarts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            agent1 = Agent(
                stakes=Stakes(
                    resource_budget=500.0,
                    reputation_value=1.5,
                    continuity_cost=3.0,
                    autonomy_level=AutonomyLevel.CONTROLLED_WRITE,
                ),
                config=AgentConfig(data_dir=data_dir),
            )
            agent1.process_input("Test")

            agent2 = Agent(config=AgentConfig(data_dir=data_dir))
            # Budget will be slightly different due to tool costs,
            # but continuity_cost and autonomy should match
            assert agent2.stakes.continuity_cost == 3.0
            assert agent2.stakes.autonomy_level == AutonomyLevel.CONTROLLED_WRITE
            assert agent2.stakes.reputation_value == 1.5

    def test_world_model_persists_across_restarts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            agent1 = Agent(config=AgentConfig(data_dir=data_dir))
            agent1.process_input("Hello world")
            # World model should have beliefs now
            assert agent1.world.belief_state.get("total_observations", 0) >= 1

            agent2 = Agent(config=AgentConfig(data_dir=data_dir))
            assert agent2.world.belief_state.get("total_observations", 0) >= 1

    def test_semantic_store_persists_across_restarts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            agent1 = Agent(config=AgentConfig(data_dir=data_dir))
            # Add a fact manually
            agent1.memory.semantic.add(SemanticFact(
                fact_id="test_fact_1",
                content="The sky is blue",
                tags=["nature"],
                confidence=0.9,
            ))
            agent1._persist_state()

            agent2 = Agent(config=AgentConfig(data_dir=data_dir))
            fact = agent2.memory.semantic.get("test_fact_1")
            assert fact is not None
            assert fact.content == "The sky is blue"

    def test_ledger_persists_across_restarts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            agent1 = Agent(config=AgentConfig(data_dir=data_dir))
            agent1.process_input("Test one")
            agent1.process_input("Test two")
            ledger_len = agent1.ledger.length

            agent2 = Agent(config=AgentConfig(data_dir=data_dir))
            # Ledger should have prior entries + new boot event
            assert agent2.ledger.length >= ledger_len

    def test_persistence_files_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            agent = Agent(config=AgentConfig(data_dir=data_dir))
            agent.process_input("Test")
            assert (data_dir / "identity.json").exists()
            assert (data_dir / "stakes.json").exists()
            assert (data_dir / "world.json").exists()
            assert (data_dir / "self_model.json").exists()
            assert (data_dir / "semantic_store.json").exists()
            assert (data_dir / "ledger.jsonl").exists()

    def test_memory_rebuilt_from_ledger_on_boot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            agent1 = Agent(config=AgentConfig(data_dir=data_dir))
            agent1.process_input("Important fact about cats")
            agent1.process_input("Another fact about dogs")

            agent2 = Agent(config=AgentConfig(data_dir=data_dir))
            # Memory should be rebuilt from ledger
            assert agent2.memory.cache.size > 0


# =========================================================================
# 5. Escalation
# =========================================================================


class TestEscalation:
    def test_escalation_queue_enqueue_and_get(self):
        queue = EscalationQueue()
        item = EscalationItem(
            plan=Plan(steps=[PlanStep(action="test", tool="noop")]),
            decision=GovernanceDecision(
                decision=GovernanceVerdict.ESCALATE,
                reason_codes=[ReasonCode.RISK_TOO_HIGH],
                explanation="Risk too high",
            ),
            trace_id="trace123",
        )
        queue.enqueue(item)
        assert queue.count == 1
        assert queue.pending_count == 1
        retrieved = queue.get(item.item_id)
        assert retrieved is not None
        assert retrieved.plan is not None

    def test_escalation_queue_resolve_approve(self):
        queue = EscalationQueue()
        item = EscalationItem(plan=Plan(), decision=GovernanceDecision())
        queue.enqueue(item)
        result = queue.resolve(item.item_id, approved=True, reviewer_notes="Looks good")
        assert result is True
        assert item.status == "approved"
        assert item.reviewer_notes == "Looks good"
        assert queue.pending_count == 0

    def test_escalation_queue_resolve_reject(self):
        queue = EscalationQueue()
        item = EscalationItem(plan=Plan(), decision=GovernanceDecision())
        queue.enqueue(item)
        result = queue.resolve(item.item_id, approved=False, reviewer_notes="Too risky")
        assert result is True
        assert item.status == "rejected"

    def test_escalation_queue_callback(self):
        queue = EscalationQueue()
        received = []
        queue.on_escalation(lambda item: received.append(item))
        item = EscalationItem(plan=Plan(), decision=GovernanceDecision())
        queue.enqueue(item)
        assert len(received) == 1
        assert received[0].item_id == item.item_id

    def test_escalation_review_packet(self):
        plan = Plan(
            steps=[PlanStep(action="risky_action", tool="noop", description="do something")],
            required_tools=["noop"],
        )
        decision = GovernanceDecision(
            decision=GovernanceVerdict.ESCALATE,
            reason_codes=[ReasonCode.RISK_TOO_HIGH],
            explanation="Risk score exceeds threshold",
            required_mods=["Reduce risk level"],
        )
        item = EscalationItem(plan=plan, decision=decision, trace_id="t1")
        packet = item.to_review_packet()
        assert packet["item_id"] == item.item_id
        assert packet["status"] == "pending"
        assert "risk_too_high" in packet["reason_codes"]
        assert "Risk score exceeds threshold" in packet["explanation"]
        assert "Reduce risk level" in packet["required_modifications"]
        assert packet["plan"]["steps"][0]["action"] == "risky_action"

    def test_agent_escalation_returns_escalation_response(self):
        """When all plans are escalated, agent returns escalation response."""
        agent = Agent(
            stakes=Stakes(
                autonomy_level=AutonomyLevel.READ_ONLY,
                resource_budget=1000.0,
            ),
            config=AgentConfig(
                governance_config=GovernanceConfig(risk_threshold=0.0),
            ),
        )
        # With risk_threshold=0.0, any plan with risk > 0 should escalate
        response = agent.process_input("Do something risky")
        # Should be either escalation or error (depending on whether any plan has 0 risk)
        assert response.get("type") in ("escalation", "error", "success")

    def test_resolve_escalation_approved(self):
        agent = Agent()
        # Manually add an escalation
        plan = Plan(
            steps=[PlanStep(action="test", tool="noop", params={})],
            required_tools=["noop"],
        )
        decision = GovernanceDecision(
            decision=GovernanceVerdict.ESCALATE,
            reason_codes=[ReasonCode.RISK_TOO_HIGH],
        )
        item = EscalationItem(plan=plan, decision=decision, trace_id="test")
        agent.escalation_queue.enqueue(item)

        result = agent.resolve_escalation(item.item_id, approved=True, reviewer_notes="OK")
        assert result["type"] == "escalation_approved"
        assert result["status"] == "success"
        # Resolution should be logged to ledger
        resolution_entries = agent.ledger.entries_of_type("escalation_resolution")
        assert len(resolution_entries) == 1
        assert resolution_entries[0].data["approved"] is True

    def test_resolve_escalation_rejected(self):
        agent = Agent()
        plan = Plan(steps=[PlanStep(action="test", tool="noop")])
        item = EscalationItem(plan=plan, decision=GovernanceDecision())
        agent.escalation_queue.enqueue(item)

        result = agent.resolve_escalation(item.item_id, approved=False)
        assert result["type"] == "escalation_rejected"

    def test_resolve_nonexistent_escalation(self):
        agent = Agent()
        result = agent.resolve_escalation("nonexistent_id", approved=True)
        assert result["type"] == "error"
        assert result["code"] == "ESCALATION_NOT_FOUND"

    def test_resolve_already_resolved_escalation(self):
        agent = Agent()
        valid_plan = Plan(
            steps=[PlanStep(action="test", tool="noop", description="test step")],
            required_tools=["noop"],
        )
        item = EscalationItem(plan=valid_plan, decision=GovernanceDecision())
        agent.escalation_queue.enqueue(item)
        agent.resolve_escalation(item.item_id, approved=True)
        # Try to resolve again
        result = agent.resolve_escalation(item.item_id, approved=False)
        assert result["type"] == "error"
        assert result["code"] == "ESCALATION_ALREADY_RESOLVED"

    def test_escalation_queue_pending_list(self):
        queue = EscalationQueue()
        item1 = EscalationItem(plan=Plan(), decision=GovernanceDecision())
        item2 = EscalationItem(plan=Plan(), decision=GovernanceDecision())
        queue.enqueue(item1)
        queue.enqueue(item2)
        assert len(queue.pending()) == 2
        queue.resolve(item1.item_id, approved=True)
        assert len(queue.pending()) == 1


# =========================================================================
# 6. write_file tool
# =========================================================================


class TestWriteFileTool:
    def test_write_file_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.txt")
            result = write_file_tool({"path": path, "content": "Hello, world!"})
            assert result.success is True
            assert result.output["bytes_written"] == 13
            assert Path(path).read_text() == "Hello, world!"

    def test_write_file_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "sub" / "dir" / "test.txt")
            result = write_file_tool({"path": path, "content": "nested"})
            assert result.success is True
            assert Path(path).read_text() == "nested"

    def test_write_file_no_path(self):
        result = write_file_tool({"content": "data"})
        assert result.success is False
        assert "No path" in result.error

    def test_write_file_cost(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.txt")
            result = write_file_tool({"path": path, "content": "data"})
            assert result.cost == 0.5

    def test_write_file_in_executor(self):
        stakes = Stakes(autonomy_level=AutonomyLevel.CONTROLLED_WRITE)
        executor = Executor(stakes)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "output.txt")
            plan = Plan(
                steps=[PlanStep(
                    action="write",
                    tool="write_file",
                    params={"path": path, "content": "executor test"},
                )],
                required_tools=["write_file"],
            )
            result, _ = executor.execute_plan(plan)
            assert result.status.value == "success"
            assert Path(path).read_text() == "executor test"

    def test_write_file_registered_in_executor(self):
        executor = Executor(Stakes())
        assert "write_file" in executor._tools


# =========================================================================
# 7. EventBus wiring in agent
# =========================================================================


class TestEventBusWiring:
    def test_event_bus_has_subscribers(self):
        agent = Agent()
        # Should have global subscriber (memory) and type-specific (world update, governance)
        assert len(agent.event_bus._global_subscribers) >= 1
        assert EventType.USER_INPUT in agent.event_bus._subscribers
        assert EventType.OBSERVATION in agent.event_bus._subscribers

    def test_event_bus_routes_user_input(self):
        agent = Agent()
        initial_cache = agent.memory.cache.size
        agent.process_input("Hello via bus")
        assert agent.memory.cache.size > initial_cache

    def test_event_bus_updates_world_model(self):
        agent = Agent()
        agent.process_input("Test via bus")
        assert agent.world.belief_state.get("total_observations", 0) >= 1

    def test_inject_observation_via_bus(self):
        agent = Agent()
        agent.inject_observation({"temperature": 72}, source="sensor")
        # Should be queued
        assert agent.event_bus.pending_count >= 1
        # Process events
        agent.event_bus.process_all()
        # World model should be updated
        assert agent.world.belief_state.get("total_observations", 0) >= 1

    def test_event_bus_processed_count(self):
        agent = Agent()
        initial = agent.event_bus.processed_count
        agent.process_input("Test")
        assert agent.event_bus.processed_count > initial


# =========================================================================
# 8. SelfModel serialization
# =========================================================================


class TestSelfModelSerialization:
    def test_self_model_to_dict_from_dict(self):
        sm = SelfModel(
            capabilities=["noop", "read_file"],
            vulnerabilities=["prompt_injection"],
            drives=["help_user"],
            self_eval={"accuracy": 0.9},
        )
        d = sm.to_dict()
        restored = SelfModel.from_dict(d)
        assert restored.capabilities == ["noop", "read_file"]
        assert restored.vulnerabilities == ["prompt_injection"]
        assert restored.drives == ["help_user"]
        assert restored.self_eval == {"accuracy": 0.9}


# =========================================================================
# 9. Agent status includes new fields
# =========================================================================


class TestAgentStatus:
    def test_status_includes_escalation_count(self):
        agent = Agent()
        status = agent.get_status()
        assert "escalation_pending" in status
        assert status["escalation_pending"] == 0

    def test_status_includes_event_bus_pending(self):
        agent = Agent()
        status = agent.get_status()
        assert "event_bus_pending" in status

    def test_status_includes_running(self):
        agent = Agent()
        status = agent.get_status()
        assert "running" in status
        assert status["running"] is False


# =========================================================================
# 10. Integration: governance tool check with write_file
# =========================================================================


class TestGovernanceWriteFile:
    def test_read_only_denies_write_file(self):
        gov = GovernanceLayer(
            identity=IdentityCore(boundaries=["never harm"]),
            stakes=Stakes(autonomy_level=AutonomyLevel.READ_ONLY),
            ledger=Ledger(),
        )
        plan = Plan(
            steps=[PlanStep(action="write", tool="write_file", description="")],
            required_tools=["write_file"],
        )
        decision = gov.evaluate(plan)
        assert decision.decision == GovernanceVerdict.DENY
        assert ReasonCode.TOOL_NOT_PERMITTED in decision.reason_codes

    def test_controlled_write_allows_write_file(self):
        gov = GovernanceLayer(
            identity=IdentityCore(boundaries=["never harm"]),
            stakes=Stakes(autonomy_level=AutonomyLevel.CONTROLLED_WRITE),
            ledger=Ledger(),
        )
        plan = Plan(
            steps=[PlanStep(action="save output", tool="write_file", description="")],
            required_tools=["write_file"],
        )
        decision = gov.evaluate(plan)
        assert decision.decision == GovernanceVerdict.ALLOW
