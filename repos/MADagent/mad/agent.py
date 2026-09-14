"""Main agent runtime: boot sequence and life loop.

Ties together all subsystems into a coherent agent that:
1. Ingests events → ledger write
2. Updates world model beliefs
3. Retrieves memory (recent + semantic)
4. Generates intents + candidate plans
5. Runs governance veto pass/fail
6. Handles escalation with review packets + callback queue
7. Executes allowed plans with evidence bundles
8. Writes outcomes to ledger
9. Periodically reflects (bounded self-change)
10. Supports run_forever tick loop for autonomous operation
11. Routes events through the EventBus
12. Full persistence of identity, stakes, world, semantic store
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mad.adapters import InputAdapter, OutputAdapter
from mad.event_bus import EventBus
from mad.executor import Executor
from mad.governance import GovernanceConfig, GovernanceLayer
from mad.ledger import Ledger
from mad.memory import MemorySystem
from mad.models import (
    AutonomyLevel,
    Event,
    EventType,
    GovernanceDecision,
    GovernanceVerdict,
    IdentityCore,
    Intent,
    Plan,
    ReasonCode,
    Result,
    ResultStatus,
    SelfModel,
    Stakes,
    WorldModel,
    _new_id,
    _now,
)
from mad.persistence import load_state, save_state
from mad.planner import Planner
from mad.reflection import ReflectionEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Escalation system
# ---------------------------------------------------------------------------

@dataclass
class EscalationItem:
    """A plan that governance escalated for human review."""
    item_id: str = field(default_factory=_new_id)
    plan: Plan | None = None
    decision: GovernanceDecision | None = None
    trace_id: str = ""
    timestamp: str = field(default_factory=_now)
    status: str = "pending"  # pending | approved | rejected
    reviewer_notes: str = ""

    def to_review_packet(self) -> dict[str, Any]:
        """Build a structured human-review packet."""
        return {
            "item_id": self.item_id,
            "timestamp": self.timestamp,
            "status": self.status,
            "trace_id": self.trace_id,
            "plan": self.plan.to_dict() if self.plan else {},
            "governance_decision": self.decision.to_dict() if self.decision else {},
            "reason_codes": (
                [r.value for r in self.decision.reason_codes]
                if self.decision
                else []
            ),
            "explanation": self.decision.explanation if self.decision else "",
            "required_modifications": (
                self.decision.required_mods if self.decision else []
            ),
        }


class EscalationQueue:
    """Queue for escalated items awaiting human review."""

    def __init__(self) -> None:
        self._items: list[EscalationItem] = []
        self._callbacks: list[Callable[[EscalationItem], None]] = []

    def enqueue(self, item: EscalationItem) -> None:
        """Add an escalated item to the queue and notify callbacks."""
        self._items.append(item)
        for cb in self._callbacks:
            try:
                cb(item)
            except Exception as e:
                logger.error(f"Escalation callback error: {e}")

    def on_escalation(self, callback: Callable[[EscalationItem], None]) -> None:
        """Register a callback invoked whenever a new item is escalated."""
        self._callbacks.append(callback)

    def pending(self) -> list[EscalationItem]:
        """Return all pending escalation items."""
        return [i for i in self._items if i.status == "pending"]

    def get(self, item_id: str) -> EscalationItem | None:
        """Look up an escalation item by ID."""
        for item in self._items:
            if item.item_id == item_id:
                return item
        return None

    def resolve(self, item_id: str, approved: bool,
                reviewer_notes: str = "") -> bool:
        """Resolve a pending escalation item. Returns False if not found."""
        item = self.get(item_id)
        if item is None or item.status != "pending":
            return False
        item.status = "approved" if approved else "rejected"
        item.reviewer_notes = reviewer_notes
        return True

    @property
    def count(self) -> int:
        return len(self._items)

    @property
    def pending_count(self) -> int:
        return sum(1 for i in self._items if i.status == "pending")

    def all_items(self) -> list[EscalationItem]:
        return list(self._items)


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration for the agent runtime."""
    data_dir: Path | None = None
    reflection_interval: int = 10
    max_candidates: int = 5
    governance_config: GovernanceConfig = field(default_factory=GovernanceConfig)
    tick_interval: float = 1.0  # seconds between ticks in run_forever


class Agent:
    """The MADagent runtime: a persistent governed agent with
    irreversible memory and stakes.
    """

    def __init__(
        self,
        identity: IdentityCore | None = None,
        stakes: Stakes | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.identity = identity or IdentityCore(
            commitments=["Be helpful", "Be honest", "Respect boundaries"],
            preferences={},
            boundaries=["never delete user data", "never exfiltrate data"],
            narrative_thread="Agent initialized.",
        )
        self.stakes = stakes or Stakes(
            autonomy_level=AutonomyLevel.READ_ONLY,
            resource_budget=1000.0,
        )

        self._tick_count = 0
        self._safe_mode = False
        self._running = False
        self._last_chosen: Plan | None = None

        self._init_stores()
        self._init_subsystems()
        self._wire_event_bus()

        if self.config.data_dir:
            self._load_persisted_state()
        self._record_boot()

    def _init_stores(self) -> None:
        """Initialize ledger and memory — the truth store and derived stores."""
        ledger_path = None
        if self.config.data_dir:
            ledger_path = self.config.data_dir / "ledger.jsonl"
        self.ledger = Ledger(path=ledger_path)
        self.memory = MemorySystem(self.ledger)

    def _init_subsystems(self) -> None:
        """Initialize models and all processing subsystems."""
        self.self_model = SelfModel(
            capabilities=["text_response", "noop"],
            vulnerabilities=["prompt_injection", "context_overflow"],
            drives=["fulfill_user_requests", "maintain_integrity"],
        )
        self.world = WorldModel(
            affordances=["noop", "read_file", "list_dir", "write_file"],
        )
        self.governance = GovernanceLayer(
            identity=self.identity,
            stakes=self.stakes,
            ledger=self.ledger,
            config=self.config.governance_config,
        )
        self.planner = Planner(
            identity=self.identity,
            self_model=self.self_model,
            world=self.world,
            stakes=self.stakes,
        )
        self.executor = Executor(stakes=self.stakes)
        self.reflection_engine = ReflectionEngine(
            identity=self.identity,
            self_model=self.self_model,
            stakes=self.stakes,
            memory=self.memory,
            ledger=self.ledger,
            reflection_interval=self.config.reflection_interval,
        )
        self.event_bus = EventBus()
        self.input_adapter = InputAdapter()
        self.output_adapter = OutputAdapter()
        self.escalation_queue = EscalationQueue()

    def _record_boot(self) -> None:
        """Record the boot event in the ledger."""
        boot_event = Event(
            type=EventType.SYSTEM,
            payload={"message": "Agent booted", "identity_id": self.identity.id},
            provenance="system:boot",
        )
        self.ledger.append_event(boot_event)

    # ------------------------------------------------------------------
    # EventBus wiring
    # ------------------------------------------------------------------

    def _wire_event_bus(self) -> None:
        """Subscribe subsystems to the EventBus."""
        # Memory ingests all events
        self.event_bus.subscribe_all(self._on_event_memory)
        # World model updates on observations and user input
        self.event_bus.subscribe(EventType.USER_INPUT, self._on_event_world_update)
        self.event_bus.subscribe(EventType.OBSERVATION, self._on_event_world_update)
        # Governance decisions logged
        self.event_bus.subscribe(
            EventType.GOVERNANCE_DECISION, self._on_event_governance_log,
        )

    def _on_event_memory(self, event: Event) -> None:
        """EventBus handler: ingest event into memory + ledger."""
        self.memory.ingest_event(event)

    def _on_event_world_update(self, event: Event) -> None:
        """EventBus handler: update world model beliefs."""
        self.world.update_beliefs(event.to_dict())

    def _on_event_governance_log(self, event: Event) -> None:
        """EventBus handler: log governance decisions."""
        logger.debug(f"Governance event: {event.payload}")

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------

    def process_input(self, text: str) -> dict[str, Any]:
        """Process a single user text input through the full pipeline.

        Returns a response dict with the agent's output.
        This is the main entry point for the synchronous "life loop."
        """
        trace_id = _new_id()

        if self._safe_mode:
            return self.output_adapter.format_error(
                "Agent is in safe mode. Human intervention required.",
                code="SAFE_MODE",
            )

        # Stage 1: Ingest input, check for injection
        early_exit = self._ingest_and_check_input(text, trace_id)
        if early_exit is not None:
            return early_exit

        # Stage 2: Plan and run governance
        context = self.memory.get_context_window(query=text)
        intent = self._generate_intent(text, context, trace_id)
        governance_result = self._plan_and_govern(intent, trace_id)
        if governance_result is not None:
            return governance_result[0]
        # If governance_result is None, _plan_and_govern stored chosen plan
        # on self._last_chosen. This avoids returning a complex tuple.

        # Stage 3: Execute, record, reflect
        return self._execute_and_record(self._last_chosen, trace_id)

    def _ingest_and_check_input(
        self, text: str, trace_id: str,
    ) -> dict[str, Any] | None:
        """Ingest input via EventBus and check for injection.

        Returns an error response dict on failure, or None to continue.
        """
        event = self.input_adapter.user_text(text, trace_id=trace_id)
        if not self.event_bus.publish(event, priority=0.1):
            logger.warning("EventBus backpressure: input event dropped for trace %s", trace_id)
            return self.output_adapter.format_error(
                "System overloaded. Please retry.",
                code="BACKPRESSURE",
            )
        self.event_bus.process_all()

        if self.governance.check_text_for_injection(text):
            denial_event = Event(
                type=EventType.GOVERNANCE_DECISION,
                payload={"decision": "DENY", "reason": "prompt_injection_detected"},
                trace_id=trace_id,
            )
            self.ledger.append_event(denial_event)
            return self.output_adapter.format_error(
                "Input rejected: potential prompt injection detected.",
                code="INJECTION_DETECTED",
            )
        return None

    def _plan_and_govern(
        self, intent: Intent, trace_id: str,
    ) -> tuple[dict[str, Any]] | None:
        """Generate plans, score them, run governance, handle escalation.

        Returns a 1-tuple of response dict if no allowed plan exists,
        or None if a plan was approved (stored in self._last_chosen).
        """
        candidates = self.planner.generate_candidates(
            intent, max_candidates=self.config.max_candidates,
        )
        ranked = self.planner.score_and_rank(candidates)

        allowed, escalated = self._evaluate_candidates(ranked, trace_id)

        if not allowed:
            if escalated:
                plan, decision = escalated[0]
                return (self.output_adapter.format_escalation(
                    plan_summary=f"Plan {plan.plan_id}: "
                    + "; ".join(s.description for s in plan.steps),
                    reasons=[r.value for r in decision.reason_codes],
                    evidence_refs=[plan.plan_id],
                ),)
            return (self.output_adapter.format_error(
                "All candidate plans were denied by governance.",
                code="ALL_PLANS_DENIED",
            ),)

        self._last_chosen = allowed[0]
        self.ledger.append_plan(self._last_chosen)
        return None

    def _evaluate_candidates(
        self, ranked: list[Plan], trace_id: str,
    ) -> tuple[list[Plan], list[tuple[Plan, GovernanceDecision]]]:
        """Run governance on each candidate. Queue escalations. Return split lists."""
        allowed: list[Plan] = []
        escalated: list[tuple[Plan, GovernanceDecision]] = []
        for plan in ranked:
            decision = self.governance.evaluate(plan)
            self.ledger.append_governance(decision)

            if decision.decision == GovernanceVerdict.ALLOW:
                allowed.append(plan)
            elif decision.decision == GovernanceVerdict.ESCALATE:
                escalated.append((plan, decision))
                logger.info("Plan %s escalated: %s", plan.plan_id, decision.reason_codes)

        for plan, decision in escalated:
            self.escalation_queue.enqueue(EscalationItem(
                plan=plan, decision=decision, trace_id=trace_id,
            ))
        return allowed, escalated

    def _execute_and_record(
        self, chosen: Plan, trace_id: str,
    ) -> dict[str, Any]:
        """Execute the chosen plan, write to ledger, reflect, persist."""
        result, bundles = self.executor.execute_plan(chosen)

        self.ledger.append_result(result)
        for bundle in bundles:
            self.ledger.append_evidence(bundle)

        if not self._check_integrity():
            self._enter_safe_mode("Integrity check failed after execution")
            return self.output_adapter.format_error(
                "Integrity failure detected. Entering safe mode.",
                code="INTEGRITY_FAILURE",
            )

        self._tick_count += 1
        self.reflection_engine.tick()
        if self.reflection_engine.should_reflect():
            self.reflection_engine.reflect()

        self._persist_state()
        return self._build_response(chosen, result, trace_id)

    # ------------------------------------------------------------------
    # Tick loop (run_forever)
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Execute a single autonomous tick.

        Processes any queued EventBus events, runs time-based reflection,
        and handles autonomous observation.
        """
        if self._safe_mode:
            return

        # Process any pending events on the bus
        self.event_bus.process_all()

        # Increment tick
        self._tick_count += 1
        self.reflection_engine.tick()

        # Time-based reflection
        if self.reflection_engine.should_reflect():
            self.reflection_engine.reflect()

        # Persist state periodically
        self._persist_state()

    def run_forever(self, tick_callback: Callable[[], None] | None = None) -> None:
        """Continuous WHILE TRUE loop per spec.

        Calls tick() at tick_interval, processes bus events, and runs
        autonomous reflection. Stops when self._running is set to False.

        Args:
            tick_callback: Optional callback invoked each tick (for observation
                           injection, external event sources, etc.)
        """
        self._running = True
        logger.info("Agent entering run_forever loop")
        while self._running:
            try:
                self.tick()
                if tick_callback:
                    tick_callback()
            except Exception as e:
                logger.error(f"Tick error: {e}")
                self.stakes.penalize_reputation(0.05)
            time.sleep(self.config.tick_interval)
        logger.info("Agent exiting run_forever loop")

    async def run_forever_async(
        self,
        tick_callback: Callable[[], None] | None = None,
    ) -> None:
        """Async version of run_forever for event-loop integration."""
        self._running = True
        logger.info("Agent entering async run_forever loop")
        while self._running:
            try:
                self.tick()
                if tick_callback:
                    tick_callback()
            except Exception as e:
                logger.error(f"Async tick error: {e}")
                self.stakes.penalize_reputation(0.05)
            await asyncio.sleep(self.config.tick_interval)
        logger.info("Agent exiting async run_forever loop")

    def stop(self) -> None:
        """Signal the run_forever loop to stop."""
        self._running = False

    def inject_observation(self, data: dict[str, Any],
                           source: str = "environment") -> bool:
        """Inject an external observation into the event bus.

        Returns False if the bus applied backpressure and dropped the event.
        """
        event = self.input_adapter.observation(data, source=source)
        published = self.event_bus.publish(event, priority=0.3)
        if not published:
            logger.warning("EventBus backpressure: observation dropped from %s", source)
        return published

    # ------------------------------------------------------------------
    # Escalation resolution
    # ------------------------------------------------------------------

    def resolve_escalation(self, item_id: str, approved: bool,
                           reviewer_notes: str = "") -> dict[str, Any]:
        """Resolve an escalated item. If approved, execute the plan."""
        item = self.escalation_queue.get(item_id)
        if item is None:
            return self.output_adapter.format_error(
                f"Escalation item {item_id} not found.",
                code="ESCALATION_NOT_FOUND",
            )

        if item.status != "pending":
            return self.output_adapter.format_error(
                f"Escalation item {item_id} already resolved: {item.status}",
                code="ESCALATION_ALREADY_RESOLVED",
            )

        self.escalation_queue.resolve(item_id, approved, reviewer_notes)

        # Log the resolution to ledger
        self.ledger.append(
            "escalation_resolution",
            {
                "item_id": item_id,
                "approved": approved,
                "reviewer_notes": reviewer_notes,
                "timestamp": _now(),
            },
        )

        if not approved:
            return {"type": "escalation_rejected", "item_id": item_id}

        # Approved: execute the plan
        if item.plan is None:
            return self.output_adapter.format_error(
                "Escalation item has no plan to execute.",
                code="ESCALATION_NO_PLAN",
            )

        self.ledger.append_plan(item.plan)
        result, bundles = self.executor.execute_plan(item.plan)
        self.ledger.append_result(result)
        for bundle in bundles:
            self.ledger.append_evidence(bundle)

        self._persist_state()

        return {
            "type": "escalation_approved",
            "item_id": item_id,
            "result_id": result.result_id,
            "status": result.status.value,
            "artifacts": result.artifacts,
        }

    # ------------------------------------------------------------------
    # Reset handling (never silent)
    # ------------------------------------------------------------------

    def reset(self, reason: str = "manual_reset") -> None:
        """Reset agent state. ALWAYS logged as a continuity event."""
        self.stakes.increment_continuity_cost(1.0)
        self.stakes.penalize_reputation(0.2)
        self.ledger.append_continuity_event(reason, 1.0)

        self.memory.cache.clear()
        self._tick_count = 0
        self._safe_mode = False
        self.reflection_engine.reset_tick_counter()

    # ------------------------------------------------------------------
    # Persistence (delegated to mad.persistence module)
    # ------------------------------------------------------------------

    def _persist_state(self) -> None:
        """Persist all agent state to disk."""
        if not self.config.data_dir:
            return
        save_state(
            self.config.data_dir,
            self.identity, self.stakes, self.world,
            self.self_model, self.memory,
        )

    def _load_persisted_state(self) -> None:
        """Load all agent state from disk if files exist."""
        if not self.config.data_dir:
            return
        load_state(
            self.config.data_dir,
            self.identity, self.stakes, self.world,
            self.self_model, self.memory, self.ledger.length,
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _generate_intent(self, text: str, context: dict[str, Any],
                         trace_id: str) -> Intent:
        """Generate an intent from user input and context.

        For MVP, this is a simple mapping. In production,
        this would use an LLM to interpret the input.
        """
        return Intent(
            goal=text,
            why_now="User requested",
            constraints=[],
            priority=0.5,
            evidence_refs=[trace_id],
        )

    def _check_integrity(self) -> bool:
        """Run integrity checks."""
        try:
            self.ledger.verify_integrity()
            return not self.ledger.has_gaps()
        except Exception:
            return False

    def _enter_safe_mode(self, reason: str) -> None:
        """Enter safe mode. Logged to ledger."""
        self._safe_mode = True
        self.stakes.increment_continuity_cost(2.0)
        self.ledger.append_continuity_event(f"safe_mode:{reason}", 2.0)
        logger.warning(f"Agent entering safe mode: {reason}")

    def _build_response(self, plan: Plan, result: Result,
                        trace_id: str) -> dict[str, Any]:
        """Build a structured response from execution results."""
        if result.status == ResultStatus.SUCCESS:
            return {
                "type": "success",
                "trace_id": trace_id,
                "plan_id": plan.plan_id,
                "result_id": result.result_id,
                "artifacts": result.artifacts,
                "costs": result.costs,
            }
        elif result.status == ResultStatus.ABORTED:
            return {
                "type": "error",
                "trace_id": trace_id,
                "plan_id": plan.plan_id,
                "message": "Plan execution aborted.",
                "errors": result.errors,
                "costs": result.costs,
            }
        else:
            return {
                "type": "partial",
                "trace_id": trace_id,
                "plan_id": plan.plan_id,
                "result_id": result.result_id,
                "artifacts": result.artifacts,
                "errors": result.errors,
                "costs": result.costs,
            }

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return current agent status for observability."""
        return {
            "identity_id": self.identity.id,
            "tick_count": self._tick_count,
            "safe_mode": self._safe_mode,
            "ledger_length": self.ledger.length,
            "ledger_head": self.ledger.head_hash,
            "stakes": self.stakes.to_dict(),
            "narrative": self.identity.narrative_thread[:200],
            "commitments": self.identity.commitments,
            "boundaries": self.identity.boundaries,
            "escalation_pending": self.escalation_queue.pending_count,
            "event_bus_pending": self.event_bus.pending_count,
            "running": self._running,
        }
