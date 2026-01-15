"""
GOAP-style planning system for Club Harness.

Adapted from LisaSimpson's deliberative_agent/planning.py.

Key differentiator from "reactive" agents:
- PLAN before acting using goal-oriented action planning
- A* search through action space to find optimal plans
- Verification points for safety
- Confidence tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappush, heappop
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple


@dataclass(frozen=True)
class WorldState:
    """
    Immutable world state for planning.

    Uses frozenset of facts for hashability (needed for A* visited set).
    """
    facts: FrozenSet[str]

    @classmethod
    def from_facts(cls, facts: List[str]) -> WorldState:
        """Create world state from list of facts."""
        return cls(frozenset(facts))

    def has_fact(self, fact: str) -> bool:
        """Check if world state contains a fact."""
        return fact in self.facts

    def has_all(self, facts: List[str]) -> bool:
        """Check if world state contains all specified facts."""
        return all(f in self.facts for f in facts)

    def has_any(self, facts: List[str]) -> bool:
        """Check if world state contains any of the specified facts."""
        return any(f in self.facts for f in facts)

    def with_facts(self, add: List[str] = None, remove: List[str] = None) -> WorldState:
        """Create new state with added/removed facts."""
        new_facts = set(self.facts)
        if add:
            new_facts.update(add)
        if remove:
            new_facts.difference_update(remove)
        return WorldState(frozenset(new_facts))

    def __hash__(self) -> int:
        return hash(self.facts)

    def __repr__(self) -> str:
        facts_str = ", ".join(sorted(self.facts)[:5])
        if len(self.facts) > 5:
            facts_str += f"... (+{len(self.facts) - 5} more)"
        return f"WorldState({facts_str})"


@dataclass
class Action:
    """
    An action that can change the world state.

    Actions have:
    - Preconditions: facts that must be true to apply
    - Effects: facts added/removed after application
    - Cost: for optimization
    - Confidence modifier: how much this action affects plan confidence
    - Reversibility: whether this action can be undone
    """
    name: str
    preconditions: List[str] = field(default_factory=list)
    effects_add: List[str] = field(default_factory=list)
    effects_remove: List[str] = field(default_factory=list)
    cost: float = 1.0
    confidence_modifier: float = 1.0  # 1.0 = no effect, <1.0 = reduces confidence
    reversible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def applicable(self, state: WorldState) -> bool:
        """Check if action can be applied in given state."""
        return state.has_all(self.preconditions)

    def apply(self, state: WorldState) -> WorldState:
        """Apply action to state, returning new state."""
        return state.with_facts(add=self.effects_add, remove=self.effects_remove)

    def __repr__(self) -> str:
        return f"Action({self.name}, cost={self.cost})"


@dataclass
class Goal:
    """
    A goal to achieve.

    Goals have:
    - Required facts: facts that must be true for goal satisfaction
    - Optional facts: facts that are nice to have
    - Dependencies: other goals that must be satisfied first
    """
    id: str
    description: str
    required_facts: List[str] = field(default_factory=list)
    optional_facts: List[str] = field(default_factory=list)
    dependencies: List[Goal] = field(default_factory=list)
    priority: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_satisfied(self, state: WorldState) -> bool:
        """Check if goal is satisfied by state."""
        return state.has_all(self.required_facts)

    def satisfaction_score(self, state: WorldState) -> float:
        """
        Calculate how satisfied the goal is (0.0 to 1.0).

        Includes optional facts in scoring.
        """
        if not self.required_facts and not self.optional_facts:
            return 1.0

        required_count = sum(1 for f in self.required_facts if state.has_fact(f))
        optional_count = sum(1 for f in self.optional_facts if state.has_fact(f))

        total_required = len(self.required_facts)
        total_optional = len(self.optional_facts)

        if total_required == 0:
            return optional_count / total_optional if total_optional else 1.0

        required_score = required_count / total_required
        optional_score = optional_count / total_optional if total_optional else 1.0

        # Required facts are more important
        return 0.8 * required_score + 0.2 * optional_score

    def can_attempt(self, state: WorldState) -> bool:
        """Check if dependencies are satisfied."""
        return all(dep.is_satisfied(state) for dep in self.dependencies)

    def unsatisfied_dependencies(self, state: WorldState) -> List[Goal]:
        """Get list of unsatisfied dependencies."""
        return [dep for dep in self.dependencies if not dep.is_satisfied(state)]

    def __repr__(self) -> str:
        return f"Goal({self.id}: {self.description[:30]}...)"


@dataclass
class Plan:
    """
    A plan to achieve a goal.

    Plans are explicit sequences of actions with:
    - Verification points for safety (after irreversible actions)
    - Confidence estimates
    - Cost estimates
    """
    steps: List[Action]
    expected_final_state: WorldState
    verification_points: Dict[int, str]  # step_index -> verification description
    confidence: float
    estimated_cost: float

    @classmethod
    def empty(cls, state: WorldState) -> Plan:
        """Create an empty plan (goal already satisfied)."""
        return cls(
            steps=[],
            expected_final_state=state,
            verification_points={},
            confidence=1.0,
            estimated_cost=0.0,
        )

    def is_empty(self) -> bool:
        """Check if this is an empty plan."""
        return len(self.steps) == 0

    def total_cost(self) -> float:
        """Calculate total cost of all steps."""
        return sum(step.cost for step in self.steps)

    def has_irreversible_steps(self) -> bool:
        """Check if plan contains irreversible actions."""
        return any(not step.reversible for step in self.steps)

    def reversible_prefix(self) -> List[Action]:
        """Get the reversible prefix of the plan."""
        prefix = []
        for step in self.steps:
            if not step.reversible:
                break
            prefix.append(step)
        return prefix

    def to_string(self) -> str:
        """Format plan as readable string."""
        if not self.steps:
            return "Empty plan (goal already satisfied)"

        lines = [f"Plan with {len(self.steps)} steps (cost: {self.estimated_cost:.1f}, confidence: {self.confidence:.0%}):"]
        for i, step in enumerate(self.steps, 1):
            verify = " [VERIFY]" if i - 1 in self.verification_points else ""
            lines.append(f"  {i}. {step.name}{verify}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        steps_str = " -> ".join(s.name for s in self.steps) if self.steps else "(empty)"
        return f"Plan({steps_str}, cost={self.estimated_cost:.1f})"


@dataclass
class PlanningResult:
    """Result of a planning attempt."""
    success: bool
    plan: Optional[Plan]
    reason: str
    explored_states: int = 0
    planning_time_ms: float = 0.0

    @classmethod
    def success_result(cls, plan: Plan, explored: int = 0) -> PlanningResult:
        """Create a successful planning result."""
        return cls(
            success=True,
            plan=plan,
            reason="Plan found",
            explored_states=explored,
        )

    @classmethod
    def failure(cls, reason: str, explored: int = 0) -> PlanningResult:
        """Create a failed planning result."""
        return cls(
            success=False,
            plan=None,
            reason=reason,
            explored_states=explored,
        )


class Planner:
    """
    GOAP-style planner using A* search.

    Key insight: PLAN before acting, unlike reactive agents
    which just respond and hope for the best.

    Uses A* search through action space to find optimal plans.
    """

    def __init__(
        self,
        actions: List[Action],
        max_depth: int = 50,
        max_explored: int = 10000,
        heuristic: Optional[Callable[[WorldState, Goal], float]] = None,
    ):
        """
        Initialize the planner.

        Args:
            actions: Available actions
            max_depth: Maximum plan length
            max_explored: Maximum states to explore before giving up
            heuristic: Optional heuristic function for A* search
        """
        self.actions = actions
        self.max_depth = max_depth
        self.max_explored = max_explored
        self.heuristic = heuristic or self._default_heuristic

    def plan(self, current: WorldState, goal: Goal) -> PlanningResult:
        """
        Find a plan to achieve the goal from the current state.

        Uses A* search through action space.

        Args:
            current: Current world state
            goal: Goal to achieve

        Returns:
            PlanningResult with plan if found
        """
        import time
        start_time = time.time()

        # Already satisfied?
        if goal.is_satisfied(current):
            return PlanningResult.success_result(Plan.empty(current))

        # Check dependencies
        if not goal.can_attempt(current):
            unsatisfied = goal.unsatisfied_dependencies(current)
            return PlanningResult.failure(
                f"Dependencies not met: {[d.id for d in unsatisfied]}"
            )

        # A* search
        counter = 0
        explored = 0

        # Priority queue: (priority, counter, state, actions, cost)
        frontier: List[Tuple[float, int, WorldState, List[Action], float]] = []
        heappush(frontier, (0, 0, current, [], 0.0))

        visited: Set[int] = set()

        while frontier and explored < self.max_explored:
            _, _, state, actions, cost = heappop(frontier)
            explored += 1

            state_hash = hash(state)
            if state_hash in visited:
                continue
            visited.add(state_hash)

            if len(actions) >= self.max_depth:
                continue

            for action in self.actions:
                if not action.applicable(state):
                    continue

                new_state = action.apply(state)
                new_actions = actions + [action]
                new_cost = cost + action.cost

                if goal.is_satisfied(new_state):
                    elapsed = (time.time() - start_time) * 1000
                    plan = Plan(
                        steps=new_actions,
                        expected_final_state=new_state,
                        verification_points=self._create_verification_points(new_actions),
                        confidence=self._estimate_confidence(new_actions),
                        estimated_cost=new_cost,
                    )
                    result = PlanningResult.success_result(plan, explored)
                    result.planning_time_ms = elapsed
                    return result

                counter += 1
                priority = new_cost + self.heuristic(new_state, goal)
                heappush(frontier, (priority, counter, new_state, new_actions, new_cost))

        # No plan found
        elapsed = (time.time() - start_time) * 1000
        if explored >= self.max_explored:
            result = PlanningResult.failure(
                f"Exploration limit reached ({self.max_explored} states)",
                explored,
            )
        else:
            result = PlanningResult.failure("No plan exists", explored)

        result.planning_time_ms = elapsed
        return result

    def plan_with_dependencies(
        self,
        current: WorldState,
        goal: Goal,
    ) -> PlanningResult:
        """
        Plan including satisfying dependencies first.

        Args:
            current: Current world state
            goal: Goal to achieve

        Returns:
            PlanningResult with combined plan
        """
        state = current
        all_steps: List[Action] = []

        # First, plan for dependencies
        for dep in goal.dependencies:
            if not dep.is_satisfied(state):
                dep_result = self.plan(state, dep)
                if not dep_result.success:
                    return PlanningResult.failure(
                        f"Cannot satisfy dependency {dep.id}: {dep_result.reason}"
                    )
                all_steps.extend(dep_result.plan.steps)
                state = dep_result.plan.expected_final_state

        # Then, plan for the main goal
        main_result = self.plan(state, goal)
        if not main_result.success:
            return main_result

        all_steps.extend(main_result.plan.steps)

        # Combine into single plan
        combined = Plan(
            steps=all_steps,
            expected_final_state=main_result.plan.expected_final_state,
            verification_points=self._create_verification_points(all_steps),
            confidence=self._estimate_confidence(all_steps),
            estimated_cost=sum(s.cost for s in all_steps),
        )

        return PlanningResult.success_result(combined)

    def find_all_plans(
        self,
        current: WorldState,
        goal: Goal,
        max_plans: int = 5,
    ) -> List[Plan]:
        """
        Find multiple alternative plans.

        Useful for presenting options to users.

        Args:
            current: Current world state
            goal: Goal to achieve
            max_plans: Maximum number of plans to find

        Returns:
            List of alternative plans
        """
        plans: List[Plan] = []
        counter = 0
        explored = 0

        frontier: List[Tuple[float, int, WorldState, List[Action], float]] = []
        heappush(frontier, (0, 0, current, [], 0.0))

        # Don't use visited set - we want to find multiple paths
        while frontier and len(plans) < max_plans and explored < self.max_explored:
            _, _, state, actions, cost = heappop(frontier)
            explored += 1

            if len(actions) >= self.max_depth:
                continue

            for action in self.actions:
                if not action.applicable(state):
                    continue

                new_state = action.apply(state)
                new_actions = actions + [action]
                new_cost = cost + action.cost

                if goal.is_satisfied(new_state):
                    plan = Plan(
                        steps=new_actions,
                        expected_final_state=new_state,
                        verification_points=self._create_verification_points(new_actions),
                        confidence=self._estimate_confidence(new_actions),
                        estimated_cost=new_cost,
                    )
                    # Only add if different from existing plans
                    if not self._is_duplicate_plan(plan, plans):
                        plans.append(plan)

                counter += 1
                priority = new_cost + self.heuristic(new_state, goal)
                heappush(frontier, (priority, counter, new_state, new_actions, new_cost))

        return plans

    def _default_heuristic(self, state: WorldState, goal: Goal) -> float:
        """
        Default heuristic for A* search.

        Counts unsatisfied required facts.
        """
        unsatisfied = sum(1 for f in goal.required_facts if not state.has_fact(f))
        return float(unsatisfied)

    def _create_verification_points(
        self,
        actions: List[Action],
    ) -> Dict[int, str]:
        """
        Create verification points for a plan.

        Verification is inserted after irreversible actions.
        """
        points: Dict[int, str] = {}
        for i, action in enumerate(actions):
            if not action.reversible:
                points[i] = f"Verify after irreversible action: {action.name}"
        return points

    def _estimate_confidence(self, actions: List[Action]) -> float:
        """
        Estimate confidence in a plan.

        Confidence compounds - each action can reduce it.
        """
        if not actions:
            return 1.0

        conf = 0.95
        for action in actions:
            conf *= action.confidence_modifier

        return max(0.0, min(1.0, conf))

    def _is_duplicate_plan(self, plan: Plan, existing: List[Plan]) -> bool:
        """Check if plan is essentially the same as an existing one."""
        plan_actions = tuple(a.name for a in plan.steps)
        for existing_plan in existing:
            existing_actions = tuple(a.name for a in existing_plan.steps)
            if plan_actions == existing_actions:
                return True
        return False


class HierarchicalPlanner:
    """
    Planner that can decompose high-level goals into subgoals.

    Enables handling complex, multi-step objectives by
    breaking them down into manageable chunks.
    """

    def __init__(
        self,
        base_planner: Planner,
        decomposers: Optional[Dict[str, Callable[[Goal], List[Goal]]]] = None,
    ):
        """
        Initialize the hierarchical planner.

        Args:
            base_planner: Planner for atomic goals
            decomposers: Goal type -> decomposition function
        """
        self.base_planner = base_planner
        self.decomposers = decomposers or {}

    def plan(self, current: WorldState, goal: Goal) -> PlanningResult:
        """
        Plan for a potentially complex goal.

        If the goal can be decomposed, does so recursively.
        """
        # Check if we have a decomposer for this goal type
        goal_type = goal.metadata.get("type", "")
        if goal_type in self.decomposers:
            subgoals = self.decomposers[goal_type](goal)
            return self._plan_for_subgoals(current, subgoals)

        # No decomposition - use base planner
        return self.base_planner.plan(current, goal)

    def _plan_for_subgoals(
        self,
        current: WorldState,
        subgoals: List[Goal],
    ) -> PlanningResult:
        """Plan for a sequence of subgoals."""
        state = current
        all_steps: List[Action] = []

        for subgoal in subgoals:
            result = self.plan(state, subgoal)
            if not result.success:
                return PlanningResult.failure(
                    f"Cannot achieve subgoal {subgoal.id}: {result.reason}"
                )
            all_steps.extend(result.plan.steps)
            state = result.plan.expected_final_state

        combined = Plan(
            steps=all_steps,
            expected_final_state=state,
            verification_points=self.base_planner._create_verification_points(all_steps),
            confidence=self.base_planner._estimate_confidence(all_steps),
            estimated_cost=sum(s.cost for s in all_steps),
        )

        return PlanningResult.success_result(combined)

    def register_decomposer(
        self,
        goal_type: str,
        decomposer: Callable[[Goal], List[Goal]],
    ) -> None:
        """
        Register a decomposition function for a goal type.

        Args:
            goal_type: Type of goal (from metadata)
            decomposer: Function that breaks goal into subgoals
        """
        self.decomposers[goal_type] = decomposer
