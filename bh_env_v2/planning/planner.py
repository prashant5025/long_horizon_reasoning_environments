"""
Hierarchical Planning System with Tree-of-Thought Beam Search
for BusinessHorizonENV v2.

Provides:
  - GoalNode / GoalTree  : hierarchical goal decomposition with progress propagation
  - ThoughtNode / TreeOfThought : beam-search action selection
  - HierarchicalPlanner  : owns both and exposes plan / update API
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..engine.types import Action, ActionType, EnvID, GoalStatus


# ────────────────────────────────────────────────────────────────────────────
# Goal Tree
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class GoalNode:
    id: str
    name: str
    level: str                          # "goal" | "subgoal" | "task"
    status: GoalStatus
    priority: float
    deadline: Optional[int]
    progress: float                     # 0.0 .. 1.0
    parent_id: Optional[str]
    children_ids: List[str]
    description: str = ""


class GoalTree:
    """Dict-backed goal hierarchy with upward progress propagation."""

    def __init__(self) -> None:
        self._nodes: Dict[str, GoalNode] = {}

    # -- mutation --------------------------------------------------------

    def add_node(self, node: GoalNode) -> None:
        self._nodes[node.id] = node
        if node.parent_id and node.parent_id in self._nodes:
            parent = self._nodes[node.parent_id]
            if node.id not in parent.children_ids:
                parent.children_ids.append(node.id)

    def get_node(self, node_id: str) -> Optional[GoalNode]:
        return self._nodes.get(node_id)

    def get_children(self, node_id: str) -> List[GoalNode]:
        node = self._nodes.get(node_id)
        if node is None:
            return []
        return [self._nodes[cid] for cid in node.children_ids if cid in self._nodes]

    def get_active_tasks(self) -> List[GoalNode]:
        return [
            n for n in self._nodes.values()
            if n.level == "task" and n.status == GoalStatus.ACTIVE
        ]

    # -- progress / completion ------------------------------------------

    def update_progress(self, node_id: str, progress: float) -> None:
        node = self._nodes.get(node_id)
        if node is None:
            return
        node.progress = max(0.0, min(1.0, progress))
        if node.progress >= 1.0:
            node.status = GoalStatus.COMPLETE
            self.propagate_completion(node_id)
        # Propagate partial progress upward
        self._propagate_progress_upward(node_id)

    def _propagate_progress_upward(self, node_id: str) -> None:
        node = self._nodes.get(node_id)
        if node is None or node.parent_id is None:
            return
        parent = self._nodes.get(node.parent_id)
        if parent is None or not parent.children_ids:
            return
        children = self.get_children(parent.id)
        if not children:
            return
        parent.progress = sum(c.progress for c in children) / len(children)
        self._propagate_progress_upward(parent.id)

    def propagate_completion(self, node_id: str) -> None:
        """When *node_id* completes, check if all siblings also complete
        so the parent can be marked COMPLETE.  Recurse upward."""
        node = self._nodes.get(node_id)
        if node is None or node.parent_id is None:
            return
        parent = self._nodes.get(node.parent_id)
        if parent is None:
            return
        siblings = self.get_children(parent.id)
        if all(s.status == GoalStatus.COMPLETE for s in siblings):
            parent.status = GoalStatus.COMPLETE
            parent.progress = 1.0
            self.propagate_completion(parent.id)

    # -- lifecycle helpers ----------------------------------------------

    def prune_overdue(self, current_step: int) -> List[str]:
        """Mark overdue tasks as ABANDONED.  Returns list of pruned ids."""
        pruned: List[str] = []
        for node in self._nodes.values():
            if (
                node.deadline is not None
                and current_step > node.deadline
                and node.status in (GoalStatus.PENDING, GoalStatus.ACTIVE)
            ):
                node.status = GoalStatus.ABANDONED
                pruned.append(node.id)
        return pruned

    def activate_ready_tasks(self) -> List[str]:
        """Activate PENDING tasks whose parent is ACTIVE (prerequisites met)."""
        activated: List[str] = []
        for node in self._nodes.values():
            if node.status != GoalStatus.PENDING:
                continue
            if node.parent_id is None:
                # Top-level goals auto-activate
                node.status = GoalStatus.ACTIVE
                activated.append(node.id)
                continue
            parent = self._nodes.get(node.parent_id)
            if parent is not None and parent.status == GoalStatus.ACTIVE:
                node.status = GoalStatus.ACTIVE
                activated.append(node.id)
        return activated

    # -- rendering ------------------------------------------------------

    def render(self) -> str:
        """Return an ASCII tree with progress bars."""
        roots = [n for n in self._nodes.values() if n.parent_id is None]
        lines: List[str] = []
        for root in roots:
            self._render_node(root, lines, indent=0)
        return "\n".join(lines)

    def _render_node(self, node: GoalNode, lines: List[str], indent: int) -> None:
        bar = self._progress_bar(node.progress)
        status_tag = node.status.value
        prefix = "  " * indent
        lines.append(
            f"{prefix}[{node.level.upper()}] {node.name}  "
            f"{bar}  ({status_tag})"
        )
        for child in self.get_children(node.id):
            self._render_node(child, lines, indent + 1)

    @staticmethod
    def _progress_bar(fraction: float, width: int = 10) -> str:
        filled = int(round(fraction * width))
        empty = width - filled
        pct = int(round(fraction * 100))
        return f"[{'█' * filled}{'░' * empty}] {pct}%"


# ────────────────────────────────────────────────────────────────────────────
# Tree-of-Thought Beam Search
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ThoughtNode:
    action: Action
    value: float
    depth: int
    children: List["ThoughtNode"]
    reasoning: str = ""


ValueFn = Callable[[Dict[str, Any], str], float]


class TreeOfThought:
    """Beam-search over action sequences with gamma-discounted rollouts."""

    _DEFAULT_SCORES: Dict[str, float] = {
        "ADVANCE_PHASE": 9.0,
        "ADVANCE_DEAL": 8.0,
        "RESOLVE_RISK": 7.0,
        "MIGRATE_COHORT": 7.0,
        "FULFILL_INSTRUCTION": 6.0,
        "ADVANCE_WORKSTREAM": 6.0,
        "RUN_POC": 5.0,
        "CONTACT_STAKEHOLDER": 5.0,
        "ALLOCATE_BUDGET": 4.0,
        "BOOST_MORALE": 4.0,
        "RESPOND_SHOCK": 8.0,
        "REVIEW_STATUS": 3.0,
        "NOOP": -1.0,
    }

    def __init__(
        self,
        beam_width: int = 4,
        beam_depth: int = 3,
        gamma: float = 0.9,
    ) -> None:
        self.beam_width = beam_width
        self.beam_depth = beam_depth
        self.gamma = gamma

    # -- public API -----------------------------------------------------

    def search(
        self,
        valid_actions: List[Action],
        value_fn: Optional[ValueFn],
        state_digest: Dict[str, Any],
    ) -> Tuple[Action, float, str]:
        """Run beam search and return (best_action, best_value, reasoning)."""
        if not valid_actions:
            noop = Action(action_type=ActionType.NOOP)
            return noop, -1.0, "no valid actions"

        vfn = value_fn if value_fn is not None else self._default_value_fn

        # Depth-0: score every valid action
        roots: List[ThoughtNode] = []
        for action in valid_actions:
            action_name = action.action_type.name
            score = vfn(state_digest, action_name)
            node = ThoughtNode(
                action=action,
                value=score,
                depth=0,
                children=[],
                reasoning=f"d0: {action_name} -> {score:.1f}",
            )
            roots.append(node)

        # Keep top-k beams
        roots.sort(key=lambda n: n.value, reverse=True)
        beams = roots[: self.beam_width]

        # Roll out deeper levels
        for d in range(1, self.beam_depth):
            discount = self.gamma ** d
            for beam in beams:
                self._expand(beam, d, discount, valid_actions, vfn, state_digest)

        # Compute cumulative values for each beam (root + discounted children)
        best_node: Optional[ThoughtNode] = None
        best_cumulative = float("-inf")

        for beam in beams:
            cumulative = self._cumulative_value(beam)
            if cumulative > best_cumulative:
                best_cumulative = cumulative
                best_node = beam

        if best_node is None:
            # Fallback: should not happen given we have beams
            noop = Action(action_type=ActionType.NOOP)
            return noop, -1.0, "beam search produced no results"

        reasoning = self._trace_reasoning(best_node)
        return best_node.action, best_cumulative, reasoning

    # -- internals ------------------------------------------------------

    def _expand(
        self,
        parent: ThoughtNode,
        depth: int,
        discount: float,
        valid_actions: List[Action],
        vfn: ValueFn,
        state_digest: Dict[str, Any],
    ) -> None:
        """Expand a parent node one level deeper."""
        # If parent already has children at this depth, expand those instead
        if parent.children:
            for child in parent.children:
                next_discount = discount * self.gamma
                self._expand(child, depth, next_discount, valid_actions, vfn, state_digest)
            return

        for action in valid_actions:
            action_name = action.action_type.name
            score = vfn(state_digest, action_name) * discount
            child = ThoughtNode(
                action=action,
                value=score,
                depth=depth,
                children=[],
                reasoning=f"d{depth}: {action_name} -> {score:.1f}",
            )
            parent.children.append(child)

        # Prune children to beam_width
        parent.children.sort(key=lambda n: n.value, reverse=True)
        parent.children = parent.children[: self.beam_width]

    def _cumulative_value(self, node: ThoughtNode) -> float:
        """Sum node value + best child path recursively."""
        if not node.children:
            return node.value
        best_child_val = max(self._cumulative_value(c) for c in node.children)
        return node.value + best_child_val

    def _trace_reasoning(self, node: ThoughtNode) -> str:
        """Build a reasoning string along the best path."""
        parts = [node.reasoning]
        current = node
        while current.children:
            current = max(current.children, key=lambda c: self._cumulative_value(c))
            parts.append(current.reasoning)
        return " | ".join(parts)

    @classmethod
    def _default_value_fn(cls, state_digest: Dict[str, Any], action_type: str) -> float:
        return cls._DEFAULT_SCORES.get(action_type, 0.0)


# ────────────────────────────────────────────────────────────────────────────
# Hierarchical Planner
# ────────────────────────────────────────────────────────────────────────────

class HierarchicalPlanner:
    """Owns a GoalTree and TreeOfThought; exposes plan() / update() API."""

    REPLAN_INTERVAL: int = 30

    def __init__(
        self,
        beam_width: int = 4,
        beam_depth: int = 3,
        gamma: float = 0.9,
    ) -> None:
        self.goal_tree = GoalTree()
        self.tot = TreeOfThought(
            beam_width=beam_width,
            beam_depth=beam_depth,
            gamma=gamma,
        )
        self._last_replan_step: int = -self.REPLAN_INTERVAL  # force first replan

    # -- goal tree initialisation per environment -----------------------

    def init_goal_tree(self, env_id: EnvID) -> None:
        builders = {
            EnvID.SALES: self._build_sales_tree,
            EnvID.PM: self._build_pm_tree,
            EnvID.HR_IT: self._build_hrit_tree,
        }
        builder = builders.get(env_id)
        if builder is not None:
            builder()
            # Activate roots and first layer
            self.goal_tree.activate_ready_tasks()

    # -- Sales (13 nodes) -----------------------------------------------

    def _build_sales_tree(self) -> None:
        tree = self.goal_tree

        # Root goal
        tree.add_node(GoalNode(
            id="s-goal", name="Close $2.4M deal", level="goal",
            status=GoalStatus.ACTIVE, priority=10.0, deadline=None,
            progress=0.0, parent_id=None, children_ids=[],
            description="Close the enterprise deal worth $2.4M",
        ))

        # Phase sub-goals
        phases = [
            ("s-sg-disco", "Discovery phase", 1, 50),
            ("s-sg-poc", "POC phase", 2, 120),
            ("s-sg-nego", "Negotiation phase", 3, 170),
            ("s-sg-close", "Closing phase", 4, 200),
        ]
        for sg_id, sg_name, priority, deadline in phases:
            tree.add_node(GoalNode(
                id=sg_id, name=sg_name, level="subgoal",
                status=GoalStatus.PENDING, priority=float(priority),
                deadline=deadline, progress=0.0, parent_id="s-goal",
                children_ids=[],
            ))

        # Tasks under Discovery
        tree.add_node(GoalNode(
            id="s-t-engage", name="Engage key stakeholders", level="task",
            status=GoalStatus.PENDING, priority=3.0, deadline=40,
            progress=0.0, parent_id="s-sg-disco", children_ids=[],
            description="Build relationships with champion and decision makers",
        ))
        tree.add_node(GoalNode(
            id="s-t-qualify", name="Qualify opportunity", level="task",
            status=GoalStatus.PENDING, priority=2.0, deadline=50,
            progress=0.0, parent_id="s-sg-disco", children_ids=[],
            description="Confirm budget authority need timeline",
        ))

        # Tasks under POC
        tree.add_node(GoalNode(
            id="s-t-runpoc", name="Run POC demonstration", level="task",
            status=GoalStatus.PENDING, priority=4.0, deadline=100,
            progress=0.0, parent_id="s-sg-poc", children_ids=[],
            description="Execute proof-of-concept to score above 80",
        ))
        tree.add_node(GoalNode(
            id="s-t-pocfollow", name="POC follow-up meetings", level="task",
            status=GoalStatus.PENDING, priority=3.0, deadline=120,
            progress=0.0, parent_id="s-sg-poc", children_ids=[],
            description="Address feedback from POC evaluation",
        ))

        # Tasks under Negotiation
        tree.add_node(GoalNode(
            id="s-t-proposal", name="Submit proposal", level="task",
            status=GoalStatus.PENDING, priority=5.0, deadline=150,
            progress=0.0, parent_id="s-sg-nego", children_ids=[],
            description="Formal proposal with pricing and terms",
        ))
        tree.add_node(GoalNode(
            id="s-t-objections", name="Handle objections", level="task",
            status=GoalStatus.PENDING, priority=4.0, deadline=170,
            progress=0.0, parent_id="s-sg-nego", children_ids=[],
            description="Resolve detractor concerns and risk flags",
        ))

        # Tasks under Closing
        tree.add_node(GoalNode(
            id="s-t-final", name="Final approval", level="task",
            status=GoalStatus.PENDING, priority=6.0, deadline=190,
            progress=0.0, parent_id="s-sg-close", children_ids=[],
            description="Get sign-off from economic buyer",
        ))
        tree.add_node(GoalNode(
            id="s-t-ink", name="Sign contract", level="task",
            status=GoalStatus.PENDING, priority=7.0, deadline=200,
            progress=0.0, parent_id="s-sg-close", children_ids=[],
            description="Execute the deal contract",
        ))

    # -- PM (10 nodes) --------------------------------------------------

    def _build_pm_tree(self) -> None:
        tree = self.goal_tree

        tree.add_node(GoalNode(
            id="p-goal", name="Deliver program", level="goal",
            status=GoalStatus.ACTIVE, priority=10.0, deadline=None,
            progress=0.0, parent_id=None, children_ids=[],
            description="Deliver the full program on time and on budget",
        ))

        workstreams = [
            ("p-sg-ws0", "Backend workstream", 3, 150),
            ("p-sg-ws1", "Frontend workstream", 3, 160),
            ("p-sg-ws2", "Integration workstream", 2, 180),
        ]
        for sg_id, sg_name, priority, deadline in workstreams:
            tree.add_node(GoalNode(
                id=sg_id, name=sg_name, level="subgoal",
                status=GoalStatus.PENDING, priority=float(priority),
                deadline=deadline, progress=0.0, parent_id="p-goal",
                children_ids=[],
            ))

        # Backend tasks
        tree.add_node(GoalNode(
            id="p-t-ws0a", name="Advance backend", level="task",
            status=GoalStatus.PENDING, priority=3.0, deadline=140,
            progress=0.0, parent_id="p-sg-ws0", children_ids=[],
            description="Complete backend workstream milestones",
        ))
        tree.add_node(GoalNode(
            id="p-t-ws0r", name="Resolve backend risks", level="task",
            status=GoalStatus.PENDING, priority=4.0, deadline=150,
            progress=0.0, parent_id="p-sg-ws0", children_ids=[],
            description="Mitigate risks flagged on backend workstream",
        ))

        # Frontend tasks
        tree.add_node(GoalNode(
            id="p-t-ws1a", name="Advance frontend", level="task",
            status=GoalStatus.PENDING, priority=3.0, deadline=150,
            progress=0.0, parent_id="p-sg-ws1", children_ids=[],
            description="Complete frontend workstream milestones",
        ))
        tree.add_node(GoalNode(
            id="p-t-ws1r", name="Resolve frontend risks", level="task",
            status=GoalStatus.PENDING, priority=4.0, deadline=160,
            progress=0.0, parent_id="p-sg-ws1", children_ids=[],
            description="Mitigate risks flagged on frontend workstream",
        ))

        # Integration tasks
        tree.add_node(GoalNode(
            id="p-t-ws2a", name="Advance integration", level="task",
            status=GoalStatus.PENDING, priority=2.0, deadline=170,
            progress=0.0, parent_id="p-sg-ws2", children_ids=[],
            description="Complete integration workstream milestones",
        ))
        tree.add_node(GoalNode(
            id="p-t-ws2r", name="Resolve integration risks", level="task",
            status=GoalStatus.PENDING, priority=3.0, deadline=180,
            progress=0.0, parent_id="p-sg-ws2", children_ids=[],
            description="Mitigate risks flagged on integration workstream",
        ))

    # -- HR_IT (10 nodes) -----------------------------------------------

    def _build_hrit_tree(self) -> None:
        tree = self.goal_tree

        tree.add_node(GoalNode(
            id="h-goal", name="Complete migration", level="goal",
            status=GoalStatus.ACTIVE, priority=10.0, deadline=None,
            progress=0.0, parent_id=None, children_ids=[],
            description="Migrate all 8000 users to new platform",
        ))

        phases = [
            ("h-sg-prep", "Preparation phase", 3, 60),
            ("h-sg-wave", "Wave migration phase", 5, 150),
            ("h-sg-valid", "Validation phase", 4, 200),
        ]
        for sg_id, sg_name, priority, deadline in phases:
            tree.add_node(GoalNode(
                id=sg_id, name=sg_name, level="subgoal",
                status=GoalStatus.PENDING, priority=float(priority),
                deadline=deadline, progress=0.0, parent_id="h-goal",
                children_ids=[],
            ))

        # Preparation tasks
        tree.add_node(GoalNode(
            id="h-t-fulfill", name="Fulfill setup instructions", level="task",
            status=GoalStatus.PENDING, priority=4.0, deadline=50,
            progress=0.0, parent_id="h-sg-prep", children_ids=[],
            description="Complete all prerequisite instructions before migration",
        ))
        tree.add_node(GoalNode(
            id="h-t-pilotmig", name="Pilot cohort migration", level="task",
            status=GoalStatus.PENDING, priority=3.0, deadline=60,
            progress=0.0, parent_id="h-sg-prep", children_ids=[],
            description="Migrate pilot cohort and verify SLA",
        ))

        # Wave migration tasks
        tree.add_node(GoalNode(
            id="h-t-waves", name="Migrate user waves", level="task",
            status=GoalStatus.PENDING, priority=5.0, deadline=140,
            progress=0.0, parent_id="h-sg-wave", children_ids=[],
            description="Migrate remaining user cohorts in waves",
        ))
        tree.add_node(GoalNode(
            id="h-t-tickets", name="Resolve support tickets", level="task",
            status=GoalStatus.PENDING, priority=4.0, deadline=150,
            progress=0.0, parent_id="h-sg-wave", children_ids=[],
            description="Keep ticket queue manageable during waves",
        ))

        # Validation tasks
        tree.add_node(GoalNode(
            id="h-t-validate", name="Validate SLA compliance", level="task",
            status=GoalStatus.PENDING, priority=4.0, deadline=180,
            progress=0.0, parent_id="h-sg-valid", children_ids=[],
            description="Confirm SLA score remains above threshold",
        ))
        tree.add_node(GoalNode(
            id="h-t-signoff", name="Stakeholder sign-off", level="task",
            status=GoalStatus.PENDING, priority=5.0, deadline=200,
            progress=0.0, parent_id="h-sg-valid", children_ids=[],
            description="Get final sign-off that migration is complete",
        ))

    # -- plan -----------------------------------------------------------

    def plan(
        self,
        step: int,
        valid_actions: List[Action],
        value_fn: Optional[ValueFn],
        state_digest: Dict[str, Any],
    ) -> Tuple[Action, Dict[str, Any]]:
        """Main planning entry-point.  Returns (action, plan_info)."""

        # Periodic replanning: prune overdue + activate ready tasks
        if step - self._last_replan_step >= self.REPLAN_INTERVAL:
            self._last_replan_step = step
            self.goal_tree.prune_overdue(step)
            self.goal_tree.activate_ready_tasks()

        # Run beam search
        best_action, best_value, reasoning = self.tot.search(
            valid_actions, value_fn, state_digest,
        )

        active_tasks = self.goal_tree.get_active_tasks()
        root_nodes = [n for n in self.goal_tree._nodes.values() if n.parent_id is None]
        goal_progress = {
            r.name: r.progress for r in root_nodes
        }

        plan_info: Dict[str, Any] = {
            "active_tasks": [t.name for t in active_tasks],
            "best_beam_value": best_value,
            "beam_reasoning": reasoning,
            "goal_progress": goal_progress,
        }
        return best_action, plan_info

    # -- update ---------------------------------------------------------

    def update(self, step: int, reward: float, milestone_hit: bool) -> None:
        """Update task progress based on reward signals."""
        active_tasks = self.goal_tree.get_active_tasks()
        if not active_tasks:
            return

        if reward > 20:
            # Large reward -> complete the highest-priority active task
            best_task = max(active_tasks, key=lambda t: t.priority)
            self.goal_tree.update_progress(best_task.id, 1.0)
        elif reward > 0:
            # Positive reward -> increment progress on all active tasks
            increment = min(reward / 100.0, 0.25)
            for task in active_tasks:
                new_progress = task.progress + increment
                self.goal_tree.update_progress(task.id, new_progress)

        if milestone_hit:
            # Activate any newly-ready tasks after completion
            self.goal_tree.activate_ready_tasks()
