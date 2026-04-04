"""
Multi-Agent Coordination — Upgrade 1 (HIGH Priority).

Three specialised department agents (Engineering, Product, Finance) with:
  - Genuine information asymmetry via ObservationFilter
  - Shared blackboard negotiation protocol
  - Shapley-inspired shared reward attribution

Builds on the existing MultiDepartmentEnvironment foundation.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..engine.types import Action, ActionType, EnvID, Event, Observation
from ..engine.environments.base import BaseEnvironment
from ..engine.environments.scaled import MultiDepartmentEnvironment
from ..memory.memory_system import MemorySystem
from ..planning.planner import HierarchicalPlanner
from ..planning.value_fn import ValueFunctionRegistry
from ..skills.skill_library import SkillLibrary, Transition
from .v2_agent import AgentContext, V2Agent


# ═════════════════════════════════════════════════════════════════════════════
#  Department Roles
# ═════════════════════════════════════════════════════════════════════════════

class Department(Enum):
    ENGINEERING = "engineering"
    PRODUCT = "product"
    FINANCE = "finance"


# Fields each department CAN see (whitelist approach)
_VISIBILITY: Dict[Department, set] = {
    Department.ENGINEERING: {
        "step", "phase", "risks_total", "risks_resolved", "team_morale",
        "ws_progress_avg", "program_delivered", "total_reward",
        # Engineering sees workstream details and risk counts, NOT budget
    },
    Department.PRODUCT: {
        "step", "phase", "risks_total", "risks_resolved", "team_morale",
        "ws_progress_avg", "program_delivered", "total_reward",
        "poc_score", "avg_relationship", "live_stakeholders",
        # Product sees stakeholder engagement and progress, limited budget
    },
    Department.FINANCE: {
        "step", "phase", "budget_runway", "total_reward",
        "deal_value", "budget_frozen", "program_delivered",
        # Finance sees budget and financials, NOT velocity/morale/risk details
    },
}

# Fields explicitly HIDDEN from each department
_HIDDEN: Dict[Department, set] = {
    Department.ENGINEERING: {"budget_runway", "deal_value", "budget_frozen"},
    Department.PRODUCT: {"budget_runway"},
    Department.FINANCE: {
        "risks_total", "risks_resolved", "team_morale", "ws_progress_avg",
        "poc_score", "avg_relationship", "live_stakeholders",
        "fulfilled_instructions", "migrated_pct", "sla_score", "ticket_queue",
    },
}


# ═════════════════════════════════════════════════════════════════════════════
#  Observation Filter
# ═════════════════════════════════════════════════════════════════════════════

class ObservationFilter:
    """Filters state_digest to enforce information asymmetry per department."""

    def __init__(self, department: Department) -> None:
        self.department = department
        self._visible = _VISIBILITY[department]
        self._hidden = _HIDDEN[department]

    def filter_digest(self, state_digest: Dict[str, Any]) -> Dict[str, Any]:
        """Return a filtered copy with only visible fields."""
        filtered: Dict[str, Any] = {}
        for key, val in state_digest.items():
            if key in self._hidden:
                continue
            if key in self._visible:
                filtered[key] = val
            # For nested dicts (e.g., multi-dept "departments" list),
            # recursively filter each sub-department
            elif key == "departments" and isinstance(val, list):
                filtered[key] = [self.filter_digest(d) for d in val]
            elif key == "shared_budget":
                # Only Finance sees the shared budget
                if self.department == Department.FINANCE:
                    filtered[key] = val
            elif key == "n_departments":
                filtered[key] = val
        return filtered

    def filter_observation(self, obs: Observation) -> Observation:
        """Filter observation text to remove hidden information."""
        # Keep structural fields, filter info dict
        filtered_info = {}
        for key, val in obs.info.items():
            if key not in self._hidden:
                filtered_info[key] = val

        return Observation(
            step=obs.step,
            phase=obs.phase,
            text=obs.text,
            events=obs.events,
            reward=obs.reward,
            done=obs.done,
            info=filtered_info,
            tags=obs.tags,
        )


# ═════════════════════════════════════════════════════════════════════════════
#  Negotiation Protocol — Shared Blackboard
# ═════════════════════════════════════════════════════════════════════════════

class MessageType(Enum):
    REQUEST = "REQUEST"       # Ask another dept for something
    INFORM = "INFORM"         # Share information
    PROPOSE = "PROPOSE"       # Propose a joint action
    ACCEPT = "ACCEPT"         # Accept a proposal
    REJECT = "REJECT"         # Reject a proposal
    ESCALATE = "ESCALATE"     # Escalate a cross-department issue


@dataclass
class Message:
    """A structured message posted to the shared blackboard."""
    msg_id: int
    sender: Department
    recipient: Optional[Department]    # None = broadcast
    msg_type: MessageType
    content: str
    priority: float = 1.0
    step: int = 0
    resolved: bool = False
    parent_id: Optional[int] = None    # For threaded replies


class Blackboard:
    """
    Shared communication channel between department agents.
    Agents post and read structured messages. Each agent only sees
    messages addressed to them or broadcast messages.
    """

    def __init__(self) -> None:
        self._messages: List[Message] = []
        self._next_id: int = 0
        self._unread: Dict[Department, List[int]] = defaultdict(list)

    def post(
        self,
        sender: Department,
        msg_type: MessageType,
        content: str,
        recipient: Optional[Department] = None,
        priority: float = 1.0,
        step: int = 0,
        parent_id: Optional[int] = None,
    ) -> int:
        """Post a message. Returns the message ID."""
        msg = Message(
            msg_id=self._next_id,
            sender=sender,
            recipient=recipient,
            msg_type=msg_type,
            content=content,
            priority=priority,
            step=step,
            parent_id=parent_id,
        )
        self._messages.append(msg)
        self._next_id += 1

        # Mark as unread for recipient(s)
        if recipient is not None:
            self._unread[recipient].append(msg.msg_id)
        else:
            # Broadcast: mark for all departments except sender
            for dept in Department:
                if dept != sender:
                    self._unread[dept].append(msg.msg_id)

        return msg.msg_id

    def read_unread(self, department: Department) -> List[Message]:
        """Read and clear unread messages for a department."""
        ids = self._unread.pop(department, [])
        return [self._messages[i] for i in ids if i < len(self._messages)]

    def read_all(self, department: Department, last_n: int = 20) -> List[Message]:
        """Read recent messages visible to a department."""
        visible: List[Message] = []
        for msg in self._messages[-last_n * 3:]:
            if msg.recipient is None or msg.recipient == department or msg.sender == department:
                visible.append(msg)
        return visible[-last_n:]

    def get_pending_requests(self, department: Department) -> List[Message]:
        """Get unresolved REQUESTs addressed to this department."""
        return [
            m for m in self._messages
            if m.recipient == department
            and m.msg_type == MessageType.REQUEST
            and not m.resolved
        ]

    def resolve(self, msg_id: int) -> None:
        """Mark a message as resolved."""
        if 0 <= msg_id < len(self._messages):
            self._messages[msg_id].resolved = True

    def context_string(self, department: Department, last_n: int = 10) -> str:
        """Render recent messages as context for an agent."""
        msgs = self.read_all(department, last_n)
        if not msgs:
            return ""
        lines = ["=== Blackboard Messages ==="]
        for m in msgs:
            direction = "BROADCAST" if m.recipient is None else f"-> {m.recipient.value}"
            lines.append(
                f"[Step {m.step}] {m.sender.value} {direction} "
                f"({m.msg_type.value}, p={m.priority:.0f}): {m.content}"
            )
        return "\n".join(lines)

    @property
    def total_messages(self) -> int:
        return len(self._messages)


# ═════════════════════════════════════════════════════════════════════════════
#  Shared Reward Attribution
# ═════════════════════════════════════════════════════════════════════════════

class RewardAttribution(Enum):
    EQUAL = "equal"
    ACTIVITY = "activity"
    SHAPLEY = "shapley"


class SharedRewardAttributor:
    """
    Distributes shared environment reward across departments.

    Three strategies:
      - EQUAL:    r / n_agents
      - ACTIVITY: proportional to action count in recent window
      - SHAPLEY:  marginal contribution estimation (approximated)
    """

    def __init__(
        self,
        strategy: RewardAttribution = RewardAttribution.SHAPLEY,
        window: int = 20,
    ) -> None:
        self.strategy = strategy
        self._window = window
        self._action_counts: Dict[Department, int] = defaultdict(int)
        self._marginal_rewards: Dict[Department, List[float]] = defaultdict(list)

    def record_action(self, dept: Department) -> None:
        """Record that a department took an action."""
        self._action_counts[dept] += 1

    def record_marginal(self, dept: Department, reward: float) -> None:
        """Record the reward obtained from a department's action (for Shapley)."""
        self._marginal_rewards[dept].append(reward)

    def attribute(
        self,
        total_reward: float,
        departments: List[Department],
    ) -> Dict[Department, float]:
        """Attribute total_reward across departments."""
        n = len(departments)
        if n == 0:
            return {}

        if self.strategy == RewardAttribution.EQUAL:
            share = total_reward / n
            return {d: share for d in departments}

        elif self.strategy == RewardAttribution.ACTIVITY:
            total_actions = sum(self._action_counts.get(d, 1) for d in departments)
            if total_actions == 0:
                total_actions = n
            return {
                d: total_reward * self._action_counts.get(d, 1) / total_actions
                for d in departments
            }

        else:  # SHAPLEY approximation
            # Use average marginal contribution as Shapley value proxy
            marginals: Dict[Department, float] = {}
            for d in departments:
                recent = self._marginal_rewards.get(d, [])[-self._window:]
                if recent:
                    marginals[d] = sum(recent) / len(recent)
                else:
                    marginals[d] = 0.0

            total_marginal = sum(abs(v) for v in marginals.values())
            if total_marginal == 0:
                share = total_reward / n
                return {d: share for d in departments}

            return {
                d: total_reward * abs(marginals[d]) / total_marginal
                for d in departments
            }

    def reset(self) -> None:
        self._action_counts.clear()
        self._marginal_rewards.clear()


# ═════════════════════════════════════════════════════════════════════════════
#  Department Agent — Specialised V2Agent with Filtered Observations
# ═════════════════════════════════════════════════════════════════════════════

class DepartmentAgent:
    """
    A specialised agent for one department.
    Wraps V2Agent with observation filtering and blackboard access.
    """

    def __init__(
        self,
        department: Department,
        dept_id: int,
        ctx: AgentContext,
        blackboard: Blackboard,
    ) -> None:
        self.department = department
        self.dept_id = dept_id
        self.agent = V2Agent(ctx)
        self.obs_filter = ObservationFilter(department)
        self.blackboard = blackboard
        self._step_count: int = 0

    def decide(self, obs: Observation, env: BaseEnvironment) -> Action:
        """Decide with filtered observations and blackboard context."""
        # Filter the observation
        filtered_obs = self.obs_filter.filter_observation(obs)

        # Check blackboard for pending requests
        pending = self.blackboard.get_pending_requests(self.department)
        if pending:
            # Auto-respond to information requests
            for req in pending[:2]:
                self._handle_request(req, env)

        # Decide using the filtered observation
        action = self.agent.decide(filtered_obs, env)

        # Tag action with dept_id for MultiDepartmentEnvironment
        action.params["dept_id"] = self.dept_id
        self._step_count += 1

        return action

    def post_step(
        self,
        obs: Observation,
        action: Action,
        shaped_reward: float,
        state_digest: Dict[str, Any],
    ) -> None:
        """Post-step with filtered state digest."""
        filtered_digest = self.obs_filter.filter_digest(state_digest)
        self.agent.post_step(obs, action, shaped_reward, filtered_digest)

        # Post significant events to blackboard
        if abs(shaped_reward) > 10:
            self.blackboard.post(
                sender=self.department,
                msg_type=MessageType.INFORM,
                content=f"Significant event (reward={shaped_reward:+.1f}): {obs.text[:100]}",
                priority=min(abs(shaped_reward) / 10, 5.0),
                step=obs.step,
            )

        # Escalate blocking situations
        if "blocked" in obs.text.lower():
            self.blackboard.post(
                sender=self.department,
                msg_type=MessageType.ESCALATE,
                content=f"Department blocked: {obs.text[:100]}",
                priority=5.0,
                step=obs.step,
            )

    def _handle_request(self, req: Message, env: BaseEnvironment) -> None:
        """Auto-respond to requests based on available information."""
        digest = env.state_digest()
        filtered = self.obs_filter.filter_digest(digest)

        # Respond with what this department can see
        info_parts = [f"{k}={v}" for k, v in filtered.items()
                      if not isinstance(v, (list, dict))]
        response = "; ".join(info_parts[:5])

        self.blackboard.post(
            sender=self.department,
            msg_type=MessageType.INFORM,
            content=f"Re: {req.content[:50]}... | {response}",
            recipient=req.sender,
            step=req.step,
            parent_id=req.msg_id,
        )
        self.blackboard.resolve(req.msg_id)


# ═════════════════════════════════════════════════════════════════════════════
#  Multi-Agent Coordinator
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MultiAgentResult:
    """Result from a multi-agent episode."""
    total_steps: int = 0
    total_reward: float = 0.0
    shaped_reward: float = 0.0
    per_dept_reward: Dict[str, float] = field(default_factory=dict)
    per_dept_actions: Dict[str, int] = field(default_factory=dict)
    messages_exchanged: int = 0
    terminal_reason: str = "timeout"
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  Multi-Agent Episode Result",
            "=" * 60,
            f"  Steps:      {self.total_steps}",
            f"  Reward:     {self.shaped_reward:+.2f}",
            f"  Terminal:   {self.terminal_reason}",
            f"  Messages:   {self.messages_exchanged}",
            f"  Elapsed:    {self.elapsed_seconds:.2f}s",
            "",
            "  Per-Department Rewards:",
        ]
        for dept, reward in self.per_dept_reward.items():
            actions = self.per_dept_actions.get(dept, 0)
            lines.append(f"    {dept:>12}: {reward:+.2f}  ({actions} actions)")
        lines.append("=" * 60)
        return "\n".join(lines)


class MultiAgentCoordinator:
    """
    Orchestrates multiple DepartmentAgents over a MultiDepartmentEnvironment.

    Manages:
      - Round-robin or priority-based department scheduling
      - Blackboard negotiation rounds between steps
      - Shared reward attribution after each step
    """

    def __init__(
        self,
        vf_registry: ValueFunctionRegistry,
        skill_library: SkillLibrary,
        reward_strategy: RewardAttribution = RewardAttribution.SHAPLEY,
        beam_width: int = 4,
        beam_depth: int = 3,
    ) -> None:
        self._vf_registry = vf_registry
        self._skill_library = skill_library
        self._reward_strategy = reward_strategy
        self._beam_width = beam_width
        self._beam_depth = beam_depth

    def run_episode(
        self,
        env: MultiDepartmentEnvironment,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> MultiAgentResult:
        """Run a full multi-agent episode."""
        import time
        t0 = time.time()

        obs = env.reset(seed)
        blackboard = Blackboard()
        attributor = SharedRewardAttributor(strategy=self._reward_strategy)

        # Create one DepartmentAgent per department
        departments = [Department.ENGINEERING, Department.PRODUCT, Department.FINANCE]
        n_depts = min(len(departments), len(env._departments))
        departments = departments[:n_depts]

        agents: List[DepartmentAgent] = []
        for i, dept in enumerate(departments):
            memory = MemorySystem()
            planner = HierarchicalPlanner(
                beam_width=self._beam_width,
                beam_depth=self._beam_depth,
            )
            planner.init_goal_tree(EnvID.PM)

            ctx = AgentContext(
                memory=memory,
                planner=planner,
                skill_library=self._skill_library,
                vf_registry=self._vf_registry,
                env_id="pm",
                beam_width=self._beam_width,
                beam_depth=self._beam_depth,
            )
            agents.append(DepartmentAgent(dept, i, ctx, blackboard))

        # Episode loop
        total_reward = 0.0
        shaped_total = 0.0
        step_counter = 0
        per_dept_reward: Dict[str, float] = {d.value: 0.0 for d in departments}
        per_dept_actions: Dict[str, int] = {d.value: 0 for d in departments}

        while not obs.done and step_counter < env.MAX_STEPS:
            # Round-robin: each department acts in turn
            for i, (dept, agent) in enumerate(zip(departments, agents)):
                if obs.done:
                    break

                # Agent decides (with filtered observations)
                action = agent.decide(obs, env)

                # Environment step
                obs = env.step(action)
                step_counter += 1

                reward = obs.reward
                total_reward += reward
                shaped_total += reward

                # Record for attribution
                attributor.record_action(dept)
                attributor.record_marginal(dept, reward)

                # Agent post-step
                state_digest = env.state_digest()
                agent.post_step(obs, action, reward, state_digest)
                per_dept_actions[dept.value] += 1

                if obs.done:
                    break

            # Negotiation round after all departments have acted
            self._negotiation_round(agents, blackboard, step_counter)

            if verbose and step_counter % 30 == 0:
                print(
                    f"  [step {step_counter:>4}] reward={shaped_total:+.1f} "
                    f"msgs={blackboard.total_messages}"
                )

        # Attribute final reward
        attributed = attributor.attribute(shaped_total, departments)
        for dept, share in attributed.items():
            per_dept_reward[dept.value] = share

        # Terminal reason
        state = env.state_digest()
        terminal = "timeout"
        if state.get("shared_budget", 1) <= 0:
            terminal = "budget_exhausted"

        elapsed = time.time() - t0

        return MultiAgentResult(
            total_steps=step_counter,
            total_reward=total_reward,
            shaped_reward=shaped_total,
            per_dept_reward=per_dept_reward,
            per_dept_actions=per_dept_actions,
            messages_exchanged=blackboard.total_messages,
            terminal_reason=terminal,
            elapsed_seconds=elapsed,
        )

    def _negotiation_round(
        self,
        agents: List[DepartmentAgent],
        blackboard: Blackboard,
        step: int,
    ) -> None:
        """One round of inter-department negotiation via blackboard."""
        for agent in agents:
            # Each agent reads unread messages
            unread = blackboard.read_unread(agent.department)
            for msg in unread:
                # Auto-propose help for escalations
                if msg.msg_type == MessageType.ESCALATE and msg.sender != agent.department:
                    blackboard.post(
                        sender=agent.department,
                        msg_type=MessageType.PROPOSE,
                        content=f"Can assist with: {msg.content[:60]}",
                        recipient=msg.sender,
                        priority=msg.priority * 0.8,
                        step=step,
                        parent_id=msg.msg_id,
                    )

            # Request information from other departments periodically
            if step % 50 == 0:
                for other_agent in agents:
                    if other_agent.department != agent.department:
                        blackboard.post(
                            sender=agent.department,
                            msg_type=MessageType.REQUEST,
                            content=f"Status update request from {agent.department.value}",
                            recipient=other_agent.department,
                            priority=1.0,
                            step=step,
                        )
