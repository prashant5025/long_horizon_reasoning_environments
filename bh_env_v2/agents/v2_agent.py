"""
V2Agent — Five-stage decision pipeline for BusinessHorizonENV v2.

Integrates all six v2 improvements on every decision step:
  Stage 1: Memory retrieval
  Stage 2: Skill library check
  Stage 3: Hierarchical planning + beam search
  Stage 4: Action selection (skill override or planner output)
  Stage 5: Post-step bookkeeping (memory record, phase tracking)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..engine.types import Action, ActionType, EnvID, Event, Observation
from ..engine.environments.base import BaseEnvironment
from ..memory.memory_system import MemorySystem
from ..planning.planner import HierarchicalPlanner
from ..planning.value_fn import ValueFunctionRegistry
from ..skills.skill_library import SkillLibrary, Transition


# ═════════════════════════════════════════════════════════════════════════════
#  Agent Context — shared references injected by the harness
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentContext:
    """References to cross-episode components owned by the harness."""
    memory: MemorySystem
    planner: HierarchicalPlanner
    skill_library: SkillLibrary
    vf_registry: ValueFunctionRegistry
    env_id: str
    beam_width: int = 4
    beam_depth: int = 3


# ═════════════════════════════════════════════════════════════════════════════
#  V2Agent
# ═════════════════════════════════════════════════════════════════════════════

class V2Agent:
    """
    Five-stage decision pipeline that integrates memory, skills,
    hierarchical planning, and the learned value function.
    """

    def __init__(self, ctx: AgentContext) -> None:
        self._ctx = ctx
        self._current_phase: int = 1
        self._trajectory: List[Transition] = []
        self._prev_state_digest: Optional[Dict[str, Any]] = None
        self._prev_action_type: Optional[str] = None
        self._executing_skill: Optional[List[str]] = None
        self._skill_id: Optional[str] = None
        self._skill_step: int = 0
        self._skill_reward_acc: float = 0.0

    # ── main entry point ──────────────────────────────────────────────────

    def decide(
        self,
        obs: Observation,
        env: BaseEnvironment,
    ) -> Action:
        """Run the five-stage pipeline and return the chosen action."""
        step = obs.step
        state_digest = env.state_digest()
        valid_actions = env.action_space()
        env_id = self._ctx.env_id
        tags = list(obs.tags)

        # Wrap raw ActionType enums into Action objects for planner/beam search
        action_types = valid_actions  # may be List[ActionType] or List[Action]
        actions: List[Action] = []
        for a in action_types:
            if isinstance(a, ActionType):
                actions.append(Action(action_type=a))
            else:
                actions.append(a)

        # ─── Stage 1: Memory Retrieval ────────────────────────────────
        query = self._build_query(obs)
        memory_context = self._ctx.memory.context_for_agent(step, query)

        # ─── Stage 2: Skill Library Check ─────────────────────────────
        # If currently executing a multi-step skill, continue it
        if self._executing_skill and self._skill_step < len(self._executing_skill):
            action = self._continue_skill(actions)
            if action is not None:
                return action
            # Skill can't continue (action not valid); abandon it
            self._finish_skill(success=False)

        # Check for a new skill match
        skill = self._ctx.skill_library.query(tags, env_id)
        if skill is not None:
            self._executing_skill = list(skill.action_sequence)
            self._skill_id = skill.skill_id
            self._skill_step = 0
            self._skill_reward_acc = 0.0
            action = self._continue_skill(actions)
            if action is not None:
                return action
            # First action not valid; abandon immediately
            self._finish_skill(success=False)

        # ─── Stage 3: Hierarchical Planning + Beam Search ─────────────
        value_fn = self._ctx.vf_registry.make_hybrid_scorer(env_id)
        best_action, plan_info = self._ctx.planner.plan(
            step, actions, value_fn, state_digest,
        )

        return best_action

    # ── post-step (called by harness after env.step) ──────────────────────

    def post_step(
        self,
        obs: Observation,
        action: Action,
        shaped_reward: float,
        state_digest: Dict[str, Any],
    ) -> None:
        """Stage 5: Post-step bookkeeping — memory recording and phase tracking."""
        step = obs.step
        env_id = self._ctx.env_id

        # Record events in memory
        for event in obs.events:
            self._ctx.memory.record_event(event)

        # If no events, create a synthetic event from the observation text
        if not obs.events and obs.text:
            synth = Event(
                step=step,
                event_type=action.action_type.name,
                text=obs.text,
                tags=list(obs.tags),
                reward=shaped_reward,
                importance=1.0 + (5.0 if abs(shaped_reward) > 10 else 0.0),
            )
            self._ctx.memory.record_event(synth)

        # Phase transition detection
        if obs.phase != self._current_phase:
            old_phase = self._current_phase
            self._current_phase = obs.phase
            self._ctx.memory.notify_phase_transition(old_phase, obs.phase, step)

        # Track trajectory for skill extraction
        transition = Transition(
            step=step,
            state_digest=state_digest,
            action_type=action.action_type.name,
            shaped_reward=shaped_reward,
            done=obs.done,
            context_tags=list(obs.tags),
            priority=abs(shaped_reward) + (10.0 if obs.done else 0.0),
        )
        self._trajectory.append(transition)

        # Accumulate skill reward if executing a skill
        if self._executing_skill is not None:
            self._skill_reward_acc += shaped_reward
            if self._skill_step >= len(self._executing_skill):
                self._finish_skill(success=shaped_reward > 0)

        # Store previous state for VF updates (done by harness)
        self._prev_state_digest = state_digest
        self._prev_action_type = action.action_type.name

    # ── accessors ─────────────────────────────────────────────────────────

    @property
    def trajectory(self) -> List[Transition]:
        return self._trajectory

    @property
    def prev_state_digest(self) -> Optional[Dict[str, Any]]:
        return self._prev_state_digest

    @property
    def prev_action_type(self) -> Optional[str]:
        return self._prev_action_type

    # ── internal helpers ──────────────────────────────────────────────────

    def _build_query(self, obs: Observation) -> str:
        """Build a retrieval query from the observation."""
        parts: List[str] = []
        if obs.tags:
            parts.extend(obs.tags)
        parts.append(f"phase={obs.phase}")
        parts.append(f"step={obs.step}")
        return " ".join(parts)

    def _continue_skill(self, valid_actions: List[Action]) -> Optional[Action]:
        """Try to execute the next action in the current skill sequence."""
        if self._executing_skill is None or self._skill_step >= len(self._executing_skill):
            return None

        target_name = self._executing_skill[self._skill_step]
        self._skill_step += 1

        # Find a matching valid action
        for action in valid_actions:
            if action.action_type.name == target_name:
                return action
        return None

    def _finish_skill(self, success: bool) -> None:
        """Record skill outcome and reset skill execution state."""
        if self._skill_id is not None:
            self._ctx.skill_library.record_outcome(
                self._skill_id,
                self._skill_reward_acc,
                success,
            )
        self._executing_skill = None
        self._skill_id = None
        self._skill_step = 0
        self._skill_reward_acc = 0.0

    def update_semantic_memory(self, step: int, state_digest: Dict[str, Any]) -> None:
        """Update semantic beliefs from episode summary (called at episode end)."""
        sem = self._ctx.memory.semantic
        for key, val in state_digest.items():
            if isinstance(val, (int, float, bool)):
                sem.update_belief(
                    concept=key,
                    belief=f"{key}={val}",
                    confidence=0.9,
                    step=step,
                )
