"""
Agent Framework Comparisons — ReAct, AutoGPT, Vanilla RAG.

Implements the PATTERNS of popular agent frameworks using the same
heuristic scoring as V2Agent, enabling fair structural comparison
without requiring LLM API calls.

Each framework is evaluated on its DECISION ARCHITECTURE, not the
underlying model.  This isolates whether the ReAct observe→reason→act
loop, AutoGPT self-prompting, or RAG retrieval→generate pattern
produces better long-horizon planning than our V2 pipeline.

Frameworks Implemented:
  1. ReActAgent      — Observe → Reason (explicit trace) → Act
  2. AutoGPTAgent    — Goal decomposition + task queue + self-critique
  3. VanillaRAGAgent — Retrieve from memory → score candidates → act
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from ..engine.types import Action, ActionType, EnvID, Observation
from ..engine.environments.base import BaseEnvironment
from ..engine.environments.sales import EnterpriseSalesPipeline
from ..engine.environments.pm import ProgramRescueEnvironment
from ..engine.environments.hr_it import ITTransformationEnv
from ..engine.reward_shaping import RewardShaper
from ..memory.memory_system import MemorySystem
from ..planning.value_fn import _context_aware_heuristic


# ═══════════════════════════════════════════════════════════════════════
#  Base Interface (matches BaselineAgent from baselines.py)
# ═══════════════════════════════════════════════════════════════════════

class FrameworkAgent:
    """Base class for agent framework implementations."""

    name: str = "FrameworkAgent"

    def decide(self, obs: Observation, env: BaseEnvironment) -> Action:
        raise NotImplementedError

    def post_step(self, obs: Observation, action: Action,
                  shaped_reward: float, state_digest: Dict[str, Any]) -> None:
        pass

    def reset(self) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════
#  1. ReAct Agent (Yao et al. 2023)
# ═══════════════════════════════════════════════════════════════════════
#
#  Pattern:  Observation → Thought (reason about state) → Action
#
#  In real ReAct, the LLM generates explicit "Thought:" traces before
#  acting.  We emulate this with a structured reasoning pipeline that
#  evaluates multiple hypotheses before committing to an action.
#
#  Key difference from V2Agent: ReAct uses a flat observe-reason-act
#  loop with no hierarchical goal decomposition, no beam search, and
#  no learned value function.  It relies entirely on per-step reasoning.

class ReActAgent(FrameworkAgent):
    """
    ReAct-style agent: Observe → Reason → Act.

    Reasoning pipeline:
      1. Observe: extract key state features
      2. Think:   generate hypotheses for each valid action
      3. Score:   rank hypotheses by heuristic + recency bonus
      4. Act:     pick highest-scoring hypothesis

    No memory between steps (stateless reasoning).
    No planning ahead (single-step horizon).
    No learned value function (pure heuristic).
    """

    name = "ReAct"

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._last_reward: float = 0.0
        self._last_action: Optional[str] = None
        self._consecutive_same: int = 0

    def reset(self) -> None:
        self._last_reward = 0.0
        self._last_action = None
        self._consecutive_same = 0

    def decide(self, obs: Observation, env: BaseEnvironment) -> Action:
        digest = env.state_digest()
        actions = env.action_space()

        # ── Observation Phase ──
        state_features = self._observe(digest, obs)

        # ── Thought Phase: generate hypotheses ──
        hypotheses: List[Tuple[ActionType, float, str]] = []
        for a in actions:
            at = a if isinstance(a, ActionType) else a.action_type
            score, reasoning = self._reason(at, digest, state_features)
            hypotheses.append((at, score, reasoning))

        # ── Action Phase: pick best hypothesis ──
        hypotheses.sort(key=lambda h: h[1], reverse=True)
        best_at, best_score, best_reason = hypotheses[0]

        # Anti-repetition: if stuck on same action, explore
        if best_at.name == self._last_action:
            self._consecutive_same += 1
            if self._consecutive_same > 5 and len(hypotheses) > 1:
                best_at, best_score, best_reason = hypotheses[1]
                self._consecutive_same = 0
        else:
            self._consecutive_same = 0

        self._last_action = best_at.name
        return Action(action_type=best_at)

    def post_step(self, obs: Observation, action: Action,
                  shaped_reward: float, state_digest: Dict[str, Any]) -> None:
        self._last_reward = shaped_reward

    def _observe(self, digest: Dict, obs: Observation) -> Dict[str, Any]:
        """Extract key features from observation."""
        return {
            "step": digest.get("step", 0),
            "phase": digest.get("phase", 1),
            "has_shock": any("shock" in t for e in obs.events for t in e.tags),
            "last_reward_positive": self._last_reward > 0,
            "morale_low": digest.get("team_morale", 100) < 40,
            "sla_danger": digest.get("sla_score", 100) < 95,
        }

    def _reason(self, action_type: ActionType, digest: Dict,
                features: Dict) -> Tuple[float, str]:
        """Generate a scored hypothesis for this action."""
        base = _context_aware_heuristic(action_type.name, digest) / 10.0

        # ReAct reasoning: adjust based on observed features
        bonus = 0.0
        reasoning_parts = [f"base={base:.2f}"]

        if features["has_shock"] and action_type == ActionType.RESPOND_SHOCK:
            bonus += 0.3
            reasoning_parts.append("shock_detected→+0.3")

        if features["morale_low"] and action_type == ActionType.BOOST_MORALE:
            bonus += 0.2
            reasoning_parts.append("morale_critical→+0.2")

        if features["sla_danger"] and action_type in (ActionType.REVIEW_STATUS, ActionType.BOOST_MORALE):
            bonus += 0.1
            reasoning_parts.append("sla_danger→+0.1")

        # Penalty for repeating failed actions
        if not features["last_reward_positive"] and action_type.name == self._last_action:
            bonus -= 0.15
            reasoning_parts.append("repeat_fail→-0.15")

        return base + bonus, " | ".join(reasoning_parts)


# ═══════════════════════════════════════════════════════════════════════
#  2. AutoGPT Agent (Significant Gravitas 2023)
# ═══════════════════════════════════════════════════════════════════════
#
#  Pattern:  Goal → Decompose → Task Queue → Execute → Self-Critique
#
#  AutoGPT maintains a persistent goal stack and task queue across steps.
#  It decomposes high-level goals into sub-tasks, executes them
#  sequentially, and periodically self-critiques to replan.
#
#  Key differences from V2Agent:
#    - Uses a flat task queue (not a hierarchical goal tree)
#    - Self-critique every N steps (not continuous beam search)
#    - No learned value function
#    - No skill library

@dataclass
class TaskItem:
    action_type: str
    priority: float
    reason: str
    created_step: int


class AutoGPTAgent(FrameworkAgent):
    """
    AutoGPT-style agent with goal decomposition and self-critique.

    Architecture:
      1. Goal Analysis:    Inspect state, determine high-level objective
      2. Task Decomposition: Break objective into ordered action queue
      3. Execution:         Pop tasks and execute
      4. Self-Critique:     Every 15 steps, evaluate progress and replan
      5. Memory:            Simple sliding window of recent (action, reward) pairs
    """

    name = "AutoGPT"
    CRITIQUE_INTERVAL = 15

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._task_queue: Deque[TaskItem] = deque()
        self._step = 0
        self._recent_history: Deque[Tuple[str, float]] = deque(maxlen=20)
        self._total_reward = 0.0
        self._env_id: Optional[str] = None

    def reset(self) -> None:
        self._task_queue.clear()
        self._step = 0
        self._recent_history.clear()
        self._total_reward = 0.0
        self._env_id = None

    def decide(self, obs: Observation, env: BaseEnvironment) -> Action:
        self._step += 1
        digest = env.state_digest()
        actions = env.action_space()
        valid_names = set(
            a.name if isinstance(a, ActionType) else a.action_type.name
            for a in actions
        )

        if self._env_id is None:
            self._env_id = getattr(env, 'env_id', None)
            if hasattr(self._env_id, 'value'):
                self._env_id = self._env_id.value

        # ── Self-Critique: periodically re-evaluate strategy ──
        if self._step % self.CRITIQUE_INTERVAL == 0 or not self._task_queue:
            self._self_critique(digest, valid_names, obs)

        # ── Execute: pop next task from queue ──
        while self._task_queue:
            task = self._task_queue[0]
            if task.action_type in valid_names:
                self._task_queue.popleft()
                return Action(action_type=ActionType[task.action_type])
            # Task not valid — skip it
            self._task_queue.popleft()

        # Fallback: pick best heuristic action
        best_at = max(
            actions,
            key=lambda a: _context_aware_heuristic(
                a.name if isinstance(a, ActionType) else a.action_type.name,
                digest
            )
        )
        at = best_at if isinstance(best_at, ActionType) else best_at.action_type
        return Action(action_type=at)

    def post_step(self, obs: Observation, action: Action,
                  shaped_reward: float, state_digest: Dict[str, Any]) -> None:
        self._recent_history.append((action.action_type.name, shaped_reward))
        self._total_reward += shaped_reward

    def _self_critique(self, digest: Dict, valid_names: Set[str],
                       obs: Observation) -> None:
        """Evaluate recent performance and generate new task queue."""
        self._task_queue.clear()

        # Analyze recent performance
        recent_rewards = [r for _, r in self._recent_history]
        avg_recent = sum(recent_rewards) / max(1, len(recent_rewards)) if recent_rewards else 0

        # Check for shock events
        has_shock = any("shock" in t for e in obs.events for t in e.tags)

        # ── Goal Decomposition ──
        # Score each valid action and create prioritized task queue
        scored_actions: List[Tuple[str, float]] = []
        for name in valid_names:
            score = _context_aware_heuristic(name, digest)

            # AutoGPT self-critique adjustments
            # If recent rewards are negative, diversify
            if avg_recent < 0:
                # Check if we've been repeating this action
                recent_actions = [a for a, _ in self._recent_history]
                repeat_count = recent_actions[-5:].count(name) if recent_actions else 0
                score -= repeat_count * 2  # Penalize repetition

            # Shock override
            if has_shock and name == "RESPOND_SHOCK":
                score += 5

            scored_actions.append((name, score))

        scored_actions.sort(key=lambda x: x[1], reverse=True)

        # Generate task queue: top actions repeated proportionally to score
        for name, score in scored_actions:
            if score <= 0:
                continue
            # Higher scored actions get more slots in the queue
            repeats = max(1, int(score / 3))
            repeats = min(repeats, self.CRITIQUE_INTERVAL)
            for _ in range(repeats):
                self._task_queue.append(TaskItem(
                    action_type=name,
                    priority=score,
                    reason=f"auto_decompose(score={score:.1f})",
                    created_step=self._step,
                ))

        # Cap queue to next critique interval
        while len(self._task_queue) > self.CRITIQUE_INTERVAL:
            self._task_queue.pop()


# ═══════════════════════════════════════════════════════════════════════
#  3. Vanilla RAG Agent
# ═══════════════════════════════════════════════════════════════════════
#
#  Pattern:  Retrieve relevant context → Generate action
#
#  Standard RAG retrieves documents from a knowledge base to ground
#  the generation step.  Here we use episodic memory as the knowledge
#  base and TF-IDF retrieval to find relevant past events.
#
#  Key differences from V2Agent:
#    - No hierarchical planning (flat retrieve→generate)
#    - No skill library
#    - No beam search
#    - No goal tree
#    - Memory IS used (this is what RAG does), but only for retrieval

class VanillaRAGAgent(FrameworkAgent):
    """
    Vanilla RAG agent: Retrieve → Score → Act.

    Architecture:
      1. Build query from current observation
      2. Retrieve top-k relevant events from episodic memory
      3. Extract action hints from retrieved events
      4. Score valid actions by heuristic + retrieval boost
      5. Pick highest-scoring action

    Uses the SAME MemorySystem as V2Agent for fair comparison,
    but without planning, skills, value function, or goal tree.
    """

    name = "Vanilla RAG"

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._memory = MemorySystem()
        self._step = 0

    def reset(self) -> None:
        self._memory = MemorySystem()
        self._step = 0

    def decide(self, obs: Observation, env: BaseEnvironment) -> Action:
        self._step += 1
        digest = env.state_digest()
        actions = env.action_space()

        # ── Retrieve: query memory for relevant context ──
        query = self._build_query(obs)
        context = self._memory.context_for_agent(obs.step, query)

        # ── Extract action hints from retrieved context ──
        action_boosts = self._extract_boosts(context)

        # ── Score: heuristic + retrieval boost ──
        scored: List[Tuple[ActionType, float]] = []
        for a in actions:
            at = a if isinstance(a, ActionType) else a.action_type
            base = _context_aware_heuristic(at.name, digest) / 10.0
            boost = action_boosts.get(at.name, 0.0)
            scored.append((at, base + boost))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Small exploration probability
        if self._rng.random() < 0.1:
            chosen = self._rng.choice(scored)[0]
        else:
            chosen = scored[0][0]

        return Action(action_type=chosen)

    def post_step(self, obs: Observation, action: Action,
                  shaped_reward: float, state_digest: Dict[str, Any]) -> None:
        # Record events in memory (RAG knowledge base)
        from ..engine.types import Event
        for event in obs.events:
            self._memory.record_event(event)

        if not obs.events and obs.text:
            synth = Event(
                step=obs.step,
                event_type=action.action_type.name,
                text=obs.text,
                tags=list(obs.tags),
                reward=shaped_reward,
                importance=1.0 + (5.0 if abs(shaped_reward) > 10 else 0.0),
            )
            self._memory.record_event(synth)

    def _build_query(self, obs: Observation) -> str:
        parts = list(obs.tags)
        parts.append(f"phase={obs.phase}")
        parts.append(f"step={obs.step}")
        return " ".join(parts)

    def _extract_boosts(self, context: str) -> Dict[str, float]:
        """Extract action type mentions from retrieved context as boosts."""
        boosts: Dict[str, float] = {}
        if not context:
            return boosts

        context_upper = context.upper()
        for at in ActionType:
            if at.name in context_upper:
                # More mentions = higher boost (capped at 0.2)
                count = context_upper.count(at.name)
                boosts[at.name] = min(0.2, count * 0.05)

        return boosts


# ═══════════════════════════════════════════════════════════════════════
#  Framework Comparison Runner
# ═══════════════════════════════════════════════════════════════════════

ENV_ID_MAP = {
    "sales": EnvID.SALES,
    "pm": EnvID.PM,
    "hr_it": EnvID.HR_IT,
}


@dataclass
class FrameworkResult:
    """Result from one framework agent."""
    framework: str
    env_id: str
    episodes: int
    rewards: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    phases: List[int] = field(default_factory=list)
    terminals: List[str] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return sum(self.rewards) / max(1, len(self.rewards))

    @property
    def std_reward(self) -> float:
        if len(self.rewards) < 2:
            return 0.0
        mu = self.mean_reward
        var = sum((r - mu) ** 2 for r in self.rewards) / (len(self.rewards) - 1)
        return math.sqrt(var)

    @property
    def success_rate(self) -> float:
        s = sum(1 for t in self.terminals
                if t in ("deal_closed", "program_delivered", "migration_complete"))
        return s / max(1, len(self.terminals))

    @property
    def mean_steps(self) -> float:
        return sum(self.steps) / max(1, len(self.steps))

    @property
    def mean_phase(self) -> float:
        return sum(self.phases) / max(1, len(self.phases))


import math

ENV_FACTORIES = {
    "sales": EnterpriseSalesPipeline,
    "pm": ProgramRescueEnvironment,
    "hr_it": ITTransformationEnv,
}


def _run_framework_episode(
    agent: FrameworkAgent,
    env_id: str,
    seed: Optional[int] = None,
) -> Tuple[float, int, int, str]:
    """Run one episode with a framework agent."""
    env = ENV_FACTORIES[env_id]()
    obs = env.reset(seed)
    shaper = RewardShaper(env_id)
    total_reward = 0.0
    step_count = 0

    while not obs.done:
        action = agent.decide(obs, env)
        obs = env.step(action)
        step_count += 1

        digest = env.state_digest()
        events_dicts = [{"event_type": e.event_type, "tags": e.tags} for e in obs.events]
        shaped, _ = shaper.shape(obs.reward, digest, obs.step, events_dicts)
        total_reward += shaped
        agent.post_step(obs, action, shaped, digest)

    final = env.state_digest()
    terminal = "timeout"
    if final.get("deal_closed"):
        terminal = "deal_closed"
    elif final.get("program_delivered"):
        terminal = "program_delivered"
    elif final.get("migrated_pct", 0) >= 100:
        terminal = "migration_complete"
    elif final.get("budget_runway", 1) <= 0:
        terminal = "budget_exhausted"
    elif final.get("team_morale", 100) <= 0:
        terminal = "morale_collapsed"
    return total_reward, step_count, obs.phase, terminal


def _run_v2_episode_for_comparison(
    env_id: str, seed: Optional[int],
) -> Tuple[float, int, int, str]:
    """Run V2Agent episode for framework comparison."""
    from ..planning.planner import HierarchicalPlanner
    from ..planning.value_fn import ValueFunctionRegistry
    from ..skills.skill_library import SkillLibrary
    from ..agents.v2_agent import AgentContext, V2Agent

    env = ENV_FACTORIES[env_id]()
    obs = env.reset(seed)
    shaper = RewardShaper(env_id)

    memory = MemorySystem()
    planner = HierarchicalPlanner(beam_width=4, beam_depth=3)
    enum_id = ENV_ID_MAP.get(env_id, EnvID.PM)
    planner.init_goal_tree(enum_id)
    vf_reg = ValueFunctionRegistry()
    skill_lib = SkillLibrary()

    ctx = AgentContext(
        memory=memory, planner=planner, skill_library=skill_lib,
        vf_registry=vf_reg, env_id=env_id,
    )
    agent = V2Agent(ctx)

    total_reward = 0.0
    step_count = 0

    while not obs.done:
        action = agent.decide(obs, env)
        obs = env.step(action)
        step_count += 1

        digest = env.state_digest()
        events_dicts = [{"event_type": e.event_type, "tags": e.tags} for e in obs.events]
        shaped, _ = shaper.shape(obs.reward, digest, obs.step, events_dicts)
        total_reward += shaped
        agent.post_step(obs, action, shaped, digest)

    final = env.state_digest()
    terminal = "timeout"
    if final.get("deal_closed"):
        terminal = "deal_closed"
    elif final.get("program_delivered"):
        terminal = "program_delivered"
    elif final.get("migrated_pct", 0) >= 100:
        terminal = "migration_complete"
    return total_reward, step_count, obs.phase, terminal


def run_framework_comparison(
    env_ids: Optional[List[str]] = None,
    episodes: int = 5,
    seed: int = 42,
) -> Dict[str, List[FrameworkResult]]:
    """
    Compare V2Agent against ReAct, AutoGPT, and Vanilla RAG patterns.
    Returns {env_id: [FrameworkResult per agent]}.
    """
    if env_ids is None:
        env_ids = ["sales", "pm", "hr_it"]

    report: Dict[str, List[FrameworkResult]] = {}

    framework_agents = [
        ("ReAct", lambda s: ReActAgent(seed=s)),
        ("AutoGPT", lambda s: AutoGPTAgent(seed=s)),
        ("Vanilla RAG", lambda s: VanillaRAGAgent(seed=s)),
    ]

    for env_id in env_ids:
        print(f"\n{'='*70}")
        print(f"  FRAMEWORK COMPARISON: {env_id.upper()}")
        print(f"{'='*70}")

        results: List[FrameworkResult] = []

        # Run framework agents
        for fw_name, make_agent in framework_agents:
            fr = FrameworkResult(framework=fw_name, env_id=env_id, episodes=episodes)
            for i in range(episodes):
                ep_seed = seed + i
                agent = make_agent(ep_seed)
                reward, steps, phase, terminal = _run_framework_episode(
                    agent, env_id, ep_seed,
                )
                fr.rewards.append(reward)
                fr.steps.append(steps)
                fr.phases.append(phase)
                fr.terminals.append(terminal)
            print(f"  {fw_name:<20s} {episodes} episodes done  "
                  f"reward={fr.mean_reward:>+8.1f}")
            results.append(fr)

        # Run V2Agent for comparison
        fr = FrameworkResult(framework="V2Agent (ours)", env_id=env_id, episodes=episodes)
        for i in range(episodes):
            ep_seed = seed + i
            reward, steps, phase, terminal = _run_v2_episode_for_comparison(
                env_id, ep_seed,
            )
            fr.rewards.append(reward)
            fr.steps.append(steps)
            fr.phases.append(phase)
            fr.terminals.append(terminal)
        print(f"  {'V2Agent (ours)':<20s} {episodes} episodes done  "
              f"reward={fr.mean_reward:>+8.1f}")
        results.append(fr)

        report[env_id] = results

    return report


def print_framework_report(report: Dict[str, List[FrameworkResult]]) -> None:
    """Print formatted framework comparison with statistical significance."""
    from ..eval.ablation import welch_t_test, _sig_marker

    for env_id, results in report.items():
        v2 = results[-1]  # Last is V2Agent

        print(f"\n{'='*115}")
        print(f"  FRAMEWORK COMPARISON: {env_id.upper()}")
        print(f"{'='*115}")
        print(f"  {'Framework':<20s} {'Mean Reward':>12s} {'Std':>8s} "
              f"{'vs V2':>10s} {'t-stat':>8s} {'p-value':>9s} {'d':>6s} "
              f"{'Steps':>7s} {'Success':>8s}")
        print(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*10} "
              f"{'-'*8} {'-'*9} {'-'*6} {'-'*7} {'-'*8}")

        for fr in results:
            if fr.framework == "V2Agent (ours)":
                print(
                    f"  {fr.framework:<20s} {fr.mean_reward:>+12.1f} "
                    f"{fr.std_reward:>8.1f}   (baseline) "
                    f"{'---':>8s} {'---':>9s} {'---':>6s} "
                    f"{fr.mean_steps:>7.0f} {fr.success_rate:>7.0%}"
                )
            else:
                delta = fr.mean_reward - v2.mean_reward
                t_stat, p_val, d = welch_t_test(v2.rewards, fr.rewards)
                sig = _sig_marker(p_val)
                print(
                    f"  {fr.framework:<20s} {fr.mean_reward:>+12.1f} "
                    f"{fr.std_reward:>8.1f} {delta:>+10.1f} "
                    f"{t_stat:>+8.2f} {p_val:>8.4f}{sig:>2s} "
                    f"{d:>+6.2f} "
                    f"{fr.mean_steps:>7.0f} {fr.success_rate:>7.0%}"
                )

        # Significance summary
        print(f"\n  Statistical Significance (Welch's t-test vs V2Agent):")
        print(f"  {'Framework':<20s} {'Δ':>10s} {'t':>8s} {'p':>9s} {'Cohen d':>9s} {'Sig':>5s}")
        print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*9} {'-'*9} {'-'*5}")

        for fr in results[:-1]:
            delta = fr.mean_reward - v2.mean_reward
            t_stat, p_val, d = welch_t_test(v2.rewards, fr.rewards)
            sig = _sig_marker(p_val)
            d_label = ("very large" if abs(d) > 1.2 else "large" if abs(d) > 0.8
                       else "medium" if abs(d) > 0.5 else "small")
            print(
                f"  {fr.framework:<20s} {delta:>+10.1f} {t_stat:>+8.2f} "
                f"{p_val:>9.4f} {d:>+9.2f} {sig:>5s}  ({d_label})"
            )


def run_and_print_framework_comparison(
    env_ids: Optional[List[str]] = None,
    episodes: int = 5,
    seed: int = 42,
) -> Dict[str, List[FrameworkResult]]:
    """Convenience: run comparison and print results."""
    report = run_framework_comparison(env_ids, episodes, seed)
    print_framework_report(report)
    return report
