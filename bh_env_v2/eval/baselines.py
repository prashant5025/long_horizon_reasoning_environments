"""
Baseline Agents & Benchmark Comparison Harness
================================================
Provides reference agents to contextualize V2Agent performance:

  1. RandomAgent       — uniform random action selection (lower bound)
  2. GreedyAgent       — picks highest-heuristic-value action each step
  3. RuleBasedAgent    — hand-crafted conditional rules per environment
  4. PlannerOnlyAgent  — hierarchical planner with NO memory/skills/VF

BenchmarkSuite runs all agents + V2Agent across environments and
produces a structured comparison report with statistical significance.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..engine.types import Action, ActionType, EnvID, Observation
from ..engine.environments.base import BaseEnvironment
from ..engine.environments.sales import EnterpriseSalesPipeline
from ..engine.environments.pm import ProgramRescueEnvironment
from ..engine.environments.hr_it import ITTransformationEnv
from ..engine.reward_shaping import RewardShaper
from ..memory.memory_system import MemorySystem
from ..planning.planner import HierarchicalPlanner
from ..planning.value_fn import ValueFunctionRegistry
from ..skills.skill_library import ExperienceReplay, SkillExtractor, SkillLibrary
from ..agents.v2_agent import AgentContext, V2Agent


# ═══════════════════════════════════════════════════════════════════════
#  Benchmark Result
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """Result from a single agent on a single environment."""
    agent_name: str
    env_id: str
    episodes: int
    rewards: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    phases: List[int] = field(default_factory=list)
    terminals: List[str] = field(default_factory=list)
    elapsed: List[float] = field(default_factory=list)

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
    def mean_steps(self) -> float:
        return sum(self.steps) / max(1, len(self.steps))

    @property
    def mean_phase(self) -> float:
        return sum(self.phases) / max(1, len(self.phases))

    @property
    def success_rate(self) -> float:
        successes = sum(
            1 for t in self.terminals
            if t in ("deal_closed", "program_delivered", "migration_complete")
        )
        return successes / max(1, len(self.terminals))

    def ci_95(self) -> Tuple[float, float]:
        """95% confidence interval for mean reward (t-approximation)."""
        n = len(self.rewards)
        if n < 2:
            return (self.mean_reward, self.mean_reward)
        se = self.std_reward / math.sqrt(n)
        # t-value for 95% CI (approximate for n > 5)
        t_val = 2.0 if n > 30 else 2.228 if n > 10 else 2.776
        return (self.mean_reward - t_val * se, self.mean_reward + t_val * se)

    def summary_row(self) -> str:
        ci_lo, ci_hi = self.ci_95()
        return (
            f"  {self.agent_name:<20s} "
            f"{self.mean_reward:>+8.1f} +/- {self.std_reward:>6.1f}  "
            f"[{ci_lo:>+7.1f}, {ci_hi:>+7.1f}]  "
            f"steps={self.mean_steps:>5.0f}  "
            f"phase={self.mean_phase:>3.1f}  "
            f"success={self.success_rate:>5.1%}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Base Agent Interface
# ═══════════════════════════════════════════════════════════════════════

class BaselineAgent:
    """Interface for baseline agents. Minimal: just decide(obs, env)."""

    name: str = "BaselineAgent"

    def decide(self, obs: Observation, env: BaseEnvironment) -> Action:
        raise NotImplementedError

    def post_step(self, obs: Observation, action: Action,
                  shaped_reward: float, state_digest: Dict[str, Any]) -> None:
        pass

    def reset(self) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════
#  1. Random Agent (Lower Bound)
# ═══════════════════════════════════════════════════════════════════════

class RandomAgent(BaselineAgent):
    """Selects a uniformly random valid action each step."""

    name = "Random"

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def decide(self, obs: Observation, env: BaseEnvironment) -> Action:
        actions = env.action_space()
        choice = self._rng.choice(actions)
        if isinstance(choice, ActionType):
            return Action(action_type=choice)
        return choice

    def reset(self) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════
#  2. Greedy Agent (Immediate Heuristic)
# ═══════════════════════════════════════════════════════════════════════

class GreedyAgent(BaselineAgent):
    """
    Picks the action with the highest immediate heuristic value.
    Heuristic scores are based on the current state digest and
    simple priority rules (e.g., resolve risks > advance work).
    No planning ahead, no memory.
    """

    name = "Greedy"

    # Priority ordering: higher = better default choice
    _ACTION_PRIORITIES = {
        ActionType.RESPOND_SHOCK: 100,
        ActionType.RESOLVE_RISK: 80,
        ActionType.FULFILL_INSTRUCTION: 70,
        ActionType.ADVANCE_WORKSTREAM: 60,
        ActionType.MIGRATE_COHORT: 55,
        ActionType.ADVANCE_DEAL: 50,
        ActionType.CONTACT_STAKEHOLDER: 40,
        ActionType.RUN_POC: 35,
        ActionType.ALLOCATE_BUDGET: 30,
        ActionType.BOOST_MORALE: 25,
        ActionType.REVIEW_STATUS: 10,
        ActionType.NOOP: 0,
    }

    def _score_action(self, action_type: ActionType,
                      digest: Dict[str, Any]) -> float:
        base = self._ACTION_PRIORITIES.get(action_type, 0)

        # Context-sensitive adjustments
        morale = digest.get("team_morale", 80)
        if morale < 40 and action_type == ActionType.BOOST_MORALE:
            base += 50  # Critical morale -> boost urgently

        budget = digest.get("budget_runway", 1_000_000)
        if budget < 500_000 and action_type == ActionType.ALLOCATE_BUDGET:
            base += 40

        if digest.get("ransomware_active") and action_type == ActionType.RESPOND_SHOCK:
            base += 60

        return float(base)

    def decide(self, obs: Observation, env: BaseEnvironment) -> Action:
        actions = env.action_space()
        digest = env.state_digest()
        best_score = -1.0
        best_action = None

        for a in actions:
            at = a if isinstance(a, ActionType) else a.action_type
            score = self._score_action(at, digest)
            if score > best_score:
                best_score = score
                best_action = at

        return Action(action_type=best_action or ActionType.NOOP)


# ═══════════════════════════════════════════════════════════════════════
#  3. Rule-Based Agent (Domain Heuristics)
# ═══════════════════════════════════════════════════════════════════════

class RuleBasedAgent(BaselineAgent):
    """
    Hand-crafted conditional strategy per environment.
    Represents what a "competent human" would do without any learning.

    Sales: stakeholder engagement -> POC -> advance deal, respond to shocks
    PM:    resolve risks -> advance workstreams -> manage morale
    HR/IT: fulfill instructions -> migrate cohorts -> respond to ransomware
    """

    name = "RuleBased"

    def __init__(self):
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def decide(self, obs: Observation, env: BaseEnvironment) -> Action:
        self._step += 1
        digest = env.state_digest()
        actions = env.action_space()
        valid_types = set(
            a if isinstance(a, ActionType) else a.action_type
            for a in actions
        )

        # Universal: always respond to shocks first
        if ActionType.RESPOND_SHOCK in valid_types:
            shock_tags = [t for e in obs.events for t in e.tags if "shock" in t.lower()]
            if shock_tags or digest.get("ransomware_active"):
                return Action(action_type=ActionType.RESPOND_SHOCK)

        env_id = env.env_id

        if env_id == EnvID.SALES:
            return self._sales_strategy(digest, valid_types, obs)
        elif env_id == EnvID.PM:
            return self._pm_strategy(digest, valid_types, obs)
        elif env_id == EnvID.HR_IT:
            return self._hrit_strategy(digest, valid_types, obs)

        return Action(action_type=ActionType.NOOP)

    def _sales_strategy(self, digest: Dict, valid: set, obs: Observation) -> Action:
        phase = digest.get("phase", 1)
        poc = digest.get("poc_score", 0)
        avg_rel = digest.get("avg_relationship", 50)

        # Always try to advance deal when possible (it's the key to progress)
        if ActionType.ADVANCE_DEAL in valid:
            return Action(action_type=ActionType.ADVANCE_DEAL)

        # Build POC score (critical for unlocking phase advances)
        if poc < 80 and ActionType.RUN_POC in valid:
            return Action(action_type=ActionType.RUN_POC)

        # Engage stakeholders periodically (but don't spam it)
        if self._step % 3 == 0 and ActionType.CONTACT_STAKEHOLDER in valid:
            return Action(action_type=ActionType.CONTACT_STAKEHOLDER)

        # Run POC as fallback
        if ActionType.RUN_POC in valid:
            return Action(action_type=ActionType.RUN_POC)

        # Contact stakeholders otherwise
        if ActionType.CONTACT_STAKEHOLDER in valid:
            return Action(action_type=ActionType.CONTACT_STAKEHOLDER)

        return Action(action_type=ActionType.REVIEW_STATUS)

    def _pm_strategy(self, digest: Dict, valid: set, obs: Observation) -> Action:
        morale = digest.get("team_morale", 80)
        risks_total = digest.get("risks_total", 0)
        risks_resolved = digest.get("risks_resolved", 0)
        unresolved = risks_total - risks_resolved

        # Critical morale
        if morale < 35 and ActionType.BOOST_MORALE in valid:
            return Action(action_type=ActionType.BOOST_MORALE)

        # Resolve risks when many are pending
        if unresolved > 5 and ActionType.RESOLVE_RISK in valid:
            return Action(action_type=ActionType.RESOLVE_RISK)

        # Advance workstreams
        if ActionType.ADVANCE_WORKSTREAM in valid:
            return Action(action_type=ActionType.ADVANCE_WORKSTREAM)

        # Periodic risk resolution
        if self._step % 5 == 0 and ActionType.RESOLVE_RISK in valid:
            return Action(action_type=ActionType.RESOLVE_RISK)

        # Periodic morale boost
        if morale < 60 and ActionType.BOOST_MORALE in valid:
            return Action(action_type=ActionType.BOOST_MORALE)

        if ActionType.ALLOCATE_BUDGET in valid:
            return Action(action_type=ActionType.ALLOCATE_BUDGET)

        return Action(action_type=ActionType.REVIEW_STATUS)

    def _hrit_strategy(self, digest: Dict, valid: set, obs: Observation) -> Action:
        migrated = digest.get("migrated_pct", 0)
        fulfilled = digest.get("fulfilled_instructions", 0)
        total_instr = digest.get("total_instructions", 300)
        sla = digest.get("sla_score", 100)

        # Ransomware response — top priority
        if digest.get("ransomware_active") and ActionType.RESPOND_SHOCK in valid:
            return Action(action_type=ActionType.RESPOND_SHOCK)

        # Interleave migration and instruction fulfillment
        # Migrate first (drives completion), fulfill periodically
        if self._step % 3 != 0 and migrated < 100 and ActionType.MIGRATE_COHORT in valid:
            return Action(action_type=ActionType.MIGRATE_COHORT)

        # Fulfill instructions every 3rd step
        if ActionType.FULFILL_INSTRUCTION in valid:
            return Action(action_type=ActionType.FULFILL_INSTRUCTION)

        # Continue migrating if can't fulfill
        if ActionType.MIGRATE_COHORT in valid:
            return Action(action_type=ActionType.MIGRATE_COHORT)

        return Action(action_type=ActionType.REVIEW_STATUS)


# ═══════════════════════════════════════════════════════════════════════
#  4. Planner-Only Agent (Ablation: planning without memory/skills/VF)
# ═══════════════════════════════════════════════════════════════════════

class PlannerOnlyAgent(BaselineAgent):
    """
    Uses the HierarchicalPlanner + beam search but with NO memory,
    NO skill library, and NO learned value function.
    This isolates the contribution of the planning component.
    """

    name = "PlannerOnly"

    def __init__(self, env_id: str, beam_width: int = 4, beam_depth: int = 3):
        self._env_id = env_id
        self._planner = HierarchicalPlanner(
            beam_width=beam_width,
            beam_depth=beam_depth,
        )
        env_id_map = {"sales": EnvID.SALES, "pm": EnvID.PM, "hr_it": EnvID.HR_IT}
        enum_id = env_id_map.get(env_id, EnvID.PM)
        self._planner.init_goal_tree(enum_id)

    def reset(self) -> None:
        env_id_map = {"sales": EnvID.SALES, "pm": EnvID.PM, "hr_it": EnvID.HR_IT}
        enum_id = env_id_map.get(self._env_id, EnvID.PM)
        self._planner = HierarchicalPlanner(beam_width=4, beam_depth=3)
        self._planner.init_goal_tree(enum_id)

    def decide(self, obs: Observation, env: BaseEnvironment) -> Action:
        actions = env.action_space()
        wrapped = []
        for a in actions:
            if isinstance(a, ActionType):
                wrapped.append(Action(action_type=a))
            else:
                wrapped.append(a)

        state_digest = env.state_digest()
        # No value function — uniform scoring
        value_fn = lambda sd, aname: 0.0
        best_action, _ = self._planner.plan(obs.step, wrapped, value_fn, state_digest)
        return best_action

    def post_step(self, obs: Observation, action: Action,
                  shaped_reward: float, state_digest: Dict[str, Any]) -> None:
        milestone_hit = any("milestone" in t.lower() for e in obs.events for t in e.tags)
        self._planner.update(obs.step, shaped_reward, milestone_hit)


# ═══════════════════════════════════════════════════════════════════════
#  Benchmark Suite
# ═══════════════════════════════════════════════════════════════════════

ENV_FACTORIES = {
    "sales": EnterpriseSalesPipeline,
    "pm": ProgramRescueEnvironment,
    "hr_it": ITTransformationEnv,
}

ENV_ID_MAP = {
    "sales": EnvID.SALES,
    "pm": EnvID.PM,
    "hr_it": EnvID.HR_IT,
}


def _run_baseline_episode(
    agent: BaselineAgent,
    env_id: str,
    seed: Optional[int] = None,
) -> Tuple[float, int, int, str]:
    """Run one episode with a baseline agent. Returns (reward, steps, phase, terminal)."""
    env = ENV_FACTORIES[env_id]()
    obs = env.reset(seed)
    reward_shaper = RewardShaper(env_id)

    total_reward = 0.0
    step_count = 0

    while not obs.done:
        action = agent.decide(obs, env)
        obs = env.step(action)
        step_count += 1

        digest = env.state_digest()
        events_dicts = [{"event_type": e.event_type, "tags": e.tags} for e in obs.events]
        shaped, _ = reward_shaper.shape(obs.reward, digest, obs.step, events_dicts)
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


def _run_v2_episode(
    env_id: str,
    seed: Optional[int] = None,
    beam_width: int = 4,
    beam_depth: int = 3,
) -> Tuple[float, int, int, str]:
    """Run one V2Agent episode for comparison."""
    env = ENV_FACTORIES[env_id]()
    obs = env.reset(seed)

    memory = MemorySystem()
    planner = HierarchicalPlanner(beam_width=beam_width, beam_depth=beam_depth)
    enum_id = ENV_ID_MAP.get(env_id, EnvID.PM)
    planner.init_goal_tree(enum_id)

    reward_shaper = RewardShaper(env_id)
    vf_registry = ValueFunctionRegistry()
    skill_library = SkillLibrary()

    ctx = AgentContext(
        memory=memory, planner=planner, skill_library=skill_library,
        vf_registry=vf_registry, env_id=env_id,
        beam_width=beam_width, beam_depth=beam_depth,
    )
    agent = V2Agent(ctx)

    if obs.events:
        for event in obs.events:
            memory.record_event(event)

    total_reward = 0.0
    step_count = 0

    while not obs.done:
        action = agent.decide(obs, env)
        obs = env.step(action)
        step_count += 1

        digest = env.state_digest()
        events_dicts = [{"event_type": e.event_type, "tags": e.tags} for e in obs.events]
        shaped, _ = reward_shaper.shape(obs.reward, digest, obs.step, events_dicts)
        total_reward += shaped

        agent.post_step(obs, action, shaped, digest)

        milestone_hit = any("milestone" in t.lower() for e in obs.events for t in e.tags)
        planner.update(obs.step, shaped, milestone_hit)

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


class BenchmarkSuite:
    """
    Runs all baseline agents + V2Agent on specified environments
    and produces a structured comparison report.

    Usage:
        suite = BenchmarkSuite(episodes=10, seed=42)
        report = suite.run(env_ids=["sales", "pm", "hr_it"])
        suite.print_report(report)
    """

    def __init__(
        self,
        episodes: int = 5,
        seed: int = 42,
        beam_width: int = 4,
        beam_depth: int = 3,
    ):
        self.episodes = episodes
        self.seed = seed
        self.beam_width = beam_width
        self.beam_depth = beam_depth

    def run(
        self,
        env_ids: Optional[List[str]] = None,
    ) -> Dict[str, List[BenchmarkResult]]:
        """Run benchmarks. Returns {env_id: [BenchmarkResult per agent]}."""
        if env_ids is None:
            env_ids = ["sales", "pm", "hr_it"]

        report: Dict[str, List[BenchmarkResult]] = {}

        for env_id in env_ids:
            print(f"\n{'='*70}")
            print(f"  BENCHMARK: {env_id.upper()}")
            print(f"{'='*70}")

            results = []

            # 1. Random Agent
            results.append(self._run_agent(
                RandomAgent(seed=self.seed), env_id, "Random",
            ))

            # 2. Greedy Agent
            results.append(self._run_agent(
                GreedyAgent(), env_id, "Greedy",
            ))

            # 3. Rule-Based Agent
            results.append(self._run_agent(
                RuleBasedAgent(), env_id, "RuleBased",
            ))

            # 4. Planner-Only Agent (ablation)
            results.append(self._run_agent(
                PlannerOnlyAgent(env_id, self.beam_width, self.beam_depth),
                env_id, "PlannerOnly",
            ))

            # 5. V2Agent (full system)
            br = BenchmarkResult(agent_name="V2Agent (full)", env_id=env_id, episodes=self.episodes)
            for i in range(self.episodes):
                ep_seed = self.seed + i
                t0 = time.time()
                reward, steps, phase, terminal = _run_v2_episode(
                    env_id, seed=ep_seed,
                    beam_width=self.beam_width, beam_depth=self.beam_depth,
                )
                elapsed = time.time() - t0
                br.rewards.append(reward)
                br.steps.append(steps)
                br.phases.append(phase)
                br.terminals.append(terminal)
                br.elapsed.append(elapsed)
            print(f"  V2Agent (full):    {self.episodes} episodes done")
            results.append(br)

            report[env_id] = results

        return report

    def _run_agent(
        self, agent: BaselineAgent, env_id: str, name: str,
    ) -> BenchmarkResult:
        br = BenchmarkResult(agent_name=name, env_id=env_id, episodes=self.episodes)

        for i in range(self.episodes):
            agent.reset()
            ep_seed = self.seed + i
            t0 = time.time()
            reward, steps, phase, terminal = _run_baseline_episode(
                agent, env_id, seed=ep_seed,
            )
            elapsed = time.time() - t0
            br.rewards.append(reward)
            br.steps.append(steps)
            br.phases.append(phase)
            br.terminals.append(terminal)
            br.elapsed.append(elapsed)

        print(f"  {name:<20s} {self.episodes} episodes done")
        return br

    @staticmethod
    def print_report(report: Dict[str, List[BenchmarkResult]]) -> None:
        """Print formatted comparison tables."""
        for env_id, results in report.items():
            print(f"\n{'='*90}")
            print(f"  BENCHMARK RESULTS: {env_id.upper()}")
            print(f"{'='*90}")
            print(f"  {'Agent':<20s} {'Mean Reward':>18s}  {'95% CI':>18s}  "
                  f"{'Steps':>7s}  {'Phase':>5s}  {'Success':>8s}")
            print(f"  {'-'*20} {'-'*18}  {'-'*18}  {'-'*7}  {'-'*5}  {'-'*8}")

            for br in results:
                print(br.summary_row())

            # Improvement over random
            random_reward = results[0].mean_reward
            v2_reward = results[-1].mean_reward
            if random_reward != 0:
                improvement = ((v2_reward - random_reward) / abs(random_reward)) * 100
            else:
                improvement = 0.0

            print(f"\n  V2Agent vs Random:  {improvement:>+.1f}% improvement")
            print(f"  V2Agent vs Greedy:  "
                  f"{v2_reward - results[1].mean_reward:>+.1f} reward delta")
            print(f"  V2Agent vs RuleBased: "
                  f"{v2_reward - results[2].mean_reward:>+.1f} reward delta")

    @staticmethod
    def to_dict(report: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Convert report to JSON-serializable dict."""
        out = {}
        for env_id, results in report.items():
            out[env_id] = []
            for br in results:
                ci_lo, ci_hi = br.ci_95()
                out[env_id].append({
                    "agent": br.agent_name,
                    "mean_reward": round(br.mean_reward, 2),
                    "std_reward": round(br.std_reward, 2),
                    "ci_95": [round(ci_lo, 2), round(ci_hi, 2)],
                    "mean_steps": round(br.mean_steps, 1),
                    "mean_phase": round(br.mean_phase, 1),
                    "success_rate": round(br.success_rate, 3),
                    "rewards": [round(r, 2) for r in br.rewards],
                })
        return out


# ═══════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

def run_benchmark(
    env_ids: Optional[List[str]] = None,
    episodes: int = 5,
    seed: int = 42,
) -> Dict[str, List[BenchmarkResult]]:
    """Run the full benchmark suite and print results."""
    suite = BenchmarkSuite(episodes=episodes, seed=seed)
    report = suite.run(env_ids)
    suite.print_report(report)
    return report
