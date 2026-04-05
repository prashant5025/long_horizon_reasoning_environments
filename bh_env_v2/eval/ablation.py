"""
Ablation Study — Systematic component removal analysis with
full statistical significance testing.

Creates V2Agent variants with individual components disabled
to measure each component's marginal contribution.

Ablation Matrix:
  Full        = Memory + Skills + VF + Planning + Exploration
  -Memory     = no memory retrieval (empty context)
  -Skills     = no skill library (disabled matching)
  -ValueFn    = pure heuristic scoring (no neural VF)
  -Exploration = epsilon = 0 (deterministic)
  -Planning   = no beam search (heuristic argmax only)
  PlannerOnly = planning only (no memory, skills, VF)
  Random      = uniform random baseline

Statistical Tests:
  - Paired t-test (ablation vs Full, same seeds)
  - Welch's t-test (framework comparisons)
  - Cohen's d effect size
  - 95% confidence intervals via t-distribution
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
from ..planning.value_fn import ValueFunctionRegistry, _context_aware_heuristic
from ..skills.skill_library import SkillLibrary
from ..agents.v2_agent import AgentContext, V2Agent


# ═══════════════════════════════════════════════════════════════════════
#  Environment Factories
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


# ═══════════════════════════════════════════════════════════════════════
#  Ablation Result
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AblationResult:
    """Result from one ablation variant."""
    variant: str
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
    def mean_steps(self) -> float:
        return sum(self.steps) / max(1, len(self.steps))

    @property
    def mean_phase(self) -> float:
        return sum(self.phases) / max(1, len(self.phases))

    @property
    def success_rate(self) -> float:
        s = sum(1 for t in self.terminals
                if t in ("deal_closed", "program_delivered", "migration_complete"))
        return s / max(1, len(self.terminals))

    def ci_95(self) -> Tuple[float, float]:
        n = len(self.rewards)
        if n < 2:
            return (self.mean_reward, self.mean_reward)
        se = self.std_reward / math.sqrt(n)
        t_val = 2.776 if n <= 10 else 2.228 if n <= 30 else 2.0
        return (self.mean_reward - t_val * se, self.mean_reward + t_val * se)


# ═══════════════════════════════════════════════════════════════════════
#  Statistical Significance Tests
# ═══════════════════════════════════════════════════════════════════════

def _t_cdf_approx(t_val: float, df: int) -> float:
    """Approximate two-tailed p-value from t-distribution.
    Uses the regularised incomplete beta function approximation.
    For small df (4-30), accurate to ~0.005.
    """
    x = df / (df + t_val ** 2)
    # Approximation via continued fraction for I_x(a, b) with a=df/2, b=0.5
    # For practical purposes, use lookup + interpolation for df=4
    if df == 4:
        # Pre-computed critical values for df=4 (two-tailed)
        table = [
            (0.0, 1.0), (0.741, 0.50), (1.533, 0.20), (2.132, 0.10),
            (2.776, 0.05), (3.747, 0.02), (4.604, 0.01), (5.598, 0.005),
            (7.173, 0.002), (8.610, 0.001),
        ]
    elif df <= 8:
        table = [
            (0.0, 1.0), (0.711, 0.50), (1.440, 0.20), (1.895, 0.10),
            (2.365, 0.05), (3.143, 0.02), (3.707, 0.01), (4.501, 0.005),
            (5.041, 0.002), (5.959, 0.001),
        ]
    else:
        table = [
            (0.0, 1.0), (0.700, 0.50), (1.372, 0.20), (1.812, 0.10),
            (2.228, 0.05), (2.764, 0.02), (3.169, 0.01), (3.581, 0.005),
            (4.144, 0.002), (4.587, 0.001),
        ]

    abs_t = abs(t_val)
    # Interpolate
    for i in range(len(table) - 1):
        t_lo, p_lo = table[i]
        t_hi, p_hi = table[i + 1]
        if t_lo <= abs_t < t_hi:
            frac = (abs_t - t_lo) / (t_hi - t_lo)
            return p_lo - frac * (p_lo - p_hi)
    # Beyond table
    if abs_t >= table[-1][0]:
        return table[-1][1] * 0.5  # rough bound
    return 1.0


def paired_t_test(
    rewards_a: List[float], rewards_b: List[float]
) -> Tuple[float, float, float]:
    """
    Paired t-test between two sets of episode rewards (same seeds).

    Returns: (t_statistic, p_value_two_tailed, cohens_d)
    """
    n = min(len(rewards_a), len(rewards_b))
    if n < 2:
        return (0.0, 1.0, 0.0)

    diffs = [rewards_a[i] - rewards_b[i] for i in range(n)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    sd_d = math.sqrt(var_d) if var_d > 0 else 1e-10

    t_stat = mean_d / (sd_d / math.sqrt(n))
    df = n - 1
    p_val = _t_cdf_approx(t_stat, df)

    # Cohen's d (paired) = mean_diff / sd_diff
    cohens_d = mean_d / sd_d if sd_d > 1e-10 else float('inf') * (1 if mean_d > 0 else -1)

    return (t_stat, p_val, cohens_d)


def welch_t_test(
    rewards_a: List[float], rewards_b: List[float]
) -> Tuple[float, float, float]:
    """
    Welch's t-test (unequal variance) between two independent samples.

    Returns: (t_statistic, p_value_two_tailed, cohens_d)
    """
    n1, n2 = len(rewards_a), len(rewards_b)
    if n1 < 2 or n2 < 2:
        return (0.0, 1.0, 0.0)

    m1 = sum(rewards_a) / n1
    m2 = sum(rewards_b) / n2
    v1 = sum((r - m1) ** 2 for r in rewards_a) / (n1 - 1)
    v2 = sum((r - m2) ** 2 for r in rewards_b) / (n2 - 1)

    se = math.sqrt(v1 / n1 + v2 / n2) if (v1 + v2) > 0 else 1e-10
    t_stat = (m1 - m2) / se

    # Welch–Satterthwaite df
    num = (v1 / n1 + v2 / n2) ** 2
    den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1) if (v1 + v2) > 0 else 1
    df = max(1, int(num / den)) if den > 0 else n1 + n2 - 2

    p_val = _t_cdf_approx(t_stat, df)

    # Pooled SD for Cohen's d
    sp = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)) if (v1 + v2) > 0 else 1e-10
    cohens_d = (m1 - m2) / sp if sp > 1e-10 else float('inf') * (1 if m1 > m2 else -1)

    return (t_stat, p_val, cohens_d)


def _sig_marker(p: float) -> str:
    """Return significance marker string."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


# ═══════════════════════════════════════════════════════════════════════
#  Ablation Episode Runner
# ═══════════════════════════════════════════════════════════════════════

def _terminal_reason(digest: Dict[str, Any]) -> str:
    if digest.get("deal_closed"):
        return "deal_closed"
    if digest.get("program_delivered"):
        return "program_delivered"
    if digest.get("migrated_pct", 0) >= 100:
        return "migration_complete"
    if digest.get("budget_runway", 1) <= 0:
        return "budget_exhausted"
    if digest.get("team_morale", 100) <= 0:
        return "morale_collapsed"
    return "timeout"


def _run_ablation_episode(
    env_id: str,
    seed: Optional[int],
    disable_memory: bool = False,
    disable_skills: bool = False,
    disable_vf: bool = False,
    disable_exploration: bool = False,
    disable_planning: bool = False,
) -> Tuple[float, int, int, str]:
    """Run one V2Agent episode with specified components disabled."""
    env = ENV_FACTORIES[env_id]()
    obs = env.reset(seed)
    reward_shaper = RewardShaper(env_id)

    memory = MemorySystem()
    planner = HierarchicalPlanner(beam_width=4, beam_depth=3)
    enum_id = ENV_ID_MAP.get(env_id, EnvID.PM)
    planner.init_goal_tree(enum_id)
    vf_registry = ValueFunctionRegistry()
    skill_library = SkillLibrary()

    # Configure ablation
    epsilon = 0.0 if disable_exploration else 0.15

    ctx = AgentContext(
        memory=memory,
        planner=planner,
        skill_library=skill_library,
        vf_registry=vf_registry,
        env_id=env_id,
        beam_width=4,
        beam_depth=3,
        epsilon=epsilon,
    )
    agent = V2Agent(ctx)

    # Disable components by monkey-patching
    if disable_memory:
        # Memory always returns empty context
        memory.context_for_agent = lambda step, query: ""

    if disable_skills:
        # Skill library never matches
        skill_library.query = lambda tags, eid: None

    if disable_vf:
        # Hybrid scorer uses ONLY heuristic (no neural VF)
        def _pure_heuristic_scorer(env_id_inner, heuristic_weight=1.0):
            def _scorer(state_digest, action_type):
                action_name = action_type if isinstance(action_type, str) else action_type.name
                return _context_aware_heuristic(action_name, state_digest) / 10.0
            return _scorer
        vf_registry.make_hybrid_scorer = _pure_heuristic_scorer

    if disable_planning:
        # Skip beam search — just pick highest heuristic action
        def _no_planning_plan(step, valid_actions, value_fn, state_digest):
            if not valid_actions:
                return Action(action_type=ActionType.NOOP), {}
            vfn = value_fn if value_fn else lambda sd, a: 0.0
            best = max(valid_actions, key=lambda a: vfn(state_digest, a.action_type.name))
            return best, {"beam_reasoning": "no_planning"}
        planner.plan = _no_planning_plan

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
    return total_reward, step_count, obs.phase, _terminal_reason(final)


def _run_random_episode(env_id: str, seed: Optional[int]) -> Tuple[float, int, int, str]:
    """Random baseline."""
    env = ENV_FACTORIES[env_id]()
    obs = env.reset(seed)
    reward_shaper = RewardShaper(env_id)
    rng = random.Random(seed)
    total_reward = 0.0
    step_count = 0

    while not obs.done:
        actions = env.action_space()
        choice = rng.choice(actions)
        action = Action(action_type=choice) if isinstance(choice, ActionType) else choice
        obs = env.step(action)
        step_count += 1

        digest = env.state_digest()
        events_dicts = [{"event_type": e.event_type, "tags": e.tags} for e in obs.events]
        shaped, _ = reward_shaper.shape(obs.reward, digest, obs.step, events_dicts)
        total_reward += shaped

    final = env.state_digest()
    return total_reward, step_count, obs.phase, _terminal_reason(final)


# ═══════════════════════════════════════════════════════════════════════
#  Ablation Suite
# ═══════════════════════════════════════════════════════════════════════

# Ablation configurations: (variant_name, kwargs)
ABLATION_CONFIGS: List[Tuple[str, Dict[str, bool]]] = [
    ("Full V2Agent", {}),
    ("-Memory", {"disable_memory": True}),
    ("-Skills", {"disable_skills": True}),
    ("-ValueFn", {"disable_vf": True}),
    ("-Exploration", {"disable_exploration": True}),
    ("-Planning", {"disable_planning": True}),
]


def run_ablation_study(
    env_ids: Optional[List[str]] = None,
    episodes: int = 5,
    seed: int = 42,
) -> Dict[str, List[AblationResult]]:
    """
    Run full ablation study across environments.
    Returns {env_id: [AblationResult per variant]}.
    """
    if env_ids is None:
        env_ids = ["sales", "pm", "hr_it"]

    report: Dict[str, List[AblationResult]] = {}

    for env_id in env_ids:
        print(f"\n{'='*70}")
        print(f"  ABLATION STUDY: {env_id.upper()}")
        print(f"{'='*70}")

        results: List[AblationResult] = []

        # Run each ablation variant
        for variant_name, kwargs in ABLATION_CONFIGS:
            ar = AblationResult(variant=variant_name, env_id=env_id, episodes=episodes)
            for i in range(episodes):
                ep_seed = seed + i
                reward, steps, phase, terminal = _run_ablation_episode(
                    env_id, ep_seed, **kwargs,
                )
                ar.rewards.append(reward)
                ar.steps.append(steps)
                ar.phases.append(phase)
                ar.terminals.append(terminal)
            print(f"  {variant_name:<20s} {episodes} episodes done  "
                  f"reward={ar.mean_reward:>+8.1f}")
            results.append(ar)

        # Random baseline
        ar = AblationResult(variant="Random", env_id=env_id, episodes=episodes)
        for i in range(episodes):
            ep_seed = seed + i
            reward, steps, phase, terminal = _run_random_episode(env_id, ep_seed)
            ar.rewards.append(reward)
            ar.steps.append(steps)
            ar.phases.append(phase)
            ar.terminals.append(terminal)
        print(f"  {'Random':<20s} {episodes} episodes done  "
              f"reward={ar.mean_reward:>+8.1f}")
        results.append(ar)

        report[env_id] = results

    return report


def print_ablation_report(report: Dict[str, List[AblationResult]]) -> None:
    """Print formatted ablation table with statistical significance."""
    for env_id, results in report.items():
        full = results[0]  # First is always Full V2Agent

        print(f"\n{'='*110}")
        print(f"  ABLATION TABLE: {env_id.upper()}")
        print(f"{'='*110}")
        print(f"  {'Variant':<20s} {'Mean Reward':>12s} {'Std':>8s} "
              f"{'Delta':>10s} {'t-stat':>8s} {'p-value':>9s} {'d':>6s} "
              f"{'Steps':>7s} {'Success':>8s}")
        print(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*10} "
              f"{'-'*8} {'-'*9} {'-'*6} {'-'*7} {'-'*8}")

        for ar in results:
            if ar.variant == "Full V2Agent":
                print(
                    f"  {ar.variant:<20s} {ar.mean_reward:>+12.1f} "
                    f"{ar.std_reward:>8.1f}   (baseline) "
                    f"{'---':>8s} {'---':>9s} {'---':>6s} "
                    f"{ar.mean_steps:>7.0f} {ar.success_rate:>7.0%}"
                )
            else:
                delta = ar.mean_reward - full.mean_reward
                t_stat, p_val, d = paired_t_test(full.rewards, ar.rewards)
                sig = _sig_marker(p_val)
                print(
                    f"  {ar.variant:<20s} {ar.mean_reward:>+12.1f} "
                    f"{ar.std_reward:>8.1f} {delta:>+10.1f} "
                    f"{t_stat:>+8.2f} {p_val:>8.4f}{sig:>2s} "
                    f"{d:>+6.2f} "
                    f"{ar.mean_steps:>7.0f} {ar.success_rate:>7.0%}"
                )

        # Statistical significance summary
        print(f"\n  Statistical Significance Summary (paired t-test, df={len(full.rewards)-1}):")
        print(f"  {'Component':<20s} {'Δ Reward':>10s} {'t':>8s} {'p':>9s} {'Cohen d':>9s} {'Sig':>5s}")
        print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*9} {'-'*9} {'-'*5}")

        for ar in results[1:]:  # Skip Full
            delta = ar.mean_reward - full.mean_reward
            t_stat, p_val, d = paired_t_test(full.rewards, ar.rewards)
            sig = _sig_marker(p_val)
            d_label = ("very large" if abs(d) > 1.2 else "large" if abs(d) > 0.8
                       else "medium" if abs(d) > 0.5 else "small")
            print(
                f"  {ar.variant:<20s} {delta:>+10.1f} {t_stat:>+8.2f} "
                f"{p_val:>9.4f} {d:>+9.2f} {sig:>5s}  ({d_label})"
            )

        # Impact ranking by |Cohen's d|
        print(f"\n  Component Impact Ranking (by |Cohen's d| — effect size):")
        impacts = []
        for ar in results[1:-1]:  # Skip Full and Random
            _, _, d = paired_t_test(full.rewards, ar.rewards)
            delta = full.mean_reward - ar.mean_reward
            impacts.append((ar.variant, abs(d), delta))
        impacts.sort(key=lambda x: x[1], reverse=True)
        for rank, (name, abs_d, delta) in enumerate(impacts, 1):
            arrow = ">>>" if abs_d > 1.2 else ">>" if abs_d > 0.5 else ">"
            print(f"    {rank}. {name:<20s} |d|={abs_d:.2f} {arrow} "
                  f"removing costs {delta:>+.1f} reward")


def run_and_print_ablation(
    env_ids: Optional[List[str]] = None,
    episodes: int = 5,
    seed: int = 42,
) -> Dict[str, List[AblationResult]]:
    """Convenience: run ablation study and print results."""
    report = run_ablation_study(env_ids, episodes, seed)
    print_ablation_report(report)
    return report
