"""
V2Harness — Episode orchestration, cross-episode learning, and CLI.

Owns the persistent components (VF registry, skill library, replay buffer,
skill extractor) and wires them into per-episode V2Agent instances.

Public API:
  harness.run_episode(env_id, seed, verbose)  -> EpisodeResult
  harness.run_n(env_id, n, seed)              -> List[EpisodeResult]
  harness.run_learning_curve(env_id, n)        -> None  (prints ASCII plot)
  harness.print_aggregate(results)             -> None
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..engine.types import Action, ActionType, EnvID, Event, Observation
from ..engine.environments.base import BaseEnvironment
from ..engine.environments.sales import EnterpriseSalesPipeline
from ..engine.environments.pm import ProgramRescueEnvironment
from ..engine.environments.hr_it import ITTransformationEnv
from ..engine.reward_shaping import RewardShaper
from ..memory.memory_system import MemorySystem
from ..planning.planner import HierarchicalPlanner
from ..planning.value_fn import ValueFunctionRegistry
from ..skills.skill_library import ExperienceReplay, SkillExtractor, SkillLibrary, Transition
from ..agents.v2_agent import AgentContext, V2Agent


# ═════════════════════════════════════════════════════════════════════════════
#  Episode Result
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class EpisodeResult:
    """Returned by run_episode with full episode statistics."""
    env_id: str
    seed: Optional[int]
    total_steps: int
    total_reward: float
    shaped_reward: float
    terminal_reason: str
    phase_reached: int
    goal_progress: Dict[str, float] = field(default_factory=dict)
    vf_stats: Optional[Dict[str, Any]] = None
    skill_stats: Optional[Dict[str, Any]] = None
    elapsed_seconds: float = 0.0
    reward_history: List[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"  Episode Result  [{self.env_id}]  seed={self.seed}",
            f"{'=' * 60}",
            f"  Steps:          {self.total_steps}",
            f"  Total Reward:   {self.total_reward:+.2f}",
            f"  Shaped Reward:  {self.shaped_reward:+.2f}",
            f"  Phase Reached:  {self.phase_reached}",
            f"  Terminal:       {self.terminal_reason}",
            f"  Elapsed:        {self.elapsed_seconds:.2f}s",
        ]
        if self.goal_progress:
            lines.append("  Goal Progress:")
            for name, prog in self.goal_progress.items():
                bar = _progress_bar(prog)
                lines.append(f"    {name}: {bar}")
        if self.vf_stats:
            lines.append("  Value Function:")
            lines.append(f"    Steps: {self.vf_stats.get('steps', 0)}")
            lines.append(f"    Offline: {self.vf_stats.get('offline_steps', 0)}")
            last = self.vf_stats.get('last_loss')
            if last is not None:
                lines.append(f"    Last Loss: {last:.6f}")
            avg = self.vf_stats.get('avg_loss_last_50')
            if avg is not None:
                lines.append(f"    Avg Loss (50): {avg:.6f}")
        if self.skill_stats:
            lines.append(f"  Skills: {self.skill_stats.get('total_skills', 0)} learned")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


def _progress_bar(fraction: float, width: int = 20) -> str:
    filled = int(round(fraction * width))
    empty = width - filled
    pct = int(round(fraction * 100))
    return f"[{'#' * filled}{'-' * empty}] {pct}%"


# ═════════════════════════════════════════════════════════════════════════════
#  Environment Factory
# ═════════════════════════════════════════════════════════════════════════════

_ENV_FACTORIES = {
    "sales": EnterpriseSalesPipeline,
    "pm": ProgramRescueEnvironment,
    "hr_it": ITTransformationEnv,
}

_ENV_ID_MAP = {
    "sales": EnvID.SALES,
    "pm": EnvID.PM,
    "hr_it": EnvID.HR_IT,
}


def _make_env(env_id_str: str) -> BaseEnvironment:
    factory = _ENV_FACTORIES.get(env_id_str)
    if factory is None:
        raise ValueError(
            f"Unknown env_id '{env_id_str}'. Choose from: {list(_ENV_FACTORIES.keys())}"
        )
    return factory()


# ═════════════════════════════════════════════════════════════════════════════
#  V2Harness
# ═════════════════════════════════════════════════════════════════════════════

class V2Harness:
    """
    Owns cross-episode persistent state and runs the V2Agent decision loop.

    Persistent components (survive across episodes):
      - ValueFunctionRegistry  (learned value functions per env)
      - SkillLibrary           (discovered skills)
      - ExperienceReplay       (prioritised replay buffer)
      - SkillExtractor         (pattern frequency tracker)

    Per-episode components (fresh each episode):
      - MemorySystem
      - HierarchicalPlanner
      - V2Agent
      - RewardShaper
    """

    def __init__(
        self,
        beam_width: int = 4,
        beam_depth: int = 3,
        replay_capacity: int = 100_000,
    ) -> None:
        self.beam_width = beam_width
        self.beam_depth = beam_depth

        # Cross-episode persistent components
        self._vf_registry = ValueFunctionRegistry()
        self._skill_library = SkillLibrary()
        self._replay_buffer = ExperienceReplay(capacity=replay_capacity)
        self._skill_extractor = SkillExtractor()
        self._episode_count: int = 0

    # ── single episode ────────────────────────────────────────────────────

    def run_episode(
        self,
        env_id: str,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> EpisodeResult:
        """Run a single episode and return the result."""
        t0 = time.time()
        self._episode_count += 1
        episode_num = self._episode_count

        # Create environment
        env = _make_env(env_id)
        obs = env.reset(seed)

        # Per-episode components
        memory = MemorySystem()
        planner = HierarchicalPlanner(
            beam_width=self.beam_width,
            beam_depth=self.beam_depth,
        )
        enum_id = _ENV_ID_MAP.get(env_id, EnvID.PM)
        planner.init_goal_tree(enum_id)

        reward_shaper = RewardShaper(env_id)

        # Agent context (links to persistent + per-episode components)
        ctx = AgentContext(
            memory=memory,
            planner=planner,
            skill_library=self._skill_library,
            vf_registry=self._vf_registry,
            env_id=env_id,
            beam_width=self.beam_width,
            beam_depth=self.beam_depth,
        )
        agent = V2Agent(ctx)

        # Record initial observation
        if obs.events:
            for event in obs.events:
                memory.record_event(event)

        # ── Episode loop ──────────────────────────────────────────────
        total_reward = 0.0
        shaped_total = 0.0
        reward_history: List[float] = []
        step_counter = 0
        prev_state_digest: Optional[Dict[str, Any]] = None
        prev_action_type: Optional[str] = None

        while not obs.done:
            state_digest = env.state_digest()

            # Agent decides
            action = agent.decide(obs, env)

            # Environment step
            obs = env.step(action)
            step_counter += 1

            # Reward shaping
            new_state_digest = env.state_digest()
            events_as_dicts = [
                {"event_type": e.event_type, "tags": e.tags}
                for e in obs.events
            ]
            shaped_reward, breakdown = reward_shaper.shape(
                obs.reward, new_state_digest, obs.step, events_as_dicts,
            )

            total_reward += obs.reward
            shaped_total += shaped_reward
            reward_history.append(shaped_reward)

            # Agent post-step bookkeeping
            agent.post_step(obs, action, shaped_reward, new_state_digest)

            # ── Harness per-step actions (Section 5.1) ────────────
            # 1. VF online update
            next_action_type = action.action_type.name  # approximate next action
            if prev_state_digest is not None and prev_action_type is not None:
                self._vf_registry.online_update(
                    env_id,
                    prev_state_digest,
                    prev_action_type,
                    shaped_reward,
                    new_state_digest,
                    next_action_type,
                    obs.done,
                )

            # 2. Offline update trigger (every 25 steps)
            if step_counter % 25 == 0 and self._replay_buffer.size >= 64:
                batch = self._replay_buffer.sample(64)
                transitions_dicts = self._transitions_to_dicts(batch)
                self._vf_registry.offline_update(env_id, transitions_dicts)

            # 3. Planner reward signal
            milestone_hit = any(
                "milestone" in t.lower()
                for e in obs.events
                for t in e.tags
            )
            planner.update(obs.step, shaped_reward, milestone_hit)

            prev_state_digest = new_state_digest
            prev_action_type = action.action_type.name

            if verbose and step_counter % 50 == 0:
                print(
                    f"  [step {obs.step:>4}] phase={obs.phase} "
                    f"reward={shaped_reward:+.1f} total={shaped_total:+.1f}"
                )

        # ── End-of-episode operations (Section 5.2) ───────────────────
        trajectory = agent.trajectory

        # Store trajectory in experience replay
        for tr in trajectory:
            self._replay_buffer.add(tr)

        # Skill extraction
        new_skills = self._skill_extractor.extract_from_trajectory(
            trajectory, env_id, episode_num,
        )
        for skill in new_skills:
            self._skill_library.register(skill)

        # Update semantic memory from final state
        final_digest = env.state_digest()
        agent.update_semantic_memory(obs.step, final_digest)

        # Terminal reason
        terminal_reason = "timeout"
        if final_digest.get("deal_closed"):
            terminal_reason = "deal_closed"
        elif final_digest.get("program_delivered"):
            terminal_reason = "program_delivered"
        elif final_digest.get("migrated_pct", 0) >= 100:
            terminal_reason = "migration_complete"
        elif final_digest.get("budget_runway", 1) <= 0:
            terminal_reason = "budget_exhausted"
        elif final_digest.get("team_morale", 100) <= 0:
            terminal_reason = "morale_collapsed"

        # Collect goal progress from planner
        root_nodes = [
            n for n in planner.goal_tree._nodes.values()
            if n.parent_id is None
        ]
        goal_progress = {n.name: n.progress for n in root_nodes}

        elapsed = time.time() - t0

        return EpisodeResult(
            env_id=env_id,
            seed=seed,
            total_steps=step_counter,
            total_reward=total_reward,
            shaped_reward=shaped_total,
            terminal_reason=terminal_reason,
            phase_reached=obs.phase,
            goal_progress=goal_progress,
            vf_stats=self._vf_registry.stats(env_id),
            skill_stats=self._skill_library.stats(),
            elapsed_seconds=elapsed,
            reward_history=reward_history,
        )

    # ── multi-episode ─────────────────────────────────────────────────────

    def run_n(
        self,
        env_id: str,
        n: int = 5,
        seed: Optional[int] = None,
    ) -> List[EpisodeResult]:
        """Run *n* episodes with optional seed offset."""
        results: List[EpisodeResult] = []
        for i in range(n):
            ep_seed = (seed + i) if seed is not None else None
            result = self.run_episode(env_id, seed=ep_seed, verbose=False)
            print(
                f"  Episode {i + 1}/{n}: steps={result.total_steps} "
                f"reward={result.shaped_reward:+.1f} "
                f"phase={result.phase_reached} "
                f"({result.terminal_reason}) "
                f"[{result.elapsed_seconds:.1f}s]"
            )
            results.append(result)
        return results

    # ── learning curve ────────────────────────────────────────────────────

    def run_learning_curve(
        self,
        env_id: str,
        n: int = 10,
        seed: Optional[int] = 42,
    ) -> None:
        """Run *n* episodes and print an ASCII learning curve."""
        print(f"Learning curve: {env_id} x {n} episodes")
        print("-" * 60)

        results = self.run_n(env_id, n, seed)

        # Print reward curve
        rewards = [r.shaped_reward for r in results]
        print(f"\n{'=' * 60}")
        print(f"  Reward Curve [{env_id}]")
        print(f"{'=' * 60}")
        self._print_ascii_curve(rewards, label="Shaped Reward")

        # Print VF loss curve
        print(f"\nValue Function Loss:")
        print(self._vf_registry.plot_ascii(env_id))

        # Print skill stats
        stats = self._skill_library.stats()
        print(f"\nSkills learned: {stats['total_skills']}")
        if stats.get("most_used"):
            mu = stats["most_used"]
            print(f"  Most used: {mu['action_sequence']} (x{mu['usage_count']})")
        print(f"Replay buffer: {self._replay_buffer.size} transitions")

    # ── aggregate printing ────────────────────────────────────────────────

    def print_aggregate(self, results: List[EpisodeResult]) -> None:
        """Print aggregate statistics for a list of episode results."""
        if not results:
            print("No results to aggregate.")
            return

        n = len(results)
        rewards = [r.shaped_reward for r in results]
        steps = [r.total_steps for r in results]
        phases = [r.phase_reached for r in results]

        avg_reward = sum(rewards) / n
        avg_steps = sum(steps) / n
        avg_phase = sum(phases) / n
        max_reward = max(rewards)
        min_reward = min(rewards)

        terminals = {}
        for r in results:
            terminals[r.terminal_reason] = terminals.get(r.terminal_reason, 0) + 1

        env_id = results[0].env_id

        print(f"\n{'=' * 60}")
        print(f"  Aggregate  [{env_id}]  n={n}")
        print(f"{'=' * 60}")
        print(f"  Avg Reward:  {avg_reward:+.2f}")
        print(f"  Max Reward:  {max_reward:+.2f}")
        print(f"  Min Reward:  {min_reward:+.2f}")
        print(f"  Avg Steps:   {avg_steps:.0f}")
        print(f"  Avg Phase:   {avg_phase:.1f}")
        print(f"  Terminals:   {terminals}")
        print(f"  Skills:      {self._skill_library.stats()['total_skills']}")
        print(f"  Replay:      {self._replay_buffer.size} transitions")
        print(f"{'=' * 60}")

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _transitions_to_dicts(transitions: List[Transition]) -> List[Dict[str, Any]]:
        """Convert Transition objects to dicts for offline_update."""
        dicts: List[Dict[str, Any]] = []
        for i, tr in enumerate(transitions):
            # Use next transition's state as next_state; if last, mark terminal
            if i + 1 < len(transitions):
                nsd = transitions[i + 1].state_digest
                nat = transitions[i + 1].action_type
                done = tr.done
            else:
                nsd = tr.state_digest
                nat = tr.action_type
                done = True
            dicts.append({
                "state_digest": tr.state_digest,
                "action_type": tr.action_type,
                "reward": tr.shaped_reward,
                "next_state_digest": nsd,
                "next_action_type": nat,
                "done": done,
            })
        return dicts

    @staticmethod
    def _print_ascii_curve(values: List[float], label: str = "Value") -> None:
        """Print a simple ASCII bar chart of values."""
        if not values:
            return
        max_val = max(abs(v) for v in values)
        if max_val == 0:
            max_val = 1.0

        print(f"  {label}:")
        for i, v in enumerate(values):
            bar_len = int(abs(v) / max_val * 40)
            if v >= 0:
                bar = "+" * bar_len
                print(f"  ep{i + 1:>3}: {v:>+10.1f} |{bar}")
            else:
                bar = "-" * bar_len
                print(f"  ep{i + 1:>3}: {v:>+10.1f} |{bar}")


# ═════════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """CLI: python -m bh_env_v2.eval.v2_harness <env_id> [--episodes N] [--seed S] [--verbose]"""
    import argparse

    parser = argparse.ArgumentParser(
        description="BusinessHorizonENV v2 — run agent episodes",
    )
    parser.add_argument(
        "env_id",
        choices=["sales", "pm", "hr_it"],
        help="Environment to run",
    )
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=1,
        help="Number of episodes (default: 1)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-step progress",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=4,
        help="Beam search width (default: 4)",
    )
    parser.add_argument(
        "--beam-depth",
        type=int,
        default=3,
        help="Beam search depth (default: 3)",
    )
    parser.add_argument(
        "--learning-curve",
        action="store_true",
        help="Run learning curve mode with ASCII plots",
    )

    # ── Upgrade flags ──
    parser.add_argument(
        "--multi-agent",
        action="store_true",
        help="Run multi-agent coordination mode (3 departments)",
    )
    parser.add_argument(
        "--llm",
        choices=["claude", "openai"],
        default=None,
        help="Use a real LLM in the planning loop",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override LLM model name",
    )
    parser.add_argument(
        "--llm-every-n",
        type=int,
        default=5,
        help="Use LLM every N steps (default: 5, heuristic otherwise)",
    )
    parser.add_argument(
        "--neural-memory",
        action="store_true",
        help="Use neural (hybrid) memory retrieval",
    )
    parser.add_argument(
        "--validate-scale",
        type=int,
        default=None,
        metavar="SCALE",
        help="Run scale validation at SCALE factor (e.g., 10)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run baseline benchmark (Random, Greedy, RuleBased, PlannerOnly, V2Agent)",
    )
    parser.add_argument(
        "--llm-benchmark",
        action="store_true",
        help="Run LLM benchmark across available models",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names for --llm-benchmark (e.g., claude-sonnet,gpt-4o)",
    )
    parser.add_argument(
        "--open-model-url",
        type=str,
        default=None,
        help="Base URL for open model API (default: http://localhost:11434/v1)",
    )

    args = parser.parse_args()

    # ── Baseline benchmark mode ──
    if args.benchmark:
        from .baselines import run_benchmark
        run_benchmark(
            env_ids=[args.env_id],
            episodes=args.episodes if args.episodes > 1 else 5,
            seed=args.seed,
        )
        return

    # ── LLM benchmark mode ──
    if args.llm_benchmark:
        from .llm_benchmark import run_llm_benchmark
        model_names = args.models.split(",") if args.models else None
        run_llm_benchmark(
            env_id=args.env_id,
            model_names=model_names,
            episodes=args.episodes if args.episodes > 1 else 3,
            seed=args.seed,
            llm_every_n=args.llm_every_n,
            open_model_url=args.open_model_url,
            verbose=args.verbose,
        )
        return

    # ── Multi-agent mode ──
    if args.multi_agent:
        from ..agents.multi_agent import (
            MultiAgentCoordinator, RewardAttribution,
        )
        from ..engine.environments.pm import ProgramRescueEnvironment
        from ..engine.environments.scaled import MultiDepartmentEnvironment

        vf_registry = ValueFunctionRegistry()
        skill_library = SkillLibrary()
        coordinator = MultiAgentCoordinator(
            vf_registry=vf_registry,
            skill_library=skill_library,
            beam_width=args.beam_width,
            beam_depth=args.beam_depth,
        )

        env = MultiDepartmentEnvironment(
            env_factories=[ProgramRescueEnvironment] * 3,
        )

        print(f"Running multi-agent episode on {args.env_id}...")
        result = coordinator.run_episode(env, seed=args.seed, verbose=args.verbose)
        print(result.summary())
        return

    # ── Scale validation mode ──
    if args.validate_scale is not None:
        from .scale_validator import ScaleValidationHarness

        validator = ScaleValidationHarness(
            beam_width=args.beam_width,
            beam_depth=args.beam_depth,
        )
        result = validator.validate(
            args.env_id,
            scale=args.validate_scale,
            seed=args.seed,
            verbose=args.verbose,
        )
        print(result.summary())
        return

    # ── Standard harness ──
    harness = V2Harness(
        beam_width=args.beam_width,
        beam_depth=args.beam_depth,
    )

    # ── LLM mode — run with LLM agent ──
    if args.llm:
        from ..agents.llm_agent import LLMAgent, LLMAgentConfig
        from ..engine.reward_shaping import RewardShaper

        config = LLMAgentConfig(
            llm_provider=args.llm,
            model=args.llm_model,
            beam_depth=1,
            beam_width=2,
            llm_every_n_steps=args.llm_every_n,
        )

        env = _make_env(args.env_id)
        obs = env.reset(args.seed)

        if args.neural_memory:
            from ..memory.neural_retrieval import NeuralMemorySystem
            memory = NeuralMemorySystem()
        else:
            memory = MemorySystem()

        planner = HierarchicalPlanner(beam_width=2, beam_depth=1)
        enum_id = _ENV_ID_MAP.get(args.env_id, EnvID.PM)
        planner.init_goal_tree(enum_id)

        ctx = AgentContext(
            memory=memory,
            planner=planner,
            skill_library=harness._skill_library,
            vf_registry=harness._vf_registry,
            env_id=args.env_id,
        )
        agent = LLMAgent(ctx, config, args.env_id)
        reward_shaper = RewardShaper(args.env_id)

        total_reward = 0.0
        step = 0
        while not obs.done:
            action = agent.decide(obs, env)
            obs = env.step(action)
            step += 1
            digest = env.state_digest()
            events_dicts = [{"event_type": e.event_type, "tags": e.tags} for e in obs.events]
            shaped, _ = reward_shaper.shape(obs.reward, digest, obs.step, events_dicts)
            total_reward += shaped
            agent.post_step(obs, action, shaped, digest)
            if args.verbose and step % 20 == 0:
                print(f"  [step {obs.step:>4}] reward={total_reward:+.1f}")

        llm_stats = agent.stats()
        print(f"\nLLM Episode [{args.env_id}]: {step} steps, reward={total_reward:+.1f}")
        print(f"  LLM calls: {llm_stats['llm_decisions']}, "
              f"heuristic: {llm_stats['heuristic_decisions']}")
        print(f"  Tokens: {llm_stats['total_tokens']}, "
              f"avg latency: {llm_stats['avg_latency_ms']:.0f}ms")
        return

    # ── Standard modes ──
    if args.learning_curve:
        harness.run_learning_curve(args.env_id, n=args.episodes, seed=args.seed)
    elif args.episodes == 1:
        result = harness.run_episode(
            args.env_id,
            seed=args.seed,
            verbose=args.verbose,
        )
        print(result.summary())
    else:
        results = harness.run_n(args.env_id, n=args.episodes, seed=args.seed)
        harness.print_aggregate(results)


if __name__ == "__main__":
    main()
