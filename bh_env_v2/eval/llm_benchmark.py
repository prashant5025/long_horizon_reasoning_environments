"""
LLM Benchmark Runner — Structured Experiments Across Model Families
====================================================================
Runs BusinessHorizonENV with real LLMs and produces comparison tables.

Supported model families:
  - Anthropic:   claude-sonnet-4-20250514, claude-haiku-4-5-20251001
  - OpenAI:      gpt-4o, gpt-4o-mini, gpt-3.5-turbo
  - Open Models: Any model served via OpenAI-compatible API
                 (Ollama, vLLM, LM Studio, Together AI, etc.)

Usage:
  python -m bh_env_v2 sales --llm-benchmark
  python -m bh_env_v2 sales --llm-benchmark --models claude-sonnet-4-20250514,gpt-4o,gpt-4o-mini
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..engine.types import Action, ActionType, EnvID, Observation
from ..engine.environments.sales import EnterpriseSalesPipeline
from ..engine.environments.pm import ProgramRescueEnvironment
from ..engine.environments.hr_it import ITTransformationEnv
from ..engine.reward_shaping import RewardShaper
from ..memory.memory_system import MemorySystem
from ..planning.planner import HierarchicalPlanner
from ..planning.value_fn import ValueFunctionRegistry
from ..skills.skill_library import SkillLibrary
from ..agents.v2_agent import AgentContext, V2Agent
from ..agents.llm_agent import (
    ClaudeLLM, OpenAILLM, LLMAgent, LLMAgentConfig, LLMInterface,
)


# ═══════════════════════════════════════════════════════════════════════
#  Model Registry
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ModelSpec:
    """Specification for an LLM model to benchmark."""
    name: str                            # Display name
    provider: str                        # "claude" | "openai" | "open"
    model_id: str                        # API model identifier
    api_key_env: str = ""                # Env var name for API key
    base_url: Optional[str] = None       # For OpenAI-compatible endpoints
    temperature: float = 0.3
    max_tokens: int = 256
    cost_per_1k_input: float = 0.0       # USD per 1K input tokens
    cost_per_1k_output: float = 0.0      # USD per 1K output tokens
    tier: str = "unknown"                # "frontier" | "mid" | "small" | "open"


# Pre-configured model registry
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # ── Anthropic ──
    "claude-sonnet": ModelSpec(
        name="Claude Sonnet 4",
        provider="claude",
        model_id="claude-sonnet-4-20250514",
        api_key_env="ANTHROPIC_API_KEY",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        tier="frontier",
    ),
    "claude-haiku": ModelSpec(
        name="Claude Haiku 4.5",
        provider="claude",
        model_id="claude-haiku-4-5-20251001",
        api_key_env="ANTHROPIC_API_KEY",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.005,
        tier="small",
    ),

    # ── OpenAI ──
    "gpt-4o": ModelSpec(
        name="GPT-4o",
        provider="openai",
        model_id="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01,
        tier="frontier",
    ),
    "gpt-4o-mini": ModelSpec(
        name="GPT-4o Mini",
        provider="openai",
        model_id="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        tier="mid",
    ),
    "gpt-3.5-turbo": ModelSpec(
        name="GPT-3.5 Turbo",
        provider="openai",
        model_id="gpt-3.5-turbo",
        api_key_env="OPENAI_API_KEY",
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        tier="small",
    ),

    # ── Open Models (via OpenAI-compatible API) ──
    "llama-3.1-8b": ModelSpec(
        name="Llama 3.1 8B",
        provider="openai",
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        api_key_env="OPEN_MODEL_API_KEY",
        base_url=None,  # Set via OPEN_MODEL_BASE_URL or --open-model-url
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        tier="open",
    ),
    "llama-3.1-70b": ModelSpec(
        name="Llama 3.1 70B",
        provider="openai",
        model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        api_key_env="OPEN_MODEL_API_KEY",
        base_url=None,
        tier="open",
    ),
    "mistral-7b": ModelSpec(
        name="Mistral 7B",
        provider="openai",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        api_key_env="OPEN_MODEL_API_KEY",
        base_url=None,
        tier="open",
    ),
    "qwen-2.5-7b": ModelSpec(
        name="Qwen 2.5 7B",
        provider="openai",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        api_key_env="OPEN_MODEL_API_KEY",
        base_url=None,
        tier="open",
    ),
    "phi-3-mini": ModelSpec(
        name="Phi-3 Mini 3.8B",
        provider="openai",
        model_id="microsoft/Phi-3-mini-4k-instruct",
        api_key_env="OPEN_MODEL_API_KEY",
        base_url=None,
        tier="open",
    ),
}


# ═══════════════════════════════════════════════════════════════════════
#  LLM Benchmark Result
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LLMBenchmarkResult:
    """Result from benchmarking a single model on a single environment."""
    model_name: str
    model_id: str
    tier: str
    env_id: str
    episodes: int
    rewards: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    phases: List[int] = field(default_factory=list)
    terminals: List[str] = field(default_factory=list)
    total_tokens: List[int] = field(default_factory=list)
    total_latency_ms: List[float] = field(default_factory=list)
    llm_decisions: List[int] = field(default_factory=list)
    parse_failures: List[int] = field(default_factory=list)
    cost_usd: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return sum(self.rewards) / max(1, len(self.rewards)) if self.rewards else 0

    @property
    def std_reward(self) -> float:
        if len(self.rewards) < 2:
            return 0.0
        mu = self.mean_reward
        return math.sqrt(sum((r - mu) ** 2 for r in self.rewards) / (len(self.rewards) - 1))

    @property
    def success_rate(self) -> float:
        successes = sum(
            1 for t in self.terminals
            if t in ("deal_closed", "program_delivered", "migration_complete")
        )
        return successes / max(1, len(self.terminals))

    @property
    def mean_tokens(self) -> float:
        return sum(self.total_tokens) / max(1, len(self.total_tokens)) if self.total_tokens else 0

    @property
    def mean_latency_ms(self) -> float:
        return sum(self.total_latency_ms) / max(1, len(self.total_latency_ms)) if self.total_latency_ms else 0

    @property
    def mean_cost(self) -> float:
        return sum(self.cost_usd) / max(1, len(self.cost_usd)) if self.cost_usd else 0

    @property
    def total_cost(self) -> float:
        return sum(self.cost_usd)


# ═══════════════════════════════════════════════════════════════════════
#  Environment Factory
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
#  LLM Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════

class LLMBenchmarkRunner:
    """
    Runs structured LLM experiments across model families.

    Usage:
        runner = LLMBenchmarkRunner()
        results = runner.run(
            env_id="sales",
            model_names=["claude-sonnet", "gpt-4o", "gpt-4o-mini"],
            episodes=3,
        )
        runner.print_report(results)
    """

    def __init__(
        self,
        llm_every_n: int = 5,
        beam_width: int = 2,
        beam_depth: int = 1,
        open_model_url: Optional[str] = None,
    ):
        self.llm_every_n = llm_every_n
        self.beam_width = beam_width
        self.beam_depth = beam_depth
        # Allow override for open model base URL
        self.open_model_url = (
            open_model_url
            or os.environ.get("OPEN_MODEL_BASE_URL")
            or "http://localhost:11434/v1"  # Ollama default
        )

    def _create_llm(self, spec: ModelSpec) -> LLMInterface:
        """Create LLM client from spec."""
        api_key = os.environ.get(spec.api_key_env, "")

        if spec.provider == "claude":
            return ClaudeLLM(
                model=spec.model_id,
                api_key=api_key,
                max_tokens=spec.max_tokens,
            )
        else:
            # OpenAI or OpenAI-compatible
            base_url = spec.base_url or (
                self.open_model_url if spec.tier == "open" else None
            )
            return OpenAILLM(
                model=spec.model_id,
                api_key=api_key or "not-needed",
                base_url=base_url,
                max_tokens=spec.max_tokens,
            )

    def _check_availability(self, spec: ModelSpec) -> Tuple[bool, str]:
        """Check if a model is available (API key set, endpoint reachable)."""
        api_key = os.environ.get(spec.api_key_env, "")

        if spec.tier != "open" and not api_key:
            return False, f"Missing env var: {spec.api_key_env}"

        return True, "OK"

    def run(
        self,
        env_id: str,
        model_names: Optional[List[str]] = None,
        episodes: int = 3,
        seed: int = 42,
        verbose: bool = False,
    ) -> List[LLMBenchmarkResult]:
        """Run benchmark across specified models."""
        if model_names is None:
            # Default: run all models with available API keys
            model_names = list(MODEL_REGISTRY.keys())

        results: List[LLMBenchmarkResult] = []

        print(f"\n{'='*80}")
        print(f"  LLM BENCHMARK: {env_id.upper()}")
        print(f"  Models: {', '.join(model_names)}")
        print(f"  Episodes: {episodes}, LLM every {self.llm_every_n} steps")
        print(f"{'='*80}")

        for model_name in model_names:
            spec = MODEL_REGISTRY.get(model_name)
            if spec is None:
                print(f"\n  [SKIP] Unknown model: {model_name}")
                continue

            available, reason = self._check_availability(spec)
            if not available:
                print(f"\n  [SKIP] {spec.name}: {reason}")
                result = LLMBenchmarkResult(
                    model_name=spec.name, model_id=spec.model_id,
                    tier=spec.tier, env_id=env_id, episodes=0,
                    errors=[reason],
                )
                results.append(result)
                continue

            print(f"\n  Running: {spec.name} ({spec.model_id})")
            result = self._run_model(spec, env_id, episodes, seed, verbose)
            results.append(result)

        return results

    def _run_model(
        self,
        spec: ModelSpec,
        env_id: str,
        episodes: int,
        seed: int,
        verbose: bool,
    ) -> LLMBenchmarkResult:
        """Run all episodes for a single model."""
        result = LLMBenchmarkResult(
            model_name=spec.name,
            model_id=spec.model_id,
            tier=spec.tier,
            env_id=env_id,
            episodes=episodes,
        )

        for ep in range(episodes):
            ep_seed = seed + ep
            print(f"    Episode {ep+1}/{episodes} (seed={ep_seed})...", end=" ", flush=True)

            try:
                ep_result = self._run_single_episode(spec, env_id, ep_seed, verbose)
                result.rewards.append(ep_result["reward"])
                result.steps.append(ep_result["steps"])
                result.phases.append(ep_result["phase"])
                result.terminals.append(ep_result["terminal"])
                result.total_tokens.append(ep_result["tokens"])
                result.total_latency_ms.append(ep_result["latency_ms"])
                result.llm_decisions.append(ep_result["llm_decisions"])
                result.parse_failures.append(ep_result.get("parse_failures", 0))
                result.cost_usd.append(ep_result["cost_usd"])

                print(
                    f"reward={ep_result['reward']:+.1f} "
                    f"tokens={ep_result['tokens']} "
                    f"cost=${ep_result['cost_usd']:.4f} "
                    f"({ep_result['terminal']})"
                )

            except Exception as e:
                result.errors.append(str(e))
                print(f"ERROR: {e}")

        return result

    def _run_single_episode(
        self,
        spec: ModelSpec,
        env_id: str,
        seed: int,
        verbose: bool,
    ) -> Dict[str, Any]:
        """Run one episode with an LLM agent."""
        env = ENV_FACTORIES[env_id]()
        obs = env.reset(seed)

        memory = MemorySystem()
        planner = HierarchicalPlanner(
            beam_width=self.beam_width,
            beam_depth=self.beam_depth,
        )
        enum_id = ENV_ID_MAP.get(env_id, EnvID.PM)
        planner.init_goal_tree(enum_id)
        reward_shaper = RewardShaper(env_id)
        vf_registry = ValueFunctionRegistry()
        skill_library = SkillLibrary()

        ctx = AgentContext(
            memory=memory, planner=planner, skill_library=skill_library,
            vf_registry=vf_registry, env_id=env_id,
            beam_width=self.beam_width, beam_depth=self.beam_depth,
        )

        config = LLMAgentConfig(
            llm_provider=spec.provider if spec.provider != "open" else "openai",
            model=spec.model_id,
            api_key=os.environ.get(spec.api_key_env, "not-needed"),
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
            beam_depth=self.beam_depth,
            beam_width=self.beam_width,
            llm_every_n_steps=self.llm_every_n,
        )

        # Patch base_url for open models
        agent = LLMAgent(ctx, config, env_id)
        if spec.base_url or spec.tier == "open":
            url = spec.base_url or self.open_model_url
            if hasattr(agent._llm, 'base_url'):
                agent._llm.base_url = url
            if hasattr(agent._llm, '_client'):
                agent._llm._client = None  # Force re-creation with new URL

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

            if verbose and step_count % 50 == 0:
                print(f"      [step {obs.step}] reward={total_reward:+.1f}")

        stats = agent.stats()
        tokens = stats["total_tokens"]

        # Estimate cost
        # Approximate 70% input, 30% output tokens
        input_tokens = int(tokens * 0.7)
        output_tokens = tokens - input_tokens
        cost = (
            (input_tokens / 1000) * spec.cost_per_1k_input
            + (output_tokens / 1000) * spec.cost_per_1k_output
        )

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

        return {
            "reward": total_reward,
            "steps": step_count,
            "phase": obs.phase,
            "terminal": terminal,
            "tokens": tokens,
            "latency_ms": stats["total_latency_ms"],
            "llm_decisions": stats["llm_decisions"],
            "cost_usd": cost,
        }

    # ── Reporting ─────────────────────────────────────────────────────

    @staticmethod
    def print_report(results: List[LLMBenchmarkResult]) -> None:
        """Print formatted comparison table."""
        if not results:
            print("No results to report.")
            return

        env_id = results[0].env_id

        print(f"\n{'='*100}")
        print(f"  LLM BENCHMARK RESULTS: {env_id.upper()}")
        print(f"{'='*100}")

        # Header
        print(f"  {'Model':<22s} {'Tier':<10s} {'Reward':>10s} {'StdDev':>8s} "
              f"{'Success':>8s} {'Tokens':>8s} {'Latency':>9s} {'Cost':>8s} {'Errors':>6s}")
        print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*6}")

        for r in results:
            if r.errors and not r.rewards:
                print(f"  {r.model_name:<22s} {r.tier:<10s} {'SKIPPED':>10s}  "
                      f"({r.errors[0][:40]})")
                continue

            latency_str = f"{r.mean_latency_ms/1000:.1f}s" if r.mean_latency_ms else "N/A"
            cost_str = f"${r.total_cost:.4f}"

            print(
                f"  {r.model_name:<22s} {r.tier:<10s} "
                f"{r.mean_reward:>+10.1f} {r.std_reward:>8.1f} "
                f"{r.success_rate:>8.1%} "
                f"{r.mean_tokens:>8.0f} "
                f"{latency_str:>9s} "
                f"{cost_str:>8s} "
                f"{len(r.errors):>6d}"
            )

        # Summary insights
        scored = [r for r in results if r.rewards]
        if len(scored) >= 2:
            best = max(scored, key=lambda r: r.mean_reward)
            cheapest = min(scored, key=lambda r: r.mean_cost) if any(r.mean_cost > 0 for r in scored) else None

            print(f"\n  Best performing:    {best.model_name} ({best.mean_reward:+.1f} reward)")
            if cheapest and cheapest.mean_cost > 0:
                print(f"  Most cost-efficient: {cheapest.model_name} "
                      f"(${cheapest.total_cost:.4f} total)")

            # Tier comparison
            tiers: Dict[str, List[LLMBenchmarkResult]] = {}
            for r in scored:
                tiers.setdefault(r.tier, []).append(r)

            if len(tiers) > 1:
                print(f"\n  Tier Comparison:")
                for tier, tier_results in sorted(tiers.items()):
                    avg_reward = sum(r.mean_reward for r in tier_results) / len(tier_results)
                    avg_cost = sum(r.mean_cost for r in tier_results) / len(tier_results)
                    print(f"    {tier:<12s}: avg_reward={avg_reward:+.1f}, "
                          f"avg_cost=${avg_cost:.4f}/episode")

    @staticmethod
    def to_dict(results: List[LLMBenchmarkResult]) -> Dict[str, Any]:
        """Convert results to JSON-serializable dict."""
        return {
            "env_id": results[0].env_id if results else "",
            "models": [
                {
                    "name": r.model_name,
                    "model_id": r.model_id,
                    "tier": r.tier,
                    "episodes": r.episodes,
                    "mean_reward": round(r.mean_reward, 2),
                    "std_reward": round(r.std_reward, 2),
                    "success_rate": round(r.success_rate, 3),
                    "mean_tokens": round(r.mean_tokens, 0),
                    "mean_latency_ms": round(r.mean_latency_ms, 0),
                    "total_cost_usd": round(r.total_cost, 4),
                    "rewards": [round(r, 2) for r in r.rewards],
                    "errors": r.errors,
                }
                for r in results
            ],
        }


# ═══════════════════════════════════════════════════════════════════════
#  Convenience Functions
# ═══════════════════════════════════════════════════════════════════════

def get_available_models() -> List[str]:
    """Return model names that have their API keys configured."""
    available = []
    for name, spec in MODEL_REGISTRY.items():
        if spec.tier == "open":
            available.append(name)  # Open models don't need API key
        elif os.environ.get(spec.api_key_env):
            available.append(name)
    return available


def run_llm_benchmark(
    env_id: str,
    model_names: Optional[List[str]] = None,
    episodes: int = 3,
    seed: int = 42,
    llm_every_n: int = 5,
    open_model_url: Optional[str] = None,
    verbose: bool = False,
) -> List[LLMBenchmarkResult]:
    """Run LLM benchmark with sensible defaults."""
    runner = LLMBenchmarkRunner(
        llm_every_n=llm_every_n,
        open_model_url=open_model_url,
    )

    if model_names is None:
        model_names = get_available_models()
        if not model_names:
            print("\nNo API keys found. Set one or more of:")
            print("  ANTHROPIC_API_KEY    — for Claude models")
            print("  OPENAI_API_KEY       — for GPT models")
            print("  OPEN_MODEL_API_KEY   — for open models (optional)")
            print("  OPEN_MODEL_BASE_URL  — endpoint for open models")
            print(f"\nAvailable model configs: {list(MODEL_REGISTRY.keys())}")
            return []

    results = runner.run(env_id, model_names, episodes, seed, verbose)
    runner.print_report(results)
    return results
