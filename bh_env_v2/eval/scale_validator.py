"""
100K+ Step Scale Validation — Upgrade 4 (LOW Priority -> IMPLEMENTED).

Validates that the framework operates correctly at 10x-100x scale:
  - Memory compression fidelity metrics (can compressed memory recall shocks?)
  - IDF corpus drift tracking over long episodes
  - StrategicInsight quality assessment after 200+ compression cycles
  - End-to-end validation harness with configurable scale
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..engine.types import Action, ActionType, EnvID, Event, Observation
from ..engine.environments.base import BaseEnvironment
from ..engine.environments.sales import EnterpriseSalesPipeline
from ..engine.environments.pm import ProgramRescueEnvironment
from ..engine.environments.hr_it import ITTransformationEnv
from ..engine.environments.scaled import ScaleConfig, ScaledEnvironment
from ..engine.reward_shaping import RewardShaper
from ..memory.memory_system import MemorySystem, EpisodicMemory
from ..planning.planner import HierarchicalPlanner
from ..planning.value_fn import ValueFunctionRegistry
from ..skills.skill_library import ExperienceReplay, SkillLibrary, Transition
from ..agents.v2_agent import AgentContext, V2Agent


# ═════════════════════════════════════════════════════════════════════════════
#  Fidelity Metrics
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FidelityReport:
    """Report on memory compression fidelity."""
    total_events: int = 0
    compressed_events: int = 0
    compression_ratio: float = 0.0

    # Shock recall: can we retrieve shock events from compressed memory?
    shock_events_total: int = 0
    shock_events_recalled: int = 0
    shock_recall_rate: float = 0.0

    # Milestone recall
    milestone_events_total: int = 0
    milestone_events_recalled: int = 0
    milestone_recall_rate: float = 0.0

    # Phase transition recall
    phase_transitions_total: int = 0
    phase_transitions_recalled: int = 0
    phase_recall_rate: float = 0.0

    # High-reward event recall
    high_reward_total: int = 0
    high_reward_recalled: int = 0
    reward_recall_rate: float = 0.0

    # Strategic insight quality
    insights_generated: int = 0
    insights_with_recommendations: int = 0
    insight_quality_score: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  Memory Fidelity Report",
            "=" * 60,
            f"  Total Events:       {self.total_events}",
            f"  Compression Ratio:  {self.compression_ratio:.1f}:1",
            "",
            "  Recall Rates:",
            f"    Shock Events:     {self.shock_recall_rate:.1%} "
            f"({self.shock_events_recalled}/{self.shock_events_total})",
            f"    Milestones:       {self.milestone_recall_rate:.1%} "
            f"({self.milestone_events_recalled}/{self.milestone_events_total})",
            f"    Phase Transitions:{self.phase_recall_rate:.1%} "
            f"({self.phase_transitions_recalled}/{self.phase_transitions_total})",
            f"    High Reward:      {self.reward_recall_rate:.1%} "
            f"({self.high_reward_recalled}/{self.high_reward_total})",
            "",
            f"  Strategic Insights: {self.insights_generated}",
            f"  Insight Quality:    {self.insight_quality_score:.2f}/1.0",
            "=" * 60,
        ]
        return "\n".join(lines)


class FidelityMetrics:
    """Measures memory retrieval quality after compression."""

    def __init__(self, memory: MemorySystem) -> None:
        self._memory = memory

    def evaluate(self, ground_truth_events: List[Event]) -> FidelityReport:
        """Evaluate fidelity against ground truth events."""
        report = FidelityReport()
        report.total_events = len(ground_truth_events)

        # Count events by type in ground truth
        shock_events: List[Event] = []
        milestone_events: List[Event] = []
        phase_events: List[Event] = []
        high_reward_events: List[Event] = []

        for event in ground_truth_events:
            tags_lower = [t.lower() for t in event.tags]
            if any("shock" in t for t in tags_lower):
                shock_events.append(event)
            if any("milestone" in t for t in tags_lower):
                milestone_events.append(event)
            if any("phase" in t for t in tags_lower):
                phase_events.append(event)
            if abs(event.reward) > 10:
                high_reward_events.append(event)

        report.shock_events_total = len(shock_events)
        report.milestone_events_total = len(milestone_events)
        report.phase_transitions_total = len(phase_events)
        report.high_reward_total = len(high_reward_events)

        # Test recall: can we retrieve each type from memory?
        report.shock_events_recalled = self._test_recall(
            shock_events, "shock event"
        )
        report.milestone_events_recalled = self._test_recall(
            milestone_events, "milestone"
        )
        report.phase_transitions_recalled = self._test_recall(
            phase_events, "phase transition"
        )
        report.high_reward_recalled = self._test_recall(
            high_reward_events, "high reward event"
        )

        # Compute rates
        report.shock_recall_rate = (
            report.shock_events_recalled / report.shock_events_total
            if report.shock_events_total > 0 else 1.0
        )
        report.milestone_recall_rate = (
            report.milestone_events_recalled / report.milestone_events_total
            if report.milestone_events_total > 0 else 1.0
        )
        report.phase_recall_rate = (
            report.phase_transitions_recalled / report.phase_transitions_total
            if report.phase_transitions_total > 0 else 1.0
        )
        report.reward_recall_rate = (
            report.high_reward_recalled / report.high_reward_total
            if report.high_reward_total > 0 else 1.0
        )

        # Compression ratio
        compressed_count = (
            len(self._memory.daily_summaries)
            + len(self._memory.milestone_summaries)
            + len(self._memory.strategic_insights)
        )
        report.compressed_events = compressed_count
        report.compression_ratio = (
            report.total_events / max(1, compressed_count)
        )

        # Strategic insight quality
        report.insights_generated = len(self._memory.strategic_insights)
        report.insights_with_recommendations = sum(
            1 for si in self._memory.strategic_insights if si.recommendation
        )
        if report.insights_generated > 0:
            report.insight_quality_score = (
                report.insights_with_recommendations / report.insights_generated
            )

        return report

    def _test_recall(self, events: List[Event], query_prefix: str) -> int:
        """Test how many events can be recalled via memory retrieval."""
        recalled = 0
        for event in events:
            # Query using the event's key terms
            query = f"{query_prefix} {event.event_type} {' '.join(event.tags[:3])}"
            results = self._memory.episodic.retrieve_relevant(query, k=10)

            # Check if any result matches the original event (by step and text overlap)
            for retrieved_event, score in results:
                if retrieved_event.step == event.step:
                    recalled += 1
                    break
                # Also accept close matches
                if (abs(retrieved_event.step - event.step) <= 2
                        and event.text[:30] in retrieved_event.text):
                    recalled += 1
                    break
        return recalled


# ═════════════════════════════════════════════════════════════════════════════
#  IDF Drift Tracker
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class IDFSnapshot:
    """Snapshot of IDF state at a point in time."""
    step: int
    num_docs: int
    vocab_size: int
    top_terms: List[Tuple[str, float]]    # (term, idf_value)
    bottom_terms: List[Tuple[str, float]]  # lowest IDF = most common


class IDFDriftTracker:
    """
    Monitors IDF corpus stability over long episodes.
    Takes periodic snapshots and measures drift between consecutive snapshots.
    """

    def __init__(self, snapshot_interval: int = 500) -> None:
        self._interval = snapshot_interval
        self._snapshots: List[IDFSnapshot] = []
        self._last_snapshot_step: int = -snapshot_interval

    def maybe_snapshot(self, step: int, episodic: EpisodicMemory) -> Optional[IDFSnapshot]:
        """Take a snapshot if interval has elapsed."""
        if step - self._last_snapshot_step < self._interval:
            return None

        self._last_snapshot_step = step
        snap = self._take_snapshot(step, episodic)
        self._snapshots.append(snap)
        return snap

    def _take_snapshot(self, step: int, episodic: EpisodicMemory) -> IDFSnapshot:
        num_docs = episodic._num_docs
        doc_freq = episodic._doc_freq

        vocab_size = len(doc_freq)

        # Compute IDF for all terms
        idf_values: List[Tuple[str, float]] = []
        for term, df in doc_freq.items():
            if df > 0 and num_docs > 0:
                idf = math.log(num_docs / df)
            else:
                idf = 0.0
            idf_values.append((term, idf))

        idf_values.sort(key=lambda x: x[1], reverse=True)

        return IDFSnapshot(
            step=step,
            num_docs=num_docs,
            vocab_size=vocab_size,
            top_terms=idf_values[:20],
            bottom_terms=idf_values[-20:] if len(idf_values) >= 20 else [],
        )

    def compute_drift(self) -> List[Dict[str, Any]]:
        """Compute drift metrics between consecutive snapshots."""
        drifts: List[Dict[str, Any]] = []

        for i in range(1, len(self._snapshots)):
            prev = self._snapshots[i - 1]
            curr = self._snapshots[i]

            # Vocabulary growth rate
            vocab_growth = curr.vocab_size - prev.vocab_size

            # Top-term stability: Jaccard similarity of top-20 terms
            prev_top = set(t for t, _ in prev.top_terms)
            curr_top = set(t for t, _ in curr.top_terms)
            jaccard = (
                len(prev_top & curr_top) / len(prev_top | curr_top)
                if prev_top | curr_top else 1.0
            )

            # IDF value drift: average absolute change in IDF for shared terms
            prev_idf = dict(prev.top_terms + prev.bottom_terms)
            curr_idf = dict(curr.top_terms + curr.bottom_terms)
            shared = set(prev_idf.keys()) & set(curr_idf.keys())
            avg_drift = 0.0
            if shared:
                avg_drift = sum(
                    abs(prev_idf[t] - curr_idf[t]) for t in shared
                ) / len(shared)

            drifts.append({
                "step_range": f"{prev.step}->{curr.step}",
                "vocab_growth": vocab_growth,
                "top_term_stability": round(jaccard, 4),
                "avg_idf_drift": round(avg_drift, 4),
                "num_docs": curr.num_docs,
                "vocab_size": curr.vocab_size,
            })

        return drifts

    def report(self) -> str:
        """Generate a human-readable drift report."""
        drifts = self.compute_drift()
        if not drifts:
            return "No IDF drift data yet (need at least 2 snapshots)."

        lines = [
            "=" * 60,
            "  IDF Corpus Drift Report",
            "=" * 60,
            f"  {'Step Range':<18} {'Vocab Growth':>12} {'Top Stability':>14} {'IDF Drift':>10}",
            "-" * 60,
        ]
        for d in drifts:
            lines.append(
                f"  {d['step_range']:<18} {d['vocab_growth']:>+12} "
                f"{d['top_term_stability']:>14.4f} {d['avg_idf_drift']:>10.4f}"
            )

        # Summary
        avg_stability = sum(d["top_term_stability"] for d in drifts) / len(drifts)
        avg_drift_val = sum(d["avg_idf_drift"] for d in drifts) / len(drifts)
        lines.extend([
            "-" * 60,
            f"  Average top-term stability: {avg_stability:.4f}",
            f"  Average IDF drift:          {avg_drift_val:.4f}",
            f"  Final vocab size:           {drifts[-1]['vocab_size']}",
            f"  Final corpus docs:          {drifts[-1]['num_docs']}",
        ])

        if avg_stability < 0.5:
            lines.append("  WARNING: High vocabulary instability detected!")
        if avg_drift_val > 1.0:
            lines.append("  WARNING: Significant IDF drift — retrieval quality may degrade.")

        lines.append("=" * 60)
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  Scale Validation Result
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ScaleValidationResult:
    """Complete validation result for a scaled episode."""
    env_id: str
    scale: int
    total_steps: int
    total_reward: float
    shaped_reward: float
    terminal_reason: str
    phase_reached: int
    elapsed_seconds: float
    fidelity: Optional[FidelityReport] = None
    idf_drift_report: str = ""
    compression_cycles: int = 0
    peak_memory_events: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  Scale Validation Result  [{self.env_id} x{self.scale}]",
            "=" * 60,
            f"  Steps:             {self.total_steps}",
            f"  Reward:            {self.shaped_reward:+.2f}",
            f"  Phase:             {self.phase_reached}",
            f"  Terminal:          {self.terminal_reason}",
            f"  Elapsed:           {self.elapsed_seconds:.2f}s",
            f"  Compression Cycles:{self.compression_cycles}",
            f"  Peak Memory Events:{self.peak_memory_events}",
        ]
        if self.fidelity:
            lines.append("")
            lines.append(self.fidelity.summary())
        if self.idf_drift_report:
            lines.append("")
            lines.append(self.idf_drift_report)
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  Scale Validation Harness
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


class ScaleValidationHarness:
    """
    Runs scaled episodes with comprehensive validation metrics.

    Validates:
      - Memory compression fidelity at scale
      - IDF corpus drift over long episodes
      - StrategicInsight quality after many compression cycles
      - Agent performance does not degrade catastrophically at scale
    """

    def __init__(
        self,
        beam_width: int = 4,
        beam_depth: int = 3,
        idf_snapshot_interval: int = 500,
    ) -> None:
        self._beam_width = beam_width
        self._beam_depth = beam_depth
        self._idf_interval = idf_snapshot_interval

    def validate(
        self,
        env_id: str,
        scale: int = 10,
        seed: int = 42,
        verbose: bool = False,
    ) -> ScaleValidationResult:
        """Run a scaled validation episode."""
        t0 = time.time()

        # Create scaled environment
        base_factory = _ENV_FACTORIES.get(env_id)
        if base_factory is None:
            raise ValueError(f"Unknown env_id: {env_id}")

        base_env = base_factory()
        if scale <= 1:
            env = base_env
        else:
            config = ScaleConfig(
                scale=scale,
                extra_risks=50 * scale,
                extra_instructions=100 * scale,
                extra_stakeholders=5 * scale,
                label=f"validate_x{scale}",
            )
            env = ScaledEnvironment(base_env, config)

        obs = env.reset(seed)

        # Create components
        memory = MemorySystem()
        vf_registry = ValueFunctionRegistry()
        skill_library = SkillLibrary()
        planner = HierarchicalPlanner(
            beam_width=self._beam_width,
            beam_depth=self._beam_depth,
        )
        enum_id = _ENV_ID_MAP.get(env_id, EnvID.PM)
        planner.init_goal_tree(enum_id)

        reward_shaper = RewardShaper(env_id)
        idf_tracker = IDFDriftTracker(snapshot_interval=self._idf_interval)

        ctx = AgentContext(
            memory=memory,
            planner=planner,
            skill_library=skill_library,
            vf_registry=vf_registry,
            env_id=env_id,
        )
        agent = V2Agent(ctx)

        # Track ground truth events for fidelity measurement
        ground_truth_events: List[Event] = []
        total_reward = 0.0
        shaped_total = 0.0
        step_counter = 0
        peak_events = 0
        compression_cycles = 0

        while not obs.done:
            state_digest = env.state_digest()
            action = agent.decide(obs, env)
            obs = env.step(action)
            step_counter += 1

            new_digest = env.state_digest()
            events_dicts = [
                {"event_type": e.event_type, "tags": e.tags}
                for e in obs.events
            ]
            shaped_reward, _ = reward_shaper.shape(
                obs.reward, new_digest, obs.step, events_dicts,
            )

            total_reward += obs.reward
            shaped_total += shaped_reward

            # Track ground truth
            for event in obs.events:
                ground_truth_events.append(event)

            agent.post_step(obs, action, shaped_reward, new_digest)

            # IDF drift tracking
            idf_tracker.maybe_snapshot(obs.step, memory.episodic)

            # Track compression cycles
            prev_insights = len(memory.strategic_insights)
            if len(memory.strategic_insights) > prev_insights:
                compression_cycles += 1

            peak_events = max(peak_events, memory.episodic.size)

            if verbose and step_counter % (200 * max(1, scale)) == 0:
                print(
                    f"  [step {obs.step:>6}] phase={obs.phase} "
                    f"reward={shaped_total:+.1f} "
                    f"memory={memory.episodic.size} "
                    f"insights={len(memory.strategic_insights)}"
                )

        # Fidelity measurement
        fidelity_metrics = FidelityMetrics(memory)
        fidelity = fidelity_metrics.evaluate(ground_truth_events)

        # IDF drift report
        idf_report = idf_tracker.report()

        compression_cycles = len(memory.strategic_insights)

        # Terminal reason
        state = env.state_digest()
        terminal = "timeout"
        if state.get("deal_closed"):
            terminal = "deal_closed"
        elif state.get("program_delivered"):
            terminal = "program_delivered"
        elif state.get("migrated_pct", 0) >= 100:
            terminal = "migration_complete"

        elapsed = time.time() - t0

        return ScaleValidationResult(
            env_id=env_id,
            scale=scale,
            total_steps=step_counter,
            total_reward=total_reward,
            shaped_reward=shaped_total,
            terminal_reason=terminal,
            phase_reached=obs.phase,
            elapsed_seconds=elapsed,
            fidelity=fidelity,
            idf_drift_report=idf_report,
            compression_cycles=compression_cycles,
            peak_memory_events=peak_events,
        )

    def validate_all_scales(
        self,
        env_id: str,
        scales: Optional[List[int]] = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> List[ScaleValidationResult]:
        """Run validation across multiple scale factors."""
        if scales is None:
            scales = [1, 5, 10]

        results: List[ScaleValidationResult] = []
        for scale in scales:
            print(f"\nValidating {env_id} at scale={scale}...")
            result = self.validate(env_id, scale=scale, seed=seed, verbose=verbose)
            print(
                f"  Done: {result.total_steps} steps, "
                f"reward={result.shaped_reward:+.1f}, "
                f"{result.elapsed_seconds:.1f}s"
            )
            if result.fidelity:
                print(
                    f"  Recall: shock={result.fidelity.shock_recall_rate:.0%} "
                    f"milestone={result.fidelity.milestone_recall_rate:.0%} "
                    f"reward={result.fidelity.reward_recall_rate:.0%}"
                )
            results.append(result)

        return results

    def print_comparison(self, results: List[ScaleValidationResult]) -> None:
        """Print a comparison table across scale factors."""
        if not results:
            print("No results to compare.")
            return

        print(f"\n{'=' * 80}")
        print(f"  Scale Comparison [{results[0].env_id}]")
        print(f"{'=' * 80}")
        header = (
            f"  {'Scale':>6} {'Steps':>8} {'Reward':>10} {'Phase':>6} "
            f"{'Shock%':>7} {'Mile%':>7} {'Time':>8} {'Insights':>9}"
        )
        print(header)
        print("-" * 80)

        for r in results:
            shock_pct = f"{r.fidelity.shock_recall_rate:.0%}" if r.fidelity else "N/A"
            mile_pct = f"{r.fidelity.milestone_recall_rate:.0%}" if r.fidelity else "N/A"
            print(
                f"  x{r.scale:<5} {r.total_steps:>8} {r.shaped_reward:>+10.1f} "
                f"{r.phase_reached:>6} {shock_pct:>7} {mile_pct:>7} "
                f"{r.elapsed_seconds:>7.1f}s {r.compression_cycles:>9}"
            )

        print(f"{'=' * 80}")
