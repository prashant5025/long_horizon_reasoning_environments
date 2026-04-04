"""
Multi-Store Memory System with 4-Level Compression
BusinessHorizonENV v2

Three memory stores (Working, Episodic, Semantic) with four compression
levels (Event -> DailySummary -> MilestoneSummary -> StrategicInsight).
Pure Python TF-IDF retrieval, no external dependencies.
"""
from __future__ import annotations

import math
import collections
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..engine.types import Event


# ─── Compression Dataclasses ───────────────────────────────────────────────────

@dataclass
class DailySummary:
    """~10:1 compression. Produced every 50 steps."""
    step_start: int
    step_end: int
    top_events: List[str] = field(default_factory=list)
    tag_frequencies: Dict[str, int] = field(default_factory=dict)
    net_reward: float = 0.0
    shock_occurred: bool = False
    milestone_occurred: bool = False
    event_count: int = 0


@dataclass
class MilestoneSummary:
    """~30:1 compression. Produced on phase transitions."""
    phase_from: int
    phase_to: int
    step: int
    outcomes: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    net_reward: float = 0.0
    shock_count: int = 0
    milestone_count: int = 0


@dataclass
class StrategicInsight:
    """~100:1 compression. Produced every 200 steps."""
    step: int
    recurring_risks: Dict[str, int] = field(default_factory=dict)
    trend_direction: str = "stable"
    recommendation: str = ""
    net_reward: float = 0.0
    period_steps: int = 0


# ─── Semantic Memory Entry ──────────────────────────────────────────────────────

@dataclass
class SemanticEntry:
    concept: str
    belief: str
    confidence: float
    last_updated: int


# ─── TF-IDF Helpers ─────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Split on whitespace and lowercase."""
    return text.lower().split()


def _compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Term frequency: count / total tokens."""
    if not tokens:
        return {}
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens)
    return {t: c / total for t, c in counts.items()}


def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors (dicts)."""
    if not a or not b:
        return 0.0
    dot = 0.0
    for term, val in a.items():
        if term in b:
            dot += val * b[term]
    if dot == 0.0:
        return 0.0
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ─── Working Memory ─────────────────────────────────────────────────────────────

class WorkingMemory:
    """Hot context buffer. Fixed-size deque of recent events. O(1) access."""

    def __init__(self, max_size: int = 20) -> None:
        self._buffer: collections.deque[Event] = collections.deque(maxlen=max_size)

    def add(self, event: Event) -> None:
        self._buffer.append(event)

    def get_all(self) -> List[Event]:
        return list(self._buffer)

    def latest(self) -> Optional[Event]:
        return self._buffer[-1] if self._buffer else None

    def __len__(self) -> int:
        return len(self._buffer)

    def __getitem__(self, idx: int) -> Event:
        return self._buffer[idx]


# ─── Episodic Memory ────────────────────────────────────────────────────────────

class EpisodicMemory:
    """
    Stores raw events with TF-IDF vectors for retrieval.
    Capped at 2000 events (oldest evicted on overflow).
    Incremental IDF corpus.
    """

    MAX_EVENTS = 2000

    def __init__(self) -> None:
        self._events: List[Event] = []
        self._tf_vectors: List[Dict[str, float]] = []
        self._importances: List[float] = []
        # IDF bookkeeping
        self._doc_freq: Dict[str, int] = {}  # term -> number of docs containing it
        self._num_docs: int = 0

    # ── importance scoring ──────────────────────────────────────────────────

    @staticmethod
    def _importance_score(event: Event) -> float:
        score = event.importance
        tags_lower = [t.lower() for t in event.tags]
        if "shock" in tags_lower:
            score += 10.0
        if "milestone" in tags_lower:
            score += 8.0
        if event.reward > 5.0:
            score += 5.0
        return score

    # ── add / evict ─────────────────────────────────────────────────────────

    def add(self, event: Event) -> None:
        tokens = _tokenize(event.text)
        tf = _compute_tf(tokens)

        # Update document frequency (each unique term in this doc)
        unique_terms = set(tokens)
        for term in unique_terms:
            self._doc_freq[term] = self._doc_freq.get(term, 0) + 1
        self._num_docs += 1

        self._events.append(event)
        self._tf_vectors.append(tf)
        self._importances.append(self._importance_score(event))

        # Evict oldest if over capacity
        if len(self._events) > self.MAX_EVENTS:
            self._evict_oldest()

    def _evict_oldest(self) -> None:
        """Remove the oldest event and decrement its doc-freq contributions."""
        old_tf = self._tf_vectors[0]
        for term in old_tf:
            self._doc_freq[term] = self._doc_freq.get(term, 1) - 1
            if self._doc_freq[term] <= 0:
                del self._doc_freq[term]
        self._num_docs -= 1
        self._events.pop(0)
        self._tf_vectors.pop(0)
        self._importances.pop(0)

    # ── TF-IDF vector construction ──────────────────────────────────────────

    def _tfidf_vector(self, tf: Dict[str, float]) -> Dict[str, float]:
        """Weight a TF vector by current IDF values."""
        if self._num_docs == 0:
            return {}
        vec: Dict[str, float] = {}
        for term, tf_val in tf.items():
            df = self._doc_freq.get(term, 0)
            if df > 0:
                idf = math.log(self._num_docs / df)
            else:
                idf = 0.0
            vec[term] = tf_val * idf
        return vec

    # ── retrieval ───────────────────────────────────────────────────────────

    def retrieve_relevant(self, query: str, k: int = 5) -> List[Tuple[Event, float]]:
        """
        Return top-k events by IDF-weighted cosine similarity to *query*,
        boosted by importance score.
        """
        if not self._events:
            return []

        query_tokens = _tokenize(query)
        query_tf = _compute_tf(query_tokens)
        query_vec = self._tfidf_vector(query_tf)

        scored: List[Tuple[int, float]] = []
        for i, tf in enumerate(self._tf_vectors):
            doc_vec = self._tfidf_vector(tf)
            sim = _cosine_similarity(query_vec, doc_vec)
            # Blend: similarity (0-1 range) + normalised importance boost
            combined = sim + self._importances[i] * 0.01
            scored.append((i, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        results: List[Tuple[Event, float]] = []
        for idx, score in scored[:k]:
            results.append((self._events[idx], score))
        return results

    @property
    def size(self) -> int:
        return len(self._events)

    def events_in_range(self, start_step: int, end_step: int) -> List[Event]:
        """Return events whose step is in [start_step, end_step]."""
        return [e for e in self._events if start_step <= e.step <= end_step]


# ─── Semantic Memory ────────────────────────────────────────────────────────────

class SemanticMemory:
    """Belief map keyed by concept string. O(1) lookup."""

    def __init__(self) -> None:
        self._beliefs: Dict[str, SemanticEntry] = {}

    def update_belief(
        self, concept: str, belief: str, confidence: float, step: int
    ) -> None:
        self._beliefs[concept] = SemanticEntry(
            concept=concept,
            belief=belief,
            confidence=confidence,
            last_updated=step,
        )

    def get_belief(self, concept: str) -> Optional[SemanticEntry]:
        return self._beliefs.get(concept)

    def all_beliefs(self) -> Dict[str, SemanticEntry]:
        return dict(self._beliefs)

    def __len__(self) -> int:
        return len(self._beliefs)


# ─── Risk Pattern Detection ─────────────────────────────────────────────────────

_RISK_PATTERNS: Dict[str, List[str]] = {
    "budget_pressure": ["budget", "cost", "overrun", "expense", "funding", "overspend"],
    "vendor_instability": ["vendor", "supplier", "contractor", "partner", "outsource"],
    "stakeholder_disengagement": [
        "disengage", "unresponsive", "detractor", "champion_lost", "ghosting",
    ],
    "morale_degradation": ["morale", "burnout", "turnover", "attrition", "fatigue"],
    "deadline_proximity": ["deadline", "overdue", "late", "delay", "slip", "behind"],
    "security_incidents": [
        "security", "breach", "ransomware", "vulnerability", "incident", "attack",
    ],
}


def _detect_risk_patterns(events: List[Event]) -> Dict[str, int]:
    """Count occurrences of each risk pattern across events."""
    counts: Dict[str, int] = {k: 0 for k in _RISK_PATTERNS}
    for event in events:
        text_lower = event.text.lower()
        tags_lower = " ".join(event.tags).lower()
        combined = text_lower + " " + tags_lower
        for pattern_name, keywords in _RISK_PATTERNS.items():
            for kw in keywords:
                if kw in combined:
                    counts[pattern_name] += 1
                    break  # count each pattern at most once per event
    return counts


def _trend_direction(recent_rewards: List[float]) -> str:
    """Simple trend: compare first-half average to second-half average."""
    if len(recent_rewards) < 4:
        return "stable"
    mid = len(recent_rewards) // 2
    first_half = sum(recent_rewards[:mid]) / max(1, mid)
    second_half = sum(recent_rewards[mid:]) / max(1, len(recent_rewards) - mid)
    diff = second_half - first_half
    if diff > 1.0:
        return "improving"
    elif diff < -1.0:
        return "declining"
    return "stable"


# ─── Memory System Orchestrator ─────────────────────────────────────────────────

class MemorySystem:
    """
    Orchestrates all three stores and four compression levels.

    Compression thresholds:
      - DailySummary   : every 50 steps
      - MilestoneSummary: on phase transition
      - StrategicInsight: every 200 steps
    """

    def __init__(self) -> None:
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()

        self.daily_summaries: List[DailySummary] = []
        self.milestone_summaries: List[MilestoneSummary] = []
        self.strategic_insights: List[StrategicInsight] = []

        self._last_daily_step: int = 0
        self._last_strategic_step: int = 0
        self._current_phase: int = 1
        self._all_events_buffer: List[Event] = []  # raw buffer for compression window

    # ── public API ──────────────────────────────────────────────────────────

    def record_event(self, event: Event) -> None:
        """Add event to all stores and trigger compression if thresholds met."""
        self.working.add(event)
        self.episodic.add(event)
        self._all_events_buffer.append(event)
        self.trigger_compression(event.step)

    def trigger_compression(self, step: int) -> None:
        """Check every threshold and produce summaries as needed."""
        # DailySummary every 50 steps
        if step - self._last_daily_step >= 50:
            self._produce_daily_summary(self._last_daily_step + 1, step)
            self._last_daily_step = step

        # StrategicInsight every 200 steps
        if step - self._last_strategic_step >= 200:
            self._produce_strategic_insight(step)
            self._last_strategic_step = step

    def notify_phase_transition(self, old_phase: int, new_phase: int, step: int) -> None:
        """Called externally when the environment transitions phases."""
        self._produce_milestone_summary(old_phase, new_phase, step)
        self._current_phase = new_phase

    def context_for_agent(self, step: int, query: str = "") -> str:
        """
        Assemble a prompt-ready context string.
        Order: most compressed first, most recent last.
          strategic -> milestone -> daily -> working memory
        """
        parts: List[str] = []

        # Strategic insights (most compressed)
        if self.strategic_insights:
            parts.append("=== Strategic Insights ===")
            for si in self.strategic_insights[-3:]:  # last 3
                risk_str = ", ".join(
                    f"{k}({v})" for k, v in si.recurring_risks.items() if v > 0
                )
                parts.append(
                    f"[Step {si.step}] Trend: {si.trend_direction} | "
                    f"Risks: {risk_str or 'none'} | "
                    f"Reward: {si.net_reward:+.1f} | "
                    f"Rec: {si.recommendation}"
                )

        # Milestone summaries
        if self.milestone_summaries:
            parts.append("=== Milestone Summaries ===")
            for ms in self.milestone_summaries[-5:]:
                outcomes_str = "; ".join(ms.outcomes[:3]) if ms.outcomes else "none"
                risks_str = "; ".join(ms.risks[:3]) if ms.risks else "none"
                parts.append(
                    f"[Phase {ms.phase_from}->{ms.phase_to} @ step {ms.step}] "
                    f"Outcomes: {outcomes_str} | Risks: {risks_str} | "
                    f"Reward: {ms.net_reward:+.1f}"
                )

        # Daily summaries (recent)
        if self.daily_summaries:
            parts.append("=== Daily Summaries ===")
            for ds in self.daily_summaries[-5:]:
                top_str = "; ".join(ds.top_events[:3]) if ds.top_events else "none"
                flags = []
                if ds.shock_occurred:
                    flags.append("SHOCK")
                if ds.milestone_occurred:
                    flags.append("MILESTONE")
                flag_str = f" [{','.join(flags)}]" if flags else ""
                parts.append(
                    f"[Steps {ds.step_start}-{ds.step_end}]{flag_str} "
                    f"Top: {top_str} | Reward: {ds.net_reward:+.1f} | "
                    f"Events: {ds.event_count}"
                )

        # Episodic retrieval (if query provided)
        if query:
            relevant = self.episodic.retrieve_relevant(query, k=5)
            if relevant:
                parts.append("=== Relevant Past Events ===")
                for ev, score in relevant:
                    parts.append(
                        f"[Step {ev.step}] ({score:.2f}) {ev.text}"
                    )

        # Working memory (most recent, least compressed)
        working_events = self.working.get_all()
        if working_events:
            parts.append("=== Recent Events (Working Memory) ===")
            for ev in working_events:
                tag_str = f" [{', '.join(ev.tags)}]" if ev.tags else ""
                parts.append(
                    f"[Step {ev.step}]{tag_str} {ev.text} (r={ev.reward:+.1f})"
                )

        return "\n".join(parts)

    # ── compression producers ───────────────────────────────────────────────

    def _produce_daily_summary(self, start_step: int, end_step: int) -> None:
        """Produce a DailySummary from events in [start_step, end_step]."""
        events = self.episodic.events_in_range(start_step, end_step)
        if not events:
            return

        # Tag frequencies
        tag_freq: Dict[str, int] = {}
        for ev in events:
            for tag in ev.tags:
                tag_freq[tag] = tag_freq.get(tag, 0) + 1

        # Top-5 by importance
        scored = sorted(
            events,
            key=lambda e: EpisodicMemory._importance_score(e),
            reverse=True,
        )
        top_events = [e.text for e in scored[:5]]

        net_reward = sum(e.reward for e in events)
        shock = any("shock" in t.lower() for e in events for t in e.tags)
        milestone = any("milestone" in t.lower() for e in events for t in e.tags)

        ds = DailySummary(
            step_start=start_step,
            step_end=end_step,
            top_events=top_events,
            tag_frequencies=tag_freq,
            net_reward=net_reward,
            shock_occurred=shock,
            milestone_occurred=milestone,
            event_count=len(events),
        )
        self.daily_summaries.append(ds)

    def _produce_milestone_summary(
        self, phase_from: int, phase_to: int, step: int
    ) -> None:
        """Merge recent DailySummaries into a MilestoneSummary on phase transition."""
        # Use up to last 3 daily summaries
        recent_ds = self.daily_summaries[-3:] if self.daily_summaries else []

        outcomes: List[str] = []
        risks: List[str] = []
        net_reward = 0.0
        shock_count = 0
        milestone_count = 0

        for ds in recent_ds:
            net_reward += ds.net_reward
            if ds.shock_occurred:
                shock_count += 1
            if ds.milestone_occurred:
                milestone_count += 1
            for text in ds.top_events:
                text_lower = text.lower()
                # Classify as outcome or risk
                risk_keywords = [
                    "risk", "threat", "fail", "block", "delay", "breach",
                    "overrun", "decline", "lost", "critical",
                ]
                if any(kw in text_lower for kw in risk_keywords):
                    risks.append(text)
                else:
                    outcomes.append(text)

        # Also check recent events if no daily summaries yet
        if not recent_ds:
            recent_events = self.episodic.events_in_range(max(0, step - 150), step)
            for ev in recent_events:
                text_lower = ev.text.lower()
                net_reward += ev.reward
                risk_keywords = [
                    "risk", "threat", "fail", "block", "delay", "breach",
                    "overrun", "decline", "lost", "critical",
                ]
                if any(kw in text_lower for kw in risk_keywords):
                    risks.append(ev.text)
                else:
                    outcomes.append(ev.text)
                tags_lower = [t.lower() for t in ev.tags]
                if "shock" in tags_lower:
                    shock_count += 1
                if "milestone" in tags_lower:
                    milestone_count += 1

        ms = MilestoneSummary(
            phase_from=phase_from,
            phase_to=phase_to,
            step=step,
            outcomes=outcomes[:10],
            risks=risks[:10],
            net_reward=net_reward,
            shock_count=shock_count,
            milestone_count=milestone_count,
        )
        self.milestone_summaries.append(ms)

    def _produce_strategic_insight(self, step: int) -> None:
        """Produce a StrategicInsight from the last 200 steps of events."""
        lookback_start = max(0, step - 200)
        events = self.episodic.events_in_range(lookback_start, step)
        if not events:
            return

        # Risk pattern detection
        recurring_risks = _detect_risk_patterns(events)

        # Trend direction from rewards
        rewards = [e.reward for e in events]
        trend = _trend_direction(rewards)
        net_reward = sum(rewards)

        # Generate recommendation based on dominant risk
        dominant_risk = max(recurring_risks, key=recurring_risks.get) if recurring_risks else ""
        dominant_count = recurring_risks.get(dominant_risk, 0)

        recommendation = _generate_recommendation(dominant_risk, dominant_count, trend)

        si = StrategicInsight(
            step=step,
            recurring_risks=recurring_risks,
            trend_direction=trend,
            recommendation=recommendation,
            net_reward=net_reward,
            period_steps=len(events),
        )
        self.strategic_insights.append(si)


def _generate_recommendation(
    dominant_risk: str, count: int, trend: str
) -> str:
    """Produce a short recommendation string from pattern analysis."""
    if count == 0:
        if trend == "improving":
            return "No recurring risks detected. Maintain current trajectory."
        elif trend == "declining":
            return "Performance declining but no clear risk pattern. Investigate root causes."
        return "Situation stable. Continue monitoring."

    risk_advice: Dict[str, str] = {
        "budget_pressure": "Reassess budget allocations and defer non-critical spend.",
        "vendor_instability": "Diversify vendor dependencies and establish contingency plans.",
        "stakeholder_disengagement": "Increase stakeholder touchpoints and re-engage champions.",
        "morale_degradation": "Prioritise team well-being initiatives and reduce workload pressure.",
        "deadline_proximity": "Re-prioritise deliverables and negotiate timeline extensions where possible.",
        "security_incidents": "Escalate security posture and conduct immediate vulnerability review.",
    }

    advice = risk_advice.get(
        dominant_risk,
        f"Address recurring {dominant_risk} pattern ({count} occurrences).",
    )

    if trend == "declining":
        advice += " URGENT: trend is declining."
    elif trend == "improving":
        advice += " Trend is improving; stay the course."

    return advice
