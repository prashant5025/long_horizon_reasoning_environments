"""
Cross-episode Skill Learning System for BusinessHorizonENV v2.

Provides experience replay, automatic skill extraction from trajectories,
and a queryable skill library that persists learned action patterns across
episodes.
"""
from __future__ import annotations

import math
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from ..engine.types import ActionType, EnvID, Event


# ─── Data Classes ───────────────────────────────────────────────────────────


@dataclass
class SkillRecord:
    """A reusable action pattern discovered through repeated experience."""
    skill_id: str
    action_sequence: List[str]   # ActionType names
    env_id: str
    trigger_tags: List[str]
    expected_reward: float
    success_rate: float = 1.0
    usage_count: int = 0
    discovery_episode: int = 0


@dataclass
class Transition:
    """A single environment transition stored in replay memory."""
    step: int
    state_digest: dict
    action_type: str
    shaped_reward: float
    done: bool
    context_tags: List[str]
    priority: float


# ─── Experience Replay ──────────────────────────────────────────────────────


class ExperienceReplay:
    """Prioritized circular buffer that persists across episodes."""

    def __init__(self, capacity: int = 50_000) -> None:
        self._capacity = capacity
        self._buffer: List[Transition] = []
        self._pos: int = 0  # write cursor for circular overwrite

    # ── mutators ──────────────────────────────────────────────────────

    def add(self, transition: Transition) -> None:
        """Add a transition; priority is |reward| + 10 if terminal."""
        transition.priority = abs(transition.shaped_reward) + (10.0 if transition.done else 0.0)

        if len(self._buffer) < self._capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._pos] = transition
        self._pos = (self._pos + 1) % self._capacity

    # ── sampling ──────────────────────────────────────────────────────

    def sample(self, n: int) -> List[Transition]:
        """Sample *n* transitions proportional to their priority."""
        if not self._buffer:
            return []

        n = min(n, len(self._buffer))
        priorities = [t.priority for t in self._buffer]
        total = sum(priorities)

        # Degenerate case: all priorities zero -> uniform sampling
        if total <= 0.0:
            return random.sample(self._buffer, n)

        sampled: List[Transition] = []
        seen_indices: set = set()

        for _ in range(n):
            threshold = random.random() * total
            cumulative = 0.0
            for idx, p in enumerate(priorities):
                cumulative += p
                if cumulative >= threshold:
                    if idx not in seen_indices:
                        seen_indices.add(idx)
                        sampled.append(self._buffer[idx])
                    break

        # If duplicates reduced sample size, fill with random unseen items
        if len(sampled) < n:
            remaining = [i for i in range(len(self._buffer)) if i not in seen_indices]
            extra = min(n - len(sampled), len(remaining))
            if extra > 0:
                for idx in random.sample(remaining, extra):
                    sampled.append(self._buffer[idx])

        return sampled

    # ── properties ────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._buffer)


# ─── Skill Extractor ────────────────────────────────────────────────────────


# Type alias for the pattern frequency key
_PatternKey = Tuple[Tuple[str, ...], str, FrozenSet[str]]


class SkillExtractor:
    """Discovers reusable skills from trajectory data using sliding windows."""

    def __init__(self) -> None:
        # Persists across calls: key -> (occurrence_count, cumulative_reward)
        self._pattern_counts: Dict[_PatternKey, Tuple[int, float]] = defaultdict(
            lambda: (0, 0.0)
        )

    # ── public API ────────────────────────────────────────────────────

    def extract_from_trajectory(
        self,
        trajectory: List[Transition],
        env_id: str,
        episode_num: int,
    ) -> List[SkillRecord]:
        """Scan trajectory with windows of length 2-6, promote frequent high-reward patterns."""
        new_skills: List[SkillRecord] = []

        for window_len in range(2, 7):  # 2..6 inclusive
            for start in range(len(trajectory) - window_len + 1):
                window = trajectory[start : start + window_len]
                cumulative_reward = sum(t.shaped_reward for t in window)

                if cumulative_reward < 10.0:
                    continue

                actions = tuple(t.action_type for t in window)
                dominant_tags = self._dominant_tags(window)
                key: _PatternKey = (actions, env_id, dominant_tags)

                prev_count, prev_reward = self._pattern_counts[key]
                new_count = prev_count + 1
                new_reward = prev_reward + cumulative_reward
                self._pattern_counts[key] = (new_count, new_reward)

                # Promote when the pattern has been seen >= 2 times
                if new_count >= 2 and prev_count < 2:
                    avg_reward = new_reward / new_count
                    skill = SkillRecord(
                        skill_id=uuid.uuid4().hex[:12],
                        action_sequence=list(actions),
                        env_id=env_id,
                        trigger_tags=sorted(dominant_tags),
                        expected_reward=avg_reward,
                        success_rate=1.0,
                        usage_count=0,
                        discovery_episode=episode_num,
                    )
                    new_skills.append(skill)

        return new_skills

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _dominant_tags(window: List[Transition]) -> FrozenSet[str]:
        """Return tags that appear in >50% of transitions in the window."""
        tag_counts: Dict[str, int] = defaultdict(int)
        n = len(window)
        for t in window:
            for tag in t.context_tags:
                tag_counts[tag] += 1

        threshold = n / 2.0
        return frozenset(tag for tag, cnt in tag_counts.items() if cnt > threshold)


# ─── Skill Library ──────────────────────────────────────────────────────────


class SkillLibrary:
    """Queryable store for learned SkillRecords with deduplication."""

    def __init__(self) -> None:
        self._skills: Dict[str, SkillRecord] = {}
        # Secondary index for deduplication: (action_tuple, env_id) -> skill_id
        self._dedup_index: Dict[Tuple[Tuple[str, ...], str], str] = {}

    # ── registration ──────────────────────────────────────────────────

    def register(self, skill: SkillRecord) -> None:
        """Add a skill or update an existing one with the same action sequence + env_id."""
        dedup_key = (tuple(skill.action_sequence), skill.env_id)

        if dedup_key in self._dedup_index:
            existing_id = self._dedup_index[dedup_key]
            existing = self._skills[existing_id]
            # Merge: keep higher expected reward, accumulate usage
            existing.expected_reward = max(existing.expected_reward, skill.expected_reward)
            existing.usage_count += skill.usage_count
            existing.trigger_tags = sorted(
                set(existing.trigger_tags) | set(skill.trigger_tags)
            )
        else:
            self._skills[skill.skill_id] = skill
            self._dedup_index[dedup_key] = skill.skill_id

    # ── querying ──────────────────────────────────────────────────────

    def query(
        self, context_tags: List[str], env_id: str
    ) -> Optional[SkillRecord]:
        """Return the most relevant skill for the current context, or None."""
        best_skill: Optional[SkillRecord] = None
        best_score: float = 0.0

        context_set = set(context_tags)

        for skill in self._skills.values():
            if skill.env_id != env_id:
                continue

            trigger_set = set(skill.trigger_tags)
            jaccard = self._jaccard(context_set, trigger_set)
            score = (
                jaccard
                * math.log(1.0 + skill.usage_count)
                * skill.expected_reward
                * skill.success_rate
            )

            if score > best_score:
                best_score = score
                best_skill = skill

        if best_score > 0.5:
            return best_skill
        return None

    # ── outcome tracking ──────────────────────────────────────────────

    def record_outcome(self, skill_id: str, reward: float, success: bool) -> None:
        """Update running averages for expected_reward and success_rate."""
        if skill_id not in self._skills:
            return

        skill = self._skills[skill_id]
        skill.usage_count += 1
        n = skill.usage_count

        # Running average for expected_reward
        skill.expected_reward = skill.expected_reward + (reward - skill.expected_reward) / n

        # Running average for success_rate
        success_val = 1.0 if success else 0.0
        skill.success_rate = skill.success_rate + (success_val - skill.success_rate) / n

    # ── introspection ─────────────────────────────────────────────────

    def all_skills(self) -> List[SkillRecord]:
        """Return all registered skills."""
        return list(self._skills.values())

    def stats(self) -> dict:
        """Summary statistics for the library."""
        skills = list(self._skills.values())
        if not skills:
            return {
                "total_skills": 0,
                "avg_success_rate": 0.0,
                "most_used": None,
            }

        avg_sr = sum(s.success_rate for s in skills) / len(skills)
        most_used = max(skills, key=lambda s: s.usage_count)

        return {
            "total_skills": len(skills),
            "avg_success_rate": round(avg_sr, 4),
            "most_used": {
                "skill_id": most_used.skill_id,
                "action_sequence": most_used.action_sequence,
                "usage_count": most_used.usage_count,
                "expected_reward": round(most_used.expected_reward, 4),
            },
        }

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0
