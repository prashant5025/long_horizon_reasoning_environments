"""
BaseEnvironment — Abstract base class for all BusinessHorizonENV environments.

Provides the contract that Sales, PM, and HR/IT environments must implement,
along with shared infrastructure: deterministic LCG RNG, page-index integration,
and event/page lifecycle helpers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..types import Action, ActionType, EnvID, Event, Observation
from ..page_index import PageIndexEngine


class BaseEnvironment(ABC):
    """Abstract base for all simulation environments."""

    MAX_STEPS: int = 200

    # ─── Construction ────────────────────────────────────────────────────

    def __init__(self, env_id: EnvID, max_steps: Optional[int] = None):
        self._env_id = env_id
        if max_steps is not None:
            self.MAX_STEPS = max_steps

        self.page_index = PageIndexEngine()

        # Deterministic LCG state (seeded in reset)
        self._lcg_seed: int = 0

    # ─── Properties ──────────────────────────────────────────────────────

    @property
    def env_id(self) -> EnvID:
        return self._env_id

    # ─── LCG Random Number Generator ────────────────────────────────────
    # Parameters match Numerical Recipes / MINSTD-variant used in many sims.
    # Fully deterministic given a seed — no stdlib dependency.

    _LCG_A: int = 1664525
    _LCG_C: int = 1013904223
    _LCG_M: int = 2 ** 32

    def _lcg_next(self) -> int:
        """Advance the LCG and return the raw 32-bit integer."""
        self._lcg_seed = (self._LCG_A * self._lcg_seed + self._LCG_C) % self._LCG_M
        return self._lcg_seed

    def _lcg_random(self) -> float:
        """Return a float in [0, 1) from the LCG."""
        return self._lcg_next() / self._LCG_M

    # ─── Event / Page Helpers ────────────────────────────────────────────

    @staticmethod
    def _make_event(
        step: int,
        event_type: str,
        text: str,
        tags: Optional[List[str]] = None,
        reward: float = 0.0,
        importance: float = 1.0,
    ) -> Event:
        """Construct an Event with sensible defaults."""
        return Event(
            step=step,
            event_type=event_type,
            text=text,
            tags=tags if tags is not None else [],
            reward=reward,
            importance=importance,
        )

    def _process_page_close(
        self,
        step: int,
        phase: int,
        prev_phase: int,
        event: Event,
    ) -> None:
        """Check whether the current page should close and, if so, close it.

        Delegates the decision to ``PageIndexEngine.should_close_page`` so that
        page-boundary heuristics remain in one place.
        """
        if self.page_index.should_close_page(event, prev_phase, phase):
            summary = f"phase={phase} step={step} | {event.event_type}: {event.text}"
            self.page_index.close_page(end_step=step, summary=summary)

    # ─── Abstract Interface ──────────────────────────────────────────────

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset the environment and return the initial observation.

        Parameters
        ----------
        seed : int, optional
            If provided, seeds the internal LCG for reproducible episodes.
        """

    @abstractmethod
    def step(self, action: Action) -> Observation:
        """Apply *action* and return the resulting observation."""

    @abstractmethod
    def state_digest(self) -> Dict[str, Any]:
        """Return a compact dict summarising the current internal state."""

    @abstractmethod
    def pending_goals(self) -> List[str]:
        """Return human-readable descriptions of outstanding goals."""

    @abstractmethod
    def action_space(self) -> List[ActionType]:
        """Return the action types valid in the current state."""

    @abstractmethod
    def render(self) -> str:
        """Produce a human-readable text rendering of the current state."""
