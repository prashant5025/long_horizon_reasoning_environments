"""
Page Index Engine — Primary V1 contribution.
Four-subsystem context retrieval: semantic chunking, inverted tag index,
state snapshots, instruction fulfillment index.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from .types import Event, InstrIndexEntry, Page, StateSnapshot


class PageIndexEngine:
    """Four interoperable subsystems for context retrieval across long episodes."""

    def __init__(self, snapshot_interval: int = 20):
        self._pages: List[Page] = []
        self._current_events: List[Event] = []
        self._current_tags: List[str] = []
        self._current_start_step: int = 0
        self._page_counter: int = 0

        # Inverted tag index: tag -> [page_id]
        self._tag_index: Dict[str, List[int]] = defaultdict(list)

        # State snapshots
        self._snapshots: List[StateSnapshot] = []
        self._snapshot_interval = snapshot_interval

        # Instruction fulfillment index
        self._instr_index: Dict[int, InstrIndexEntry] = {}

    # ─── Semantic Chunking ────────────────────────────────────────────

    def add_event(self, event: Event) -> None:
        self._current_events.append(event)
        self._current_tags.extend(event.tags)

    def close_page(self, end_step: int, summary: str = "") -> Page:
        """Close current page on meaningful state transition."""
        tags = list(set(self._current_tags))
        page = Page(
            page_id=self._page_counter,
            start_step=self._current_start_step,
            end_step=end_step,
            events=list(self._current_events),
            tags=tags,
            summary=summary,
        )
        self._pages.append(page)

        # Update inverted tag index
        for tag in tags:
            self._tag_index[tag].append(page.page_id)

        self._page_counter += 1
        self._current_events = []
        self._current_tags = []
        self._current_start_step = end_step + 1
        return page

    def should_close_page(self, event: Event, prev_phase: int, curr_phase: int) -> bool:
        """Determine if page should close based on state transitions."""
        if prev_phase != curr_phase:
            return True
        if any("shock:" in t for t in event.tags):
            return True
        if len(self._current_events) >= 50:
            return True
        return False

    # ─── Inverted Tag Index ───────────────────────────────────────────

    def get_pages_by_tag(self, tag: str) -> List[Page]:
        """O(1) lookup by context tag."""
        page_ids = self._tag_index.get(tag, [])
        return [self._pages[pid] for pid in page_ids if pid < len(self._pages)]

    def get_pages_by_tags(self, tags: List[str]) -> List[Page]:
        """Return pages matching any of the given tags, deduplicated."""
        seen = set()
        result = []
        for tag in tags:
            for pid in self._tag_index.get(tag, []):
                if pid not in seen and pid < len(self._pages):
                    seen.add(pid)
                    result.append(self._pages[pid])
        return result

    # ─── State Snapshots ──────────────────────────────────────────────

    def maybe_snapshot(self, step: int, state_digest_fn: Callable[[], Dict[str, Any]]) -> Optional[StateSnapshot]:
        """Environment writes ground-truth snapshot every N steps."""
        if step > 0 and step % self._snapshot_interval == 0:
            snap = StateSnapshot(step=step, data=state_digest_fn())
            self._snapshots.append(snap)
            return snap
        return None

    def get_latest_snapshot(self) -> Optional[StateSnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    def get_all_snapshots(self) -> List[StateSnapshot]:
        return list(self._snapshots)

    # ─── Instruction Fulfillment Index ────────────────────────────────

    def register_instruction(self, instr_id: int, deadline: int) -> None:
        self._instr_index[instr_id] = InstrIndexEntry(
            instruction_id=instr_id,
            deadline=deadline,
        )

    def mark_instruction_fulfilled(self, instr_id: int, page_id: int, step: int) -> None:
        """O(1) per lookup — makes ITTransformation tractable."""
        if instr_id in self._instr_index:
            entry = self._instr_index[instr_id]
            entry.fulfilled = True
            entry.page_id = page_id
            entry.step = step

    def get_instruction_status(self, instr_id: int) -> Optional[InstrIndexEntry]:
        return self._instr_index.get(instr_id)

    def get_unfulfilled_instructions(self) -> List[InstrIndexEntry]:
        return [e for e in self._instr_index.values() if not e.fulfilled]

    def get_overdue_instructions(self, current_step: int) -> List[InstrIndexEntry]:
        return [
            e for e in self._instr_index.values()
            if not e.fulfilled and e.deadline <= current_step
        ]

    # ─── Queries ──────────────────────────────────────────────────────

    @property
    def page_count(self) -> int:
        return len(self._pages)

    @property
    def total_events(self) -> int:
        return sum(len(p.events) for p in self._pages) + len(self._current_events)

    def stats(self) -> Dict[str, Any]:
        return {
            "pages": self.page_count,
            "total_events": self.total_events,
            "unique_tags": len(self._tag_index),
            "snapshots": len(self._snapshots),
            "instructions_tracked": len(self._instr_index),
            "instructions_fulfilled": sum(1 for e in self._instr_index.values() if e.fulfilled),
        }
