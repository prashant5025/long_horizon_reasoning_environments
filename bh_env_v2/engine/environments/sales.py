"""
EnterpriseSalesPipeline — EXTREME difficulty environment.

$2.4M enterprise deal negotiation across 11 stakeholders, 6 phases,
3 adversarial shocks, and 340 max steps.  Relationship decay forces
constant multi-stakeholder engagement while POC milestones and phase
gates must be cleared to close the deal.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..types import (
    Action,
    ActionType,
    EnvID,
    Event,
    Observation,
    SalesState,
    Severity,
    Stakeholder,
)
from ..page_index import PageIndexEngine
from .base import BaseEnvironment


# ─── Constants ───────────────────────────────────────────────────────────────

_STAKEHOLDER_DEFS: List[Dict[str, Any]] = [
    {"id": 0, "name": "Alice Chen", "role": "CEO", "engagement": 50.0},
    {"id": 1, "name": "Bob Martinez", "role": "CTO", "engagement": 50.0},
    {"id": 2, "name": "Carol Wang", "role": "CFO", "engagement": 50.0},
    {"id": 3, "name": "David Kim", "role": "VP Sales", "engagement": 50.0},
    {"id": 4, "name": "Eve Johnson", "role": "VP Engineering", "engagement": 50.0},
    {"id": 5, "name": "Frank Liu", "role": "VP Product", "engagement": 50.0},
    {"id": 6, "name": "Grace Park", "role": "Director IT", "engagement": 50.0},
    {"id": 7, "name": "Hank Davis", "role": "Director Security", "engagement": 50.0},
    {"id": 8, "name": "Irene Scott", "role": "Legal Counsel", "engagement": 50.0},
    {"id": 9, "name": "Jack Thompson", "role": "Procurement Lead", "engagement": 50.0},
    {"id": 10, "name": "Karen Adams", "role": "End User Champion", "engagement": 55.0, "is_champion": True},
]

_PHASE_NAMES: Dict[int, str] = {
    1: "Discovery",
    2: "Qualification",
    3: "Technical Evaluation",
    4: "Proposal / Negotiation",
    5: "Procurement",
    6: "Close",
}

_SHOCK_CHAMPION_STEP = 80
_SHOCK_BUDGET_FREEZE_STEP = 170
_SHOCK_COMPETITOR_STEP = 260

_BUDGET_FREEZE_DURATION = 15  # steps the freeze lasts


class EnterpriseSalesPipeline(BaseEnvironment):
    """EXTREME-difficulty enterprise sales simulation.

    Difficulty : EXTREME
    Max steps  : 340
    Deal value : $2.4 M
    """

    MAX_STEPS: int = 340

    # ─── Construction ────────────────────────────────────────────────────

    def __init__(self) -> None:
        super().__init__(env_id=EnvID.SALES, max_steps=self.MAX_STEPS)
        self._state = SalesState()
        self._budget_freeze_remaining: int = 0
        self._contacted_this_step: set[int] = set()

    # ─── Reset ───────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Observation:
        if seed is not None:
            self._lcg_seed = seed
        else:
            self._lcg_seed = 42

        self.page_index = PageIndexEngine()

        self._state = SalesState(
            step=0,
            phase=1,
            stakeholders=self._build_stakeholders(),
            poc_score=0.0,
            deal_value=2_400_000.0,
            budget_frozen=False,
            deal_closed=False,
            shocks_triggered=[],
            total_reward=0.0,
        )
        self._budget_freeze_remaining = 0
        self._contacted_this_step = set()

        intro_event = self._make_event(
            step=0,
            event_type="reset",
            text="Enterprise sales pipeline initiated. $2.4M deal, 11 stakeholders.",
            tags=["reset", "sales"],
        )
        self.page_index.add_event(intro_event)

        return Observation(
            step=0,
            phase=1,
            text="Enterprise sales pipeline initiated. 11 stakeholders identified. "
            "Target deal value: $2.4M. Begin stakeholder engagement.",
            events=[intro_event],
            reward=0.0,
            done=False,
            info=self.state_digest(),
        )

    # ─── Step ────────────────────────────────────────────────────────────

    def step(self, action: Action) -> Observation:
        s = self._state
        prev_phase = s.phase
        events: List[Event] = []
        step_reward: float = 0.0

        s.step += 1
        self._contacted_this_step = set()

        # --- relationship decay ---
        for sh in s.stakeholders:
            if sh.active:
                sh.engagement -= 2.0

        # --- disengage check ---
        for sh in s.stakeholders:
            if sh.active and sh.engagement < 20.0:
                sh.active = False
                penalty = -10.0
                step_reward += penalty
                ev = self._make_event(
                    step=s.step,
                    event_type="disengage",
                    text=f"{sh.name} ({sh.role}) disengaged (engagement={sh.engagement:.1f}).",
                    tags=["disengage", sh.role.lower().replace(" ", "_")],
                    reward=penalty,
                )
                events.append(ev)
                self.page_index.add_event(ev)

        # --- budget freeze tick ---
        if self._budget_freeze_remaining > 0:
            self._budget_freeze_remaining -= 1
            if self._budget_freeze_remaining == 0:
                s.budget_frozen = False
                ev = self._make_event(
                    step=s.step,
                    event_type="shock_end",
                    text="Budget freeze lifted.",
                    tags=["shock:budget_freeze", "resolved"],
                )
                events.append(ev)
                self.page_index.add_event(ev)

        # --- adversarial shocks ---
        shock_events, shock_reward = self._process_shocks(s)
        events.extend(shock_events)
        step_reward += shock_reward

        # --- action processing ---
        action_events, action_reward = self._process_action(action, s)
        events.extend(action_events)
        step_reward += action_reward

        # --- POC milestones ---
        poc_events, poc_reward = self._check_poc_milestones(s)
        events.extend(poc_events)
        step_reward += poc_reward

        # --- terminal check ---
        done = s.deal_closed or s.step >= self.MAX_STEPS
        if s.deal_closed:
            close_ev = self._make_event(
                step=s.step,
                event_type="deal_closed",
                text=f"Deal closed at ${s.deal_value:,.0f}! Terminal state reached.",
                tags=["terminal", "deal_closed"],
                reward=500.0,
            )
            events.append(close_ev)
            self.page_index.add_event(close_ev)
            step_reward += 500.0

        s.total_reward += step_reward

        # --- page index bookkeeping ---
        if events:
            last_ev = events[-1]
            self._process_page_close(s.step, s.phase, prev_phase, last_ev)
        self.page_index.maybe_snapshot(s.step, self.state_digest)

        # --- observation text ---
        active_count = sum(1 for sh in s.stakeholders if sh.active)
        obs_text = (
            f"Step {s.step}/{self.MAX_STEPS} | Phase {s.phase} ({_PHASE_NAMES.get(s.phase, '?')}) | "
            f"POC {s.poc_score:.0f} | Active stakeholders {active_count}/11 | "
            f"Reward this step: {step_reward:+.1f}"
        )

        return Observation(
            step=s.step,
            phase=s.phase,
            text=obs_text,
            events=events,
            reward=step_reward,
            done=done,
            info=self.state_digest(),
        )

    # ─── Action Processing ───────────────────────────────────────────────

    def _process_action(self, action: Action, s: SalesState) -> tuple[List[Event], float]:
        events: List[Event] = []
        reward: float = 0.0

        if action.action_type == ActionType.CONTACT_STAKEHOLDER:
            reward, events = self._do_contact(action, s)
        elif action.action_type == ActionType.RUN_POC:
            reward, events = self._do_run_poc(s)
        elif action.action_type == ActionType.ADVANCE_DEAL:
            reward, events = self._do_advance_deal(s)
        elif action.action_type == ActionType.RESPOND_SHOCK:
            reward, events = self._do_respond_shock(action, s)
        elif action.action_type == ActionType.REVIEW_STATUS:
            reward, events = self._do_review_status(s)
        elif action.action_type == ActionType.NOOP:
            ev = self._make_event(
                step=s.step,
                event_type="noop",
                text="No action taken this step.",
                tags=["noop"],
            )
            events.append(ev)
            self.page_index.add_event(ev)
        # unknown actions silently ignored

        return events, reward

    def _do_contact(self, action: Action, s: SalesState) -> tuple[float, List[Event]]:
        events: List[Event] = []
        target_id = action.target_id
        if target_id is None:
            ev = self._make_event(
                step=s.step,
                event_type="error",
                text="CONTACT_STAKEHOLDER requires a target_id.",
                tags=["error"],
            )
            events.append(ev)
            self.page_index.add_event(ev)
            return 0.0, events

        sh = self._find_stakeholder(s, target_id)
        if sh is None or not sh.active:
            ev = self._make_event(
                step=s.step,
                event_type="error",
                text=f"Stakeholder {target_id} not found or inactive.",
                tags=["error"],
            )
            events.append(ev)
            self.page_index.add_event(ev)
            return 0.0, events

        # engagement boost: 8-15 via LCG
        boost = 8.0 + self._lcg_random() * 7.0  # [8, 15)
        sh.engagement = min(100.0, sh.engagement + boost)
        self._contacted_this_step.add(target_id)

        ev = self._make_event(
            step=s.step,
            event_type="contact",
            text=f"Contacted {sh.name} ({sh.role}). Engagement +{boost:.1f} -> {sh.engagement:.1f}.",
            tags=["contact", sh.role.lower().replace(" ", "_")],
            reward=0.0,
        )
        events.append(ev)
        self.page_index.add_event(ev)
        return 0.0, events

    def _do_run_poc(self, s: SalesState) -> tuple[float, List[Event]]:
        events: List[Event] = []
        # POC score increase: 5-12 via LCG
        increase = 5.0 + self._lcg_random() * 7.0  # [5, 12)
        s.poc_score = min(100.0, s.poc_score + increase)

        ev = self._make_event(
            step=s.step,
            event_type="poc",
            text=f"POC executed. Score +{increase:.1f} -> {s.poc_score:.1f}.",
            tags=["poc"],
            reward=0.0,
        )
        events.append(ev)
        self.page_index.add_event(ev)
        return 0.0, events

    def _do_advance_deal(self, s: SalesState) -> tuple[float, List[Event]]:
        events: List[Event] = []

        if s.budget_frozen:
            ev = self._make_event(
                step=s.step,
                event_type="blocked",
                text="Cannot advance deal: budget is frozen.",
                tags=["blocked", "budget_freeze"],
            )
            events.append(ev)
            self.page_index.add_event(ev)
            return 0.0, events

        required_poc = s.phase * 15
        if s.poc_score < required_poc:
            ev = self._make_event(
                step=s.step,
                event_type="blocked",
                text=f"Cannot advance deal: POC score {s.poc_score:.0f} < required {required_poc}.",
                tags=["blocked", "poc_insufficient"],
            )
            events.append(ev)
            self.page_index.add_event(ev)
            return 0.0, events

        if s.phase >= 6:
            # close the deal
            s.deal_closed = True
            ev = self._make_event(
                step=s.step,
                event_type="advance",
                text="Phase 6 complete. Deal is being closed.",
                tags=["advance", "close"],
                reward=8.0,
            )
            events.append(ev)
            self.page_index.add_event(ev)
            return 8.0, events

        s.phase += 1
        reward = 8.0
        ev = self._make_event(
            step=s.step,
            event_type="advance",
            text=f"Deal advanced to Phase {s.phase} ({_PHASE_NAMES.get(s.phase, '?')}). +{reward:.0f} reward.",
            tags=["advance", f"phase_{s.phase}"],
            reward=reward,
        )
        events.append(ev)
        self.page_index.add_event(ev)
        return reward, events

    def _do_respond_shock(self, action: Action, s: SalesState) -> tuple[float, List[Event]]:
        events: List[Event] = []
        shock_name = action.params.get("shock", "")
        ev = self._make_event(
            step=s.step,
            event_type="respond_shock",
            text=f"Responding to shock: {shock_name}.",
            tags=["respond_shock", f"shock:{shock_name}"],
        )
        events.append(ev)
        self.page_index.add_event(ev)
        return 0.0, events

    def _do_review_status(self, s: SalesState) -> tuple[float, List[Event]]:
        events: List[Event] = []
        active = [sh for sh in s.stakeholders if sh.active]
        avg_eng = sum(sh.engagement for sh in active) / max(1, len(active))
        ev = self._make_event(
            step=s.step,
            event_type="review",
            text=(
                f"Status review: Phase {s.phase}, POC {s.poc_score:.0f}, "
                f"{len(active)} active stakeholders, avg engagement {avg_eng:.1f}."
            ),
            tags=["review"],
        )
        events.append(ev)
        self.page_index.add_event(ev)
        return 0.0, events

    # ─── Shocks ──────────────────────────────────────────────────────────

    def _process_shocks(self, s: SalesState) -> tuple[List[Event], float]:
        events: List[Event] = []
        reward: float = 0.0

        # Shock 1: Champion promoted (step ~80)
        if s.step == _SHOCK_CHAMPION_STEP and "champion_promoted" not in s.shocks_triggered:
            s.shocks_triggered.append("champion_promoted")
            champion = self._find_champion(s)
            if champion is not None:
                champion.active = False
                morale_hit = -30.0
                reward += morale_hit
                ev = self._make_event(
                    step=s.step,
                    event_type="shock",
                    text=(
                        f"SHOCK: {champion.name} (End User Champion) has been promoted "
                        f"out of the project. Champion removed. Morale hit: {morale_hit}."
                    ),
                    tags=["shock:champion_promoted"],
                    reward=morale_hit,
                    importance=3.0,
                )
            else:
                ev = self._make_event(
                    step=s.step,
                    event_type="shock",
                    text="SHOCK: Champion promoted, but champion was already inactive.",
                    tags=["shock:champion_promoted"],
                    importance=2.0,
                )
            events.append(ev)
            self.page_index.add_event(ev)

        # Shock 2: Budget freeze (step ~170)
        if s.step == _SHOCK_BUDGET_FREEZE_STEP and "budget_freeze" not in s.shocks_triggered:
            s.shocks_triggered.append("budget_freeze")
            s.budget_frozen = True
            self._budget_freeze_remaining = _BUDGET_FREEZE_DURATION
            ev = self._make_event(
                step=s.step,
                event_type="shock",
                text=(
                    f"SHOCK: Company-wide budget freeze! ADVANCE_DEAL blocked for "
                    f"{_BUDGET_FREEZE_DURATION} steps."
                ),
                tags=["shock:budget_freeze"],
                importance=3.0,
            )
            events.append(ev)
            self.page_index.add_event(ev)

        # Shock 3: Competitor threat (step ~260)
        if s.step == _SHOCK_COMPETITOR_STEP and "competitor_threat" not in s.shocks_triggered:
            s.shocks_triggered.append("competitor_threat")
            for sh in s.stakeholders:
                if sh.active:
                    sh.engagement = max(0.0, sh.engagement - 15.0)
            ev = self._make_event(
                step=s.step,
                event_type="shock",
                text="SHOCK: Competitor threat! All stakeholder engagement drops by 15.",
                tags=["shock:competitor_threat"],
                importance=3.0,
            )
            events.append(ev)
            self.page_index.add_event(ev)

        return events, reward

    # ─── POC Milestones ──────────────────────────────────────────────────

    def _check_poc_milestones(self, s: SalesState) -> tuple[List[Event], float]:
        events: List[Event] = []
        reward: float = 0.0

        # Check milestones only on the step they are first crossed
        poc_key_60 = "poc_milestone_60"
        poc_key_80 = "poc_milestone_80"

        if s.poc_score >= 80.0 and poc_key_80 not in s.shocks_triggered:
            s.shocks_triggered.append(poc_key_80)
            # Also mark 60 as triggered if not already
            if poc_key_60 not in s.shocks_triggered:
                s.shocks_triggered.append(poc_key_60)
            reward += 30.0
            ev = self._make_event(
                step=s.step,
                event_type="milestone",
                text=f"POC milestone: score >= 80 reached ({s.poc_score:.0f}). +30 bonus.",
                tags=["milestone", "poc_80"],
                reward=30.0,
            )
            events.append(ev)
            self.page_index.add_event(ev)
        elif s.poc_score >= 60.0 and poc_key_60 not in s.shocks_triggered:
            s.shocks_triggered.append(poc_key_60)
            reward += 15.0
            ev = self._make_event(
                step=s.step,
                event_type="milestone",
                text=f"POC milestone: score >= 60 reached ({s.poc_score:.0f}). +15 bonus.",
                tags=["milestone", "poc_60"],
                reward=15.0,
            )
            events.append(ev)
            self.page_index.add_event(ev)

        return events, reward

    # ─── Abstract Implementations ────────────────────────────────────────

    def state_digest(self) -> Dict[str, Any]:
        return self._state.digest()

    def action_space(self) -> List[ActionType]:
        s = self._state
        actions: List[ActionType] = [
            ActionType.REVIEW_STATUS,
            ActionType.NOOP,
        ]

        # CONTACT_STAKEHOLDER if any active stakeholders
        if any(sh.active for sh in s.stakeholders):
            actions.append(ActionType.CONTACT_STAKEHOLDER)

        # RUN_POC always available (capped at 100 internally)
        actions.append(ActionType.RUN_POC)

        # ADVANCE_DEAL available if not budget frozen and poc meets threshold
        if not s.budget_frozen and s.poc_score >= s.phase * 15:
            actions.append(ActionType.ADVANCE_DEAL)

        # RESPOND_SHOCK if any shocks active
        if s.budget_frozen or (s.step >= _SHOCK_CHAMPION_STEP and "champion_promoted" in s.shocks_triggered):
            actions.append(ActionType.RESPOND_SHOCK)

        return actions

    def render(self) -> str:
        s = self._state
        lines = [
            "=" * 60,
            "  ENTERPRISE SALES PIPELINE",
            "=" * 60,
            f"  Step       : {s.step} / {self.MAX_STEPS}",
            f"  Phase      : {s.phase} ({_PHASE_NAMES.get(s.phase, '?')})",
            f"  POC Score  : {s.poc_score:.1f}",
            f"  Deal Value : ${s.deal_value:,.0f}",
            f"  Deal Closed: {s.deal_closed}",
            f"  Budget Frozen: {s.budget_frozen}",
            f"  Total Reward : {s.total_reward:.1f}",
            "",
            "  Stakeholders:",
        ]
        for sh in s.stakeholders:
            status = "ACTIVE" if sh.active else "INACTIVE"
            champ = " [CHAMPION]" if sh.is_champion else ""
            lines.append(
                f"    [{sh.id:2d}] {sh.name:<20s} {sh.role:<22s} "
                f"eng={sh.engagement:5.1f}  {status}{champ}"
            )
        lines.append("")
        lines.append(f"  Shocks triggered: {', '.join(s.shocks_triggered) or 'none'}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def pending_goals(self) -> List[str]:
        s = self._state
        goals: List[str] = []

        if not s.deal_closed:
            goals.append(f"Close the ${s.deal_value:,.0f} deal (current phase: {s.phase}/6).")

        required_poc = s.phase * 15
        if s.poc_score < required_poc:
            goals.append(
                f"Raise POC score to {required_poc} (currently {s.poc_score:.0f}) "
                f"to unlock phase advancement."
            )

        disengaged = [sh for sh in s.stakeholders if not sh.active]
        if disengaged:
            goals.append(
                f"Lost {len(disengaged)} stakeholder(s) to disengagement — "
                f"maintain relationships to prevent further losses."
            )

        low_eng = [sh for sh in s.stakeholders if sh.active and sh.engagement < 35.0]
        if low_eng:
            names = ", ".join(sh.name for sh in low_eng)
            goals.append(f"Urgent: re-engage low-engagement stakeholders: {names}.")

        if s.budget_frozen:
            goals.append(
                f"Budget freeze active ({self._budget_freeze_remaining} steps remaining). "
                f"Deal advancement blocked."
            )

        if "champion_promoted" in s.shocks_triggered:
            goals.append("Champion has been promoted — rebuild advocacy network.")

        if "competitor_threat" in s.shocks_triggered:
            goals.append("Competitor threat detected — reinforce stakeholder engagement.")

        if not goals:
            goals.append("All objectives met.")

        return goals

    # ─── Internal Helpers ────────────────────────────────────────────────

    @staticmethod
    def _build_stakeholders() -> List[Stakeholder]:
        return [
            Stakeholder(
                id=d["id"],
                name=d["name"],
                role=d["role"],
                engagement=d.get("engagement", 50.0),
                is_champion=d.get("is_champion", False),
            )
            for d in _STAKEHOLDER_DEFS
        ]

    @staticmethod
    def _find_stakeholder(s: SalesState, sid: int) -> Optional[Stakeholder]:
        for sh in s.stakeholders:
            if sh.id == sid:
                return sh
        return None

    @staticmethod
    def _find_champion(s: SalesState) -> Optional[Stakeholder]:
        for sh in s.stakeholders:
            if sh.is_champion and sh.active:
                return sh
        return None
