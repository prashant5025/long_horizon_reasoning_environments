"""
ProgramRescue Environment (PM) -- EXTREME difficulty.

4 workstreams, 47 risk items, 3 shocks, 420 max steps.
Simulates a multi-workstream program rescue where the agent must manage
risks, morale, budget, and workstream progress across 5 delivery phases.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..types import (
    PMState,
    Workstream,
    Risk,
    ActionType,
    EnvID,
    Action,
    Observation,
    Event,
    Severity,
)
from .base import BaseEnvironment


# ── Constants ───────────────────────────────────────────────────────────────

MAX_STEPS = 420

_WORKSTREAM_DEFS = [
    {"name": "Backend Migration",  "dependencies": []},
    {"name": "Frontend Rewrite",   "dependencies": [0]},
    {"name": "Data Pipeline",      "dependencies": [0]},
    {"name": "QA Automation",      "dependencies": [0, 1]},
]

_RISK_DESCRIPTIONS = [
    "Legacy API contract mismatch",
    "Insufficient test coverage on auth module",
    "Database migration rollback plan missing",
    "CI/CD pipeline timeout on large builds",
    "Third-party SDK license compliance gap",
    "Container orchestration memory limits",
    "Cross-service authentication token expiry",
    "Unpatched CVE in dependency tree",
    "Load balancer health-check misconfiguration",
    "Stale feature flags in production",
    "Monitoring blind-spot in payment service",
    "Data retention policy non-compliance",
    "Rate limiter configuration drift",
    "Secrets rotation overdue",
    "Schema migration lacks idempotency guard",
    "Frontend bundle size regression",
    "Accessibility audit findings unresolved",
    "Mobile breakpoint layout regression",
    "Internationalisation string extraction incomplete",
    "WebSocket reconnection logic fragile",
    "Cache invalidation race condition",
    "Event sourcing replay gap",
    "Search index replication lag",
    "File upload size limit inconsistency",
    "GraphQL N+1 query pattern detected",
    "Background job retry storm risk",
    "Message queue dead-letter backlog",
    "Service mesh mTLS certificate rotation",
    "Terraform state lock contention",
    "Deployment canary metric threshold too lax",
    "Disaster recovery RTO exceeds SLA",
    "PII logging in debug mode",
    "Vendor API deprecation approaching",
    "DNS TTL too long for failover",
    "Feature toggle evaluation performance",
    "Distributed tracing context propagation gap",
    "Horizontal pod autoscaler flapping",
    "Object storage lifecycle policy missing",
    "Webhook delivery retry budget exhausted",
    "Multi-region consistency model undefined",
    "ETL job SLA at risk from upstream delay",
    "Streaming pipeline checkpoint lag",
    "ML model serving cold start latency",
    "A/B test statistical power insufficient",
    "Compliance audit evidence collection gap",
    "Penetration test finding remediation overdue",
    "Capacity forecast model stale",
]

_SEVERITY_CYCLE = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]

# Phase thresholds: average workstream progress required to advance phase.
_PHASE_THRESHOLDS = [0.0, 20.0, 40.0, 60.0, 85.0]  # phases 1-5

_SHOCK_ARCHITECT_STEP = 100
_SHOCK_VENDOR_STEP = 220
_SHOCK_SCOPE_STEP = 340

_VENDOR_BLOCK_DURATION = 30

_MORALE_DECAY_RATE = 0.05
_BUDGET_INITIAL = 6_000_000.0


class ProgramRescueEnvironment(BaseEnvironment):
    """EXTREME-difficulty programme rescue simulation.

    The agent must balance risk resolution, workstream advancement,
    budget allocation, and team morale across 420 steps while surviving
    three scripted shock events.
    """

    MAX_STEPS = MAX_STEPS

    # ── Construction ────────────────────────────────────────────────────

    def __init__(self) -> None:
        super().__init__(env_id=EnvID.PM, max_steps=MAX_STEPS)
        self._state = PMState()
        self._vendor_blocked_remaining: int = 0

    # ── Public Interface ────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Observation:
        if seed is not None:
            self._lcg_seed = seed
        else:
            self._lcg_seed = 42

        self.page_index = type(self.page_index)(snapshot_interval=20)

        # Initialise state --------------------------------------------------
        self._state = PMState(
            step=0,
            phase=1,
            team_morale=80.0,
            budget=_BUDGET_INITIAL,
            budget_initial=_BUDGET_INITIAL,
            program_delivered=False,
            shocks_triggered=[],
            total_reward=0.0,
        )

        # Build workstreams
        self._state.workstreams = [
            Workstream(
                id=i,
                name=d["name"],
                progress=0.0,
                blocked=False,
                dependencies=list(d["dependencies"]),
            )
            for i, d in enumerate(_WORKSTREAM_DEFS)
        ]

        # Generate 47 risks deterministically via LCG ----------------------
        self._state.risks = []
        for rid in range(47):
            sev_idx = self._lcg_next() % 4
            ws_id = self._lcg_next() % 4
            self._state.risks.append(
                Risk(
                    id=rid,
                    description=_RISK_DESCRIPTIONS[rid],
                    severity=_SEVERITY_CYCLE[sev_idx],
                    resolved=False,
                    workstream_id=ws_id,
                    step_created=0,
                )
            )

        self._vendor_blocked_remaining = 0

        # Opening page / event
        evt = self._make_event(
            step=0,
            event_type="reset",
            text="ProgramRescue environment initialised. 4 workstreams, 47 risks, $6M budget.",
            tags=["reset", "pm"],
        )
        self.page_index.add_event(evt)

        return Observation(
            step=0,
            phase=1,
            text=(
                "Program rescue initiated. You manage 4 workstreams, 47 risks, "
                "and a $6M budget over 420 steps. Team morale is 80. "
                "Deliver the programme before time runs out."
            ),
            events=[evt],
            reward=0.0,
            done=False,
            info=self.state_digest(),
        )

    def step(self, action: Action) -> Observation:
        s = self._state
        prev_phase = s.phase
        events: List[Event] = []
        step_reward = 0.0

        s.step += 1

        # ── Morale decay ─────────────────────────────────────────────
        s.team_morale = max(0.0, s.team_morale - _MORALE_DECAY_RATE)

        # ── Vendor block countdown ───────────────────────────────────
        if self._vendor_blocked_remaining > 0:
            self._vendor_blocked_remaining -= 1
            if self._vendor_blocked_remaining == 0:
                ws2 = s.workstreams[2]
                ws2.blocked = False
                evt = self._make_event(
                    step=s.step,
                    event_type="vendor_unblock",
                    text="Data Pipeline workstream unblocked after vendor replacement.",
                    tags=["vendor", "unblock"],
                )
                events.append(evt)
                self.page_index.add_event(evt)

        # ── Shock processing ─────────────────────────────────────────
        shock_events = self._process_shocks(s.step)
        for se in shock_events:
            if se.reward != 0.0:
                step_reward += se.reward
            events.append(se)
            self.page_index.add_event(se)

        # ── Action processing ────────────────────────────────────────
        action_reward, action_events = self._process_action(action)
        step_reward += action_reward
        events.extend(action_events)
        for ae in action_events:
            self.page_index.add_event(ae)

        # ── Phase check ──────────────────────────────────────────────
        new_phase = self._compute_phase()
        if new_phase != s.phase:
            phase_evt = self._make_event(
                step=s.step,
                event_type="phase_milestone",
                text=f"Programme advanced to phase {new_phase}.",
                tags=["phase", f"phase:{new_phase}"],
                reward=25.0,
            )
            step_reward += 25.0
            events.append(phase_evt)
            self.page_index.add_event(phase_evt)
            s.phase = new_phase

        # ── Terminal check ───────────────────────────────────────────
        done = False
        if self._check_program_delivered():
            s.program_delivered = True
            done = True
            step_reward += 600.0
            term_evt = self._make_event(
                step=s.step,
                event_type="program_delivered",
                text="Programme successfully delivered!",
                tags=["terminal", "delivered"],
                reward=600.0,
                importance=5.0,
            )
            events.append(term_evt)
            self.page_index.add_event(term_evt)

        if s.step >= self.MAX_STEPS and not done:
            done = True
            timeout_evt = self._make_event(
                step=s.step,
                event_type="timeout",
                text="Programme ran out of time. Delivery failed.",
                tags=["terminal", "timeout"],
                importance=5.0,
            )
            events.append(timeout_evt)
            self.page_index.add_event(timeout_evt)

        if s.budget <= 0 and not done:
            done = True
            budget_evt = self._make_event(
                step=s.step,
                event_type="budget_exhausted",
                text="Budget exhausted. Programme cancelled.",
                tags=["terminal", "budget"],
                importance=5.0,
            )
            events.append(budget_evt)
            self.page_index.add_event(budget_evt)

        if s.team_morale <= 0 and not done:
            done = True
            morale_evt = self._make_event(
                step=s.step,
                event_type="morale_collapse",
                text="Team morale collapsed. Programme abandoned.",
                tags=["terminal", "morale"],
                importance=5.0,
            )
            events.append(morale_evt)
            self.page_index.add_event(morale_evt)

        # ── Bookkeeping ──────────────────────────────────────────────
        s.total_reward += step_reward

        # Page-close check
        for evt in events:
            self._process_page_close(s.step, s.phase, prev_phase, evt)

        # Snapshot
        self.page_index.maybe_snapshot(s.step, self.state_digest)

        obs_text = self._build_observation_text(action, events, step_reward)
        return Observation(
            step=s.step,
            phase=s.phase,
            text=obs_text,
            events=events,
            reward=step_reward,
            done=done,
            info=self.state_digest(),
            tags=[e.event_type for e in events],
        )

    def state_digest(self) -> Dict[str, Any]:
        return self._state.digest()

    def pending_goals(self) -> List[str]:
        goals: List[str] = []
        s = self._state

        unresolved = sum(1 for r in s.risks if not r.resolved)
        if unresolved > 0:
            goals.append(f"Resolve {unresolved} outstanding risks")

        for ws in s.workstreams:
            if ws.progress < 100.0:
                goals.append(f"Advance '{ws.name}' to 100% (currently {ws.progress:.1f}%)")

        if s.team_morale < 50:
            goals.append(f"Restore team morale (currently {s.team_morale:.1f})")

        if not s.program_delivered:
            goals.append("Deliver the programme before step 420")

        return goals

    def action_space(self) -> List[ActionType]:
        return [
            ActionType.RESOLVE_RISK,
            ActionType.ADVANCE_WORKSTREAM,
            ActionType.ALLOCATE_BUDGET,
            ActionType.BOOST_MORALE,
            ActionType.RESPOND_SHOCK,
            ActionType.REVIEW_STATUS,
            ActionType.NOOP,
        ]

    def render(self) -> str:
        s = self._state
        lines = [
            f"=== ProgramRescue  Step {s.step}/{self.MAX_STEPS}  Phase {s.phase}/5 ===",
            f"Morale: {s.team_morale:.1f}   Budget: ${s.budget:,.0f}   Reward: {s.total_reward:.1f}",
            "",
            "Workstreams:",
        ]
        for ws in s.workstreams:
            status = "BLOCKED" if ws.blocked else "active"
            dep_str = ", ".join(str(d) for d in ws.dependencies) if ws.dependencies else "none"
            lines.append(
                f"  [{ws.id}] {ws.name:25s} {ws.progress:5.1f}%  ({status})  deps=[{dep_str}]"
            )

        resolved = sum(1 for r in s.risks if r.resolved)
        total = len(s.risks)
        crit = sum(1 for r in s.risks if not r.resolved and r.severity == Severity.CRITICAL)
        high = sum(1 for r in s.risks if not r.resolved and r.severity == Severity.HIGH)
        lines.append("")
        lines.append(f"Risks: {resolved}/{total} resolved  ({crit} CRITICAL, {high} HIGH open)")

        if s.shocks_triggered:
            lines.append(f"Shocks triggered: {', '.join(s.shocks_triggered)}")

        return "\n".join(lines)

    # ── Shock Processing ────────────────────────────────────────────────

    def _process_shocks(self, step: int) -> List[Event]:
        s = self._state
        events: List[Event] = []

        # Shock 1: Architect departure (~step 100)
        if step == _SHOCK_ARCHITECT_STEP and "architect_departure" not in s.shocks_triggered:
            s.shocks_triggered.append("architect_departure")
            s.team_morale = max(0.0, s.team_morale - 25.0)
            ws0 = s.workstreams[0]
            ws0.progress = max(0.0, ws0.progress * 0.80)
            evt = self._make_event(
                step=step,
                event_type="shock",
                text=(
                    "SHOCK: Lead architect has departed. Team morale drops by 25. "
                    "Backend Migration progress reduced by 20%."
                ),
                tags=["shock:architect_departure", "morale", "workstream:0"],
                importance=4.0,
            )
            events.append(evt)

        # Shock 2: Vendor bankruptcy (~step 220)
        if step == _SHOCK_VENDOR_STEP and "vendor_bankrupt" not in s.shocks_triggered:
            s.shocks_triggered.append("vendor_bankrupt")
            ws2 = s.workstreams[2]
            ws2.blocked = True
            self._vendor_blocked_remaining = _VENDOR_BLOCK_DURATION
            evt = self._make_event(
                step=step,
                event_type="shock",
                text=(
                    "SHOCK: Critical vendor has gone bankrupt. "
                    f"Data Pipeline workstream blocked for {_VENDOR_BLOCK_DURATION} steps. "
                    "Immediate budget impact: -$20K equivalent reward penalty."
                ),
                tags=["shock:vendor_bankrupt", "workstream:2", "budget"],
                reward=-20.0,
                importance=4.0,
            )
            events.append(evt)

        # Shock 3: Scope creep (~step 340)
        if step == _SHOCK_SCOPE_STEP and "scope_creep" not in s.shocks_triggered:
            s.shocks_triggered.append("scope_creep")
            s.budget = max(0.0, s.budget - 500_000.0)
            # Add 15 new risks
            base_id = len(s.risks)
            for i in range(15):
                sev_idx = self._lcg_next() % 4
                ws_id = self._lcg_next() % 4
                s.risks.append(
                    Risk(
                        id=base_id + i,
                        description=f"Scope-creep risk #{i+1}: new requirement injection",
                        severity=_SEVERITY_CYCLE[sev_idx],
                        resolved=False,
                        workstream_id=ws_id,
                        step_created=step,
                    )
                )
            evt = self._make_event(
                step=step,
                event_type="shock",
                text=(
                    "SHOCK: Major scope creep detected. 15 new risks injected. "
                    "Budget reduced by $500K."
                ),
                tags=["shock:scope_creep", "risk", "budget"],
                importance=4.0,
            )
            events.append(evt)

        return events

    # ── Action Processing ───────────────────────────────────────────────

    def _process_action(self, action: Action) -> tuple[float, List[Event]]:
        s = self._state
        reward = 0.0
        events: List[Event] = []

        at = action.action_type

        if at == ActionType.RESOLVE_RISK:
            reward, events = self._do_resolve_risk(action)
        elif at == ActionType.ADVANCE_WORKSTREAM:
            reward, events = self._do_advance_workstream(action)
        elif at == ActionType.ALLOCATE_BUDGET:
            reward, events = self._do_allocate_budget(action)
        elif at == ActionType.BOOST_MORALE:
            reward, events = self._do_boost_morale(action)
        elif at == ActionType.RESPOND_SHOCK:
            reward, events = self._do_respond_shock(action)
        elif at == ActionType.REVIEW_STATUS:
            reward, events = self._do_review_status()
        elif at == ActionType.NOOP:
            evt = self._make_event(
                step=s.step,
                event_type="noop",
                text="No action taken this step.",
                tags=["noop"],
            )
            events.append(evt)
        else:
            evt = self._make_event(
                step=s.step,
                event_type="invalid_action",
                text=f"Action {at.name} is not valid in the PM environment.",
                tags=["invalid"],
            )
            events.append(evt)

        return reward, events

    # ── Individual Action Handlers ──────────────────────────────────────

    def _do_resolve_risk(self, action: Action) -> tuple[float, List[Event]]:
        s = self._state
        target = action.target_id
        if target is None:
            # Resolve first unresolved risk
            for r in s.risks:
                if not r.resolved:
                    target = r.id
                    break
        if target is None:
            evt = self._make_event(
                step=s.step,
                event_type="resolve_risk_fail",
                text="No unresolved risks remaining.",
                tags=["risk"],
            )
            return 0.0, [evt]

        risk = self._find_risk(target)
        if risk is None or risk.resolved:
            evt = self._make_event(
                step=s.step,
                event_type="resolve_risk_fail",
                text=f"Risk {target} not found or already resolved.",
                tags=["risk"],
            )
            return 0.0, [evt]

        risk.resolved = True
        severity_rewards = {
            Severity.CRITICAL: 15.0,
            Severity.HIGH: 10.0,
            Severity.MEDIUM: 5.0,
            Severity.LOW: 2.0,
        }
        reward = severity_rewards.get(risk.severity, 2.0)

        evt = self._make_event(
            step=s.step,
            event_type="resolve_risk",
            text=f"Resolved {risk.severity.value} risk: {risk.description}",
            tags=["risk", f"severity:{risk.severity.value}", f"workstream:{risk.workstream_id}"],
            reward=reward,
        )
        return reward, [evt]

    def _do_advance_workstream(self, action: Action) -> tuple[float, List[Event]]:
        s = self._state
        ws_id = action.target_id
        if ws_id is None:
            ws_id = 0

        if ws_id < 0 or ws_id >= len(s.workstreams):
            evt = self._make_event(
                step=s.step,
                event_type="advance_fail",
                text=f"Invalid workstream id {ws_id}.",
                tags=["workstream", "invalid"],
            )
            return 0.0, [evt]

        ws = s.workstreams[ws_id]

        if ws.blocked:
            evt = self._make_event(
                step=s.step,
                event_type="advance_fail",
                text=f"Workstream '{ws.name}' is blocked.",
                tags=["workstream", "blocked", f"workstream:{ws_id}"],
            )
            return 0.0, [evt]

        # Check dependencies: all dependencies must have progress >= 20%
        for dep_id in ws.dependencies:
            dep_ws = s.workstreams[dep_id]
            if dep_ws.progress < 20.0:
                evt = self._make_event(
                    step=s.step,
                    event_type="advance_fail",
                    text=(
                        f"Cannot advance '{ws.name}': dependency '{dep_ws.name}' "
                        f"only at {dep_ws.progress:.1f}% (need 20%)."
                    ),
                    tags=["workstream", "dependency", f"workstream:{ws_id}"],
                )
                return 0.0, [evt]

        # Progress 3-8% based on LCG
        advance = 3.0 + (self._lcg_next() % 6)  # 3..8
        # Morale multiplier: scale by morale/100
        morale_factor = max(0.3, s.team_morale / 100.0)
        advance *= morale_factor

        old_progress = ws.progress
        ws.progress = min(100.0, ws.progress + advance)
        actual_advance = ws.progress - old_progress

        reward = actual_advance * 0.1

        evt = self._make_event(
            step=s.step,
            event_type="advance_workstream",
            text=(
                f"Advanced '{ws.name}' by {actual_advance:.1f}% "
                f"(now {ws.progress:.1f}%)."
            ),
            tags=["workstream", f"workstream:{ws_id}"],
            reward=reward,
        )
        return reward, [evt]

    def _do_allocate_budget(self, action: Action) -> tuple[float, List[Event]]:
        s = self._state
        amount = action.params.get("amount", 100_000.0)
        ws_id = action.target_id
        if ws_id is None:
            ws_id = 0

        if ws_id < 0 or ws_id >= len(s.workstreams):
            evt = self._make_event(
                step=s.step,
                event_type="budget_fail",
                text=f"Invalid workstream id {ws_id}.",
                tags=["budget", "invalid"],
            )
            return 0.0, [evt]

        amount = min(amount, s.budget)
        if amount <= 0:
            evt = self._make_event(
                step=s.step,
                event_type="budget_fail",
                text="No budget remaining to allocate.",
                tags=["budget"],
            )
            return 0.0, [evt]

        ws = s.workstreams[ws_id]
        if ws.blocked:
            evt = self._make_event(
                step=s.step,
                event_type="budget_fail",
                text=f"Cannot allocate budget to blocked workstream '{ws.name}'.",
                tags=["budget", "blocked", f"workstream:{ws_id}"],
            )
            return 0.0, [evt]

        s.budget -= amount
        # Budget boost: every $100K gives 2% progress
        boost = (amount / 100_000.0) * 2.0
        old_progress = ws.progress
        ws.progress = min(100.0, ws.progress + boost)
        actual_boost = ws.progress - old_progress

        reward = actual_boost * 0.1

        evt = self._make_event(
            step=s.step,
            event_type="allocate_budget",
            text=(
                f"Allocated ${amount:,.0f} to '{ws.name}'. "
                f"Progress boosted by {actual_boost:.1f}% (now {ws.progress:.1f}%). "
                f"Budget remaining: ${s.budget:,.0f}."
            ),
            tags=["budget", f"workstream:{ws_id}"],
            reward=reward,
        )
        return reward, [evt]

    def _do_boost_morale(self, action: Action) -> tuple[float, List[Event]]:
        s = self._state
        # Boost is 5-10 based on LCG
        boost = 5.0 + (self._lcg_next() % 6)  # 5..10
        old_morale = s.team_morale
        s.team_morale = min(100.0, s.team_morale + boost)
        actual_boost = s.team_morale - old_morale

        # Small budget cost for morale events
        morale_cost = 25_000.0
        s.budget = max(0.0, s.budget - morale_cost)

        reward = actual_boost * 0.2

        evt = self._make_event(
            step=s.step,
            event_type="boost_morale",
            text=(
                f"Team-building event boosted morale by {actual_boost:.1f} "
                f"(now {s.team_morale:.1f}). Cost: ${morale_cost:,.0f}."
            ),
            tags=["morale"],
            reward=reward,
        )
        return reward, [evt]

    def _do_respond_shock(self, action: Action) -> tuple[float, List[Event]]:
        s = self._state

        if not s.shocks_triggered:
            evt = self._make_event(
                step=s.step,
                event_type="respond_shock_fail",
                text="No shocks to respond to.",
                tags=["shock"],
            )
            return 0.0, [evt]

        latest_shock = s.shocks_triggered[-1]
        reward = 0.0
        text = ""

        if latest_shock == "architect_departure":
            # Partial morale recovery and progress restoration
            s.team_morale = min(100.0, s.team_morale + 10.0)
            s.workstreams[0].progress = min(100.0, s.workstreams[0].progress + 5.0)
            reward = 8.0
            text = (
                "Responded to architect departure: hired interim lead. "
                "Morale +10, Backend Migration +5%."
            )
        elif latest_shock == "vendor_bankrupt":
            # Reduce block duration
            reduced = min(10, self._vendor_blocked_remaining)
            self._vendor_blocked_remaining = max(0, self._vendor_blocked_remaining - reduced)
            if self._vendor_blocked_remaining == 0:
                s.workstreams[2].blocked = False
            reward = 10.0
            text = (
                f"Responded to vendor bankruptcy: expedited replacement. "
                f"Block reduced by {reduced} steps."
            )
        elif latest_shock == "scope_creep":
            # Resolve a few of the new risks automatically
            resolved_count = 0
            for r in s.risks:
                if not r.resolved and r.step_created == _SHOCK_SCOPE_STEP and resolved_count < 5:
                    r.resolved = True
                    resolved_count += 1
            reward = 12.0
            text = (
                f"Responded to scope creep: triaged and resolved {resolved_count} "
                "of the injected risks."
            )
        else:
            text = f"Acknowledged shock: {latest_shock}."
            reward = 2.0

        evt = self._make_event(
            step=s.step,
            event_type="respond_shock",
            text=text,
            tags=["shock", f"shock:{latest_shock}"],
            reward=reward,
        )
        return reward, [evt]

    def _do_review_status(self) -> tuple[float, List[Event]]:
        s = self._state
        resolved = sum(1 for r in s.risks if r.resolved)
        total = len(s.risks)
        avg_progress = sum(w.progress for w in s.workstreams) / max(1, len(s.workstreams))

        text = (
            f"Status review: Phase {s.phase}/5, "
            f"Risks {resolved}/{total} resolved, "
            f"Avg progress {avg_progress:.1f}%, "
            f"Morale {s.team_morale:.1f}, "
            f"Budget ${s.budget:,.0f}."
        )
        evt = self._make_event(
            step=s.step,
            event_type="review_status",
            text=text,
            tags=["review"],
        )
        # Small reward for situational awareness
        return 0.5, [evt]

    # ── Internal Helpers ────────────────────────────────────────────────

    def _find_risk(self, risk_id: int) -> Optional[Risk]:
        for r in self._state.risks:
            if r.id == risk_id:
                return r
        return None

    def _compute_phase(self) -> int:
        avg_progress = sum(w.progress for w in self._state.workstreams) / max(
            1, len(self._state.workstreams)
        )
        phase = 1
        for i, threshold in enumerate(_PHASE_THRESHOLDS):
            if avg_progress >= threshold:
                phase = i + 1
        return min(phase, 5)

    def _check_program_delivered(self) -> bool:
        """Programme is delivered when all workstreams reach 100%."""
        return all(ws.progress >= 100.0 for ws in self._state.workstreams)

    def _build_observation_text(
        self, action: Action, events: List[Event], reward: float
    ) -> str:
        s = self._state
        parts = [
            f"Step {s.step}/{self.MAX_STEPS} | Phase {s.phase}/5 | "
            f"Morale {s.team_morale:.1f} | Budget ${s.budget:,.0f} | "
            f"Reward this step: {reward:.1f}"
        ]
        for evt in events:
            parts.append(f"  - {evt.text}")
        return "\n".join(parts)
