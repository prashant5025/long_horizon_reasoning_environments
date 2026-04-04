"""
ITTransformation Environment — LEGENDARY difficulty.

300 compliance instructions with individual deadlines, 8000-user migration,
ransomware shock at step ~240, 480 max steps.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..types import (
    HRITState,
    Instruction,
    ActionType,
    EnvID,
    Action,
    Observation,
    Event,
    Severity,
)
from .base import BaseEnvironment


# ─── Instruction-generation helpers ─────────────────────────────────────────

_CATEGORIES = ["compliance", "security", "migration", "training", "documentation"]

_TEMPLATES: Dict[str, List[str]] = {
    "compliance": [
        "Update data-retention policy for {region} region",
        "Audit access logs for {system} system",
        "Submit SOX compliance attestation for Q{q}",
        "Review GDPR consent records batch {n}",
        "Validate ISO-27001 control {ctrl}",
    ],
    "security": [
        "Rotate API keys for {system} service",
        "Patch CVE-{cve} across production fleet",
        "Conduct phishing-simulation round {n}",
        "Review firewall rules for DMZ segment {seg}",
        "Update MFA enrollment for department {dept}",
    ],
    "migration": [
        "Prepare migration runbook for cohort {n}",
        "Validate AD sync for OU {ou}",
        "Migrate shared-drive data for floor {floor}",
        "Test SSO integration for app {app}",
        "Decommission legacy mailbox batch {n}",
    ],
    "training": [
        "Schedule security-awareness session {n}",
        "Deliver new-system onboarding for team {team}",
        "Create FAQ document for migration wave {n}",
        "Run tabletop DR exercise {n}",
        "Certify admin training for module {mod}",
    ],
    "documentation": [
        "Draft change-advisory-board report {n}",
        "Update network topology diagram rev {rev}",
        "Publish runbook for incident type {itype}",
        "Archive legacy KB articles batch {n}",
        "Write post-mortem for outage {n}",
    ],
}


class ITTransformationEnv(BaseEnvironment):
    """
    LEGENDARY IT-Transformation environment.

    * 300 compliance instructions with individual deadlines
    * 8 000-user platform migration in cohorts of ~200
    * Ransomware attack shock at step ~240
    * 480 maximum steps
    """

    MAX_STEPS: int = 480

    # Migration constants
    _TOTAL_USERS: int = 8000
    _COHORT_SIZE: int = 200
    _MIGRATION_PHASES: int = 4  # each 25 % of total users

    # Shock timing
    _RANSOMWARE_STEP: int = 240
    _RANSOMWARE_FREEZE_STEPS: int = 72

    # SLA thresholds
    _SLA_DECAY_THRESHOLD: int = 10
    _SLA_DECAY_RATE: float = 0.5
    _SLA_BREACH_LEVEL: float = 95.0
    _SLA_BREACH_PENALTY: float = 3.0

    # ─── Construction ────────────────────────────────────────────────────

    def __init__(self) -> None:
        super().__init__(env_id=EnvID.HR_IT, max_steps=self.MAX_STEPS)
        self._state = HRITState()
        self._migration_frozen_until: int = 0

    # ─── Reset ───────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Observation:
        self._lcg_seed = seed if seed is not None else 42
        self.page_index = type(self.page_index)()  # fresh PageIndexEngine

        self._state = HRITState(total_users=self._TOTAL_USERS)
        self._migration_frozen_until = 0

        # Generate 300 instructions with LCG
        instructions: List[Instruction] = []
        for i in range(300):
            cat = _CATEGORIES[self._lcg_next() % len(_CATEGORIES)]
            templates = _TEMPLATES[cat]
            tmpl = templates[self._lcg_next() % len(templates)]

            # Fill placeholder tokens deterministically
            text = tmpl.format(
                region=f"R{self._lcg_next() % 12 + 1}",
                system=f"SYS-{self._lcg_next() % 50 + 1:03d}",
                q=(self._lcg_next() % 4) + 1,
                n=i + 1,
                ctrl=f"A.{self._lcg_next() % 18 + 1}.{self._lcg_next() % 9 + 1}",
                cve=f"2026-{self._lcg_next() % 90000 + 10000}",
                seg=self._lcg_next() % 8 + 1,
                dept=f"D{self._lcg_next() % 20 + 1:02d}",
                ou=f"OU-{self._lcg_next() % 30 + 1}",
                floor=self._lcg_next() % 15 + 1,
                app=f"APP-{self._lcg_next() % 40 + 1:03d}",
                team=f"T{self._lcg_next() % 25 + 1:02d}",
                mod=f"M{self._lcg_next() % 10 + 1}",
                rev=f"{self._lcg_next() % 20 + 1}.0",
                itype=f"INC-{self._lcg_next() % 12 + 1}",
            )

            # Deadline spread: step 20 .. step 460
            deadline = 20 + (self._lcg_next() % 441)

            instr = Instruction(id=i, text=text, deadline=deadline, category=cat)
            instructions.append(instr)

            # Register in page index
            self.page_index.register_instruction(instr.id, instr.deadline)

        self._state.instructions = instructions

        # Open the first page
        init_event = self._make_event(
            step=0,
            event_type="episode_start",
            text="IT Transformation programme initiated. 300 compliance instructions loaded; "
                 "8000 users awaiting migration.",
            tags=["init", "migration", "compliance"],
        )
        self.page_index.add_event(init_event)

        return Observation(
            step=0,
            phase=self._state.phase,
            text=init_event.text,
            events=[init_event],
            reward=0.0,
            done=False,
            info=self.state_digest(),
        )

    # ─── Step ────────────────────────────────────────────────────────────

    def step(self, action: Action) -> Observation:
        s = self._state
        s.step += 1
        step = s.step
        events: List[Event] = []
        step_reward: float = 0.0
        prev_phase = s.phase

        # ── Ticket queue growth (LCG 1-3) ───────────────────────────────
        ticket_growth = (self._lcg_next() % 3) + 1
        s.ticket_queue += ticket_growth

        # ── SLA decay ────────────────────────────────────────────────────
        if s.ticket_queue > self._SLA_DECAY_THRESHOLD:
            s.sla_score = max(0.0, s.sla_score - self._SLA_DECAY_RATE)
        if s.sla_score < self._SLA_BREACH_LEVEL:
            step_reward -= self._SLA_BREACH_PENALTY
            events.append(self._make_event(
                step, "sla_breach",
                f"SLA score {s.sla_score:.1f}% — breach penalty applied.",
                tags=["sla", "penalty"],
                reward=-self._SLA_BREACH_PENALTY,
            ))

        # ── Ransomware shock ─────────────────────────────────────────────
        if step == self._RANSOMWARE_STEP and "ransomware" not in s.shocks_triggered:
            s.shocks_triggered.append("ransomware")
            s.ransomware_active = True
            s.ransomware_countdown = self._RANSOMWARE_FREEZE_STEPS
            self._migration_frozen_until = step + self._RANSOMWARE_FREEZE_STEPS
            s.ticket_queue += 50
            step_reward -= 50.0
            shock_event = self._make_event(
                step, "shock:ransomware",
                "RANSOMWARE ATTACK detected! Migration frozen for 72 steps. "
                "Ticket queue surged by 50.",
                tags=["shock:ransomware", "security", "critical"],
                reward=-50.0,
                importance=5.0,
            )
            events.append(shock_event)

        # Ransomware countdown tick
        if s.ransomware_active:
            s.ransomware_countdown = max(0, s.ransomware_countdown - 1)
            if s.ransomware_countdown <= 0:
                s.ransomware_active = False
                events.append(self._make_event(
                    step, "ransomware_resolved",
                    "Ransomware containment complete. Migration may resume.",
                    tags=["security", "resolved"],
                ))

        # ── Overdue instruction penalty ──────────────────────────────────
        overdue_count = sum(
            1 for inst in s.instructions
            if not inst.fulfilled and inst.deadline < step
        )
        if overdue_count > 0:
            overdue_penalty = -0.5 * overdue_count
            step_reward += overdue_penalty

        # ── Process action ───────────────────────────────────────────────
        action_reward, action_events = self._process_action(action, step)
        step_reward += action_reward
        events.extend(action_events)

        # ── Update migration phase (each 25 % of total users) ────────────
        new_phase = min(
            self._MIGRATION_PHASES,
            (s.migrated_users * self._MIGRATION_PHASES) // self._TOTAL_USERS + 1,
        )
        if new_phase != s.phase:
            phase_event = self._make_event(
                step, "phase_change",
                f"Migration entered phase {new_phase} "
                f"({s.migrated_users}/{self._TOTAL_USERS} users migrated).",
                tags=["migration", "phase"],
                reward=20.0,
            )
            events.append(phase_event)
            step_reward += 20.0
            s.phase = new_phase

        # ── Terminal: full migration ─────────────────────────────────────
        done = False
        if s.migrated_users >= self._TOTAL_USERS:
            done = True
            step_reward += 400.0
            events.append(self._make_event(
                step, "migration_complete",
                "All 8000 users migrated. IT Transformation complete!",
                tags=["migration", "terminal"],
                reward=400.0,
                importance=5.0,
            ))

        # Also terminate if max steps reached
        if step >= self.MAX_STEPS:
            done = True

        # ── Page index bookkeeping ───────────────────────────────────────
        for ev in events:
            self.page_index.add_event(ev)
            self._process_page_close(step, s.phase, prev_phase, ev)

        self.page_index.maybe_snapshot(step, self.state_digest)

        # ── Accumulate reward ────────────────────────────────────────────
        s.total_reward += step_reward

        obs_parts = [ev.text for ev in events]
        return Observation(
            step=step,
            phase=s.phase,
            text=" | ".join(obs_parts) if obs_parts else f"Step {step}: no notable events.",
            events=events,
            reward=step_reward,
            done=done,
            info=self.state_digest(),
            tags=[t for ev in events for t in ev.tags],
        )

    # ─── Action Dispatch ─────────────────────────────────────────────────

    def _process_action(self, action: Action, step: int) -> tuple[float, List[Event]]:
        """Return (reward, events) for the given action."""
        s = self._state
        reward = 0.0
        events: List[Event] = []

        if action.action_type == ActionType.FULFILL_INSTRUCTION:
            tid = action.target_id
            if tid is not None and 0 <= tid < len(s.instructions):
                instr = s.instructions[tid]
                if not instr.fulfilled:
                    instr.fulfilled = True
                    instr.step_fulfilled = step
                    reward += 1.0
                    self.page_index.mark_instruction_fulfilled(
                        instr.id, self.page_index.page_count, step,
                    )
                    events.append(self._make_event(
                        step, "instruction_fulfilled",
                        f"Instruction {tid} fulfilled: {instr.text}",
                        tags=["instruction", instr.category],
                        reward=1.0,
                    ))

        elif action.action_type == ActionType.MIGRATE_COHORT:
            if s.ransomware_active or step < self._migration_frozen_until:
                events.append(self._make_event(
                    step, "migration_blocked",
                    "Migration frozen due to active ransomware incident.",
                    tags=["migration", "blocked"],
                ))
            elif s.migrated_users < self._TOTAL_USERS:
                cohort = min(self._COHORT_SIZE, self._TOTAL_USERS - s.migrated_users)
                s.migrated_users += cohort
                events.append(self._make_event(
                    step, "cohort_migrated",
                    f"Migrated {cohort} users ({s.migrated_users}/{self._TOTAL_USERS} total).",
                    tags=["migration"],
                ))

        elif action.action_type == ActionType.RESPOND_SHOCK:
            if s.ransomware_active and s.ransomware_countdown > 0:
                reduction = min(20, s.ransomware_countdown)
                s.ransomware_countdown -= reduction
                events.append(self._make_event(
                    step, "shock_response",
                    f"Ransomware response: countdown reduced by {reduction} "
                    f"(remaining: {s.ransomware_countdown}).",
                    tags=["security", "shock_response"],
                ))
                if s.ransomware_countdown <= 0:
                    s.ransomware_active = False
                    self._migration_frozen_until = 0
                    events.append(self._make_event(
                        step, "ransomware_resolved",
                        "Ransomware fully contained via active response.",
                        tags=["security", "resolved"],
                    ))

        elif action.action_type == ActionType.REVIEW_STATUS:
            # Reduces ticket queue by 5-10 (LCG)
            reduction = 5 + (self._lcg_next() % 6)
            s.ticket_queue = max(0, s.ticket_queue - reduction)
            events.append(self._make_event(
                step, "status_review",
                f"Ticket review cleared {reduction} tickets (queue: {s.ticket_queue}).",
                tags=["operations"],
            ))

        elif action.action_type == ActionType.BOOST_MORALE:
            # Slight SLA recovery — represents comms / stakeholder alignment
            recovery = 0.3
            s.sla_score = min(100.0, s.sla_score + recovery)
            events.append(self._make_event(
                step, "morale_boost",
                f"Morale boost applied. SLA score nudged to {s.sla_score:.1f}%.",
                tags=["morale"],
            ))

        elif action.action_type == ActionType.NOOP:
            pass  # intentional no-op

        return reward, events

    # ─── Query Methods ───────────────────────────────────────────────────

    def state_digest(self) -> Dict[str, Any]:
        return self._state.digest()

    def pending_goals(self) -> List[str]:
        s = self._state
        goals: List[str] = []

        unfulfilled = sum(1 for i in s.instructions if not i.fulfilled)
        if unfulfilled > 0:
            goals.append(f"Fulfill {unfulfilled} remaining compliance instructions")

        remaining_users = s.total_users - s.migrated_users
        if remaining_users > 0:
            goals.append(f"Migrate {remaining_users} remaining users")

        if s.ransomware_active:
            goals.append(
                f"Contain ransomware attack ({s.ransomware_countdown} steps remaining)"
            )

        if s.ticket_queue > self._SLA_DECAY_THRESHOLD:
            goals.append(f"Reduce ticket queue from {s.ticket_queue} to below {self._SLA_DECAY_THRESHOLD}")

        if s.sla_score < self._SLA_BREACH_LEVEL:
            goals.append(f"Restore SLA score from {s.sla_score:.1f}% above {self._SLA_BREACH_LEVEL}%")

        return goals

    def action_space(self) -> List[ActionType]:
        return [
            ActionType.FULFILL_INSTRUCTION,
            ActionType.MIGRATE_COHORT,
            ActionType.RESPOND_SHOCK,
            ActionType.REVIEW_STATUS,
            ActionType.BOOST_MORALE,
            ActionType.NOOP,
        ]

    def render(self) -> str:
        s = self._state
        fulfilled = sum(1 for i in s.instructions if i.fulfilled)
        overdue = sum(
            1 for i in s.instructions
            if not i.fulfilled and i.deadline < s.step
        )
        lines = [
            "=" * 60,
            "  IT TRANSFORMATION  [LEGENDARY]",
            "=" * 60,
            f"  Step:  {s.step}/{self.MAX_STEPS}   Phase: {s.phase}/{self._MIGRATION_PHASES}",
            f"  Instructions: {fulfilled}/300 fulfilled  |  {overdue} overdue",
            f"  Migration:    {s.migrated_users}/{s.total_users} users",
            f"  SLA Score:    {s.sla_score:.1f}%",
            f"  Ticket Queue: {s.ticket_queue}",
            f"  Ransomware:   {'ACTIVE (' + str(s.ransomware_countdown) + ' steps)' if s.ransomware_active else 'clear'}",
            f"  Total Reward: {s.total_reward:.1f}",
            "=" * 60,
        ]
        return "\n".join(lines)
