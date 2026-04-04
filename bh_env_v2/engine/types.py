"""
BusinessHorizonENV Type System
All dataclasses and enums for the simulation framework.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


# ─── Enums ───────────────────────────────────────────────────────────────────

class ActionType(Enum):
    CONTACT_STAKEHOLDER = auto()
    RUN_POC = auto()
    ADVANCE_DEAL = auto()
    RESOLVE_RISK = auto()
    ADVANCE_WORKSTREAM = auto()
    ALLOCATE_BUDGET = auto()
    FULFILL_INSTRUCTION = auto()
    MIGRATE_COHORT = auto()
    RESPOND_SHOCK = auto()
    BOOST_MORALE = auto()
    REVIEW_STATUS = auto()
    NOOP = auto()


class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class EnvID(Enum):
    SALES = "sales"
    PM = "pm"
    HR_IT = "hr_it"


class GoalStatus(Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETE = "COMPLETE"
    BLOCKED = "BLOCKED"
    ABANDONED = "ABANDONED"


# ─── Core Game Objects ───────────────────────────────────────────────────────

@dataclass
class Stakeholder:
    id: int
    name: str
    role: str
    engagement: float = 50.0
    is_champion: bool = False
    is_detractor: bool = False
    active: bool = True


@dataclass
class Risk:
    id: int
    description: str
    severity: Severity
    resolved: bool = False
    workstream_id: Optional[int] = None
    step_created: int = 0


@dataclass
class Workstream:
    id: int
    name: str
    progress: float = 0.0
    blocked: bool = False
    dependencies: List[int] = field(default_factory=list)


@dataclass
class Instruction:
    id: int
    text: str
    deadline: int
    fulfilled: bool = False
    step_fulfilled: Optional[int] = None
    category: str = "general"


@dataclass
class Event:
    step: int
    event_type: str
    text: str
    tags: List[str] = field(default_factory=list)
    reward: float = 0.0
    importance: float = 1.0


@dataclass
class RewardEvent:
    step: int
    base_reward: float
    shaped_reward: float
    source: str
    tags: List[str] = field(default_factory=list)


@dataclass
class Action:
    action_type: ActionType
    target_id: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = [self.action_type.name]
        if self.target_id is not None:
            parts.append(f"target={self.target_id}")
        return f"Action({', '.join(parts)})"


@dataclass
class Observation:
    step: int
    phase: int
    text: str
    events: List[Event] = field(default_factory=list)
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


# ─── State Dataclasses ───────────────────────────────────────────────────────

@dataclass
class SalesState:
    step: int = 0
    phase: int = 1
    stakeholders: List[Stakeholder] = field(default_factory=list)
    poc_score: float = 0.0
    deal_value: float = 2_400_000.0
    budget_frozen: bool = False
    deal_closed: bool = False
    shocks_triggered: List[str] = field(default_factory=list)
    total_reward: float = 0.0

    def digest(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "phase": self.phase,
            "poc_score": self.poc_score,
            "deal_value": self.deal_value,
            "budget_frozen": self.budget_frozen,
            "deal_closed": self.deal_closed,
            "avg_relationship": sum(s.engagement for s in self.stakeholders if s.active) / max(1, sum(1 for s in self.stakeholders if s.active)),
            "live_stakeholders": sum(1 for s in self.stakeholders if s.active),
            "total_reward": self.total_reward,
        }


@dataclass
class PMState:
    step: int = 0
    phase: int = 1
    workstreams: List[Workstream] = field(default_factory=list)
    risks: List[Risk] = field(default_factory=list)
    team_morale: float = 80.0
    budget: float = 6_000_000.0
    budget_initial: float = 6_000_000.0
    program_delivered: bool = False
    shocks_triggered: List[str] = field(default_factory=list)
    total_reward: float = 0.0

    def digest(self) -> Dict[str, Any]:
        ws_progress = [w.progress for w in self.workstreams]
        return {
            "step": self.step,
            "phase": self.phase,
            "risks_total": len(self.risks),
            "risks_resolved": sum(1 for r in self.risks if r.resolved),
            "team_morale": self.team_morale,
            "budget_runway": self.budget,
            "ws_progress_avg": sum(ws_progress) / max(1, len(ws_progress)),
            "program_delivered": self.program_delivered,
            "total_reward": self.total_reward,
        }


@dataclass
class HRITState:
    step: int = 0
    phase: int = 1
    instructions: List[Instruction] = field(default_factory=list)
    migrated_users: int = 0
    total_users: int = 8000
    sla_score: float = 100.0
    ticket_queue: int = 0
    ransomware_active: bool = False
    ransomware_countdown: int = 0
    shocks_triggered: List[str] = field(default_factory=list)
    total_reward: float = 0.0

    def digest(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "phase": self.phase,
            "fulfilled_instructions": sum(1 for i in self.instructions if i.fulfilled),
            "total_instructions": len(self.instructions),
            "migrated_pct": (self.migrated_users / self.total_users) * 100 if self.total_users > 0 else 0,
            "sla_score": self.sla_score,
            "ticket_queue": self.ticket_queue,
            "ransomware_active": self.ransomware_active,
            "total_reward": self.total_reward,
        }


# ─── Page Index Types ────────────────────────────────────────────────────────

@dataclass
class Page:
    page_id: int
    start_step: int
    end_step: int
    events: List[Event] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class InstrIndexEntry:
    instruction_id: int
    fulfilled: bool = False
    page_id: Optional[int] = None
    step: Optional[int] = None
    deadline: int = 0


@dataclass
class StateSnapshot:
    step: int
    data: Dict[str, Any] = field(default_factory=dict)
