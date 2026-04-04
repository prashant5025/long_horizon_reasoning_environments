"""
Environment Scale — V2 Improvement 6.
ScaledEnvironment wrapper and MultiDepartmentEnvironment for 10x-100x scale.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..types import Action, ActionType, EnvID, Observation, Event, Risk, Instruction, Workstream, Severity
from .base import BaseEnvironment


@dataclass
class ScaleConfig:
    """Configuration for scaling environment parameters."""
    scale: int = 1
    extra_risks: int = 0
    extra_instructions: int = 0
    extra_stakeholders: int = 0
    label: str = "default"

    @classmethod
    def default(cls) -> ScaleConfig:
        return cls(scale=1, label="default")

    @classmethod
    def large(cls) -> ScaleConfig:
        return cls(scale=10, extra_risks=100, extra_instructions=500,
                   extra_stakeholders=20, label="large")

    @classmethod
    def xlarge(cls) -> ScaleConfig:
        return cls(scale=100, extra_risks=500, extra_instructions=2000,
                   extra_stakeholders=50, label="xlarge")

    @classmethod
    def multi_dept(cls, n_depts: int = 3) -> ScaleConfig:
        return cls(scale=5, extra_risks=50, extra_instructions=200,
                   extra_stakeholders=10, label=f"multi_dept_{n_depts}")


class ScaledEnvironment(BaseEnvironment):
    """
    Wraps any BaseEnvironment. Multiplies timing-dependent mechanics:
    shock trigger steps, instruction deadlines, phase transition thresholds,
    and MAX_STEPS.
    """

    def __init__(self, base_env: BaseEnvironment, config: ScaleConfig) -> None:
        self._base = base_env
        self._config = config
        self._scale = config.scale
        self.MAX_STEPS = base_env.MAX_STEPS * config.scale
        super().__init__(env_id=base_env.env_id, max_steps=self.MAX_STEPS)

    @property
    def env_id(self) -> EnvID:
        return self._base.env_id

    def reset(self, seed: Optional[int] = None) -> Observation:
        obs = self._base.reset(seed)
        # Scale the max steps
        self._base.MAX_STEPS = self.MAX_STEPS

        # Add extra entities at scale
        self._add_extra_entities()

        obs.info["scale"] = self._scale
        obs.info["scaled_max_steps"] = self.MAX_STEPS
        return obs

    def _add_extra_entities(self) -> None:
        """Extend entity lists with generated items at scale."""
        cfg = self._config
        state = self._base._state if hasattr(self._base, '_state') else None
        if state is None:
            return

        # Add extra risks for PM environment
        if hasattr(state, 'risks') and cfg.extra_risks > 0:
            base_count = len(state.risks)
            severities = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]
            for i in range(cfg.extra_risks):
                sev = severities[self._base._lcg_next() % len(severities)]
                state.risks.append(Risk(
                    id=base_count + i,
                    description=f"Scaled risk {i}: {sev.value} severity issue",
                    severity=sev,
                    step_created=0,
                ))

        # Add extra instructions for HR/IT environment
        if hasattr(state, 'instructions') and cfg.extra_instructions > 0:
            base_count = len(state.instructions)
            categories = ["compliance", "security", "migration", "training", "documentation"]
            for i in range(cfg.extra_instructions):
                deadline = 20 + (self._base._lcg_next() % (self.MAX_STEPS - 40))
                cat = categories[self._base._lcg_next() % len(categories)]
                instr = Instruction(
                    id=base_count + i,
                    text=f"Scaled {cat} instruction {i}",
                    deadline=deadline,
                    category=cat,
                )
                state.instructions.append(instr)
                self._base.page_index.register_instruction(instr.id, instr.deadline)

    def step(self, action: Action) -> Observation:
        return self._base.step(action)

    def state_digest(self) -> Dict[str, Any]:
        d = self._base.state_digest()
        d["scale"] = self._scale
        return d

    def pending_goals(self) -> List[str]:
        return self._base.pending_goals()

    def action_space(self) -> List[Action]:
        return self._base.action_space()

    def render(self) -> str:
        header = f"[SCALED x{self._scale} | {self._config.label}]\n"
        return header + self._base.render()


class MultiDepartmentEnvironment(BaseEnvironment):
    """
    Runs N independent BaseEnvironment instances (departments) simultaneously
    under a shared corporate budget. The agent routes attention via dept_id.
    """

    def __init__(self, env_factories: List, shared_budget: float = 20_000_000.0,
                 scale_config: Optional[ScaleConfig] = None) -> None:
        """
        env_factories: list of callables that return BaseEnvironment instances.
        """
        self._factories = env_factories
        self._shared_budget = shared_budget
        self._initial_budget = shared_budget
        self._departments: List[BaseEnvironment] = []
        self._dept_names = ["Engineering", "Product", "Finance"]
        self._scale_config = scale_config or ScaleConfig.multi_dept(len(env_factories))
        self._current_dept = 0
        self._cross_dept_blockers: Dict[int, List[int]] = {}  # dept -> [blocking_depts]
        self.MAX_STEPS = 420 * self._scale_config.scale
        super().__init__(env_id=EnvID.PM, max_steps=self.MAX_STEPS)

    @property
    def env_id(self) -> EnvID:
        return EnvID.PM  # primary type

    def reset(self, seed: Optional[int] = None) -> Observation:
        if seed is not None:
            self._lcg_seed = seed
        self._shared_budget = self._initial_budget
        self._departments = []
        for i, factory in enumerate(self._factories):
            env = factory()
            dept_seed = (seed or 42) + i * 1000
            env.reset(dept_seed)
            if self._scale_config.scale > 1:
                env.MAX_STEPS = env.MAX_STEPS * self._scale_config.scale
            self._departments.append(env)

        self._cross_dept_blockers = {}
        self._current_dept = 0

        return Observation(
            step=0,
            phase=1,
            text=f"MultiDepartment environment initialized: {len(self._departments)} departments, "
                 f"shared budget ${self._shared_budget:,.0f}",
            tags=["multi_dept", "init"],
        )

    def step(self, action: Action) -> Observation:
        dept_id = action.params.get("dept_id", self._current_dept)
        if dept_id < 0 or dept_id >= len(self._departments):
            dept_id = 0

        # Check cross-department blockers
        if dept_id in self._cross_dept_blockers:
            blockers = self._cross_dept_blockers[dept_id]
            if blockers:
                return Observation(
                    step=self._departments[dept_id]._state.step if hasattr(self._departments[dept_id], '_state') else 0,
                    phase=1,
                    text=f"Department {self._dept_names[dept_id]} is blocked by departments: "
                         f"{[self._dept_names[b] for b in blockers]}",
                    tags=["blocked", f"dept:{dept_id}"],
                    reward=-1.0,
                )

        obs = self._departments[dept_id].step(action)
        obs.tags.append(f"dept:{dept_id}")
        obs.tags.append(f"dept_name:{self._dept_names[dept_id]}")

        # Deduct from shared budget if applicable
        if action.action_type == ActionType.ALLOCATE_BUDGET:
            cost = action.params.get("amount", 100_000)
            self._shared_budget -= cost
            if self._shared_budget < 0:
                obs.reward -= 10.0
                obs.text += " [WARNING: Shared budget exceeded!]"

        return obs

    def state_digest(self) -> Dict[str, Any]:
        return {
            "departments": [d.state_digest() for d in self._departments],
            "shared_budget": self._shared_budget,
            "n_departments": len(self._departments),
            "cross_dept_blockers": dict(self._cross_dept_blockers),
        }

    def pending_goals(self) -> List[str]:
        goals = []
        for i, dept in enumerate(self._departments):
            for g in dept.pending_goals():
                goals.append(f"[{self._dept_names[i]}] {g}")
        return goals

    def action_space(self) -> List[Action]:
        """Return actions for all departments with dept_id param."""
        all_actions = []
        for i, dept in enumerate(self._departments):
            for at in dept.action_space():
                # Wrap raw ActionType into Action with dept_id param
                if isinstance(at, ActionType):
                    action = Action(action_type=at, params={"dept_id": i})
                else:
                    at.params["dept_id"] = i
                    action = at
                all_actions.append(action)
        return all_actions

    def render(self) -> str:
        lines = [
            "=" * 60,
            f"  MULTI-DEPARTMENT ENVIRONMENT ({len(self._departments)} depts)",
            f"  Shared Budget: ${self._shared_budget:,.0f} / ${self._initial_budget:,.0f}",
            "=" * 60,
        ]
        for i, dept in enumerate(self._departments):
            lines.append(f"\n--- {self._dept_names[i]} ---")
            lines.append(dept.render())
        return "\n".join(lines)
