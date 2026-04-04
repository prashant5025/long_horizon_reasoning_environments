"""
Advanced Reward Shaping — V2 Improvement 5.
Three additive shaping layers: potential-based progress, critical path bonus,
dependency resolution bonus.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class CriticalPathNode:
    node_id: str
    weight: float = 1.0
    completed: bool = False
    dependencies: List[str] = field(default_factory=list)


class PotentialFunction:
    """
    Potential-based shaping: F(s, s') = gamma * phi(s') - phi(s).
    Ng et al. 1999 guarantees this preserves optimal policy.
    """

    def __init__(self, feature_weights: Dict[str, float], gamma: float = 0.95):
        self._weights = feature_weights
        self._gamma = gamma
        self._prev_potential: Optional[float] = None

    def phi(self, state_digest: Dict[str, Any]) -> float:
        """Compute potential as weighted sum of normalised progress metrics."""
        total = 0.0
        for key, weight in self._weights.items():
            val = state_digest.get(key, 0.0)
            if isinstance(val, bool):
                val = 1.0 if val else 0.0
            total += weight * float(val)
        return total

    def compute(self, state_digest: Dict[str, Any]) -> float:
        """Return delta_phi = gamma * phi(s') - phi(s)."""
        current = self.phi(state_digest)
        if self._prev_potential is None:
            self._prev_potential = current
            return 0.0
        delta = self._gamma * current - self._prev_potential
        self._prev_potential = current
        return delta

    def reset(self) -> None:
        self._prev_potential = None


class CriticalPathTracker:
    """DAG of project milestones with bonuses on advance."""

    def __init__(self) -> None:
        self._nodes: Dict[str, CriticalPathNode] = {}
        self._miss_check_interval = 15
        self._last_check_step = 0

    def add_node(self, node: CriticalPathNode) -> None:
        self._nodes[node.node_id] = node

    def complete_node(self, node_id: str) -> float:
        """Mark node complete and return bonus. +5 * (1 + weight)."""
        node = self._nodes.get(node_id)
        if node is None or node.completed:
            return 0.0
        # Check dependencies
        for dep_id in node.dependencies:
            dep = self._nodes.get(dep_id)
            if dep and not dep.completed:
                return 0.0
        node.completed = True
        return 5.0 * (1.0 + node.weight)

    def check_miss_penalty(self, step: int) -> float:
        """Every 15 steps, penalize -3 if stuck on uncompleted nodes."""
        if step - self._last_check_step < self._miss_check_interval:
            return 0.0
        self._last_check_step = step
        uncompleted = sum(1 for n in self._nodes.values() if not n.completed)
        if uncompleted > 0 and len(self._nodes) > 0:
            completion_ratio = sum(1 for n in self._nodes.values() if n.completed) / len(self._nodes)
            if completion_ratio < 0.5:
                return -3.0
        return 0.0

    def reset(self) -> None:
        for node in self._nodes.values():
            node.completed = False
        self._last_check_step = 0


class DependencyGraph:
    """Track dependency resolution bonuses."""

    def __init__(self) -> None:
        self._edges: Dict[str, List[str]] = {}  # blocker -> [blocked_by]
        self._resolved: Set[str] = set()

    def add_dependency(self, blocked: str, blocker: str) -> None:
        if blocked not in self._edges:
            self._edges[blocked] = []
        self._edges[blocked].append(blocker)

    def resolve(self, node_id: str) -> float:
        """Resolve a blocking node. Return +3 * unlocked_count."""
        if node_id in self._resolved:
            return 0.0
        self._resolved.add(node_id)
        unlocked = 0
        for blocked, blockers in self._edges.items():
            if node_id in blockers:
                remaining = [b for b in blockers if b not in self._resolved]
                if len(remaining) == 0:
                    unlocked += 1
        return 3.0 * unlocked

    def reset(self) -> None:
        self._resolved.clear()


class RewardShaper:
    """
    Orchestrates three additive shaping layers after each environment step.
    """

    def __init__(self, env_id: str) -> None:
        self.env_id = env_id
        self.potential = self._make_potential(env_id)
        self.critical_path = CriticalPathTracker()
        self.dependency_graph = DependencyGraph()
        self._init_critical_path(env_id)
        self._init_dependencies(env_id)

    def _make_potential(self, env_id: str) -> PotentialFunction:
        """Environment-specific potential function weights."""
        if env_id == "sales":
            return PotentialFunction({
                "poc_score": 0.3 / 100.0,
                "avg_relationship": 0.25 / 100.0,
                "phase": 0.3 / 6.0,
                "live_stakeholders": 0.15 / 11.0,
            })
        elif env_id == "pm":
            return PotentialFunction({
                "risks_resolved": 0.2 / 47.0,
                "ws_progress_avg": 0.3 / 100.0,
                "budget_runway": 0.2 / 6_000_000.0,
                "phase": 0.3 / 5.0,
            })
        else:  # hr_it
            return PotentialFunction({
                "fulfilled_instructions": 0.3 / 300.0,
                "migrated_pct": 0.3 / 100.0,
                "sla_score": 0.2 / 100.0,
                "ticket_queue": -0.2 / 100.0,
            })

    def _init_critical_path(self, env_id: str) -> None:
        """Set up environment-specific milestone DAGs."""
        if env_id == "sales":
            nodes = [
                CriticalPathNode("discovery", weight=1.0),
                CriticalPathNode("poc_60", weight=1.5, dependencies=["discovery"]),
                CriticalPathNode("poc_80", weight=2.0, dependencies=["poc_60"]),
                CriticalPathNode("negotiation", weight=1.5, dependencies=["poc_80"]),
                CriticalPathNode("close", weight=3.0, dependencies=["negotiation"]),
            ]
        elif env_id == "pm":
            nodes = [
                CriticalPathNode("risk_triage", weight=1.0),
                CriticalPathNode("ws_backend_50", weight=1.5, dependencies=["risk_triage"]),
                CriticalPathNode("ws_frontend_50", weight=1.5, dependencies=["risk_triage"]),
                CriticalPathNode("integration", weight=2.0, dependencies=["ws_backend_50", "ws_frontend_50"]),
                CriticalPathNode("delivery", weight=3.0, dependencies=["integration"]),
            ]
        else:  # hr_it
            nodes = [
                CriticalPathNode("compliance_50pct", weight=1.0),
                CriticalPathNode("migration_phase1", weight=1.5),
                CriticalPathNode("migration_phase2", weight=2.0, dependencies=["migration_phase1"]),
                CriticalPathNode("migration_phase3", weight=2.5, dependencies=["migration_phase2"]),
                CriticalPathNode("full_migration", weight=3.0, dependencies=["migration_phase3", "compliance_50pct"]),
            ]
        for node in nodes:
            self.critical_path.add_node(node)

    def _init_dependencies(self, env_id: str) -> None:
        """Set up dependency tracking."""
        if env_id == "pm":
            self.dependency_graph.add_dependency("qa_automation", "backend_migration")
            self.dependency_graph.add_dependency("qa_automation", "frontend_rewrite")
            self.dependency_graph.add_dependency("deployment", "qa_automation")
            self.dependency_graph.add_dependency("deployment", "data_pipeline")
        elif env_id == "hr_it":
            self.dependency_graph.add_dependency("migration_phase2", "security_audit")
            self.dependency_graph.add_dependency("migration_phase3", "migration_phase2")
            self.dependency_graph.add_dependency("go_live", "compliance_complete")

    def shape(self, base_reward: float, state_digest: Dict[str, Any],
              step: int, events: List[Dict[str, Any]] | None = None) -> Tuple[float, Dict[str, float]]:
        """
        Apply three shaping layers. Returns (total_shaped_reward, breakdown).
        """
        breakdown: Dict[str, float] = {"base": base_reward}

        # Layer 1: Potential-based progress shaping
        progress_bonus = self.potential.compute(state_digest)
        progress_bonus = max(-3.0, min(3.0, progress_bonus))
        breakdown["progress"] = progress_bonus

        # Layer 2: Critical path bonus/penalty
        cp_bonus = self.critical_path.check_miss_penalty(step)
        if events:
            for evt in events:
                evt_type = evt.get("event_type", "") if isinstance(evt, dict) else getattr(evt, "event_type", "")
                if "milestone" in str(evt_type).lower() or "phase" in str(evt_type).lower():
                    cp_bonus += self._check_critical_path_advance(state_digest)
        breakdown["critical_path"] = cp_bonus

        # Layer 3: Dependency resolution
        dep_bonus = 0.0
        if events:
            for evt in events:
                tags = evt.get("tags", []) if isinstance(evt, dict) else getattr(evt, "tags", [])
                for tag in tags:
                    if "resolved:" in str(tag):
                        dep_bonus += self.dependency_graph.resolve(str(tag).split(":")[-1])
        breakdown["dependency"] = dep_bonus

        total = base_reward + progress_bonus + cp_bonus + dep_bonus
        breakdown["total"] = total
        return total, breakdown

    def _check_critical_path_advance(self, state_digest: Dict[str, Any]) -> float:
        """Check if any critical path nodes should be completed based on state."""
        bonus = 0.0
        if self.env_id == "sales":
            poc = state_digest.get("poc_score", 0)
            phase = state_digest.get("phase", 1)
            if poc >= 60:
                bonus += self.critical_path.complete_node("poc_60")
            if poc >= 80:
                bonus += self.critical_path.complete_node("poc_80")
            if phase >= 2:
                bonus += self.critical_path.complete_node("discovery")
            if phase >= 4:
                bonus += self.critical_path.complete_node("negotiation")
            if state_digest.get("deal_closed"):
                bonus += self.critical_path.complete_node("close")
        elif self.env_id == "pm":
            resolved = state_digest.get("risks_resolved", 0)
            ws_avg = state_digest.get("ws_progress_avg", 0)
            if resolved >= 10:
                bonus += self.critical_path.complete_node("risk_triage")
            if ws_avg >= 50:
                bonus += self.critical_path.complete_node("ws_backend_50")
                bonus += self.critical_path.complete_node("ws_frontend_50")
            if ws_avg >= 75:
                bonus += self.critical_path.complete_node("integration")
            if state_digest.get("program_delivered"):
                bonus += self.critical_path.complete_node("delivery")
        else:  # hr_it
            fulfilled = state_digest.get("fulfilled_instructions", 0)
            migrated = state_digest.get("migrated_pct", 0)
            if fulfilled >= 150:
                bonus += self.critical_path.complete_node("compliance_50pct")
            if migrated >= 25:
                bonus += self.critical_path.complete_node("migration_phase1")
            if migrated >= 50:
                bonus += self.critical_path.complete_node("migration_phase2")
            if migrated >= 75:
                bonus += self.critical_path.complete_node("migration_phase3")
            if migrated >= 100:
                bonus += self.critical_path.complete_node("full_migration")
        return bonus

    def reset(self) -> None:
        self.potential.reset()
        self.critical_path.reset()
        self.dependency_graph.reset()
