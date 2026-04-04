"""
Learned Value Function for BusinessHorizonENV v2.

Pure NumPy neural network with TD(0) learning, Adam optimizer,
target-network sync, and a hybrid heuristic/learned scorer.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..engine.types import ActionType, EnvID

# ── Ordered action list (follows ActionType enum declaration order) ──────────
_ACTION_ORDER: List[str] = [a.name for a in ActionType]
_NUM_ACTIONS: int = len(_ACTION_ORDER)  # 12

# ── Heuristic prior scores (same defaults used in the planner) ───────────────
_HEURISTIC_SCORES: Dict[str, float] = {
    "ADVANCE_DEAL": 8,
    "RESOLVE_RISK": 7,
    "ADVANCE_WORKSTREAM": 6,
    "FULFILL_INSTRUCTION": 6,
    "MIGRATE_COHORT": 5,
    "CONTACT_STAKEHOLDER": 5,
    "RUN_POC": 5,
    "ALLOCATE_BUDGET": 4,
    "RESPOND_SHOCK": 7,
    "BOOST_MORALE": 4,
    "REVIEW_STATUS": 3,
    "NOOP": 1,
}


# ═════════════════════════════════════════════════════════════════════════════
#  Feature Extractor
# ═════════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """Converts a (state_digest, action_type) pair into a 32-d float32 vector."""

    MAX_STEPS: int = 480

    # ------------------------------------------------------------------ #
    def extract(self, state_digest: dict, action_type: str) -> np.ndarray:
        """Return a feature vector of shape (32,) dtype float32."""
        f = np.zeros(32, dtype=np.float32)

        step = float(state_digest.get("step", 0))
        phase = float(state_digest.get("phase", 0))
        max_steps = float(self.MAX_STEPS)

        # Dims 0-1: normalised phase and step
        f[0] = phase / 6.0
        f[1] = step / max_steps

        # Dims 2-13: one-hot action encoding (12 slots)
        action_name = action_type if isinstance(action_type, str) else action_type.name
        for idx, name in enumerate(_ACTION_ORDER):
            if name == action_name:
                f[2 + idx] = 1.0
                break

        # Dims 13-16 (overlap on 13 is intentional: first Sales dim overwrites
        # last action one-hot only when both fire — but 12 actions map to
        # indices 2..13 so index 13 is the 12th action *and* the first Sales
        # feature. We keep the spec exactly as given.)
        # Sales features
        f[13] = float(state_digest.get("poc_score", 0)) / 100.0
        f[14] = float(state_digest.get("avg_relationship", 0)) / 100.0
        f[15] = float(state_digest.get("live_stakeholders", 0)) / 11.0
        f[16] = 1.0 if state_digest.get("budget_frozen", False) else 0.0

        # Dims 17-20: PM features
        f[17] = float(state_digest.get("risks_resolved", 0)) / 47.0
        f[18] = float(state_digest.get("team_morale", 0)) / 100.0
        f[19] = float(state_digest.get("ws_progress_avg", 0)) / 100.0
        f[20] = float(state_digest.get("budget_runway", 0)) / 6_000_000.0

        # Dims 21-24: HR/IT features
        f[21] = float(state_digest.get("fulfilled_instructions", 0)) / 300.0
        f[22] = float(state_digest.get("migrated_pct", 0)) / 100.0
        f[23] = float(state_digest.get("sla_score", 0)) / 100.0
        f[24] = min(float(state_digest.get("ticket_queue", 0)) / 100.0, 1.0)

        # Dims 25-27: reward features
        total_reward = float(state_digest.get("total_reward", 0))
        if total_reward > 0:
            f[25] = 1.0
        elif total_reward < 0:
            f[25] = -1.0
        else:
            f[25] = 0.0
        f[26] = math.log(abs(total_reward) + 1.0) / 10.0
        f[27] = phase / 6.0  # phase fraction (same as dim 0)

        # Dim 28: heuristic prior
        f[28] = _HEURISTIC_SCORES.get(action_name, 1) / 10.0

        # Dims 29-31: pressure / alignment features
        f[29] = 1.0 - step / max_steps  # time pressure

        # Budget / resource pressure (env-specific heuristic)
        budget_runway = float(state_digest.get("budget_runway", 0))
        ticket_queue = float(state_digest.get("ticket_queue", 0))
        morale = float(state_digest.get("team_morale", 100))
        resource_pressure = 0.0
        if budget_runway > 0:
            # PM env: low budget => high pressure
            resource_pressure = max(resource_pressure,
                                    1.0 - min(budget_runway / 6_000_000.0, 1.0))
        if ticket_queue > 0:
            resource_pressure = max(resource_pressure,
                                    min(ticket_queue / 100.0, 1.0))
        if morale < 100:
            resource_pressure = max(resource_pressure,
                                    1.0 - morale / 100.0)
        f[30] = resource_pressure

        # Goal alignment: 1 if action matches a plausible high-value action
        goal_alignment = 0.0
        if _HEURISTIC_SCORES.get(action_name, 0) >= 6:
            goal_alignment = 1.0
        f[31] = goal_alignment

        return f


# ═════════════════════════════════════════════════════════════════════════════
#  Value Network (pure NumPy MLP)
# ═════════════════════════════════════════════════════════════════════════════

class ValueNetwork:
    """
    Architecture: Linear(32,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,1)
    Trained with Adam, L2 regularisation, and per-parameter gradient clipping.
    """

    def __init__(self, seed: int = 42) -> None:
        rng = np.random.RandomState(seed)

        # He / Kaiming initialisation  std = sqrt(2 / fan_in)
        self.W1 = rng.randn(32, 64).astype(np.float32) * np.float32(math.sqrt(2.0 / 32))
        self.b1 = np.zeros(64, dtype=np.float32)

        self.W2 = rng.randn(64, 32).astype(np.float32) * np.float32(math.sqrt(2.0 / 64))
        self.b2 = np.zeros(32, dtype=np.float32)

        self.W3 = rng.randn(32, 1).astype(np.float32) * np.float32(math.sqrt(2.0 / 32))
        self.b3 = np.zeros(1, dtype=np.float32)

        # Adam hyper-parameters
        self.lr: float = 5e-4
        self.beta1: float = 0.9
        self.beta2: float = 0.999
        self.eps: float = 1e-8
        self.l2_lambda: float = 1e-4
        self.clip_norm: float = 0.5

        # Adam moments  (m = first moment, v = second moment)
        self._param_names = ["W1", "b1", "W2", "b2", "W3", "b3"]
        self.m: Dict[str, np.ndarray] = {n: np.zeros_like(getattr(self, n)) for n in self._param_names}
        self.v: Dict[str, np.ndarray] = {n: np.zeros_like(getattr(self, n)) for n in self._param_names}
        self.t: int = 0  # Adam timestep

    # ── forward pass ─────────────────────────────────────────────────────
    def forward(self, x: np.ndarray) -> float:
        """Forward pass.  *x* is (32,) or (B,32).  Returns scalar float."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Layer 1
        z1 = x @ self.W1 + self.b1          # (B,64)
        a1 = np.maximum(z1, 0.0)            # ReLU

        # Layer 2
        z2 = a1 @ self.W2 + self.b2         # (B,32)
        a2 = np.maximum(z2, 0.0)            # ReLU

        # Output
        out = a2 @ self.W3 + self.b3        # (B,1)
        return float(out.squeeze())

    # ── backward pass (single sample MSE) ────────────────────────────────
    def backward(self, x: np.ndarray, target: float) -> float:
        """
        One gradient step on MSE loss = 0.5*(pred - target)^2.
        Returns the loss (float).
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        B = x.shape[0]

        # ── Forward (store activations) ──
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)

        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0.0)

        out = a2 @ self.W3 + self.b3  # (B,1)
        pred = out.squeeze()

        loss = float(0.5 * np.mean((pred - target) ** 2))

        # ── Backward ──
        # d_out: (B,1)
        d_out = ((pred - target) / B).reshape(B, 1)

        # Layer 3
        dW3 = a2.T @ d_out                  # (32,1)
        db3 = d_out.sum(axis=0)              # (1,)
        d_a2 = d_out @ self.W3.T             # (B,32)

        # ReLU2
        d_z2 = d_a2 * (z2 > 0).astype(np.float32)

        # Layer 2
        dW2 = a1.T @ d_z2                   # (64,32)
        db2 = d_z2.sum(axis=0)              # (32,)
        d_a1 = d_z2 @ self.W2.T             # (B,64)

        # ReLU1
        d_z1 = d_a1 * (z1 > 0).astype(np.float32)

        # Layer 1
        dW1 = x.T @ d_z1                    # (32,64)
        db1 = d_z1.sum(axis=0)              # (64,)

        grads: Dict[str, np.ndarray] = {
            "W1": dW1, "b1": db1,
            "W2": dW2, "b2": db2,
            "W3": dW3, "b3": db3,
        }

        # ── Adam update with L2 reg and gradient clipping ──
        self.t += 1
        for name in self._param_names:
            g = grads[name]

            # L2 regularisation gradient
            param = getattr(self, name)
            g = g + self.l2_lambda * param

            # Per-parameter gradient clipping (clip to max norm 0.5)
            g_norm = np.linalg.norm(g)
            if g_norm > self.clip_norm:
                g = g * (self.clip_norm / g_norm)

            # Adam moments
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (g ** 2)

            m_hat = self.m[name] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1.0 - self.beta2 ** self.t)

            setattr(self, name,
                    param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))

        return loss

    # ── weight copy ──────────────────────────────────────────────────────
    def copy_weights_to(self, other: "ValueNetwork") -> None:
        """Copy all weights and biases (but *not* optimiser state) to *other*."""
        for name in self._param_names:
            setattr(other, name, getattr(self, name).copy())


# ═════════════════════════════════════════════════════════════════════════════
#  Value Function Trainer (online TD + offline batch)
# ═════════════════════════════════════════════════════════════════════════════

class ValueFunctionTrainer:
    """Wraps an online_net and a target_net; provides TD(0) updates."""

    SYNC_EVERY: int = 50

    def __init__(self, seed: int = 42) -> None:
        self.online_net = ValueNetwork(seed=seed)
        self.target_net = ValueNetwork(seed=seed + 1)
        # Initialise target with online weights
        self.online_net.copy_weights_to(self.target_net)

        self.feature_extractor = FeatureExtractor()
        self.step_counter: int = 0
        self.loss_history: List[float] = []
        self.offline_steps: int = 0

    # ── helpers ───────────────────────────────────────────────────────────
    def _normalise_reward(self, reward: float) -> float:
        return float(np.clip(reward / 20.0, -1.0, 1.0))

    def sync_target(self) -> None:
        self.online_net.copy_weights_to(self.target_net)

    # ── online (single transition) ───────────────────────────────────────
    def online_update(
        self,
        state_digest: dict,
        action_type: str,
        reward: float,
        next_state_digest: dict,
        next_action_type: str,
        done: bool,
        gamma: float = 0.95,
    ) -> float:
        """One-step TD(0) update. Returns loss."""
        norm_r = self._normalise_reward(reward)

        features = self.feature_extractor.extract(state_digest, action_type)

        if done:
            target = norm_r
        else:
            next_features = self.feature_extractor.extract(
                next_state_digest, next_action_type
            )
            target = norm_r + gamma * self.target_net.forward(next_features)

        target = float(np.clip(target, -3.0, 3.0))

        loss = self.online_net.backward(features, target)
        self.loss_history.append(loss)

        self.step_counter += 1
        if self.step_counter % self.SYNC_EVERY == 0:
            self.sync_target()

        return loss

    # ── offline (batch of transitions) ───────────────────────────────────
    def offline_update(
        self,
        transitions: List[Dict[str, Any]],
        gamma: float = 0.95,
    ) -> float:
        """Batch TD(0) update over up to 64 transitions. Returns mean loss."""
        batch = transitions[:64]
        if not batch:
            return 0.0

        features_list: List[np.ndarray] = []
        targets_list: List[float] = []

        for tr in batch:
            sd = tr["state_digest"]
            at = tr["action_type"]
            r = tr["reward"]
            nsd = tr["next_state_digest"]
            nat = tr["next_action_type"]
            d = tr["done"]

            feat = self.feature_extractor.extract(sd, at)
            norm_r = self._normalise_reward(r)

            if d:
                t_val = norm_r
            else:
                next_feat = self.feature_extractor.extract(nsd, nat)
                t_val = norm_r + gamma * self.target_net.forward(next_feat)

            t_val = float(np.clip(t_val, -3.0, 3.0))
            features_list.append(feat)
            targets_list.append(t_val)

        X = np.stack(features_list, axis=0)           # (B, 32)
        T = np.array(targets_list, dtype=np.float32)  # (B,)

        loss = self.online_net.backward(X, T)
        self.loss_history.append(loss)

        self.offline_steps += len(batch)
        self.step_counter += 1
        if self.step_counter % self.SYNC_EVERY == 0:
            self.sync_target()

        return loss


# ═════════════════════════════════════════════════════════════════════════════
#  Value Function Registry (per-environment trainers + hybrid scorer)
# ═════════════════════════════════════════════════════════════════════════════

class ValueFunctionRegistry:
    """
    Manages one ValueFunctionTrainer per env_id.
    Provides a hybrid heuristic/learned scorer whose heuristic weight
    decays as more offline data is consumed.
    """

    def __init__(self) -> None:
        self._trainers: Dict[str, ValueFunctionTrainer] = {}
        self._feature_extractor = FeatureExtractor()

    # ── trainer access ───────────────────────────────────────────────────
    def get_or_create(self, env_id: str) -> ValueFunctionTrainer:
        if env_id not in self._trainers:
            seed = hash(env_id) % (2 ** 31)
            self._trainers[env_id] = ValueFunctionTrainer(seed=seed)
        return self._trainers[env_id]

    # ── delegated updates ────────────────────────────────────────────────
    def online_update(
        self,
        env_id: str,
        state_digest: dict,
        action_type: str,
        reward: float,
        next_state_digest: dict,
        next_action_type: str,
        done: bool,
        gamma: float = 0.95,
    ) -> float:
        trainer = self.get_or_create(env_id)
        return trainer.online_update(
            state_digest, action_type, reward,
            next_state_digest, next_action_type, done, gamma,
        )

    def offline_update(
        self,
        env_id: str,
        transitions: List[Dict[str, Any]],
        gamma: float = 0.95,
    ) -> float:
        trainer = self.get_or_create(env_id)
        return trainer.offline_update(transitions, gamma)

    # ── hybrid scorer ────────────────────────────────────────────────────
    def make_hybrid_scorer(
        self,
        env_id: str,
        heuristic_weight: float = 0.5,
    ) -> Callable[[dict, str], float]:
        """
        Return a callable(state_digest, action_type) -> float that blends
        learned value with the heuristic prior.

        The heuristic weight *w* decays exponentially as the trainer
        accumulates offline experience:
            w_eff = w * exp(-offline_steps / 500)
        so the learned signal dominates over time.
        """
        trainer = self.get_or_create(env_id)
        extractor = self._feature_extractor

        def _scorer(state_digest: dict, action_type: str) -> float:
            features = extractor.extract(state_digest, action_type)
            learned = trainer.online_net.forward(features)

            action_name = action_type if isinstance(action_type, str) else action_type.name
            heuristic = _HEURISTIC_SCORES.get(action_name, 1) / 10.0

            # Exponential decay of heuristic contribution
            decay = math.exp(-trainer.offline_steps / 500.0)
            w = heuristic_weight * decay

            return (1.0 - w) * learned + w * heuristic

        return _scorer

    # ── diagnostics ──────────────────────────────────────────────────────
    def stats(self, env_id: str) -> Dict[str, Any]:
        trainer = self.get_or_create(env_id)
        history = trainer.loss_history
        return {
            "env_id": env_id,
            "steps": trainer.step_counter,
            "offline_steps": trainer.offline_steps,
            "total_losses": len(history),
            "last_loss": history[-1] if history else None,
            "avg_loss_last_50": (
                float(np.mean(history[-50:])) if len(history) >= 50
                else (float(np.mean(history)) if history else None)
            ),
            "min_loss": float(np.min(history)) if history else None,
            "max_loss": float(np.max(history)) if history else None,
            "loss_history": history,
        }

    def plot_ascii(self, env_id: str, width: int = 60, height: int = 15) -> str:
        """Return a simple ASCII loss-curve plot."""
        trainer = self.get_or_create(env_id)
        history = trainer.loss_history
        if not history:
            return f"[{env_id}] No loss history yet."

        # Down-sample to *width* buckets
        arr = np.array(history, dtype=np.float64)
        if len(arr) > width:
            bucket_size = len(arr) / width
            buckets = []
            for i in range(width):
                lo = int(i * bucket_size)
                hi = int((i + 1) * bucket_size)
                buckets.append(float(np.mean(arr[lo:hi])))
            arr = np.array(buckets)

        lo_val = float(np.min(arr))
        hi_val = float(np.max(arr))
        span = hi_val - lo_val if hi_val != lo_val else 1.0

        lines: List[str] = []
        lines.append(f"Loss curve [{env_id}]  steps={trainer.step_counter}")
        lines.append(f"  {hi_val:>10.6f} |")

        grid = [[" "] * len(arr) for _ in range(height)]
        for col, val in enumerate(arr):
            row = int((val - lo_val) / span * (height - 1))
            row = height - 1 - row  # invert so high values are at the top
            grid[row][col] = "*"

        for r, row_data in enumerate(grid):
            label = ""
            if r == 0:
                label = f"  {hi_val:>10.6f} |"
            elif r == height - 1:
                label = f"  {lo_val:>10.6f} |"
            else:
                label = "             |"
            lines.append(label + "".join(row_data))

        lines.append("             +" + "-" * len(arr))
        lines.append(f"              0{' ' * (len(arr) - 5)}step={len(history)}")
        return "\n".join(lines)

    def compare_with_heuristic(
        self,
        env_id: str,
        state_digest: dict,
        action_types: List[str],
    ) -> str:
        """Return a text table comparing learned value vs heuristic for each action."""
        trainer = self.get_or_create(env_id)
        extractor = self._feature_extractor

        rows: List[str] = []
        header = f"{'Action':<28} {'Learned':>9} {'Heuristic':>10} {'Delta':>8}"
        rows.append(header)
        rows.append("-" * len(header))

        for at in action_types:
            feat = extractor.extract(state_digest, at)
            learned = trainer.online_net.forward(feat)

            action_name = at if isinstance(at, str) else at.name
            heuristic = _HEURISTIC_SCORES.get(action_name, 1) / 10.0

            delta = learned - heuristic
            rows.append(
                f"{action_name:<28} {learned:>9.4f} {heuristic:>10.4f} {delta:>+8.4f}"
            )

        return "\n".join(rows)
