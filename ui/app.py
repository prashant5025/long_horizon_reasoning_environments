"""
BusinessHorizonENV v2 — Web Dashboard
Flask + Flask-SocketIO backend that wraps the simulation harness
and streams real-time step data to the browser.
"""
from __future__ import annotations

import sys
import os
import time
import threading
from dataclasses import asdict
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from bh_env_v2.engine.types import Action, ActionType, EnvID, Event, Observation
from bh_env_v2.engine.environments.sales import EnterpriseSalesPipeline
from bh_env_v2.engine.environments.pm import ProgramRescueEnvironment
from bh_env_v2.engine.environments.hr_it import ITTransformationEnv
from bh_env_v2.engine.reward_shaping import RewardShaper
from bh_env_v2.memory.memory_system import MemorySystem
from bh_env_v2.planning.planner import HierarchicalPlanner
from bh_env_v2.planning.value_fn import ValueFunctionRegistry
from bh_env_v2.skills.skill_library import (
    ExperienceReplay, SkillExtractor, SkillLibrary, Transition,
)
from bh_env_v2.agents.v2_agent import AgentContext, V2Agent

# ═══════════════════════════════════════════════════════════════════════
#  App Setup
# ═══════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["SECRET_KEY"] = "bh-env-v2-dashboard"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ═══════════════════════════════════════════════════════════════════════
#  Environment Factories
# ═══════════════════════════════════════════════════════════════════════

ENV_FACTORIES = {
    "sales": EnterpriseSalesPipeline,
    "pm": ProgramRescueEnvironment,
    "hr_it": ITTransformationEnv,
}

ENV_ID_MAP = {
    "sales": EnvID.SALES,
    "pm": EnvID.PM,
    "hr_it": EnvID.HR_IT,
}

ENV_INFO = {
    "sales": {
        "name": "Enterprise Sales Pipeline",
        "difficulty": "EXTREME",
        "max_steps": 340,
        "description": "Close a $2.4M enterprise deal across 6 phases with 11 stakeholders.",
        "phases": 6,
    },
    "pm": {
        "name": "Program Rescue",
        "difficulty": "EXTREME",
        "max_steps": 420,
        "description": "Rescue a failing $6M program with 4 workstreams and 47 risks.",
        "phases": 4,
    },
    "hr_it": {
        "name": "IT Transformation",
        "difficulty": "LEGENDARY",
        "max_steps": 480,
        "description": "Migrate 8,000 users while handling 300 instructions and ransomware.",
        "phases": 8,
    },
}

# ═══════════════════════════════════════════════════════════════════════
#  Global State (cross-episode persistent)
# ═══════════════════════════════════════════════════════════════════════

_vf_registry = ValueFunctionRegistry()
_skill_library = SkillLibrary()
_replay_buffer = ExperienceReplay(capacity=100_000)
_skill_extractor = SkillExtractor()
_episode_count = 0
_active_run: Optional[threading.Thread] = None
_stop_flag = threading.Event()

# Per-run episode results cache
_run_results: List[Dict[str, Any]] = []


def _reset_persistent():
    """Reset cross-episode state for a fresh run."""
    global _vf_registry, _skill_library, _replay_buffer, _skill_extractor, _episode_count, _run_results
    _vf_registry = ValueFunctionRegistry()
    _skill_library = SkillLibrary()
    _replay_buffer = ExperienceReplay(capacity=100_000)
    _skill_extractor = SkillExtractor()
    _episode_count = 0
    _run_results = []


# ═══════════════════════════════════════════════════════════════════════
#  Simulation Runner (runs in background thread, emits via SocketIO)
# ═══════════════════════════════════════════════════════════════════════

def _run_simulation(
    env_id: str,
    episodes: int,
    seed: int,
    beam_width: int,
    beam_depth: int,
    reset_state: bool,
):
    """Run episodes in a background thread, emitting per-step and per-episode events."""
    global _episode_count, _active_run

    if reset_state:
        _reset_persistent()

    try:
        for ep in range(episodes):
            if _stop_flag.is_set():
                socketio.emit("run_stopped", {"message": "Run stopped by user"})
                break

            _episode_count += 1
            episode_num = _episode_count
            t0 = time.time()

            # Create environment
            env = ENV_FACTORIES[env_id]()
            ep_seed = seed + ep if seed is not None else None
            obs = env.reset(ep_seed)

            # Per-episode components
            memory = MemorySystem()
            planner = HierarchicalPlanner(
                beam_width=beam_width,
                beam_depth=beam_depth,
            )
            enum_id = ENV_ID_MAP.get(env_id, EnvID.PM)
            planner.init_goal_tree(enum_id)
            reward_shaper = RewardShaper(env_id)

            ctx = AgentContext(
                memory=memory,
                planner=planner,
                skill_library=_skill_library,
                vf_registry=_vf_registry,
                env_id=env_id,
                beam_width=beam_width,
                beam_depth=beam_depth,
            )
            agent = V2Agent(ctx)

            # Record initial events
            if obs.events:
                for event in obs.events:
                    memory.record_event(event)

            socketio.emit("episode_start", {
                "episode": episode_num,
                "env_id": env_id,
                "max_steps": ENV_INFO[env_id]["max_steps"],
                "seed": ep_seed,
            })

            # ── Episode loop ──
            total_reward = 0.0
            shaped_total = 0.0
            reward_history = []
            step_counter = 0
            prev_state_digest = None
            prev_action_type = None
            events_log = []

            while not obs.done:
                if _stop_flag.is_set():
                    break

                state_digest = env.state_digest()
                action = agent.decide(obs, env)
                obs = env.step(action)
                step_counter += 1

                new_state_digest = env.state_digest()
                events_as_dicts = [
                    {"event_type": e.event_type, "tags": e.tags}
                    for e in obs.events
                ]
                shaped_reward, breakdown = reward_shaper.shape(
                    obs.reward, new_state_digest, obs.step, events_as_dicts,
                )

                total_reward += obs.reward
                shaped_total += shaped_reward
                reward_history.append(shaped_reward)

                agent.post_step(obs, action, shaped_reward, new_state_digest)

                # VF online update
                if prev_state_digest is not None and prev_action_type is not None:
                    _vf_registry.online_update(
                        env_id, prev_state_digest, prev_action_type,
                        shaped_reward, new_state_digest,
                        action.action_type.name, obs.done,
                    )

                # Offline update every 25 steps
                if step_counter % 25 == 0 and _replay_buffer.size >= 64:
                    batch = _replay_buffer.sample(64)
                    dicts = _transitions_to_dicts(batch)
                    _vf_registry.offline_update(env_id, dicts)

                # Planner reward signal
                milestone_hit = any(
                    "milestone" in t.lower()
                    for e in obs.events for t in e.tags
                )
                planner.update(obs.step, shaped_reward, milestone_hit)

                prev_state_digest = new_state_digest
                prev_action_type = action.action_type.name

                # Collect event texts for log
                step_events = [e.text for e in obs.events]
                events_log.extend(
                    {"step": obs.step, "text": e.text, "type": e.event_type, "importance": e.importance}
                    for e in obs.events
                )

                # ── Emit step data to frontend ──
                # Collect goal progress
                goal_progress = {}
                for nid, node in planner.goal_tree._nodes.items():
                    if node.parent_id is None:
                        goal_progress[node.name] = round(node.progress, 3)

                # Memory stats
                working_count = len(memory.working._buffer) if hasattr(memory.working, '_buffer') else 0
                episodic_count = len(memory.episodic._events) if hasattr(memory.episodic, '_events') else 0
                semantic_count = len(memory.semantic._beliefs) if hasattr(memory.semantic, '_beliefs') else 0
                compression_stats = {
                    "daily_summaries": len(memory._daily_summaries) if hasattr(memory, '_daily_summaries') else 0,
                    "milestone_summaries": len(memory._milestone_summaries) if hasattr(memory, '_milestone_summaries') else 0,
                    "strategic_insights": len(memory._strategic_insights) if hasattr(memory, '_strategic_insights') else 0,
                }

                step_data = {
                    "episode": episode_num,
                    "step": obs.step,
                    "phase": obs.phase,
                    "action": action.action_type.name,
                    "target_id": action.target_id,
                    "reward": round(obs.reward, 2),
                    "shaped_reward": round(shaped_reward, 2),
                    "total_reward": round(total_reward, 2),
                    "shaped_total": round(shaped_total, 2),
                    "events": step_events[:3],
                    "state": _sanitize_digest(new_state_digest),
                    "goal_progress": goal_progress,
                    "memory": {
                        "working": working_count,
                        "episodic": episodic_count,
                        "semantic": semantic_count,
                        "compression": compression_stats,
                    },
                    "done": obs.done,
                    "breakdown": {k: round(v, 3) for k, v in breakdown.items()} if breakdown else {},
                }
                socketio.emit("step_update", step_data)

                # Throttle slightly to prevent overwhelming the browser
                if step_counter % 5 == 0:
                    time.sleep(0.01)

            # ── End-of-episode ──
            trajectory = agent.trajectory
            for tr in trajectory:
                _replay_buffer.add(tr)

            new_skills = _skill_extractor.extract_from_trajectory(
                trajectory, env_id, episode_num,
            )
            for skill in new_skills:
                _skill_library.register(skill)

            final_digest = env.state_digest()
            agent.update_semantic_memory(obs.step, final_digest)

            # Terminal reason
            terminal_reason = "timeout"
            if final_digest.get("deal_closed"):
                terminal_reason = "deal_closed"
            elif final_digest.get("program_delivered"):
                terminal_reason = "program_delivered"
            elif final_digest.get("migrated_pct", 0) >= 100:
                terminal_reason = "migration_complete"
            elif final_digest.get("budget_runway", 1) <= 0:
                terminal_reason = "budget_exhausted"
            elif final_digest.get("team_morale", 100) <= 0:
                terminal_reason = "morale_collapsed"

            # Goal progress
            root_progress = {}
            for nid, node in planner.goal_tree._nodes.items():
                if node.parent_id is None:
                    root_progress[node.name] = round(node.progress, 3)

            elapsed = time.time() - t0

            episode_result = {
                "episode": episode_num,
                "env_id": env_id,
                "seed": ep_seed,
                "total_steps": step_counter,
                "total_reward": round(total_reward, 2),
                "shaped_reward": round(shaped_total, 2),
                "terminal_reason": terminal_reason,
                "phase_reached": obs.phase,
                "goal_progress": root_progress,
                "elapsed_seconds": round(elapsed, 2),
                "reward_history": [round(r, 2) for r in reward_history],
                "skills_learned": _skill_library.stats()["total_skills"],
                "replay_size": _replay_buffer.size,
                "final_state": _sanitize_digest(final_digest),
                "events_log": events_log[-50:],  # last 50 events
            }
            _run_results.append(episode_result)
            socketio.emit("episode_end", episode_result)

        # ── All episodes done ──
        socketio.emit("run_complete", {
            "total_episodes": len(_run_results),
            "results": _run_results,
        })

    except Exception as e:
        socketio.emit("run_error", {"error": str(e)})
        import traceback
        traceback.print_exc()
    finally:
        _active_run = None
        _stop_flag.clear()


def _sanitize_digest(d: Dict[str, Any]) -> Dict[str, Any]:
    """Make state digest JSON-serializable with rounded numbers."""
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, 2)
        elif isinstance(v, (int, bool, str)):
            out[k] = v
        elif isinstance(v, list):
            out[k] = len(v)
        else:
            out[k] = str(v)
    return out


def _transitions_to_dicts(transitions: List[Transition]) -> List[Dict[str, Any]]:
    dicts = []
    for i, tr in enumerate(transitions):
        if i + 1 < len(transitions):
            nsd = transitions[i + 1].state_digest
            nat = transitions[i + 1].action_type
            done = tr.done
        else:
            nsd = tr.state_digest
            nat = tr.action_type
            done = True
        dicts.append({
            "state_digest": tr.state_digest,
            "action_type": tr.action_type,
            "reward": tr.shaped_reward,
            "next_state_digest": nsd,
            "next_action_type": nat,
            "done": done,
        })
    return dicts


# ═══════════════════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/environments")
def get_environments():
    return jsonify(ENV_INFO)


@app.route("/api/status")
def get_status():
    return jsonify({
        "running": _active_run is not None and _active_run.is_alive(),
        "episode_count": _episode_count,
        "skills": _skill_library.stats()["total_skills"],
        "replay_size": _replay_buffer.size,
        "results": _run_results,
    })


# ═══════════════════════════════════════════════════════════════════════
#  SocketIO Events
# ═══════════════════════════════════════════════════════════════════════

@socketio.on("connect")
def handle_connect():
    emit("connected", {
        "message": "Connected to BusinessHorizonENV Dashboard",
        "environments": ENV_INFO,
    })


@socketio.on("start_run")
def handle_start_run(data):
    global _active_run, _stop_flag

    if _active_run is not None and _active_run.is_alive():
        emit("run_error", {"error": "A simulation is already running. Stop it first."})
        return

    env_id = data.get("env_id", "sales")
    episodes = int(data.get("episodes", 1))
    seed = int(data.get("seed", 42))
    beam_width = int(data.get("beam_width", 4))
    beam_depth = int(data.get("beam_depth", 3))
    reset_state = data.get("reset_state", True)

    if env_id not in ENV_FACTORIES:
        emit("run_error", {"error": f"Unknown environment: {env_id}"})
        return

    _stop_flag.clear()
    _active_run = threading.Thread(
        target=_run_simulation,
        args=(env_id, episodes, seed, beam_width, beam_depth, reset_state),
        daemon=True,
    )
    _active_run.start()
    emit("run_started", {
        "env_id": env_id,
        "episodes": episodes,
        "seed": seed,
    })


@socketio.on("stop_run")
def handle_stop_run():
    _stop_flag.set()
    emit("run_stopping", {"message": "Stopping simulation..."})


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  BusinessHorizonENV v2 Dashboard")
    print("  http://localhost:5000\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
