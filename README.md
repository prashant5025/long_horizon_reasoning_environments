# BusinessHorizonENV

**A Long-Horizon LLM Evaluation Framework for Multi-Step Enterprise Business Simulations**

BusinessHorizonENV is purpose-built to test and improve the planning capabilities of language model agents in complex, multi-step enterprise business simulations. Where standard LLM benchmarks measure single-turn reasoning, this framework demands that an agent maintain a coherent strategy across hundreds to tens of thousands of steps, manage competing stakeholder relationships, recover from unexpected shocks, and track hundreds of interdependent instructions simultaneously.

---

## Key Features

- **Three simulation environments** of escalating difficulty (EXTREME to LEGENDARY)
- **10,000-50,000 step episodes** without exceeding context window limits
- **Hierarchical planning** with Tree-of-Thought beam search
- **Multi-store memory** with 4-level compression (100:1 ratio at strategic level)
- **Cross-episode skill learning** via experience replay and automatic pattern extraction
- **Learned value function** — pure NumPy neural network with TD(0) and target networks
- **Advanced reward shaping** — potential-based (Ng et al. 1999), critical path, dependency bonuses
- **Zero external dependencies** beyond NumPy (all other components use Python stdlib)
- **Deterministic simulations** via seeded LCG RNG for full reproducibility

---

## The Three Environments

| Environment | Class | Difficulty | Max Steps | Description |
|---|---|---|---|---|
| **Enterprise Sales Pipeline** | `EnterpriseSalesPipeline` | EXTREME | 340 | Close a $2.4M deal across 11 stakeholders, 6 phases, and 3 adversarial shocks (champion departure, budget freeze, competitor threat) |
| **Program Rescue** | `ProgramRescueEnvironment` | EXTREME | 420 | Rescue a failing $6M program with 4 workstreams, 47 risks, team morale decay, and 3 shocks (architect departure, vendor bankruptcy, scope creep) |
| **IT Transformation** | `ITTransformationEnv` | LEGENDARY | 480 | Migrate 8,000 users across 4 phases while fulfilling 300 compliance instructions, maintaining SLA, and surviving a ransomware attack |

Each environment features:
- **Scripted adversarial shocks** at predetermined steps that force strategy adaptation
- **Relationship/morale decay** requiring constant multi-stakeholder engagement
- **Phase gates** with prerequisites that prevent shortcut strategies
- **Sparse, delayed rewards** that cannot be gamed by short-sighted policies

---

## Installation

### Prerequisites
- Python 3.10+
- NumPy

### Setup
```bash
git clone https://github.com/<your-username>/BusinessHorizonENV.git
cd BusinessHorizonENV
pip install numpy
```

No other dependencies required. All components beyond the value function use Python standard library.

---

## Quick Start

### CLI

```bash
# Run a single Sales episode with verbose output
python -m bh_env_v2 sales --seed 42 --verbose

# Run 5 PM episodes and print aggregate stats
python -m bh_env_v2 pm -n 5 --seed 42

# Run a 10-episode learning curve with ASCII plots
python -m bh_env_v2 hr_it -n 10 --seed 42 --learning-curve
```

### Python API

```python
from bh_env_v2.eval.v2_harness import V2Harness

# Create harness (owns all cross-episode persistent state)
harness = V2Harness(beam_width=4, beam_depth=3)

# Single episode
result = harness.run_episode('pm', seed=42, verbose=True)
print(result.summary())

# Multi-episode with skill accumulation
results = harness.run_n('hr_it', n=5)
harness.print_aggregate(results)

# Learning curve with loss tracking
harness.run_learning_curve('sales', n=10)

# Inspect the trained value function
print(harness._vf_registry.plot_ascii('pm'))
print(harness._vf_registry.compare_with_heuristic('pm', state, actions))
```

### CLI Reference

| Flag | Description | Default |
|---|---|---|
| `env_id` | Environment: `sales`, `pm`, or `hr_it` | (required) |
| `--episodes, -n` | Number of episodes to run | `1` |
| `--seed, -s` | Random seed for reproducibility | `42` |
| `--verbose, -v` | Print per-step progress every 50 steps | off |
| `--beam-width` | Beam search width for planning | `4` |
| `--beam-depth` | Beam search lookahead depth | `3` |
| `--learning-curve` | Print ASCII reward/loss plots after run | off |

---

## Architecture

The framework is built on a six-layer architecture with six targeted v2 improvements:

```
+-----------------------------------------------------------+
|                      V2 Harness                           |
|  (episode orchestration, cross-episode persistence, CLI)  |
+-----------------------------------------------------------+
|                       V2 Agent                            |
|  (5-stage decision pipeline per step)                     |
+------+----------+---------+----------+--------------------+
|Memory|  Skills   |Planning | Value Fn | Reward Shaping     |
|System|  Library  |  Tree   | (NumPy)  | (3 layers)         |
+------+----------+---------+----------+--------------------+
|              Page Index Engine                             |
|  (semantic chunking, inverted index, snapshots)           |
+-----------------------------------------------------------+
|        Base Environment (Sales / PM / HR_IT)              |
|  (deterministic LCG RNG, action/observation loop)         |
+-----------------------------------------------------------+
```

### V2 Agent Decision Pipeline (5 stages per step)

| Stage | Component | Purpose |
|---|---|---|
| 1 | **Memory Retrieval** | Query episodic memory via TF-IDF; assemble hierarchical context (strategic -> milestone -> daily -> working) |
| 2 | **Skill Check** | Query skill library for learned action patterns matching current context tags |
| 3 | **Hierarchical Planning** | Goal tree decomposition + Tree-of-Thought beam search with hybrid value function |
| 4 | **Action Selection** | Execute matched skill sequence or return planner's best action |
| 5 | **Post-Step Bookkeeping** | Record events to memory, detect phase transitions, track trajectory |

### The Six V2 Improvements

#### 1. Rich Multi-Store Memory Retrieval
Three memory stores inspired by cognitive memory models:

| Store | Capacity | Access | Purpose |
|---|---|---|---|
| **Working Memory** | 20 events | O(1) deque | Hot context buffer of most recent events |
| **Episodic Memory** | 2,000 events | TF-IDF cosine similarity | Long-term event store with importance-weighted retrieval |
| **Semantic Memory** | Unbounded | O(1) dict | Belief map of learned concepts with confidence scores |

#### 2. Hierarchical Planning + Tree of Thought
- **GoalTree**: 3-level decomposition (Goal -> SubGoal -> Task) with automatic upward progress propagation
- **TreeOfThought**: Beam search (width=4, depth=3, gamma=0.9) over action sequences
- **HierarchicalPlanner**: Periodic replanning every 30 steps, environment-specific goal trees (10-13 nodes)

#### 3. Multi-Level Memory Compression
Four compression levels triggered automatically:

| Level | Trigger | Ratio | Output |
|---|---|---|---|
| Raw Events | Every step | 1:1 | `Event` objects |
| Daily Summary | Every 50 steps | ~10:1 | Top events, tag frequencies, net reward |
| Milestone Summary | Phase transition | ~30:1 | Outcomes, risks, shock/milestone counts |
| Strategic Insight | Every 200 steps | ~100:1 | Risk pattern detection, trend analysis, recommendations |

#### 4. Cross-Episode Skill Learning
- **SkillExtractor**: Sliding window (length 2-6) scans trajectories for high-reward sub-sequences
- **ExperienceReplay**: Prioritised circular buffer (100K capacity), persists across episodes
- **SkillLibrary**: Jaccard-based matching with usage-weighted relevance scoring

#### 5. Advanced Reward Shaping
Three additive layers through the `RewardShaper`:

| Layer | Method | Guarantee |
|---|---|---|
| **Progress** | Potential-based shaping: F(s,s') = gamma * phi(s') - phi(s) | Preserves optimal policy (Ng et al. 1999) |
| **Critical Path** | DAG milestone bonuses (+5 * weight) and miss penalties (-3) | Environment-specific milestone graphs |
| **Dependency** | Resolution bonuses (+3 per unlocked node) | Tracks blocking relationships |

#### 6. Environment Scale
- **ScaledEnvironment**: Wraps any environment, multiplies all timing (10x-100x scale)
- **MultiDepartmentEnvironment**: N independent departments under shared corporate budget with cross-department blockers

### Learned Value Function
Pure NumPy neural network replacing the static heuristic:

- **FeatureExtractor**: 32-dimensional state-action encoding (phase, step, one-hot action, environment-specific features, heuristic prior, pressure signals)
- **ValueNetwork**: 32 -> ReLU(64) -> ReLU(32) -> Linear(1), He/Kaiming init, Adam optimizer with L2 regularization and gradient clipping
- **Target Network**: Frozen copy synced every 50 steps (prevents bootstrap instability, per Mnih et al. 2015)
- **Training**: Online TD(0) per step + offline minibatch (64 samples) every 25 steps
- **Hybrid Scorer**: Blends learned value with heuristic prior, heuristic weight decays exponentially as training accumulates

---

## Project Structure

```
bh_env_v2/
|-- __init__.py
|-- __main__.py                  # CLI entry: python -m bh_env_v2
|
|-- engine/
|   |-- __init__.py
|   |-- types.py                 # All dataclasses and enums (Action, Event, Observation, etc.)
|   |-- page_index.py            # Semantic chunking, inverted tag index, state snapshots
|   |-- reward_shaping.py        # 3-layer reward shaping (potential, critical path, dependency)
|   |-- environments/
|       |-- __init__.py
|       |-- base.py              # Abstract base with LCG RNG and page-index integration
|       |-- sales.py             # EnterpriseSalesPipeline (EXTREME, 340 steps)
|       |-- pm.py                # ProgramRescueEnvironment (EXTREME, 420 steps)
|       |-- hr_it.py             # ITTransformationEnv (LEGENDARY, 480 steps)
|       |-- scaled.py            # ScaledEnvironment + MultiDepartmentEnvironment
|
|-- memory/
|   |-- __init__.py
|   |-- memory_system.py         # Working/Episodic/Semantic memory + 4-level compression
|   |-- neural_retrieval.py      # [NEW] Neural/hybrid embedding retrieval (Upgrade 3)
|
|-- planning/
|   |-- __init__.py
|   |-- planner.py               # GoalTree + TreeOfThought + HierarchicalPlanner
|   |-- value_fn.py              # FeatureExtractor, ValueNetwork, Trainer, Registry
|
|-- skills/
|   |-- __init__.py
|   |-- skill_library.py         # ExperienceReplay, SkillExtractor, SkillLibrary
|
|-- agents/
|   |-- __init__.py
|   |-- v2_agent.py              # V2Agent 5-stage decision pipeline
|   |-- multi_agent.py           # [NEW] Multi-agent coordination (Upgrade 1)
|   |-- llm_agent.py             # [NEW] Real LLM integration (Upgrade 2)
|
|-- eval/
    |-- __init__.py
    |-- v2_harness.py            # V2Harness orchestration, learning curves, CLI
    |-- scale_validator.py       # [NEW] 100K+ step scale validation (Upgrade 4)
```

**~8,100 lines of Python** across 27 files.

---

## Design Decisions

### Why TF-IDF and Not Neural Embeddings for Memory
The observation vocabulary is machine-generated and structurally constrained (e.g., `Step 47 phase=2. poc_score=72.`). TF-IDF with IDF weighting correctly handles the rare-is-informative property without adding a PyTorch/ONNX dependency (400MB+) or 10-50ms latency per retrieval. Neural embeddings should be revisited when a real LLM agent generates free-text observations with high vocabulary variance.

### Why Pure NumPy for the Neural Network
The ValueNetwork is tiny (~4,300 parameters). Manual backpropagation made diagnosing three training instabilities (reward explosion, bootstrap spiral, trace accumulation) far easier. Forward/backward pass takes <1ms, so CUDA transfer would provide no speedup.

### Why TD(0) and Not TD(lambda)
TD(lambda) with eligibility traces was tried and produced catastrophically large weight updates. The trace accumulates continuously during 400+ steps of zero reward; when a non-zero reward finally arrives, the gradient magnitude is proportional to trajectory length. TD(0) is numerically stable, and offline minibatch replay provides the long-range credit assignment.

### Environment-Written vs Agent-Written Snapshots
Snapshots are written by the environment (ground truth), not the agent (beliefs that may contain misconceptions). This is deliberate for a rigorous evaluation framework.

### Potential-Based Reward Shaping
The Ng et al. 1999 guarantee ensures F(s,s') = gamma * phi(s') - phi(s) leaves the optimal policy unchanged. Shaped rewards provide dense gradient signal without introducing local optima.

---

## Training Stability Notes

Three critical instabilities were encountered and resolved during value function development:

| Problem | Cause | Fix |
|---|---|---|
| **Reward Explosion** | Un-normalised rewards (range -11,000 to +600) produced MSE loss ~10^10 | Fixed-range normalisation: reward/20, clipped to [-1, +1] |
| **Bootstrap Spiral** | Online network used for both prediction and bootstrap target (Deadly Triad) | Target network synced every 50 steps (per DQN, Mnih et al. 2015) |
| **Trace Accumulation** | TD(lambda) traces accumulated over 400+ zero-reward steps | Removed eligibility traces entirely; use plain TD(0) + offline replay |

---

## Advanced Features (Upgrades 1-4)

All four planned upgrades have been implemented and are accessible via CLI flags or Python API.

### Upgrade 1: Multi-Agent Coordination

Three specialised department agents (Engineering, Product, Finance) with genuine information asymmetry.

```bash
python -m bh_env_v2 pm --multi-agent --seed 42 --verbose
```

**Components:**
- **ObservationFilter**: Whitelists state fields per department. Engineering cannot see budget runway; Finance cannot see velocity metrics or morale; neither sees the full risk register.
- **Blackboard Protocol**: Structured message passing (REQUEST, INFORM, PROPOSE, ACCEPT, REJECT, ESCALATE) between agents. Auto-responses to information requests and escalation proposals.
- **SharedRewardAttributor**: Three strategies — EQUAL (r/n), ACTIVITY (proportional to actions), SHAPLEY (marginal contribution estimation).

```python
from bh_env_v2.agents.multi_agent import MultiAgentCoordinator, RewardAttribution
from bh_env_v2.engine.environments.pm import ProgramRescueEnvironment
from bh_env_v2.engine.environments.scaled import MultiDepartmentEnvironment

coordinator = MultiAgentCoordinator(vf_registry, skill_library, reward_strategy=RewardAttribution.SHAPLEY)
env = MultiDepartmentEnvironment(env_factories=[ProgramRescueEnvironment] * 3)
result = coordinator.run_episode(env, seed=42, verbose=True)
print(result.summary())
```

### Upgrade 2: Real LLM Agent Integration

Places Claude or GPT in the planning loop with latency-aware scheduling.

```bash
# Using Claude (requires ANTHROPIC_API_KEY env var)
python -m bh_env_v2 sales --llm claude --seed 42 --verbose

# Using OpenAI (requires OPENAI_API_KEY env var)
python -m bh_env_v2 pm --llm openai --llm-model gpt-4o --llm-every-n 10

# Optional: pip install anthropic   OR   pip install openai
```

**Components:**
- **ClaudeLLM / OpenAILLM**: API wrappers with retry logic and JSON response parsing
- **PromptBuilder**: Constructs system/user prompts from memory context, state digest, and valid actions
- **LLMAgent**: Queries the LLM every N steps (default: 5), uses heuristic planner in between for latency budgeting
- Beam depth auto-reduced from 3 to 1 for API-bound decisions (~500ms per LLM call vs <1ms heuristic)

### Upgrade 3: Neural Memory Retrieval

Dense embedding retrieval blended with TF-IDF for hybrid scoring.

```bash
python -m bh_env_v2 sales --neural-memory --seed 42
```

**Components:**
- **SentenceTransformerModel**: Wraps sentence-transformers for semantic embeddings (optional: `pip install sentence-transformers`)
- **NumpyEmbeddingModel**: Lightweight random-projection baseline (NumPy only, no extra deps)
- **HybridEpisodicMemory**: Blends `alpha * neural_score + (1-alpha) * tfidf_score`, falls back to TF-IDF when neural unavailable
- **NeuralMemorySystem**: Drop-in replacement for `MemorySystem` with all compression logic inherited

### Upgrade 4: 100K+ Step Scale Validation

Validates framework correctness and memory fidelity at 10x-100x scale.

```bash
# Validate PM at 5x scale (2,100 steps)
python -m bh_env_v2 pm --validate-scale 5 --seed 42 --verbose

# Validate at 10x scale (4,200 steps)
python -m bh_env_v2 pm --validate-scale 10 --seed 42
```

**Components:**
- **FidelityMetrics**: Measures recall of shock events, milestones, phase transitions, and high-reward events from compressed memory
- **IDFDriftTracker**: Periodic snapshots of IDF corpus; measures vocabulary growth, top-term stability (Jaccard), and average IDF value drift
- **ScaleValidationHarness**: Runs scaled episodes with comprehensive reporting

```python
from bh_env_v2.eval.scale_validator import ScaleValidationHarness

validator = ScaleValidationHarness()
results = validator.validate_all_scales('pm', scales=[1, 5, 10], seed=42)
validator.print_comparison(results)
```

### Complete CLI Reference

| Flag | Description | Default |
|---|---|---|
| `env_id` | Environment: `sales`, `pm`, or `hr_it` | (required) |
| `--episodes, -n` | Number of episodes | `1` |
| `--seed, -s` | Random seed | `42` |
| `--verbose, -v` | Per-step progress | off |
| `--beam-width` | Beam search width | `4` |
| `--beam-depth` | Beam search depth | `3` |
| `--learning-curve` | ASCII reward/loss plots | off |
| `--multi-agent` | Multi-agent coordination mode | off |
| `--llm {claude,openai}` | Use real LLM in planning loop | off |
| `--llm-model MODEL` | Override LLM model name | auto |
| `--llm-every-n N` | LLM call frequency (steps) | `5` |
| `--neural-memory` | Hybrid neural+TF-IDF retrieval | off |
| `--validate-scale N` | Run scale validation at Nx | off |

---

## Example Output

### Single Episode (Sales)
```
============================================================
  Episode Result  [sales]  seed=42
============================================================
  Steps:          18
  Total Reward:   +483.00
  Shaped Reward:  +529.78
  Phase Reached:  6
  Terminal:       deal_closed
  Elapsed:        0.48s
  Goal Progress:
    Close $2.4M deal: [###########---------] 53%
  Value Function:
    Steps: 17
    Last Loss: 0.955464
  Skills: 8 learned
============================================================
```

### Learning Curve (5 Episodes)
```
Learning curve: sales x 5 episodes
------------------------------------------------------------
  Episode 1/5: steps=18 reward=+529.8 phase=6 (deal_closed) [0.2s]
  Episode 2/5: steps=19 reward=+529.8 phase=6 (deal_closed) [0.1s]
  Episode 3/5: steps=17 reward=+542.9 phase=6 (deal_closed) [0.1s]
  Episode 4/5: steps=18 reward=+532.8 phase=6 (deal_closed) [0.1s]
  Episode 5/5: steps=18 reward=+529.8 phase=6 (deal_closed) [0.1s]

  Shaped Reward:
  ep  1:     +529.8 |+++++++++++++++++++++++++++++++++++++++
  ep  2:     +529.8 |+++++++++++++++++++++++++++++++++++++++
  ep  3:     +542.9 |++++++++++++++++++++++++++++++++++++++++
  ep  4:     +532.8 |+++++++++++++++++++++++++++++++++++++++
  ep  5:     +529.8 |+++++++++++++++++++++++++++++++++++++++
```

### Multi-Agent Coordination
```
============================================================
  Multi-Agent Episode Result
============================================================
  Steps:      2100
  Reward:     +19491.00
  Terminal:   timeout
  Messages:   1257
  Elapsed:    36.40s

  Per-Department Rewards:
     engineering: +6497.00  (700 actions)
         product: +6497.00  (700 actions)
         finance: +6497.00  (700 actions)
============================================================
```

### Scale Validation (5x)
```
============================================================
  Scale Validation Result  [pm x5]
============================================================
  Steps:             1301
  Reward:            +11160.50
  Compression Cycles:6
  Peak Memory Events:1306

============================================================
  Memory Fidelity Report
============================================================
  Compression Ratio:  40.8:1

  Recall Rates:
    Shock Events:     1.6% (14/864)
    Milestones:       100.0% (0/0)
    Phase Transitions:100.0% (0/0)

  Strategic Insights: 6
  Insight Quality:    1.00/1.0

============================================================
  IDF Corpus Drift Report
============================================================
  Step Range         Vocab Growth  Top Stability  IDF Drift
------------------------------------------------------------
  1->501                     +508         0.1250     3.5530
  501->1001                    +8         0.9048     0.8138
------------------------------------------------------------
  Average top-term stability: 0.5149
  Average IDF drift:          2.1834
============================================================
```

---

## Reward Structure

### Sales
| Event | Reward |
|---|---|
| Contact stakeholder | +2 |
| POC score milestones (25/50/75) | +15/+20/+30 |
| Phase advance | +8 |
| Deal closed | +500 |
| Shock events | -30 |
| Relationship decay | -0.5/step per low-engagement stakeholder |

### PM
| Event | Reward |
|---|---|
| Risk resolved | +5 to +12 (by severity) |
| Workstream advance | +5 to +10 |
| Phase advance | +10 |
| Program delivered | +500 |
| Morale collapse | -100 |
| Budget exhausted | -200 |

### HR/IT
| Event | Reward |
|---|---|
| Instruction fulfilled | +2 |
| Cohort migrated (200 users) | +10 |
| Full migration complete | +300 |
| SLA breach per step | -3 |
| Overdue instruction per step | -1 |
| Ransomware shock | -50 |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
