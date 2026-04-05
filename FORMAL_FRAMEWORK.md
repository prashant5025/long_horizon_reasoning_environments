# Formal Framework: BusinessHorizonENV v2

**A Mathematical Foundation for Long-Horizon LLM Evaluation**

---

## 1. Problem Formulation

### 1.1 Extended-Horizon POMDP

We formalize each BusinessHorizonENV environment as a **Partially Observable Markov Decision Process** (POMDP) extended for long-horizon operation:

```
M = (S, A, T, R, Omega, O, gamma, H)
```

| Symbol | Definition |
|--------|-----------|
| `S` | State space (environment-specific, see Section 2) |
| `A` | Action space, `\|A\| = 12` action types (Section 1.2) |
| `T: S x A -> Delta(S)` | Stochastic transition function (LCG-deterministic given seed) |
| `R: S x A x S -> R` | Reward function (base + shaped, Section 3) |
| `Omega` | Observation space (partial view of state) |
| `O: S -> Omega` | Observation function (`state_digest()`) |
| `gamma in (0, 1)` | Discount factor, `gamma = 0.95` |
| `H` | Horizon length: `H in {340, 420, 480}` |

**Key distinction from standard RL benchmarks**: `H >> 100` creates challenges in credit assignment, memory management, and planning depth that standard benchmarks (`H ~ 50-200`) do not test.

### 1.2 Action Space

The action space is shared across all environments:

```
A = {CONTACT_STAKEHOLDER, RUN_POC, ADVANCE_DEAL, RESOLVE_RISK,
     ADVANCE_WORKSTREAM, ALLOCATE_BUDGET, FULFILL_INSTRUCTION,
     MIGRATE_COHORT, RESPOND_SHOCK, BOOST_MORALE, REVIEW_STATUS, NOOP}
```

The **valid action set** `A_valid(s) subset A` is state-dependent. Not all actions are valid in all states, enforcing context-dependent decision-making.

### 1.3 Phased Structure

Each environment has a phase function `phi: S -> {1, ..., P}` where `P in {4, 6, 8}`. Phases introduce:
- Non-stationary dynamics: `T_phi != T_phi'` for `phi != phi'`
- Irreversible state transitions (phase cannot decrease)
- Shock events at predetermined steps (exogenous perturbations)

**Definition 1 (Shock Event)**: A shock at step `t_shock` is an exogenous state perturbation:
```
s_{t_shock+1} = sigma(s_{t_shock})
```
where `sigma: S -> S` is a deterministic corruption function that degrades multiple state variables simultaneously.

---

## 2. Environment State Spaces

### 2.1 EnterpriseSalesPipeline (`H = 340, P = 6`)

```
S_sales = (step, phase, {stakeholder_i}_{i=1}^{11}, poc_score, deal_value,
           budget_frozen, deal_closed, shocks_triggered)
```

where each `stakeholder_i = (engagement_i, is_champion_i, is_detractor_i, active_i)`.

**Dimension**: `|S_sales| ~ 11 * 4 + 6 = 50` continuous/discrete variables.

**Shock schedule**: `sigma_1` at `t=80` (champion departure), `sigma_2` at `t=170` (budget freeze), `sigma_3` at `t=260` (competitor threat).

### 2.2 ProgramRescueEnvironment (`H = 420, P = 4`)

```
S_pm = (step, phase, {workstream_j}_{j=1}^{4}, {risk_k}_{k=1}^{47},
        team_morale, budget, program_delivered, shocks_triggered)
```

where `workstream_j = (progress_j, blocked_j, dependencies_j)` and `risk_k = (severity_k, resolved_k)`.

**Dimension**: `|S_pm| ~ 4 * 3 + 47 * 2 + 5 = 111` variables.

**Shock schedule**: `sigma_1` at `t=100`, `sigma_2` at `t=220`, `sigma_3` at `t=340`.

### 2.3 ITTransformationEnv (`H = 480, P = 8`)

```
S_hrit = (step, phase, {instruction_l}_{l=1}^{300}, migrated_users,
          total_users, sla_score, ticket_queue, ransomware_active,
          shocks_triggered)
```

**Dimension**: `|S_hrit| ~ 300 * 2 + 7 = 607` variables.

**Shock schedule**: Ransomware attack `sigma_1` at `t=240`.

---

## 3. Reward Shaping

### 3.1 Potential-Based Shaping (Ng et al., 1999)

**Theorem 1 (Policy Invariance)**: The shaped reward `R' = R + F` preserves the set of optimal policies when the shaping function has the form:
```
F(s, s') = gamma * Phi(s') - Phi(s)
```
for any bounded potential function `Phi: S -> R`.

**Our implementation**: We define the potential as a weighted linear combination of normalized progress metrics:
```
Phi(s) = sum_{i} w_i * f_i(s)
```
where `f_i` are environment-specific features (e.g., `poc_score / 100`, `morale / 100`, `migrated_pct / 100`) and `w_i` are hand-tuned weights.

### 3.2 Three-Layer Shaping Decomposition

The total shaped reward decomposes as:
```
R_shaped(s, a, s') = R_base(s, a, s') + F_potential(s, s') + B_critical(s, a) + B_dependency(s, a)
```

| Component | Source | Purpose |
|-----------|--------|---------|
| `R_base` | Environment | Sparse task reward |
| `F_potential` | Potential-based (Thm 1) | Dense progress signal |
| `B_critical` | Critical path DAG | Milestone ordering bonus |
| `B_dependency` | Dependency graph | Prerequisite resolution bonus |

**Note**: Only `F_potential` satisfies Theorem 1 exactly. `B_critical` and `B_dependency` are auxiliary bonuses that may alter the optimal policy but empirically improve learning speed by 40-60% in early episodes.

---

## 4. Memory System

### 4.1 Three-Store Architecture

The agent maintains three memory stores `M = (M_W, M_E, M_S)`:

| Store | Type | Capacity | Access Pattern |
|-------|------|----------|----------------|
| `M_W` (Working) | FIFO deque | `\|M_W\| = 20` | Last-k events, O(1) |
| `M_E` (Episodic) | TF-IDF indexed | `\|M_E\| <= 2000` | Similarity search, O(n * v) |
| `M_S` (Semantic) | Key-value beliefs | `\|M_S\|` unbounded | Exact lookup, O(1) |

### 4.2 TF-IDF Retrieval Formalization

For episodic memory retrieval, given query `q` and stored event `e_i`:

```
score(q, e_i) = cosine(tfidf(q), tfidf(e_i)) * importance(e_i)
```

where:
```
tfidf(d)_t = tf(t, d) * idf(t)
tf(t, d) = count(t in d) / |d|
idf(t) = log(N / (1 + df(t)))
```

`N` = total documents in corpus, `df(t)` = document frequency of term `t`.

**Complexity**: Retrieval is `O(|M_E| * |V|)` where `|V|` is vocabulary size. For `H = 480`, this grows linearly but remains tractable (`< 5ms` per query on commodity hardware).

### 4.3 Four-Level Compression

To manage memory growth over long horizons, we apply progressive compression:

```
Raw Events -(50 steps)-> DailySummary -(phase change)-> MilestoneSummary -(200 steps)-> StrategicInsight
```

| Level | Compression Ratio | Trigger | Information Preserved |
|-------|------------------|---------|-----------------------|
| Level 0: Raw Event | 1:1 | Every step | Full event text, tags, reward |
| Level 1: DailySummary | ~10:1 | Every 50 steps | Top events, tag frequencies, net reward |
| Level 2: MilestoneSummary | ~30:1 | Phase transition | Outcomes, risks, shock/milestone counts |
| Level 3: StrategicInsight | ~100:1 | Every 200 steps | Recurring risks, trend, recommendation |

**Proposition 1 (Memory Boundedness)**: For any episode of length `H`, the total memory footprint is bounded:
```
|M_total| <= |M_W| + H + floor(H/50) + P + floor(H/200) = O(H)
```
In practice, with compression, the *active* memory accessed per query is `O(|M_W| + k)` where `k` is the retrieval budget (default `k = 5`).

### 4.4 Hybrid Neural Retrieval (Optional)

When enabled, retrieval blends dense and sparse scores:
```
score_hybrid(q, e_i) = alpha * cosine(emb(q), emb(e_i)) + (1 - alpha) * score_tfidf(q, e_i)
```

where `emb: text -> R^d` is either a sentence-transformer (`d = 384`) or a lightweight NumPy random projection (`d = 128`), and `alpha in [0, 1]` (default `alpha = 0.5`).

---

## 5. Planning System

### 5.1 Hierarchical Goal Decomposition

The planner maintains a goal tree `G = (V, E)` where `V` contains nodes at three levels:

```
V = V_goal union V_subgoal union V_task
```

Each node `v in V` has attributes `(name, status, priority, deadline, progress)`.

**Progress propagation**: For a parent node `v` with children `C(v)`:
```
progress(v) = (1 / |C(v)|) * sum_{c in C(v)} progress(c)
```

### 5.2 Tree-of-Thought Beam Search

At each decision step, we perform beam search over action sequences:

**Algorithm**: `BeamSearch(s, A_valid, value_fn, width=4, depth=3)`
```
1. Initialize beam B_0 = {(s, [], 0)}   // (state, action_seq, cumulative_score)
2. For d = 1, ..., depth:
   a. B_candidates = {}
   b. For each (s_i, seq_i, score_i) in B_{d-1}:
      For each a in A_valid(s_i):
        score_new = score_i + gamma^d * value_fn(s_i, a)
        B_candidates = B_candidates union {(s_i, seq_i + [a], score_new)}
   c. B_d = top-width elements of B_candidates by score_new
3. Return argmax_{(s, seq, score) in B_depth} score  // first action of best sequence
```

**Complexity**: `O(width * |A| * depth)` evaluations per step. With defaults (`width=4, |A|=12, depth=3`), this is `144` evaluations per step — negligible compared to environment step cost.

### 5.3 Value Function

The learned value function `V_theta: S x A -> R` is a 3-layer MLP:
```
V_theta(s, a) = W_3 * ReLU(W_2 * ReLU(W_1 * phi(s, a) + b_1) + b_2) + b_3
```

Architecture: `32 -> 64 -> 32 -> 1` (implemented in pure NumPy).

**Training**: Online TD(0) updates:
```
delta = r + gamma * V_{theta^-}(s', a') - V_theta(s, a)
theta <- theta + alpha * delta * grad_theta V_theta(s, a)
```

where `theta^-` is a target network updated via Polyak averaging (`tau = 0.01`).

Optimizer: Adam (`alpha = 0.001, beta_1 = 0.9, beta_2 = 0.999`).

---

## 6. Cross-Episode Learning

### 6.1 Experience Replay

A circular buffer `D` of capacity `|D| = 100,000` stores transitions:
```
D = {(s_t, a_t, r_t, s_{t+1}, done_t)}
```

Sampling is prioritized by `|r_t| + 10 * done_t`, biasing toward high-reward and terminal transitions.

### 6.2 Skill Extraction

**Definition 2 (Skill)**: A skill `sigma = (context, action_sequence, value)` where:
- `context subset Tags` is the set of tags under which the skill was discovered
- `action_sequence in A^*` is a sequence of `2-6` actions
- `value` is the cumulative reward observed during execution

**Extraction**: Sliding window of sizes `[2, 3, 4, 5, 6]` over episode trajectories. Patterns observed `>= 2` times with positive reward are registered as skills.

**Matching**: Jaccard similarity between current context tags and skill context:
```
match(tags, sigma) = |tags intersect sigma.context| / |tags union sigma.context|
```

Skill is activated if `match > 0.3` and `sigma.value > 0`.

---

## 7. Multi-Agent Coordination

### 7.1 Information Asymmetry

In multi-agent mode, the full state `s` is projected to department-specific partial observations:
```
o_d = pi_d(s)    for d in {ENGINEERING, PRODUCT, FINANCE}
```

where `pi_d: S -> O_d` is a whitelist-based filter that removes department-inaccessible fields.

**Property 1**: No agent can reconstruct the full state from its observation alone:
```
For all d: dim(o_d) < dim(s)
```

### 7.2 Blackboard Communication

Agents communicate via a shared blackboard `BB`:
```
BB = {(sender, recipient, type, content, priority, resolved)}
```

Message types: `{REQUEST, INFORM, PROPOSE, ACCEPT, REJECT, ESCALATE}`

### 7.3 Reward Attribution

Given joint reward `R` and `n` agents, we support three attribution schemes:

| Scheme | Formula | Properties |
|--------|---------|------------|
| Equal | `r_i = R / n` | Fair, ignores contribution |
| Activity | `r_i = R * (actions_i / sum_j actions_j)` | Proportional to effort |
| Shapley | `r_i = sum_{C subset N\{i}} (|C|!(n-|C|-1)!/n!) * [v(C union {i}) - v(C)]` | Game-theoretically fair |

**Shapley approximation**: We use permutation sampling with `m = 100` permutations, giving an approximation error of `O(1/sqrt(m))`.

---

## 8. Complexity Analysis

### 8.1 Per-Step Computational Cost

| Component | Complexity | Typical Time |
|-----------|-----------|-------------|
| Environment step | `O(|S|)` | `< 0.1ms` |
| Memory retrieval | `O(|M_E| * |V|)` | `< 5ms` |
| Beam search | `O(W * |A| * D)` | `< 1ms` |
| VF forward pass | `O(d_1 * d_2 + d_2 * d_3)` | `< 0.1ms` |
| Skill matching | `O(|Skills| * |Tags|)` | `< 0.1ms` |
| **Total (heuristic)** | | **`< 10ms/step`** |
| **Total (with LLM)** | + API call | **`~500-2000ms/step`** |

### 8.2 Per-Episode Space

```
O(H + |M_E| + |replay_buffer|) = O(H + 2000 + 100,000) = O(100,000)
```

### 8.3 Scaling Behavior

For scale factor `k` (extending horizon by `k`x):
- Memory growth: `O(k * H)` raw events, `O(k * H / 50)` daily summaries
- TF-IDF vocabulary: `O(sqrt(k * H))` (Heaps' law)
- IDF drift: Empirically measured at `drift = 2.18` for `k = 5` (see scale validation)

---

## 9. Theoretical Guarantees and Limitations

### 9.1 What We Guarantee

1. **Reward shaping policy invariance** (Theorem 1): The potential-based component `F` preserves optimal policies. Formally verified by the structure `F(s,s') = gamma * Phi(s') - Phi(s)`.

2. **Deterministic reproducibility**: Given seed `s`, the LCG RNG produces identical episode trajectories. The LCG parameters `(a=1664525, c=1013904223, m=2^32)` produce period `2^32`.

3. **Memory boundedness** (Proposition 1): Memory consumption is `O(H)` with compression, preventing unbounded growth.

### 9.2 What We Do NOT Guarantee

1. **Convergence of the value function**: The MLP value function with online TD(0) is not guaranteed to converge in the function approximation setting (Tsitsiklis & Van Roy, 1997). We use a target network and small learning rate as practical stabilization.

2. **Optimality of beam search**: Beam search with finite width is a greedy approximation. It may miss globally optimal action sequences that require temporarily low-value actions.

3. **Skill extraction completeness**: The sliding-window approach may miss skills that span more than 6 steps or require non-contiguous action patterns.

4. **Shapley value exactness**: The permutation-sampling approximation introduces error bounded by `O(R_max / sqrt(m))`.

---

## 10. Evaluation Protocol

### 10.1 Baseline Agents

To contextualize results, we compare against:

| Baseline | Description | Purpose |
|----------|-------------|---------|
| `Random` | Uniform random action | Lower bound on any strategy |
| `Greedy` | Single-step heuristic max | Tests whether planning adds value |
| `RuleBased` | Hand-crafted domain rules | Human-level heuristic ceiling |
| `PlannerOnly` | Planner without memory/VF | Ablation: isolates planner contribution |
| `V2Agent` | Full system | Upper bound (heuristic agent) |
| `LLM-{model}` | Real LLM in planning loop | Evaluates LLM long-horizon capability |

### 10.2 Metrics

For each agent across `n >= 5` episodes with different seeds:

1. **Mean shaped reward** with 95% confidence interval
2. **Success rate**: fraction of episodes reaching terminal goal
3. **Mean phase reached**: progress through environment phases
4. **Efficiency**: steps to completion (lower is better for successful episodes)
5. **LLM-specific**: tokens used, latency, cost, parse failure rate

### 10.3 Statistical Methodology

- Report mean +/- standard deviation and 95% CI (t-distribution)
- Compare agent pairs using Welch's t-test (unequal variance)
- Effect size reported as Cohen's d
- Minimum 5 episodes per configuration (different seeds)

---

## References

1. Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. *ICML*.
2. Tsitsiklis, J. N., & Van Roy, B. (1997). An analysis of temporal-difference learning with function approximation. *IEEE Transactions on Automatic Control*.
3. Yao, S., et al. (2023). Tree of Thoughts: Deliberate problem solving with large language models. *NeurIPS*.
4. Shapley, L. S. (1953). A value for n-person games. *Contributions to the Theory of Games*.
5. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*.
6. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*.
