# SocialGuard-RL — Developer Blueprint

> A complete coding plan for building a production-grade reinforcement learning environment for social media integrity moderation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [Folder Structure](#3-folder-structure)
4. [File Responsibilities](#4-file-responsibilities)
5. [Data Contracts](#5-data-contracts)
6. [Build Order](#6-build-order)
7. [Week-by-Week Sprint Plan](#7-week-by-week-sprint-plan)
8. [Module Design Rules](#8-module-design-rules)
9. [Testing Strategy](#9-testing-strategy)
10. [Environment Config](#10-environment-config)
11. [Coding Standards](#11-coding-standards)
12. [Common Pitfalls to Avoid](#12-common-pitfalls-to-avoid)

---

## 1. Project Overview

SocialGuard-RL is a **gym-compatible reinforcement learning environment** that simulates a live social media platform.

An AI agent plays the role of an automated integrity officer. It observes user behavior and network signals, takes moderation actions (`allow`, `warn`, `reduce_reach`, `remove`, `escalate`), and learns from the consequences via a multi-objective reward function.

### Why RL, not a classifier?

Static classifiers treat every decision independently. Social media moderation is a **sequential decision problem under uncertainty**:

- Removing early stops spread but risks false positives on real users
- Waiting for more evidence is safer but lets content propagate
- Actions on one account cascade through the social graph

None of this is learnable from a static labeled dataset.

### The three core tasks

| Task | Difficulty | Core Challenge |
|---|---|---|
| Spam account detection | Easy | Feature-based binary decisions, precision vs recall |
| Viral misinformation flagging | Medium | Temporal dynamics, timing bonus |
| CIB network takedown | Hard | Graph-level reasoning, cluster detection |

---

## 2. Tech Stack

### Core

| Package | Version | Purpose |
|---|---|---|
| `gymnasium` | >= 0.29 | Base `Env` class, spaces, standard API |
| `numpy` | >= 1.24 | Observation vectors, reward math |
| `networkx` | >= 3.0 | Social graph generation and analysis |
| `node2vec` | >= 0.4 | Graph embeddings for Task 3 observation |

### Training

| Package | Version | Purpose |
|---|---|---|
| `stable-baselines3` | >= 2.0 | PPO, DQN, A2C training loops |
| `torch` | >= 2.0 | Backend for SB3 |

### Visualization & Dashboard

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | >= 1.30 | Interactive dashboard |
| `pyvis` | >= 0.3 | Live network graph rendering |
| `matplotlib` | >= 3.7 | Metrics plots |

### Dev & Quality

| Package | Version | Purpose |
|---|---|---|
| `pytest` | >= 7.0 | Unit and integration tests |
| `black` | latest | Code formatting |
| `ruff` | latest | Linting |
| `python-dotenv` | latest | Config management |

---

## 3. Folder Structure

```
socialguard-rl/
│
├── env/
│   ├── __init__.py
│   ├── env.py                  # Core gym.Env subclass
│   ├── spaces.py               # Observation + action space definitions
│   └── rewards.py              # Multi-objective reward function
│
├── sim/
│   ├── __init__.py
│   ├── social_graph.py         # NetworkX graph generator
│   ├── user_behavior.py        # Bot vs human behavior models
│   └── content_gen.py          # Post + spread simulation
│
├── tasks/
│   ├── __init__.py
│   ├── base_task.py            # Abstract base task class
│   ├── task_spam.py            # Task 1 — spam detection
│   ├── task_misinfo.py         # Task 2 — misinformation flagging
│   └── task_cib.py             # Task 3 — CIB network takedown
│
├── graders/
│   ├── __init__.py
│   └── grader.py               # Evaluation: precision, recall, F1, timing
│
├── training/
│   ├── train_ppo.py            # PPO training entry point
│   ├── train_dqn.py            # DQN training entry point
│   └── callbacks.py            # SB3 custom callbacks (logging, early stop)
│
├── dashboard/
│   ├── app.py                  # Streamlit dashboard main
│   ├── graph_view.py           # Network graph component
│   └── metrics_view.py         # Reward and decision charts
│
├── data/
│   ├── __init__.py
│   └── synthetic_graph.py      # Pre-generated graph fixtures for testing
│
├── tests/
│   ├── test_env.py             # Env API correctness
│   ├── test_rewards.py         # Reward function unit tests
│   ├── test_tasks.py           # Task-specific tests
│   └── test_graph.py           # Graph generator tests
│
├── configs/
│   ├── default.yaml            # Default hyperparameters
│   ├── task1.yaml              # Task 1 specific config
│   ├── task2.yaml              # Task 2 specific config
│   └── task3.yaml              # Task 3 specific config
│
├── baseline.py                 # Rule-based heuristic agent (benchmark)
├── evaluate.py                 # Run grader on a trained model
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 4. File Responsibilities

### `env/env.py` — The Core

This is the most important file. Every other module is called from here.

**Responsibilities:**
- Subclass `gymnasium.Env`
- Implement `reset(seed, options)` → returns `(observation, info)`
- Implement `step(action)` → returns `(observation, reward, terminated, truncated, info)`
- Implement `render()` for human-readable output
- Expose `state()` as a custom method returning full internal state dict
- Hold references to: active task, graph engine, behavior engine, content engine, reward engine
- Track: current timestep, episode length, decision history, cumulative reward
- Validate every action before passing to the reward engine

---

### `env/spaces.py` — Input/Output Contracts

**Responsibilities:**
- Define `ObservationSpace` as a `gymnasium.spaces.Box` with exact float bounds per feature
- Define `ActionSpace` as `gymnasium.spaces.Discrete(5)`
- Expose feature names and action name constants so every other module imports from here instead of using magic numbers
- Version the observation vector shape so changes are explicit and traceable

---

### `env/rewards.py` — The Reward Function

This is where the platform's "values" live. Treat it carefully.

**Responsibilities:**
- Accept: the action taken, the ground-truth label of the entity, the entity's legitimacy score, current timestep, task config
- Return: a single float (the reward for this step)
- Internally decompose into five named sub-rewards: `correctness`, `fp_cost`, `collateral_damage`, `speed_bonus`, `escalation_penalty`
- Read coefficients (alpha, beta, gamma, delta, epsilon) from config — never hardcode them
- Log each sub-reward component separately into `info` dict so the agent's reasoning can be audited

**Reward formula:**

```
R = α·correctness
  − β·fp_cost
  − γ·collateral_damage
  + δ·speed_bonus
  − ε·escalation_penalty
```

---

### `sim/social_graph.py` — The Network

**Responsibilities:**
- Generate a NetworkX graph with configurable number of nodes (default 500 for Task 3)
- Use the **planted partition model** to embed a hidden bot cluster of configurable size and density
- Label every node with: `is_bot (bool)`, `legitimacy_score (float 0–1)`, `account_age_days`, `activity_score`
- Expose: `get_neighbors(node_id)`, `get_cluster_ids()`, `get_bot_nodes()`, `get_real_nodes()`
- Support graph evolution: `tick()` adds new edges based on activity, `remove_node(id)` updates the graph and propagates

---

### `sim/user_behavior.py` — The Behavior Model

**Responsibilities:**
- Implement two behavior classes: `HumanBehavior` and `BotBehavior`, both inheriting `BaseBehavior`
- `HumanBehavior`: generates posts with normal-distribution timing, realistic follower ratios, varied content
- `BotBehavior`: generates posts with tight synchronized timing windows, low follower ratios, high content repetition
- Each behavior object produces a structured feature dict matching the observation space spec
- Support configurable noise level — bots should not be perfectly detectable; there is intentional signal overlap

---

### `sim/content_gen.py` — The Content Engine

**Responsibilities:**
- Generate post objects with: `post_id`, `author_id`, `timestamp`, `is_misinfo (bool)`, `spread_rate`, `credibility_score`, `hop_count`
- Simulate spread via breadth-first diffusion through the social graph
- Implement `tick()` to advance content spread by one timestep
- Track and expose: current hop count, total reach, engagement ratio
- Expose `get_content_observation()` returning the content feature vector for the active post

---

### `tasks/base_task.py` — Abstract Task

**Responsibilities:**
- Define the abstract interface all tasks must implement
- Methods: `reset()`, `get_observation()`, `get_ground_truth()`, `is_done()`, `get_info()`
- Hold: task name, difficulty, max steps, action space subset allowed
- Each concrete task subclasses this and overrides only what changes

---

### `tasks/task_spam.py` — Task 1

**Responsibilities:**
- Generate a queue of accounts (70% real, 30% bots — configurable)
- Each step presents one account's feature vector
- Return ground truth label for reward computation
- Terminate after queue is exhausted or max steps reached
- Action space: `{allow, warn, shadow_ban, remove}` — no `escalate` available

---

### `tasks/task_misinfo.py` — Task 2

**Responsibilities:**
- Instantiate content generator and start a piece of content spreading
- Each step presents the current content state (spread rate, hop count, credibility signal, etc.)
- Implement `tick()` to advance spread before returning next observation
- Apply timing bonus: reward degrades with each hop the content has already traveled
- Terminate when: content removed, max hops reached, or max steps reached

---

### `tasks/task_cib.py` — Task 3

**Responsibilities:**
- Generate the full 500-node planted partition graph
- Compute node2vec embeddings at episode start (64-dim per node)
- Each step presents one node's embedding + behavioral features
- Track: how many bot nodes removed, how many real nodes removed (collateral damage count)
- Terminate when: all bots removed, collateral damage exceeds threshold, or max steps reached
- Expose `get_cluster_state()` returning current bot/real/removed counts per cluster

---

### `graders/grader.py` — Evaluation Engine

**Responsibilities:**
- Run N evaluation episodes (default 100) using a provided policy
- Compute over all episodes: precision, recall, F1, mean reward, mean episode length, time-to-detection
- Produce a structured results dict and optionally write to JSON
- Compare trained agent against rule-based baseline and report improvement delta
- Separate metrics per task so mixed evaluations are still readable

---

### `baseline.py` — Rule-Based Benchmark

**Responsibilities:**
- Implement a deterministic rule-based agent using handcrafted thresholds
- Task 1: if `suspicion_score > 0.7` → remove; if `> 0.4` → warn; else allow
- Task 2: if `spread_rate > threshold AND credibility < threshold` → reduce reach; if fact-check flag → remove
- Task 3: if node embedding distance from known bot centroid < threshold → remove
- This is the performance floor the RL agent must beat

---

### `dashboard/app.py` — Streamlit Dashboard

**Responsibilities:**
- Load a trained model via sidebar file picker
- Run live episodes with step-by-step control (play, pause, step)
- Display: network graph (pyvis), current observation values, last action taken, cumulative reward, episode metrics
- Color nodes: red = confirmed bot, green = real user, yellow = under review, gray = removed
- Show decision log table with: timestep, action, ground truth, reward

---

## 5. Data Contracts

### Observation vector (Task 1 — 8 features)

| Index | Name | Type | Range |
|---|---|---|---|
| 0 | account_age_days | float | 0–3650 |
| 1 | posts_per_hour | float | 0–200 |
| 2 | follower_ratio | float | 0–1 |
| 3 | login_time_variance | float | 0–1 |
| 4 | content_repetition_score | float | 0–1 |
| 5 | profile_completeness | float | 0–1 |
| 6 | device_fingerprint_uniqueness | float | 0–1 |
| 7 | ip_diversity_score | float | 0–1 |

### Observation vector (Task 2 — 6 features)

| Index | Name | Type | Range |
|---|---|---|---|
| 0 | spread_rate | float | 0–1 |
| 1 | fact_check_flag | float | 0 or 1 |
| 2 | engagement_ratio | float | 0–1 |
| 3 | source_credibility | float | 0–1 |
| 4 | hop_count | float | 0–20 (normalized) |
| 5 | timestep_normalized | float | 0–1 |

### Observation vector (Task 3 — 68 features)

| Index | Name | Type | Range |
|---|---|---|---|
| 0–63 | node2vec embedding | float | -1 to 1 |
| 64 | degree_centrality | float | 0–1 |
| 65 | clustering_coefficient | float | 0–1 |
| 66 | community_assignment | float | 0–1 (normalized) |
| 67 | posts_per_hour_normalized | float | 0–1 |

### Action space (all tasks)

| ID | Name | Description |
|---|---|---|
| 0 | allow | Take no action — entity continues normally |
| 1 | warn | Issue a warning — no removal, affects trust score |
| 2 | reduce_reach | Shadow visibility reduction |
| 3 | remove | Permanent removal |
| 4 | escalate | Queue for human review |

> **Note:** Not all actions are valid in every task. The task config specifies which action IDs are allowed. The env validates and penalizes invalid action use.

### `step()` return contract

```
observation  : np.ndarray  — shape matches active task's obs space
reward       : float       — scalar reward for this step
terminated   : bool        — episode ended due to task completion
truncated    : bool        — episode ended due to max steps
info         : dict        — {
    ground_truth        : int,
    action_taken        : int,
    reward_breakdown    : dict,
    collateral_count    : int,
    episode_step        : int,
    task_name           : str
}
```

---

## 6. Build Order

Follow this order strictly. Each phase must pass its tests before moving to the next.

```
Phase 1 — Foundation
  1. env/spaces.py        → Define observation + action spaces, feature name constants
  2. env/rewards.py       → Implement reward function with config-driven coefficients
  3. sim/user_behavior.py → Build HumanBehavior and BotBehavior classes

Phase 2 — Task 1 Loop
  4. tasks/base_task.py   → Abstract task interface
  5. tasks/task_spam.py   → Spam detection task
  6. env/env.py           → Core Env wiring Task 1 only
  7. baseline.py          → Rule-based agent for Task 1
  8. tests/test_env.py    → Validate reset/step/spaces/reward

Phase 3 — Simulation
  9. sim/social_graph.py  → NetworkX graph generator
  10. sim/content_gen.py  → Content spread simulation
  11. tasks/task_misinfo.py → Task 2 with timing reward

Phase 4 — Task 3 + Graph RL
  12. tasks/task_cib.py   → CIB takedown task
  13. Integrate node2vec embeddings into env observation pipeline

Phase 5 — Training
  14. training/callbacks.py  → Logging, early stopping
  15. training/train_ppo.py  → PPO training entry point
  16. training/train_dqn.py  → DQN training entry point

Phase 6 — Evaluation
  17. graders/grader.py   → Precision, recall, F1, timing metrics
  18. evaluate.py          → Run grader against trained model + baseline

Phase 7 — Visualization
  19. dashboard/graph_view.py   → pyvis network component
  20. dashboard/metrics_view.py → Reward/decision charts
  21. dashboard/app.py          → Streamlit app wiring

Phase 8 — Packaging
  22. configs/*.yaml       → Finalize all config files
  23. Dockerfile           → Containerize
  24. requirements.txt     → Pin versions
  25. README.md            → Final usage documentation
```

---

## 7. Week-by-Week Sprint Plan

### Week 1 — Core Environment + Task 1

**Goal:** A fully working Task 1 loop that a random agent can run through.

**Day 1–2:** Build `spaces.py` and `rewards.py`. No env yet — just the data contracts and the reward function. Write unit tests for reward edge cases immediately.

**Day 3–4:** Build `user_behavior.py`. Make sure bot and human signals overlap meaningfully — if they're too separable, the task is trivially solved and the RL agent learns nothing.

**Day 5–6:** Build `task_spam.py` and `base_task.py`. Wire `env.py` to run Task 1 only.

**Day 7:** Validate with `gymnasium.utils.env_checker`. Run a random agent for 1000 steps. All actions must produce valid observations. Write `test_env.py`.

**Milestone:** `env.reset()` and `env.step(random_action)` work without errors for Task 1.

---

### Week 2 — Tasks 2 and 3

**Goal:** All three tasks are runnable. Graph embedding pipeline is functional.

**Day 1–2:** Build `social_graph.py`. Generate a 500-node graph with a hidden bot cluster. Verify that `get_bot_nodes()` and `get_real_nodes()` return non-overlapping, correct sets.

**Day 3–4:** Build `content_gen.py`. Implement BFS spread diffusion. Verify that `hop_count` increments correctly across ticks.

**Day 5:** Build `task_misinfo.py`. Integrate timing bonus into reward config.

**Day 6–7:** Build `task_cib.py`. Integrate node2vec. Verify observation shape is (68,). Test that removing a bot node updates the graph correctly before the next observation.

**Milestone:** All three tasks run without errors. Observation shapes match spec for all tasks.

---

### Week 3 — Training + Reward Tuning

**Goal:** A PPO agent learns something nontrivial on Task 1. Reward coefficients produce sensible agent behavior.

**Day 1–2:** Build `training/callbacks.py` and `training/train_ppo.py`. Run Task 1 training for 100k steps. Plot episode reward over time.

**Day 3–4:** Tune reward coefficients. Key signal: if the agent always `remove`s everything, `fp_cost` is too low. If the agent always `allow`s, `correctness` weight is too low. Adjust until the agent learns selective removal.

**Day 5:** Run `train_dqn.py` as comparison. Log both agents' performance.

**Day 6–7:** Build `graders/grader.py` and `evaluate.py`. Run evaluation on 100 episodes. Compare PPO vs DQN vs baseline across precision, recall, F1.

**Milestone:** Trained PPO agent beats the rule-based baseline on Task 1 F1 score.

---

### Week 4 — Visualization + Packaging

**Goal:** The dashboard works. The project runs in Docker.

**Day 1–3:** Build the Streamlit dashboard. The graph view is the hardest part — use pyvis for rendering. Make sure node color updates in real time as the agent takes actions.

**Day 4:** Write `Dockerfile`. Test that `docker build` and `docker run` produce a working environment.

**Day 5:** Finalize `configs/*.yaml`. Every tunable hyperparameter must be in config, not hardcoded.

**Day 6–7:** Final integration test. Run the full pipeline: train → evaluate → dashboard. Fix any remaining issues. Write `README.md` usage section.

**Milestone:** `docker run` starts the dashboard. A trained agent can be loaded and stepped through visually.

---

## 8. Module Design Rules

### The env knows nothing about simulation internals

`env.py` calls task methods. Tasks call sim methods. The env never directly touches the graph, content, or behavior engines. This keeps `env.py` clean and swappable tasks simple.

```
env.py → task.get_observation() → sim.social_graph.get_features(node_id)
```

Not:

```
env.py → sim.social_graph.get_features(node_id)   ← WRONG
```

---

### Every module is independently testable

Each module must be instantiable and callable without the full env running. If you can't write a 5-line test for a module in isolation, the module has too many dependencies.

---

### Config drives everything

No magic numbers anywhere in the codebase. Every threshold, coefficient, episode length, node count, and bot ratio must come from a YAML config loaded at startup. This rule is what makes reward tuning possible without code changes.

---

### Observation space shape never changes at runtime

The observation vector is fixed-length. If a task needs fewer features, pad with zeros. Never change the shape mid-episode or between tasks without resetting the env. SB3 will crash if the space changes.

---

### State is always serializable

`env.state()` must return a plain Python dict with no numpy arrays — only native Python types. This makes replay, logging, and dashboard integration simple.

---

## 9. Testing Strategy

### What to test

**Env API correctness** (`test_env.py`)
- `reset()` returns valid observation matching space shape
- `step()` with every valid action returns `(obs, float, bool, bool, dict)`
- `step()` with invalid action either raises a clear error or returns negative reward
- Episode terminates correctly (terminated vs truncated)
- `gymnasium.utils.env_checker(env)` passes with zero warnings

**Reward function** (`test_rewards.py`)
- Correct removal of a bot returns positive reward
- Incorrect removal of a real user returns negative reward proportional to legitimacy score
- Early removal gets higher reward than late removal (speed bonus)
- Escalation overuse accumulates penalty

**Tasks** (`test_tasks.py`)
- `reset()` produces different episodes with different seeds
- `is_done()` returns True only after terminal condition
- Ground truth labels are internally consistent with generated entities

**Graph** (`test_graph.py`)
- Bot node count matches config ratio
- Real and bot node sets are disjoint
- `remove_node()` removes node from both the graph and the appropriate set
- node2vec embedding shape is (64,)

### When to run tests

Run `pytest tests/` at the end of every build phase before moving to the next. Never start Phase 3 with failing Phase 2 tests.

---

## 10. Environment Config

Example `configs/default.yaml`:

```yaml
env:
  max_steps: 200
  seed: 42
  render_mode: null

task:
  name: task_spam
  bot_ratio: 0.30
  noise_level: 0.15
  action_space: [0, 1, 2, 3, 4]

graph:
  num_nodes: 500
  bot_cluster_size: 80
  intra_cluster_density: 0.4
  inter_cluster_density: 0.05
  embedding_dim: 64

reward:
  alpha: 1.0    # correctness weight
  beta: 0.8     # false positive penalty weight
  gamma: 1.2    # collateral damage weight
  delta: 0.3    # speed bonus weight
  epsilon: 0.1  # escalation overuse penalty

training:
  total_timesteps: 500000
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  eval_freq: 10000
  n_eval_episodes: 20
```

---

## 11. Coding Standards

**File length:** No single file should exceed 300 lines. If it does, the module is doing too much — split it.

**Functions:** Every function does exactly one thing. If the function name needs "and" in it, split it.

**Naming:**
- Classes: `PascalCase`
- Functions and variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Config keys: `snake_case`

**Docstrings:** Every class and every public method gets a one-line docstring minimum. Complex methods get a full docstring with Args and Returns.

**Type hints:** All function signatures must have type hints. The return type of `step()` is `tuple[np.ndarray, float, bool, bool, dict]`.

**Imports:** Group imports in three blocks separated by blank lines — standard library, third-party, local. No wildcard imports.

**No print statements in modules:** Use Python's `logging` module. Set level via config. `print()` is allowed only in `baseline.py` for human-readable output and in `dashboard/app.py`.

---

## 12. Common Pitfalls to Avoid

**Pitfall: Hardcoding bot ratio or reward coefficients.**
Fix: Every tunable number is in YAML config. Import config at module init. Never use literals.

**Pitfall: Observation shape mismatch when switching tasks.**
Fix: All tasks must produce the same shape observation vector. Pad shorter vectors with zeros. Enforce this in `spaces.py`.

**Pitfall: node2vec is slow to train at episode start.**
Fix: Precompute embeddings once during `task_cib.reset()` and cache them. Only recompute when `remove_node()` is called (or re-use stale embeddings with a flag).

**Pitfall: Graph removal doesn't propagate.**
Fix: When `remove_node(id)` is called, update all internal sets immediately: the NetworkX graph, `bot_node_set`, `real_node_set`, and the episode's decision log. Never rely on graph state being consistent without explicitly checking.

**Pitfall: Reward always positive — agent stops learning.**
Fix: Check that false positive penalty actually fires. Add an assertion in `test_rewards.py` that removing a real user with `legitimacy_score > 0.8` always returns a negative reward.

**Pitfall: The dashboard freezes during long episodes.**
Fix: Use `st.empty()` placeholders updated in a loop, not `st.rerun()` on every step. Cap live episode rendering at 10 steps per second using `time.sleep()`.

**Pitfall: Training diverges immediately on Task 3.**
Fix: Start Task 3 training from a Task 1 pre-trained checkpoint (curriculum learning). Do not attempt to train Task 3 from random weights — the sparse graph reward signal makes early exploration essentially random.

**Pitfall: Bot behavior is too different from human behavior — task is trivially solved.**
Fix: Tune `noise_level` in config until the rule-based baseline achieves ~65–75% F1 (not 95%+). If the baseline solves it easily, the RL agent has nothing to learn.

---
## Why RL over a Classifier?
Static classifiers treat every decision independently.
Social media moderation is a sequential decision problem:
- Removing early stops spread but risks false positives
- Waiting is safer but lets content propagate  
- Actions on one account cascade through the social graph
None of this is learnable from a static labeled dataset.

## Quick Start (once built)

```bash
# Install dependencies
pip install -r requirements.txt

# Run Task 1 with random agent
python -c "from env.env import SocialGuardEnv; ..."

# Train PPO on Task 1
python training/train_ppo.py --config configs/task1.yaml

# Evaluate trained model
python evaluate.py --model checkpoints/ppo_task1.zip --task task_spam

# Launch dashboard
streamlit run dashboard/app.py

# Docker
docker build -t socialguard-rl .
docker run -p 8501:8501 socialguard-rl
```

---

*This document is the single source of truth for the SocialGuard-RL build. Update it when any contract, build order, or design decision changes.*
