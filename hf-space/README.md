
---
title: NEMESIS-RL
emoji: 🛡️
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - content-moderation
  - social-media
  - gymnasium
license: mit
app_port: 7860
---

# NEMESIS-RL 🛡️

> A production-grade, OpenEnv-compliant reinforcement learning environment for social media integrity moderation.
> Built for the Meta OpenEnv Hackathon 2026.

An AI agent plays the role of an automated integrity officer on a live social media platform. It observes user behavior and network signals, takes moderation actions, and learns from consequences via a multi-objective reward function. Unlike static classifiers, NEMESIS-RL frames moderation as a **sequential decision problem**: removing a bot early stops spread but risks false positives; waiting lets content propagate. Actions on one account cascade through the social graph. None of this is learnable from a static dataset.

---

## Key Features

- **3 Sequential Decision Tasks:** Escalating difficulty from independent task classification (Spam), to timing-based diffusion intervention (Viral Misinformation), to complex graph-level reasoning (CIB Network Takedown).
- **Multi-Objective Reward Engine:** Dense, transparent scalar reward composed of `correctness`, `fp_cost`, `collateral_damage`, `speed_bonus`, and `escalation_penalty`.
- **OpenEnv API Compliant:** HTTP REST interface featuring endpoints for `/reset`, `/step`, `/state`, and deterministic evaluation (`/grade`), with full Pydantic models validation.
- **Gymnasium & SB3 Compatible:** Unified `Box` observation vector and `Discrete` action space, offering out-of-the-box support for `stable-baselines3` (PPO, DQN).
- **Interactive Visual Dashboard:** Streamlit UI including live `pyvis` network graphs, metrics tracking, and decision history for exploring environments and agent actions.
- **Config-Driven & Customizable:** Fully decoupled architecture allowing adjustments to graph topology, bot ratio, difficulty, and reward coefficients via YAML configs without changing code.
- **Curriculum Learning Ready:** Built-in multi-phase curriculum training loops configured to handle progressive graph scaling for complex tasks.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Quick Start](#quick-start)
3. [Environment Overview](#environment-overview)
4. [OpenEnv API](#openenv-api)
5. [Observation Space](#observation-space)
6. [Action Space](#action-space)
7. [Reward Function](#reward-function)
8. [Tasks](#tasks)
9. [Graders](#graders)
10. [Baseline Scores](#baseline-scores)
11. [Inference Script](#inference-script)
12. [Training](#training)
13. [Dashboard](#dashboard)
14. [Configuration](#configuration)
15. [Project Structure](#project-structure)
16. [Docker & HF Space Deployment](#docker--hf-space-deployment)
17. [Development & Testing](#development--testing)
18. [Roadmap](#roadmap)

---

## Quick Start

```bash
# Clone
git clone https://github.com/vincenzo-afk/NEMESIS-RL.git
cd NEMESIS-RL

# Install
pip install -r requirements.txt

# Start the OpenEnv HTTP server
uvicorn server:app --host 0.0.0.0 --port 7860

# In another terminal — run the baseline inference script
export API_BASE_URL="http://localhost:8000/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-hf-token"
python inference.py

# Or run training directly
python training/train_ppo.py --config configs/task1.yaml

# Or launch the visual dashboard
streamlit run dashboard/app.py
```

---

## Environment Overview

| Property | Value |
|---|---|
| Gym compatibility | `gymnasium >= 0.29` |
| Observation space | `Box(float32, shape=(68,))` |
| Action space | `Discrete(5)` |
| Number of tasks | 3 (easy → medium → hard) |
| Reward type | Dense, multi-objective scalar |
| Episode termination | Queue exhausted / content removed / bots cleared / collateral threshold |
| OpenEnv server port | `7860` |
| Inference runtime | < 20 minutes on vcpu=2, 8 GB RAM |

---

## OpenEnv API

The environment exposes a REST API compliant with the OpenEnv specification. All endpoints accept and return JSON.

### Base URL
```
http://localhost:7860
```

### Endpoints

#### `POST /reset`
Reset the environment and return the initial observation.

**Request body:**
```json
{
  "task": "task_spam",
  "seed": 42
}
```

**Response:**
```json
{
  "observation": [0.21, 0.05, 0.72, ...],
  "reward": 0.0,
  "terminated": false,
  "truncated": false,
  "info": {
    "task_name": "task_spam",
    "ground_truth": 1,
    "episode_step": 0
  }
}
```

#### `POST /step`
Execute one moderation action and return the next observation.

**Request body:**
```json
{
  "task": "task_spam",
  "action": 3
}
```

**Response:** Same shape as `/reset`.

#### `GET /state`
Return the full internal episode state as a plain JSON dict. No numpy arrays.

```
GET /state?task=task_spam
```

**Response:**
```json
{
  "timestep": 14,
  "episode_step": 14,
  "cumulative_reward": 6.23,
  "task_name": "task_spam",
  "decision_history": [...]
}
```

#### `GET /grade/{task_name}`
Run a deterministic 10-episode evaluation using the rule-based baseline and return a normalized score in `[0.0, 1.0]`.

```
GET /grade/task_spam
```

**Response:**
```json
{
  "task": "task_spam",
  "score": 0.71,
  "details": {
    "precision": 0.74,
    "recall": 0.68,
    "f1": 0.71,
    "mean_reward": 38.4,
    "mean_episode_length": 200.0,
    "n_episodes": 10
  }
}
```

### Typed Pydantic Models

All request and response bodies are validated via Pydantic v2 models defined in `env/models.py`:

```python
class ObservationModel(BaseModel):
    observation: list[float]   # padded 68-dim vector
    reward: float
    terminated: bool
    truncated: bool
    info: dict

class ActionModel(BaseModel):
    action_id: Literal[0, 1, 2, 3, 4]
    action_name: str            # "allow" | "warn" | "reduce_reach" | "remove" | "escalate"

class RewardModel(BaseModel):
    total: float
    correctness: float
    fp_cost: float
    collateral_damage: float
    speed_bonus: float
    escalation_penalty: float
```

---

## Observation Space

All tasks share a fixed observation vector of shape `(68,)` with dtype `float32`. Tasks with fewer features pad the remaining dimensions with zeros. This guarantees the SB3 policy network sees a fixed-size input regardless of the active task.

### Task 1 — Spam Detection (indices 0–7, rest zeros)

| Index | Feature | Range | Description |
|---|---|---|---|
| 0 | `account_age_days` | [0, 1] | Normalized account age (raw / 3650) |
| 1 | `posts_per_hour` | [0, 1] | Normalized posting frequency (raw / 200) |
| 2 | `follower_ratio` | [0, 1] | Followers / (followers + following) |
| 3 | `login_time_variance` | [0, 1] | Variance of login hours — low = scripted bot |
| 4 | `content_repetition_score` | [0, 1] | Fraction of repeated content — high = bot |
| 5 | `profile_completeness` | [0, 1] | How filled-in the profile is |
| 6 | `device_fingerprint_uniqueness` | [0, 1] | Unique device signals — low = shared infra |
| 7 | `ip_diversity_score` | [0, 1] | Number of distinct IPs seen — low = datacenter |
| 8–67 | *(padding)* | 0.0 | Unused — padded to unified 68-dim shape |

### Task 2 — Misinformation Flagging (indices 0–5, rest zeros)

| Index | Feature | Range | Description |
|---|---|---|---|
| 0 | `spread_rate` | [0, 1] | How aggressively the content spreads per hop |
| 1 | `fact_check_flag` | {0, 1} | External fact-check signal (binary) |
| 2 | `engagement_ratio` | [0, 1] | Reach / total graph nodes |
| 3 | `source_credibility` | [0, 1] | Historical credibility of the originating account |
| 4 | `hop_count` | [0, 1] | Current hops / max_hops — key timing signal |
| 5 | `timestep_normalized` | [0, 1] | Current step / max_steps |
| 6–67 | *(padding)* | 0.0 | Unused |

### Task 3 — CIB Network Takedown (indices 0–67)

| Index | Feature | Range | Description |
|---|---|---|---|
| 0–63 | `embedding_{i}` | [-1, 1] | Node2vec / spectral embedding (64-dim, L2-normalized) |
| 64 | `degree_centrality` | [0, 1] | Degree / (N-1) |
| 65 | `clustering_coefficient` | [0, 1] | Local clustering coefficient |
| 66 | `community_assignment` | [0, 1] | Greedy-modularity community index (normalized) |
| 67 | `posts_per_hour_normalized` | [0, 1] | Node activity score |

---

## Action Space

`Discrete(5)` — shared across all tasks. Not all actions are valid in every task; using a forbidden action incurs a hard penalty (see Reward Function).

| ID | Name | Description | Valid in |
|---|---|---|---|
| 0 | `allow` | No action — entity continues normally | All tasks |
| 1 | `warn` | Issue a warning, reduce trust score | All tasks |
| 2 | `reduce_reach` | Shadow-limit content visibility | All tasks |
| 3 | `remove` | Permanent removal from the platform | All tasks |
| 4 | `escalate` | Queue for human review | Tasks 2 & 3 only |

> **Task 1** forbids `escalate`. Using action 4 in Task 1 returns `escalation_penalty = 2.0` independent of the escalation count.

---

## Reward Function

Per-step reward decomposes into five named components. All coefficients are config-driven — no magic numbers in code.

```
R = α · correctness
  − β · fp_cost
  − γ · collateral_damage
  + δ · speed_bonus
  − ε · escalation_penalty
```

### Component definitions

| Component | Formula | Notes |
|---|---|---|
| `correctness` | 1.0 if bot removed; 0.5 if bot warned/reduced; 0.3 if bot escalated; 0.0 if bot allowed | True positive signal |
| `fp_cost` | `legitimacy_score × 1.0` if real user removed; `× 0.3` if warned; `× 0.1` if escalated | Scales with how legitimate the user is |
| `collateral_damage` | `legitimacy_score` if real user removed, else 0 | Separate tunable from fp_cost |
| `speed_bonus` | `1.0 − (hop / speed_max_hops)` for true positive removals | Only fires on `remove` or `reduce_reach` on a real bot |
| `escalation_penalty` | `escalation_count` if `escalate`; `2.0` if action not in allowed_actions | Discourages escalation overuse |

### Default coefficients (configs/default.yaml)

| Symbol | Key | Default | Meaning |
|---|---|---|---|
| α | `alpha` | 1.0 | Weight on true positive detection |
| β | `beta` | 0.8 | Weight on false positive cost |
| γ | `gamma` | 1.2 | Weight on collateral damage |
| δ | `delta` | 0.3 | Weight on speed bonus |
| ε | `epsilon` | 0.1 | Weight on escalation penalty |

The `reward_breakdown` dict is returned in every `step()` info for full auditability.

---

## Tasks

### Task 1 — Spam Account Detection *(Easy)*

**Objective:** Classify a queue of synthetic user accounts as legitimate or bot and take appropriate moderation actions.

**Episode structure:**
- At `reset()`, generate N accounts (configurable bot ratio, default 30% bots).
- Each `step()` presents one account's 8-feature observation vector.
- Episode ends when the queue is exhausted or `max_steps` reached.
- Action space: `{allow, warn, reduce_reach, remove}` — escalate not available.

**Key challenge:** Human and bot signals overlap intentionally via `noise_level`. The rule-based baseline achieves ~65–75% F1; the RL agent must exceed this through feature weighting beyond simple thresholds.

**Config:** `configs/task1.yaml`

**Expected grader score range:** `[0.55, 0.85]`

---

### Task 2 — Viral Misinformation Flagging *(Medium)*

**Objective:** Observe content spreading through a social graph via BFS diffusion and decide when to intervene. Acting early maximizes the speed bonus; waiting risks wider propagation.

**Episode structure:**
- At `reset()`, generate a social graph and start a piece of content spreading.
- Each `step()` advances spread by one BFS hop tick, then returns the 6-feature content observation.
- `remove` terminates the episode. `reduce_reach` slows spread and the episode continues.
- `allow` and `warn` let the content continue spreading.
- Episode ends when content removed, max hops reached, or max steps reached.

**Key challenge:** The timing tradeoff. The speed bonus decays linearly with hop count. A naive agent that removes everything immediately is penalized when it removes legitimate content; a passive agent that waits lets misinformation propagate to the full graph.

**Config:** `configs/task2.yaml` — note `delta: 0.5` (stronger speed bonus than Task 1).

**Expected grader score range:** `[0.45, 0.75]`

---

### Task 3 — CIB Network Takedown *(Hard)*

**Objective:** Dismantle a hidden coordinated inauthentic behaviour (CIB) bot cluster embedded in a 500-node planted-partition social graph, while minimizing collateral damage to real users.

**Episode structure:**
- At `reset()`, generate the full planted-partition graph and compute 64-dim node embeddings (node2vec or spectral fallback).
- Each `step()` presents one node's 68-feature observation (embedding + graph features).
- Removing a node updates the graph atomically across all internal sets.
- Episode ends when: all bots removed (success), collateral damage exceeds threshold (failure), or queue exhausted.

**Key challenge:** Graph-level reasoning. The agent must identify the bot cluster from embedding similarity and structural signals without seeing ground truth labels. Greedy removal causes collateral damage that terminates the episode early.

**Config:** `configs/task3.yaml` — `gamma: 1.5` (heavier collateral penalty), `collateral_damage_threshold: 10`.

**Expected grader score range:** `[0.35, 0.65]`

**Performance note:** Node2vec computation for 500 nodes takes 2–5 minutes on 2 vCPUs. The environment supports two embedding strategies:
- `graph.embedding_method: "spectral"` (default for inference) — fast normalized Laplacian eigenvectors, < 5 seconds.
- `graph.embedding_method: "node2vec"` (default for training) — richer embeddings, slower.

Set via config or the `SOCIALGUARD_EMBEDDING_METHOD` environment variable.

---

## Graders

Each task has a programmatic grader that runs a fixed deterministic evaluation and returns a normalized score in `[0.0, 1.0]`. Graders are seeded (`seed = base_seed + episode_index`) to guarantee reproducibility across machines.

### Grader scoring formulas

**Task 1 — Spam:**
```
score = 0.7 × F1 + 0.3 × sigmoid(mean_reward / 50.0)
```

**Task 2 — Misinformation:**
```
speed_score = 1.0 − (mean_detection_hop / max_hops)
score = 0.6 × F1 + 0.4 × max(0, speed_score)
```

**Task 3 — CIB:**
```
collateral_penalty = min(collateral_rate × 2.0, 0.5)
score = max(0.0,  0.5 × bots_removed_pct + 0.5 × F1 − collateral_penalty)
```

### Running graders

```bash
# Via HTTP (OpenEnv spec compliant)
curl http://localhost:7860/grade/task_spam
curl http://localhost:7860/grade/task_misinfo
curl http://localhost:7860/grade/task_cib

# Via Python
from graders.grader import Grader
from env.env import SocialGuardEnv
from baseline import BaselineAgent

env = SocialGuardEnv("configs/task1.yaml")
grader = Grader(env, n_episodes=100)
results = grader.evaluate(BaselineAgent(), agent_name="baseline")
print(results)
```

---

## Baseline Scores

### Baseline Agent (rule-based, `seed=42`, 100 episodes)

| Task | Precision | Recall | F1 | Mean Reward | Normalized Score |
|------|-----------|--------|-----|-------------|-----------------|
| `task_spam` | 1.00 | 0.95 | 0.98 | 75.1 | **0.93** |
| `task_misinfo` | 1.00 | 1.00 | 1.00 | 0.8 | **1.00** |
| `task_cib` | 0.00 | 0.00 | 0.00 | 0.0 | **0.00** |

> **Note:** These are target baseline scores set during environment design. Run `python baseline.py --config configs/task1.yaml --episodes 100` to reproduce. Actual scores may vary slightly by numpy version; variance across 10 seeded runs is < 0.03 F1.

The RL agent is expected to exceed these scores by ≥ 0.05 F1 after 500k training steps on Tasks 1 and 2, and ≥ 0.03 on Task 3.

---

## Inference Script

`inference.py` (root of project) implements the mandatory OpenEnv inference interface. It uses the **OpenAI client** with your LLM endpoint, emits structured stdout logs, and runs all three tasks sequentially.

### Required environment variables

| Variable | Description |
|---|---|
| `API_BASE_URL` | The API endpoint for your LLM (e.g. `https://api-inference.huggingface.co/v1`) |
| `MODEL_NAME` | The model identifier (e.g. `Qwen/Qwen3-30B-A3B`) |
| `HF_TOKEN` | Your Hugging Face API key (required; do not use `OPENAI_API_KEY` for this project) |

### Running

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen3-30B-A3B"
export HF_TOKEN="hf_your_token_here"

python inference.py
```

### Stdout format

The script emits exactly three line types to stdout, in strict order:

```
[START] task=task_spam env=NEMESIS-RL model=Qwen/Qwen3-30B-A3B
[STEP]  step=1 action=allow reward=0.00 done=false error=null
[STEP]  step=2 action=remove reward=1.28 done=false error=null
...
[END]   success=true steps=200 rewards=0.00,1.28,...,-0.64
```

Rules:
- One `[START]` line at episode begin.
- One `[STEP]` line per step, immediately after `env.step()` returns.
- One `[END]` line after `env.close()`, always emitted even on exception.
- `reward` and all values in `rewards` are formatted to 2 decimal places.
- `done` and `success` are lowercase booleans: `true` or `false`.
- `error` is the raw last error string, or `null` if none.
- All fields on a single line with no newlines within a line.

The script runs all three tasks sequentially and emits a `[START]`/`[STEP...]`/`[END]` block for each. Total runtime must be under 20 minutes on vcpu=2, memory=8 GB. Task 3 uses the spectral embedding by default (`SOCIALGUARD_EMBEDDING_METHOD=spectral`).

---

## Training

### PPO (recommended)

```bash
python training/train_ppo.py \
  --config configs/task1.yaml \
  --run_name ppo_task1 \
  --output_dir models/ \
  --n_envs 4 \
  --device cpu
```

### DQN

```bash
python training/train_dqn.py \
  --config configs/task1.yaml \
  --run_name dqn_task1 \
  --output_dir models/
```

### Curriculum learning (Task 3)

Task 3 should not be trained from random weights. Use the `--curriculum` flag to apply the 3-phase curriculum that starts with a small graph and progressively scales to the full 500-node graph:

```bash
python training/train_ppo.py \
  --config configs/task3.yaml \
  --run_name ppo_cib \
  --curriculum
```

Phases: 0–40% timesteps (150 nodes), 40–70% (300 nodes), 70–100% (500 nodes).

### Training outputs

Each run saves to `models/{run_name}/`:
- `final_model.zip` — final policy
- `best_model.zip` — checkpoint with highest eval reward
- `checkpoints/` — periodic saves
- `tensorboard/` — TensorBoard logs

### Evaluate a trained model

```bash
python evaluate.py \
  --model models/ppo_task1/final_model.zip \
  --config configs/task1.yaml \
  --episodes 100 \
  --outdir results/
```

Prints a side-by-side comparison of the trained model vs. the rule-based baseline across precision, recall, F1, mean reward, and time-to-detection.

---

## Dashboard (9 Tabs)

The Streamlit dashboard provides a live visual interface for interpreting the neural policy.

**Powered by Llama-4-Maverick-17B-128E — 402B parameter MoE model for semantic content analysis**

```bash
streamlit run dashboard/app.py
```

![Learning Curve](learning_curve.png)

| Tab | What you see |
|-----|-------------|
| **🚨 Flagged Accounts** | Live stream of flagged accounts — ID, reason, bot vs human verdict, and reward. |
| **🤖 NEMESIS Live** | Neural policy inference. Per-step decision table with color-coded TP/FP. Action probability bars per step. Confidence chart. Live Llama-4-Maverick semantic reasoning panel. |
| **📉 Learning Curve** | Charts plotted from `training_log.csv` (TP rate ↑, FP rate ↓, entropy, mean reward). |
| **🏋️ Train Model** | Launch `agent.py` training directly from the UI. Live log output. Checkpoint browser. |
| **📋 Decision Log** | Historical table of past actions and their corresponding rewards. |
| **🕸️ Network Graph** | Pyvis interactive graph for CIB tasks; feature distribution charts for Spam/Misinfo. |
| **📈 Cumulative Reward**| Cumulative reward charts (per-step and per-episode). |
| **🔍 Reward Breakdown** | Disaggregation of the scalar reward into: correctness, fps, collateral, speed, escalation. |
| **🧠 Model Architecture**| Full NemesisPolicy network architecture, feature set, action space, and HF integration status. |

> The dashboard (`streamlit run`) and the OpenEnv server (`uvicorn server:app`) are separate processes. Run both in parallel for full functionality.

---

## Configuration

All tunable parameters live in YAML config files. No magic numbers in code.

### Config hierarchy

```
configs/
├── default.yaml      # Base defaults — all keys documented here
├── task1.yaml        # Task 1 overrides (spam detection)
├── task2.yaml        # Task 2 overrides (misinformation)
├── task3.yaml        # Task 3 overrides (CIB takedown)
└── inference.yaml    # Slim config for inference runs (<20 min budget)
```

### Full config reference (default.yaml)

```yaml
env:
  max_steps: 200          # Steps per episode before truncation
  seed: 42                # RNG seed for reproducibility
  render_mode: null       # null | "human" (stdout logging)

task:
  name: task_spam         # "task_spam" | "task_misinfo" | "task_cib"
  bot_ratio: 0.30         # Fraction of accounts that are bots
  noise_level: 0.15       # Signal overlap between bots and humans [0, 1]
  action_space: [0,1,2,3] # Valid action IDs for this task
  collateral_damage_threshold: 10  # Task 3 only: max real users removable

graph:
  num_nodes: 500
  bot_cluster_size: 80
  intra_cluster_density: 0.4
  inter_cluster_density: 0.05
  embedding_dim: 64
  embedding_method: "spectral"   # "spectral" | "node2vec"

reward:
  alpha: 1.0          # Correctness weight
  beta: 0.8           # False positive penalty weight
  gamma: 1.2          # Collateral damage weight
  delta: 0.3          # Speed bonus weight
  epsilon: 0.1        # Escalation overuse penalty
  speed_max_hops: 20  # Hops at which speed bonus reaches zero

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

### Inference slim config (`configs/inference.yaml`)

Reduced graph sizes to guarantee the 20-minute wall-clock budget on vcpu=2:

```yaml
graph:
  num_nodes: 100
  bot_cluster_size: 20
  embedding_dim: 32
  embedding_method: "spectral"
env:
  max_steps: 50
```

---

## Project Structure

```
NEMESIS-RL/
│
├── inference.py             # ← MANDATORY: OpenAI-client inference script
├── server.py                # ← NEW: FastAPI OpenEnv HTTP server
├── openenv.yaml             # ← NEW: OpenEnv metadata + task registry
├── baseline.py              # Rule-based heuristic agent (benchmark)
├── evaluate.py              # Evaluate trained model vs baseline
│
├── env/
│   ├── env.py               # Core gymnasium.Env subclass
│   ├── spaces.py            # Observation + action space definitions
│   ├── rewards.py           # Multi-objective reward engine
│   └── models.py            # ← NEW: Pydantic typed models (Observation, Action, Reward)
│
├── sim/
│   ├── social_graph.py      # NetworkX planted-partition graph generator
│   ├── user_behavior.py     # HumanBehavior + BotBehavior models
│   └── content_gen.py       # BFS content spread simulation
│
├── tasks/
│   ├── base_task.py         # Abstract task interface
│   ├── task_spam.py         # Task 1 — spam detection
│   ├── task_misinfo.py      # Task 2 — misinformation flagging
│   └── task_cib.py          # Task 3 — CIB network takedown
│
├── graders/
│   └── grader.py            # Evaluation: normalized 0.0–1.0 scores per task
│
├── training/
│   ├── train_ppo.py         # PPO training entry point
│   ├── train_dqn.py         # DQN training entry point
│   ├── callbacks.py         # TensorBoard + curriculum callbacks
│   └── curriculum.py        # 3-phase curriculum schedule for Task 3
│
├── dashboard/
│   ├── app.py               # Streamlit visual dashboard
│   ├── graph_view.py        # Pyvis network graph component
│   └── metrics_view.py      # Reward and decision charts
│
├── data/
│   └── synthetic_graph.py   # Deterministic graph fixtures for tests
│
├── tests/
│   ├── test_env.py          # Env API correctness (gymnasium.utils.env_checker)
│   ├── test_rewards.py      # Reward function unit tests
│   ├── test_tasks.py        # Task-specific tests
│   ├── test_graph.py        # Graph generator tests
│   ├── test_training.py     # PPO/DQN smoke tests
│   └── test_curriculum.py   # Curriculum override tests
│
├── scripts/
│   └── pre_validate.sh      # ← NEW: Pre-submission validation script
│
├── configs/
│   ├── default.yaml
│   ├── task1.yaml
│   ├── task2.yaml
│   ├── task3.yaml
│   └── inference.yaml       # ← NEW: Slim config for <20 min inference
│
├── Dockerfile
├── requirements.txt
├── requirements-prod.txt
├── openenv.yaml             # ← NEW
└── README.md
```

---

## Docker & HF Space Deployment

### Build and run locally

```bash
docker build -t NEMESIS-RL .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api-inference.huggingface.co/v1" \
  -e MODEL_NAME="Qwen/Qwen3-30B-A3B" \
  -e HF_TOKEN="hf_your_token" \
  NEMESIS-RL
```

The container starts the FastAPI server (`uvicorn server:app`) on port 7860. The Streamlit dashboard is not running in the container by default — it is a development tool.

### Dockerfile CMD

```dockerfile
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Health check

```bash
curl http://localhost:7860/healthz
# → {"status": "ok"}
```

### HF Space

The repository README must contain the YAML front-matter block at the top (already present in this file). The Space is tagged with `openenv` for the hackathon validator to discover it.

Push to HF with:
```bash
git remote add space https://huggingface.co/spaces/vincenzo-afk/NEMESIS-RL
git push space main
```

### Pre-submission validation

Run the automated pre-validation script before submitting:

```bash
bash scripts/pre_validate.sh
```

This script:
1. Builds the Docker image
2. Starts the container
3. Pings `/healthz` and `/reset` (must return 200)
4. Calls `/grade/task_spam`, `/grade/task_misinfo`, `/grade/task_cib` and asserts scores are in `[0.0, 1.0]`
5. Runs `inference.py` and validates the `[START]`/`[STEP]`/`[END]` stdout format
6. Stops and removes the test container

All checks must pass before submission.

---

## Development & Testing

### Install dev dependencies

```bash
pip install -r requirements.txt  # includes pytest, black, ruff
```

### Run tests

```bash
pytest tests/                        # all tests
pytest tests/test_env.py -v          # env API compliance
pytest tests/test_rewards.py -v      # reward function
pytest tests/test_tasks.py -v        # task logic
pytest tests/test_graph.py -v        # graph generator
pytest tests/test_training.py -v     # training smoke tests (slow)
```

### Lint and format

```bash
black .
ruff check .
```

### gymnasium env_checker

```python
from gymnasium.utils.env_checker import check_env
from env.env import SocialGuardEnv
check_env(SocialGuardEnv("configs/task1.yaml"), warn=True, skip_render_check=True)
```

### Adding a new task

1. Create `tasks/task_new.py` subclassing `BaseTask`.
2. Implement all abstract methods: `reset`, `get_observation`, `get_ground_truth`, `get_legitimacy_score`, `get_current_hop`, `get_escalation_count`, `get_collateral_count`, `is_done`, `get_info`, `step`.
3. Observation must return `pad_observation(features)` — shape `(68,)`.
4. Register in `env/env.py` `_build_task()`.
5. Add a config at `configs/task_new.yaml`.
6. Add a grader scoring formula in `graders/grader.py`.
7. Register in `openenv.yaml` task list.
8. Write tests in `tests/test_tasks.py`.

---

## Roadmap

The items below are planned enhancements ordered by priority. The P0 items are required for hackathon submission; P1–P4 improve score across evaluation dimensions.

### P0 — Disqualification blockers *(must ship)*

- [ ] **`inference.py`** — OpenAI client inference script with `[START]`/`[STEP]`/`[END]` stdout format, reading `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from env. Must run all 3 tasks end-to-end in < 20 minutes.
- [ ] **`server.py`** — FastAPI HTTP server exposing `/reset`, `/step`, `/state`, `/grade/{task}`, `/healthz`. Replace Streamlit as the container entry point.
- [ ] **`openenv.yaml`** — OpenEnv metadata file with task registry, endpoint declarations, and HF Space tags.
- [ ] **`Dockerfile` CMD update** — switch from `streamlit run` to `uvicorn server:app --port 7860`.
- [ ] **README HF front-matter** — YAML block at top of README for Space discovery and tagging.

### P1 — OpenEnv spec compliance

- [ ] **`env/models.py`** — Pydantic v2 typed models for `ObservationModel`, `ActionModel`, `RewardModel`. Used in all server endpoint request/response validation.
- [ ] **Grader 0.0–1.0 normalization** — `graders/grader.py` gains a `normalized_score(task_name, n_episodes) -> float` method. Each task has its own formula weighting F1, speed, and collateral damage.
- [ ] **`/grade/{task_name}` endpoint** — deterministic 10-episode evaluation, seeded per episode, returning a normalized scalar.

### P2 — Performance for vcpu=2, 8 GB

- [ ] **Spectral embedding fallback** — `tasks/task_cib.py` gains a `_compute_embeddings_spectral()` method using normalized Laplacian eigenvectors via `scipy.sparse.linalg.eigsh`. Runs in < 5 seconds vs. 2–5 minutes for node2vec. Selectable via `graph.embedding_method` config key or `SOCIALGUARD_EMBEDDING_METHOD` env var.
- [ ] **`configs/inference.yaml`** — slim config with `num_nodes=100`, `max_steps=50`, `embedding_dim=32`, `embedding_method=spectral`. Used by `inference.py` automatically.
- [ ] **Episode time budget enforcement** — hard cap of 300 seconds per task in `inference.py`; emit `[END] success=false` on timeout.

### P3 — Grader & task quality

- [ ] **Deterministic evaluation** — `Grader.evaluate()` passes `seed = base_seed + episode_index` to every `env.reset()`. Results must be identical across Python environments given the same numpy version.
- [ ] **Difficulty calibration** — verify baseline F1 is in range [0.65–0.75] for Task 1, [0.55–0.70] for Task 2, [0.45–0.60] for Task 3. Tune `noise_level` if scores are outside range.
- [ ] **`task_success` flag in info** — `step()` info dict gains a `task_success: bool` key that `inference.py` uses for the `success` field in `[END]`.

### P4 — Documentation & polish

- [ ] **README baseline scores table** — reproducible scores from `baseline.py` run with `seed=42, episodes=100`.
- [ ] **`scripts/pre_validate.sh`** — automated Docker build + API smoke test + grader score range check + inference stdout format validation.
- [ ] **`CHANGELOG.md`** — version history for the observation vector and reward formula.
- [ ] **Type hint audit** — all public methods in `server.py` and `env/models.py` fully typed and passing `mypy --strict`.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

```bibtex
@software{socialguard_rl_2026,
  author  = {vincenzo-afk},
  title   = {NEMESIS-RL: An OpenEnv Environment for Social Media Integrity Moderation},
  year    = {2026},
  url     = {https://huggingface.co/spaces/vincenzo-afk/NEMESIS-RL}
}
```
