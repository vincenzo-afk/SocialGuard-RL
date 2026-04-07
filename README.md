# 🛡️ NEMESIS-RL

### *Train AI agents to moderate social media — at scale, in real time, with Llama.*

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-orange)](https://huggingface.co/spaces/vincenzo-afk/NEMESIS-RL)
[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-compatible-green)](openenv.yaml)
[![Llama Powered](https://img.shields.io/badge/🦙%20Llama-4%20Maverick-purple)](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct)

> **Meta invests $13 billion per year into AI infrastructure.** NEMESIS-RL is the training ground that makes that investment pay off — a production-grade, OpenEnv-compliant reinforcement learning environment where Llama-based agents learn to make real-time moderation decisions across three escalating difficulty levels: spam, misinformation, and coordinated inauthentic behavior.

---

## Why This Matters

Social media moderation is one of the hardest sequential decision problems in AI. Meta processes **billions of posts per day** across Facebook, Instagram, and Threads, each requiring a real-time judgment call: allow, warn, restrict, remove, or escalate to a human reviewer. Static classifiers fail because moderation is **not an independent-decision problem** — removing a bot early stops a network attack; waiting for more evidence lets the campaign spread. Acting on node A cascades through the social graph to nodes B, C, and D.

**NEMESIS-RL frames this correctly, as a sequential decision problem under uncertainty** — exactly the problem class where RL has outperformed classical ML at scale (protein folding, chip placement, game-playing). This environment is the training infrastructure that lets Llama-based agents learn that nuanced tradeoff.

---

## What It Does

NEMESIS-RL exposes a Gymnasium-compatible RL environment over a clean REST API, fully compliant with the OpenEnv specification. An agent sends a `POST /step` with an integer action (0–4) and receives an observation, a scalar reward, a termination flag, and a structured info dict. The grader at `GET /grade/{task}` runs 10 deterministic evaluation episodes and returns a normalized score in `[0.0, 1.0]`.

### Three Task Tracks

| Task | Difficulty | Core Challenge | Grading Formula |
|------|-----------|----------------|----------------|
| `task_spam` | Easy | Feature-based binary moderation: 8-feature account fingerprint, 30% bot ratio with configurable signal overlap | `0.7 × F1 + 0.3 × sigmoid(mean_reward / 50)` |
| `task_misinfo` | Medium | BFS content diffusion through a 500-node social graph; act early for the speed bonus or let misinformation spread | `0.6 × F1 + 0.4 × (1 − mean_hop / max_hops)` |
| `task_cib` | Hard | Dismantle a hidden bot cluster embedded in a 500-node planted-partition graph; node2vec or spectral embeddings; collateral damage terminates the episode | `0.5 × recall + 0.5 × F1 − min(collateral_rate × 2, 0.5)` |

### Agent Types

- **`BaselineAgent`** (`baseline.py`) — Deterministic rule-based heuristic using suspicion score thresholds. This is the performance floor the RL agent must beat.
- **`NemesisPolicy`** (`model.py`) — Custom SB3 `ActorCriticPolicy` with `NemesisMlpExtractor`: 5 tabular dims + 384-dim sentence embedding (all-MiniLM-L6-v2 locally; Llama-4-Maverick-17B via HuggingFace Inference API when `HF_TOKEN` is set) → 389-dim input → [512 → 256 → 128] FC backbone → actor/critic heads.
- **`LLMAgent`** (`inference.py`) — Calls any OpenAI-compatible endpoint (e.g., HuggingFace Inference API with Llama-4-Maverick) to pick an action from the observation JSON. Runs all 3 tasks end-to-end in < 20 minutes.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NEMESIS-RL                           │
│                                                                  │
│  ┌──────────┐   POST /reset    ┌─────────────────────────────┐  │
│  │          │ ───────────────► │      SocialGuardEnv         │  │
│  │  Agent   │                  │  ┌─────────────────────┐    │  │
│  │ (LLM /   │   POST /step     │  │  Task Router        │    │  │
│  │  PPO /   │ ───────────────► │  │  task_spam          │    │  │
│  │ Baseline)│                  │  │  task_misinfo       │    │  │
│  │          │ ◄─────────────── │  │  task_cib           │    │  │
│  └──────────┘  obs + reward +  │  └─────────────────────┘    │  │
│                terminated +     │  ┌─────────────────────┐    │  │
│                info             │  │  RewardEngine       │    │  │
│                                  │  │  α·correctness      │    │  │
│  ┌──────────┐                  │  │  −β·fp_cost          │    │  │
│  │ Grader   │  GET /grade/{t}  │  │  −γ·collateral       │    │  │
│  │[0.0,1.0] │ ───────────────► │  │  +δ·speed_bonus      │    │  │
│  └──────────┘                  │  │  −ε·escalation       │    │  │
│                                  │  └─────────────────────┘    │  │
│  ┌──────────────────────────┐  └─────────────────────────────┘  │
│  │  Streamlit Dashboard     │                                    │
│  │  - NEMESIS Live Inference│  ┌─────────────────────────────┐  │
│  │  - Learning Curve Charts │  │  Social Graph (NetworkX)    │  │
│  │  - Network Graph (pyvis) │  │  Planted-partition model    │  │
│  │  - Reward Breakdown      │  │  node2vec / spectral embeds │  │
│  └──────────────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Observation Space

All tasks share a fixed `Box(float32, shape=(68,))` observation vector. Tasks with fewer features zero-pad to 68 dims — the SB3 policy network always sees the same input shape.

| Task | Active Dims | Key Features |
|------|-------------|-------------|
| Spam | 0–7 | `account_age_days`, `posts_per_hour`, `follower_ratio`, `login_time_variance`, `content_repetition_score`, `profile_completeness`, `device_fingerprint_uniqueness`, `ip_diversity_score` |
| Misinfo | 0–5 | `spread_rate`, `fact_check_flag`, `engagement_ratio`, `source_credibility`, `hop_count` (normalized), `timestep_normalized` |
| CIB | 0–67 | `embedding_0`…`embedding_63` (node2vec/spectral, L2-normalized), `degree_centrality`, `clustering_coefficient`, `community_assignment`, `posts_per_hour_normalized` |

### Action Space

`Discrete(5)` — shared across tasks. Invalid actions incur `escalation_penalty = 2.0`.

| ID | Name | Task 1 | Task 2 | Task 3 |
|----|------|--------|--------|--------|
| 0 | `allow` | ✅ | ✅ | ✅ |
| 1 | `warn` | ✅ | ✅ | ✅ |
| 2 | `reduce_reach` | ✅ | ✅ | ✅ |
| 3 | `remove` | ✅ | ✅ | ✅ |
| 4 | `escalate` | ❌ | ✅ | ✅ |

---

## Quick Start

### Docker (recommended)

```bash
docker build -t NEMESIS-RL .
docker run -p 7860:7860 \
  -e HF_TOKEN="hf_your_token_here" \
  -e MODEL_NAME="meta-llama/Llama-4-Maverick-17B-128E-Instruct" \
  -e API_BASE_URL="https://api-inference.huggingface.co/v1" \
  NEMESIS-RL
```

### Local

```bash
git clone https://github.com/vincenzo-afk/NEMESIS-RL.git
cd NEMESIS-RL
pip install -r requirements.txt

# Start the OpenEnv server
uvicorn server:app --host 0.0.0.0 --port 7860
```

### API

```bash
# Reset to a new episode (task_spam)
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "task_spam", "seed": 42}' | python3 -m json.tool

# Take a moderation action (action 3 = remove)
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task": "task_spam", "action": 3}' | python3 -m json.tool

# Grade the baseline agent on task_cib
curl http://localhost:7860/grade/task_cib

# Grade all three tasks at once
curl http://localhost:7860/grade/all

# Health check
curl http://localhost:7860/healthz
```

### Run the LLM Inference Script

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-4-Maverick-17B-128E-Instruct"
export HF_TOKEN="hf_your_token_here"
python inference.py
```

Stdout format (required by OpenEnv):
```
[START] task=task_spam env=NEMESIS-RL model=meta-llama/Llama-4-Maverick-17B-128E-Instruct
[STEP]  step=1 action=allow reward=0.00 done=false error=null
[STEP]  step=2 action=remove reward=1.28 done=false error=null
...
[END]   success=true steps=200 rewards=0.00,1.28,...
```

### Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

### Train with PPO

```bash
# Task 1
python training/train_ppo.py --config configs/task1.yaml --run_name ppo_spam --n_envs 4

# Task 3 with curriculum learning (required — do NOT train from scratch)
python training/train_ppo.py --config configs/task3.yaml --run_name ppo_cib --curriculum
```

---

## Benchmarks

### Baseline Agent (rule-based, `seed=42`, 100 episodes)

| Task | Precision | Recall | F1 | Mean Reward | Normalized Score |
|------|-----------|--------|-----|-------------|-----------------|
| `task_spam` | 0.74 | 0.68 | 0.71 | 38.4 | **0.71** |
| `task_misinfo` | 0.61 | 0.55 | 0.58 | 22.1 | **0.56** |
| `task_cib` | 0.52 | 0.48 | 0.50 | 15.7 | **0.43** |

### NEMESIS PPO Agent Training Progress (from `training_log.csv`)

| Cycle | Episodes | TP Rate | FP Rate | Mean Reward | Policy Entropy |
|-------|----------|---------|---------|-------------|---------------|
| 1 | 10 | 0.52 | 0.28 | 12.50 | 1.55 |
| 3 | 30 | 0.64 | 0.18 | 22.10 | 1.21 |
| 5 | 50 | 0.76 | 0.11 | 29.40 | 0.85 |
| 8 | 80 | 0.88 | 0.04 | 38.90 | 0.39 |
| 10 | 100 | 0.93 | 0.02 | 43.50 | 0.25 |

TP rate rises from 0.52 → **0.93** (+79%). FP rate drops from 0.28 → **0.02** (−93%). Policy entropy collapses as the agent becomes decisively selective — exactly the behavior profile a production content moderation system needs.

---

## Built for Meta's AI Safety Mission

Meta's content moderation teams process **hundreds of billions of content decisions per year** across Facebook, Instagram, WhatsApp, and Threads. The challenge is not binary classification — it is sequential policy under adversarial conditions: bot networks adapt, misinformation mutates, and coordinated campaigns evolve their tactics faster than human reviewers can respond.

NEMESIS-RL is purpose-built to be the simulation layer that trains the next generation of Meta Llama-powered moderation agents:

**Llama-4-Maverick as the Policy Network.** `inference.py` uses the OpenAI client interface to call any Llama endpoint. The `LLMAgent` class formats the 68-dim observation vector as a JSON prompt, calls `meta-llama/Llama-4-Maverick-17B-128E-Instruct` via the HuggingFace Inference API, and parses the action (0–4) from the response. Drop in Meta's internal endpoint and the infrastructure is ready.

**Reward Engineering That Reflects Meta's Values.** The `RewardEngine` (`env/rewards.py`) encodes a precise policy:
- `α · correctness` — reward correct threat detection
- `−β · fp_cost` — penalize false positives proportional to user legitimacy (a verified account costs more to wrongly remove than a new account)
- `−γ · collateral_damage` — separately tunable penalty for removing legitimate users
- `+δ · speed_bonus` — early intervention is rewarded for spread-timing tasks; this is what prevents misinformation from reaching a million users before action is taken
- `−ε · escalation_penalty` — agents learn to reserve human review for genuinely uncertain cases

**Scales to Meta's Graph Topology.** Task 3 uses the planted partition model (`networkx.stochastic_block_model`) to embed a hidden bot cluster inside a large social graph — a direct analog to the coordinated inauthentic behavior networks that Meta's Trust & Safety team dismantles. Node embeddings are computed via spectral decomposition (< 5 seconds on CPU) or node2vec (richer, slower), and the agent must reason over 64-dimensional graph structure features to find the cluster without the collateral damage of mass removals.

**Drop-in Training Infrastructure.** The Gymnasium API, Stable-Baselines3 compatibility, Docker container, and OpenEnv REST interface mean any team can spin this up on Meta's internal compute, swap in their own model checkpoint, and run curriculum-scheduled training on progressively harder graph sizes — exactly the scaling curriculum needed to produce a robust policy.

> *"AI is the defining technology of our time. We're going to invest heavily in it."* — Mark Zuckerberg, 2024. NEMESIS-RL is the training environment that turns that investment into a safer platform.

---

## Reward Function

```
R = α · correctness
  − β · fp_cost
  − γ · collateral_damage
  + δ · speed_bonus
  − ε · escalation_penalty
```

All coefficients are YAML-configurable — no magic numbers in code.

| Coefficient | Default | Meaning |
|-------------|---------|---------|
| `alpha` = 1.0 | Reward correct detection |
| `beta` = 0.8 | Penalty for false positives, scaled by `legitimacy_score` |
| `gamma` = 1.2 | Separate collateral damage penalty (heavier for CIB: 1.5) |
| `delta` = 0.3 | Speed bonus — decays linearly from 1.0 (hop 0) to 0.0 (hop 20) |
| `epsilon` = 0.1 | Escalation overuse penalty — grows with `escalation_count` |

---

## Project Structure

```
NEMESIS-RL/
│
├── inference.py             # OpenAI-client inference script (Llama integration)
├── server.py                # FastAPI OpenEnv HTTP server
├── baseline.py              # Rule-based heuristic agent (benchmark)
├── evaluate.py              # Baseline vs trained model comparison
├── agent.py                 # NEMESIS-RL training agent + inference runner
├── model.py                 # NemesisPolicy, NemesisMlpExtractor, Llama API call
├── openenv.yaml             # OpenEnv metadata + task registry
│
├── env/
│   ├── env.py               # Core gymnasium.Env — SocialGuardEnv
│   ├── spaces.py            # Observation (68,) + Action Discrete(5) spaces
│   ├── rewards.py           # RewardEngine — 5-component reward with config coefficients
│   └── models.py            # Pydantic v2 models for server validation
│
├── sim/
│   ├── social_graph.py      # NetworkX planted-partition graph, tick(), remove_node()
│   ├── user_behavior.py     # HumanBehavior + BotBehavior with configurable noise
│   └── content_gen.py       # BFS content spread simulation (Post + ContentEngine)
│
├── tasks/
│   ├── base_task.py         # Abstract BaseTask interface
│   ├── task_spam.py         # Task 1 — account queue with bot ratio
│   ├── task_misinfo.py      # Task 2 — BFS spread + timing bonus
│   └── task_cib.py          # Task 3 — graph RL with node2vec/spectral embeddings
│
├── graders/
│   └── grader.py            # Grader.evaluate() + normalized_score() + compare_agents()
│
├── training/
│   ├── train_ppo.py         # PPO entry point (SubprocVecEnv, EvalCallback)
│   ├── train_dqn.py         # DQN entry point
│   ├── callbacks.py         # TensorboardCallback + CurriculumCallback
│   └── curriculum.py        # 3-phase CIB curriculum schedule
│
├── dashboard/
│   ├── app.py               # Streamlit dashboard (11 tabs)
│   ├── graph_view.py        # Pyvis network graph + decision log injection
│   └── metrics_view.py      # Reward + decision charts
│
├── data/
│   └── synthetic_graph.py   # Deterministic graph fixtures for tests
│
├── tests/
│   ├── test_env.py          # gymnasium.utils.env_checker + 1000-step random agent
│   ├── test_rewards.py      # RewardEngine unit tests (8 test classes)
│   ├── test_tasks.py        # Task-specific tests (TaskSpam, TaskMisinfo, TaskCIB)
│   ├── test_graph.py        # SocialGraph contract tests
│   ├── test_grader.py       # Grader normalized_score bounds + baseline smoke
│   ├── test_training.py     # PPO/DQN smoke tests
│   ├── test_curriculum.py   # Curriculum override tests
│   └── test_nemesis.py      # NemesisNetBackbone + NemesisMlpExtractor + PPO tests
│
├── configs/
│   ├── default.yaml         # Base defaults — all keys documented
│   ├── task1.yaml           # Spam: max_steps=200, bot_ratio=0.30, delta=0.3
│   ├── task2.yaml           # Misinfo: delta=0.5 (stronger speed bonus)
│   ├── task3.yaml           # CIB: gamma=1.5, collateral_threshold=10
│   └── inference.yaml       # Slim: num_nodes=100, embedding_method=spectral
│
├── scripts/
│   └── pre_validate.sh      # Docker build + API smoke test + inference format check
│
├── Dockerfile               # python:3.12-slim, CMD uvicorn server:app --port 7860
├── requirements.txt
├── requirements-prod.txt
├── requirements-training.txt
└── openenv.yaml
```

---

## Roadmap

### Phase 1 — Foundation ✅
- [x] `env/spaces.py` — Unified `Box(68,)` observation + `Discrete(5)` action space
- [x] `env/rewards.py` — 5-component `RewardEngine` with config-driven coefficients
- [x] `sim/user_behavior.py` — `HumanBehavior` + `BotBehavior` with noise overlap

### Phase 2 — Task 1 Loop ✅
- [x] `tasks/base_task.py` + `tasks/task_spam.py` — Account queue with configurable bot ratio
- [x] `env/env.py` — Core `SocialGuardEnv` gymnasium API compliance
- [x] `baseline.py` — Rule-based `BaselineAgent` (performance floor)

### Phase 3 — Simulation ✅
- [x] `sim/social_graph.py` — NetworkX planted-partition graph
- [x] `sim/content_gen.py` — BFS content spread simulation
- [x] `tasks/task_misinfo.py` — Task 2 with timing reward

### Phase 4 — Task 3 + Graph RL ✅
- [x] `tasks/task_cib.py` — 500-node graph RL with node2vec embeddings
- [x] `_compute_embeddings_spectral()` — Fast spectral fallback (< 5 sec on 500 nodes)

### Phase 5 — Training ✅
- [x] `training/callbacks.py` — `TensorboardCallback` + `CurriculumCallback`
- [x] `training/train_ppo.py` + `training/train_dqn.py` — Full SB3 training pipelines
- [x] `training/curriculum.py` — 3-phase curriculum: 150 → 300 → 500 nodes

### Phase 6 — Evaluation ✅
- [x] `graders/grader.py` — Deterministic evaluation with normalized scores
- [x] `evaluate.py` — Baseline vs trained model side-by-side comparison

### Phase 7 — OpenEnv API ✅
- [x] `server.py` — FastAPI `/reset`, `/step`, `/state`, `/grade/{task}`, `/grade/all`, `/healthz`, `/metrics`
- [x] `inference.py` — LLM inference with `[START]`/`[STEP]`/`[END]` stdout protocol
- [x] `openenv.yaml` — OpenEnv metadata registry

### Phase 8 — Dashboard ✅
- [x] `dashboard/app.py` — 11-tab Streamlit dashboard
- [x] `dashboard/graph_view.py` — Pyvis network + decision log injection
- [x] NEMESIS PPO training from UI + live checkpoint browser

### Phase 9 — Hardening 🔲
- [ ] Full test coverage for `grader.py`, `callbacks.py`, `baseline.py`
- [ ] BUG-1: Per-step FN inflation in grader metrics
- [ ] BUG-2: Race condition in `server.py` environment registry
- [ ] Pin PyTorch version in `requirements-prod.txt`
- [ ] Add Docker health check for GPU availability
- [ ] Authenticated dashboard (`SOCIALGUARD_DASHBOARD_TOKEN` wired)

---

## Development

```bash
# Run full test suite
pytest tests/ -v

# Lint + format
black .
ruff check .

# gymnasium env_checker
python -c "
from gymnasium.utils.env_checker import check_env
from env.env import SocialGuardEnv
check_env(SocialGuardEnv('configs/task1.yaml'), warn=True, skip_render_check=True)
print('env_checker passed')
"

# Pre-submission validation
bash scripts/pre_validate.sh
```

---

## License

MIT — see [LICENSE](LICENSE).

## Contributors

Built by **vincenzo-afk** for the **Meta Llama Impact Hackathon 2026**.

---

## Citation

```bibtex
@software{socialguard_rl_2026,
  author  = {vincenzo-afk},
  title   = {NEMESIS-RL: An OpenEnv Environment for Social Media Integrity Moderation},
  year    = {2026},
  url     = {https://huggingface.co/spaces/vincenzo-afk/NEMESIS-RL},
  note    = {Meta Llama Impact Hackathon 2026}
}
```
