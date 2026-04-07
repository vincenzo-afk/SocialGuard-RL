# CHANGELOG

All notable changes to NEMESIS-RL are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.3.0] — 2026-04-02

### Added
- **`inference.py`**: Mandatory OpenAI-client inference script. Runs all 3 tasks sequentially with strict `[START]`/`[STEP]`/`[END]` stdout protocol. Enforces 5-minute per-task wall-clock budget via `signal.SIGALRM` (Unix) and a manual monotonic clock fallback (Windows). Falls back to `BaselineAgent` if the `openai` package is not installed.
- **`server.py`**: Complete FastAPI OpenEnv HTTP server. Exposes `/healthz`, `/reset`, `/step`, `/state`, and `/grade/{task_name}`. Lazy-initialized per-task environment registry. Validated via Pydantic v2 models.
- **`env/models.py`**: Pydantic v2 request/response models — `ObservationModel`, `ActionModel`, `RewardModel`, `ResetRequest`, `StepRequest`.
- **`openenv.yaml`**: OpenEnv metadata file with task registry, endpoint declarations, and HF Space discovery tags.
- **`configs/inference.yaml`**: Slim configuration for the inference budget: 100 nodes, `embedding_method=spectral`, `max_steps=50`.
- **`scripts/pre_validate.sh`**: Automated pre-submission pipeline — Docker build, API smoke-test, `/grade` score range check, inference stdout format validation.
- **`graders/grader.py`** — `normalized_score()`: Per-task normalized scoring in `[0.0, 1.0]` using the README formulas:
  - Task 1: `0.7 × F1 + 0.3 × sigmoid(mean_reward / 50.0)`
  - Task 2: `0.6 × F1 + 0.4 × max(0, 1 − mean_hop / max_hops)`
  - Task 3: `0.5 × recall + 0.5 × F1 − min(collateral_rate × 2, 0.5)`
- **`tasks/task_cib.py`** — `_compute_embeddings_spectral()`: Normalized Laplacian spectral embeddings via `scipy.sparse.linalg.eigsh`. Runs in < 5 seconds vs 2–5 minutes for node2vec. Selected via `graph.embedding_method: spectral` or `SOCIALGUARD_EMBEDDING_METHOD=spectral` env var.
- **`env/env.py`** — `task_success` bool added to `step()` info dict. `True` when episode terminates cleanly without collateral overrun or truncation.
- **Deterministic evaluation**: `Grader.evaluate()` now seeds each episode as `base_seed + episode_index` for fully reproducible cross-run results.

### Changed
- `Dockerfile`: CMD switched from `streamlit run dashboard/app.py` to `uvicorn server:app --host 0.0.0.0 --port 7860`. EXPOSE changed from `8501` to `7860`. HEALTHCHECK updated to `/healthz`.
- `requirements.txt` / `requirements-prod.txt`: Added `fastapi`, `uvicorn`, `pydantic>=2.0`, `openai`, `scipy`.

---

## [1.2.0] — 2026-03-29

### Added
- **`training/train_ppo.py`**: Full PPO training entry point with `n_envs`, `device`, `eval_freq`, and `run_name` arguments. Saves `final_model.zip`, `best_model.zip`, periodic checkpoints, and TensorBoard logs.
- **`training/train_dqn.py`**: DQN training entry point sharing the same callback/eval infrastructure.
- **`training/callbacks.py`**: `SocialGuardEvalCallback` (TensorBoard metrics), `DashboardWindowCallback` (training progress logging).
- **`training/curriculum.py`**: `CurriculumSchedule` — 3-phase curriculum for Task 3: 150 nodes (0–40%), 300 nodes (40–70%), 500 nodes (70–100%).

### Fixed
- `dashboard/app.py`: Runtime crashes resolved — fixed `pyvis` render pipeline, added `SOCIALGUARD_DASHBOARD_TOKEN` authentication, replaced broken model path resolution.
- `.gitignore`: Added `models/`, `*.zip`, `__pycache__/`, `.env`, `*.log` exclusions.

---

## [1.1.0] — 2026-03-21

### Added
- **Task 3 — CIB Network Takedown** (`tasks/task_cib.py`): full planted-partition graph environment, node2vec embeddings, 68-dim observation space (64-dim embedding + 4 graph features).
- **Task 2 — Viral Misinformation Flagging** (`tasks/task_misinfo.py`): BFS content spread simulation with timing-based speed bonus.
- **`sim/social_graph.py`**: NetworkX planted-partition graph generator with configurable bot cluster sizes and density.
- **`sim/content_gen.py`**: BFS content spread simulation for Task 2 dynamics.
- **`sim/user_behavior.py`**: `HumanBehavior` and `BotBehavior` feature generation models.
- **`graders/grader.py`**: Evaluation engine with F1, precision, recall, mean reward, time-to-detection, and collateral damage metrics.
- **`dashboard/graph_view.py`**: Pyvis network graph with node coloring (green=real, red=bot, yellow=under review, gray=removed). Decision log injection via injected JS.

### Changed
- Observation space unified to `Box(float32, shape=(68,))` across all tasks via zero-padding.
- Action space: global `Discrete(5)` — `allow`, `warn`, `reduce_reach`, `remove`, `escalate`.

---

## [1.0.0] — 2026-03-16

### Added
- **Task 1 — Spam Account Detection** (`tasks/task_spam.py`): feature-based queue moderation environment.
- **`env/env.py`**: Core `SocialGuardEnv` gymnasium.Env subclass. Config-driven, full gymnasium API compliance.
- **`env/rewards.py`**: Multi-objective `RewardEngine` — `correctness`, `fp_cost`, `collateral_damage`, `speed_bonus`, `escalation_penalty` with YAML-configurable coefficients.
- **`env/spaces.py`**: Unified observation (`Box(68,)`) and action (`Discrete(5)`) spaces.
- **`baseline.py`**: Rule-based `BaselineAgent` — computes suspicion score from raw features; multi-task dispatch.
- **`evaluate.py`**: Trained model vs. baseline side-by-side evaluator with JSON output.
- **`configs/default.yaml`**, `task1.yaml`, `task2.yaml`, `task3.yaml`: Full YAML config hierarchy with documented defaults.
- MIT License.

### Observation Space (v1.0)
| Version | Dim | Changes |
|---------|-----|---------|
| 1.0 | 68 | Initial — Task 1 uses indices 0–7, rest zero-padded |
| 1.1 | 68 | Task 2 added (indices 0–5). Task 3 added (all 68 dims used). |

### Reward Formula (v1.0)
```
R = α·correctness − β·fp_cost − γ·collateral_damage + δ·speed_bonus − ε·escalation_penalty
```
Default: α=1.0, β=0.8, γ=1.2, δ=0.3, ε=0.1
