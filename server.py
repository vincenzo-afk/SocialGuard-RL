from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Any
import threading
import numpy as np

from env.env import SocialGuardEnv
from env.models import ObservationModel, ResetRequest, StepRequest
from env.spaces import ACTION_NAMES

app = FastAPI(
    title="SocialGuard-RL OpenEnv Server",
    description="OpenEnv-compliant API for social media integrity moderation RL environment.",
    version="1.0.0",
)

# [FAIL] #30: Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Environment registry — one env per task, lazily initialized
# ---------------------------------------------------------------------------

_envs: dict[str, SocialGuardEnv] = {}
_locks: dict[str, threading.Lock] = {}

TASK_CONFIG_MAP = {
    "task_spam":    "configs/task1.yaml",
    "task_misinfo": "configs/task2.yaml",
    "task_cib":     "configs/task3.yaml",
}


def get_env_and_lock(task_name: str) -> tuple[SocialGuardEnv, threading.Lock]:
    """Return (or lazily create) the environment and its lock for the given task."""
    if task_name not in TASK_CONFIG_MAP:
        # [FAIL] #33: Unknown task should return HTTP 404
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task '{task_name}'. Valid: {list(TASK_CONFIG_MAP.keys())}",
        )
    
    if task_name not in _locks:
        _locks[task_name] = threading.Lock()
        
    with _locks[task_name]:
        if task_name not in _envs:
            try:
                _envs[task_name] = SocialGuardEnv(TASK_CONFIG_MAP[task_name])
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Env init failed: {exc}")
    
    return _envs[task_name], _locks[task_name]


def _deep_cast_numpy(obj: Any) -> Any:
    """Recursively cast NumPy types to pure Python types for JSON serialization."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_deep_cast_numpy(i) for i in obj]
    if isinstance(obj, dict):
        return {str(k): _deep_cast_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [_deep_cast_numpy(i) for i in obj]
    return obj


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/healthz", tags=["meta"])
def healthz():
    """Health probe — returns 200 {"status": "ok"} when the server is ready."""
    return {"status": "ok"}


@app.post("/reset", response_model=ObservationModel, tags=["env"])
def reset_env(req: ResetRequest):
    """Reset the environment for the given task and return the initial observation."""
    env, lock = get_env_and_lock(req.task)
    with lock:
        try:
            obs, info = env.reset(seed=req.seed)
            return ObservationModel(
                observation=obs.tolist(),
                reward=0.0,
                terminated=False,
                truncated=False,
                info=_deep_cast_numpy(info),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step", response_model=ObservationModel, tags=["env"])
def step_env(req: StepRequest):
    """Execute one moderation action and return the next observation + reward."""
    env, lock = get_env_and_lock(req.task)
    with lock:
        if env._task is None:
            raise HTTPException(status_code=400, detail="Call /reset before /step.")
        try:
            obs, reward, terminated, truncated, info = env.step(req.action)
            return ObservationModel(
                observation=obs.tolist(),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=_deep_cast_numpy(info),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state", tags=["env"])
def get_state(task: str):
    """Return the full internal episode state as a JSON-serialisable dict."""
    env, lock = get_env_and_lock(task)
    with lock:
        if env._task is None:
            raise HTTPException(status_code=400, detail="Environment has not been reset yet.")
        state_dict = env.state()
        return _deep_cast_numpy(state_dict)


@app.get("/grade/{task_name}", tags=["grading"])
def grade_task(task_name: str, n_episodes: int = 10, seed: int = 42):
    """
    Run a deterministic evaluation using the rule-based baseline agent.
    Returns a normalized score in [0.0, 1.0] plus detailed metrics.
    """
    from graders.grader import Grader
    from baseline import BaselineAgent

    env, lock = get_env_and_lock(task_name)
    with lock:
        grader = Grader(env, n_episodes=n_episodes)
        try:
            agent = BaselineAgent()
            results = grader.evaluate(agent, agent_name="baseline")
            task_metrics = results.get("tasks", {}).get(task_name, results)

            # Compute the normalized score per README formulae
            score = grader.normalized_score(task_name, task_metrics)

            return {
                "task": task_name,
                "score": round(score, 4),
                "details": {
                    "precision": _deep_cast_numpy(task_metrics.get("precision", 0.0)),
                    "recall": _deep_cast_numpy(task_metrics.get("recall", 0.0)),
                    "f1": _deep_cast_numpy(task_metrics.get("f1", 0.0)),
                    "mean_reward": _deep_cast_numpy(task_metrics.get("mean_reward", 0.0)),
                    "mean_episode_length": _deep_cast_numpy(task_metrics.get("mean_episode_length", 0.0)),
                    "n_episodes": n_episodes,
                },
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
