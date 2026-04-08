from collections import deque
import hmac
import multiprocessing as mp
import os
import threading
import time
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
import numpy as np
import uvicorn
import yaml

from env.env import SocialGuardEnv
from env.models import ObservationModel, ResetRequest, StepRequest

os.environ.setdefault("SOCIALGUARD_EMBEDDING_METHOD", "spectral")

app = FastAPI(
    title="NEMESIS-RL OpenEnv Server",
    description="OpenEnv-compliant API for social media integrity moderation RL environment.",
    version="1.0.0",
    contact={"name": "NEMESIS-RL"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

_envs: dict[str, SocialGuardEnv] = {}
_locks: dict[str, threading.Lock] = {}
_grade_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()
_recent_calls: deque[dict[str, Any]] = deque(maxlen=25)
_server_started_at = time.time()
_total_steps_served = 0
_task_reset_counts: dict[str, int] = {}
_task_step_counts: dict[str, int] = {}

TASK_CONFIG_MAP = {
    "task_spam": "configs/task1.yaml",
    "task_misinfo": "configs/task2.yaml",
    "task_cib": "configs/task3.yaml",
}

SCORE_FORMULA_MAP = {
    "task_spam": "0.7 * F1 + 0.3 * sigmoid(mean_reward / 50)",
    "task_misinfo": "0.6 * F1 + 0.4 * max(0, 1 - mean_hop/max_hops)",
    "task_cib": "0.5 * recall + 0.5 * F1 - min(collateral_rate*2, 0.5)",
}


def get_env_and_lock(task_name: str) -> tuple[SocialGuardEnv, threading.Lock]:
    if task_name not in TASK_CONFIG_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task_name}'. Valid: {list(TASK_CONFIG_MAP.keys())}",
        )
    # Bug #2 fix: check BOTH dicts atomically to avoid KeyError race condition
    # where _envs[task] is set but _locks[task] is not yet written by a concurrent thread.
    if task_name in _envs and task_name in _locks:
        return _envs[task_name], _locks[task_name]

    with _registry_lock:
        if task_name in _envs and task_name in _locks:
            return _envs[task_name], _locks[task_name]

        lock = _locks.setdefault(task_name, threading.Lock())
        try:
            env = SocialGuardEnv(TASK_CONFIG_MAP[task_name])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Env init failed: {exc}")
        _envs[task_name] = env
        return env, lock


def _deep_cast_numpy(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_deep_cast_numpy(i) for i in obj]
    if isinstance(obj, dict):
        return {str(k): _deep_cast_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_cast_numpy(i) for i in obj]
    return obj


def _score_formula(task_name: str) -> str:
    return SCORE_FORMULA_MAP.get(task_name, "F1")


def _round_optional(value: Any, digits: int) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _default_grade_result(task_name: str, requested_episodes: int, reason: str = "") -> dict[str, Any]:
    details: dict[str, Any] = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "mean_reward": 0.0,
        "mean_episode_length": 0.0,
        "time_to_detection": None,
        "mean_collateral": 0.0,
        "n_episodes": 0,
        "requested_n_episodes": max(0, int(requested_episodes)),
        "agent": "rule-based baseline",
        "status": "empty",
    }
    if reason:
        details["reason"] = reason
    return {
        "task": task_name,
        "score": 0.0,
        "score_formula": _score_formula(task_name),
        "details": details,
    }


def _ensure_env_session(task_name: str, seed: Optional[int] = None) -> bool:
    env, lock = get_env_and_lock(task_name)
    with lock:
        if getattr(env, "_task", None) is None:
            env.reset(seed=seed)
            return True
    return False


def _is_empty_grading_error(exc: HTTPException) -> bool:
    detail = str(getattr(exc, "detail", ""))
    return exc.status_code == 400 or "No metrics compiled for task" in detail


@app.middleware("http")
async def track_recent_calls(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-NEMESIS-Version"] = app.version
    _recent_calls.appendleft(
        {
            "method": request.method,
            "path": request.url.path,
            "status": int(response.status_code),
            "elapsed_ms": round(float(elapsed_ms), 2),
        }
    )
    return response


@app.middleware("http")
async def require_api_token(request: Request, call_next):
    token = os.environ.get("SOCIALGUARD_API_TOKEN", "").strip()
    if not token:
        return await call_next(request)

    path = request.url.path
    if path in {"/", "/healthz", "/openapi.json"} or path.startswith("/docs") or path.startswith("/redoc"):
        return await call_next(request)

    provided = request.headers.get("Authorization", "").strip()
    if not hmac.compare_digest(provided, f"Bearer {token}"):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    return await call_next(request)


@app.get("/healthz", tags=["meta"])
def healthz():
    return {
        "status": "ok",
        "version": app.version,
        "uptime_seconds": int(max(0, time.time() - _server_started_at)),
        "total_steps_served": int(_total_steps_served),
    }


@app.get("/", tags=["meta"])
def root():
    index_path = os.path.join("dashboard", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)


@app.post("/reset", response_model=ObservationModel, tags=["env"])
def reset_env(req: Optional[ResetRequest] = Body(default=None)):
    if req is None:
        req = ResetRequest()
    env, lock = get_env_and_lock(req.task)
    with lock:
        try:
            obs, info = env.reset(seed=req.seed)
            _task_reset_counts[req.task] = int(_task_reset_counts.get(req.task, 0)) + 1
            return ObservationModel(
                observation=obs.tolist(),
                reward=0.0,
                terminated=False,
                truncated=False,
                info=_deep_cast_numpy(info),
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step", response_model=ObservationModel, tags=["env"])
def step_env(req: StepRequest):
    global _total_steps_served
    env, lock = get_env_and_lock(req.task)
    with lock:
        if getattr(env, "_task", None) is None:
            raise HTTPException(status_code=400, detail="Call /reset before /step.")
        try:
            obs, reward, terminated, truncated, info = env.step(req.action)
            with _registry_lock:
                _total_steps_served += 1
            _task_step_counts[req.task] = int(_task_step_counts.get(req.task, 0)) + 1
            return ObservationModel(
                observation=obs.tolist(),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=_deep_cast_numpy(info),
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state", tags=["env"])
def get_state(task: str):
    env, lock = get_env_and_lock(task)
    with lock:
        if env._task is None:
            raise HTTPException(status_code=400, detail="Environment has not been reset yet.")
        return _deep_cast_numpy(env.state())


@app.get("/config/{task_name}", tags=["meta"])
def get_task_config(task_name: str):
    config_path = TASK_CONFIG_MAP.get(task_name)
    if not config_path:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task_name}'. Valid: {list(TASK_CONFIG_MAP.keys())}",
        )
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail=f"Config not found for task '{task_name}'.")

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load config: {exc}")

    task_cfg = data.get("task", {}) if isinstance(data, dict) else {}
    env_cfg = data.get("env", {}) if isinstance(data, dict) else {}
    reward_cfg = data.get("reward", {}) if isinstance(data, dict) else {}

    return {
        "task": task_name,
        "bot_ratio": task_cfg.get("bot_ratio"),
        "noise_level": task_cfg.get("noise_level"),
        "max_steps": env_cfg.get("max_steps"),
        "collateral_threshold": task_cfg.get("collateral_damage_threshold", 0),
        "reward": reward_cfg,
        "raw": data,
    }


@app.get("/grade/all", tags=["grading"])
def grade_all(n_episodes: int = 10, seed: int = 42):
    requested_episodes = max(0, int(n_episodes))
    if not TASK_CONFIG_MAP:
        return {
            "combined_score": 0.0,
            "status": "empty",
            "message": "No grading tasks are configured.",
        }

    results: dict[str, Any] = {}
    total = 0.0
    initialized_tasks: list[str] = []
    empty_tasks: list[str] = []

    for task in TASK_CONFIG_MAP:
        if requested_episodes == 0:
            graded = _default_grade_result(task, requested_episodes, "No grading episodes requested.")
            empty_tasks.append(task)
            results[task] = graded
            continue

        if _ensure_env_session(task, seed=seed):
            initialized_tasks.append(task)

        try:
            graded = grade_task(task, n_episodes=requested_episodes, seed=seed)
        except HTTPException as exc:
            if not _is_empty_grading_error(exc):
                raise
            graded = _default_grade_result(task, requested_episodes, str(exc.detail))
            empty_tasks.append(task)

        results[task] = graded
        total += float(graded["score"])

    completed_tasks = [
        task
        for task in TASK_CONFIG_MAP
        if int(results.get(task, {}).get("details", {}).get("n_episodes", 0)) > 0
    ]
    results["combined_score"] = round(total / len(TASK_CONFIG_MAP), 4)
    results["status"] = "ok" if completed_tasks else "empty"
    results["message"] = (
        "Grading completed."
        if completed_tasks
        else "No completed grading data available yet; returning default scores."
    )
    results["completed_tasks"] = completed_tasks
    if initialized_tasks:
        results["initialized_tasks"] = initialized_tasks
    if empty_tasks:
        results["empty_tasks"] = empty_tasks
    return results


@app.get("/grade/{task_name}", tags=["grading"])
def grade_task(task_name: str, n_episodes: int = 10, seed: int = 42):
    if task_name not in TASK_CONFIG_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task_name}'. Valid: {list(TASK_CONFIG_MAP.keys())}",
        )

    with _registry_lock:
        grade_lock = _grade_locks.setdefault(task_name, threading.Lock())

    with grade_lock:
        timeout_s = float(max(10, min(600, 15 * max(1, int(n_episodes)))))
        ctx = mp.get_context("spawn")
        out_queue: mp.Queue = ctx.Queue()
        proc = ctx.Process(target=_grade_worker, args=(task_name, int(n_episodes), int(seed), out_queue))
        proc.start()
        proc.join(timeout_s)
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            raise HTTPException(
                status_code=504,
                detail=f"Grading timed out after {timeout_s:.0f}s for task={task_name}",
            )
        if out_queue.empty():
            raise HTTPException(status_code=500, detail=f"Grading failed without output for task={task_name}")
        result = out_queue.get()
        if not result.get("ok", False):
            raise HTTPException(status_code=500, detail=str(result.get("error", "Unknown grading error")))
        return result["payload"]


@app.get("/metrics", tags=["meta"])
def metrics():
    lines = []
    for task, env in _envs.items():
        lines.append(f'nemesis_env_timestep{{task="{task}"}} {getattr(env, "_timestep", 0)}')
        lines.append(f'nemesis_env_episode_step{{task="{task}"}} {getattr(env, "_episode_step", 0)}')
        lines.append(
            f'nemesis_env_cumulative_reward{{task="{task}"}} {float(getattr(env, "_cumulative_reward", 0.0)):.4f}'
        )
    for task in TASK_CONFIG_MAP:
        lines.append(f'nemesis_env_episode_count{{task="{task}"}} {int(_task_reset_counts.get(task, 0))}')
        lines.append(f'nemesis_env_step_count{{task="{task}"}} {int(_task_step_counts.get(task, 0))}')
    lines.append(f"nemesis_server_total_steps_served {int(_total_steps_served)}")
    lines.append(f"nemesis_server_uptime_seconds {int(max(0, time.time() - _server_started_at))}")
    return Response(content="\n".join(lines) or "# no envs initialized\n", media_type="text/plain")

@app.get("/recent_calls", tags=["meta"])
def recent_calls():
    return list(_recent_calls)


# Bug #1 fix: _grade_worker MUST be defined before if __name__ == "__main__" so that
# spawned child processes (mp.get_context("spawn")) can pickle and import it correctly.
def _grade_worker(task_name: str, n_episodes: int, seed: int, out_queue: mp.Queue) -> None:
    """Run grading in an isolated child process so timeouts can terminate cleanly."""
    try:
        from baseline import BaselineAgent
        from graders.grader import Grader

        env = SocialGuardEnv(TASK_CONFIG_MAP[task_name])
        try:
            grader = Grader(env, n_episodes=n_episodes)
            agent = BaselineAgent(task_name=task_name)
            results = grader.evaluate(agent, agent_name="baseline", base_seed=seed)
            task_metrics = results.get("tasks", {}).get(task_name, {})
            if not task_metrics:
                out_queue.put({"ok": False, "error": f"No metrics compiled for task {task_name}"})
                return

            score = grader.normalized_score(task_name, task_metrics)
            out_queue.put(
                {
                    "ok": True,
                    "payload": {
                        "task": task_name,
                        "score": round(score, 4),
                        "score_formula": _score_formula(task_name),
                        "details": {
                            "precision": round(float(task_metrics.get("precision", 0.0)), 4),
                            "recall": round(float(task_metrics.get("recall", 0.0)), 4),
                            "f1": round(float(task_metrics.get("f1", 0.0)), 4),
                            "mean_reward": round(float(task_metrics.get("mean_reward", 0.0)), 4),
                            "mean_episode_length": round(float(task_metrics.get("mean_episode_length", 0.0)), 1),
                            "time_to_detection": _round_optional(task_metrics.get("time_to_detection"), 1),
                            "mean_collateral": round(float(task_metrics.get("mean_collateral", 0.0)), 2),
                            "n_episodes": n_episodes,
                            "agent": "rule-based baseline",
                        },
                    },
                }
            )
        finally:
            env.close()
    except Exception as exc:
        out_queue.put({"ok": False, "error": str(exc)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
