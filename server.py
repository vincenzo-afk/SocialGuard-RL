from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import threading
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import numpy as np
import uvicorn

from env.env import SocialGuardEnv
from env.models import ObservationModel, ResetRequest, StepRequest

app = FastAPI(
    title="SocialGuard-RL OpenEnv Server",
    description="OpenEnv-compliant API for social media integrity moderation RL environment.",
    version="1.0.0",
    contact={"name": "SocialGuard-RL"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_envs: dict[str, SocialGuardEnv] = {}
_locks: dict[str, threading.Lock] = {}
_grade_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()
_recent_calls: deque[dict[str, Any]] = deque(maxlen=25)

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
    if task_name in _envs:
        return _envs[task_name], _locks[task_name]

    with _registry_lock:
        if task_name in _envs:
            return _envs[task_name], _locks[task_name]
        if task_name not in _locks:
            _locks[task_name] = threading.Lock()

    try:
        new_env = SocialGuardEnv(TASK_CONFIG_MAP[task_name])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Env init failed: {exc}")

    with _registry_lock:
        if task_name not in _envs:
            _envs[task_name] = new_env

    return _envs[task_name], _locks[task_name]


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


def _default_grade_result(task_name: str, requested_episodes: int, reason: str = "") -> dict[str, Any]:
    details: dict[str, Any] = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "mean_reward": 0.0,
        "mean_episode_length": 0.0,
        "time_to_detection": 0.0,
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
    _recent_calls.appendleft(
        {
            "method": request.method,
            "path": request.url.path,
            "status": int(response.status_code),
            "elapsed_ms": round(float(elapsed_ms), 2),
        }
    )
    return response


@app.get("/healthz", tags=["meta"])
def healthz():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse, tags=["meta"])
def root():
    rows = "\n".join(
        f"<li><code>{c['method']} {c['path']}</code> -> {c['status']} ({c['elapsed_ms']:.2f}ms)</li>"
        for c in list(_recent_calls)[:10]
    ) or "<li>No calls yet.</li>"
    return f"""<!DOCTYPE html>
<html><head><title>SocialGuard-RL</title>
<meta http-equiv=\"refresh\" content=\"30\">
<style>
  body{{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;background:#fafafa;color:#1a1a1a}}
  h1{{font-size:22px;font-weight:600;margin-bottom:4px}}
  .tag{{background:#e8f0fe;color:#1a56db;padding:2px 8px;border-radius:4px;font-size:12px}}
  .card{{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:16px;margin:12px 0}}
  .endpoint{{font-family:monospace;font-size:13px;background:#f3f4f6;padding:2px 6px;border-radius:4px}}
  a{{color:#2563eb;text-decoration:none}}
</style></head><body>
<h1>SocialGuard-RL <span class=\"tag\">OpenEnv</span></h1>
<p style=\"color:#6b7280;font-size:14px\">Social media integrity moderation RL environment</p>
<div class=\"card\"><strong>Quick links</strong><br>
  <a href=\"/docs\">API Docs (Swagger)</a> ·
  <a href=\"/grade/task_spam\">Grade Spam</a> ·
  <a href=\"/grade/task_misinfo\">Grade Misinfo</a> ·
  <a href=\"/grade/task_cib\">Grade CIB</a> ·
  <a href=\"/grade/all\">Grade All</a> ·
  <a href=\"/healthz\">Health</a> ·
  <a href=\"/metrics\">Metrics</a>
</div>
<div class=\"card\"><strong>Endpoints</strong>
  <ul style=\"font-size:14px;line-height:2\">
    <li><span class=\"endpoint\">POST /reset</span> - start a new episode</li>
    <li><span class=\"endpoint\">POST /step</span> - take a moderation action</li>
    <li><span class=\"endpoint\">GET /state</span> - full episode state</li>
    <li><span class=\"endpoint\">GET /grade/{{task}}</span> - normalized score [0,1]</li>
  </ul>
</div>
<div class=\"card\"><strong>Recent API calls</strong>
  <ul style=\"font-size:13px;line-height:1.8\">{rows}</ul>
</div>
</body></html>"""


@app.post("/reset", response_model=ObservationModel, tags=["env"])
def reset_env(req: ResetRequest):
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
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step", response_model=ObservationModel, tags=["env"])
def step_env(req: StepRequest):
    env, lock = get_env_and_lock(req.task)
    with lock:
        if getattr(env, "_task", None) is None:
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
    from baseline import BaselineAgent
    from graders.grader import Grader

    if task_name not in TASK_CONFIG_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task_name}'. Valid: {list(TASK_CONFIG_MAP.keys())}",
        )

    with _registry_lock:
        grade_lock = _grade_locks.setdefault(task_name, threading.Lock())

    with grade_lock:
        env = SocialGuardEnv(TASK_CONFIG_MAP[task_name])
        grader = Grader(env, n_episodes=n_episodes)
        try:
            agent = BaselineAgent()
            timeout_s = float(max(10, min(600, 15 * max(1, int(n_episodes)))))

            def _run_eval() -> dict[str, Any]:
                return grader.evaluate(agent, agent_name="baseline", base_seed=seed)

            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_run_eval)
                try:
                    results = fut.result(timeout=timeout_s)
                except FuturesTimeoutError:
                    raise HTTPException(
                        status_code=504,
                        detail=f"Grading timed out after {timeout_s:.0f}s for task={task_name}",
                    )

            task_metrics = results.get("tasks", {}).get(task_name, {})
            if not task_metrics:
                raise HTTPException(status_code=500, detail=f"No metrics compiled for task {task_name}")

            if task_name == "task_cib":
                actual_threshold = float(getattr(env, "_task_cfg", {}).get("collateral_damage_threshold", 10.0))
                task_metrics["collateral_threshold"] = actual_threshold

            score = grader.normalized_score(task_name, task_metrics)

            return {
                "task": task_name,
                "score": round(score, 4),
                "score_formula": _score_formula(task_name),
                "details": {
                    "precision": round(float(task_metrics.get("precision", 0.0)), 4),
                    "recall": round(float(task_metrics.get("recall", 0.0)), 4),
                    "f1": round(float(task_metrics.get("f1", 0.0)), 4),
                    "mean_reward": round(float(task_metrics.get("mean_reward", 0.0)), 4),
                    "mean_episode_length": round(float(task_metrics.get("mean_episode_length", 0.0)), 1),
                    "time_to_detection": round(float(task_metrics.get("time_to_detection", 0.0)), 1),
                    "mean_collateral": round(float(task_metrics.get("mean_collateral", 0.0)), 2),
                    "n_episodes": n_episodes,
                    "agent": "rule-based baseline",
                },
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            env.close()


@app.get("/metrics", tags=["meta"])
def metrics():
    lines = []
    for task, env in _envs.items():
        lines.append(f'socialguard_env_timestep{{task="{task}"}} {getattr(env, "_timestep", 0)}')
        lines.append(
            f'socialguard_env_cumulative_reward{{task="{task}"}} {float(getattr(env, "_cumulative_reward", 0.0)):.4f}'
        )
    return "\n".join(lines) or "# no envs initialized"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
