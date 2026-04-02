"""
inference.py — Mandatory OpenEnv inference script for SocialGuard-RL.

Uses the OpenAI client to call an LLM as the moderation policy.
Emits structured [START]/[STEP]/[END] lines to stdout.
Runs all three tasks sequentially within a 20-minute wall-clock budget.

Required environment variables:
    API_BASE_URL  — LLM API base URL, e.g. https://api-inference.huggingface.co/v1
    MODEL_NAME    — model identifier, e.g. meta-llama/Llama-4-Maverick-17B-128E-Instruct
    HF_TOKEN      — Hugging Face API token

Usage::

    export API_BASE_URL="https://api-inference.huggingface.co/v1"
    export MODEL_NAME="Qwen/Qwen3-30B-A3B"
    export HF_TOKEN="hf_..."
    python inference.py
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from typing import Any

import numpy as np

from env.env import SocialGuardEnv

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "task_spam":    "configs/task1.yaml",
    "task_misinfo": "configs/task2.yaml",
    "task_cib":     "configs/inference.yaml",
}

# Per-task episode budget (seconds). Total must fit in 20 min.
TASK_TIMEOUT_SECONDS = 300  # 5 min each; 3 tasks = 15 min max
ENV_NAME = "socialguard-rl"

# Use spectral embeddings for fast inference on task_cib
os.environ.setdefault("SOCIALGUARD_EMBEDDING_METHOD", "spectral")


# ---------------------------------------------------------------------------
# LLM agent wrapper
# ---------------------------------------------------------------------------

class LLMAgent:
    """Wraps an OpenAI-compatible LLM as a moderation policy agent."""

    SYSTEM_PROMPT = (
        "You are an automated social media integrity officer. "
        "You receive a JSON observation vector describing a user, content, or network node. "
        "You must choose one moderation action:\n"
        "  0=allow, 1=warn, 2=reduce_reach, 3=remove, 4=escalate\n"
        "Respond with ONLY the integer action number and nothing else."
    )

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def _format_obs(self, obs: np.ndarray) -> str:
        """Convert observation array to a compact JSON string for the prompt."""
        return json.dumps({"observation": [round(float(x), 4) for x in obs]})

    def act(self, obs: np.ndarray) -> int:
        """Call the LLM and parse an integer action from its response."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": self._format_obs(obs)},
                ],
                max_tokens=8,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            action = int(raw.split()[0])
            if action not in range(5):
                raise ValueError(f"Action {action} out of range")
            return action
        except Exception as exc:
            logger.warning("LLM call failed (%s); falling back to allow", exc)
            return 0  # allow on error


# ---------------------------------------------------------------------------
# Stdout helpers — strict format per README
# ---------------------------------------------------------------------------

def emit_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={ENV_NAME} model={model}", flush=True)


def emit_step(step: int, action_name: str, reward: float, done: bool, error: str | None) -> None:
    error_str = error if error is not None else "null"
    done_str = "true" if done else "false"
    print(
        f"[STEP]  step={step} action={action_name} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def emit_end(success: bool, steps: int, rewards: list[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END]   success={success_str} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Timeout handler
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("Task timed out")


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

def run_task(
    task_name: str,
    agent: LLMAgent,
    model_name: str,
    seed: int = 42,
) -> None:
    """Run one full task episode, emitting START / STEP... / END lines."""
    emit_start(task_name, model_name)

    step_count = 0
    all_rewards: list[float] = []
    last_error: str | None = None
    success = False
    env = None

    use_signal = hasattr(signal, "SIGALRM")
    if use_signal:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(TASK_TIMEOUT_SECONDS)

    task_start = time.monotonic()

    try:
        config_path = TASK_CONFIGS[task_name]
        env = SocialGuardEnv(config_path)

        obs, info = env.reset(seed=seed)
        terminated = truncated = False

        while not (terminated or truncated):
            # Manual timeout check for Windows compatibility
            if time.monotonic() - task_start > TASK_TIMEOUT_SECONDS:
                raise TimeoutError("Task timed out (manual check)")

            step_count += 1
            error: str | None = None
            reward: float = 0.0

            try:
                action = agent.act(obs)
            except Exception as exc:
                action = 0
                error = str(exc)
                last_error = error

            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception as exc:
                reward = 0.0
                terminated = True
                error = str(exc)
                last_error = error

            done = terminated or truncated
            action_name = {0: "allow", 1: "warn", 2: "reduce_reach", 3: "remove", 4: "escalate"}.get(action, "allow")
            all_rewards.append(float(reward))

            emit_step(step_count, action_name, float(reward), done, error)

        # Determine success from task_success flag if available
        success = bool(info.get("task_success", not truncated))

    except TimeoutError:
        last_error = "timeout"
        success = False
    except Exception as exc:
        last_error = str(exc)
        success = False
        logger.exception("Unexpected error in task %s", task_name)
    finally:
        if use_signal:
            signal.alarm(0)  # cancel alarm
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        
        emit_end(success, step_count, all_rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    api_base_url = os.environ.get("API_BASE_URL", "")
    model_name   = os.environ.get("MODEL_NAME", "baseline")
    hf_token     = os.environ.get("HF_TOKEN", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")

    # Build LLM client — falls back to baseline agent if no API URL
    try:
        from openai import OpenAI
        if not hf_token:
            if openai_api_key:
                print(
                    "[WARN] OPENAI_API_KEY is set but this project requires HF_TOKEN.",
                    file=sys.stderr,
                )
            print("[ERROR] HF_TOKEN is not set. Exiting.", file=sys.stderr)
            sys.exit(1)
        if not api_base_url or not model_name:
            print("[ERROR] API_BASE_URL and MODEL_NAME must be set.", file=sys.stderr)
            sys.exit(1)
        client = OpenAI(
            api_key=hf_token or "no-key",
            base_url=api_base_url,
        )
        agent = LLMAgent(client, model_name)
    except ImportError:
        print("[WARN] openai package not installed; using baseline heuristics.", file=sys.stderr)
        from baseline import BaselineAgent
        class _FallbackAgent:
            def __init__(self) -> None:
                self._inner = BaselineAgent()
            def act(self, obs: np.ndarray) -> int:
                return self._inner.act(obs)
        agent = _FallbackAgent()  # type: ignore[assignment]

    total_start = time.monotonic()

    for task_name in TASK_CONFIGS:
        run_task(task_name, agent, model_name, seed=42)  # type: ignore[arg-type]

        elapsed = time.monotonic() - total_start
        remaining = (20 * 60) - elapsed
        if remaining <= 0:
            print("[WARN] 20-minute budget exhausted — skipping remaining tasks.", file=sys.stderr)
            break


if __name__ == "__main__":
    main()
