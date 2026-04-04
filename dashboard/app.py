"""
dashboard/app.py — Streamlit Dashboard for SocialGuard-RL.

Run live evaluation episodes step-by-step or automatically, displaying the
network state, latest actions, cumulative reward, and decision logs.

Features:
  - Load any trained .zip model or use the rule-based baseline.
  - Step forward manually or run continuously at up to 10 steps/second.
  - Live pyvis network graph with node coloring:
      green (real), red (bot), yellow (under review), gray (removed).
  - Metrics cards: total reward, bots removed (TP), innocents removed (FP), timestep.
  - Decision log table: timestep, action, ground truth, reward.
  - Cumulative reward chart (per-step).
  - Optional token-protected access via SOCIALGUARD_DASHBOARD_TOKEN env var.

Usage::
    streamlit run dashboard/app.py
"""

import hmac
import os
import time
import atexit
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from env.env import SocialGuardEnv
from baseline import BaselineAgent
from env.spaces import ACTION_NAMES, ACTION_REMOVE
from dashboard.graph_view import generate_graph_base_html, apply_decision_log
from dashboard.metrics_view import render_metrics_cards, render_decision_log, render_reward_chart


st.set_page_config(
    layout="wide",
    page_title="SocialGuard-RL Dashboard",
    page_icon="🛡️",
)

# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def _resolve_trusted_path(path_str: str, trusted_dir: str, suffixes: tuple[str, ...]) -> Path:
    root = (Path.cwd() / trusted_dir).resolve()
    p = Path(path_str)
    p = (Path.cwd() / p).resolve() if not p.is_absolute() else p.resolve()
    if root not in p.parents:
        raise ValueError(f"Path must be under `{trusted_dir}/`: {p}")
    if suffixes and p.suffix.lower() not in suffixes:
        raise ValueError(f"Path must end with one of {suffixes}: {p}")
    return p


def _check_token() -> None:
    """Block access unless a valid token is provided (if SOCIALGUARD_DASHBOARD_TOKEN is set)."""
    dashboard_token = os.environ.get("SOCIALGUARD_DASHBOARD_TOKEN", "").strip()
    if not dashboard_token:
        return
    provided = st.sidebar.text_input("🔑 Dashboard Token", type="password", key="token_input")
    if not provided:
        st.warning("Enter the dashboard token in the sidebar to continue.")
        st.stop()
    if not hmac.compare_digest(provided.strip(), dashboard_token):
        st.sidebar.error("❌ Invalid token.")
        st.stop()


# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------

def get_env(config_path: str) -> SocialGuardEnv:
    cfg_path = _resolve_trusted_path(config_path, "configs", (".yaml", ".yml"))
    cfg_mtime = cfg_path.stat().st_mtime
    if (
        "sg_env" not in st.session_state
        or st.session_state.get("sg_env_config_path") != str(cfg_path)
        or st.session_state.get("sg_env_config_mtime") != cfg_mtime
    ):
        if "sg_env" in st.session_state:
            try:
                st.session_state.sg_env.close()
            except Exception:
                pass
        st.session_state.sg_env = SocialGuardEnv(config_path=str(cfg_path))
        st.session_state.sg_env_config_path = str(cfg_path)
        st.session_state.sg_env_config_mtime = cfg_mtime
        # Invalidate graph cache when env changes
        st.session_state.pop("graph_base_html", None)
        st.session_state.pop("graph_base_sig", None)
    return st.session_state.sg_env


def _close_dashboard_env_on_exit() -> None:
    """Best-effort cleanup when Streamlit process exits/reloads."""
    try:
        env = st.session_state.get("sg_env")
        if env is not None:
            env.close()
    except Exception:
        pass


atexit.register(_close_dashboard_env_on_exit)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "step": 0,
        "running": False,
        "obs": None,
        "terminated": False,
        "truncated": False,
        "ep_reward": 0.0,
        "cumulative_rewards": [],   # per-step cumulative reward values
        "episode_rewards": [],      # per-episode totals for chart
        "log": [],
        "decision_log": {},
        "tp": 0,
        "fp": 0,
        "last_breakdown": {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_episode_state() -> None:
    st.session_state.step = 0
    st.session_state.terminated = False
    st.session_state.truncated = False
    st.session_state.ep_reward = 0.0
    st.session_state.cumulative_rewards = []
    st.session_state.log = []
    st.session_state.decision_log = {}
    st.session_state.tp = 0
    st.session_state.fp = 0
    st.session_state.last_breakdown = {}
    st.session_state.running = False
    # Invalidate graph cache so it's rebuilt for the new episode
    st.session_state.pop("graph_base_html", None)
    st.session_state.pop("graph_base_sig", None)


# ---------------------------------------------------------------------------
# Step logic
# ---------------------------------------------------------------------------

def step_agent(env: SocialGuardEnv, agent: Any) -> None:
    """Execute a single step using the loaded agent and update session state."""
    if st.session_state.terminated or st.session_state.truncated:
        return

    obs = st.session_state.obs

    if hasattr(agent, "predict"):
        action_arr, _ = agent.predict(obs, deterministic=True)
        action = int(action_arr.item() if hasattr(action_arr, "item") else action_arr)
    else:
        action = agent.act(obs)

    try:
        obs, reward, terminated, truncated, info = env.step(action)
    except Exception as e:
        st.session_state.running = False
        st.session_state.terminated = True
        st.session_state.truncated = True
        st.error(f"env.step() failed: {e}")
        return

    st.session_state.obs = obs
    st.session_state.terminated = terminated
    st.session_state.truncated = truncated
    st.session_state.ep_reward += float(reward)
    st.session_state.step += 1
    st.session_state.cumulative_rewards.append(st.session_state.ep_reward)

    gt = info.get("ground_truth", -1)
    act_str = ACTION_NAMES.get(action, str(action))

    if action == ACTION_REMOVE:
        if gt == 1:
            st.session_state.tp += 1
        elif gt == 0:
            st.session_state.fp += 1

    flagged_account = info.get("flagged_account", "N/A")
    flagged_reason = info.get("flagged_reason", "N/A")

    st.session_state.log.append({
        "Timestep": st.session_state.step,
        "Action": act_str,
        "Account": flagged_account,
        "Reason": flagged_reason,
        "Ground Truth": "Bot" if gt == 1 else "Real" if gt == 0 else "N/A",
        "Reward": round(float(reward), 4),
        "Cumulative": round(st.session_state.ep_reward, 4),
    })
    st.session_state.last_breakdown = info.get("reward_breakdown", {})

    # Track node decision for graph coloring (task_cib)
    state_dict = env.state()
    active_task = state_dict.get("active_task", "")
    if "cib" in active_task.lower():
        node_idx = info.get("entity_id")
        if node_idx is not None and int(node_idx) >= 0:
            st.session_state.decision_log[int(node_idx)] = {
                "action": action,
                "ground_truth": gt,
            }


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _get_live_graph(env: SocialGuardEnv) -> Any:
    """Try to extract the live NetworkX graph from the env's task."""
    task = env._task
    if task is not None and hasattr(task, "_graph") and task._graph is not None:
        if hasattr(task._graph, "graph"):
            return task._graph.graph
    return None


def _build_graph_html(env: SocialGuardEnv) -> str:
    """Return cached or freshly-generated pyvis HTML for the current episode graph."""
    import networkx as nx

    live_graph = _get_live_graph(env)

    if live_graph is not None:
        # Stable content-derived signature (not object id) to avoid stale caches.
        node_sample = tuple(sorted(int(n) for n in list(live_graph.nodes())[:32]))
        edge_sample = tuple(
            sorted(
                f"{int(min(u, v))}-{int(max(u, v))}"
                for u, v in list(live_graph.edges())[:64]
            )
        )
        sig = (
            live_graph.number_of_nodes(),
            live_graph.number_of_edges(),
            node_sample,
            edge_sample,
        )
    else:
        sig = ("dummy", 30, 0)

    if st.session_state.get("graph_base_sig") != sig:
        if live_graph is not None:
            G = live_graph
        else:
            # Fallback: generate a small illustrative placeholder
            G = nx.barabasi_albert_graph(30, 2, seed=42)
            # Mark a few nodes as bots for visual demonstration
            for n in list(G.nodes())[:5]:
                G.nodes[n]["is_bot"] = True

        st.session_state.graph_base_html = generate_graph_base_html(G)
        st.session_state.graph_base_sig = sig

    return apply_decision_log(
        st.session_state.graph_base_html,
        st.session_state.decision_log,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    init_session_state()

    # --- Auth ---
    _check_token()

    # --- Sidebar ---
    st.sidebar.title("⚙️ Configuration")

    config_file = st.sidebar.selectbox(
        "Task Config",
        options=[
            "configs/task1.yaml",
            "configs/task2.yaml",
            "configs/task3.yaml",
        ],
        index=0,
    )
    agent_type = st.sidebar.selectbox(
        "Agent",
        ["Rule-based Baseline", "Trained Model"],
    )

    model_file = ""
    if agent_type == "Trained Model":
        model_file = st.sidebar.text_input("Model path (.zip)", value="models/ppo_task1/final_model.zip")

    auto_speed = st.sidebar.slider("Auto-play speed (steps/sec)", min_value=2, max_value=10, value=5)

    st.sidebar.divider()

    if st.sidebar.button("🔄 Reset Episode", use_container_width=True):
        try:
            env = get_env(config_file)
            st.session_state.obs, _ = env.reset()
            reset_episode_state()
        except Exception as e:
            st.sidebar.error(f"Reset failed: {e}")
        st.rerun()

    # --- Load env ---
    try:
        env = get_env(config_file)
    except Exception as e:
        st.error(f"❌ Failed to load environment: {e}")
        return

    if st.session_state.obs is None:
        try:
            st.session_state.obs, _ = env.reset()
        except Exception as e:
            st.error(f"❌ env.reset() failed: {e}")
            return

    # --- Load agent ---
    try:
        if agent_type == "Rule-based Baseline":
            agent = BaselineAgent()
        else:
            if not model_file:
                st.error("Please enter a model path.")
                return
            from evaluate import load_model
            agent = load_model(model_file)
    except ValueError as e:
        st.error(f"❌ Invalid model path: {e}")
        st.info("Use a `.zip` model under the `models/` directory.")
        return
    except Exception as e:
        st.error(f"❌ Failed to load agent: {e}")
        return

    # --- Header ---
    st.title("🛡️ SocialGuard-RL Dashboard")

    task_label = config_file.replace("configs/", "").replace(".yaml", "")
    cols = st.columns([2, 1, 1])
    cols[0].caption(f"**Task:** `{task_label}` · **Agent:** {agent_type}")
    if st.session_state.terminated or st.session_state.truncated:
        success = st.session_state.terminated and not st.session_state.truncated
        cols[1].success("✅ Episode Complete") if success else cols[1].warning("⏹️ Episode Ended")
    else:
        cols[1].info("▶️ Running...")

    # --- Metrics ---
    render_metrics_cards(
        st.session_state.ep_reward,
        st.session_state.tp,
        st.session_state.fp,
        st.session_state.step,
    )

    # --- Controls ---
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⏭️ Step Forward", use_container_width=True, disabled=(st.session_state.terminated or st.session_state.truncated)):
            step_agent(env, agent)
            st.rerun()
    with c2:
        if st.button("▶️ Play", use_container_width=True, disabled=st.session_state.running):
            st.session_state.running = True
            st.rerun()
    with c3:
        if st.button("⏸️ Pause", use_container_width=True, disabled=not st.session_state.running):
            st.session_state.running = False
            st.rerun()

    # --- Main Tabs ---
    tab_log, tab_graph, tab_rewards, tab_breakdown = st.tabs(
        ["📋 Decision Log", "🕸️ Network Graph", "📈 Reward Chart", "🔍 Last Reward"]
    )

    with tab_log:
        render_decision_log(list(reversed(st.session_state.log[-100:])))

    with tab_graph:
        state_dict = env.state()
        active_task = state_dict.get("active_task", "")

        if "cib" in active_task.lower():
            st.caption("Node colors: 🟢 Real user · 🔴 Bot · 🟡 Under review · ⬜ Removed")
            try:
                html = _build_graph_html(env)
                components.html(html, height=640, scrolling=False)
            except Exception as e:
                st.warning(f"Graph unavailable: {e}")
        elif "spam" in active_task.lower():
            if st.session_state.obs is not None:
                obs = st.session_state.obs
                feature_names = [
                    "age",
                    "posts/hr",
                    "follower_ratio",
                    "login_var",
                    "content_rep",
                    "profile",
                    "device",
                    "ip_div",
                ]
                df_feat = pd.DataFrame(
                    {"Feature": feature_names, "Value": [float(obs[i]) for i in range(8)]}
                )
                st.bar_chart(df_feat.set_index("Feature"))
                gt = st.session_state.log[-1]["Ground Truth"] if st.session_state.log else "?"
                st.caption(f"Current account ground truth: **{gt}**")
            else:
                st.info("Reset episode to populate task features.")
        elif "misinfo" in active_task.lower():
            if st.session_state.obs is not None:
                obs = st.session_state.obs
                feat_names = [
                    "spread_rate",
                    "fact_check",
                    "engagement",
                    "credibility",
                    "hop_count",
                    "timestep",
                ]
                df_feat = pd.DataFrame(
                    {"Feature": feat_names, "Value": [float(obs[i]) for i in range(6)]}
                )
                st.bar_chart(df_feat.set_index("Feature"))
            else:
                st.info("Reset episode to populate task features.")
        else:
            st.info("No active task graph available yet.")

    with tab_rewards:
        if st.session_state.cumulative_rewards:
            df = pd.DataFrame({
                "Step": range(1, len(st.session_state.cumulative_rewards) + 1),
                "Cumulative Reward": st.session_state.cumulative_rewards,
            }).set_index("Step")
            st.line_chart(df)
        else:
            st.info("Cumulative reward chart will appear after the first step.")

        if st.session_state.episode_rewards:
            df_eps = pd.DataFrame({
                "Episode": range(1, len(st.session_state.episode_rewards) + 1),
                "Total Reward": st.session_state.episode_rewards,
            }).set_index("Episode")
            st.caption("Episode reward history")
            st.bar_chart(df_eps)

    with tab_breakdown:
        if st.session_state.log:
            breakdown = st.session_state.get("last_breakdown", {}) or {}
            if breakdown:
                cols = st.columns(5)
                labels = ["correctness", "fp_cost", "collateral_damage", "speed_bonus", "escalation_penalty"]
                for col, lbl in zip(cols, labels):
                    col.metric(lbl, f"{float(breakdown.get(lbl, 0.0)):.3f}")
                st.caption(f"Total: **{float(breakdown.get('total', 0.0)):.4f}**")
            else:
                st.info("Step forward to see reward breakdown.")
        else:
            st.info("No steps yet.")

    # --- Auto-play loop ---
    if st.session_state.running:
        if st.session_state.terminated or st.session_state.truncated:
            st.session_state.running = False
            st.session_state.episode_rewards.append(st.session_state.ep_reward)
            st.sidebar.success(f"Episode finished. Reward: {st.session_state.ep_reward:.2f}")
            with st.expander("📊 Episode Summary", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Reward", f"{st.session_state.ep_reward:.2f}")
                c2.metric("Bots Removed", st.session_state.tp)
                c3.metric("Innocents Removed", st.session_state.fp)
                c4.metric("Steps", st.session_state.step)
                if st.session_state.tp + st.session_state.fp > 0:
                    precision = st.session_state.tp / (st.session_state.tp + st.session_state.fp)
                    st.progress(precision, text=f"Precision: {precision:.1%}")
        else:
            step_agent(env, agent)
            time.sleep(max(0.0, 1.0 / max(float(auto_speed), 1.0)))
            st.rerun()


if __name__ == "__main__":
    main()
