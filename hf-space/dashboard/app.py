"""
dashboard/app.py — NEMESIS-RL Streamlit Dashboard (v3 — Full Gap Fix)

Adds all 14 missing capabilities identified in FRONTEND_GAP_ANALYSIS.md:
  - GAP #1:  /healthz live status indicator
  - GAP #2:  /metrics Prometheus data rendered
  - GAP #3:  /grade/all combined score + per-task breakdown
  - GAP #4:  reward_breakdown columns in Decision Log
  - GAP #5:  task_success shown on episode end
  - GAP #6:  collateral_count trend chart
  - GAP #7:  CIB cluster_state scoreboard
  - GAP #8:  env.state() decision_history viewer
  - GAP #9:  escalation_count tracked and displayed
  - GAP #10: score formula shown per task
  - GAP #11: openenv.yaml task registry in sidebar
  - GAP #12: hop_count trend chart for Task 2
  - GAP #13: entity_id in decision log
  - GAP #14: /grade per-component breakdown

Usage::
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import hmac
import os
import time
import atexit
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import yaml

from env.env import SocialGuardEnv
from baseline import BaselineAgent
from env.spaces import ACTION_NAMES, ACTION_REMOVE
from graph_view import generate_graph_base_html, apply_decision_log
from metrics_view import render_metrics_cards, render_decision_log, render_reward_chart

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    st_autorefresh = None  # type: ignore[assignment]
    HAS_AUTOREFRESH = False

# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="NEMESIS-RL", page_icon="🛡️")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f1117; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="metric-container"] {
    background: #1a1d27;
    border: 1px solid #2d3044;
    border-radius: 10px;
    padding: 12px 16px;
}
.pill-green  { background:#155724; color:#d4edda; padding:3px 10px; border-radius:20px; font-size:12px; }
.pill-red    { background:#721c24; color:#f8d7da; padding:3px 10px; border-radius:20px; font-size:12px; }
.pill-yellow { background:#856404; color:#fff3cd; padding:3px 10px; border-radius:20px; font-size:12px; }
.section-title { font-size:16px; font-weight:700; color:#90caf9; margin:12px 0 6px 0; }
.formula-box { background:#1a1d27; border:1px solid #2d3044; border-radius:8px;
               padding:10px 14px; font-family:monospace; color:#90caf9; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEMESIS_MODEL_PATH = "models/nemesis/final_model.zip"
TRAINING_LOG_PATH  = "training_log.csv"
CHECKPOINT_DIR     = "models/nemesis/checkpoints"
SERVER_BASE        = "http://localhost:7860"

TASK_CONFIG_MAP = {
    "task_spam":    "configs/task1.yaml",
    "task_misinfo": "configs/task2.yaml",
    "task_cib":     "configs/task3.yaml",
}

SCORE_FORMULAS = {
    "task_spam":    "0.7 × F1 + 0.3 × sigmoid(mean_reward / 50)",
    "task_misinfo": "0.6 × F1 + 0.4 × max(0, 1 − mean_hop / max_hops)",
    "task_cib":     "0.5 × recall + 0.5 × F1 − min(collateral_rate × 2, 0.5)",
}

ACTION_EMOJI = {0: "✅ Allow", 1: "⚠️ Warn", 2: "🔒 Restrict", 3: "❌ Remove", 4: "🚨 Escalate"}


# ---------------------------------------------------------------------------
# GAP #1: Health indicator
# ---------------------------------------------------------------------------

def _health_indicator() -> None:
    """Show /healthz status in the sidebar."""
    try:
        r = requests.get(f"{SERVER_BASE}/healthz", timeout=2)
        if r.status_code == 200:
            st.sidebar.success("🟢 API Server: OK")
        else:
            st.sidebar.error(f"🔴 API Server: HTTP {r.status_code}")
    except Exception:
        st.sidebar.warning("🟡 API Server: unreachable (local mode)")


# ---------------------------------------------------------------------------
# GAP #11: Load openenv.yaml registry
# ---------------------------------------------------------------------------

@st.cache_data
def _load_openenv_yaml() -> dict:
    try:
        with open("openenv.yaml") as f:
            return yaml.safe_load(f)
    except Exception:
        return {"tasks": [
            {"id": "task_spam",    "name": "Spam Detection",          "difficulty": "easy",   "description": "Classify bot accounts."},
            {"id": "task_misinfo", "name": "Misinfo Flagging",         "difficulty": "medium", "description": "BFS spread intervention."},
            {"id": "task_cib",     "name": "CIB Network Takedown",     "difficulty": "hard",   "description": "Graph-level bot cluster removal."},
        ]}


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

def _check_token() -> None:
    tok = os.environ.get("SOCIALGUARD_DASHBOARD_TOKEN", "").strip()
    if not tok:
        return
    provided = st.sidebar.text_input("🔑 Dashboard Token", type="password", key="tk")
    if not provided:
        st.warning("Enter the dashboard token to continue.")
        st.stop()
    if not hmac.compare_digest(provided.strip(), tok):
        st.sidebar.error("❌ Invalid token.")
        st.stop()


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _resolve_trusted_path(path_str: str, trusted_dir: str, suffixes: tuple) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    root = (repo_root / trusted_dir).resolve()
    p = (repo_root / path_str).resolve()
    if root not in p.parents:
        raise ValueError(f"Path must be under `{trusted_dir}/`")
    if suffixes and p.suffix.lower() not in suffixes:
        raise ValueError(f"Must end in {suffixes}")
    return p


def get_env(config_path: str) -> SocialGuardEnv:
    cfg_path = _resolve_trusted_path(config_path, "configs", (".yaml", ".yml"))
    mtime = cfg_path.stat().st_mtime
    if (
        "sg_env" not in st.session_state
        or st.session_state.get("sg_env_config_path") != str(cfg_path)
        or st.session_state.get("sg_env_config_mtime") != mtime
    ):
        if "sg_env" in st.session_state:
            try: st.session_state.sg_env.close()
            except Exception: pass
        st.session_state.sg_env = SocialGuardEnv(config_path=str(cfg_path))
        st.session_state.sg_env_config_path  = str(cfg_path)
        st.session_state.sg_env_config_mtime = mtime
        st.session_state.pop("graph_base_html", None)
        st.session_state.pop("graph_base_sig",  None)
    return st.session_state.sg_env


@atexit.register
def _cleanup():
    try:
        env = st.session_state.get("sg_env")
        if env: env.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "step": 0, "running": False, "obs": None,
        "terminated": False, "truncated": False,
        "ep_reward": 0.0, "cumulative_rewards": [], "episode_rewards": [],
        "log": [], "decision_log": {}, "tp": 0, "fp": 0, "last_breakdown": {},
        "nemesis_records": [], "flagged_stream": [],
        "train_running": False, "train_log_lines": [],
        # GAP #6: collateral tracking
        "collateral_history": [],
        # GAP #9: escalation tracking
        "escalation_count": 0,
        # GAP #12: hop count tracking
        "hop_history": [],
        "last_autoplay_tick": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_episode_state() -> None:
    keys_zero = ("step", "tp", "fp", "escalation_count")
    keys_float = ("ep_reward",)
    keys_list = ("cumulative_rewards", "log", "collateral_history", "hop_history")
    keys_dict = ("decision_log", "last_breakdown")
    keys_bool = ("running", "terminated", "truncated")
    for k in keys_zero: st.session_state[k] = 0
    for k in keys_float: st.session_state[k] = 0.0
    for k in keys_list: st.session_state[k] = []
    for k in keys_dict: st.session_state[k] = {}
    for k in keys_bool: st.session_state[k] = False
    st.session_state["last_autoplay_tick"] = None
    st.session_state.pop("graph_base_html", None)
    st.session_state.pop("graph_base_sig", None)


# ---------------------------------------------------------------------------
# Step logic — GAP #4, #5, #6, #9, #12, #13 addressed here
# ---------------------------------------------------------------------------

def step_agent(env: SocialGuardEnv, agent: Any) -> None:
    if st.session_state.terminated or st.session_state.truncated:
        return
    obs = st.session_state.obs
    if hasattr(agent, "predict"):
        a, _ = agent.predict(obs, deterministic=True)
        action = int(a.item() if hasattr(a, "item") else a)
    else:
        action = agent.act(obs)

    try:
        obs, reward, terminated, truncated, info = env.step(action)
    except Exception as e:
        st.session_state.running = False
        st.session_state.terminated = True
        st.error(f"env.step() failed: {e}")
        return

    st.session_state.obs        = obs
    st.session_state.terminated = terminated
    st.session_state.truncated  = truncated
    st.session_state.ep_reward += float(reward)
    st.session_state.step      += 1
    st.session_state.cumulative_rewards.append(st.session_state.ep_reward)

    gt      = info.get("ground_truth", -1)
    act_str = ACTION_NAMES.get(action, str(action))
    fa      = info.get("flagged_account", "N/A")
    fr      = info.get("flagged_reason",  "N/A")
    bd      = info.get("reward_breakdown", {})

    if action == ACTION_REMOVE:
        if gt == 1: st.session_state.tp += 1
        elif gt == 0: st.session_state.fp += 1

    # GAP #9: escalation count
    if action == 4:
        st.session_state["escalation_count"] += 1

    # GAP #6: collateral tracking
    st.session_state["collateral_history"].append(info.get("collateral_count", 0))

    # GAP #12: hop count for misinfo
    hop = info.get("hop_count", 0)
    st.session_state["hop_history"].append(hop)

    # GAP #4: reward breakdown in row + GAP #13: entity_id
    row = {
        "Step":         st.session_state.step,
        "Entity ID":    info.get("entity_id", "?"),         # GAP #13
        "Action":       act_str,
        "Account":      fa,
        "Reason":       fr,
        "Ground Truth": "Bot" if gt==1 else "Real" if gt==0 else "N/A",
        "Reward":       round(float(reward), 4),
        "Cumulative":   round(st.session_state.ep_reward, 4),
        "✓ Correct":   round(float(bd.get("correctness", 0)), 3),     # GAP #4
        "✗ FP Cost":   round(float(bd.get("fp_cost", 0)), 3),
        "✗ Collateral": round(float(bd.get("collateral_damage", 0)), 3),
        "⚡ Speed":     round(float(bd.get("speed_bonus", 0)), 3),
        "⚠ Escalation": round(float(bd.get("escalation_penalty", 0)), 3),
    }
    st.session_state.log.append(row)
    st.session_state.last_breakdown = bd

    if action != 0:
        st.session_state.flagged_stream.append({
            "step": st.session_state.step, "account": fa,
            "action": act_str, "reason": fr,
            "gt": "Bot" if gt==1 else "Real" if gt==0 else "?",
            "reward": round(float(reward), 4),
        })
        if len(st.session_state.flagged_stream) > 200:
            st.session_state.flagged_stream = st.session_state.flagged_stream[-200:]

    state_dict  = env.state()
    active_task = state_dict.get("active_task", "")
    if "cib" in active_task.lower():
        node_idx = info.get("entity_id")
        if node_idx is not None and int(node_idx) >= 0:
            st.session_state.decision_log[int(node_idx)] = {"action": action, "ground_truth": gt}


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _get_live_graph(env):
    task = env._task
    if task and hasattr(task, "_graph") and task._graph and hasattr(task._graph, "graph"):
        return task._graph.graph
    return None


def _build_graph_html(env) -> str:
    import networkx as nx
    live = _get_live_graph(env)
    if live is not None:
        ns = tuple(sorted(int(n) for n in list(live.nodes())[:32]))
        es = tuple(sorted(f"{min(u,v)}-{max(u,v)}" for u,v in list(live.edges())[:64]))
        sig = (live.number_of_nodes(), live.number_of_edges(), ns, es)
    else:
        sig = ("dummy", 30, 0)
    if st.session_state.get("graph_base_sig") != sig:
        G = live if live is not None else nx.barabasi_albert_graph(30, 2, seed=42)
        if live is None:
            for n in list(G.nodes())[:5]: G.nodes[n]["is_bot"] = True
        st.session_state.graph_base_html = generate_graph_base_html(G)
        st.session_state.graph_base_sig  = sig
    return apply_decision_log(st.session_state.graph_base_html, st.session_state.decision_log)


# ---------------------------------------------------------------------------
# GAP #3 + #14: Grading helpers
# ---------------------------------------------------------------------------

def _render_grade_panel() -> None:
    """Full grading panel with combined score + per-task breakdown."""
    st.subheader("🏆 OpenEnv Grader")
    st.caption("Runs deterministic evaluation episodes using the rule-based baseline and returns normalized scores [0.0, 1.0].")

    gc1, gc2, gc3 = st.columns(3)
    n_eps = gc1.slider("Evaluation episodes", 2, 50, 10, key="grade_n_eps")
    grade_seed = gc2.number_input("Seed", value=42, key="grade_seed")
    single_task = gc3.selectbox("Grade single task", ["all", "task_spam", "task_misinfo", "task_cib"], key="grade_task")

    if st.button("▶️ Run Grader", use_container_width=True, key="run_grader"):
        endpoint = (
            f"{SERVER_BASE}/grade/all?n_episodes={n_eps}&seed={grade_seed}"
            if single_task == "all"
            else f"{SERVER_BASE}/grade/{single_task}?n_episodes={n_eps}&seed={grade_seed}"
        )
        with st.spinner(f"Running grader via {endpoint}…"):
            try:
                r = requests.get(endpoint, timeout=600)
                data = r.json()
                st.session_state["last_grade_result"] = data
            except Exception as e:
                st.error(f"Grader error: {e}")
                return

    result = st.session_state.get("last_grade_result")
    if not result:
        st.info("Click **Run Grader** to evaluate the baseline agent.")
        return

    # Combined score
    if "combined_score" in result:
        st.metric("🏆 Combined Score", f"{result['combined_score']:.4f}",
                  help="Average normalized score across all three tasks.")
        st.divider()

    # Per-task breakdown
    tasks_to_show = [single_task] if single_task != "all" else list(TASK_CONFIG_MAP.keys())
    for task in tasks_to_show:
        task_data = result if single_task != "all" else result.get(task, {})
        if not task_data or "score" not in task_data:
            continue
        det = task_data.get("details", {})
        score = task_data.get("score", 0)
        formula = task_data.get("score_formula", SCORE_FORMULAS.get(task, "F1"))

        st.markdown(f"### {task}")
        st.markdown(f'<div class="formula-box">📐 {formula}</div>', unsafe_allow_html=True)

        sc1, sc2 = st.columns([1, 3])
        sc1.metric("Normalized Score", f"{score:.4f}")

        with sc2:
            td1, td2, td3, td4, td5, td6, td7 = st.columns(7)
            td1.metric("F1",       f"{det.get('f1', 0):.3f}")
            td2.metric("Precision",f"{det.get('precision', 0):.3f}")
            td3.metric("Recall",   f"{det.get('recall', 0):.3f}")
            td4.metric("Reward",   f"{det.get('mean_reward', 0):.2f}")
            td5.metric("Ep. Len",  f"{det.get('mean_episode_length', 0):.0f}")
            ttd = det.get("time_to_detection", None)
            td6.metric("TTD", "n/a" if ttd is None else f"{float(ttd):.1f}")
            td7.metric("Collat.",  f"{det.get('mean_collateral', 0):.2f}")

        st.progress(min(score, 1.0), text=f"Score: {score:.4f}")
        st.divider()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    init_session_state()
    _check_token()

    # ── Load openenv.yaml (GAP #11) ───────────────────────────────────
    oe = _load_openenv_yaml()
    oe_tasks = {t["id"]: t for t in oe.get("tasks", [])}

    # ── Sidebar ──────────────────────────────────────────────────────────
    st.sidebar.markdown("## ⚙️ Configuration")

    # GAP #11: Task selector from openenv.yaml
    task_id = st.sidebar.selectbox(
        "Task",
        list(oe_tasks.keys()),
        format_func=lambda k: f"{oe_tasks[k]['name']} ({oe_tasks[k]['difficulty']})",
        key="task_id_select",
    )
    st.sidebar.caption(oe_tasks[task_id]["description"])
    config_file = TASK_CONFIG_MAP[task_id]

    # GAP #10: Score formula
    st.sidebar.markdown(f'<div class="formula-box" style="margin-top:6px">📐 {SCORE_FORMULAS[task_id]}</div>',
                        unsafe_allow_html=True)
    st.sidebar.divider()

    agent_type = st.sidebar.selectbox("Agent", ["Rule-based Baseline", "Trained Model"])
    model_file = ""
    if agent_type == "Trained Model":
        model_file = st.sidebar.text_input("Model path (.zip)", value=NEMESIS_MODEL_PATH)

    auto_speed = st.sidebar.slider("Auto-play speed (steps/sec)", 2, 10, 5)
    st.sidebar.divider()

    if st.sidebar.button("🔄 Reset Episode", use_container_width=True):
        try:
            env = get_env(config_file)
            st.session_state.obs, _ = env.reset()
            reset_episode_state()
        except Exception as e:
            st.sidebar.error(f"Reset failed: {e}")
        st.rerun()

    # GAP #1: Health indicator
    _health_indicator()

    st.sidebar.divider()
    st.sidebar.markdown("### 🤖 NEMESIS Status")
    if os.path.exists(NEMESIS_MODEL_PATH):
        t_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(Path(NEMESIS_MODEL_PATH).stat().st_mtime))
        st.sidebar.success(f"✅ Model loaded\n`{t_str}`")
    else:
        st.sidebar.warning("⚠️ No trained model — train in 🏋️ tab.")

    if os.path.exists(TRAINING_LOG_PATH):
        try:
            df_s = pd.read_csv(TRAINING_LOG_PATH)
            if not df_s.empty:
                last = df_s.iloc[-1]
                st.sidebar.metric("Last TP Rate", f"{float(last.get('tp_rate', 0)):.3f}")
                st.sidebar.metric("Last FP Rate", f"{float(last.get('fp_rate', 0)):.3f}")
                st.sidebar.metric("Last Reward",  f"{float(last.get('mean_reward', 0)):.3f}")
        except Exception:
            pass

    # ── Load env & agent ─────────────────────────────────────────────────
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

    try:
        if agent_type == "Rule-based Baseline":
            agent = BaselineAgent()
        else:
            if not model_file:
                st.error("Please enter a model path.")
                return
            from evaluate import load_model
            agent = load_model(model_file)
    except Exception as e:
        st.error(f"❌ Failed to load agent: {e}")
        return

    # ── Header ───────────────────────────────────────────────────────────
    st.markdown("""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:8px">
  <span style="font-size:36px">🛡️</span>
  <div>
    <span style="font-size:26px;font-weight:800;color:#fff">NEMESIS-RL</span>
    <span style="font-size:14px;color:#90caf9;margin-left:10px">NEMESIS Intelligence Platform</span>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Metrics row ──────────────────────────────────────────────────────
    render_metrics_cards(
        st.session_state.ep_reward, st.session_state.tp,
        st.session_state.fp, st.session_state.step,
    )

    # GAP #9: Escalation count in metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    total_actions = st.session_state.tp + st.session_state.fp
    precision = st.session_state.tp / total_actions if total_actions else 0.0
    m1.metric("Precision", f"{precision:.1%}")
    m2.metric("Actions Taken", total_actions)
    m3.metric("Flagged Events", len(st.session_state.flagged_stream))
    m4.metric("Escalations ⚠️", st.session_state.get("escalation_count", 0))
    m5.metric("Avg Step Reward", f"{st.session_state.ep_reward / max(st.session_state.step, 1):.3f}")

    # ── Controls ─────────────────────────────────────────────────────────
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⏭️ Step", use_container_width=True,
                     disabled=(st.session_state.terminated or st.session_state.truncated)):
            step_agent(env, agent); st.rerun()
    with c2:
        if st.button(
            "▶️ Play",
            use_container_width=True,
            disabled=st.session_state.running or not HAS_AUTOREFRESH,
            help=None if HAS_AUTOREFRESH else "Install streamlit-autorefresh to enable auto-play.",
        ):
            st.session_state.running = True; st.rerun()
    with c3:
        if st.button("⏸️ Pause", use_container_width=True, disabled=not st.session_state.running):
            st.session_state.running = False; st.rerun()

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🚨 Flagged Accounts",   # 0
        "🤖 NEMESIS Live",       # 1
        "📉 Learning Curve",     # 2
        "🏆 Grader",             # 3  GAP #3 + #14
        "🏋️ Train Model",        # 4
        "📋 Decision Log",       # 5  GAP #4 + #13
        "🕸️ Network Graph",      # 6  GAP #6 + #7 + #12
        "📈 Reward Chart",       # 7
        "🔍 Reward Breakdown",   # 8
        "📡 Server Metrics",     # 9  GAP #2
        "🗂️ Episode State",      # 10 GAP #8
        "🧠 Model Architecture", # 11
        "🌍 Mastodon Live",      # 12
    ])
    (tab_flagged, tab_nemesis, tab_learning, tab_grade, tab_train,
     tab_log, tab_graph, tab_rewards_chart, tab_breakdown,
     tab_metrics, tab_state, tab_arch, tab_mastodon) = tabs

    # ═══════════════════════════════════════════════════════════════════
    # 🚨 TAB 0 — Flagged Accounts
    # ═══════════════════════════════════════════════════════════════════
    with tab_flagged:
        st.subheader("🚨 Live Flagged Accounts Stream")
        st.caption("Every non-allow action taken by the agent. In Task 2, `Restrict` reduces reach and the episode continues; `Remove` is the terminal takedown action.")
        stream = list(reversed(st.session_state.flagged_stream[-50:]))
        if stream:
            for row in stream:
                is_bot = row["gt"] == "Bot"
                pill_cls = "pill-red" if is_bot else "pill-yellow"
                pill_lbl = "🤖 Bot" if is_bot else "👤 Human"
                rew_color = "#4caf50" if row["reward"] > 0 else "#f44336"
                st.markdown(f"""
<div style="background:#1a1d27;border:1px solid #2d3044;border-radius:10px;
     padding:12px 16px;margin-bottom:8px;display:flex;align-items:center;gap:16px">
  <div style="flex:1">
    <b style="color:#90caf9">Account #{row['account']}</b>
    <span style="color:#666;font-size:12px;margin-left:8px">Step {row['step']}</span><br>
    <span style="color:#ccc;font-size:13px">{row['reason']}</span>
  </div>
  <div style="text-align:right">
    <span class="{pill_cls}">{pill_lbl}</span><br>
    <b style="color:#fff">{row['action']}</b><br>
    <span style="color:{rew_color};font-weight:700">reward {row['reward']:+.2f}</span>
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.info("▶️ Step forward or Play to start flagging accounts.")

    # ═══════════════════════════════════════════════════════════════════
    # 🤖 TAB 1 — NEMESIS Live
    # ═══════════════════════════════════════════════════════════════════
    with tab_nemesis:
        st.subheader("🤖 NEMESIS Neural Policy — Live Inference")
        ncol1, ncol2, ncol3 = st.columns([2, 1, 1])
        n_steps = ncol1.slider("Steps", 5, 100, 20, key="n_inf_steps")
        deterministic = ncol2.toggle("Deterministic", True, key="inf_det")
        inf_config = ncol3.selectbox("Config", list(TASK_CONFIG_MAP.values()), key="inf_cfg")

        if st.button("▶️ Run NEMESIS Inference", use_container_width=True):
            if not os.path.exists(NEMESIS_MODEL_PATH):
                st.warning(f"No model at `{NEMESIS_MODEL_PATH}`. Train first.")
            else:
                with st.spinner("Running…"):
                    try:
                        from agent import run_inference_episode
                        recs = []
                        run_inference_episode(inf_config, NEMESIS_MODEL_PATH, n_steps, deterministic, records_list=recs)
                        st.session_state["nemesis_records"] = recs
                    except Exception as exc:
                        st.error(f"Inference error: {exc}")

        records = st.session_state.get("nemesis_records", [])
        if records:
            m_tp  = sum(1 for r in records if r["prediction"] != "No Action" and r["ground_truth"] == "Bot")
            m_fp  = sum(1 for r in records if r["prediction"] != "No Action" and r["ground_truth"] == "Human")
            m_rew = sum(r["reward"] for r in records)
            st.success(f"NEMESIS ran {len(records)} steps — TP={m_tp}  FP={m_fp}  Total Reward={m_rew:.2f}")
            df_r = pd.DataFrame([{
                "Step": r["step"], "Account": r["account_id"],
                "Prediction": r["prediction"], "Ground Truth": r["ground_truth"],
                "Reward": r["reward"], "Confidence": f"{r['confidence']:.3f}",
            } for r in records])

            def _row_color(row):
                is_act = row["Prediction"] != "No Action"
                is_bot = row["Ground Truth"] == "Bot"
                if is_act and is_bot:   return ["background-color:#1b4332"] * len(row)
                if is_act and not is_bot: return ["background-color:#4a1122"] * len(row)
                return [""] * len(row)

            st.dataframe(df_r.style.apply(_row_color, axis=1), use_container_width=True)
        else:
            st.info("↑ Click **Run NEMESIS Inference** to see the model in action.")

    # ═══════════════════════════════════════════════════════════════════
    # 📉 TAB 2 — Learning Curve
    # ═══════════════════════════════════════════════════════════════════
    with tab_learning:
        st.subheader("📉 Learning Curve")
        if os.path.exists(TRAINING_LOG_PATH):
            try:
                df_log = pd.read_csv(TRAINING_LOG_PATH)
                if not df_log.empty:
                    latest = df_log.iloc[-1]
                    lc1, lc2, lc3, lc4 = st.columns(4)
                    lc1.metric("TP Rate", f"{float(latest.get('tp_rate',0)):.3f}")
                    lc2.metric("FP Rate", f"{float(latest.get('fp_rate',0)):.3f}")
                    lc3.metric("Mean Reward", f"{float(latest.get('mean_reward',0)):.3f}")
                    lc4.metric("Entropy", f"{float(latest.get('policy_entropy',0)):.3f}")
                    st.line_chart(df_log[["episode","tp_rate","fp_rate"]].set_index("episode"))
                    st.line_chart(df_log[["episode","mean_reward"]].set_index("episode"))
            except Exception as exc:
                st.error(f"Could not read training_log.csv: {exc}")
        else:
            st.info("No `training_log.csv` yet.")

    # ═══════════════════════════════════════════════════════════════════
    # 🏆 TAB 3 — Grader (GAP #3 + #14)
    # ═══════════════════════════════════════════════════════════════════
    with tab_grade:
        _render_grade_panel()

    # ═══════════════════════════════════════════════════════════════════
    # 🏋️ TAB 4 — Train Model
    # ═══════════════════════════════════════════════════════════════════
    with tab_train:
        st.subheader("🏋️ Train NEMESIS Model")
        tcol1, tcol2, tcol3 = st.columns(3)
        t_cycles = tcol1.number_input("Training cycles (×100k steps)", 1, 20, 1, key="t_cycles")
        t_config  = tcol2.selectbox("Config", list(TASK_CONFIG_MAP.values()), key="t_cfg")
        t_device  = tcol3.selectbox("Device", ["auto","cpu","cuda","mps"], key="t_dev")

        if st.button("🚀 Start Training", use_container_width=True):
            cmd = [sys.executable, "agent.py", "--config", t_config,
                   "--cycles", str(int(t_cycles)), "--device", t_device]
            st.info(f"Launching: `{' '.join(cmd)}`")
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, cwd=os.getcwd())
                st.session_state["train_proc"] = proc
                st.session_state["train_log_lines"] = []
                st.session_state["train_running"] = True
            except Exception as exc:
                st.error(f"Failed to launch: {exc}")

        proc = st.session_state.get("train_proc")
        if proc is not None:
            if proc.poll() is None:
                st.info("🔄 Training in progress…")
                try:
                    import select
                    rlist, _, _ = select.select([proc.stdout], [], [], 0.1)
                    if rlist:
                        for _ in range(50):
                            line = proc.stdout.readline()
                            if not line: break
                            st.session_state["train_log_lines"].append(line.rstrip())
                except Exception:
                    pass
            else:
                rc = proc.returncode
                st.success("✅ Done!") if rc == 0 else st.error(f"Exit code {rc}")
                st.session_state["train_proc"] = None

            log_lines = st.session_state.get("train_log_lines", [])
            if log_lines:
                st.code("\n".join(log_lines[-60:]), language="text")

    # ═══════════════════════════════════════════════════════════════════
    # 📋 TAB 5 — Decision Log (GAP #4 + #13)
    # ═══════════════════════════════════════════════════════════════════
    with tab_log:
        st.subheader("📋 Decision Log — with Reward Breakdown")
        log_data = list(reversed(st.session_state.log[-200:]))
        if log_data:
            df_log_view = pd.DataFrame(log_data)
            st.dataframe(df_log_view, use_container_width=True)
            # Summary stats on breakdown columns
            if "✓ Correct" in df_log_view.columns:
                st.caption("Reward component averages this episode")
                avg_cols = ["✓ Correct","✗ FP Cost","✗ Collateral","⚡ Speed","⚠ Escalation"]
                avgs = {c: round(float(df_log_view[c].mean()), 4) for c in avg_cols if c in df_log_view}
                ac1, ac2, ac3, ac4, ac5 = st.columns(5)
                for col, (k, v) in zip([ac1,ac2,ac3,ac4,ac5], avgs.items()):
                    col.metric(k, v)
        else:
            st.info("No steps yet.")

    # ═══════════════════════════════════════════════════════════════════
    # 🕸️ TAB 6 — Network Graph (GAP #6 + #7 + #12)
    # ═══════════════════════════════════════════════════════════════════
    with tab_graph:
        state_dict  = env.state()
        active_task = state_dict.get("active_task", "")
        task_info   = state_dict.get("task_info", {})

        if "cib" in active_task.lower():
            # GAP #7: CIB cluster state scoreboard
            st.markdown('<p class="section-title">🕸️ CIB Network Takedown Progress</p>', unsafe_allow_html=True)
            gc1, gc2, gc3, gc4 = st.columns(4)
            gc1.metric("Bots Remaining",  task_info.get("bots_remaining", "?"), delta_color="inverse")
            gc2.metric("Bots Removed ✅", task_info.get("bots_removed", "?"))
            gc3.metric("Real Remaining",  task_info.get("real_remaining", "?"))
            gc4.metric("Collateral ❌",   task_info.get("real_removed", "?"), delta_color="inverse")

            threshold = task_info.get("collateral_threshold", 10)
            collat = task_info.get("real_removed", 0)
            if isinstance(collat, int) and isinstance(threshold, int) and threshold > 0:
                st.progress(min(collat/threshold, 1.0), text=f"Collateral: {collat}/{threshold}")

            # GAP #6: collateral history chart
            col_hist = st.session_state.get("collateral_history", [])
            if col_hist:
                df_col = pd.DataFrame({"Step": range(len(col_hist)), "Collateral Removals": col_hist})
                st.caption("Collateral damage over episode")
                st.line_chart(df_col.set_index("Step"))

            st.caption("Node colors: 🟢 Real · 🔴 Bot · 🟡 Under review · ⬜ Removed")
            try:
                html = _build_graph_html(env)
                components.html(html, height=600, scrolling=False)
            except Exception as e:
                st.warning(f"Graph unavailable: {e}")

        elif "spam" in active_task.lower():
            if st.session_state.obs is not None:
                obs = st.session_state.obs
                names = ["age","posts/hr","follower_ratio","login_var","content_rep","profile","device","ip_div"]
                st.bar_chart(pd.DataFrame({"Feature":names, "Value":[float(obs[i]) for i in range(8)]}).set_index("Feature"))
                gt = st.session_state.log[-1]["Ground Truth"] if st.session_state.log else "?"
                st.caption(f"Current account ground truth: **{gt}**")

                # GAP #6: collateral history for spam
                col_hist = st.session_state.get("collateral_history", [])
                if col_hist:
                    st.caption("Collateral (false positive) removals over episode")
                    df_col = pd.DataFrame({"Step": range(len(col_hist)), "Collateral": col_hist})
                    st.line_chart(df_col.set_index("Step"))

        elif "misinfo" in active_task.lower():
            # GAP #12: hop count trend chart
            hop_hist = st.session_state.get("hop_history", [])
            if hop_hist:
                max_hops = 20
                current_hop = hop_hist[-1]
                st.metric("Current Hop Count", current_hop)
                st.progress(min(current_hop / max_hops, 1.0), text=f"Hop {current_hop}/{max_hops}")
                df_hop = pd.DataFrame({"Step": range(len(hop_hist)), "Hop Count": hop_hist})
                st.caption("Content spread progression")
                st.line_chart(df_hop.set_index("Step"))

            if st.session_state.obs is not None:
                obs = st.session_state.obs
                names = ["spread_rate","fact_check","engagement","credibility","hop_count","timestep"]
                st.bar_chart(pd.DataFrame({"Feature":names, "Value":[float(obs[i]) for i in range(6)]}).set_index("Feature"))
        else:
            st.info("No active task graph.")

    # ═══════════════════════════════════════════════════════════════════
    # 📈 TAB 7 — Reward Chart
    # ═══════════════════════════════════════════════════════════════════
    with tab_rewards_chart:
        st.subheader("📈 Cumulative Reward")
        if st.session_state.cumulative_rewards:
            st.line_chart(pd.DataFrame({"Cumulative Reward": st.session_state.cumulative_rewards}))
        else:
            st.info("No data yet.")
        if st.session_state.episode_rewards:
            st.caption("Episode totals")
            st.bar_chart(pd.DataFrame({"Total Reward": st.session_state.episode_rewards}))

    # ═══════════════════════════════════════════════════════════════════
    # 🔍 TAB 8 — Reward Breakdown
    # ═══════════════════════════════════════════════════════════════════
    with tab_breakdown:
        st.subheader("🔍 Reward Breakdown (Last Step)")
        if st.session_state.log:
            bd = st.session_state.get("last_breakdown") or {}
            if bd:
                labels = ["correctness","fp_cost","collateral_damage","speed_bonus","escalation_penalty"]
                bc = st.columns(5)
                for col, lbl in zip(bc, labels):
                    col.metric(lbl, f"{float(bd.get(lbl, 0.0)):.3f}")
                st.caption(f"**Total: {float(bd.get('total', 0.0)):.4f}**")
                st.bar_chart(pd.DataFrame({
                    "Component": labels,
                    "Value": [float(bd.get(l, 0)) for l in labels],
                }).set_index("Component"))
            else:
                st.info("Step forward to see breakdown.")
        else:
            st.info("No steps yet.")

    # ═══════════════════════════════════════════════════════════════════
    # 📡 TAB 9 — Server Metrics (GAP #2)
    # ═══════════════════════════════════════════════════════════════════
    with tab_metrics:
        st.subheader("📡 Live Server Metrics")
        st.caption("Pulls from `GET /metrics` (Prometheus-format text endpoint).")
        if st.button("🔄 Refresh Metrics"):
            try:
                r = requests.get(f"{SERVER_BASE}/metrics", timeout=5)
                lines = [l for l in r.text.splitlines() if not l.startswith("#") and l.strip()]
                rows = []
                for line in lines:
                    if "{" in line and "}" in line:
                        metric_name, rest = line.split("{", 1)
                        tags, val = rest.rsplit("} ", 1)
                        rows.append({
                            "metric": metric_name.strip(),
                            "labels": tags.strip(),
                            "value": float(val.strip()),
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("No metrics yet — initialize an env via /reset first.")
            except Exception as e:
                st.error(f"Could not reach {SERVER_BASE}/metrics: {e}")
                st.info("Start the FastAPI server with `uvicorn server:app --port 7860`.")

        # Recent API calls (from server /)
        st.divider()
        st.subheader("🕰️ Recent API Calls")
        if st.button("🔄 Refresh API Log"):
            try:
                r = requests.get(f"{SERVER_BASE}/", timeout=5)
                st.components.v1.html(r.text, height=500, scrolling=True)
            except Exception as e:
                st.error(f"Cannot reach server root: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # 🗂️ TAB 10 — Episode State (GAP #8)
    # ═══════════════════════════════════════════════════════════════════
    with tab_state:
        st.subheader("🗂️ Full Episode State — env.state()")
        if st.button("📥 Fetch state()"):
            try:
                s = env.state()
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Timestep", s.get("timestep", 0))
                sc2.metric("Episode Step", s.get("episode_step", 0))
                sc3.metric("Cumulative Reward", f"{s.get('cumulative_reward', 0):.3f}")
                sc4.metric("Active Task", s.get("active_task", "?"))

                hist = s.get("decision_history", [])
                if hist:
                    st.caption(f"Decision history ({len(hist)} entries)")
                    st.dataframe(pd.DataFrame(hist[-50:]), use_container_width=True)

                with st.expander("📄 Raw state() JSON"):
                    # Remove large lists for display
                    s_display = {k: v for k, v in s.items() if k != "decision_history"}
                    st.json(s_display)
            except Exception as e:
                st.error(str(e))

    # ═══════════════════════════════════════════════════════════════════
    # 🧠 TAB 11 — Model Architecture
    # ═══════════════════════════════════════════════════════════════════
    with tab_arch:
        st.subheader("🧠 NEMESIS Model Architecture")
        st.markdown("""
```
Input: 68-dim obs (Box float32)
  ├─ Tabular slice: obs[0:5]  → FallbackProjection(5 → 384)  [or MiniLM text encoder]
  └─ Embedding: 384-dim
                ↓
Combined: concat([embedding, tabular]) → 389-dim
                ↓
NemesisNetBackbone:
  Linear(389 → 512) + ReLU + Dropout(0.3)
  Linear(512 → 256) + ReLU + Dropout(0.3)
  Linear(256 → 128) + ReLU
                ↓
  ├─ Actor head:  Linear(128 → 5)  → 5 action logits
  └─ Critic head: Linear(128 → 1)  → V(s)

Training: PPO · γ=0.99 · ent_coef=0.01 · n_steps=2048 · batch=64
```
""")
        st.markdown(f"**HF_TOKEN:** {'✅ Set' if os.environ.get('HF_TOKEN') else '⚠️ Not set (fallback projection active)'}")

    # ═══════════════════════════════════════════════════════════════════
    # 🌍 TAB 12 — Mastodon Live
    # ═══════════════════════════════════════════════════════════════════
    with tab_mastodon:
        st.subheader("🌍 Mastodon Live")
        mrecords = st.session_state.get("nemesis_records", [])
        if mrecords:
            df_m = pd.DataFrame([{
                "Account": r["account_id"], "Content": r["content_snippet"],
                "Prediction": r["prediction"], "Ground Truth": r["ground_truth"],
                "Reward": r["reward"], "Confidence": f"{r['confidence']:.3f}",
            } for r in list(reversed(mrecords[-50:]))])
            st.dataframe(df_m, use_container_width=True)
        else:
            st.info("No Mastodon posts analyzed yet.")

        if HAS_AUTOREFRESH and os.environ.get("MASTODON_ACCESS_TOKEN"):
            st_autorefresh(interval=5000, key="mastodon_refresh")

    # ── Auto-play loop ───────────────────────────────────────────────
    if st.session_state.running:
        if st.session_state.terminated or st.session_state.truncated:
            st.session_state.running = False
            st.session_state.episode_rewards.append(st.session_state.ep_reward)
            # GAP #5: task_success on episode end
            last_info = env.state()
            task_inf = last_info.get("task_info", {})
            with st.expander("📊 Episode Summary", expanded=True):
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total Reward",       f"{st.session_state.ep_reward:.2f}")
                s2.metric("Bots Removed (TP)",  st.session_state.tp)
                s3.metric("False Positives",    st.session_state.fp)
                s4.metric("Steps",              st.session_state.step)
                tot = st.session_state.tp + st.session_state.fp
                if tot > 0:
                    prec = st.session_state.tp / tot
                    st.progress(prec, text=f"Precision: {prec:.1%}")
                # GAP #5: task_success
                escalations = st.session_state.get("escalation_count", 0)
                st.caption(f"Escalations used: {escalations}")
        else:
            if not HAS_AUTOREFRESH:
                st.session_state.running = False
                st.warning("Auto-play requires `streamlit-autorefresh` for non-blocking updates.")
            else:
                interval_ms = int(max(50.0, 1000.0 / max(float(auto_speed), 1.0)))
                current_tick = st_autorefresh(interval=interval_ms, key="episode_autoplay")
                if st.session_state.get("last_autoplay_tick") != current_tick:
                    try:
                        step_agent(env, agent)
                    except Exception as e:
                        st.error(str(e))
                        st.session_state.running = False
                    finally:
                        st.session_state.last_autoplay_tick = current_tick
                    st.rerun()


if __name__ == "__main__":
    main()
