"""
dashboard/app.py — NEMESIS-RL Streamlit Dashboard (v2)

Full-featured dashboard showing:
  - NEMESIS model architecture overview
  - Live flagged accounts stream with account ID, reason, confidence
  - NEMESIS inference tab: per-step decisions with color-coded TP/FP table
  - Action probability bars per inference step
  - Learning curve charts from training_log.csv
  - Training controls (launch training cycle from UI)
  - Reward breakdown panel
  - Network graph (CIB task)
  - Llama-4-Maverick analysis panel

Usage::
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import hmac
import os
import time
import atexit
import threading
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Dict

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

from env.env import SocialGuardEnv
from baseline import BaselineAgent
from env.spaces import ACTION_NAMES, ACTION_REMOVE
from dashboard.graph_view import generate_graph_base_html, apply_decision_log
from dashboard.metrics_view import render_metrics_cards, render_decision_log, render_reward_chart

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    st_autorefresh = None  # type: ignore[assignment]
    HAS_AUTOREFRESH = False

# ---------------------------------------------------------------------------
# Page config + custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    layout="wide",
    page_title="NEMESIS-RL",
    page_icon="🛡️",
)

st.markdown("""
<style>
/* Dark background on sidebar */
[data-testid="stSidebar"] { background: #0f1117; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #1a1d27;
    border: 1px solid #2d3044;
    border-radius: 10px;
    padding: 12px 16px;
}

/* Tab styling */
[data-testid="stTabs"] button { font-size: 13px; font-weight: 600; }

/* Status pills */
.pill-green  { background:#155724; color:#d4edda; padding:3px 10px; border-radius:20px; font-size:12px; }
.pill-red    { background:#721c24; color:#f8d7da; padding:3px 10px; border-radius:20px; font-size:12px; }
.pill-yellow { background:#856404; color:#fff3cd; padding:3px 10px; border-radius:20px; font-size:12px; }
.pill-gray   { background:#383d47; color:#ccc;     padding:3px 10px; border-radius:20px; font-size:12px; }

/* Confidence bar */
.conf-bar-bg { background:#2d3044; border-radius:6px; height:10px; margin-top:4px; }
.conf-bar-fill { background:linear-gradient(90deg,#00b4d8,#0077b6); border-radius:6px; height:10px; }

/* Section heading */
.section-title { font-size:16px; font-weight:700; color:#90caf9; margin:12px 0 6px 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NEMESIS_MODEL_PATH = "models/nemesis/final_model.zip"
TRAINING_LOG_PATH  = "training_log.csv"
CHECKPOINT_DIR     = "models/nemesis/checkpoints"

ACTION_EMOJI = {0: "✅ Allow", 1: "⚠️ Warn", 2: "🔒 Restrict", 3: "❌ Remove", 4: "🚨 Escalate"}
REASON_EMOJI = {
    "Spam/Phishing Activity":          "📧",
    "Misinformation Spread":           "📰",
    "Coordinated Inauthentic Behavior":"🤖",
    "False Positive (Organic User)":   "👤",
    "Normal Activity":                 "✓",
    "N/A":                             "—",
}

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
        "last_autoplay_tick": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_episode_state() -> None:
    for k in (
        "step", "terminated", "truncated", "ep_reward", "cumulative_rewards",
        "episode_rewards", "log", "decision_log", "tp", "fp",
        "last_breakdown", "running", "flagged_stream", "last_autoplay_tick",
    ):
        st.session_state[k] = 0 if k in ("step","tp","fp") else (
            0.0 if k in ("ep_reward",) else ([] if k in ("cumulative_rewards","log") else (
            {} if k in ("decision_log","last_breakdown") else (
            [] if k in ("episode_rewards", "flagged_stream") else None if k == "last_autoplay_tick" else False)))
        )
    st.session_state.pop("graph_base_html", None)
    st.session_state.pop("graph_base_sig",  None)


def _format_delta(series: pd.Series) -> str:
    if series.empty:
        return "+0.000"
    delta_val = series.diff().iloc[-1]
    return f"{float(delta_val):+.3f}" if pd.notna(delta_val) else "+0.000"

# ---------------------------------------------------------------------------
# Step logic
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

    gt       = info.get("ground_truth", -1)
    act_str  = ACTION_NAMES.get(action, str(action))
    fa       = info.get("flagged_account", "N/A")
    fr       = info.get("flagged_reason",  "N/A")

    if action == ACTION_REMOVE:
        if gt == 1: st.session_state.tp += 1
        elif gt == 0: st.session_state.fp += 1

    row = {
        "Timestep": st.session_state.step, "Action": act_str,
        "Account": fa, "Reason": fr,
        "Ground Truth": "Bot" if gt == 1 else "Real" if gt == 0 else "N/A",
        "Reward": round(float(reward), 4),
        "Cumulative": round(st.session_state.ep_reward, 4),
    }
    st.session_state.log.append(row)
    st.session_state.last_breakdown = info.get("reward_breakdown", {})

    # Flagged stream (only non-allow actions)
    if action != 0:
        st.session_state.flagged_stream.append({
            "step": st.session_state.step,
            "account": fa,
            "action": act_str,
            "reason": fr,
            "gt": "Bot" if gt == 1 else "Real" if gt == 0 else "?",
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
# Helpers: model info
# ---------------------------------------------------------------------------

def _model_info_html() -> str:
    return """
<div style="font-family:monospace;font-size:12px;background:#0f1117;color:#90caf9;
     padding:16px;border-radius:10px;border:1px solid #2d3044;line-height:1.8">
<b style="color:#fff;font-size:14px">🧠 NemesisPolicy Architecture</b><br><br>
<b style="color:#f9a825">Encoder: </b> Llama-4-Maverick-17B-128E-Instruct (HF API)<br>
<b style="color:#f9a825">Fallback: </b> all-MiniLM-L6-v2 → FallbackProjection (5→384)<br>
<b style="color:#f9a825">Input:   </b> 5 tabular dims + 384 embedding <b>= 389 dims</b><br><br>
<b style="color:#80cbc4">FC Layers:</b><br>
&nbsp;&nbsp;Linear(389 → 512) + ReLU + Dropout(0.3)<br>
&nbsp;&nbsp;Linear(512 → 256) + ReLU + Dropout(0.3)<br>
&nbsp;&nbsp;Linear(256 → 128) + ReLU<br><br>
<b style="color:#80cbc4">Output:</b><br>
&nbsp;&nbsp;Actor head  → 5 logits (No Action / Warn / Restrict / Suspend / Ban)<br>
&nbsp;&nbsp;Critic head → V(s) scalar<br><br>
<b style="color:#ce93d8">Training: PPO · γ=0.99 · ent_coef=0.01 · 100k steps/cycle</b><br>
<b style="color:#ce93d8">Reward:   TP=+1.0 · FP=-0.5 · Miss=-1.0 · TN=0.0</b>
</div>"""

# ---------------------------------------------------------------------------
# Confidence bar HTML
# ---------------------------------------------------------------------------

def _conf_bars_html(probs: Dict[str, float]) -> str:
    bars = ""
    colors = ["#00b4d8","#f9a825","#ef5350","#ce93d8","#ff7043"]
    for i, (label, p) in enumerate(probs.items()):
        pct = max(0.0, min(1.0, p)) * 100
        c = colors[i % len(colors)]
        bars += (
            f'<div style="margin:4px 0">'
            f'<span style="color:#ccc;font-size:11px;width:80px;display:inline-block">{label}</span>'
            f'<div style="display:inline-block;vertical-align:middle;width:60%;'
            f'background:#2d3044;border-radius:4px;height:8px;margin:0 8px">'
            f'<div style="width:{pct:.1f}%;background:{c};border-radius:4px;height:8px"></div></div>'
            f'<span style="color:{c};font-size:11px">{p:.3f}</span>'
            f'</div>'
        )
    return f'<div style="font-family:monospace;padding:8px 0">{bars}</div>'

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    init_session_state()
    _check_token()

    # ── Sidebar ──────────────────────────────────────────────────────────
    st.sidebar.markdown("## ⚙️ Configuration")

    config_file = st.sidebar.selectbox(
        "Task Config",
        ["configs/task1.yaml", "configs/task2.yaml", "configs/task3.yaml"],
    )
    agent_type = st.sidebar.selectbox("Baseline Agent", ["Rule-based Baseline", "Trained Model"])
    model_file = ""
    if agent_type == "Trained Model":
        model_file = st.sidebar.text_input("Model path (.zip)", value="models/ppo_task1/final_model.zip")

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

    # NEMESIS model status in sidebar
    st.sidebar.divider()
    st.sidebar.markdown("### 🤖 NEMESIS Status")
    if os.path.exists(NEMESIS_MODEL_PATH):
        mtime = Path(NEMESIS_MODEL_PATH).stat().st_mtime
        t_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
        st.sidebar.success(f"✅ Model loaded\n`{t_str}`")
        ckpts = len(list(Path(CHECKPOINT_DIR).glob("*.zip"))) if Path(CHECKPOINT_DIR).exists() else 0
        st.sidebar.caption(f"Checkpoints: **{ckpts}**")
    else:
        st.sidebar.warning("⚠️ No trained model found.\nRun training below.")

    if os.path.exists(TRAINING_LOG_PATH):
        try:
            df_s = pd.read_csv(TRAINING_LOG_PATH)
            if not df_s.empty:
                last = df_s.iloc[-1]
                st.sidebar.metric("Last TP Rate",  f"{float(last.get('tp_rate',0)):.3f}")
                st.sidebar.metric("Last FP Rate",  f"{float(last.get('fp_rate',0)):.3f}")
                st.sidebar.metric("Last Reward",   f"{float(last.get('mean_reward',0)):.3f}")
                st.sidebar.metric("Cycles Done",   int(df_s["cycle"].max()))
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
            agent = BaselineAgent(task_name=str(env._task_cfg.get("name", "task_spam")))
        else:
            if not model_file:
                st.error("Please enter a model path.")
                return
            from evaluate import load_model
            agent = load_model(model_file)
    except Exception as e:
        if agent_type == "Trained Model":
            st.error(
                "❌ Failed to load trained model. Use a `.zip` under `models/` and ensure runtime dependencies are installed."
            )
        else:
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

    task_label = config_file.replace("configs/", "").replace(".yaml", "").upper()
    hcol1, hcol2, hcol3 = st.columns([3,1,1])
    hcol1.caption(f"Task: **{task_label}** · Agent: **{agent_type}**")
    if st.session_state.terminated or st.session_state.truncated:
        hcol2.success("✅ Episode Done")
    else:
        hcol2.info("▶️ Running")

    # ── Metrics row ──────────────────────────────────────────────────────
    render_metrics_cards(
        st.session_state.ep_reward, st.session_state.tp,
        st.session_state.fp,        st.session_state.step,
    )

    # Extra metrics: precision, flagged count
    ex1, ex2, ex3, ex4 = st.columns(4)
    total_actions   = st.session_state.tp + st.session_state.fp
    precision       = st.session_state.tp / total_actions if total_actions else 0.0
    flagged_total   = len(st.session_state.flagged_stream)
    ex1.metric("Precision",     f"{precision:.1%}")
    ex2.metric("Actions Taken", total_actions)
    ex3.metric("Flagged Events",flagged_total)
    ex4.metric("Avg Step Reward",
               f"{st.session_state.ep_reward / max(st.session_state.step,1):.3f}")

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
            st.session_state.running = True
            st.session_state.last_autoplay_tick = None
            st.rerun()
    with c3:
        if st.button("⏸️ Pause", use_container_width=True, disabled=not st.session_state.running):
            st.session_state.running = False
            st.session_state.last_autoplay_tick = None
            st.rerun()

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🚨 Flagged Accounts",
        "🤖 NEMESIS Live",
        "📉 Learning Curve",
        "🏋️ Train Model",
        "📋 Decision Log",
        "🕸️ Network Graph",
        "📈 Cumulative Reward",
        "🔍 Reward Breakdown",
        "🧠 Model Architecture",
        "🌍 Mastodon Live",
        "⚖️ Mastodon Statistics",
    ])
    (tab_flagged, tab_nemesis, tab_learning, tab_train,
     tab_log, tab_graph, tab_rewards, tab_breakdown, tab_arch,
     tab_mastodon, tab_mastodon_stats) = tabs

    # ═══════════════════════════════════════════════════════════════════
    # 🚨 TAB 1 — Flagged Accounts
    # ═══════════════════════════════════════════════════════════════════
    with tab_flagged:
        st.subheader("🚨 Live Flagged Accounts Stream")
        st.caption("Every non-allow action taken by the agent. In Task 2, `Restrict` reduces reach and the episode continues; `Remove` is the terminal takedown action.")

        stream = list(reversed(st.session_state.flagged_stream[-50:]))
        if stream:
            for row in stream:
                emoji = REASON_EMOJI.get(row["reason"], "❓")
                is_bot = row["gt"] == "Bot"
                pill_cls = "pill-red" if is_bot else "pill-yellow"
                pill_lbl = "🤖 Bot" if is_bot else "👤 Human"
                rew_color = "#4caf50" if row["reward"] > 0 else "#f44336"
                st.markdown(f"""
<div style="background:#1a1d27;border:1px solid #2d3044;border-radius:10px;
     padding:12px 16px;margin-bottom:8px;display:flex;align-items:center;gap:16px">
  <span style="font-size:22px">{emoji}</span>
  <div style="flex:1">
    <b style="color:#90caf9">Account #{row['account']}</b>
    <span style="color:#666;font-size:12px;margin-left:8px">Step {row['step']}</span><br>
    <span style="color:#ccc;font-size:13px">{row['reason']}</span>
  </div>
  <div style="text-align:right">
    <span class="{pill_cls}">{pill_lbl}</span><br>
    <span style="color:#aaa;font-size:12px">{ACTION_EMOJI.get(0,'')}&nbsp;→&nbsp;
      <b style="color:#fff">{row['action']}</b></span><br>
    <span style="color:{rew_color};font-weight:700">reward {row['reward']:+.2f}</span>
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.info("▶️ Step forward or Play to start flagging accounts. All non-allow actions appear here.")

        if stream:
            df_stream = pd.DataFrame(list(reversed(st.session_state.flagged_stream)))
            with st.expander("📄 Full flagged log table"):
                st.dataframe(df_stream, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════
    # 🤖 TAB 2 — NEMESIS Live Inference
    # ═══════════════════════════════════════════════════════════════════
    with tab_nemesis:
        st.subheader("🤖 NEMESIS Neural Policy — Live Inference")
        st.markdown("""
> Runs the **trained NEMESIS PPO policy** (NemesisPolicy backbone, 389-dim input)
> on a fresh episode. The model calls **Llama-4-Maverick-17B-128E-Instruct** via
> HuggingFace Inference API for semantic content encoding when `HF_TOKEN` is set.
""")
        ncol1, ncol2, ncol3 = st.columns([2, 1, 1])
        with ncol1:
            n_steps = st.slider("Steps to run", 5, 100, 20, key="n_inf_steps")
        with ncol2:
            deterministic = st.toggle("Deterministic", True, key="inf_det")
        with ncol3:
            inf_config = st.selectbox("Config", ["configs/task1.yaml","configs/task2.yaml","configs/task3.yaml"], key="inf_cfg")

        run_col, llama_col = st.columns(2)
        run_btn   = run_col.button("▶️ Run NEMESIS Inference", use_container_width=True, key="run_nem")
        llama_btn = llama_col.button("🦙 Analyze with Llama-4-Maverick", use_container_width=True, key="run_llama")

        if run_btn:
            if not os.path.exists(NEMESIS_MODEL_PATH):
                st.warning(f"No model at `{NEMESIS_MODEL_PATH}`. Train first (🏋️ tab).")
            else:
                with st.spinner("Running NEMESIS inference…"):
                    try:
                        from agent import run_inference_episode
                        recs_list = st.session_state.get("nemesis_records", [])
                        recs_list.clear()
                        recs = run_inference_episode(
                            config_path=inf_config,
                            model_path=NEMESIS_MODEL_PATH,
                            n_steps=n_steps,
                            deterministic=deterministic,
                            records_list=recs_list
                        )
                        st.session_state["nemesis_records"] = recs
                    except Exception as exc:
                        st.error(f"Inference error: {exc}")

        if llama_btn:
            obs_now = st.session_state.obs
            if obs_now is not None:
                with st.spinner("Calling Llama-4-Maverick via HuggingFace API…"):
                    try:
                        from model import analyze_content_with_llama
                        snippet = (
                            f"age={float(obs_now[0]):.2f} posts_hr={float(obs_now[1]):.2f} "
                            f"follower_ratio={float(obs_now[2]):.2f} "
                            f"content_rep={float(obs_now[4]):.2f}"
                        )
                        result = analyze_content_with_llama(snippet, {"task": config_file})
                        st.session_state["llama_result"] = result
                    except Exception as exc:
                        st.error(f"Llama error: {exc}")
            else:
                st.warning("Reset episode first to have an observation to analyze.")

        # Llama result panel
        llama_res = st.session_state.get("llama_result")
        if llama_res:
            st.markdown('<p class="section-title">🦙 Llama-4-Maverick Analysis</p>', unsafe_allow_html=True)
            lc1, lc2 = st.columns([1, 3])
            risk = float(llama_res.get("risk_score", 0.5))
            risk_color = "#ef5350" if risk > 0.6 else ("#f9a825" if risk > 0.3 else "#4caf50")
            lc1.markdown(f"""
<div style="background:#1a1d27;border-radius:10px;padding:20px;text-align:center">
  <div style="font-size:40px;font-weight:800;color:{risk_color}">{risk:.2f}</div>
  <div style="color:#aaa;font-size:12px">Risk Score</div>
</div>""", unsafe_allow_html=True)
            cats  = ", ".join(llama_res.get("categories", [])) or "None"
            reason= llama_res.get("reasoning", "")[:300]
            lc2.markdown(f"**Categories:** {cats}")
            lc2.markdown(f"**Reasoning:** {reason}")

        # Inference records
        records = st.session_state.get("nemesis_records", [])
        if records:
            m_tp  = sum(1 for r in records if r["prediction"] != "No Action" and r["ground_truth"] == "Bot")
            m_fp  = sum(1 for r in records if r["prediction"] != "No Action" and r["ground_truth"] == "Human")
            m_rew = sum(r["reward"] for r in records)
            avg_conf = np.mean([r["confidence"] for r in records])

            st.success(f"NEMESIS ran {len(records)} steps — TP={m_tp}  FP={m_fp}  Total Reward={m_rew:.2f}  Avg Confidence={avg_conf:.3f}")

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Steps",          len(records))
            mc2.metric("True Positives", m_tp)
            mc3.metric("False Positives",m_fp)
            mc4.metric("Total Reward",   f"{m_rew:.2f}")
            mc5.metric("Avg Confidence", f"{avg_conf:.3f}")

            # Color-coded table
            df_r = pd.DataFrame([{
                "Step":         r["step"],
                "Account":      r["account_id"],
                "Content":      r["content_snippet"][:55] + "…",
                "Prediction":   r["prediction"],
                "Ground Truth": r["ground_truth"],
                "Reward":       r["reward"],
                "Conf":         f"{r['confidence']:.3f}",
                "Flagged Reason": r["flagged_reason"],
            } for r in records])

            def _row_color(row):
                is_act = row["Prediction"] != "No Action"
                is_bot = row["Ground Truth"] == "Bot"
                if is_act and is_bot:   return ["background-color:#1b4332"] * len(row)
                if is_act and not is_bot: return ["background-color:#4a1122"] * len(row)
                return [""] * len(row)

            st.dataframe(df_r.style.apply(_row_color, axis=1), use_container_width=True)

            # Per-step probability bars
            st.markdown('<p class="section-title">📊 Action Probability per Step</p>', unsafe_allow_html=True)
            selected_step = st.selectbox(
                "Select step to inspect probability distribution",
                [r["step"] for r in records], key="prob_step"
            )
            rec = next((r for r in records if r["step"] == selected_step), None)
            if rec and "probs" in rec:
                st.markdown(_conf_bars_html(rec["probs"]), unsafe_allow_html=True)
                st.markdown(f"""
<div style="background:#1a1d27;padding:12px;border-radius:8px;font-size:13px;color:#ccc;margin-top:8px">
  🏷️ <b>Account:</b> {rec['account_id']} &nbsp;|&nbsp;
  🎯 <b>Prediction:</b> {rec['prediction']} &nbsp;|&nbsp;
  🔖 <b>Ground Truth:</b> {rec['ground_truth']} &nbsp;|&nbsp;
  💰 <b>Reward:</b> <span style="color:{'#4caf50' if rec['reward']>=0 else '#f44336'}">{rec['reward']:+.3f}</span> &nbsp;|&nbsp;
  🤔 <b>Confidence:</b> {rec['confidence']:.3f}<br>
  🚩 <b>Flagged Reason:</b> {rec['flagged_reason']}
</div>""", unsafe_allow_html=True)

            # Confidence line chart
            cdf = pd.DataFrame({"Step": [r["step"] for r in records],
                                  "Confidence": [r["confidence"] for r in records]}).set_index("Step")
            st.caption("Model confidence over episode")
            st.line_chart(cdf)

        elif not run_btn:
            st.info("↑ Click **Run NEMESIS Inference** to see the model in action.")

    # ═══════════════════════════════════════════════════════════════════
    # 📉 TAB 3 — Learning Curve
    # ═══════════════════════════════════════════════════════════════════
    with tab_learning:
        st.subheader("📉 NEMESIS Learning Curve")
        st.caption("From `training_log.csv`. TP rate ↑ and FP rate ↓ prove the model is learning.")

        lrefresh = st.button("🔄 Refresh", key="lc_refresh")
        if os.path.exists(TRAINING_LOG_PATH):
            try:
                df_log = pd.read_csv(TRAINING_LOG_PATH)
                if df_log.empty:
                    st.info("training_log.csv is empty — run a training cycle.")
                else:
                    latest = df_log.iloc[-1]
                    lc1, lc2, lc3, lc4 = st.columns(4)
                    lc1.metric("TP Rate",    f"{float(latest.get('tp_rate',0)):.3f}",
                               delta=_format_delta(df_log["tp_rate"]))
                    lc2.metric("FP Rate",    f"{float(latest.get('fp_rate',0)):.3f}",
                               delta=_format_delta(df_log["fp_rate"]), delta_color="inverse")
                    lc3.metric("Mean Reward",f"{float(latest.get('mean_reward',0)):.3f}",
                               delta=_format_delta(df_log["mean_reward"]))
                    lc4.metric("Entropy",    f"{float(latest.get('policy_entropy',0)):.3f}",
                               delta=_format_delta(df_log["policy_entropy"]), delta_color="inverse")

                    st.markdown('<p class="section-title">TP Rate vs FP Rate</p>', unsafe_allow_html=True)
                    st.line_chart(df_log[["episode","tp_rate","fp_rate"]].set_index("episode"))

                    st.markdown('<p class="section-title">Mean Episode Reward</p>', unsafe_allow_html=True)
                    st.line_chart(df_log[["episode","mean_reward"]].set_index("episode"))

                    if "policy_entropy" in df_log.columns:
                        st.markdown('<p class="section-title">Policy Entropy (↓ = more decisive)</p>', unsafe_allow_html=True)
                        st.line_chart(df_log[["episode","policy_entropy"]].set_index("episode"))

                    with st.expander("📄 Raw training_log.csv"):
                        st.dataframe(df_log, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not read training_log.csv: {exc}")
        else:
            st.info("No `training_log.csv` yet. Train the model in the 🏋️ tab first.")

    # ═══════════════════════════════════════════════════════════════════
    # 🏋️ TAB 4 — Train Model
    # ═══════════════════════════════════════════════════════════════════
    with tab_train:
        st.subheader("🏋️ Train NEMESIS Model")
        st.markdown("""
Train the **NemesisPolicy PPO** directly from the dashboard.
- Runs `python3 agent.py` in a subprocess
- Checkpoints saved every 10,000 steps
- Resumes from last checkpoint automatically
- `training_log.csv` updated after each cycle
""")
        tcol1, tcol2, tcol3 = st.columns(3)
        with tcol1:
            t_cycles = st.number_input("Training cycles (×100k steps)", 1, 20, 1, key="t_cycles")
        with tcol2:
            t_config  = st.selectbox("Config", ["configs/task1.yaml","configs/task2.yaml","configs/task3.yaml"], key="t_cfg")
        with tcol3:
            t_device  = st.selectbox("Device", ["auto","cpu","cuda","mps"], key="t_dev")

        train_col1, train_col2 = st.columns(2)
        start_train = train_col1.button("🚀 Start Training", use_container_width=True, key="start_train")
        clear_ckpts = train_col2.button("🗑️ Clear Checkpoints (fresh start)", use_container_width=True, key="clear_ckpt")

        if clear_ckpts:
            import glob, shutil
            removed = 0
            for f in glob.glob(os.path.join(CHECKPOINT_DIR, "*.zip")):
                os.remove(f); removed += 1
            if os.path.exists(NEMESIS_MODEL_PATH):
                os.remove(NEMESIS_MODEL_PATH); removed += 1
            st.success(f"Removed {removed} file(s). Next training will start fresh.")

        if start_train:
            cmd = [
                sys.executable, "agent.py",
                "--config", t_config,
                "--cycles", str(int(t_cycles)),
                "--device", t_device,
            ]
            st.info(f"Launching: `{' '.join(cmd)}`")
            st.session_state["train_proc_cmd"] = " ".join(cmd)
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=os.getcwd(),
                )
                st.session_state["train_proc"] = proc
                st.session_state["train_log_lines"] = []
                st.session_state["train_running"] = True
                st.success("✅ Training launched. Refresh this tab to see live output.")
            except Exception as exc:
                st.error(f"Failed to launch training: {exc}")

        # Show live output if training is running
        proc = st.session_state.get("train_proc")
        if proc is not None:
            if proc.poll() is None:
                st.info("🔄 Training in progress…")
                # Drain available lines non-blocking
                import select
                try:
                    rlist, _, _ = select.select([proc.stdout], [], [], 0.1)
                    if rlist:
                        for _ in range(50):
                            line = proc.stdout.readline()
                            if not line: break
                            st.session_state["train_log_lines"].append(line.rstrip())
                except Exception:
                    pass
            else:
                st.session_state["train_running"] = False
                rc = proc.returncode
                if rc == 0:
                    st.success("✅ Training complete!")
                else:
                    st.error(f"Training exited with code {rc}")
                st.session_state["train_proc"] = None

            log_lines = st.session_state.get("train_log_lines", [])
            if log_lines:
                st.markdown('<p class="section-title">Training Output</p>', unsafe_allow_html=True)
                st.code("\n".join(log_lines[-60:]), language="text")

        # Checkpoint browser
        st.markdown('<p class="section-title">Checkpoints</p>', unsafe_allow_html=True)
        if Path(CHECKPOINT_DIR).exists():
            import glob
            ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.zip")))
            if ckpts:
                ckpt_data = []
                for c in ckpts:
                    p = Path(c)
                    ckpt_data.append({
                        "File": p.name,
                        "Size (KB)": round(p.stat().st_size/1024, 1),
                        "Modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
                    })
                st.dataframe(pd.DataFrame(ckpt_data), use_container_width=True)
            else:
                st.caption("No checkpoints yet.")
        else:
            st.caption("Checkpoint directory does not exist yet.")

    # ═══════════════════════════════════════════════════════════════════
    # 📋 TAB 5 — Decision Log
    # ═══════════════════════════════════════════════════════════════════
    with tab_log:
        st.subheader("📋 Decision Log")
        render_decision_log(list(reversed(st.session_state.log[-100:])))

    # ═══════════════════════════════════════════════════════════════════
    # 🕸️ TAB 6 — Network Graph
    # ═══════════════════════════════════════════════════════════════════
    with tab_graph:
        state_dict  = env.state()
        active_task = state_dict.get("active_task", "")

        if "cib" in active_task.lower():
            st.caption("Node colors: 🟢 Real · 🔴 Bot · 🟡 Under review · ⬜ Removed")
            try:
                html = _build_graph_html(env)
                components.html(html, height=640, scrolling=False)
            except Exception as e:
                st.warning(f"Graph unavailable: {e}")
        elif "spam" in active_task.lower():
            if st.session_state.obs is not None:
                obs = st.session_state.obs
                names = ["age","posts/hr","follower_ratio","login_var","content_rep","profile","device","ip_div"]
                st.bar_chart(pd.DataFrame({"Feature":names,"Value":[float(obs[i]) for i in range(8)]}).set_index("Feature"))
                gt = st.session_state.log[-1]["Ground Truth"] if st.session_state.log else "?"
                st.caption(f"Ground truth: **{gt}**")
            else:
                st.info("Reset episode to populate.")
        elif "misinfo" in active_task.lower():
            if st.session_state.obs is not None:
                obs   = st.session_state.obs
                names = ["spread_rate","fact_check","engagement","credibility","hop_count","timestep"]
                st.bar_chart(pd.DataFrame({"Feature":names,"Value":[float(obs[i]) for i in range(6)]}).set_index("Feature"))
            else:
                st.info("Reset episode to populate.")
        else:
            st.info("No active task graph.")

    # ═══════════════════════════════════════════════════════════════════
    # 📈 TAB 7 — Cumulative Reward
    # ═══════════════════════════════════════════════════════════════════
    with tab_rewards:
        st.subheader("📈 Cumulative Reward")
        if st.session_state.cumulative_rewards:
            df = pd.DataFrame({
                "Step": range(1, len(st.session_state.cumulative_rewards)+1),
                "Cumulative Reward": st.session_state.cumulative_rewards,
            }).set_index("Step")
            st.line_chart(df)
        else:
            st.info("No data yet.")
        if st.session_state.episode_rewards:
            df_eps = pd.DataFrame({
                "Episode": range(1, len(st.session_state.episode_rewards)+1),
                "Total Reward": st.session_state.episode_rewards,
            }).set_index("Episode")
            st.caption("Episode totals")
            st.bar_chart(df_eps)

    # ═══════════════════════════════════════════════════════════════════
    # 🔍 TAB 8 — Reward Breakdown
    # ═══════════════════════════════════════════════════════════════════
    with tab_breakdown:
        st.subheader("🔍 Reward Breakdown (Last Step)")
        if st.session_state.log:
            breakdown = st.session_state.get("last_breakdown") or {}
            if breakdown:
                labels = ["correctness","fp_cost","collateral_damage","speed_bonus","escalation_penalty"]
                bc = st.columns(5)
                for col, lbl in zip(bc, labels):
                    col.metric(lbl, f"{float(breakdown.get(lbl, 0.0)):.3f}")
                st.caption(f"**Total: {float(breakdown.get('total', 0.0)):.4f}**")

                # Breakdown bar
                vals = {l: float(breakdown.get(l, 0.0)) for l in labels}
                df_bd = pd.DataFrame({"Component": list(vals.keys()), "Value": list(vals.values())}).set_index("Component")
                st.bar_chart(df_bd)
            else:
                st.info("Step forward to see breakdown.")
        else:
            st.info("No steps yet.")

    # ═══════════════════════════════════════════════════════════════════
    # 🧠 TAB 9 — Model Architecture
    # ═══════════════════════════════════════════════════════════════════
    with tab_arch:
        st.subheader("🧠 NEMESIS Model Architecture")
        st.markdown(_model_info_html(), unsafe_allow_html=True)

        st.markdown('<p class="section-title">Action Space</p>', unsafe_allow_html=True)
        arch_df = pd.DataFrame([
            {"ID": k, "Label": v, "Reward (TP)": "+1.0" if k==3 else "varies",
             "Description": {
                 0:"Allow — no action taken",1:"Warn — flag account",
                 2:"Restrict — reduce reach and continue monitoring",3:"Remove — suspend/takedown immediately",
                 4:"Escalate — human review required"
             }[k]}
            for k, v in ACTION_EMOJI.items()
        ])
        st.dataframe(arch_df, use_container_width=True)

        st.markdown('<p class="section-title">HuggingFace Integration</p>', unsafe_allow_html=True)
        hf_status = "✅ HF_TOKEN set" if os.environ.get("HF_TOKEN") else "⚠️ HF_TOKEN not set (using fallback projection)"
        st.info(f"**Llama-4-Maverick-17B-128E-Instruct** via HF Inference API\n\n{hf_status}")
        st.code("export HF_TOKEN=hf_xxx...\nstreamlit run dashboard/app.py", language="bash")

        st.markdown('<p class="section-title">Tabular Features (obs dims 0–4)</p>', unsafe_allow_html=True)
        feat_df = pd.DataFrame([
            {"Dim": 0, "Feature": "account_age_days",      "Range": "[0, 1]"},
            {"Dim": 1, "Feature": "posts_per_hour",         "Range": "[0, 1]"},
            {"Dim": 2, "Feature": "follower_ratio",         "Range": "[0, 1]"},
            {"Dim": 3, "Feature": "login_time_variance",    "Range": "[0, 1]"},
            {"Dim": 4, "Feature": "content_repetition",     "Range": "[0, 1]"},
        ])
        st.dataframe(feat_df, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════
    # 🌍 TAB 10 — Mastodon Live
    # ═══════════════════════════════════════════════════════════════════
    with tab_mastodon:
        st.subheader("🌍 Mastodon Live (Real-time)")
        st.write("Displays the last 50 Mastodon posts analyzed by the agent.")
        if HAS_AUTOREFRESH and os.environ.get("MASTODON_ACCESS_TOKEN"):
            st_autorefresh(interval=5000, key="mastodon_refresh")

        mrecords = st.session_state.get("nemesis_records", [])
        if mrecords:
            recent = list(reversed(mrecords[-50:]))
            df_m = pd.DataFrame([{
                "Account ID": r["account_id"],
                "Content Snippet": r["content_snippet"],
                "Prediction": r["prediction"],
                "Ground Truth": r["ground_truth"],
                "Reward": r["reward"],
                "Confidence": f"{r['confidence']:.3f}",
                "Flagged Reason": r["flagged_reason"],
            } for r in recent])
            
            def _row_color(row):
                is_act = row["Prediction"] != "No Action"
                is_bot = row["Ground Truth"] == "Bot"
                if is_act and is_bot:   return ["background-color:#1b4332"] * len(row)
                if is_act and not is_bot: return ["background-color:#4a1122"] * len(row)
                return [""] * len(row)
                
            st.dataframe(df_m.style.apply(_row_color, axis=1), use_container_width=True)
        else:
            st.info("No Mastodon posts analyzed yet. Ensure MASTODON_ACCESS_TOKEN is set and run NEMESIS inference.")

    # ═══════════════════════════════════════════════════════════════════
    # ⚖️ TAB 11 — Mastodon Statistics
    # ═══════════════════════════════════════════════════════════════════
    with tab_mastodon_stats:
        st.subheader("⚖️ Mastodon Statistics")
        mrecords = st.session_state.get("nemesis_records", [])
        if mrecords:
            cat_counts = {}
            total = len(mrecords)
            for r in mrecords:
                cats = r.get("categories", [])
                for c in cats:
                    cat_counts[c] = cat_counts.get(c, 0) + 1
            
            if cat_counts:
                st.markdown("### Toxicity Distribution Metrics")
                df_c = pd.DataFrame([{
                    "Category": k, 
                    "Percentage": f"{(v/total)*100:.1f}%",
                    "Count": v
                } for k, v in cat_counts.items()])
                st.dataframe(df_c, use_container_width=True)
            else:
                st.info("No policy violations detected yet in the current session.")
        else:
            st.info("No Mastodon posts analyzed yet.")

    # ── Auto-play loop ───────────────────────────────────────────────
    if st.session_state.running:
        if st.session_state.terminated or st.session_state.truncated:
            st.session_state.running = False
            st.session_state.last_autoplay_tick = None
            st.session_state.episode_rewards.append(st.session_state.ep_reward)
            st.sidebar.success(f"Episode finished · Reward: {st.session_state.ep_reward:.2f}")
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
        else:
            if not HAS_AUTOREFRESH:
                st.session_state.running = False
                st.warning(
                    "Auto-play requires `streamlit-autorefresh` for non-blocking updates."
                )
            else:
                interval_ms = int(max(50.0, 1000.0 / max(float(auto_speed), 1.0)))
                current_tick = st_autorefresh(interval=interval_ms, key="episode_autoplay")
                if st.session_state.get("last_autoplay_tick") != current_tick:
                    st.session_state.last_autoplay_tick = current_tick
                    step_agent(env, agent)
                    st.rerun()


if __name__ == "__main__":
    main()
