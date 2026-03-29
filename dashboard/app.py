"""
dashboard/app.py — Streamlit Dashboard for SocialGuard-RL.

Run live evaluation episodes step-by-step or automatically, displaying the
network state, latest actions, cumulative reward, and decision logs.

Usage::
    streamlit run dashboard/app.py
"""

import time
import argparse
import os
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from pathlib import Path

from env.env import SocialGuardEnv
from baseline import BaselineAgent
from env.spaces import ACTION_NAMES, ACTION_REMOVE
from dashboard.graph_view import generate_graph_base_html, apply_decision_log
from dashboard.metrics_view import render_metrics_cards, render_decision_log, render_reward_chart
from evaluate import load_model

st.set_page_config(layout="wide", page_title="SocialGuard-RL Dashboard")

def _resolve_trusted_path(path_str: str, trusted_dir: str, suffixes: tuple[str, ...]) -> Path:
    root = (Path.cwd() / trusted_dir).resolve()
    p = Path(path_str)
    p = (Path.cwd() / p).resolve() if not p.is_absolute() else p.resolve()
    if root not in p.parents:
        raise ValueError(f"Path must be under `{trusted_dir}/`: {p}")
    if suffixes and p.suffix.lower() not in suffixes:
        raise ValueError(f"Path must end with one of {suffixes}: {p}")
    return p

def get_env(config_path: str) -> SocialGuardEnv:
    """Load env. Caches are possible but environment holds internal state directly."""
    cfg_path = _resolve_trusted_path(config_path, "configs", (".yaml", ".yml"))
    if (
        "sg_env" not in st.session_state
        or st.session_state.get("sg_env_config_path") != str(cfg_path)
    ):
        if "sg_env" in st.session_state:
            try:
                st.session_state.sg_env.close()
            except Exception:
                pass
        st.session_state.sg_env = SocialGuardEnv(config_path=str(cfg_path))
        st.session_state.sg_env_config_path = str(cfg_path)
    return st.session_state.sg_env

def init_session_state():
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "running" not in st.session_state:
        st.session_state.running = False
    if "obs" not in st.session_state:
        st.session_state.obs = None
    if "terminated" not in st.session_state:
        st.session_state.terminated = False
    if "truncated" not in st.session_state:
        st.session_state.truncated = False
    if "ep_reward" not in st.session_state:
        st.session_state.ep_reward = 0.0
    if "log" not in st.session_state:
        st.session_state.log = []
    if "rewards" not in st.session_state:
        st.session_state.rewards = []
    if "decision_log" not in st.session_state:
        st.session_state.decision_log = {}
    if "tp" not in st.session_state:
        st.session_state.tp = 0
    if "fp" not in st.session_state:
        st.session_state.fp = 0

def step_agent(env: SocialGuardEnv, agent):
    """Execute a single step using the loaded agent."""
    if st.session_state.terminated or st.session_state.truncated:
        return
        
    obs = st.session_state.obs
    
    if hasattr(agent, "predict"):
        action_arr, _ = agent.predict(obs, deterministic=True)
        action = int(action_arr.item() if hasattr(action_arr, "item") else action_arr)
    else:
        action = agent.act(obs)
        
    obs, reward, terminated, truncated, info = env.step(action)
    
    st.session_state.obs = obs
    st.session_state.terminated = terminated
    st.session_state.truncated = truncated
    st.session_state.ep_reward += reward
    st.session_state.step += 1
    
    gt = info.get("ground_truth", -1)
    act_str = ACTION_NAMES.get(action, str(action))
    
    if action == ACTION_REMOVE:
        if gt == 1:
            st.session_state.tp += 1
        elif gt == 0:
            st.session_state.fp += 1
    
    log_entry = {
        "Timestep": st.session_state.step,
        "Action Taken": act_str,
        "Ground Truth": "Bot" if gt == 1 else "Real" if gt == 0 else "N/A",
        "Reward": round(reward, 4)
    }
    st.session_state.log.append(log_entry)
    
    # Store decision by node_id if this is a graph task
    st_dict = env.state()
    active_task = st_dict.get("active_task", "")
    
    if "cib" in active_task.lower():
        # Requires deeper integration; for now we use episode_step
        node_id = st.session_state.step  # Example placeholder for demo
        st.session_state.decision_log[node_id] = {
            "action": action,
            "ground_truth": gt
        }


def main():
    init_session_state()
    
    st.sidebar.title("Configuration")
    dashboard_token = os.environ.get("SOCIALGUARD_DASHBOARD_TOKEN", "").strip()
    if dashboard_token:
        provided = st.sidebar.text_input("Dashboard Token", type="password")
        if provided.strip() != dashboard_token:
            st.sidebar.error("Invalid token.")
            st.stop()
    else:
        st.sidebar.warning("No `SOCIALGUARD_DASHBOARD_TOKEN` set (dashboard is unauthenticated).")

    model_file = st.sidebar.text_input("Path to Model (.zip)", value="models/ppo_task1.zip")
    config_file = st.sidebar.text_input("Path to Config (.yaml)", value="configs/task1.yaml")
    
    agent_type = st.sidebar.selectbox("Agent to Run", ["Trained Model", "Rule-based Baseline"])
    
    if st.sidebar.button("Reset Environment"):
        env = get_env(config_file)
        st.session_state.obs, _ = env.reset()
        st.session_state.terminated = False
        st.session_state.truncated = False
        st.session_state.ep_reward = 0.0
        st.session_state.step = 0
        st.session_state.log = []
        st.session_state.decision_log = {}
        st.session_state.tp = 0
        st.session_state.fp = 0
        st.session_state.running = False
        st.rerun()

    try:
        env = get_env(config_file)
    except Exception as e:
        st.sidebar.error(f"Invalid config path: {e}")
        return
    if st.session_state.obs is None:
        st.session_state.obs, _ = env.reset()

    st.title("SocialGuard-RL Dashboard")

    try:
        if agent_type == "Rule-based Baseline":
            agent = BaselineAgent()
        else:
            try:
                _ = _resolve_trusted_path(model_file, "models", (".zip",))
            except Exception as e:
                st.sidebar.error(f"Invalid model path: {e}")
                return
            agent = load_model(model_file)
    except Exception as e:
        st.error(f"Failed to load agent: {e}")
        return

    # Controls
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Step Forward (1)", use_container_width=True):
            step_agent(env, agent)
            st.rerun()
    with col2:
        if st.button("Play (Auto)", use_container_width=True):
            st.session_state.running = True
            st.rerun()
    with col3:
        if st.button("Stop (Pause)", use_container_width=True):
            st.session_state.running = False
            st.rerun()

    # Metric Cards Placeholder
    metric_placeholder = st.empty()
    with metric_placeholder:
        render_metrics_cards(
            st.session_state.ep_reward,
            st.session_state.tp,
            st.session_state.fp,
            st.session_state.step,
        )

    t1, t2 = st.tabs(["Dashboard", "Network Graph"])

    with t1:
        log_placeholder = st.empty()
        with log_placeholder:
            render_decision_log(list(reversed(st.session_state.log[-50:])))
        render_reward_chart(st.session_state.rewards)
            
    with t2:
        graph_placeholder = st.empty()
        # For tasks that don't export graph, this will render empty
        state = env.state()
        active_task = state.get("active_task", "")
        # Real graph rendering requires sim node access.
        # Fallback to simple generic graph if none exists:
        if "graph" in state:
            st.write("Graph data from Env state exists.")
        
        # Example to render graph using a dummy graph if not task_cib
        try:
            import networkx as nx
            G = nx.fast_gnp_random_graph(20, 0.1)
            sig = (int(G.number_of_nodes()), int(G.number_of_edges()))
            if st.session_state.get("graph_base_sig") != sig:
                st.session_state.graph_base_html = generate_graph_base_html(G)
                st.session_state.graph_base_sig = sig
            html = apply_decision_log(st.session_state.graph_base_html, st.session_state.decision_log)
            # Actually Pyvis is handled here
            with graph_placeholder:
                components.html(html, height=620)
        except Exception:
            st.warning("Could not render internal graph.")

    # Execution Loop
    if st.session_state.running:
        if st.session_state.terminated or st.session_state.truncated:
            st.session_state.running = False
            st.session_state.rewards.append(st.session_state.ep_reward)
            st.sidebar.success("Episode Finished.")
        else:
            step_agent(env, agent)
            time.sleep(0.1)  # 10 steps per second cap
            st.rerun()


if __name__ == "__main__":
    main()
