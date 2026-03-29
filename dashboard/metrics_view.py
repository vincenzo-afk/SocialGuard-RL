"""
dashboard/metrics_view.py — Reward and decision charts for SocialGuard-RL.

Helper components for render lines and metric dashboards in Streamlit.
"""

import pandas as pd
import streamlit as st


def render_metrics_cards(episode_reward: float, tp: int, fp: int, episode_len: int) -> None:
    """Render standard key metrics across the top of the dashboard."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reward", f"{episode_reward:.2f}")
    with col2:
        st.metric("Bots Removed (TP)", tp)
    with col3:
        st.metric("Innocents Removed (FP)", fp)
    with col4:
        st.metric("Current Timestep", episode_len)


def render_reward_chart(rewards: list[float]) -> None:
    """Render a cumulative running reward chart over time."""
    if not rewards:
        st.info("Waiting for data to chart reward progression...")
        return
        
    df = pd.DataFrame(rewards, columns=["Episode Reward"])
    st.line_chart(df)


def render_decision_log(log: list[dict]) -> None:
    """Render a table view of past decisions made."""
    if not log:
        st.info("Decision log is empty.")
        return
        
    df = pd.DataFrame(log)
    st.dataframe(df, use_container_width=True)
