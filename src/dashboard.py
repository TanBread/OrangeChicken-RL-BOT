import streamlit as st
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "logs"
METRICS_FILE = LOG_DIR / "metrics.jsonl"

st.set_page_config(page_title="OrangeChicken RL", layout="wide")
st.title("OrangeChicken RL")


@st.cache_data(ttl=2)
def load_metrics():
    if not METRICS_FILE.exists():
        return []
    data = []
    with open(METRICS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


data = load_metrics()

if not data:
    st.info("No training data yet. Start training with `python run.py`")
    st.stop()

import pandas as pd
import plotly.graph_objects as go

df = pd.DataFrame(data)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Games", f"{len(df):,}")

def fmt_steps(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    return f"{n:,}"

if "steps" in df.columns:
    c2.metric("Steps", fmt_steps(df['steps'].iloc[-1]))
else:
    c2.metric("Steps", "N/A")

if "goals" in df.columns:
    c3.metric("Goals/Game (last 1k)", f"{df.tail(1000)['goals'].sum() / min(1000, len(df)):.2f}")
else:
    c3.metric("Goals/Game", "N/A")

c4.metric("Avg Reward", f"{df['reward'].mean():.0f}")

if "tgpt" in df.columns and len(df) > 1 and df["total_time"].iloc[-1] > 0:
    c5.metric("TG/m", f"{df['tgpt'].iloc[-1]:.0f}")
else:
    c5.metric("TG/m", "N/A")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Reward (last 500 games)")
    last500 = df.tail(500)
    fig_reward = go.Figure()
    fig_reward.add_trace(go.Scatter(
        x=last500["game"], y=last500["reward"],
        mode="lines",
        line=dict(color="#636EFA", width=1.5, shape="linear"),
        name="Reward"
    ))
    fig_reward.update_layout(
        xaxis_title="Game", yaxis_title="Reward",
        template="plotly_dark",
        height=400, margin=dict(l=40, r=20, t=20, b=40)
    )
    st.plotly_chart(fig_reward, use_container_width=True)

with col_right:
    if "goals" in df.columns:
        st.subheader("Goals (last 500 games)")
        last500 = df.tail(500)
        fig_goals = go.Figure()
        fig_goals.add_trace(go.Scatter(
            x=last500["game"], y=last500["goals"],
            mode="lines",
            line=dict(color="#00CC96", width=1.5, shape="linear"),
            name="Goals"
        ))
        fig_goals.update_layout(
            xaxis_title="Game", yaxis_title="Goals",
            yaxis=dict(rangemode="tozero"),
            template="plotly_dark",
            height=400, margin=dict(l=40, r=20, t=20, b=40)
        )
        st.plotly_chart(fig_goals, use_container_width=True)
    else:
        st.subheader("Game Progress")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(df) + 1)), y=df["reward"],
            mode="lines",
            line=dict(color="#00CC96", width=1.5),
            name="Reward over time"
        ))
        fig.update_layout(
            xaxis_title="Game", yaxis_title="Reward",
            template="plotly_dark",
            height=400, margin=dict(l=40, r=20, t=20, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

if st.sidebar.button("Refresh"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total games:** {len(df):,}")
if "goals" in df.columns:
    st.sidebar.markdown(f"**Total goals:** {df['goals'].sum():,}")
