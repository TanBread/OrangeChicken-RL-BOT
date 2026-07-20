import streamlit as st
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.absolute()
LOG_DIR = BASE_DIR / "logs"
METRICS_FILE = LOG_DIR / "metrics.jsonl"

st.set_page_config(page_title="RL Training", layout="wide")
st.title("RL Training Dashboard")


@st.cache_data(ttl=2)
def load_metrics():
    if not METRICS_FILE.exists():
        return []
    data = []
    with open(METRICS_FILE) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


data = load_metrics()

if not data:
    st.info("No training data yet. Start training with `python run.py`")
    st.stop()

import pandas as pd
df = pd.DataFrame(data)

col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
col1.metric("Games", len(df))
col2.metric("Total Steps", f"{df['steps'].iloc[-1]:,}")
col3.metric("Total Goals", int(df["goals"].sum()))
col4.metric("Best Reward", f"{df['reward'].max():.1f}")
col5.metric("Latest Reward", f"{df['reward'].iloc[-1]:.1f}")
col6.metric("TGPT (Sim)", f"{df['tgpt'].iloc[-1] / 3600:.1f}h")
col7.metric("TGPTT (Real)", f"{df['tgptt'].iloc[-1] / 3600:.1f}h")
col8.metric("Speed", f"{df['tgpt'].iloc[-1] / df['tgptt'].iloc[-1]:.1f}x")

st.subheader("Reward per Game")
st.line_chart(df.set_index("game")["reward"], use_container_width=True)

st.subheader("Goals per Game")
st.bar_chart(df.set_index("game")["goals"], use_container_width=True)

st.subheader("Loss per Game")
st.line_chart(df.set_index("game")["loss"], use_container_width=True)

st.subheader("Recent Games")
recent = df.tail(20)[["game", "steps", "reward", "loss", "goals", "game_time"]].copy()
recent["reward"] = recent["reward"].round(1)
recent["loss"] = recent["loss"].round(4)
recent.columns = ["Game", "Steps", "Reward", "Loss", "Goals", "Time (s)"]
st.dataframe(recent, use_container_width=True)

if st.sidebar.button("Refresh"):
    st.cache_data.clear()
    st.rerun()
