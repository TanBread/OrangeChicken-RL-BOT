# OrangeChicken RL Bot

A Rocket League reinforcement learning bot trained with PPO using RLGym and RocketSim.

## Features

- PPO training with configurable opponent count (1v1, 1v2, 1v3)
- Replay-based training via Ballchasing API (download, extract, behavioral clone)
- Streamlit dashboard for real-time training metrics
- 3D game viewer with multiple camera modes
- Hardware benchmarking for optimal environment count

## Requirements

- Python 3.10+
- Rocket League installed
- GPU recommended (CUDA ONLY)

## Install

```bash
git clone https://github.com/TanBread/OrangeChicken-RL-BOT.git
cd OrangeChicken-RL-BOT
pip install -r requirements.txt
```

## Usage

```bash
python run.py
```

Select a mode:
1. **Train** — Start RL training (by time or number of games)
2. **Train from Replays + RL** — Behavioral clone from replays, then fine-tune with RL
3. **Download & Extract Replays** — Fetch replays from Ballchasing and prepare training data
4. **Dashboard** — Open Streamlit training monitor at `http://localhost:8501`
5. **Watch Bot** — Visualize the bot playing in a 3D viewer

## Project Structure

| File | Description |
|------|-------------|
| `run.py` | Main entry point with interactive menu |
| `env.py` | RLGym environment setup (observations, actions, rewards) |
| `collect.py` | PPO training loop with multiprocessing |
| `train_bc.py` | Behavioral cloning from replay data |
| `download_replays.py` | Ballchasing API replay downloader |
| `extract_replay.py` | Replay data extraction |
| `dashboard.py` | Streamlit training dashboard |
| `watch.py` | 3D OpenGL game viewer |
| `bench.py` | Hardware benchmarking tool |
| `monitor.py` | CPU usage monitor |

## Tech Stack

- [RLGym](https://rlgym.org/) — Rocket League gym environment
- [RocketSim](https://github.com/RLBot/rocketSim) — Physics simulation
- [PyTorch](https://pytorch.org/) — Neural network framework
- [Streamlit](https://streamlit.io/) — Training dashboard
- [Pygame](https://pygame.org/) + OpenGL — 3D visualization
