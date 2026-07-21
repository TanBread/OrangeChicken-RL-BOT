import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collect import ActorCritic
from env import OBS_SIZE, ACT_SIZE


MODEL_DIR = Path("models")
LOG_DIR = Path("logs")


def train_bc(data_dir="replay_data", epochs=50, batch_size=256, lr=3e-4,
             save_every=10, model_name="rl_bc"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    data_path = Path(data_dir)
    obs_file = data_path / "observations.npy"
    acts_file = data_path / "actions.npy"

    if not obs_file.exists() or not acts_file.exists():
        print(f"Error: Data not found in {data_dir}")
        print("Run extract_replay.py first!")
        return

    obs = np.load(obs_file)
    acts = np.load(acts_file)
    print(f"Loaded {len(obs)} frames from {data_dir}")

    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    acts_tensor = torch.tensor(acts, dtype=torch.float32, device=device)

    model = ActorCritic(OBS_SIZE, ACT_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    MODEL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    metrics_file = LOG_DIR / "bc_metrics.jsonl"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")
    print(f"Training: {epochs} epochs, batch_size={batch_size}, lr={lr}\n")

    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        indices = torch.randperm(len(obs_tensor))
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(obs_tensor), batch_size):
            end = min(start + batch_size, len(obs_tensor))
            batch_idx = indices[start:end]

            batch_obs = obs_tensor[batch_idx]
            batch_acts = acts_tensor[batch_idx]

            logits, value, log_std = model(batch_obs)
            pred_actions = torch.tanh(logits)

            action_loss = F.mse_loss(pred_actions, batch_acts)

            value_loss = F.mse_loss(value.squeeze(-1), torch.zeros_like(value.squeeze(-1)))

            loss = action_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        elapsed = time.time()

        print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_DIR / f"{model_name}_best.pt")

        if epoch % save_every == 0:
            torch.save(model.state_dict(), MODEL_DIR / f"{model_name}_epoch{epoch}.pt")

            with open(metrics_file, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch,
                    "loss": float(avg_loss),
                    "best_loss": float(best_loss),
                    "frames": len(obs),
                }) + "\n")

    torch.save(model.state_dict(), MODEL_DIR / f"{model_name}_final.pt")
    print(f"\nDone! Best loss: {best_loss:.6f}")
    print(f"Models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="replay_data", help="Folder with extracted replay data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--model-name", default="rl_bc")
    args = parser.parse_args()

    train_bc(data_dir=args.data, epochs=args.epochs, batch_size=args.batch_size,
             lr=args.lr, save_every=args.save_every, model_name=args.model_name)
