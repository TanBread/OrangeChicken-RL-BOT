import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


MODEL_DIR = Path("models")
LOG_DIR = Path("logs")


class ActorCritic(nn.Module):
    def __init__(self, obs_size=18, act_size=8):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, act_size)
        self.critic = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.zeros(act_size))

    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared), self.log_std.exp()

    def get_actions_and_values(self, obs_batch):
        with torch.no_grad():
            logits, value, std = self(obs_batch)
            dist = torch.distributions.Normal(logits, std)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(-1)
            return actions.cpu().numpy(), log_probs.cpu().numpy(), value.squeeze(-1).cpu().numpy()

    def evaluate(self, obs, actions):
        logits, value, std = self(obs)
        dist = torch.distributions.Normal(logits, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value.squeeze(-1), entropy


def rl(total_games=100, save_every=10, n_envs=8, gamma=0.99, gae_lambda=0.95,
       clip_range=0.2, ent_coef=0.01, vf_coef=0.5, lr=3e-4, n_epochs=4):
    from env import make_env, OBS_SIZE, ACT_SIZE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} with {n_envs} parallel environments", flush=True)

    envs = [make_env(team_size=1, tick_skip=32) for _ in range(n_envs)]
    model = ActorCritic(OBS_SIZE, ACT_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    MODEL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    metrics_file = LOG_DIR / "metrics.jsonl"
    best_reward = -float("inf")

    global_step = 0
    games_done = 0
    train_start = time.time()

    obs_list = [None] * n_envs
    agent_ids_list = [None] * n_envs
    ep_rewards = [0.0] * n_envs
    ep_goals = [0] * n_envs
    dones = [False] * n_envs

    for i, env in enumerate(envs):
        obs_dict = env.reset()
        agent_ids = list(obs_dict.keys())
        agent_id = [a for a in agent_ids if "blue" in a][0]
        obs_list[i] = obs_dict
        agent_ids_list[i] = (agent_ids, agent_id)

    batch_obs = []
    batch_acts = []
    batch_rews = []
    batch_dones = []
    batch_vals = []
    batch_logprobs = []

    while games_done < total_games:
        obs_batch = []
        active = []

        for i in range(n_envs):
            if not dones[i]:
                obs = obs_list[i][agent_ids_list[i][1]].flatten()
                obs_batch.append(obs)
                active.append(i)

        if not active:
            for i in range(n_envs):
                obs_dict = envs[i].reset()
                agent_ids = list(obs_dict.keys())
                agent_id = [a for a in agent_ids if "blue" in a][0]
                obs_list[i] = obs_dict
                agent_ids_list[i] = (agent_ids, agent_id)
                dones[i] = False
            continue

        obs_tensor = torch.tensor(np.array(obs_batch), dtype=torch.float32).to(device)
        actions, log_probs, values = model.get_actions_and_values(obs_tensor)

        for idx, i in enumerate(active):
            agent_ids, agent_id = agent_ids_list[i]
            action = actions[idx]
            actions_dict = {a: action for a in agent_ids}

            next_obs_dict, rewards, terminated, truncated = envs[i].step(actions_dict)

            reward = rewards[agent_id]
            done_flag = terminated[agent_id] or truncated[agent_id]

            if reward >= 5.0:
                ep_goals[i] += 1

            batch_obs.append(obs_tensor[idx])
            batch_acts.append(torch.tensor(action, dtype=torch.float32))
            batch_rews.append(reward)
            batch_dones.append(done_flag)
            batch_vals.append(values[idx])
            batch_logprobs.append(log_probs[idx])

            ep_rewards[i] += reward
            obs_list[i] = next_obs_dict
            global_step += 1

            if done_flag:
                games_done += 1
                elapsed = time.time() - train_start
                avg = elapsed / games_done
                print(f"Game {games_done}/{total_games} | Reward: {ep_rewards[i]:.1f} | Goals: {ep_goals[i]} | {avg:.1f}s/game", flush=True)

                if games_done % save_every == 0:
                    tgpt = games_done * 60.0
                    with open(metrics_file, "a") as f:
                        f.write(json.dumps({
                            "game": games_done,
                            "steps": global_step,
                            "reward": float(ep_rewards[i]),
                            "loss": 0.0,
                            "goals": ep_goals[i],
                            "tgpt": round(tgpt, 1),
                            "tgptt": round(elapsed, 1),
                            "total_time": round(elapsed, 1),
                        }) + "\n")
                        f.flush()

                if ep_rewards[i] > best_reward:
                    best_reward = ep_rewards[i]
                    torch.save(model.state_dict(), MODEL_DIR / "rl_best.pt")

                if games_done >= total_games:
                    break

                obs_dict = envs[i].reset()
                agent_ids = list(obs_dict.keys())
                agent_id = [a for a in agent_ids if "blue" in a][0]
                obs_list[i] = obs_dict
                agent_ids_list[i] = (agent_ids, agent_id)
                ep_rewards[i] = 0
                ep_goals[i] = 0
                dones[i] = False

        if games_done >= total_games:
            break

        if len(batch_obs) < n_envs:
            continue

        obs_tensor = torch.stack(batch_obs).to(device)
        act_tensor = torch.stack(batch_acts).to(device)
        rew_tensor = torch.tensor(batch_rews, dtype=torch.float32).to(device)
        done_tensor = torch.tensor(batch_dones, dtype=torch.float32).to(device)
        val_tensor = torch.tensor(batch_vals, dtype=torch.float32).to(device)
        old_logprobs = torch.tensor(batch_logprobs, dtype=torch.float32).to(device)

        advantages = torch.zeros_like(rew_tensor)
        last_gae = 0
        for t in reversed(range(len(rew_tensor))):
            next_val = val_tensor[t + 1] if t < len(rew_tensor) - 1 else 0
            delta = rew_tensor[t] + gamma * next_val * (1 - done_tensor[t]) - val_tensor[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - done_tensor[t]) * last_gae

        returns = advantages + val_tensor

        for _ in range(n_epochs):
            new_logprobs, new_values, entropy = model.evaluate(obs_tensor, act_tensor)
            ratio = (new_logprobs - old_logprobs).exp()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_values, returns)
            entropy_loss = -entropy.mean()

            loss = actor_loss + vf_coef * critic_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        batch_obs.clear()
        batch_acts.clear()
        batch_rews.clear()
        batch_dones.clear()
        batch_vals.clear()
        batch_logprobs.clear()

    torch.save(model.state_dict(), MODEL_DIR / "rl_final.pt")
    elapsed = time.time() - train_start
    print(f"\nDone. {games_done} games in {elapsed:.1f}s ({elapsed/games_done:.2f}s/game). Best reward: {best_reward:.1f}", flush=True)

    for env in envs:
        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--envs", type=int, default=240)
    parser.add_argument("--lr", type=float, default=3e-4)

    args = parser.parse_args()
    rl(total_games=args.games, save_every=args.save_every, n_envs=args.envs, lr=args.lr)
