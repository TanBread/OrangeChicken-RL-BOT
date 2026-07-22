import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from multiprocessing import Process, Queue, Pipe


MODEL_DIR = Path("models")
LOG_DIR = Path("logs")


class ActorCritic(nn.Module):
    def __init__(self, obs_size=18, act_size=8, hidden=12288):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, act_size)
        self.critic = nn.Linear(hidden, 1)
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


def trainer_process_fn(ctrl_pipe, resp_pipe, device, n_epochs, gamma, gae_lambda,
                       clip_range, ent_coef, vf_coef, lr):
    model = ActorCritic()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    while True:
        msg = ctrl_pipe.recv()
        if msg is None:
            break

        obs_n, acts_n, rews_n, dones_n, vals_n, old_lp_n = msg
        n = len(rews_n)

        obs_t = torch.from_numpy(obs_n).float().to(device)
        act_t = torch.from_numpy(acts_n).float().to(device)
        rew_t = torch.from_numpy(rews_n).float().to(device)
        done_t = torch.from_numpy(dones_n).float().to(device)
        val_t = torch.from_numpy(vals_n).float().to(device)
        old_lp = torch.from_numpy(old_lp_n).float().to(device)

        advantages = torch.zeros_like(rew_t)
        last_gae = 0
        for t in reversed(range(n)):
            next_val = val_t[t + 1] if t < n - 1 else 0
            delta = rew_t[t] + gamma * next_val * (1 - done_t[t]) - val_t[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - done_t[t]) * last_gae
        returns = advantages + val_t

        t0 = time.time()
        mb_size = 4096
        n = len(rews_n)
        for _ in range(n_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, mb_size):
                idx = perm[start:start + mb_size]
                mb_obs = obs_t[idx]
                mb_act = act_t[idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]
                mb_old = old_lp[idx]

                new_lp, new_val, entropy = model.evaluate(mb_obs, mb_act)
                ratio = (new_lp - mb_old).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_val, mb_ret)
                entropy_loss = -entropy.mean()
                loss = actor_loss + vf_coef * critic_loss + ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        train_time = time.time() - t0
        new_state = {k: v.cpu() for k, v in model.state_dict().items()}
        resp_pipe.send((train_time, new_state))

    resp_pipe.send(None)


def env_worker(q_in, q_out, n_envs, stats_queue, orange_size=1):
    from env import make_env

    envs = [make_env(team_size=1, orange_size=orange_size, tick_skip=32) for _ in range(n_envs)]
    obs_list = []
    ep_rewards = [0.0] * n_envs
    ep_goals = [0] * n_envs

    for env in envs:
        obs_dict = env.reset()
        agent_ids = list(obs_dict.keys())
        agent_id = [a for a in agent_ids if "blue" in a][0]
        obs_list.append((obs_dict, agent_ids, agent_id))

    def get_obs():
        return [obs_list[i][0][obs_list[i][2]].flatten() for i in range(n_envs)]

    q_out.put(get_obs())

    step_counts = [0] * n_envs
    random_offsets = [random.randint(0, 200) for _ in range(n_envs)]

    while True:
        msg = q_in.get()
        if msg == "stop":
            break

        next_obs_list = []
        rewards_list = []
        dones_list = []

        for i in range(n_envs):
            obs_dict, agent_ids, agent_id = obs_list[i]
            actions_dict = {}
            for a in agent_ids:
                if a == agent_id:
                    actions_dict[a] = msg[i]
                else:
                    actions_dict[a] = np.random.uniform(-1, 1, 8).astype(np.float32)

            next_obs_dict, rew, terminated, truncated = envs[i].step(actions_dict)
            done = terminated[agent_id] or truncated[agent_id]
            reward = rew[agent_id]
            step_counts[i] += 1

            ep_rewards[i] += reward
            if reward >= 5.0:
                ep_goals[i] += 1

            if done and step_counts[i] > random_offsets[i]:
                stats_queue.put({
                    "reward": ep_rewards[i],
                    "goals": ep_goals[i],
                })
                next_obs_dict = envs[i].reset()
                agent_ids = list(next_obs_dict.keys())
                agent_id = [a for a in agent_ids if "blue" in a][0]
                ep_rewards[i] = 0.0
                ep_goals[i] = 0
                random_offsets[i] = random.randint(20, 80)
                step_counts[i] = 0

            obs_list[i] = (next_obs_dict, agent_ids, agent_id)
            next_obs_list.append(next_obs_dict[agent_id].flatten())
            rewards_list.append(reward)
            dones_list.append(done)

        q_out.put({"obs": next_obs_list, "rewards": rewards_list, "dones": dones_list})


def rl(total_games=100, save_every=10, n_workers=8, total_envs=192,
       gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
       vf_coef=0.5, lr=3e-4, n_epochs=6, batch_steps=96, time_limit=None, orange_size=1):
    from env import OBS_SIZE, ACT_SIZE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs_per_worker = total_envs // n_workers
    total_envs = n_workers * envs_per_worker
    batch_size = total_envs * batch_steps

    print(f"Training on: {device} | {n_workers} workers x {envs_per_worker} envs = {total_envs} envs", flush=True)
    print(f"Batch: {batch_steps} steps/worker = {batch_size} transitions | {n_epochs} PPO epochs", flush=True)

    MODEL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    metrics_file = LOG_DIR / "metrics.jsonl"

    model = ActorCritic(OBS_SIZE, ACT_SIZE).to(device)
    count = sum(p.numel() for p in model.parameters())
    print(f"Model: {count:,} params\n", flush=True)

    ctrl_send, ctrl_recv = Pipe(True)
    resp_recv, resp_send = Pipe(True)

    trainer = Process(target=trainer_process_fn,
                      args=(ctrl_recv, resp_send, device, n_epochs, gamma,
                            gae_lambda, clip_range, ent_coef, vf_coef, lr),
                      daemon=True)
    trainer.start()
    ctrl_recv.close()
    resp_send.close()

    q_ins = []
    q_outs = []
    workers = []
    stats_queue = Queue()

    t_start = time.time()
    for _ in range(n_workers):
        q_in = Queue()
        q_out = Queue()
        p = Process(target=env_worker, args=(q_in, q_out, envs_per_worker, stats_queue, orange_size), daemon=True)
        p.start()
        q_ins.append(q_in)
        q_outs.append(q_out)
        workers.append(p)

    all_obs = []
    for g in range(n_workers):
        obs = q_outs[g].get()
        all_obs.extend(obs)
    t_ready = time.time()
    print(f"Envs ready in {t_ready - t_start:.1f}s\n", flush=True)

    batch_obs_np = []
    batch_acts_np = []
    batch_rews = []
    batch_dones = []
    batch_vals_np = []
    batch_logprobs_np = []

    games_done = 0
    global_step = 0
    train_start = time.time()
    training = False
    deadline = train_start + time_limit if time_limit else float('inf')

    while games_done < total_games and time.time() < deadline:
        obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32, device=device)
        actions, log_probs, values = model.get_actions_and_values(obs_tensor)

        obs_np = obs_tensor.cpu().numpy()

        offset = 0
        for g in range(n_workers):
            q_ins[g].put(actions[offset:offset + envs_per_worker].tolist())
            offset += envs_per_worker

        all_obs = []
        for g in range(n_workers):
            result = q_outs[g].get()
            base = g * envs_per_worker
            batch_obs_np.append(obs_np[base:base + envs_per_worker])
            batch_acts_np.append(actions[base:base + envs_per_worker])
            batch_rews.extend(result["rewards"])
            batch_dones.extend(result["dones"])
            batch_vals_np.append(values[base:base + envs_per_worker])
            batch_logprobs_np.append(log_probs[base:base + envs_per_worker])
            global_step += envs_per_worker
            all_obs.extend(result["obs"])

        while not stats_queue.empty():
            stats = stats_queue.get()
            games_done += 1
            elapsed = time.time() - train_start
            tgm = games_done / (elapsed / 60) if elapsed > 0 else 0
            avg = elapsed / games_done
            status = "training" if training else "collecting"
            print(f"Game {games_done}/{total_games} | Reward: {stats['reward']:.1f} | Goals: {stats['goals']} | {tgm:.0f} TG/m | {tgm:.1f}x | {avg:.1f}s/game | {status}", flush=True)

            if games_done % save_every == 0:
                save_path = MODEL_DIR / f"rl_game{games_done}.pt"
                torch.save(model.state_dict(), save_path)
                # Rolling checkpoint cleanup: keep only last 3
                checkpoints = sorted(MODEL_DIR.glob("rl_game*.pt"), key=lambda p: p.stat().st_mtime)
                while len(checkpoints) > 3:
                    old = checkpoints.pop(0)
                    old.unlink(missing_ok=True)
                with open(metrics_file, "a") as f:
                    f.write(json.dumps({
                        "game": games_done,
                        "steps": global_step,
                        "reward": float(stats["reward"]),
                        "loss": 0.0,
                        "goals": stats["goals"],
                        "tgpt": round(games_done * 60.0 / elapsed, 1) if elapsed > 0 else 0,
                        "tgptt": round(elapsed, 1),
                        "total_time": round(elapsed, 1),
                    }) + "\n")
                    f.flush()

            if games_done >= total_games:
                break

        if not training and len(batch_obs_np) >= batch_size:
            batch_data = (
                np.concatenate(batch_obs_np, axis=0),
                np.concatenate(batch_acts_np, axis=0),
                np.array(batch_rews, dtype=np.float32),
                np.array(batch_dones, dtype=np.float32),
                np.concatenate(batch_vals_np, axis=0),
                np.concatenate(batch_logprobs_np, axis=0),
            )
            ctrl_send.send(batch_data)
            batch_obs_np = []
            batch_acts_np = []
            batch_rews = []
            batch_dones = []
            batch_vals_np = []
            batch_logprobs_np = []
            training = True

        if training and resp_recv.poll():
            result = resp_recv.recv()
            if result is not None:
                train_time, new_state = result
                model.load_state_dict({k: v.to(device) for k, v in new_state.items()})
                training = False

    ctrl_send.send(None)

    for q in q_ins:
        q.put("stop")

    for g in range(n_workers):
        try:
            q_outs[g].get(timeout=5)
        except Exception:
            pass

    for p in workers:
        p.join(timeout=3)
        if p.is_alive():
            p.terminate()

    trainer.join(timeout=5)
    if trainer.is_alive():
        trainer.terminate()

    torch.save(model.state_dict(), MODEL_DIR / "rl_final.pt")
    elapsed = time.time() - train_start
    print(f"\nDone. {games_done} games in {elapsed:.1f}s ({elapsed / games_done:.2f}s/game)", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--envs", type=int, default=192)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--time", type=float, default=None, help="Time limit in seconds")
    parser.add_argument("--orange", type=int, default=1, help="Number of orange opponents")

    args = parser.parse_args()
    rl(total_games=args.games, save_every=args.save_every,
       n_workers=args.cores, total_envs=args.envs, lr=args.lr,
       batch_steps=args.batch_steps, n_epochs=args.epochs,
       time_limit=args.time, orange_size=args.orange)
