import time
import torch
import numpy as np
import sys
from pathlib import Path
from multiprocessing import Process, Queue

sys.path.insert(0, str(Path(__file__).parent / "src"))
from env import make_env, OBS_SIZE, ACT_SIZE
from collect import ActorCritic


PROJECT_DIR = Path(__file__).parent


def env_worker(q_in, q_out, n_envs, stats_queue):
    envs = [make_env(team_size=1, tick_skip=32) for _ in range(n_envs)]
    obs_list = []

    for env in envs:
        obs_dict = env.reset()
        agent_ids = list(obs_dict.keys())
        agent_id = [a for a in agent_ids if "blue" in a][0]
        obs_list.append((obs_dict, agent_ids, agent_id))

    games_done = [0] * n_envs
    ep_rewards = [0.0] * n_envs

    def get_obs():
        return [obs_list[i][0][obs_list[i][2]].flatten() for i in range(n_envs)]

    q_out.put({"obs": get_obs()})

    while True:
        msg = q_in.get()
        if msg == "stop":
            break

        for i in range(n_envs):
            obs_dict, agent_ids, agent_id = obs_list[i]
            actions_dict = {a: msg[i] for a in agent_ids}

            next_obs_dict, rew, terminated, truncated = envs[i].step(actions_dict)
            done = terminated[agent_id] or truncated[agent_id]
            reward = rew[agent_id]

            ep_rewards[i] += reward

            if done:
                stats_queue.put({"reward": ep_rewards[i]})
                games_done[i] += 1
                ep_rewards[i] = 0.0
                next_obs_dict = envs[i].reset()
                agent_ids = list(next_obs_dict.keys())
                agent_id = [a for a in agent_ids if "blue" in a][0]

            obs_list[i] = (next_obs_dict, agent_ids, agent_id)

        q_out.put({"obs": [obs_list[i][0][obs_list[i][2]].flatten() for i in range(n_envs)]})

    q_out.put("done")


def run_single_config(envs_per_core, n_cores, duration=15):
    total_envs = envs_per_core * n_cores
    q_ins = []
    q_outs = []
    workers = []
    stats_queue = Queue()

    for _ in range(n_cores):
        q_in = Queue()
        q_out = Queue()
        p = Process(target=env_worker, args=(q_in, q_out, envs_per_core, stats_queue), daemon=True)
        p.start()
        q_ins.append(q_in)
        q_outs.append(q_out)
        workers.append(p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(OBS_SIZE, ACT_SIZE).to(device)

    all_obs = []
    for g in range(n_cores):
        result = q_outs[g].get(timeout=120)
        all_obs.extend(result["obs"])

    games_done = 0
    total_reward = 0.0
    start = time.time()

    while time.time() - start < duration:
        obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32, device=device)
        with torch.no_grad():
            actions, _, _ = model.get_actions_and_values(obs_tensor)

        offset = 0
        for g in range(n_cores):
            q_ins[g].put(actions[offset:offset + envs_per_core].tolist())
            offset += envs_per_core

        all_obs = []
        for g in range(n_cores):
            result = q_outs[g].get(timeout=30)
            all_obs.extend(result["obs"])

        while not stats_queue.empty():
            stats = stats_queue.get()
            games_done += 1
            total_reward += stats["reward"]

    elapsed = time.time() - start

    for q in q_ins:
        q.put("stop")

    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    games_per_min = (games_done / elapsed) * 60 if elapsed > 0 else 0
    games_per_sec = games_done / elapsed if elapsed > 0 else 0
    tg_tt_ratio = games_per_sec / elapsed if elapsed > 0 else 0
    avg_reward = total_reward / games_done if games_done > 0 else 0

    return {
        "envs_per_core": envs_per_core,
        "total_envs": total_envs,
        "games": games_done,
        "games_per_min": games_per_min,
        "tg_tt_ratio": tg_tt_ratio,
        "avg_reward": avg_reward,
        "elapsed": elapsed,
    }


def run_benchmark_batch(configs, n_cores, duration=15):
    results = []

    for envs_per_core in configs:
        print(f"  Testing {envs_per_core} envs/core ({envs_per_core * n_cores} total)...", end=" ", flush=True)
        result = run_single_config(envs_per_core, n_cores, duration)
        results.append(result)
        print(f"{result['games_per_min']:.1f} games/min | TG/s:TT = {result['tg_tt_ratio']:.4f} | avg reward: {result['avg_reward']:.0f}")

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: No CUDA GPU found, benchmark will be slow")

    cores = int(input("\nHow many cores to use? [16]: ").strip() or "16")
    duration = int(input("Seconds per test? [15]: ").strip() or "15")

    configs = list(range(5, 201, 5))
    print(f"\nTesting {len(configs)} configs with {cores} cores, {duration}s each")
    print(f"Estimated time: ~{len(configs) * duration // 60} min\n")

    print("=" * 60)
    results = run_benchmark_batch(configs, cores, duration)
    print("=" * 60)

    best = max(results, key=lambda r: r["tg_tt_ratio"])

    print(f"\n{'='*60}")
    print(f"BEST: {best['envs_per_core']} envs/core ({best['total_envs']} total)")
    print(f"  {best['games_per_min']:.1f} games/min | TG/s:TT = {best['tg_tt_ratio']:.4f}")
    print(f"  {best['games']} games in {best['elapsed']:.1f}s | Avg reward: {best['avg_reward']:.0f}")
    print(f"{'='*60}")

    print(f"\nAll results:")
    print(f"{'Envs/Core':>10} {'Total':>8} {'Games/min':>10} {'TG/s:TT':>10} {'Avg Reward':>12}")
    print("-" * 52)
    for r in sorted(results, key=lambda x: x["envs_per_core"]):
        marker = " <-- BEST" if r == best else ""
        print(f"{r['envs_per_core']:>10} {r['total_envs']:>8} {r['games_per_min']:>10.1f} {r['tg_tt_ratio']:>10.4f} {r['avg_reward']:>12.0f}{marker}")

    config_path = PROJECT_DIR / "benchmark_config.txt"
    with open(config_path, "w") as f:
        f.write(f"{best['total_envs']}\n")
    print(f"\nSaved to benchmark_config.txt - use --envs {best['total_envs']}")

    input("\nPress Enter to exit...")
