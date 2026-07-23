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


def env_worker(q_in, q_out, n_envs, stats_queue, offset=0):
    if offset > 0:
        time.sleep(offset)
    envs = [make_env(team_size=1, tick_skip=32) for _ in range(n_envs)]
    obs_list = []

    for env in envs:
        obs_dict = env.reset()
        agent_ids = list(obs_dict.keys())
        agent_id = [a for a in agent_ids if "blue" in a][0]
        obs_list.append((obs_dict, agent_ids, agent_id))

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
                ep_rewards[i] = 0.0
                next_obs_dict = envs[i].reset()
                agent_ids = list(next_obs_dict.keys())
                agent_id = [a for a in agent_ids if "blue" in a][0]

            obs_list[i] = (next_obs_dict, agent_ids, agent_id)

        q_out.put({"obs": [obs_list[i][0][obs_list[i][2]].flatten() for i in range(n_envs)]})

    q_out.put("done")


def run_single_config(envs_per_core, n_cores, min_games=3):
    q_ins = []
    q_outs = []
    workers = []
    stats_queue = Queue()

    for i in range(n_cores):
        q_in = Queue()
        q_out = Queue()
        offset = i * 0.1
        p = Process(target=env_worker, args=(q_in, q_out, envs_per_core, stats_queue, offset))
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

    while games_done < min_games:
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

    games_per_sec = games_done / elapsed if elapsed > 0 else 0
    avg_reward = total_reward / games_done if games_done > 0 else 0

    return {
        "envs_per_core": envs_per_core,
        "total_envs": n_cores * envs_per_core,
        "games": games_done,
        "games_per_sec": games_per_sec,
        "avg_reward": avg_reward,
        "elapsed": elapsed,
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: No CUDA GPU found, benchmark will be slow")

    n_cores = 16
    min_games = int(input("\nMin games per config? [3]: ").strip() or "3")

    configs = list(range(1, 151))

    print(f"{len(configs)} configs | {n_cores} cores each | {min_games} games min")
    print("Stopping early once TG/s plateaus\n")
    print("=" * 70)

    all_results = []
    best_tgs = 0

    for i, envs_per_core in enumerate(configs):
        total = n_cores * envs_per_core
        print(f"[{i+1}/{len(configs)}] {envs_per_core}/core ({total} total)...", end=" ", flush=True)
        result = run_single_config(envs_per_core, n_cores, min_games)
        all_results.append(result)
        print(f"{result['games']} games in {result['elapsed']:.1f}s | {result['games_per_sec']:.2f} TG/s | reward: {result['avg_reward']:.0f}")

        if result["games_per_sec"] > best_tgs:
            best_tgs = result["games_per_sec"]

        if len(all_results) >= 4:
            last3 = [r["games_per_sec"] for r in all_results[-3:]]
            trend = last3[2] - last3[0]
            if trend < 0:
                print(f"\nTG/s trending down (avg last 3: {sum(last3)/3:.2f}). Stopping at {envs_per_core}/core.")
                break

    print("\n" + "=" * 70)

    best = max(all_results, key=lambda r: r["games_per_sec"])

    print(f"\nBEST: {best['envs_per_core']}/core ({best['total_envs']} total)")
    print(f"  {best['games_per_sec']:.2f} TG/s")
    print(f"  {best['games']} games in {best['elapsed']:.1f}s | Avg reward: {best['avg_reward']:.0f}")

    print(f"\n{'Envs/Core':>10} {'Total':>8} {'Games':>6} {'Time':>6} {'TG/s':>8} {'Reward':>8}")
    print("-" * 55)
    for r in sorted(all_results, key=lambda x: x["envs_per_core"]):
        marker = " <--" if r == best else ""
        print(f"{r['envs_per_core']:>10} {r['total_envs']:>8} {r['games']:>6} {r['elapsed']:>5.1f}s {r['games_per_sec']:>8.2f} {r['avg_reward']:>8.0f}{marker}")

    config_path = PROJECT_DIR / "benchmark_config.txt"
    with open(config_path, "w") as f:
        f.write(f"{best['total_envs']}\n")
    print(f"\nSaved to benchmark_config.txt - use --envs {best['total_envs']}")

    input("\nPress Enter to exit...")
