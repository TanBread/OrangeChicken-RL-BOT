import time, torch, numpy as np, sys
from multiprocessing import Process, Queue
from env import make_env, OBS_SIZE, ACT_SIZE
from collect import ActorCritic


def env_worker(q_in, q_out, n_envs):
    envs = [make_env(team_size=1, tick_skip=32) for _ in range(n_envs)]
    obs_list = []

    for env in envs:
        obs_dict = env.reset()
        agent_ids = list(obs_dict.keys())
        agent_id = [a for a in agent_ids if 'blue' in a][0]
        obs_list.append((obs_dict, agent_ids, agent_id))

    def get_obs():
        return [obs_list[i][0][obs_list[i][2]].flatten() for i in range(n_envs)]

    q_out.put({"obs": get_obs()})

    while True:
        msg = q_in.get()
        if msg == 'stop':
            break

        for i in range(n_envs):
            obs_dict, agent_ids, agent_id = obs_list[i]
            actions_dict = {a: msg[i] for a in agent_ids}
            next_obs_dict, rew, terminated, truncated = envs[i].step(actions_dict)
            done = terminated[agent_id] or truncated[agent_id]

            if done:
                next_obs_dict = envs[i].reset()
                agent_ids = list(next_obs_dict.keys())
                agent_id = [a for a in agent_ids if "blue" in a][0]

            obs_list[i] = (next_obs_dict, agent_ids, agent_id)

        q_out.put({"obs": [obs_list[i][0][obs_list[i][2]].flatten() for i in range(n_envs)]})

    q_out.put("done")


def run_benchmark(n_envs, groups, duration=10):
    total_envs = n_envs * groups
    q_ins = []
    q_outs = []
    workers = []

    for _ in range(groups):
        q_in = Queue()
        q_out = Queue()
        p = Process(target=env_worker, args=(q_in, q_out, n_envs), daemon=True)
        p.start()
        q_ins.append(q_in)
        q_outs.append(q_out)
        workers.append(p)

    model = ActorCritic(OBS_SIZE, ACT_SIZE).cuda()

    all_obs = []
    for g in range(groups):
        result = q_outs[g].get(timeout=120)
        all_obs.extend(result["obs"])

    start = time.time()
    while time.time() - start < duration:
        obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32).cuda()
        with torch.no_grad():
            actions, _, _ = model.get_actions_and_values(obs_tensor)

        offset = 0
        for g in range(groups):
            q_ins[g].put(actions[offset:offset + n_envs].tolist())
            offset += n_envs

        all_obs = []
        for g in range(groups):
            result = q_outs[g].get(timeout=30)
            all_obs.extend(result["obs"])

    elapsed = time.time() - start

    for q in q_ins:
        q.put('stop')

    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    return total_envs / elapsed


if __name__ == '__main__':
    torch.cuda.set_device(0)
    cores = int(input("How many cores? [16]: ").strip() or "16")
    runs = int(input("Runs per config? [3]: ").strip() or "3")

    configs = list(range(5, 201, 5))
    print(f"\nTesting {len(configs)} configs x {runs} runs with {cores} cores\n")

    best = 0
    best_envs = 1

    for envs_per_core in configs:
        vals = []
        for run in range(runs):
            val = run_benchmark(envs_per_core, cores, duration=10)
            vals.append(val)
        avg = sum(vals) / len(vals)
        print(f"  {envs_per_core:3d} envs/core ({envs_per_core*cores:5d} total): {avg:8.0f} envs/s")

        if avg > best:
            best = avg
            best_envs = envs_per_core

    print(f"\n{'='*50}")
    print(f"BEST: {best_envs} envs/core ({best_envs * cores} total) = {best:.0f} envs/s")
    print(f"{'='*50}")

    with open("benchmark_config.txt", "w") as f:
        f.write(f"{best_envs * cores}\n")
    print(f"\nSaved to benchmark_config.txt - use --envs {best_envs * cores}")

    input("\nPress Enter to exit...")
