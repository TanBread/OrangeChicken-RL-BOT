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

    q_out.put(get_obs())
    games_done = 0
    while True:
        msg = q_in.get()
        if msg == 'stop':
            break
        for i in range(n_envs):
            obs_dict, agent_ids, agent_id = obs_list[i]
            actions_dict = {a: msg[i] for a in agent_ids}
            next_obs_dict, rewards, terminated, truncated = envs[i].step(actions_dict)
            done = terminated[agent_id] or truncated[agent_id]
            if done:
                games_done += 1
                obs_dict = envs[i].reset()
                agent_ids = list(obs_dict.keys())
                agent_id = [a for a in agent_ids if 'blue' in a][0]
            obs_list[i] = (obs_dict, agent_ids, agent_id)
        q_out.put(get_obs())
    q_out.put(games_done)


def run_benchmark(n_envs, groups, duration=10):
    total_envs = n_envs * groups
    q_ins = []
    q_outs = []
    workers = []

    t0 = time.time()
    for _ in range(groups):
        q_in = Queue()
        q_out = Queue()
        p = Process(target=env_worker, args=(q_in, q_out, n_envs), daemon=True)
        p.start()
        q_ins.append(q_in)
        q_outs.append(q_out)
        workers.append(p)
    t1 = time.time()
    print(f"  Spawn time: {t1 - t0:.1f}s ({groups} workers, {total_envs} envs)")

    model = ActorCritic(OBS_SIZE, ACT_SIZE).cuda()
    start = time.time()

    first_game = None
    while time.time() - start < duration:
        all_obs = []
        alive = []
        for g in range(groups):
            msg = q_outs[g].get()
            if isinstance(msg, list):
                all_obs.extend(msg)
                alive.append(g)

        if not all_obs:
            break

        if first_game is None:
            first_game = time.time() - start
            print(f"  First obs ready: {first_game:.1f}s after benchmark start")

        obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32).cuda()
        with torch.no_grad():
            actions, _, _ = model.get_actions_and_values(obs_tensor)

        offset = 0
        for g in alive:
            q_ins[g].put(actions[offset:offset+n_envs].tolist())
            offset += n_envs

    elapsed = time.time() - start

    for q in q_ins:
        q.put('stop')

    real_games = 0
    for g in range(groups):
        while True:
            msg = q_outs[g].get(timeout=10)
            if isinstance(msg, int):
                real_games += msg
                break

    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    tgs = real_games / elapsed
    print(f"  {total_envs:4d} envs ({groups}x{n_envs}): {tgs:6.1f} TG/s ({elapsed:.1f}s, {real_games} games)")
    return tgs


if __name__ == '__main__':
    torch.cuda.set_device(0)
    t_import = time.time()
    print("=== 60s games, 192 envs (6x32), 60s benchmark ===")
    tgs = run_benchmark(32, 6, duration=60)
    total_startup = time.time() - t_import
    tgm = tgs * 60
    print(f"\nResult: {tgm:.0f} TG/m")
    print(f"Total startup: {total_startup:.1f}s")
