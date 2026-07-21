import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from env import make_env, OBS_SIZE, ACT_SIZE
from collect import ActorCritic


def play(model_path="models/rl_best.pt", num_games=3, fps=60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(OBS_SIZE, ACT_SIZE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    env = make_env(team_size=1, tick_skip=32)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-4200, 4200)
    ax.set_ylim(-5200, 5200)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#0f0f23')

    ax.plot([-4200, 4200, 4200, -4200, -4200], [-5200, -5200, 5200, 5200, -5200], 'w-', linewidth=2)
    ax.plot([-4200, 4200], [0, 0], 'w--', linewidth=1, alpha=0.3)
    ax.add_patch(plt.Rectangle((-893, -5120), 1786, 200, fill=True, color='#2ecc71', alpha=0.3))
    ax.add_patch(plt.Rectangle((-893, 4920), 1786, 200, fill=True, color='#e74c3c', alpha=0.3))

    ball_dot, = ax.plot([], [], 'o', color='white', markersize=12, zorder=5)
    car_blue, = ax.plot([], [], 's', color='#3498db', markersize=16, zorder=5)
    car_orange, = ax.plot([], [], 's', color='#e67e22', markersize=16, zorder=5)
    trail_ball, = ax.plot([], [], '-', color='white', linewidth=1, alpha=0.4)
    trail_blue, = ax.plot([], [], '-', color='#3498db', linewidth=1, alpha=0.4)

    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
                        fontsize=11, color='white', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    ball_history = deque(maxlen=100)
    blue_history = deque(maxlen=100)

    obs_dict = env.reset()
    agent_ids = list(obs_dict.keys())
    blue_id = [a for a in agent_ids if "blue" in a][0]
    orange_id = [a for a in agent_ids if "orange" in a][0]
    obs = obs_dict[blue_id].flatten()

    games_done = 0
    total_goals = 0
    paused = [False]

    def on_key(event):
        if event.key == ' ':
            paused[0] = not paused[0]
        elif event.key == 'q':
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)

    def update(frame):
        if paused[0]:
            return ball_dot, car_blue, car_orange, trail_ball, trail_blue, info_text

        nonlocal obs, games_done, total_goals, obs_dict, agent_ids, blue_id, orange_id

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, value, std = model(obs_tensor)
            dist = torch.distributions.Normal(logits, std)
            action = dist.sample().squeeze(0).cpu().numpy()

        actions_dict = {a: action for a in agent_ids}
        next_obs_dict, rewards, terminated, truncated = env.step(actions_dict)
        done = terminated[blue_id] or truncated[blue_id]

        state = env.state
        ball_pos = state.ball.position[:2]
        ball_history.append(ball_pos.copy())

        blue_pos = state.cars[blue_id].physics.position[:2]
        blue_history.append(blue_pos.copy())

        orange_pos = state.cars[orange_id].physics.position[:2]

        ball_dot.set_data([ball_pos[0]], [ball_pos[1]])
        car_blue.set_data([blue_pos[0]], [blue_pos[1]])
        car_orange.set_data([orange_pos[0]], [orange_pos[1]])

        if len(ball_history) > 1:
            bx, by = zip(*ball_history)
            trail_ball.set_data(bx, by)
        if len(blue_history) > 1:
            bx, by = zip(*blue_history)
            trail_blue.set_data(bx, by)

        boost = state.cars[blue_id].boost_amount
        vel = np.linalg.norm(state.cars[blue_id].physics.linear_velocity)
        ball_vel = np.linalg.norm(state.ball.linear_velocity)

        info_text.set_text(
            f"Game: {games_done + 1}/{num_games}  [SPACE] pause  [Q] quit\n"
            f"Reward: {rewards[blue_id]:.1f}\n"
            f"Boost: {boost:.0f}%  Speed: {vel:.0f}\n"
            f"Ball speed: {ball_vel:.0f}\n"
            f"Goals scored: {total_goals}"
        )

        if done:
            if rewards[blue_id] > 5:
                total_goals += 1
            games_done += 1
            if games_done >= num_games:
                info_text.set_text(f"FINISHED!\nGames: {games_done}\nGoals: {total_goals}\n\nPress Q to quit")
                return ball_dot, car_blue, car_orange, trail_ball, trail_blue, info_text

            obs_dict = env.reset()
            agent_ids = list(obs_dict.keys())
            blue_id = [a for a in agent_ids if "blue" in a][0]
            orange_id = [a for a in agent_ids if "orange" in a][0]
            ball_history.clear()
            blue_history.clear()
        else:
            obs_dict = next_obs_dict
            obs = obs_dict[blue_id].flatten()

        return ball_dot, car_blue, car_orange, trail_ball, trail_blue, info_text

    ani = animation.FuncAnimation(fig, update, frames=num_games * 60 * 60,
                                  interval=1000 // fps, blit=True, repeat=False)

    plt.title("OrangeChicken RL Bot", color='white', fontsize=16)
    plt.tight_layout()
    plt.show()

    print(f"\nResults: {games_done} games, {total_goals} goals")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/rl_best.pt")
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    play(model_path=args.model, num_games=args.games, fps=args.fps)
