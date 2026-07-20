import gymnasium as gym
import numpy as np
from typing import List, Dict, Any
from rlgym.api import RLGym, ActionParser, ObsBuilder, RewardFunction, AgentID
from rlgym.rocket_league.api import GameState, Car
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition
from rlgym.rocket_league.state_mutators import KickoffMutator, FixedTeamSizeMutator, MutatorSequence


OBS_SIZE = 18
ACT_SIZE = 8

SIDE_WALL_Y = 5120
GOAL_HEIGHT = 642.775


class SimpleObs(ObsBuilder):
    def get_obs_space(self, agent):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32)

    def reset(self, agents, initial_state, shared_info):
        pass

    def build_obs(self, agents, state: GameState, shared_info):
        obs_dict = {}
        ball = state.ball

        for agent_id in agents:
            car = state.cars[agent_id]

            car_pos = car.physics.position / 2300.0
            car_vel = car.physics.linear_velocity / 2300.0
            ball_pos = ball.position / 2300.0
            ball_vel = ball.linear_velocity / 2300.0

            on_ground = 1.0 if any(car.wheels_with_contact) else 0.0
            boost = car.boost_amount / 100.0

            obs = np.concatenate([
                car_pos,
                car_vel,
                ball_pos,
                ball_vel,
                [boost],
                [on_ground],
                [car.is_supersonic],
                car.physics.euler_angles,
            ]).astype(np.float32)

            obs_dict[agent_id] = obs

        return obs_dict


class ContinuousAction(ActionParser):
    def get_action_space(self, agent):
        return gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

    def reset(self, agents, initial_state, shared_info):
        pass

    def parse_actions(self, actions, state, shared_info):
        parsed = {}
        for agent_id, action in actions.items():
            a = np.array(action, dtype=np.float32).flatten()[:8]
            parsed[agent_id] = np.array([[
                np.clip(a[0], -1, 1),
                np.clip(a[1], -1, 1),
                np.clip(a[2], -1, 1),
                np.clip(a[3], -1, 1),
                np.clip(a[4], -1, 1),
                1.0 if a[5] > 0 else 0,
                1.0 if a[6] > 0 else 0,
                1.0 if a[7] > 0 else 0,
            ]], dtype=np.float32)
        return parsed


class DenseReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.last_state = None

    def reset(self, agents, initial_state, shared_info):
        self.last_state = initial_state

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        if self.last_state is None:
            self.last_state = state
            return {a: 0.0 for a in agents}

        rewards = {}
        ball = state.ball

        for agent_id in agents:
            car = state.cars[agent_id]
            prev_car = self.last_state.cars.get(agent_id)

            reward = 0.0

            ball_pos = ball.position
            car_pos = car.physics.position
            car_vel = car.physics.linear_velocity
            ball_vel = ball.linear_velocity

            their_goal_y = -SIDE_WALL_Y if car.team_num == 0 else SIDE_WALL_Y
            their_goal = np.array([0.0, their_goal_y, GOAL_HEIGHT / 2])

            dist_to_ball = np.linalg.norm(ball_pos - car_pos)

            reward += max(0, (5000 - dist_to_ball) / 5000) * 0.1

            if dist_to_ball < 300:
                reward += 0.5

            ball_to_goal = their_goal - ball_pos
            ball_speed = np.linalg.norm(ball_vel)
            if ball_speed > 100:
                ball_dir = ball_vel / (ball_speed + 1e-6)
                goal_dir = ball_to_goal / (np.linalg.norm(ball_to_goal) + 1e-6)
                alignment = np.dot(ball_dir, goal_dir)
                if alignment > 0:
                    reward += alignment * ball_speed / 2300 * 0.3

            if prev_car:
                vel_toward_ball = np.dot(car_vel, (ball_pos - car_pos) / (dist_to_ball + 1e-6))
                reward += max(0, vel_toward_ball / 2300) * 0.1

            if state.goal_scored:
                if (state.scoring_team == 0 and car.team_num == 0) or \
                   (state.scoring_team == 1 and car.team_num == 1):
                    reward += 10.0
                else:
                    reward -= 5.0

            rewards[agent_id] = float(reward)

        self.last_state = state
        return rewards


def make_env(team_size=1, tick_skip=32, game_speed=100):
    engine = RocketSimEngine(rlbot_delay=False)

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=team_size, orange_size=team_size),
        KickoffMutator()
    )

    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=SimpleObs(),
        action_parser=ContinuousAction(),
        reward_fn=DenseReward(),
        transition_engine=engine,
        termination_cond=GoalCondition(),
        truncation_cond=TimeoutCondition(60),
    )

    return env
