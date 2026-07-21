import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path


OBS_SIZE = 18
ACT_SIZE = 8

SIDE_WALL_Y = 5120
GOAL_HEIGHT = 642.775


def extract_from_replay(replay_path):
    import analyzerl_parser

    df = analyzerl_parser.parse_replay(
        replay_path=str(replay_path),
        output='frames',
        return_type='pandas',
        workers=1,
    )

    observations = []
    actions = []

    for _, row in df.iterrows():
        car_x = row.get('blue_player_1_pos_x', 0.0) or 0.0
        car_y = row.get('blue_player_1_pos_y', 0.0) or 0.0
        car_z = row.get('blue_player_1_pos_z', 0.0) or 0.0
        car_vel_x = row.get('blue_player_1_vel_x', 0.0) or 0.0
        car_vel_y = row.get('blue_player_1_vel_y', 0.0) or 0.0
        car_vel_z = row.get('blue_player_1_vel_z', 0.0) or 0.0

        ball_x = row.get('ball_pos_x', 0.0) or 0.0
        ball_y = row.get('ball_pos_y', 0.0) or 0.0
        ball_z = row.get('ball_pos_z', 0.0) or 0.0
        ball_vel_x = row.get('ball_vel_x', 0.0) or 0.0
        ball_vel_y = row.get('ball_vel_y', 0.0) or 0.0
        ball_vel_z = row.get('ball_vel_z', 0.0) or 0.0

        boost = (row.get('blue_player_1_boost', 0) or 0) / 100.0

        rot_x = row.get('blue_player_1_rot_x', 0.0) or 0.0
        rot_y = row.get('blue_player_1_rot_y', 0.0) or 0.0
        rot_z = row.get('blue_player_1_rot_z', 0.0) or 0.0

        on_ground = 1.0 if car_z < 50 else 0.0
        supersonic = 1.0 if row.get('blue_player_1_supersonic', False) else 0.0

        obs = np.array([
            car_x / 2300.0, car_y / 2300.0, car_z / 2300.0,
            car_vel_x / 2300.0, car_vel_y / 2300.0, car_vel_z / 2300.0,
            ball_x / 2300.0, ball_y / 2300.0, ball_z / 2300.0,
            ball_vel_x / 2300.0, ball_vel_y / 2300.0, ball_vel_z / 2300.0,
            boost,
            on_ground,
            supersonic,
            rot_x, rot_y, rot_z,
        ], dtype=np.float32)

        throttle = row.get('blue_player_1_throttle', 0.0)
        if pd.isna(throttle):
            throttle = 0.0
        steer = row.get('blue_player_1_steer', 0.0)
        if pd.isna(steer):
            steer = 0.0
        boost_active = 1.0 if row.get('blue_player_1_boost_active', False) else 0.0
        jump_active = 1.0 if row.get('blue_player_1_jump_active', False) else 0.0
        dbl_jump = 1.0 if row.get('blue_player_1_double_jump_active', False) else 0.0
        handbrake = 1.0 if row.get('blue_player_1_handbrake', False) else 0.0
        dodge = 1.0 if row.get('blue_player_1_dodge_active', False) else 0.0
        flipped = 1.0 if row.get('blue_player_1_flipped', False) else 0.0

        action = np.array([
            np.clip(throttle, -1, 1),
            np.clip(steer, -1, 1),
            np.clip(throttle, -1, 1),
            0.0,
            0.0,
            1.0 if boost_active > 0.5 else 0,
            1.0 if jump_active > 0.5 else 0,
            1.0 if dbl_jump > 0.5 or dodge > 0.5 else 0,
        ], dtype=np.float32)

        observations.append(obs)
        actions.append(action)

    return np.array(observations, dtype=np.float32), np.array(actions, dtype=np.float32)


def process_replay_folder(folder_path, output_path):
    folder = Path(folder_path)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    replay_files = list(folder.glob("*.replay"))
    print(f"Found {len(replay_files)} replay files")

    all_obs = []
    all_acts = []

    for i, replay_file in enumerate(replay_files):
        print(f"\n[{i+1}/{len(replay_files)}] Processing: {replay_file.name}")
        try:
            obs, acts = extract_from_replay(replay_file)
            print(f"  Extracted {len(obs)} frames")
            all_obs.append(obs)
            all_acts.append(acts)
        except Exception as e:
            print(f"  Error: {e}")

    if all_obs:
        all_obs = np.concatenate(all_obs, axis=0)
        all_acts = np.concatenate(all_acts, axis=0)

        np.save(output / "observations.npy", all_obs)
        np.save(output / "actions.npy", all_acts)

        print(f"\nTotal frames: {len(all_obs)}")
        print(f"Saved to: {output / 'observations.npy'}")
        print(f"Saved to: {output / 'actions.npy'}")
    else:
        print("No data extracted!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="replay_files", help="Folder with .replay files")
    parser.add_argument("--output", default="replay_data", help="Output folder for extracted data")
    args = parser.parse_args()

    process_replay_folder(args.input, args.output)
