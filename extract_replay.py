import os
import struct
import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

OBS_SIZE = 18
ACT_SIZE = 8


def extract_from_replay(replay_path):
    import analyzerl_parser

    df = analyzerl_parser.parse_replay(
        replay_path=str(replay_path),
        output='frames',
        return_type='pandas',
        workers=3,
    )

    n = len(df)
    if n == 0:
        return np.empty((0, OBS_SIZE), dtype=np.float32), np.empty((0, ACT_SIZE), dtype=np.float32)

    px = df.get('blue_player_1_pos_x', 0).fillna(0).values.astype(np.float32)
    py = df.get('blue_player_1_pos_y', 0).fillna(0).values.astype(np.float32)
    pz = df.get('blue_player_1_pos_z', 0).fillna(0).values.astype(np.float32)
    vx = df.get('blue_player_1_vel_x', 0).fillna(0).values.astype(np.float32)
    vy = df.get('blue_player_1_vel_y', 0).fillna(0).values.astype(np.float32)
    vz = df.get('blue_player_1_vel_z', 0).fillna(0).values.astype(np.float32)
    bx = df.get('ball_pos_x', 0).fillna(0).values.astype(np.float32)
    by = df.get('ball_pos_y', 0).fillna(0).values.astype(np.float32)
    bz = df.get('ball_pos_z', 0).fillna(0).values.astype(np.float32)
    bvx = df.get('ball_vel_x', 0).fillna(0).values.astype(np.float32)
    bvy = df.get('ball_vel_y', 0).fillna(0).values.astype(np.float32)
    bvz = df.get('ball_vel_z', 0).fillna(0).values.astype(np.float32)
    boost = (df.get('blue_player_1_boost', 0).fillna(0).values.astype(np.float32)) / 100.0
    rx = df.get('blue_player_1_rot_x', 0).fillna(0).values.astype(np.float32)
    ry = df.get('blue_player_1_rot_y', 0).fillna(0).values.astype(np.float32)
    rz = df.get('blue_player_1_rot_z', 0).fillna(0).values.astype(np.float32)
    on_ground = (pz < 50).astype(np.float32)
    supersonic = df.get('blue_player_1_supersonic', False).fillna(False).values.astype(np.float32)

    observations = np.column_stack([
        px / 2300.0, py / 2300.0, pz / 2300.0,
        vx / 2300.0, vy / 2300.0, vz / 2300.0,
        bx / 2300.0, by / 2300.0, bz / 2300.0,
        bvx / 2300.0, bvy / 2300.0, bvz / 2300.0,
        boost, on_ground, supersonic,
        rx, ry, rz,
    ]).astype(np.float32)

    throttle = df.get('blue_player_1_throttle', 0).fillna(0).values.astype(np.float32)
    steer = df.get('blue_player_1_steer', 0).fillna(0).values.astype(np.float32)
    boost_active = (df.get('blue_player_1_boost_active', False).fillna(False).values.astype(np.float32))
    jump_active = (df.get('blue_player_1_jump_active', False).fillna(False).values.astype(np.float32))
    dbl_jump = (df.get('blue_player_1_double_jump_active', False).fillna(False).values.astype(np.float32))
    dodge = (df.get('blue_player_1_dodge_active', False).fillna(False).values.astype(np.float32))

    actions = np.column_stack([
        np.clip(throttle, -1, 1),
        np.clip(steer, -1, 1),
        np.clip(throttle, -1, 1),
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        (boost_active > 0.5).astype(np.float32),
        (jump_active > 0.5).astype(np.float32),
        ((dbl_jump > 0.5) | (dodge > 0.5)).astype(np.float32),
    ]).astype(np.float32)

    return observations, actions


def _worker(replay_path):
    os.environ["RAYON_NUM_THREADS"] = "1"
    try:
        obs, acts = extract_from_replay(replay_path)
        return len(obs), obs, acts
    except Exception as e:
        return 0, None, None


def _write_npy_header(f, shape, dtype=np.float32):
    header = f"{{'descr': '{dtype().dtype.char}{dtype().dtype.itemsize}', 'fortran_order': False, 'shape': {shape}, }}"
    header = header.encode('utf-8')
    header_len = len(header)
    magic = b'\x93NUMPY\x01\x00'
    f.write(magic)
    f.write(struct.pack('<H', header_len))
    f.write(header)
    padding = (16 - (magic.nbytes + 2 + header_len) % 16) % 16
    f.write(b'\x00' * padding)


def _append_raw(path, data):
    with open(path, 'ab') as f:
        f.write(data.tobytes())


def _finalize_npy(path, total_rows, shape_per_row, dtype=np.float32):
    raw_path = str(path) + '.raw'
    path_final = str(path)
    with open(path_final, 'wb') as out:
        _write_npy_header(out, (total_rows,) + shape_per_row, dtype)
        with open(raw_path, 'rb') as raw:
            while True:
                chunk = raw.read(1024 * 1024 * 64)
                if not chunk:
                    break
                out.write(chunk)
    os.remove(raw_path)


def process_replay_folder(folder_path, output_path, workers=None, limit=None):
    folder = Path(folder_path)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    replay_files = sorted(folder.glob("*.replay"))
    total = len(replay_files)
    print(f"Found {total} replay files in {folder.name}")

    processed_file = output / "processed_replays.txt"
    processed = set()
    if processed_file.exists():
        raw = processed_file.read_text().strip()
        if raw:
            processed = set(raw.split("\n"))
        print(f"Resuming: {len(processed)} replays already processed")

    remaining = [f for f in replay_files if f.name not in processed]
    if limit:
        remaining = remaining[:limit]
    print(f"Remaining: {len(remaining)} replays to extract")

    if not remaining:
        print("Nothing to do!")
        return

    if workers is None:
        workers = min(cpu_count(), 22)
    print(f"Extracting with {workers} workers...")

    obs_raw = output / "observations.npy.raw"
    acts_raw = output / "actions.npy.raw"
    total_frames = 0
    t_start = time.time()

    with Pool(workers) as pool:
        with open(processed_file, "a") as pf:
            for i, (count, obs, acts) in enumerate(pool.imap_unordered(_worker, remaining)):
                if count > 0:
                    total_frames += count
                    _append_raw(obs_raw, obs)
                    _append_raw(acts_raw, acts)

                pf.write(remaining[i].name + "\n")
                if (i + 1) % 100 == 0:
                    pf.flush()

                elapsed = time.time() - t_start
                fps = total_frames / elapsed if elapsed > 0 else 0
                done = i + 1
                pct = done / len(remaining) * 100

                if done % 50 == 0 or done == len(remaining):
                    print(f"[{done}/{len(remaining)}] {pct:.1f}% | {total_frames:,} frames | {fps:,.0f} frames/s | {elapsed:.0f}s elapsed", flush=True)

    print(f"\nFinalizing npy files ({total_frames:,} frames)...")
    _finalize_npy(output / "observations.npy", total_frames, (OBS_SIZE,))
    _finalize_npy(output / "actions.npy", total_frames, (ACT_SIZE,))

    elapsed = time.time() - t_start
    print(f"Done! {total_frames:,} frames in {elapsed:.0f}s ({total_frames / elapsed:,.0f} frames/s)")
    print(f"Saved to: {output / 'observations.npy'}")
    print(f"Saved to: {output / 'actions.npy'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="replay_files", help="Folder with .replay files")
    parser.add_argument("--output", default="replay_data", help="Output folder for extracted data")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Max replays to extract this run")
    args = parser.parse_args()

    process_replay_folder(args.input, args.output, args.workers, args.limit)
