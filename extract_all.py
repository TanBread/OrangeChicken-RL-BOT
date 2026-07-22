import subprocess
import sys
import os
import argparse
from pathlib import Path

os.environ["RAYON_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=32, help="Number of workers to use (2 per core, 16 core limit)")
args = parser.parse_args()

INPUT_BASE = Path(r"D:\RLReplays")
OUTPUT = Path(__file__).parent / "replay_data"
PROCESSED = OUTPUT / "processed_replays.txt"
WORKERS = args.workers

modes = ["1v1", "2v2", "3v3"]

for mode in modes:
    input_dir = INPUT_BASE / mode
    if not input_dir.exists():
        continue

    all_replays = list(input_dir.glob("*.replay"))
    processed = set()
    if PROCESSED.exists():
        raw = PROCESSED.read_text().strip()
        if raw:
            processed = set(raw.split("\n"))
    remaining = [f for f in all_replays if f.name not in processed]

    if not remaining:
        print(f"{mode}: done")
        continue

    print(f"{mode}: {len(remaining)} replays left", flush=True)

    result = subprocess.run(
        [sys.executable, "extract_replay.py",
         "--input", str(input_dir),
         "--output", str(OUTPUT),
         "--workers", str(WORKERS)],
        env={**os.environ, "RAYON_NUM_THREADS": "1"},
    )

print("All done!")
