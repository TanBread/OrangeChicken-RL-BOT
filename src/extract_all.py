import subprocess
import sys
import os
import argparse
from pathlib import Path

os.environ["RAYON_NUM_THREADS"] = "1"

PROJECT_DIR = Path(__file__).parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--input", default=str(PROJECT_DIR / "replay_files"), help="Base folder with replay subfolders (1v1, 2v2, 3v3)")
parser.add_argument("--output", default=str(PROJECT_DIR / "replay_data"), help="Output folder for extracted data")
parser.add_argument("--workers", type=int, default=16, help="Number of workers")
args = parser.parse_args()

INPUT_BASE = Path(args.input)
OUTPUT = Path(args.output)
PROCESSED = OUTPUT / "processed_replays.txt"
WORKERS = args.workers
SRC_DIR = Path(__file__).parent

modes = ["1v1", "2v2", "3v3"]

for mode in modes:
    input_dir = INPUT_BASE / mode
    if not input_dir.exists():
        print(f"{mode}: folder not found, skipping")
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
        [sys.executable, str(SRC_DIR / "extract_replay.py"),
         "--input", str(input_dir),
         "--output", str(OUTPUT),
         "--workers", str(WORKERS)],
        env={**os.environ, "RAYON_NUM_THREADS": "1"},
    )

print("All done!")
