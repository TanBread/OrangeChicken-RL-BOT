import subprocess
import sys
from pathlib import Path

BENCH_CONFIG = Path(__file__).parent / "benchmark_config.txt"

def get_envs():
    if BENCH_CONFIG.exists():
        return BENCH_CONFIG.read_text().strip()
    return "2400"

if __name__ == "__main__":
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy>=2.0.0"])
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "rlgym-rocket-league"])
    subprocess.run([sys.executable, "-m", "pip", "install", "gymnasium>=0.28.0", "torch>=1.12.0"])
    subprocess.run([sys.executable, "-m", "pip", "install", "streamlit>=1.28.0", "pandas>=2.0.0"])

    print("\nSelect mode:")
    print("  1. Train")
    print("  2. Dashboard")
    print("  3. Watch Bot")
    print("  4. Download Replays (Ballchasing)")
    print("  5. Extract Replays")
    print("  6. Train from Replays (Behavioral Cloning)")
    print("  7. Train from Replays + RL (fine-tune)")
    mode = input("\nMode? [1]: ").strip() or "1"

    if mode == "1":
        duration = input("Training time? [30m]: ").strip() or "30m"
        total_secs = 0
        if "h" in duration:
            parts = duration.split("h")
            total_secs += int(parts[0]) * 3600
            if parts[1]:
                total_secs += int(parts[1].replace("m", "").replace("s", "")) * 60
        elif "m" in duration:
            total_secs += int(duration.replace("m", "")) * 60
        elif "s" in duration:
            total_secs += int(duration.replace("s", ""))
        else:
            total_secs = int(duration) * 60
        print(f"\nStarting RL training ({duration})...")
        subprocess.run([sys.executable, "-u", "collect.py", "--games", "999999", "--save-every", "1",
                        "--envs", get_envs(), "--cores", "16", "--batch-steps", "64", "--epochs", "1",
                        "--time", str(total_secs)])

    elif mode == "2":
        print("\nOpening dashboard on http://localhost:8501...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py",
                        "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"])

    elif mode == "3":
        model = input("Model path? [models/rl_best.pt]: ").strip() or "models/rl_best.pt"
        games = input("How many games? [3]: ").strip() or "3"
        fps = input("FPS? [30]: ").strip() or "30"
        print(f"\nWatching bot play ({games} games)...")
        subprocess.run([sys.executable, "watch.py", "--model", model, "--games", games, "--fps", fps])

    elif mode == "4":
        token = input("Ballchasing API token: ").strip()
        count = input("How many replays? [200]: ").strip() or "200"
        print("\nSelect minimum rank:")
        print("  1. Diamond 1 (D1)")
        print("  2. Diamond 3 (D3)")
        print("  3. Champion 1 (C1)")
        print("  4. Champion 3 (C3)")
        print("  5. Grand Champion 1 (GC1)")
        print("  6. Grand Champion 3 (GC3)")
        print("  7. Supersonic Legend (SSL)")
        rank_choice = input("\nRank? [5]: ").strip() or "5"
        rank_map = {
            "1": "diamond-1", "2": "diamond-3",
            "3": "champion-1", "4": "champion-3",
            "5": "grand-champion", "6": "grand-champion",
            "7": "grand-champion",
        }
        min_rank = rank_map.get(rank_choice, "grand-champion")
        print(f"\nDownloading 1v1 replays from Ballchasing...")
        subprocess.run([sys.executable, "download_replays.py", "--token", token,
                        "--playlist", "ranked-duels", "--count", count,
                        "--min-rank", min_rank])

    elif mode == "5":
        replay_dir = input("Replay folder? [replay_files]: ").strip() or "replay_files"
        output_dir = input("Output folder? [replay_data]: ").strip() or "replay_data"
        print(f"\nExtracting replays from {replay_dir}...")
        subprocess.run([sys.executable, "extract_replay.py", "--input", replay_dir, "--output", output_dir])

    elif mode == "6":
        data_dir = input("Data folder? [replay_data]: ").strip() or "replay_data"
        epochs = input("Epochs? [100]: ").strip() or "100"
        print(f"\nTraining behavioral cloning ({epochs} epochs)...")
        subprocess.run([sys.executable, "-u", "train_bc.py", "--data", data_dir, "--epochs", epochs])

    elif mode == "7":
        data_dir = input("Data folder? [replay_data]: ").strip() or "replay_data"
        epochs = input("BC epochs? [100]: ").strip() or "100"
        print(f"\nStep 1: Behavioral cloning ({epochs} epochs)...")
        subprocess.run([sys.executable, "-u", "train_bc.py", "--data", data_dir, "--epochs", epochs,
                        "--model-name", "rl_bc"])
        print(f"\nStep 2: Fine-tuning with RL...")
        duration = input("RL time? [30m]: ").strip() or "30m"
        total_secs = 0
        if "h" in duration:
            parts = duration.split("h")
            total_secs += int(parts[0]) * 3600
            if parts[1]:
                total_secs += int(parts[1].replace("m", "").replace("s", "")) * 60
        elif "m" in duration:
            total_secs += int(duration.replace("m", "")) * 60
        elif "s" in duration:
            total_secs += int(duration.replace("s", ""))
        else:
            total_secs = int(duration) * 60
        subprocess.run([sys.executable, "-u", "collect.py", "--games", "999999", "--save-every", "1",
                        "--envs", get_envs(), "--cores", "16", "--batch-steps", "64", "--epochs", "1",
                        "--time", str(total_secs)])

    print("\nDone! Press Enter to exit.")
    input()
