import subprocess
import sys
from pathlib import Path

BENCH_CONFIG = Path(__file__).parent / "benchmark_config.txt"

def get_envs():
    if BENCH_CONFIG.exists():
        return BENCH_CONFIG.read_text().strip()
    return "1200"

if __name__ == "__main__":
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy>=2.0.0"])
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "rlgym-rocket-league"])
    subprocess.run([sys.executable, "-m", "pip", "install", "gymnasium>=0.28.0", "torch>=1.12.0"])
    subprocess.run([sys.executable, "-m", "pip", "install", "streamlit>=1.28.0", "pandas>=2.0.0"])

    print("\nSelect mode:")
    print("  1. Train")
    print("  2. Train from Replays + RL")
    print("  3. Download & Extract Replays")
    print("  4. Dashboard")
    print("  5. Watch Bot")
    mode = input("\nMode? [1]: ").strip() or "1"

    if mode == "1":
        print("\nOpponent count:")
        print("  1. 1v1")
        print("  2. 1v2")
        print("  3. 1v3")
        opp = input("\n? [1]: ").strip() or "1"
        orange_size = {"1": 1, "2": 2, "3": 3}.get(opp, 1)

        print("\nTrain by:")
        print("  1. Time")
        print("  2. Games")
        train_type = input("\n? [1]: ").strip() or "1"

        if train_type == "2":
            games = input("How many games? [1000]: ").strip() or "1000"
            print(f"\nStarting RL training ({games} games, 1v{orange_size})...")
            subprocess.run([sys.executable, "-u", "collect.py", "--games", games, "--save-every", "10",
                            "--envs", get_envs(), "--cores", "16", "--batch-steps", "96", "--epochs", "6",
                            "--orange", str(orange_size)])
        else:
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
            print(f"\nStarting RL training ({duration}, 1v{orange_size})...")
            subprocess.run([sys.executable, "-u", "collect.py", "--games", "999999", "--save-every", "1",
                            "--envs", get_envs(), "--cores", "16", "--batch-steps", "96", "--epochs", "6",
                            "--orange", str(orange_size), "--time", str(total_secs)])

    elif mode == "2":
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
                        "--envs", get_envs(), "--cores", "16", "--batch-steps", "96", "--epochs", "6",
                        "--time", str(total_secs)])

    elif mode == "3":
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
        print(f"\nExtracting replays...")
        subprocess.run([sys.executable, "extract_replay.py", "--input", "replay_files", "--output", "replay_data"])

    elif mode == "4":
        print("\nOpening dashboard on http://localhost:8501...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py",
                        "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"])

    elif mode == "5":
        model = input("Model path? [models/rl_best.pt]: ").strip() or "models/rl_best.pt"
        games = input("How many games? [3]: ").strip() or "3"
        fps = input("FPS? [30]: ").strip() or "30"
        print(f"\nWatching bot play ({games} games)...")
        subprocess.run([sys.executable, "watch.py", "--model", model, "--games", games, "--fps", fps])

    print("\nDone! Press Enter to exit.")
    input()
