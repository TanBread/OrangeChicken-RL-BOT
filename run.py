import subprocess
import sys

if __name__ == "__main__":
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy>=2.0.0"])
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "rlgym-rocket-league"])
    subprocess.run([sys.executable, "-m", "pip", "install", "gymnasium>=0.28.0", "torch>=1.12.0"])
    subprocess.run([sys.executable, "-m", "pip", "install", "streamlit>=1.28.0", "pandas>=2.0.0"])

    print("\nSelect mode:")
    print("  1. Train")
    print("  2. Dashboard")
    print("  3. Train + Dashboard")
    mode = input("\nMode? [3]: ").strip() or "3"

    if mode == "1":
        games = input("How many games? [100]: ").strip() or "100"
        save_every = input("Save every N games? [10]: ").strip() or "10"
        print(f"\nStarting RL training ({games} games)...")
        subprocess.run([sys.executable, "-u", "collect.py", "--games", games, "--save-every", save_every])

    elif mode == "2":
        print("\nOpening dashboard on http://localhost:8501...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py",
                        "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"])

    else:
        games = input("How many games? [100]: ").strip() or "100"
        save_every = input("Save every N games? [10]: ").strip() or "10"

        import threading

        def start_dashboard():
            subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py",
                            "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"])

        t = threading.Thread(target=start_dashboard, daemon=True)
        t.start()
        print("\nDashboard on http://localhost:8501")
        print(f"Starting RL training ({games} games)...\n")
        subprocess.run([sys.executable, "-u", "collect.py", "--games", games, "--save-every", save_every])

    print("\nDone! Press Enter to exit.")
    input()
