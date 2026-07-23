import os
import sys
import time
import json
import requests
from pathlib import Path


API_BASE = "https://ballchasing.com/api"


class BallchasingAPI:
    def __init__(self, token):
        self.token = token
        self.headers = {"Authorization": token}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def ping(self):
        r = self.session.get(f"{API_BASE}/")
        if r.status_code == 200:
            data = r.json()
            print(f"Connected! Account type: {data.get('type', 'unknown')}")
            return True
        elif r.status_code == 401:
            print("Invalid API token!")
            return False
        return False

    def list_replays(self, playlist=None, min_rank=None, max_rank=None,
                     count=200, sort_by="upload-date", sort_dir="desc"):
        params = {
            "count": min(count, 200),
            "sort-by": "replay-date",
            "sort-dir": sort_dir,
        }
        if playlist:
            params["playlist"] = playlist
        if min_rank:
            params["min-rank"] = min_rank
        if max_rank:
            params["max-rank"] = max_rank

        r = self.session.get(f"{API_BASE}/replays", params=params)
        r.raise_for_status()
        return r.json()

    def get_replay(self, replay_id):
        r = self.session.get(f"{API_BASE}/replays/{replay_id}")
        r.raise_for_status()
        return r.json()

    def download_replay(self, replay_id, output_path):
        r = self.session.get(f"{API_BASE}/replays/{replay_id}/file", stream=True)
        r.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True


def download_replays(token, output_dir="replay_files", count=50,
                     playlist="ranked-standard", min_rank=None, max_rank=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    api = BallchasingAPI(token)
    if not api.ping():
        return

    print(f"\nSearching for replays...")
    print(f"  Playlist: {playlist or 'any'}")
    print(f"  Min rank: {min_rank or 'any'}")
    print(f"  Max rank: {max_rank or 'any'}")
    print(f"  Count: {count}")

    replays_data = api.list_replays(
        playlist=playlist,
        min_rank=min_rank,
        max_rank=max_rank,
        count=min(count, 200),
    )

    replays = replays_data.get("list", [])
    print(f"\nFound {len(replays)} replays")

    downloaded = 0
    for i, replay in enumerate(replays):
        replay_id = replay["id"]
        title = replay.get("replay_title", "untitled")
        filename = f"{replay_id}.replay"
        filepath = output / filename

        if filepath.exists():
            print(f"  [{i+1}/{len(replays)}] Skip (exists): {title[:50]}")
            downloaded += 1
            continue

        try:
            print(f"  [{i+1}/{len(replays)}] Downloading: {title[:50]}...", end=" ", flush=True)
            api.download_replay(replay_id, filepath)
            print("OK")
            downloaded += 1
            time.sleep(1.0)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print("Rate limited, waiting 30s...")
                time.sleep(30)
            else:
                print(f"Error: {e}")
                time.sleep(2)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)

    print(f"\nDone! {downloaded}/{len(replays)} replays saved to {output}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="Ballchasing API token")
    parser.add_argument("--output", default="replay_files", help="Output folder")
    parser.add_argument("--count", type=int, default=50, help="Number of replays to download")
    parser.add_argument("--playlist", default="ranked-standard",
                        choices=["ranked-duels", "ranked-doubles", "ranked-standard",
                                 "unranked-duels", "unranked-doubles", "unranked-standard"],
                        help="Playlist to filter by")
    parser.add_argument("--min-rank", default=None,
                        choices=["bronze-1", "silver-1", "gold-1", "platinum-1",
                                 "diamond-1", "champion-1", "grand-champion"],
                        help="Minimum rank filter")
    parser.add_argument("--max-rank", default=None,
                        choices=["silver-3", "gold-3", "platinum-3", "diamond-3",
                                 "champion-3", "grand-champion"],
                        help="Maximum rank filter")
    args = parser.parse_args()

    download_replays(
        token=args.token,
        output_dir=args.output,
        count=args.count,
        playlist=args.playlist,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
    )
