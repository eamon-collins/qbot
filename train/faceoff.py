#Faceoff: Arena tournament for Quoridor models.
#Runs round-robin matches between models and reports results, highlighting upsets.

import argparse
import logging
import os
import subprocess
import sys
import itertools
from pathlib import Path
from datetime import datetime

def get_project_root():
    return Path(__file__).parent.parent

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def get_model_timestamp(path: Path) -> float:
    return path.stat().st_mtime

def find_recent_models(model_dir: Path, n: int = 3) -> list[Path]:
    """Find the n most recent best_*.pt models."""
    if not model_dir.exists():
        return []

    files = list(model_dir.glob("best_*.pt"))
    # Sort by mtime descending (newest first)
    files.sort(key=get_model_timestamp, reverse=True)
    return files[:n]

def run_match(qbot_path: Path, model_older: Path, model_newer: Path, 
              games: int, sims: int, threads: int, 
              temp: float, temp_drop: int, max_memory: int) -> dict:
    """
    Run a match between two models.
    model_older is passed as '-m' (Current/Defender).
    model_newer is passed as '--candidate' (Challenger).
    """

    cmd = [
        str(qbot_path),
        "--arena",
        "-m", str(model_older),
        "--candidate", str(model_newer),
        "-t", str(threads),
        "--arena-games", str(games),
        "-n", str(sims),
        "--temperature", str(temp),
        "--temp-drop", str(temp_drop),
        "--max-memory", str(max_memory),
    ]

    print(f"\n{'='*80}")
    print(f"MATCH: {model_older.name} (Old) vs {model_newer.name} (New)")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")

    result = {
        "model_older": model_older,
        "model_newer": model_newer,
        "winner": None,
        "is_upset": False,
        "output_lines": []
    }

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True
        )

        # Stream output to console and capture it
        for line in proc.stdout:
            print(line, end='') 
            result["output_lines"].append(line.strip())

        proc.wait()

        full_output = "\n".join(result["output_lines"])

        # Determine winner
        # qbot outputs "Candidate wins!" if the candidate (newer) performs significantly better.
        # If the candidate fails to win significantly (or loses), the older model 'retains the title'.

        if "Candidate wins!" in full_output or "promoted successfully" in full_output:
            # Newer model won (Expected result)
            result["winner"] = model_newer
            result["is_upset"] = False
        else:
            # Older model won or held draw (Upset / Failure to improve)
            result["winner"] = model_older
            result["is_upset"] = True

    except Exception as e:
        print(f"Error running match: {e}")

    return result

def main():
    parser = argparse.ArgumentParser(description="Run round-robin faceoff between models")
    parser.add_argument('models', metavar='MODEL', type=str, nargs='*',
                        help='Paths to .pt models to test. If empty, uses 3 most recent best_ models.')
    parser.add_argument('--games', type=int, default=100, help='Games per match')
    parser.add_argument('--sims', type=int, default=500, help='MCTS simulations per move')
    parser.add_argument('--threads', type=int, default=20, help='Number of threads')
    parser.add_argument('--temp', type=float, default=0.1, help='Temperature')
    parser.add_argument('--temp-drop', type=int, default=10, help='Ply to drop temperature')
    parser.add_argument('--max-memory', type=int, default=30, help='Max memory in GB')

    args = parser.parse_args()
    setup_logging()

    root = get_project_root()
    qbot_path = root / "build" / "qbot"
    model_dir = root / "model"

    if not qbot_path.exists():
        print(f"Error: qbot binary not found at {qbot_path}")
        sys.exit(1)

    # 1. Select Models
    models = []
    if args.models:
        for m in args.models:
            p = Path(m).resolve()
            if not p.exists():
                print(f"Error: Model {p} not found")
                sys.exit(1)
            models.append(p)
        # Sort manually provided models by time as well to determine 'old' vs 'new'
        models.sort(key=get_model_timestamp)
    else:
        print("No models specified. Finding 3 most recent 'best_*.pt' models in model/...")
        models = find_recent_models(model_dir, n=3)
        # find_recent_models returns Newest -> Oldest. 
        # We reverse it to get Oldest -> Newest for listing
        models.reverse()

        if len(models) < 2:
            print("Error: Need at least 2 models to run a faceoff.")
            print(f"Found {len(models)} in {model_dir}")
            sys.exit(1)

    print(f"\nTournament Roster (Ordered by Age: Oldest -> Newest):")
    for i, m in enumerate(models):
        ts = datetime.fromtimestamp(get_model_timestamp(m)).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {i+1}. {m.name}  (Modified: {ts})")

    # 2. Run Round Robin
    results = []

    # Generate all unique pairs
    pairs = list(itertools.combinations(models, 2))
    print(f"\nScheduled {len(pairs)} matches.\n")

    for m1, m2 in pairs:
        # Determine chronological order for this specific pair
        t1 = get_model_timestamp(m1)
        t2 = get_model_timestamp(m2)

        # Ensure m_older is the one with the smaller timestamp
        if t1 < t2:
            m_older, m_newer = m1, m2
        else:
            m_older, m_newer = m2, m1

        match_res = run_match(qbot_path, m_older, m_newer, 
                              args.games, args.sims, args.threads, 
                              args.temp, args.temp_drop, args.max_memory)
        results.append(match_res)

    # 3. Final Summary
    print(f"\n\n{'#'*80}")
    print("TOURNAMENT RESULTS SUMMARY")
    print(f"{'#'*80}")

    upset_count = 0

    for res in results:
        m_old = res["model_older"]
        m_new = res["model_newer"]
        winner = res["winner"]
        is_upset = res["is_upset"]

        if is_upset:
            upset_count += 1
            status = "\033[91m*** UPSET ***\033[0m" # Red text if terminal supports it
        else:
            status = "\033[92mExpected\033[0m"      # Green text

        print(f"{m_old.name} (old) vs {m_new.name} (new) -> Winner: {winner.name}  [{status}]")

    print(f"\nTotal Matches: {len(results)}")
    print(f"Total Upsets:  {upset_count}")
    print(f"{'#'*80}\n")

if __name__ == "__main__":
    main()
