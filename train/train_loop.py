#!/usr/bin/env python3
"""
AlphaZero-style training loop for Quoridor.

Full loop:
1. Self-play: Generate games using current best model, save to samples/tree_X.qbot
2. Train: Train candidate model on generated data, export to model/candidate.pt
3. Arena: Evaluate candidate vs current_best, promote if candidate wins
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

import torch

from resnet import QuoridorValueNet, train


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_next_iteration(samples_dir: str) -> int:
    """Find the next iteration number based on existing tree files."""
    samples_path = Path(samples_dir)
    if not samples_path.exists():
        return 0

    max_iter = -1
    for f in samples_path.glob("tree_*.qbot"):
        match = re.match(r"tree_(\d+)\.qbot", f.name)
        if match:
            max_iter = max(max_iter, int(match.group(1)))

    return max_iter + 1


def run_selfplay(tree_path: str, model_path: str, num_games: int,
                 simulations: int, num_threads: int,
                 temperature: float = 1.0, temp_drop_ply: int = 30) -> bool:
    """Run self-play games using NN-only MCTS evaluation."""
    qbot_path = get_project_root() / "build" / "qbot"

    if not qbot_path.exists():
        logging.error(f"qbot not found at {qbot_path}")
        return False

    if not model_path or not os.path.exists(model_path):
        logging.error(f"Model file required for self-play: {model_path}")
        return False

    # Ensure output directory exists
    tree_dir = Path(tree_path).parent
    tree_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(qbot_path),
        "--selfplay",
        "-m", model_path,
        "-g", str(num_games),
        "-n", str(simulations),
        "-t", str(num_threads),
        "-s", tree_path,
        "--temperature", str(temperature),
        "--temp-drop", str(temp_drop_ply),
    ]

    logging.info(f"Running: {' '.join(cmd)}")
    logging.info(f"Playing {num_games} self-play games...")

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, _ = proc.communicate(timeout=3600 * 4)  # 4 hour timeout
        if stdout:
            lines = stdout.decode().strip().split('\n')
            for line in lines[-10:]:
                logging.info(f"  {line}")

        return proc.returncode == 0

    except subprocess.TimeoutExpired:
        logging.error("Self-play timed out after 4 hours")
        proc.kill()
        proc.wait()
        return False
    except Exception as e:
        logging.error(f"Error running self-play: {e}")
        return False


def check_tree_has_games(tree_path: str) -> int:
    """Check if tree has completed games using leopard. Returns count."""
    leopard_path = get_project_root() / "build" / "leopard"

    if not leopard_path.exists():
        logging.error(f"leopard not found at {leopard_path}")
        return 0

    try:
        result = subprocess.run(
            [str(leopard_path), tree_path],
            capture_output=True,
            timeout=60
        )

        stderr = result.stderr.decode()
        for line in stderr.split('\n'):
            if "terminal nodes" in line:
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])
        return 0

    except Exception as e:
        logging.error(f"Error checking tree: {e}")
        return 0


def train_model(model: QuoridorValueNet, tree_path: str, epochs: int,
                batch_size: int) -> bool:
    """Train the model on the current tree."""
    logging.info(f"Training model for {epochs} epochs...")

    try:
        train(model, tree_path, batch_size, epochs)
        return True
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return False


def export_model(model: QuoridorValueNet, export_path: str) -> bool:
    """Export model to TorchScript for C++ inference."""
    logging.info(f"Exporting model to {export_path}")

    # Ensure output directory exists
    export_dir = Path(export_path).parent
    export_dir.mkdir(parents=True, exist_ok=True)

    try:
        model.eval()
        model.cpu()

        # New 6-channel input format: (batch, 6, 9, 9)
        # Channels: my_pawn, opp_pawn, h_walls, v_walls, my_fences, opp_fences
        example_input = torch.zeros(1, 6, 9, 9)

        traced = torch.jit.trace(model, example_input)
        traced.save(export_path)

        return True
    except Exception as e:
        logging.error(f"Error exporting model: {e}")
        return False


def save_checkpoint(model: QuoridorValueNet, checkpoint_path: str) -> bool:
    """Save model weights checkpoint."""
    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving checkpoint: {e}")
        return False


def run_arena(current_model: str, candidate_model: str, num_threads: int, num_games: int,
              simulations: int, win_threshold: float = 0.55,
              temperature: float = 1.0, temp_drop_ply: int = 30) -> tuple[bool, bool]:
    """
    Run arena evaluation between candidate and current model.
    Returns (success, candidate_won).
    """
    qbot_path = get_project_root() / "build" / "qbot"

    if not qbot_path.exists():
        logging.error(f"qbot not found at {qbot_path}")
        return False, False

    cmd = [
        str(qbot_path),
        "--arena",
        "-m", current_model,
        "--candidate", candidate_model,
        "-t", str(num_threads),
        "--arena-games", str(num_games),
        "-n", str(simulations),
        "--win-threshold", str(win_threshold),
        "--temperature", str(temperature),
        "--temp-drop", str(temp_drop_ply),
    ]

    logging.info(f"Running arena: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=3600 * 2)
        output = result.stdout.decode() + result.stderr.decode()

        # Log last lines of output
        lines = output.strip().split('\n')
        for line in lines[-15:]:
            logging.info(f"  {line}")

        if result.returncode != 0:
            logging.error("Arena evaluation failed")
            return False, False

        # Check if candidate won by looking for "Candidate wins!" or "promoted"
        candidate_won = "Candidate wins!" in output or "promoted successfully" in output

        return True, candidate_won

    except subprocess.TimeoutExpired:
        logging.error("Arena evaluation timed out")
        return False, False
    except Exception as e:
        logging.error(f"Error running arena: {e}")
        return False, False


def main():
    parser = argparse.ArgumentParser(
        description='AlphaZero-style training loop for Quoridor',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Directories
    parser.add_argument('--samples-dir', type=str, default='samples',
                        dest='samples_dir',
                        help='Directory for tree samples (tree_X.qbot files)')
    parser.add_argument('--model-dir', type=str, default='model',
                        dest='model_dir',
                        help='Directory for models')

    # Training parameters
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of training iterations')
    parser.add_argument('--games', type=int, default=500,
                        help='Self-play games per iteration')
    parser.add_argument('--simulations', type=int, default=800,
                        help='MCTS simulations per move')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=64, dest='batch_size',
                        help='Training batch size')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of threads for self-play')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for move selection')
    parser.add_argument('--temp-drop', type=int, default=30, dest='temp_drop',
                        help='Ply at which to drop temperature to 0')

    # Arena parameters
    parser.add_argument('--arena-games', type=int, default=100, dest='arena_games',
                        help='Number of arena games for evaluation')
    parser.add_argument('--arena-sims', type=int, default=400, dest='arena_sims',
                        help='MCTS simulations per move in arena')
    parser.add_argument('--arena-temperature', type=float, default=0.2,
                        dest='arena_temperature',
                        help='Temperature for arena move selection')
    parser.add_argument('--arena-temp-drop', type=int, default=30,
                        dest='arena_temp_drop',
                        help='Ply at which arena drops temperature to 0')
    parser.add_argument('--win-threshold', type=float, default=0.55,
                        dest='win_threshold',
                        help='Win rate threshold for model promotion')

    # Options
    parser.add_argument('--skip-arena', action='store_true', dest='skip_arena',
                        help='Skip arena evaluation (always promote candidate)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', dest='log_level',
                        help='Logging level')

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    # Resolve paths
    project_root = get_project_root()
    samples_dir = project_root / args.samples_dir
    model_dir = project_root / args.model_dir

    current_best_pt = model_dir / "current_best.pt"
    candidate_pt = model_dir / "candidate.pt"
    current_best_weights = model_dir / "current_best.model"
    candidate_weights = model_dir / "candidate.model"

    # Ensure directories exist
    samples_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 60)
    logging.info("Quoridor AlphaZero Training Loop")
    logging.info("=" * 60)
    logging.info(f"Samples dir:     {samples_dir}")
    logging.info(f"Model dir:       {model_dir}")
    logging.info(f"Current best:    {current_best_pt}")
    logging.info(f"Candidate:       {candidate_pt}")
    logging.info(f"Iterations:      {args.iterations}")
    logging.info(f"Games/iter:      {args.games}")
    logging.info(f"Sims/move:       {args.simulations}")
    logging.info(f"Temperature:     {args.temperature} (drops at ply {args.temp_drop})")
    logging.info(f"Epochs:          {args.epochs} per iteration")
    logging.info(f"Threads:         {args.threads}")
    logging.info(f"Arena games:     {args.arena_games}")
    logging.info(f"Arena temp:      {args.arena_temperature} (drops at ply {args.arena_temp_drop})")
    logging.info(f"Win threshold:   {args.win_threshold * 100}%")
    logging.info("=" * 60)

    # Initialize model
    model = QuoridorValueNet()

    # Load existing best model weights if available
    if current_best_weights.exists():
        logging.info(f"Loading existing model from {current_best_weights}")
        model.load_state_dict(torch.load(current_best_weights))
    elif current_best_pt.exists():
        logging.info("No weights file found, will use existing TorchScript model")
    else:
        logging.info("Starting with fresh random model")

    # Use GPU if available
    if torch.cuda.is_available():
        model.cuda()
        logging.info("Using CUDA")

    # Export initial model if needed
    if not current_best_pt.exists():
        logging.info("Exporting initial model as current_best...")
        if not export_model(model, str(current_best_pt)):
            logging.error("Failed to export initial model")
            return
        save_checkpoint(model, str(current_best_weights))

    # Find starting iteration
    start_iter = get_next_iteration(str(samples_dir))
    logging.info(f"Starting from iteration {start_iter}")

    promotions = 0
    rejections = 0

    for iteration in range(start_iter, start_iter + args.iterations):
        logging.info("")
        logging.info(f"{'=' * 20} ITERATION {iteration} {'=' * 20}")

        tree_path = samples_dir / f"tree_{iteration}.qbot"

        # Phase 1: Self-play
        logging.info(f"[Phase 1] Self-play ({args.games} games)...")
        if not run_selfplay(str(tree_path), str(current_best_pt), args.games,
                            args.simulations, args.threads,
                            args.temperature, args.temp_drop):
            logging.error("Self-play failed, retrying iteration...")
            continue

        # Check if tree has completed games
        num_games = check_tree_has_games(str(tree_path))
        if num_games == 0:
            logging.warning("No completed games in tree, skipping training")
            continue

        logging.info(f"Tree has {num_games} terminal nodes")

        # Derive .qsamples path from tree path
        samples_path = tree_path.with_suffix('.qsamples')
        if not samples_path.exists():
            logging.error(f"Training samples not found at {samples_path}")
            logging.error("Self-play should generate .qsamples files automatically")
            continue

        logging.info(f"Using training samples: {samples_path}")

        # Phase 2: Train candidate model
        logging.info(f"[Phase 2] Training candidate neural network...")

        # Reload current best weights before training
        if current_best_weights.exists():
            model.load_state_dict(torch.load(current_best_weights))
            if torch.cuda.is_available():
                model.cuda()

        if not train_model(model, str(samples_path), args.epochs, args.batch_size):
            logging.error("Training failed, skipping iteration...")
            continue

        # Export candidate
        save_checkpoint(model, str(candidate_weights))
        export_model(model, str(candidate_pt))

        # Phase 3: Arena evaluation
        if args.skip_arena:
            logging.info("[Phase 3] Arena skipped, promoting candidate...")
            candidate_won = True
            arena_success = True
        else:
            logging.info(f"[Phase 3] Arena evaluation ({args.arena_games} games)...")
            arena_success, candidate_won = run_arena(
                str(current_best_pt), str(candidate_pt), args.threads,
                args.arena_games, args.arena_sims, args.win_threshold,
                args.arena_temperature, args.arena_temp_drop
            )

            if not arena_success:
                logging.error("Arena evaluation failed, keeping current model...")
                rejections += 1
                continue

        # Handle result
        if candidate_won:
            promotions += 1
            logging.info(f"Candidate PROMOTED! (total promotions: {promotions})")
            # Copy candidate to current_best
            shutil.copy(str(candidate_pt), str(current_best_pt))
            shutil.copy(str(candidate_weights), str(current_best_weights))
        else:
            rejections += 1
            logging.info(f"Candidate rejected. (total rejections: {rejections})")
            # Reload current best weights for next iteration
            if current_best_weights.exists():
                model.load_state_dict(torch.load(current_best_weights))
                if torch.cuda.is_available():
                    model.cuda()

        logging.info(f"Iteration {iteration} complete. "
                     f"Promotions: {promotions}, Rejections: {rejections}")

    # Final summary
    logging.info("")
    logging.info("=" * 60)
    logging.info("Training complete!")
    logging.info(f"  Total promotions: {promotions}")
    logging.info(f"  Total rejections: {rejections}")
    logging.info(f"  Final model: {current_best_pt}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
