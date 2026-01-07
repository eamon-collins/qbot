#!/usr/bin/env python3
"""
AlphaZero-style training loop for Quoridor.

Alternates between:
1. Training neural network on completed games in the tree
2. Building more tree using the trained neural network
"""

import argparse
import logging
import os
import subprocess
from pathlib import Path

import torch

from resnet import QuoridorValueNet, train


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


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

    # Load existing tree if it exists
    if os.path.exists(tree_path):
        cmd.extend(["-l", tree_path])

    logging.info(f"Running: {' '.join(cmd)}")
    logging.info(f"Playing {num_games} self-play games...")

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Self-play runs to completion
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

        # Parse stderr for terminal node count
        stderr = result.stderr.decode()
        for line in stderr.split('\n'):
            if "terminal nodes" in line:
                # "Found X terminal nodes (completed games)"
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

    try:
        model.eval()
        model.cpu()  # Ensure model is on CPU for export

        # Create example inputs for tracing
        # meta has 3 elements: p1_fences, p2_fences, turn_indicator
        example_pawn = torch.zeros(1, 2, 9, 9)
        example_wall = torch.zeros(1, 2, 8, 8)
        example_meta = torch.zeros(1, 3)

        traced = torch.jit.trace(model, (example_pawn, example_wall, example_meta))
        traced.save(export_path)

        return True
    except Exception as e:
        logging.error(f"Error exporting model: {e}")
        return False


def save_checkpoint(model: QuoridorValueNet, checkpoint_path: str) -> bool:
    """Save model weights checkpoint."""
    try:
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving checkpoint: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='AlphaZero-style training loop for Quoridor',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Paths
    parser.add_argument('--tree', type=str, default='tree.qbot',
                        help='Path to tree file')
    parser.add_argument('--model', type=str, default='treestyle.model',
                        help='Path to model weights file')
    parser.add_argument('--export-model', type=str, default='treestyle.pt',
                        dest='export_model',
                        help='Path to exported TorchScript model')

    # Training parameters
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of train/selfplay iterations')
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

    # Options
    parser.add_argument('--skip-initial-build', action='store_true',
                        dest='skip_initial_build',
                        help='Skip initial tree building (use existing tree)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', dest='log_level',
                        help='Logging level')

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    # Initialize model
    model = QuoridorValueNet()

    # Load existing model if available
    if os.path.exists(args.model):
        logging.info(f"Loading existing model from {args.model}")
        model.load_state_dict(torch.load(args.model))

    # Use GPU if available
    if torch.cuda.is_available():
        model.cuda()
        logging.info("Using CUDA")

    # Resolve paths
    tree_path = os.path.abspath(args.tree)
    model_path = os.path.abspath(args.model)
    export_path = os.path.abspath(args.export_model)

    logging.info("=" * 60)
    logging.info("Quoridor AlphaZero Training Loop (Self-Play)")
    logging.info("=" * 60)
    logging.info(f"Tree file:      {tree_path}")
    logging.info(f"Model file:     {model_path}")
    logging.info(f"Export file:    {export_path}")
    logging.info(f"Iterations:     {args.iterations}")
    logging.info(f"Games/iter:     {args.games}")
    logging.info(f"Sims/move:      {args.simulations}")
    logging.info(f"Temperature:    {args.temperature} (drops at ply {args.temp_drop})")
    logging.info(f"Epochs:         {args.epochs} per iteration")
    logging.info(f"Threads:        {args.threads}")
    logging.info("=" * 60)

    # Export initial model (required for self-play)
    if not os.path.exists(export_path):
        logging.info("Exporting initial model for self-play...")
        if not export_model(model, export_path):
            logging.error("Failed to export initial model")
            return

    for iteration in range(args.iterations):
        logging.info("")
        logging.info(f"{'=' * 20} ITERATION {iteration + 1}/{args.iterations} {'=' * 20}")

        # Phase 1: Self-play (skip on first iteration if requested and tree exists)
        if iteration == 0 and args.skip_initial_build and os.path.exists(tree_path):
            logging.info("Skipping initial self-play (using existing tree)")
        else:
            logging.info(f"[Phase 1] Self-play ({args.games} games)...")
            if not run_selfplay(tree_path, export_path, args.games, args.simulations,
                                args.threads, args.temperature, args.temp_drop):
                logging.error("Self-play failed")
                continue

        # Check if tree has completed games
        num_games = check_tree_has_games(tree_path)
        if num_games == 0:
            logging.warning("No completed games in tree, skipping training")
            continue

        logging.info(f"Tree has {num_games} terminal nodes (completed games)")

        # Phase 2: Train neural network
        logging.info(f"[Phase 2] Training neural network...")
        if not train_model(model, tree_path, args.epochs, args.batch_size):
            logging.error("Training failed")
            continue

        # Save checkpoint and export updated model for next iteration
        save_checkpoint(model, model_path)
        export_model(model, export_path)

        logging.info(f"Iteration {iteration + 1} complete")

    # Final export
    logging.info("")
    logging.info("=" * 60)
    logging.info("Training complete!")
    export_model(model, export_path)
    save_checkpoint(model, model_path)
    logging.info(f"Final model saved to {model_path}")
    logging.info(f"TorchScript model saved to {export_path}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
