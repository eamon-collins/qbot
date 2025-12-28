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
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

from resnet import QuoridorValueNet, train


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def run_qbot(tree_path: str, model_path: str | None, duration_seconds: int,
             num_threads: int) -> bool:
    """Run qbot to build tree for a specified duration."""
    qbot_path = get_project_root() / "build" / "qbot"

    if not qbot_path.exists():
        logging.error(f"qbot not found at {qbot_path}")
        return False

    cmd = [
        str(qbot_path),
        "-b",  # Training mode
        "-t", str(num_threads),
        "-s", tree_path,
    ]

    # Load existing tree if it exists
    if os.path.exists(tree_path):
        cmd.extend(["-l", tree_path])

    # Use model if provided
    if model_path and os.path.exists(model_path):
        cmd.extend(["-m", model_path])

    logging.info(f"Running: {' '.join(cmd)}")
    logging.info(f"Building tree for {duration_seconds} seconds...")

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Wait for duration then send SIGINT for graceful shutdown
        time.sleep(duration_seconds)
        proc.send_signal(signal.SIGINT)

        # Wait for graceful shutdown (with timeout)
        try:
            stdout, _ = proc.communicate(timeout=30)
            if stdout:
                # Print last few lines of output
                lines = stdout.decode().strip().split('\n')
                for line in lines[-5:]:
                    logging.info(f"  {line}")
        except subprocess.TimeoutExpired:
            logging.warning("qbot didn't shut down gracefully, killing...")
            proc.kill()
            proc.wait()

        return proc.returncode == 0 or proc.returncode == -2  # -2 is SIGINT

    except Exception as e:
        logging.error(f"Error running qbot: {e}")
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
                        help='Number of train/build iterations')
    parser.add_argument('--tree-time', type=int, default=300, dest='tree_time',
                        help='Seconds to build tree per iteration')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=64, dest='batch_size',
                        help='Training batch size')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of threads for tree building')

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
    logging.info("Quoridor AlphaZero Training Loop")
    logging.info("=" * 60)
    logging.info(f"Tree file:      {tree_path}")
    logging.info(f"Model file:     {model_path}")
    logging.info(f"Export file:    {export_path}")
    logging.info(f"Iterations:     {args.iterations}")
    logging.info(f"Tree time:      {args.tree_time}s per iteration")
    logging.info(f"Epochs:         {args.epochs} per iteration")
    logging.info(f"Threads:        {args.threads}")
    logging.info("=" * 60)

    for iteration in range(args.iterations):
        logging.info("")
        logging.info(f"{'=' * 20} ITERATION {iteration + 1}/{args.iterations} {'=' * 20}")

        # Phase 1: Build tree (skip on first iteration if requested)
        if iteration == 0 and args.skip_initial_build:
            logging.info("Skipping initial tree build")
        else:
            # Export model for C++ before tree building (if we have trained it)
            if iteration > 0 or os.path.exists(args.model):
                if not export_model(model, export_path):
                    logging.warning("Failed to export model, building without NN")
                    export_path_for_qbot = None
                else:
                    export_path_for_qbot = export_path
            else:
                export_path_for_qbot = None

            logging.info(f"[Phase 1] Building tree...")
            if not run_qbot(tree_path, export_path_for_qbot, args.tree_time, args.threads):
                logging.error("Tree building failed")
                continue

        # Check if tree has completed games
        num_games = check_tree_has_games(tree_path)
        if num_games == 0:
            logging.warning("No completed games in tree, skipping training")
            logging.info("Try running longer or with more depth-first exploration")
            continue

        logging.info(f"Tree has {num_games} completed games")

        # Phase 2: Train neural network
        logging.info(f"[Phase 2] Training neural network...")
        if not train_model(model, tree_path, args.epochs, args.batch_size):
            logging.error("Training failed")
            continue

        # Save checkpoint
        save_checkpoint(model, model_path)

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
