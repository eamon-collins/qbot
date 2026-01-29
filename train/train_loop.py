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
import gc
import copy
import ctypes
import shutil
import psutil
import subprocess
import sys
import traceback
import json
from datetime import datetime
from pathlib import Path

import torch

from resnet import QuoridorNet, train
from model_utils import (
    compute_model_hash,
    find_samples_for_model,
    sync_samples_from_remote,
    push_model_to_remote,
    update_symlink
)


def setup_logging(log_level: str) -> Path:
    """Set up logging with both console and file handlers. Returns log file path."""
    project_root = Path(__file__).parent.parent
    log_dir = project_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # File handler - captures everything
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    return log_file

def log_memory(label):
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / (1024**3)
    logging.info(f"[MEMORY] {label}: {mem_gb:.2f} GB")

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_next_iteration(samples_dir: str) -> int:
    """Find the next iteration number based on existing tree files."""
    samples_path = Path(samples_dir)
    if not samples_path.exists():
        return 0

    max_iter = -1
    for f in samples_path.glob("tree_*.qsamples"):
        match = re.search(r'_(\d+)_[a-f0-9]+\.qsamples$', f.name)
        if match:
            max_iter = max(max_iter, int(match.group(1)))

    return max_iter + 1


def run_selfplay(tree_path: str, model_path: str, num_games: int,
                 simulations: int, num_threads: int, games_per_thread: int,
                 temperature: float = 1.0, temp_drop_ply: int = 30,
                 max_memory: int = 30, batch_size: int = 256, max_wait: float = 1.0, model_id: str = "") -> bool:
    """Run self-play games using NN-only MCTS evaluation."""
    qbot_path = get_project_root() / "build" / "qbot"

    if not qbot_path.exists():
        logging.error(f"qbot not found at {qbot_path}")
        return False

    # Check if model_path is a valid file (resolve symlink if needed)
    resolved_model_path = Path(model_path).resolve()
    if not resolved_model_path.exists():
        logging.error(f"Model file required for self-play: {model_path} (resolved: {resolved_model_path})")
        return False

    # Ensure output directory exists
    tree_dir = Path(tree_path).parent
    tree_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(qbot_path),
        "--selfplay",
        "-m", str(resolved_model_path),
        "-g", str(num_games),
        "-n", str(simulations),
        "-t", str(num_threads),
        "-s", tree_path,
        "--batch-size", str(batch_size),
        "--max-wait", str(max_wait),
        "--temperature", str(temperature),
        "--temp-drop", str(temp_drop_ply),
        "--max-memory", str(max_memory),
        "--games-per-thread", str(games_per_thread),
    ]

    # Add model-id if provided
    if model_id:
        cmd.extend(["--model-id", model_id])

    logging.info(f"Running: {' '.join(cmd)}")
    logging.info(f"Playing {num_games} self-play games...")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True
        )

        # Stream output line by line
        for line in proc.stdout:
            line = line.rstrip('\n')
            if line:
                logging.info(f"  {line}")

        proc.wait(timeout=3600 * 4)  # 4 hour timeout
        return proc.returncode == 0

    except subprocess.TimeoutExpired:
        logging.error("Self-play timed out after 4 hours")
        proc.kill()
        proc.wait()
        return False
    except Exception as e:
        logging.error(f"Error running self-play: {e}")
        return False


def train_model(model: QuoridorNet, training_files: list[str], epochs: int,
                batch_size: int, max_samples: int = 2000000) -> bool:
    """
    Train the model on one or more training files.
    """
    if not training_files:
        logging.error("No training files provided")
        return False

    logging.info(f"Training model for {epochs} epochs...")

    try:
        # Pass all files to train() - it will concatenate them into a single dataset
        train(model, training_files, batch_size, epochs, max_samples)
        return True
    except Exception as e:
        traceback.print_exc()
        logging.error(f"Error training model: {e}")
        return False


def export_model(model: QuoridorNet, export_path: str) -> bool:
    """Export model to TorchScript for C++ inference."""
    logging.info(f"Exporting model to {export_path}")

    # Ensure output directory exists
    export_dir = Path(export_path).parent
    export_dir.mkdir(parents=True, exist_ok=True)

    try:
        model.eval()
        model.cpu()

        # New 6-channel input format: (batch, 6, 9, 9)
        export_model = copy.deepcopy(model)
        export_model.eval()
        export_model.half()
        if torch.cuda.is_available():
            export_model.cuda()
            example_input = torch.zeros(1, 6, 9, 9).half().cuda()
        else:
            example_input = torch.zeros(1, 6, 9, 9).half()
        traced = torch.jit.trace(export_model, example_input)
        traced.save(export_path)

        return True
    except Exception as e:
        logging.error(f"Error exporting model: {e}")
        return False


def save_checkpoint(model: QuoridorNet, checkpoint_path: str) -> bool:
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
              simulations: int, promote_alpha: float = 0.07,
              temperature: float = 1.0, temp_drop_ply: int = 30, max_memory: int = 30) -> tuple[bool, bool]:
    """
    Run arena evaluation between candidate and current model.
    Returns (success, candidate_won).
    """
    qbot_path = get_project_root() / "build" / "qbot"

    if not qbot_path.exists():
        logging.error(f"qbot not found at {qbot_path}")
        return False, False

    # Resolve symlinks for qbot
    current_resolved = Path(current_model).resolve()
    candidate_resolved = Path(candidate_model).resolve()

    cmd = [
        str(qbot_path),
        "--arena",
        "-m", str(current_resolved),
        "--candidate", str(candidate_resolved),
        #10x as arena doesn't have multiple games per worker and also frees memory as it goes
        "-t", str(10*num_threads),
        "--arena-games", str(num_games),
        "-n", str(simulations),
        "--promote-alpha", str(promote_alpha),
        "--temperature", str(temperature),
        "--temp-drop", str(temp_drop_ply),
        "--max-memory", str(max_memory),
    ]

    logging.info(f"Running arena: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True
        )

        # Stream output line by line, collect for result checking
        output_lines = []
        for line in proc.stdout:
            line = line.rstrip('\n')
            if line:
                logging.info(f"  {line}")
                output_lines.append(line)

        proc.wait(timeout=3600 * 2)  # 2 hour timeout

        if proc.returncode != 0:
            logging.error("Arena evaluation failed")
            return False, False

        # Check if candidate won by looking for "Candidate wins!" or "promoted"
        output = '\n'.join(output_lines)
        candidate_won = "Candidate wins!" in output or "promoted successfully" in output

        return True, candidate_won

    except subprocess.TimeoutExpired:
        logging.error("Arena evaluation timed out")
        proc.kill()
        proc.wait()
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
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of training iterations')
    parser.add_argument('--games', type=int, default=500,
                        help='Self-play games per iteration')
    parser.add_argument('--simulations', type=int, default=500,
                        help='MCTS simulations per move')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=1024, dest='batch_size',
                        help='Training batch size')
    parser.add_argument('--inference-batch-size', type=int, default=256, dest='inference_batch_size',
                        help='Inference batch size')
    parser.add_argument('--max-wait', type=float, default=1.0, dest='max_wait',
                        help='Inference batch size')
    parser.add_argument('--threads', type=int, default=20,
                        help='Number of threads for self-play')
    parser.add_argument('--games-per-thread', type=int, default=16,
                        help='Number of games per thread for self-play')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for move selection')
    parser.add_argument('--temp-drop', type=int, default=30, dest='temp_drop',
                        help='Ply at which to drop temperature to 0')
    parser.add_argument('--max-memory', type=int, default=35, dest='max_memory',
                        help='max memory in GB, resets pool at 80%')

    # Arena parameters
    parser.add_argument('--arena-games', type=int, default=200, dest='arena_games',
                        help='Number of arena games for evaluation')
    parser.add_argument('--arena-sims', type=int, default=500, dest='arena_sims',
                        help='MCTS simulations per move in arena')
    parser.add_argument('--arena-temperature', type=float, default=0.0,
                        dest='arena_temperature',
                        help='Temperature for arena move selection')
    parser.add_argument('--arena-temp-drop', type=int, default=30,
                        dest='arena_temp_drop',
                        help='Ply at which arena drops temperature to 0')
    parser.add_argument('--win-threshold', type=float, default=0.55,
                        dest='win_threshold',
                        help='Win rate threshold for model promotion')
    parser.add_argument('--promote-alpha', type=float, default=0.07,dest='promote_alpha',
                        help='p value threshold for model promotions')
    parser.add_argument('--max-samples', type=float, default=2000000,dest='max_samples',
                        help='will load last n samples in sorting datasets')
    # Options
    parser.add_argument('--skip-arena', action='store_true', dest='skip_arena',
                        help='Skip arena evaluation (always promote candidate)')
    parser.add_argument('--skip-selfplay', action='store_true', dest='skip_selfplay',
                        help='Skip selfplay (use pre-existing qsample files from this model)')
    parser.add_argument('--only-selfplay', action='store_true', dest='only_selfplay',
                        help='does not train or perform evals, only generates samples with existing model')
    parser.add_argument('--all-samples', action='store_true', dest='all_samples',
                        help='use all available samples in the sample file instead of just this model_id')
    parser.add_argument('--sync-remotes', action='store_true', dest='sync_remotes',
                        help='Sync samples from remote before training and push promoted models to remote')
    parser.add_argument('--big-model', dest="big_model", help='Use model with 6m parameters instead of 500k',
                        action='store_true', default=False)
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', dest='log_level',
                        help='Logging level')

    args = parser.parse_args()

    log_file = setup_logging(args.log_level)

    # Resolve paths
    project_root = get_project_root()
    samples_dir = project_root / args.samples_dir
    model_dir = project_root / args.model_dir

    # Symlink paths
    current_best_pt_link = model_dir / "current_best.pt"
    current_best_weights_link = model_dir / "current_best.model"
    candidate_pt_link = model_dir / "candidate.pt"
    candidate_weights_link = model_dir / "candidate.model"

    # Ensure directories exist
    samples_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model parameters
    if args.big_model:
        num_channels = 128
        num_blocks = 15
    else:
        # Default QuoridorNet params
        num_channels = 64
        num_blocks = 6

    logging.info("=" * 60)
    logging.info("Quoridor AlphaZero Training Loop")
    logging.info("=" * 60)
    logging.info(f"Log file:        {log_file}")
    logging.info(f"Samples dir:     {samples_dir}")
    logging.info(f"Model dir:       {model_dir}")
    logging.info(f"Model config:    {num_blocks} blocks, {num_channels} channels")
    logging.info(f"Current best:    {current_best_pt_link}")
    logging.info(f"Candidate:       {candidate_pt_link}")
    logging.info("=" * 60)

    # Initialize model
    model = QuoridorNet(num_channels=num_channels, num_blocks=num_blocks)

    # Load existing best model weights if available
    if current_best_weights_link.exists():
        logging.info(f"Loading existing model from {current_best_weights_link}")
        model.load_state_dict(torch.load(current_best_weights_link))
    elif current_best_pt_link.exists():
        logging.info("No weights file found, will use existing TorchScript model")
    else:
        logging.info("Starting with fresh random model")

    # Use GPU if available
    if torch.cuda.is_available():
        model.cuda()

    # Export initial model if needed
    if not current_best_pt_link.exists():
        logging.info("Exporting initial model as current_best...")

        # Save to temp files first
        temp_pt = model_dir / "initial_temp.pt"
        temp_weights = model_dir / "initial_temp.model"

        if not export_model(model, str(temp_pt)):
            logging.error("Failed to export initial model")
            return
        save_checkpoint(model, str(temp_weights))

        # Compute hash and rename
        initial_hash = compute_model_hash(str(temp_pt))

        best_pt_name = f"best_{initial_hash}_{num_blocks}x{num_channels}.pt"
        best_weights_name = f"best_{initial_hash}_{num_blocks}x{num_channels}.model"

        final_pt_path = model_dir / best_pt_name
        final_weights_path = model_dir / best_weights_name

        shutil.move(temp_pt, final_pt_path)
        shutil.move(temp_weights, final_weights_path)

        # Create symlinks
        update_symlink(current_best_pt_link, final_pt_path)
        update_symlink(current_best_weights_link, final_weights_path)
        logging.info(f"Initialized {best_pt_name}")

    ## see what sample numbering we're starting on
    sample_num = get_next_iteration(str(samples_dir))

    promotions = 0
    rejections = 0

    for iteration in range(args.iterations):
        logging.info("")
        logging.info(f"{'=' * 20} ITERATION {iteration} {'=' * 20}")

        # Compute model hash for current_best model (resolve symlink to get actual file)
        try:
            # We compute hash from the file pointed to by current_best_pt_link
            model_hash = compute_model_hash(str(current_best_pt_link))
            logging.info(f"Current model hash: {model_hash}")
        except Exception as e:
            logging.error(f"Failed to compute model hash: {e}")
            model_hash = ""

        tree_path = samples_dir / f"tree_{sample_num}.qbot"

        # Phase 1: Self-play
        if not args.skip_selfplay:
            logging.info(f"[Phase 1] Self-play ({args.games} games)...")
            # Pass the symlink path; run_selfplay resolves it
            if not run_selfplay(str(tree_path), str(current_best_pt_link), args.games,
                                args.simulations, args.threads, args.games_per_thread,
                                args.temperature, args.temp_drop, args.max_memory,
                                args.inference_batch_size, args.max_wait, model_hash):
                logging.error("Self-play failed, retrying iteration...")
                continue

            sample_num += 1
            if args.only_selfplay:
                #hit it again baby
                continue

        # Sync remote samples if requested
        if args.sync_remotes:
            sync_samples_from_remote(samples_dir)

        # Find matching samples
        matching_samples = find_samples_for_model(str(samples_dir), model_hash, args.all_samples)

        if not matching_samples:
            logging.error(f"No training samples found for model hash {model_hash}")
            logging.error("Self-play should generate .qsamples files automatically")
            continue


        # Phase 2: Train candidate model
        logging.info(f"[Phase 2] Training candidate neural network...")

        # Reload current best weights if needed
        if iteration == 0:
            if current_best_weights_link.exists():
                model.load_state_dict(torch.load(current_best_weights_link))
                if torch.cuda.is_available():
                    model.cuda()
        else:
            if candidate_weights_link.exists():
                model.load_state_dict(torch.load(candidate_weights_link))
                if torch.cuda.is_available():
                    model.cuda()

        # Train
        training_files = [str(p) for p in matching_samples]
        if not train_model(model, training_files, args.epochs, args.batch_size, args.max_samples):
            logging.error("Training failed, skipping iteration...")
            continue

        # Save candidate to temp files first
        temp_cand_pt = model_dir / "temp_candidate.pt"
        temp_cand_weights = model_dir / "temp_candidate.model"

        save_checkpoint(model, str(temp_cand_weights))
        export_model(model, str(temp_cand_pt))

        # Compute hash and rename to candidate_<hash>...
        cand_hash = compute_model_hash(str(temp_cand_pt))

        cand_pt_name = f"candidate_{cand_hash}_{num_blocks}x{num_channels}.pt"
        cand_weights_name = f"candidate_{cand_hash}_{num_blocks}x{num_channels}.model"

        cand_pt_path = model_dir / cand_pt_name
        cand_weights_path = model_dir / cand_weights_name

        # Move temp to formatted candidate name (overwriting if exists from previous run)
        shutil.move(temp_cand_pt, cand_pt_path)
        shutil.move(temp_cand_weights, cand_weights_path)

        # Update candidate symlinks
        update_symlink(candidate_pt_link, cand_pt_path)
        update_symlink(candidate_weights_link, cand_weights_path)

        logging.info(f"Saved candidate: {cand_pt_name}")

        gc.collect()
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass


        # Phase 3: Arena evaluation
        if args.skip_arena:
            logging.info("[Phase 3] Arena skipped, promoting candidate...")
            candidate_won = True
            arena_success = True
        else:
            logging.info(f"[Phase 3] Arena evaluation ({args.arena_games} games)...")
            # Pass symlinks
            arena_success, candidate_won = run_arena(
                str(current_best_pt_link), str(candidate_pt_link), args.threads,
                args.arena_games, args.arena_sims, args.promote_alpha,
                args.arena_temperature, args.arena_temp_drop, args.max_memory
            )

            if not arena_success:
                logging.error("Arena evaluation failed, keeping current model...")
                rejections += 1
                continue

        # Handle result
        if candidate_won:
            promotions += 1
            logging.info(f"Candidate PROMOTED! (total promotions: {promotions})")

            # Construct best filename
            best_pt_name = f"best_{cand_hash}_{num_blocks}x{num_channels}.pt"
            best_weights_name = f"best_{cand_hash}_{num_blocks}x{num_channels}.model"

            best_pt_path = model_dir / best_pt_name
            best_weights_path = model_dir / best_weights_name

            # Rename candidate files to best files
            # Note: We move the files that the candidate symlink currently points to
            try:
                # We can just move the paths we established earlier
                shutil.move(cand_pt_path, best_pt_path)
                shutil.move(cand_weights_path, best_weights_path)

                # Update current_best symlinks
                update_symlink(current_best_pt_link, best_pt_path)
                update_symlink(current_best_weights_link, best_weights_path)

                # Re-point candidate symlinks to the new best file (so they aren't broken)? 
                # Or leave them broken until next training? 
                # Better to remove them so we don't accidentally use old candidate
                if candidate_pt_link.exists():
                    candidate_pt_link.unlink()
                if candidate_weights_link.exists():
                    candidate_weights_link.unlink()

            except Exception as e:
                logging.error(f"Error during promotion file operations: {e}")
                # Try to recover by reloading old best
                if current_best_weights_link.exists():
                    model.load_state_dict(torch.load(current_best_weights_link))
                continue

            # Push promoted model to remote if requested
            if args.sync_remotes:
                push_model_to_remote(model_dir)
        else:
            rejections += 1
            logging.info(f"Candidate rejected. (total rejections: {rejections})")
            # Reload current best weights for next iteration
            if current_best_weights_link.exists():
                model.load_state_dict(torch.load(current_best_weights_link))
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
    logging.info(f"  Final model: {current_best_pt_link.resolve().name}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
