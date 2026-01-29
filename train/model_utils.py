"""Utilities for model management and identification."""
import hashlib
import json
import logging
import re
import subprocess
import os
from pathlib import Path

# Remote sync configuration
REMOTE_HOST = "eserver"
REMOTE_SAMPLES_PATH = "~/repos/qbot/samples"
REMOTE_MODEL_PATH = "~/repos/qbot/model"


def compute_model_hash(model_path: str, hash_length: int = 8) -> str:
    """
    Compute a short hash identifier for a model file.
    Resolves symlinks to hash the actual file content.

    Args:
        model_path: Path to the .pt model file
        hash_length: Number of hex characters to return (default: 8)

    Returns:
        Hex string hash of the model file
    """
    path = Path(model_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        # Read in chunks to handle large files
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()[:hash_length]


def update_symlink(link_path: Path, target_path: Path):
    """
    Safely update a symlink to point to target_path.
    Uses relative paths if possible for portability.
    """
    link_path = Path(link_path)
    target_path = Path(target_path)

    # Calculate relative path from link dir to target
    try:
        relative_target = target_path.relative_to(link_path.parent)
    except ValueError:
        # Not in same tree, use absolute
        relative_target = target_path.resolve()

    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()

    link_path.symlink_to(relative_target)


def find_samples_for_model(samples_dir: str, model_hash: str, all_samples: bool = False) -> list[Path]:
    """
    Find all .qsamples files generated with a specific model version.

    Args:
        samples_dir: Directory containing sample files
        model_hash: Model hash identifier to match

    Returns:
        List of Path objects for matching sample files, sorted by iteration number
    """
    samples_path = Path(samples_dir)
    if not samples_path.exists():
        return []

    # Pattern: tree_<iter#>_<modelhash>.qsamples
    matching_files = []
    if all_samples:
        for file in samples_path.glob(f"tree_*.qsamples"):
            matching_files.append(file)
    else:
        for file in samples_path.glob(f"tree_*_{model_hash}.qsamples"):
            matching_files.append(file)

    # Sort by iteration number (extract from filename)
    def get_iter_num(path: Path) -> int:
        try:
            parts = path.stem.split('_')
            if len(parts) < 2:
                return (0, "")

            raw_val = parts[1]

            match = re.match(r"(\d+)([a-z]?)", raw_val, re.I)
            if match:
                num_part = int(match.group(1))
                char_part = match.group(2)

                # We return (-num_part) to sort 100 before 99 (descending)
                # We return char_part to sort "78" before "78a" (ascending)
                return (-num_part, char_part)

        except (ValueError, IndexError):
            pass
        return (0, "")

    matching_files.sort(key=get_iter_num, reverse=True)
    return matching_files


def find_all_model_hashes(samples_dir: str) -> dict[str, list[Path]]:
    """
    Group all sample files by model hash.
    """
    samples_path = Path(samples_dir)
    if not samples_path.exists():
        return {}

    hash_to_files = {}

    for file in samples_path.glob("tree_*_*.qsamples"):
        # Extract hash from "tree_123_abc.qsamples"
        parts = file.stem.split('_')
        if len(parts) >= 3:
            model_hash = parts[2]
            if model_hash not in hash_to_files:
                hash_to_files[model_hash] = []
            hash_to_files[model_hash].append(file)

    # Sort each list by iteration number
    for files in hash_to_files.values():
        files.sort(key=lambda p: int(p.stem.split('_')[1]) if p.stem.split('_')[1].isdigit() else 0)

    return hash_to_files


# Remote sync functions

def load_sync_state(samples_dir: Path) -> dict:
    """Load mapping of remote filenames to local filenames."""
    state_file = samples_dir / ".sync_state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {}


def save_sync_state(samples_dir: Path, state: dict):
    """Save mapping of remote filenames to local filenames."""
    state_file = samples_dir / ".sync_state.json"
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def get_remote_files() -> list[str]:
    """Get list of .qsamples files on remote server."""
    try:
        result = subprocess.run(
            ["ssh", REMOTE_HOST, f"ls {REMOTE_SAMPLES_PATH}/*.qsamples 2>/dev/null || true"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            logging.warning(f"Could not list remote files: {result.stderr}")
            return []

        files = [Path(line.strip()).name for line in result.stdout.strip().split('\n') if line.strip()]
        return files
    except Exception as e:
        logging.error(f"Error listing remote files: {e}")
        return []


def parse_sample_filename(filename: str) -> tuple[int, str] | None:
    """Extract number and hash from filename like tree_1_598921ac.qsamples."""
    match = re.match(r'tree_(\d+)([a-z]?)_([0-9a-f]+)\.qsamples', filename)
    if match:
        num = int(match.group(1))
        hash_val = match.group(3)
        return (num, hash_val)
    return None


def get_local_max_sample_number(samples_dir: Path) -> int:
    """Get the highest number used in local sample files."""
    max_num = 0
    for filename in samples_dir.glob("tree_*.qsamples"):
        match = re.match(r'tree_(\d+)([a-z]?)_([0-9a-f]+)\.qsamples', filename.name)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)
    return max_num


def get_local_max_sample_number_for_hash(samples_dir: Path, hash_val: str) -> int | None:
    """Get the highest number used in local sample files with a specific hash."""
    max_num = None
    for filename in samples_dir.glob(f"tree_*_{hash_val}.qsamples"):
        match = re.match(r'tree_(\d+)([a-z]?)_([0-9a-f]+)\.qsamples', filename.name)
        if match:
            num = int(match.group(1))
            if max_num is None or num > max_num:
                max_num = num
    return max_num


def get_next_alpha_suffix(samples_dir: Path, local_num: int, hash_val: str) -> str:
    """Get the next alphabetic suffix for a given number/hash combo."""
    pattern = re.compile(rf'tree_{local_num}([a-z]?)_{hash_val}\.qsamples')
    existing_suffixes = []

    for filename in samples_dir.glob(f"tree_{local_num}*_{hash_val}.qsamples"):
        match = pattern.match(filename.name)
        if match:
            suffix = match.group(1)
            existing_suffixes.append(suffix)

    if not existing_suffixes:
        return ''

    # If there's a file with no suffix, next should be 'a'
    if '' in existing_suffixes:
        if 'a' not in existing_suffixes:
            return 'a'

    # Find the highest suffix and increment
    alpha_suffixes = [s for s in existing_suffixes if s]
    if not alpha_suffixes:
        return 'a'

    highest = max(alpha_suffixes)
    next_char = chr(ord(highest) + 1)

    if next_char > 'z':
        raise ValueError(f"Exhausted alphabetic suffixes for tree_{local_num}_{hash_val}")

    return next_char


def sync_samples_from_remote(samples_dir: Path) -> bool:
    """
    Sync sample files from remote server to local directory.
    """

    sync_state = load_sync_state(samples_dir)
    remote_files = get_remote_files()

    if not remote_files:
        logging.info("No remote files found or remote not accessible")
        return False

    # Get current overall max number (for hashes we haven't seen locally)
    overall_max = get_local_max_sample_number(samples_dir)

    # Track files to sync and temporarily created filenames
    files_to_sync = []
    temp_files = []  # Track files we're about to create in this batch

    for remote_file in remote_files:
        # Check if we already have a mapping for this remote file
        if remote_file in sync_state:
            local_file = sync_state[remote_file]
            if not (samples_dir / local_file).exists():
                logging.warning(f"State file has {remote_file} but local file {local_file} missing")
            # Always add to sync list (rsync -u will skip if local is up to date)
            files_to_sync.append((remote_file, local_file))
            continue

        # Parse remote filename for new files
        parsed = parse_sample_filename(remote_file)
        if not parsed:
            logging.warning(f"Skipping invalid remote filename: {remote_file}")
            continue

        remote_num, hash_val = parsed

        # Find the most recent local sample with this hash
        local_num_for_hash = get_local_max_sample_number_for_hash(samples_dir, hash_val)

        if local_num_for_hash is None:
            # No local files with this hash, assign a new number
            overall_max += 1
            local_num = overall_max
        else:
            # Use the same number as the most recent local file with this hash
            local_num = local_num_for_hash

        # Get alphabetic suffix (need to account for files we're about to create)
        alpha_suffix = get_next_alpha_suffix(samples_dir, local_num, hash_val)

        # Check if we're creating a file with this num/hash in this batch
        # If so, increment the suffix
        while True:
            local_file = f"tree_{local_num}{alpha_suffix}_{hash_val}.qsamples"
            if local_file not in temp_files:
                break
            # This suffix is taken by a file we're syncing in this batch, get next
            if alpha_suffix == '':
                alpha_suffix = 'a'
            else:
                alpha_suffix = chr(ord(alpha_suffix) + 1)
                if alpha_suffix > 'z':
                    raise ValueError(f"Exhausted alphabetic suffixes for tree_{local_num}_{hash_val}")

        files_to_sync.append((remote_file, local_file))
        temp_files.append(local_file)

    if not files_to_sync:
        logging.info("No new remote samples to sync")
        return False

    # Perform the sync
    synced_count = 0
    for remote_file, local_file in files_to_sync:
        remote_path = f"{REMOTE_HOST}:{REMOTE_SAMPLES_PATH}/{remote_file}"
        local_path = samples_dir / local_file

        try:
            result = subprocess.run(
                ["rsync", "-azu", "--progress", remote_path, str(local_path)],
                capture_output=True,
                text=True,
                timeout=600
            )
            if "qsamples" in result.stdout:
                logging.info(f"Syncing {remote_file} -> {local_file}")

            if result.returncode == 0:
                sync_state[remote_file] = local_file
                save_sync_state(samples_dir, sync_state)
                synced_count += 1
            else:
                logging.error(f"  Failed to sync {remote_file}: {result.stderr}")

        except Exception as e:
            logging.error(f"  Error syncing {remote_file}: {e}")

    if synced_count > 0:
        return True

    return False


def push_model_to_remote(model_dir: Path) -> bool:
    """
    Push current_best model to remote server.
    Resolves symlinks to push the actual file content to the remote's static filename.
    """
    logging.info("Pushing promoted model to remote server...")

    # We use the symlink but resolve it to get the real file path
    local_model_pt_link = model_dir / "current_best.pt"
    local_model_weights_link = model_dir / "current_best.model"

    if not local_model_pt_link.exists():
        logging.error("Cannot push model: current_best.pt not found")
        return False

    # Resolve to actual file: best_<hash>...pt
    real_pt_path = local_model_pt_link.resolve()

    try:
        # Push TorchScript model
        # Source is the unique file, Dest is the generic "current_best.pt" on remote
        remote_path_pt = f"{REMOTE_HOST}:{REMOTE_MODEL_PATH}/current_best.pt"
        result = subprocess.run(
            ["rsync", "-az", "--progress", str(real_pt_path), remote_path_pt],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            logging.error(f"Failed to push model .pt: {result.stderr}")
            return False

        logging.info("  Pushed current_best.pt")

        # Push weights if they exist
        if local_model_weights_link.exists():
            real_weights_path = local_model_weights_link.resolve()
            remote_path_weights = f"{REMOTE_HOST}:{REMOTE_MODEL_PATH}/current_best.model"
            result = subprocess.run(
                ["rsync", "-az", "--progress", str(real_weights_path), remote_path_weights],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                logging.warning(f"Failed to push model weights: {result.stderr}")
            else:
                logging.info("  Pushed current_best.model")

        logging.info("Model successfully pushed to remote server")
        return True

    except Exception as e:
        logging.error(f"Error pushing model to remote: {e}")
        return False
