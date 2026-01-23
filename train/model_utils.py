"""Utilities for model management and identification."""
import hashlib
import re
from pathlib import Path


def compute_model_hash(model_path: str, hash_length: int = 8) -> str:
    """
    Compute a short hash identifier for a model file.

    Uses SHA256 of the file contents and returns the first hash_length characters.
    This provides a stable identifier for tracking which samples were generated
    with which model version.

    Args:
        model_path: Path to the .pt model file
        hash_length: Number of hex characters to return (default: 8)

    Returns:
        Hex string hash of the model file
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    sha256 = hashlib.sha256()
    with open(model_path, 'rb') as f:
        # Read in chunks to handle large files
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()[:hash_length]


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

    Args:
        samples_dir: Directory containing sample files

    Returns:
        Dictionary mapping model_hash -> list of sample file paths
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
