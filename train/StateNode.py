import ctypes
import subprocess
import numpy as np
import torch
import random
from typing import Optional
import logging
from io import BufferedReader
import os
import struct
from torch.utils.data import Dataset
# Action space constants (must match resnet.py)
NUM_PAWN_ACTIONS = 81   # 9x9 board destinations
NUM_WALL_ACTIONS = 128  # 8x8 * 2 orientations
NUM_ACTIONS = NUM_PAWN_ACTIONS + NUM_WALL_ACTIONS  # 209 total


# =============================================================================
# Training Sample Format (.qsamples files)
# =============================================================================
# These contain pre-computed training samples with MCTS visit distributions.
# This is the preferred format for AlphaZero-style training.

# struct TrainingSampleHeader { // 64 bytes
#     uint32_t magic = 0x51534D50;  // "QSMP"
#     uint16_t version = 1;
#     uint16_t flags = 0;
#     uint32_t sample_count;
#     uint32_t reserved1;
#     uint64_t timestamp;
#     uint8_t reserved[40];
# };

QSMP_MAGIC = 0x51534D50
QSMP_HEADER_SIZE = 64

# struct CompactState { // 24 bytes
#     uint8_t p1_row, p1_col, p2_row, p2_col;  // 4 bytes
#     uint8_t p1_fences, p2_fences;             // 2 bytes
#     uint8_t flags, reserved;                  // 2 bytes
#     uint64_t fences_horizontal;               // 8 bytes
#     uint64_t fences_vertical;                 // 8 bytes
# };

# struct TrainingSample { // 864 bytes
#     CompactState state;                       // 24 bytes
#     float policy[209];                        // 836 bytes
#     float value;                              // 4 bytes
# };

TRAINING_SAMPLE_SIZE = 872

# Numpy dtype for fast binary loading matching the C++ struct layout
# This maps exactly to the 864-byte TrainingSample struct
QSMP_DTYPE = np.dtype([
    ('p1_row', 'u1'), ('p1_col', 'u1'), ('p2_row', 'u1'), ('p2_col', 'u1'),
    ('p1_fences', 'u1'), ('p2_fences', 'u1'),
    ('flags', 'u1'), ('reserved_byte', 'u1'),
    ('fences_h', '<u8'), ('fences_v', '<u8'),
    ('policy', '<f4', (209,)),
    ('wins', '<u4'),
    ('losses', '<u4'),
    ('value', '<f4')
])

def load_qsamples_raw(file_path: str, max_count: int = None):
    """
    Load raw sample data from a v2 .qsamples file.

    Returns:
        raw_data: numpy structured array with all fields
    """
    with open(file_path, 'rb') as f:
        header_data = f.read(QSMP_HEADER_SIZE)
        if len(header_data) < QSMP_HEADER_SIZE:
            raise ValueError(f"Invalid .qsamples file: {file_path}")

        magic, version, flags, sample_count = struct.unpack('<IHHI', header_data[:12])
        if magic != QSMP_MAGIC:
            raise ValueError(f"Invalid magic: {hex(magic)}")
        if version < 2:
            raise ValueError(f"Unsupported version {version}. Run convert_qsamples.py first.")

        if max_count is not None:
            count = min(int(sample_count), int(max_count))
        else:
            count = int(sample_count)

        if count == 0:
            return None

        print(f"loading {count} samples of length {QSMP_DTYPE.itemsize}")
        raw_data = np.fromfile(f, dtype=QSMP_DTYPE, count=count)

    return raw_data



def _raw_to_state_tensor(raw_data: np.ndarray) -> np.ndarray:
    """Vectorized conversion of raw sample data to state tensors."""
    N = len(raw_data)
    states = np.zeros((N, 6, 9, 9), dtype=np.float32)

    p1_row = raw_data['p1_row']
    p1_col = raw_data['p1_col']
    p2_row = raw_data['p2_row']
    p2_col = raw_data['p2_col']
    p1_fences = raw_data['p1_fences']
    p2_fences = raw_data['p2_fences']
    flags = raw_data['flags']

    is_p1_turn = (flags & 0x04) != 0

    # print(raw_data)
    my_row = np.where(is_p1_turn, p1_row, p2_row)
    my_col = np.where(is_p1_turn, p1_col, p2_col)
    my_fences = np.where(is_p1_turn, p1_fences, p2_fences)

    opp_row = np.where(is_p1_turn, p2_row, p1_row)
    opp_col = np.where(is_p1_turn, p2_col, p1_col)
    opp_fences = np.where(is_p1_turn, p2_fences, p1_fences)

    batch_idx = np.arange(N)
    states[batch_idx, 0, my_row, my_col] = 1.0
    states[batch_idx, 1, opp_row, opp_col] = 1.0

    bit_indices = np.arange(64, dtype=np.uint64)
    masks = (1 << bit_indices)

    walls_h_flat = (raw_data['fences_h'][:, None] & masks) > 0
    walls_v_flat = (raw_data['fences_v'][:, None] & masks) > 0

    states[:, 2, :8, :8] = walls_h_flat.reshape(N, 8, 8)
    states[:, 3, :8, :8] = walls_v_flat.reshape(N, 8, 8)

    states[:, 4, :, :] = (my_fences[:, None, None] / 10.0)
    states[:, 5, :, :] = (opp_fences[:, None, None] / 10.0)

    return states

class TrainingSampleDataset(Dataset):
    """
    Dataset that expands samples based on wins/losses count.

    For each position with W wins and L losses, creates W+L virtual samples:
    - W samples with value target +1 (win)
    - L samples with value target -1 (loss)

    Memory-efficient: stores raw data once, expands indices on-demand.

    Usage:
        dataset = TrainingSampleDataset(['file1.qsamples', 'file2.qsamples'])
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
    """

    def __init__(self, samples_paths: list[str], max_samples: int = 5_000_000):
        self.samples_paths = samples_paths
        self.max_samples = max_samples

        all_raw = []
        total_loaded = 0

        for path in reversed(samples_paths):
            remaining = int(max_samples - total_loaded)
            if remaining <= 0:
                break

            raw = load_qsamples_raw(path, max_count=remaining)
            if raw is not None:
                all_raw.append(raw)
                total_loaded += len(raw)
                logging.info(f"  + {path}: {len(raw)} samples")

        if not all_raw:
            raise ValueError("No samples found in provided files")

        self.raw_data = np.concatenate(all_raw)
        logging.info(f"Loaded {len(self.raw_data)} base samples")

        wins = self.raw_data['wins']
        losses = self.raw_data['losses']
        games_per_sample = wins + losses
        games_per_sample = np.maximum(games_per_sample, 1)

        self.cumsum = np.zeros(len(self.raw_data) + 1, dtype=np.int64)
        self.cumsum[1:] = np.cumsum(games_per_sample)
        self.total_samples = int(self.cumsum[-1])

        logging.info(f"Expanded to {self.total_samples} virtual samples "
                     f"({self.total_samples / len(self.raw_data):.2f}x expansion)")

        self.states = torch.from_numpy(_raw_to_state_tensor(self.raw_data))
        self.policies = torch.from_numpy(self.raw_data['policy'].copy())

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        base_idx = np.searchsorted(self.cumsum[1:], idx, side='right')
        offset = idx - self.cumsum[base_idx]
        wins = int(self.raw_data['wins'][base_idx])

        value = 1.0 if offset < wins else -1.0

        return (
            self.states[base_idx],
            self.policies[base_idx],
            torch.tensor([value], dtype=torch.float32)
        )
    # def __getitem__(self, idx):
    #     idx = int(idx)
    #     base_idx = int(np.searchsorted(self.cumsum[1:], idx, side='right'))
    #     offset = idx - int(self.cumsum[base_idx])
    #     wins = int(self.raw_data['wins'][base_idx])
    #     
    #     value = 1.0 if offset < wins else -1.0
    #     
    #     return (
    #         self.states[base_idx],
    #         self.policies[base_idx],
    #         torch.tensor([value], dtype=torch.float32)
    #     )

def load_qsamples_fast(file_path: str, max_count: int = None):
    """
    Vectorized loader for .qsamples files. 
    Loads the file 100x faster than struct.unpack loop.
    """
    with open(file_path, 'rb') as f:
        # Check header
        header_data = f.read(QSMP_HEADER_SIZE)
        if len(header_data) < QSMP_HEADER_SIZE:
            raise ValueError(f"Invalid .qsamples file: {file_path}")

        magic, version, flags, sample_count = struct.unpack('<IHHI', header_data[:12])
        if magic != QSMP_MAGIC:
            raise ValueError(f"Invalid magic: {hex(magic)}")

        # Limit count if requested
        if max_count is not None:
            count = min(sample_count, max_count)
        else:
            count = sample_count

        if count == 0:
            return None, None, None

        # 1. READ RAW BINARY (Zero-copy if possible)
        # This reads all data into a numpy structured array in one go
        raw_data = np.fromfile(f, dtype=QSMP_DTYPE, count=count)

    # 2. VECTORIZED STATE GENERATION
    # Create empty tensor (N, 6, 9, 9)
    N = len(raw_data)
    states = np.zeros((N, 6, 9, 9), dtype=np.float32)

    # Extract columns for cleaner code
    p1_row = raw_data['p1_row']
    p1_col = raw_data['p1_col']
    p2_row = raw_data['p2_row']
    p2_col = raw_data['p2_col']
    p1_fences = raw_data['p1_fences']
    p2_fences = raw_data['p2_fences']
    flags = raw_data['flags']

    # Identify current player (bit 2 is FLAG_P1_TO_MOVE)
    is_p1_turn = (flags & 0x04) != 0

    # Vectorized "Current Player" perspective swap
    my_row = np.where(is_p1_turn, p1_row, p2_row)
    my_col = np.where(is_p1_turn, p1_col, p2_col)
    my_fences = np.where(is_p1_turn, p1_fences, p2_fences)

    opp_row = np.where(is_p1_turn, p2_row, p1_row)
    opp_col = np.where(is_p1_turn, p2_col, p1_col)
    opp_fences = np.where(is_p1_turn, p2_fences, p1_fences)

    # Channel 0 & 1: Pawns (One-hot fancy indexing)
    batch_idx = np.arange(N)
    states[batch_idx, 0, my_row, my_col] = 1.0
    states[batch_idx, 1, opp_row, opp_col] = 1.0

    # Channel 2 & 3: Walls
    # Use broadcasting to unpack bits. 
    # Create a mask of shape (1, 64) -> (N, 64)
    bit_indices = np.arange(64, dtype=np.uint64)
    masks = (1 << bit_indices)

    # Expand fences to (N, 64) boolean mask
    # We must cast to int64 or uint64 to avoid overflow during bitshift
    walls_h_flat = (raw_data['fences_h'][:, None] & masks) > 0
    walls_v_flat = (raw_data['fences_v'][:, None] & masks) > 0

    # Reshape (N, 64) -> (N, 8, 8)
    walls_h_8x8 = walls_h_flat.reshape(N, 8, 8)
    walls_v_8x8 = walls_v_flat.reshape(N, 8, 8)

    # Assign to 9x9 grid (leaving last row/col zero as padding)
    states[:, 2, :8, :8] = walls_h_8x8
    states[:, 3, :8, :8] = walls_v_8x8

    # Channel 4 & 5: Fences (Broadcasting)
    # (N,) -> (N, 1, 1) -> Broadcast to (N, 9, 9)
    states[:, 4, :, :] = (my_fences[:, None, None] / 10.0)
    states[:, 5, :, :] = (opp_fences[:, None, None] / 10.0)

    # 3. PREPARE OUTPUTS
    # Convert structured policy array to standard float array
    # View as (N, 209) float32
    policies = raw_data['policy'].copy() # Copy to ensure contiguous memory

    # Values (N, 1)
    values = raw_data['value'][:, None].copy()

    return (
        torch.from_numpy(states), 
        torch.from_numpy(policies), 
        torch.from_numpy(values)
    )

class MultiFileTrainingSampleDataset:
    """
    Dataset loader that concatenates multiple .qsamples files into a single dataset.

    This is useful for training on accumulated samples from the same model version,
    where multiple self-play runs generated separate sample files.

    By default, loads all samples from all files into memory and shuffles them.

    Usage:
        with MultiFileTrainingSampleDataset(['tree_0.qsamples', 'tree_1.qsamples'], 64) as dataset:
            for batch in dataset.generate_batches():
                train_step(model, batch)
    """
    def __init__(self, samples_paths: list[str], batch_size: int, max_sample_num: int = 3000000):
        self.samples_paths = samples_paths
        self.batch_size = batch_size
        self.max_sample_num = max_sample_num

        self.states = None
        self.policies = None
        self.values = None

        self.__enter__()

    def __len__(self):
        return len(self.states) if self.states is not None else 0

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]

    def __enter__(self):
        logging.info(f"Loading samples from {len(self.samples_paths)} files...")

        all_states = []
        all_policies = []
        all_values = []
        total_loaded = 0

        # Load latest files first
        self.samples_paths.reverse()

        for path in self.samples_paths:
            remaining = self.max_sample_num - total_loaded
            if remaining <= 0:
                break

            s, p, v = load_qsamples_fast(path, max_count=remaining)
            if s is not None:
                all_states.append(s)
                all_policies.append(p)
                all_values.append(v)
                total_loaded += len(s)
                logging.info(f"  + {path}: {len(s)} samples")

        if not all_states:
            raise ValueError("No samples found in provided files")

        # Concatenate everything into one massive tensor
        logging.info(f"Concatenating {total_loaded} samples...")
        self.states = torch.cat(all_states)
        self.policies = torch.cat(all_policies)
        self.values = torch.cat(all_values)

        # Free temp lists
        del all_states, all_policies, all_values

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def generate_batches(self):
        if self.states is None:
            raise RuntimeError("Dataset not initialized")

        N = len(self.states)
        indices = torch.randperm(N)

        for i in range(0, N, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield (
                self.states[batch_indices],
                self.policies[batch_indices],
                self.values[batch_indices]
            )

# class MultiFileTrainingSampleDataset:
#     """
#     Dataset loader that concatenates multiple .qsamples files into a single dataset.
#
#     This is useful for training on accumulated samples from the same model version,
#     where multiple self-play runs generated separate sample files.
#
#     By default, loads all samples from all files into memory and shuffles them.
#
#     Usage:
#         with MultiFileTrainingSampleDataset(['tree_0.qsamples', 'tree_1.qsamples'], 64) as dataset:
#             for batch in dataset.generate_batches():
#                 train_step(model, batch)
#     """
#
#     def __init__(self, samples_paths: list[str], batch_size: int, max_sample_num: int = 4500000):
#         """
#         Args:
#             samples_paths: List of .qsamples file paths to load
#             batch_size: Batch size for training
#         """
#         self.samples_paths = samples_paths
#         self.batch_size = batch_size
#         self.total_samples = 0
#         self.max_sample_num = max_sample_num
#         self.samples = None
#         self.__enter__()
#
#     def __len__(self):
#         # In-memory mode is required for this efficient shuffling
#         return len(self.samples) if self.samples else 0
#
#     def __getitem__(self, idx):
#         # Returns (state, policy, value) for a single index
#         return self.samples[idx]
#
#     def __enter__(self):
#         # Load all samples into memory from all files
#         logging.info(f"Loading samples from {len(self.samples_paths)} files into memory...")
#         self.samples = []
#
#         ## reverse list so we load in the latest samples first, so if we stop early we leave out the oldest
#         self.samples_paths.reverse()
#         for path in self.samples_paths:
#             with open(path, 'rb') as f:
#                 header_data = f.read(QSMP_HEADER_SIZE)
#                 if len(header_data) < QSMP_HEADER_SIZE:
#                     raise ValueError(f"Invalid .qsamples file: {path}")
#
#                 magic, version, flags, sample_count = struct.unpack('<IHHI', header_data[:12])
#                 if magic != QSMP_MAGIC:
#                     raise ValueError(f"Invalid .qsamples file: bad magic in {path}")
#                 if version > 1:
#                     raise ValueError(f"Unsupported .qsamples version in {path}: {version}")
#
#                 logging.info(f"  {path}: {sample_count} samples")
#                 self.total_samples += sample_count
#
#                 # Load all samples from this file
#                 for _ in range(sample_count):
#                     sample_data = f.read(TRAINING_SAMPLE_SIZE)
#                     if len(sample_data) < TRAINING_SAMPLE_SIZE:
#                         break
#                     self.samples.append(self._parse_sample(sample_data))
#
#                 if self.total_samples >= self.max_sample_num:
#                     break
#
#         logging.info(f"Loaded {self.total_samples} total samples into memory")
#
#         return self
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         pass
#
#     def _parse_sample(self, data: bytes) -> tuple:
#         """Parse a single TrainingSample from bytes."""
#         # CompactState (24 bytes)
#         p1_row, p1_col, p2_row, p2_col = struct.unpack('BBBB', data[0:4])
#         p1_fences, p2_fences = struct.unpack('BB', data[4:6])
#         flags = data[6]
#         fences_h, fences_v = struct.unpack('<QQ', data[8:24])
#
#         # Policy (209 floats = 836 bytes)
#         policy = np.frombuffer(data[24:24+NUM_ACTIONS*4], dtype=np.float32).copy()
#         policy_sum = np.sum(policy)
#         if not (0.99 < policy_sum < 1.01):
#             logging.warning(f"CORRUPT DATA? Policy sums to {policy_sum}, expected 1.0")
#
#         # Value (1 float = 4 bytes)
#         value = struct.unpack('<f', data[24+NUM_ACTIONS*4:24+NUM_ACTIONS*4+4])[0]
#
#         # Convert to tensor format
#         state = self._state_to_tensor(
#             p1_row, p1_col, p2_row, p2_col,
#             p1_fences, p2_fences, flags,
#             fences_h, fences_v
#         )
#
#         return (
#             torch.from_numpy(state),
#             torch.from_numpy(policy),
#             torch.tensor([value], dtype=torch.float32)
#         )
#
#     def _state_to_tensor(self, p1_row, p1_col, p2_row, p2_col,
#                          p1_fences, p2_fences, flags,
#                          fences_h, fences_v) -> np.ndarray:
#         """Convert compact state to 6-channel tensor."""
#         tensor = np.zeros((6, 9, 9), dtype=np.float32)
#
#         # FLAG_P1_TO_MOVE = 0x04
#         is_p1_turn = (flags & 0x04) != 0
#
#         # Determine current player and opponent
#         if is_p1_turn:
#             my_row, my_col, my_fences = p1_row, p1_col, p1_fences
#             opp_row, opp_col, opp_fences = p2_row, p2_col, p2_fences
#         else:
#             my_row, my_col, my_fences = p2_row, p2_col, p2_fences
#             opp_row, opp_col, opp_fences = p1_row, p1_col, p1_fences
#
#         # Channel 0: Current player's pawn
#         tensor[0, my_row, my_col] = 1.0
#
#         # Channel 1: Opponent's pawn
#         tensor[1, opp_row, opp_col] = 1.0
#
#         # Channel 2: Horizontal walls
#         for r in range(8):
#             for c in range(8):
#                 if (fences_h >> (r * 8 + c)) & 1:
#                     tensor[2, r, c] = 1.0
#
#         # Channel 3: Vertical walls
#         for r in range(8):
#             for c in range(8):
#                 if (fences_v >> (r * 8 + c)) & 1:
#                     tensor[3, r, c] = 1.0
#
#         # Channel 4: Current player's fences
#         tensor[4, :, :] = my_fences / 10.0
#
#         # Channel 5: Opponent's fences
#         tensor[5, :, :] = opp_fences / 10.0
#
#         return tensor
#
#     def generate_batches(self):
#         """
#         Generate batches of training data from all files.
#
#         In-memory mode (default): Shuffles all samples before generating batches.
#         """
#         if self.samples is None:
#             raise RuntimeError("Dataset not initialized with context manager")
#
#         # Shuffle samples for better training
#         indices = list(range(len(self.samples)))
#         random.shuffle(indices)
#
#         # Generate batches from shuffled indices
#         for i in range(0, len(indices), self.batch_size):
#             batch_indices = indices[i:i + self.batch_size]
#             states = [self.samples[idx][0] for idx in batch_indices]
#             policies = [self.samples[idx][1] for idx in batch_indices]
#             values = [self.samples[idx][2] for idx in batch_indices]
#
#             yield (
#                 torch.stack(states),
#                 torch.stack(policies),
#                 torch.stack(values)
#             )


# =============================================================================
# Legacy Tree Format (.qbot files)
# =============================================================================
# These are pruned tree files containing SerializedNode structs.
# Policy targets are approximated (uniform distribution) since visit counts
# aren't stored per-child.

# Mirror the C++ SerializedNode structure from storage.h
# struct SerializedNode {
#     uint32_t first_child;
#     uint32_t next_sibling;
#     uint32_t parent;
#     uint8_t p1_row, p1_col, p1_fences;
#     uint8_t p2_row, p2_col, p2_fences;
#     uint16_t move_data;
#     uint8_t flags;
#     uint8_t reserved;
#     uint16_t ply;
#     uint64_t fences_horizontal;
#     uint64_t fences_vertical;
#     uint32_t visits;
#     float total_value;
#     float prior;
#     float terminal_value;
# };
# static_assert(sizeof(SerializedNode) == 56, "SerializedNode should be 56 bytes");

class SerializedNode(ctypes.Structure):
    _pack_ = 1  # Ensure no padding
    _fields_ = [
        # Tree structure (12 bytes)
        ("first_child", ctypes.c_uint32),
        ("next_sibling", ctypes.c_uint32),
        ("parent", ctypes.c_uint32),
        # Player 1 state (3 bytes)
        ("p1_row", ctypes.c_uint8),
        ("p1_col", ctypes.c_uint8),
        ("p1_fences", ctypes.c_uint8),
        # Player 2 state (3 bytes)
        ("p2_row", ctypes.c_uint8),
        ("p2_col", ctypes.c_uint8),
        ("p2_fences", ctypes.c_uint8),
        # Move and flags (6 bytes)
        ("move_data", ctypes.c_uint16),
        ("flags", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),
        ("ply", ctypes.c_uint16),
        # Fence grid (16 bytes)
        ("fences_horizontal", ctypes.c_uint64),
        ("fences_vertical", ctypes.c_uint64),
        # Statistics (16 bytes)
        ("visits", ctypes.c_uint32),
        ("total_value", ctypes.c_float),
        ("prior", ctypes.c_float),
        ("terminal_value", ctypes.c_float),
    ]

assert ctypes.sizeof(SerializedNode) == 56, f"SerializedNode should be 56 bytes, got {ctypes.sizeof(SerializedNode)}"


# Default leopard location (relative to train/ directory)
DEFAULT_LEOPARD_PATH = os.path.join(os.path.dirname(__file__), "..", "build", "leopard")


class QuoridorDataset:
    """
    Dataset loader for Quoridor training data.

    Reads binary SerializedNode structs via leopard and converts to tensors.
    All states are presented from the current player's perspective.

    Output format per sample:
        state: (6, 9, 9) tensor - current-player-perspective board representation
        policy_target: (209,) tensor - MCTS visit distribution (placeholder for now)
        value_target: (1,) tensor - game outcome from current player's view
    """

    def __init__(self, load_model: Optional[str], batch_size: int, leopard_path: str = DEFAULT_LEOPARD_PATH):
        self.load_model = load_model
        self.batch_size = batch_size
        self.leopard_path = leopard_path
        self.process = None

    def __enter__(self):
        if not os.path.exists(self.leopard_path):
            raise FileNotFoundError(f"leopard binary not found at {self.leopard_path}. Run 'make leopard' in build/")

        cmd = [self.leopard_path]
        if self.load_model:
            cmd.append(self.load_model)

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            self.process.terminate()
            self.process.wait()

    def _state_to_unified_tensor(self, state: SerializedNode) -> np.ndarray:
        """
        Convert SerializedNode to current-player-perspective 6-channel 9x9 tensor.

        The board is always presented from the CURRENT PLAYER's perspective:
        - Channel 0 is always "my" pawn, Channel 1 is always opponent's pawn
        - When it's P2's turn, we swap P1/P2 positions and fence counts

        Channels:
            [0] Current player's pawn position (one-hot)
            [1] Opponent's pawn position (one-hot)
            [2] Horizontal walls (padded 8x8 -> 9x9, last row/col = 0)
            [3] Vertical walls (padded 8x8 -> 9x9, last row/col = 0)
            [4] Current player's fences remaining / 10 (constant plane)
            [5] Opponent's fences remaining / 10 (constant plane)
        """
        tensor = np.zeros((6, 9, 9), dtype=np.float32)

        # FLAG_P1_TO_MOVE = 0x04
        is_p1_turn = (state.flags & 0x04) != 0

        # Determine current player and opponent based on whose turn it is
        if is_p1_turn:
            my_row, my_col, my_fences = state.p1_row, state.p1_col, state.p1_fences
            opp_row, opp_col, opp_fences = state.p2_row, state.p2_col, state.p2_fences
        else:
            my_row, my_col, my_fences = state.p2_row, state.p2_col, state.p2_fences
            opp_row, opp_col, opp_fences = state.p1_row, state.p1_col, state.p1_fences

        # Channel 0: Current player's pawn position (one-hot)
        tensor[0, my_row, my_col] = 1.0

        # Channel 1: Opponent's pawn position (one-hot)
        tensor[1, opp_row, opp_col] = 1.0

        # Channel 2: Horizontal walls (8x8 grid, padded to 9x9)
        for r in range(8):
            for c in range(8):
                bit_idx = r * 8 + c
                if (state.fences_horizontal >> bit_idx) & 1:
                    tensor[2, r, c] = 1.0

        # Channel 3: Vertical walls (8x8 grid, padded to 9x9)
        for r in range(8):
            for c in range(8):
                bit_idx = r * 8 + c
                if (state.fences_vertical >> bit_idx) & 1:
                    tensor[3, r, c] = 1.0

        # Channel 4: Current player's fences remaining (normalized, constant plane)
        tensor[4, :, :] = my_fences / 10.0

        # Channel 5: Opponent's fences remaining (normalized, constant plane)
        tensor[5, :, :] = opp_fences / 10.0

        return tensor

    def _create_policy_target(self, state: SerializedNode) -> np.ndarray:
        """
        Create policy target tensor.

        NOTE: The current storage format doesn't include per-child visit counts,
        which are needed for proper policy training. This is a placeholder that
        creates a uniform distribution. Proper policy targets will require C++
        changes to store MCTS visit distributions.

        For now, we use uniform distribution over all actions. This will be
        replaced once the C++ side stores visit counts.
        """
        # Placeholder: uniform distribution over all actions
        # In proper AlphaZero, this would be visit_counts / sum(visit_counts)
        policy = np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS
        return policy

    def _state_to_tensors(self, state: SerializedNode):
        """Convert SerializedNode to training tensors."""
        # Unified state tensor
        state_tensor = self._state_to_unified_tensor(state)

        # Policy target (placeholder - see _create_policy_target)
        policy_target = self._create_policy_target(state)

        # Value target: Q-value from current player's perspective in [-1, +1]
        value_target = np.array([state.terminal_value], dtype=np.float32)

        return (
            torch.from_numpy(state_tensor),
            torch.from_numpy(policy_target),
            torch.from_numpy(value_target)
        )

    def generate_batches(self):
        """Generate batches of training data."""
        if not self.process:
            raise RuntimeError("Dataset not initialized with context manager")

        states, policies, values = [], [], []

        # Set up buffers for efficient ctypes reading
        node_size = ctypes.sizeof(SerializedNode)
        read_chunk_size = 1024 * 16
        data_buf = bytearray(read_chunk_size)
        mview = memoryview(data_buf)
        buf_addr = ctypes.addressof((ctypes.c_ubyte).from_buffer(mview))
        offset = 0
        reader = BufferedReader(self.process.stdout)
        read_data = reader.read(read_chunk_size)
        mview[:len(read_data)] = read_data

        while True:
            try:
                remaining_data = len(read_data) - offset
                if remaining_data < node_size:
                    next_chunk = reader.read(offset)
                    if not next_chunk:
                        if remaining_data:
                            logging.error("Tree file ended with incomplete node")
                        break
                    read_data = read_data[offset:] + next_chunk
                    mview[:len(read_data)] = read_data
                    offset = 0
                    continue

                node = SerializedNode.from_address(buf_addr + offset)
                state_t, policy_t, value_t = self._state_to_tensors(node)

                states.append(state_t)
                policies.append(policy_t)
                values.append(value_t)

                offset += node_size

                if len(states) == self.batch_size:
                    yield (
                        torch.stack(states),
                        torch.stack(policies),
                        torch.stack(values)
                    )
                    states, policies, values = [], [], []

            except Exception as e:
                logging.error(f"Error processing state: {e}")
                raise e

        # Yield remaining samples
        if states:
            yield (
                torch.stack(states),
                torch.stack(policies),
                torch.stack(values)
            )
