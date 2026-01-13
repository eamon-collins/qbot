import ctypes
import subprocess
import numpy as np
import torch
from typing import Optional
import logging
from io import BufferedReader
import os
import struct

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

TRAINING_SAMPLE_SIZE = 864


class TrainingSampleDataset:
    """
    Dataset loader for .qsamples files containing pre-computed training samples.

    These files contain state + MCTS visit distribution + game outcome, making
    them ideal for AlphaZero-style training where policy targets come from
    MCTS search rather than move labels.

    By default, loads all samples into memory and shuffles them for better training.
    Use stream=True for memory-efficient streaming from disk (no shuffling).

    Output format per sample:
        state: (6, 9, 9) tensor - current-player-perspective board representation
        policy_target: (209,) tensor - MCTS visit distribution
        value_target: (1,) tensor - game outcome from current player's view
    """

    def __init__(self, samples_path: str, batch_size: int, stream: bool = False):
        self.samples_path = samples_path
        self.batch_size = batch_size
        self.stream = stream
        self.file = None
        self.samples = None  # In-memory storage when not streaming

    def __enter__(self):
        self.file = open(self.samples_path, 'rb')
        # Read and validate header
        header_data = self.file.read(QSMP_HEADER_SIZE)
        if len(header_data) < QSMP_HEADER_SIZE:
            raise ValueError(f"Invalid .qsamples file: too short")

        magic, version, flags, sample_count = struct.unpack('<IHHI', header_data[:12])
        if magic != QSMP_MAGIC:
            raise ValueError(f"Invalid .qsamples file: bad magic {hex(magic)}")
        if version > 1:
            raise ValueError(f"Unsupported .qsamples version: {version}")

        self.sample_count = sample_count

        if not self.stream:
            # Load all samples into memory
            logging.info(f"Loading {sample_count} samples from {self.samples_path} into memory...")
            self.samples = []
            for _ in range(sample_count):
                sample_data = self.file.read(TRAINING_SAMPLE_SIZE)
                if len(sample_data) < TRAINING_SAMPLE_SIZE:
                    break
                self.samples.append(self._parse_sample(sample_data))
            self.file.close()
            self.file = None
            logging.info(f"Loaded {len(self.samples)} samples into memory")
        else:
            logging.info(f"Opened {self.samples_path}: {sample_count} samples (streaming mode)")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def _parse_sample(self, data: bytes) -> tuple:
        """Parse a single TrainingSample from bytes."""
        # CompactState (24 bytes)
        p1_row, p1_col, p2_row, p2_col = struct.unpack('BBBB', data[0:4])
        p1_fences, p2_fences = struct.unpack('BB', data[4:6])
        flags = data[6]
        # reserved = data[7]
        fences_h, fences_v = struct.unpack('<QQ', data[8:24])

        # Policy (209 floats = 836 bytes)
        policy = np.frombuffer(data[24:24+NUM_ACTIONS*4], dtype=np.float32).copy()

        # Value (1 float = 4 bytes)
        value = struct.unpack('<f', data[24+NUM_ACTIONS*4:24+NUM_ACTIONS*4+4])[0]

        # Convert to tensor format
        state = self._state_to_tensor(
            p1_row, p1_col, p2_row, p2_col,
            p1_fences, p2_fences, flags,
            fences_h, fences_v
        )

        return (
            torch.from_numpy(state),
            torch.from_numpy(policy),
            torch.tensor([value], dtype=torch.float32)
        )

    def _state_to_tensor(self, p1_row, p1_col, p2_row, p2_col,
                         p1_fences, p2_fences, flags,
                         fences_h, fences_v) -> np.ndarray:
        """Convert compact state to 6-channel tensor."""
        tensor = np.zeros((6, 9, 9), dtype=np.float32)

        # FLAG_P1_TO_MOVE = 0x04
        is_p1_turn = (flags & 0x04) != 0

        # Determine current player and opponent
        if is_p1_turn:
            my_row, my_col, my_fences = p1_row, p1_col, p1_fences
            opp_row, opp_col, opp_fences = p2_row, p2_col, p2_fences
        else:
            my_row, my_col, my_fences = p2_row, p2_col, p2_fences
            opp_row, opp_col, opp_fences = p1_row, p1_col, p1_fences

        # Channel 0: Current player's pawn
        tensor[0, my_row, my_col] = 1.0

        # Channel 1: Opponent's pawn
        tensor[1, opp_row, opp_col] = 1.0

        # Channel 2: Horizontal walls
        for r in range(8):
            for c in range(8):
                if (fences_h >> (r * 8 + c)) & 1:
                    tensor[2, r, c] = 1.0

        # Channel 3: Vertical walls
        for r in range(8):
            for c in range(8):
                if (fences_v >> (r * 8 + c)) & 1:
                    tensor[3, r, c] = 1.0

        # Channel 4: Current player's fences
        tensor[4, :, :] = my_fences / 10.0

        # Channel 5: Opponent's fences
        tensor[5, :, :] = opp_fences / 10.0

        return tensor

    def generate_batches(self):
        """
        Generate batches of training data.

        In-memory mode (default): Shuffles samples before generating batches.
        Streaming mode: Reads samples sequentially from disk.
        """
        if self.stream:
            # Streaming mode: read from file
            if not self.file:
                raise RuntimeError("Dataset not initialized with context manager")

            states, policies, values = [], [], []

            for _ in range(self.sample_count):
                sample_data = self.file.read(TRAINING_SAMPLE_SIZE)
                if len(sample_data) < TRAINING_SAMPLE_SIZE:
                    break

                state_t, policy_t, value_t = self._parse_sample(sample_data)
                states.append(state_t)
                policies.append(policy_t)
                values.append(value_t)

                if len(states) == self.batch_size:
                    yield (
                        torch.stack(states),
                        torch.stack(policies),
                        torch.stack(values)
                    )
                    states, policies, values = [], [], []

            # Yield remaining samples
            if states:
                yield (
                    torch.stack(states),
                    torch.stack(policies),
                    torch.stack(values)
                )
        else:
            # In-memory mode: shuffle and generate batches
            if self.samples is None:
                raise RuntimeError("Dataset not initialized with context manager")

            # Shuffle samples for better training
            import random
            indices = list(range(len(self.samples)))
            random.shuffle(indices)

            # Generate batches from shuffled indices
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                states = [self.samples[idx][0] for idx in batch_indices]
                policies = [self.samples[idx][1] for idx in batch_indices]
                values = [self.samples[idx][2] for idx in batch_indices]

                yield (
                    torch.stack(states),
                    torch.stack(policies),
                    torch.stack(values)
                )


class MultiFileTrainingSampleDataset:
    """
    Dataset loader that concatenates multiple .qsamples files into a single dataset.

    This is useful for training on accumulated samples from the same model version,
    where multiple self-play runs generated separate sample files.

    By default, loads all samples from all files into memory and shuffles them.
    Use stream=True for memory-efficient streaming (no shuffling, files read sequentially).

    Usage:
        with MultiFileTrainingSampleDataset(['tree_0.qsamples', 'tree_1.qsamples'], 64) as dataset:
            for batch in dataset.generate_batches():
                train_step(model, batch)
    """

    def __init__(self, samples_paths: list[str], batch_size: int, stream: bool = False):
        """
        Args:
            samples_paths: List of .qsamples file paths to load
            batch_size: Batch size for training
            stream: If False (default), load all into memory and shuffle. If True, stream from disk.
        """
        self.samples_paths = samples_paths
        self.batch_size = batch_size
        self.stream = stream
        self.total_samples = 0
        self.samples = None  # In-memory storage when not streaming

    def __enter__(self):
        if not self.stream:
            # Load all samples into memory from all files
            logging.info(f"Loading samples from {len(self.samples_paths)} files into memory...")
            self.samples = []

            for path in self.samples_paths:
                with open(path, 'rb') as f:
                    header_data = f.read(QSMP_HEADER_SIZE)
                    if len(header_data) < QSMP_HEADER_SIZE:
                        raise ValueError(f"Invalid .qsamples file: {path}")

                    magic, version, flags, sample_count = struct.unpack('<IHHI', header_data[:12])
                    if magic != QSMP_MAGIC:
                        raise ValueError(f"Invalid .qsamples file: bad magic in {path}")
                    if version > 1:
                        raise ValueError(f"Unsupported .qsamples version in {path}: {version}")

                    logging.info(f"  {path}: {sample_count} samples")

                    # Load all samples from this file
                    for _ in range(sample_count):
                        sample_data = f.read(TRAINING_SAMPLE_SIZE)
                        if len(sample_data) < TRAINING_SAMPLE_SIZE:
                            break
                        self.samples.append(self._parse_sample(sample_data))

            self.total_samples = len(self.samples)
            logging.info(f"Loaded {self.total_samples} total samples into memory")
        else:
            # Streaming mode: just count samples
            logging.info(f"Opening {len(self.samples_paths)} files for streaming...")
            for path in self.samples_paths:
                with open(path, 'rb') as f:
                    header_data = f.read(QSMP_HEADER_SIZE)
                    if len(header_data) < QSMP_HEADER_SIZE:
                        raise ValueError(f"Invalid .qsamples file: {path}")

                    magic, version, flags, sample_count = struct.unpack('<IHHI', header_data[:12])
                    if magic != QSMP_MAGIC:
                        raise ValueError(f"Invalid .qsamples file: bad magic in {path}")
                    if version > 1:
                        raise ValueError(f"Unsupported .qsamples version in {path}: {version}")

                    self.total_samples += sample_count
                    logging.info(f"  {path}: {sample_count} samples")

            logging.info(f"Total samples across all files: {self.total_samples} (streaming mode)")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _parse_sample(self, data: bytes) -> tuple:
        """Parse a single TrainingSample from bytes."""
        # CompactState (24 bytes)
        p1_row, p1_col, p2_row, p2_col = struct.unpack('BBBB', data[0:4])
        p1_fences, p2_fences = struct.unpack('BB', data[4:6])
        flags = data[6]
        fences_h, fences_v = struct.unpack('<QQ', data[8:24])

        # Policy (209 floats = 836 bytes)
        policy = np.frombuffer(data[24:24+NUM_ACTIONS*4], dtype=np.float32).copy()

        # Value (1 float = 4 bytes)
        value = struct.unpack('<f', data[24+NUM_ACTIONS*4:24+NUM_ACTIONS*4+4])[0]

        # Convert to tensor format
        state = self._state_to_tensor(
            p1_row, p1_col, p2_row, p2_col,
            p1_fences, p2_fences, flags,
            fences_h, fences_v
        )

        return (
            torch.from_numpy(state),
            torch.from_numpy(policy),
            torch.tensor([value], dtype=torch.float32)
        )

    def _state_to_tensor(self, p1_row, p1_col, p2_row, p2_col,
                         p1_fences, p2_fences, flags,
                         fences_h, fences_v) -> np.ndarray:
        """Convert compact state to 6-channel tensor."""
        tensor = np.zeros((6, 9, 9), dtype=np.float32)

        # FLAG_P1_TO_MOVE = 0x04
        is_p1_turn = (flags & 0x04) != 0

        # Determine current player and opponent
        if is_p1_turn:
            my_row, my_col, my_fences = p1_row, p1_col, p1_fences
            opp_row, opp_col, opp_fences = p2_row, p2_col, p2_fences
        else:
            my_row, my_col, my_fences = p2_row, p2_col, p2_fences
            opp_row, opp_col, opp_fences = p1_row, p1_col, p1_fences

        # Channel 0: Current player's pawn
        tensor[0, my_row, my_col] = 1.0

        # Channel 1: Opponent's pawn
        tensor[1, opp_row, opp_col] = 1.0

        # Channel 2: Horizontal walls
        for r in range(8):
            for c in range(8):
                if (fences_h >> (r * 8 + c)) & 1:
                    tensor[2, r, c] = 1.0

        # Channel 3: Vertical walls
        for r in range(8):
            for c in range(8):
                if (fences_v >> (r * 8 + c)) & 1:
                    tensor[3, r, c] = 1.0

        # Channel 4: Current player's fences
        tensor[4, :, :] = my_fences / 10.0

        # Channel 5: Opponent's fences
        tensor[5, :, :] = opp_fences / 10.0

        return tensor

    def generate_batches(self):
        """
        Generate batches of training data from all files.

        In-memory mode (default): Shuffles all samples before generating batches.
        Streaming mode: Reads samples sequentially from files.
        """
        if self.stream:
            # Streaming mode: read from files sequentially
            states, policies, values = [], [], []

            # Iterate through all files
            for path in self.samples_paths:
                with open(path, 'rb') as f:
                    # Skip header
                    f.read(QSMP_HEADER_SIZE)

                    # Read all samples from this file
                    while True:
                        sample_data = f.read(TRAINING_SAMPLE_SIZE)
                        if len(sample_data) < TRAINING_SAMPLE_SIZE:
                            break

                        state_t, policy_t, value_t = self._parse_sample(sample_data)
                        states.append(state_t)
                        policies.append(policy_t)
                        values.append(value_t)

                        # Yield batch when full
                        if len(states) == self.batch_size:
                            yield (
                                torch.stack(states),
                                torch.stack(policies),
                                torch.stack(values)
                            )
                            states, policies, values = [], [], []

            # Yield remaining samples
            if states:
                yield (
                    torch.stack(states),
                    torch.stack(policies),
                    torch.stack(values)
                )
        else:
            # In-memory mode: shuffle and generate batches
            if self.samples is None:
                raise RuntimeError("Dataset not initialized with context manager")

            # Shuffle samples for better training
            import random
            indices = list(range(len(self.samples)))
            random.shuffle(indices)

            # Generate batches from shuffled indices
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                states = [self.samples[idx][0] for idx in batch_indices]
                policies = [self.samples[idx][1] for idx in batch_indices]
                values = [self.samples[idx][2] for idx in batch_indices]

                yield (
                    torch.stack(states),
                    torch.stack(policies),
                    torch.stack(values)
                )


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
