import ctypes
import subprocess
import numpy as np
import torch
from typing import Optional
import logging
from io import BufferedReader
import os


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

    def _fences_to_wall_tensor(self, horizontal: int, vertical: int) -> np.ndarray:
        """
        Convert 64-bit fence bitmaps to 2x8x8 wall tensor.

        The FenceGrid stores fences at intersection points:
        - bit r*8+c = fence placed at intersection (r,c)
        - horizontal[r,c] blocks movement between rows r and r+1
        - vertical[r,c] blocks movement between columns c and c+1
        """
        wall_tensor = np.zeros((2, 8, 8), dtype=np.float32)

        for r in range(8):
            for c in range(8):
                bit_idx = r * 8 + c
                if (horizontal >> bit_idx) & 1:
                    wall_tensor[0, r, c] = 1
                if (vertical >> bit_idx) & 1:
                    wall_tensor[1, r, c] = 1

        return wall_tensor

    def _state_to_tensors(self, state: SerializedNode):
        # Create pawn state tensor (2 x 9 x 9)
        pawn = np.zeros((2, 9, 9), dtype=np.float32)

        # P1 pawn position
        pawn[0, state.p1_row, state.p1_col] = 1

        # P2 pawn position
        pawn[1, state.p2_row, state.p2_col] = 1

        # Wall positions from fence bitmaps
        wall = self._fences_to_wall_tensor(state.fences_horizontal, state.fences_vertical)

        # Meta features (remaining fences)
        meta = np.array([state.p1_fences, state.p2_fences], dtype=np.float32)

        # Target value: game outcome z ∈ {-1, +1}
        # leopard outputs actual game outcomes in terminal_value field
        # +1 = P1 won, -1 = P2 won
        target = np.array([state.terminal_value], dtype=np.float32)

        return (
            torch.from_numpy(pawn),
            torch.from_numpy(wall),
            torch.from_numpy(meta),
            torch.from_numpy(target)
        )

    def generate_batches(self):
        if not self.process:
            raise RuntimeError("Dataset not initialized with context manager")

        pawns, walls, metas, targets = [], [], [], []

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

                state = SerializedNode.from_address(buf_addr + offset)
                pawn, wall, meta, target = self._state_to_tensors(state)

                pawns.append(pawn)
                walls.append(wall)
                metas.append(meta)
                targets.append(target)

                offset += node_size

                if len(pawns) == self.batch_size:
                    yield (
                        torch.stack(pawns),
                        torch.stack(walls),
                        torch.stack(metas),
                        torch.stack(targets)
                    )
                    pawns, walls, metas, targets = [], [], [], []

            except Exception as e:
                logging.error(f"Error processing state: {e}")
                raise e

        # Yield remaining samples
        if pawns:
            yield (
                torch.stack(pawns),
                torch.stack(walls),
                torch.stack(metas),
                torch.stack(targets)
            )
