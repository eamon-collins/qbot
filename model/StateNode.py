import ctypes
import subprocess
import numpy as np
import torch
from typing import Optional
import logging
from io import BufferedReader


# Mirror the C++ StateNode structure
class StateNode(ctypes.Structure):
    _fields_ = [
        ("move", ctypes.c_int * 3),     # type, row, col
        ("p1", ctypes.c_int * 3),       # row, col, numFences
        ("p2", ctypes.c_int * 3),       # row, col, numFences
        ("turn", ctypes.c_bool),
        ("gamestate", (ctypes.c_bool * 9) * 17),  # 17x9 bool array
        ("score", ctypes.c_double),
        ("visits", ctypes.c_int),
        ("ply", ctypes.c_int)
    ]


class QuoridorDataset:
    def __init__(self, load_model: Optional[str], batch_size: int):
        self.load_model = load_model
        self.batch_size = batch_size
        self.process = None
        
    def __enter__(self):
        cmd = ["./leopard"]
        if self.load_model:
            cmd.extend([self.load_model])
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE
            # stderr=subprocess.PIPE,
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            self.process.terminate()
            self.process.wait()

    def _state_to_tensors(self, state: StateNode):
        # Create pawn state tensor (2 x 9 x 9)
        pawn = np.zeros((2, 9, 9), dtype=np.float32)
        
        # P1 pawn position
        pawn[0, state.p1[0], state.p1[1]] = 1
        
        # P2 pawn position
        pawn[1, state.p2[0], state.p2[1]] = 1
        
        # Wall positions
        wall = self._gamestate_to_wall_tensor(state.gamestate)
        
        # Meta features
        meta = np.array([state.p1[2], state.p2[2]], dtype=np.float32)
        
        # Target value (normalized score to [-1, 1])
        target = np.array([np.tanh(state.score)], dtype=np.float32)
        
        return (
            torch.from_numpy(pawn),
            torch.from_numpy(wall),
            torch.from_numpy(meta),
            torch.from_numpy(target)
        )

    def _gamestate_to_wall_tensor(self, gamestate) -> np.ndarray:
        """
        Convert 17x9 gamestate matrix into 2x8x8 wall tensor.
        Returns numpy array with shape (2,8,8) where:
            - First channel (0) contains horizontal walls 
            - Second channel (1) contains vertical walls
        Values are 1 where a wall exists, 0 otherwise
        """
        wall_tensor = np.zeros((2, 8, 8), dtype=np.float32)
        
        # Horizontal walls come from odd rows in gamestate
        # A horizontal wall exists if two adjacent cells in an odd row are True
        for i in range(8):  # 8 possible horizontal wall rows
            row = 2*i + 1  # odd rows contain horizontal walls
            for j in range(8):  # 8 possible horizontal wall positions per row
                if gamestate[row][j] and gamestate[row][j+1]:
                    wall_tensor[0, i, j] = 1
                    
        # Vertical walls come from even rows in gamestate
        # A vertical wall exists if two adjacent even rows have True in same column
        for i in range(8):  # 8 possible vertical wall rows
            row = 2*i  # even rows contain vertical walls
            for j in range(8):  # 8 possible vertical wall positions per row
                if gamestate[row][j] and gamestate[row+2][j]:
                    wall_tensor[1, i, j] = 1
                    
        return wall_tensor

    def generate_batches(self):
        if not self.process:
            raise RuntimeError("Dataset not initialized with context manager")
            
        pawns, walls, metas, targets = [], [], [], []
        
        # Set up buffers for quick ctypes reading
        node_size = ctypes.sizeof(StateNode)
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

                # print(len(read_data))
                # print(read_data[offset:ctypes.sizeof(StateNode)])
                    
                state = StateNode.from_address(buf_addr + offset)
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
                # continue
                raise e
                
        # Yield remaining samples
        if pawns:
            yield (
                torch.stack(pawns),
                torch.stack(walls),
                torch.stack(metas),
                torch.stack(targets)
            )
