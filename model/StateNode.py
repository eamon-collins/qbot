import ctypes

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
        cmd = ["leopard"]
        if self.load_model:
            cmd.extend(["-l", self.load_model])
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            self.process.terminate()
            self.process.wait()

    def _state_to_tensors(self, state: StateNode):
        # Create board state tensor (3 x 9 x 9)
        board = np.zeros((3, 9, 9), dtype=np.float32)
        
        # P1 pawn position
        board[0, state.p1[0], state.p1[1]] = 1
        
        # P2 pawn position
        board[1, state.p2[0], state.p2[1]] = 1
        
        # Wall positions
        for i in range(17):
            for j in range(9):
                if state.gamestate[i][j]:
                    board[2, i//2, j] = 1
        
        # Meta features
        meta = np.array([state.p1[2], state.p2[2]], dtype=np.float32)
        
        # Target value (normalized score to [-1, 1])
        target = np.array([np.tanh(state.score)], dtype=np.float32)
        
        return (
            torch.from_numpy(board),
            torch.from_numpy(meta),
            torch.from_numpy(target)
        )

    def generate_batches(self):
        if not self.process:
            raise RuntimeError("Dataset not initialized with context manager")
            
        boards, metas, targets = [], [], []
        
        while True:
            # Read binary data for one StateNode
            try:
                raw_data = self.process.stdout.buffer.read(ctypes.sizeof(StateNode))
                if not raw_data:
                    break
                    
                state = StateNode.from_buffer_copy(raw_data)
                board, meta, target = self._state_to_tensors(state)
                
                boards.append(board)
                metas.append(meta)
                targets.append(target)
                
                if len(boards) == self.batch_size:
                    yield (
                        torch.stack(boards),
                        torch.stack(metas),
                        torch.stack(targets)
                    )
                    boards, metas, targets = [], [], []
                    
            except Exception as e:
                logging.error(f"Error processing state: {e}")
                continue
                
        # Yield remaining samples
        if boards:
            yield (
                torch.stack(boards),
                torch.stack(metas),
                torch.stack(targets)
            )
