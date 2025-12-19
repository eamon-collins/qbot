# Quoridor MCTS Bot

## Build

```bash
mkdir -p build && cd build
cmake ..
make              # Debug build (qbot)
make fast         # Optimized build (-O3 -march=native -flto)
make leopard      # Tree file dumper for training
make test_game    # Build game tests
make test_storage # Build storage tests
```

With neural network inference (requires libtorch):
```bash
conda activate qenv
cmake .. -DENABLE_INFERENCE=ON
make
```

## Run Tests

```bash
# All tests
./test_game
./test_storage

# Useful specific tests (verbose output when run alone)
# builds a tree for 3 seconds and prints how many nodes it created
./test_game --gtest_filter=GameTest.BenchmarkBuildTree
# builds tree at random with low branching factor until someone wins, then prints whole game path.
./test_game --gtest_filter=GameTest.BuildTreeUntilWinAndPrintPath
```

## GUI Visualization

Run the pygame GUI server (requires pygame and websockets):
```bash
python gui/main.py
```

With the GUI running on port 8765, tests like `BuildTreeUntilWinAndPrintPath` will visualize the game path instead of printing to stdout.

## Training

Generate training data from a tree file:
```bash
./leopard /path/to/tree.qbot > training_data.bin
```

Train with `resnet.py` (from `train/` directory):
```bash
# Train new model on tree file
python resnet.py --load-tree /path/to/tree.qbot --save-model model.pt --epochs 100

# Continue training existing model
python resnet.py --load-tree /path/to/tree.qbot --load-model model.pt --save-model model.pt

# Export model for C++ inference
python resnet.py --load-model model.pt --save-model model_traced.pt --export
```

Options:
- `--load-tree` - Tree file to train on
- `--load-model` - Load existing model weights
- `--save-model` - Save model weights after training
- `--export` - Export TorchScript model for C++ inference (requires `--load-model`)
- `--batch-size` - Training batch size (default: 64)
- `--epochs` - Number of training epochs (default: 100)
- `--log-level` - DEBUG, INFO, WARNING, ERROR (default: INFO)
