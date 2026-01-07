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

### Complete Training Workflow

The training loop alternates between:
1. **Tree building**: MCTS explores the game tree, accumulating visit statistics
2. **Neural network training**: Train on the tree's value estimates
3. **Integration**: Use the improved NN to guide MCTS evaluation

#### Step 1: Build a search tree with MCTS

```bash
# Build tree with 8 threads (Ctrl+C to stop, auto-saves on exit)
./qbot -b -t 8 -s tree.qbot

# Continue building from existing tree
./qbot -b -t 8 -l tree.qbot -s tree.qbot
```

Command line options:
- `-b, --train` - Training mode (self-play)
- `-t, --threads N` - Number of MCTS threads (default: 4)
- `-s, --save FILE` - Save tree to file (default: tree.qbot)
- `-l, --load FILE` - Load existing tree
- `-m, --model FILE` - Load TorchScript model for NN evaluation
- `-v, --verbose` - Verbose output

#### Step 2: Train neural network on the tree

```bash
cd train/

# Train new model
python resnet.py \
    --load-tree ../build/tree.qbot \
    --save-model model.pt \
    --epochs 100 \
    --batch-size 64

# Continue training existing model
python resnet.py \
    --load-tree ../build/tree.qbot \
    --load-model model.pt \
    --save-model model.pt \
    --epochs 50
```

#### Step 3: Export model for C++ inference

```bash
python resnet.py \
    --load-model model.pt \
    --save-model model_traced.pt \
    --export
```

#### Step 4: Use trained model in MCTS

```bash
# Build tree with NN-guided evaluation
./qbot -b -t 8 -l tree.qbot -s tree.qbot -m model_traced.pt
```

#### Iterative Training Loop

For best results, repeat steps 1-4:
```bash
# Initial tree building (no model)
./qbot -b -t 8 -s tree.qbot &
sleep 3600 && kill %1  # Run for 1 hour

# Train model on tree
cd train
python resnet.py --load-tree ../build/tree.qbot --save-model model.pt --epochs 100
python resnet.py --load-model model.pt --save-model ../build/model_traced.pt --export

# Continue tree building with model
cd ../build
./qbot -b -t 8 -l tree.qbot -s tree.qbot -m model_traced.pt &
sleep 3600 && kill %1

# Retrain model on improved tree
cd ../train
python resnet.py --load-tree ../build/tree.qbot --load-model model.pt --save-model model.pt --epochs 50
# ... repeat
```

### AlphaZero-Style Training

For pure neural network training without rollouts, use self-play mode:

#### Self-Play

Generate training data by playing complete games with NN evaluation:

```bash
# Run 500 self-play games, 800 sims/move, save tree for training
./qbot --selfplay -m model/current_best.pt -g 500 -n 800 -s tree.qbot

# With temperature settings (exploration vs exploitation)
./qbot --selfplay -m model/current_best.pt -g 1000 -n 800 \
    --temperature 1.0 --temp-drop 30 -s tree.qbot
```

Options:
- `--selfplay` - Enable self-play mode (requires model)
- `-g, --games N` - Number of games to play (default: 1000)
- `-n, --simulations N` - MCTS simulations per move (default: 800)
- `--temperature T` - Softmax temperature for move selection (default: 1.0)
- `--temp-drop PLY` - Ply to drop temperature to 0 for deterministic play (default: 30)

#### Arena (Model Evaluation)

Evaluate a candidate model against the current best:

```bash
# Evaluate candidate vs current, promote if candidate wins >= 55%
./qbot --arena -m model/current_best.pt --candidate model/new_model.pt \
    --arena-games 100 -n 400

# Custom threshold and output path
./qbot --arena -m model/current_best.pt --candidate model/new_model.pt \
    --arena-games 200 -n 800 --win-threshold 0.55 --best-model model/current_best.pt
```

Options:
- `--arena` - Enable arena mode
- `-m, --model` - Current best model path
- `--candidate` - Candidate model to evaluate
- `--arena-games N` - Number of games to play (default: 100)
- `--win-threshold F` - Win rate to replace best model (default: 0.55)
- `--best-model PATH` - Where to save winning model (default: model/current_best.pt)

The arena alternates which model plays as P1/P2 for fairness. If the candidate wins >= 55% of decisive games, it replaces the current best.

#### Automated Training Loop

Use `train_loop.py` to automate the self-play → train → evaluate cycle:

```bash
cd train/
python train_loop.py --iterations 10 --games 500 --simulations 800 --epochs 50
```

### Tree Memory Limits

The tree will automatically limit expansion when approaching memory bounds:
- Default limit: 40GB
- Below 80%: expand all nodes
- 80-95%: only expand high-visit nodes
- Above 95%: use NN evaluation only (no expansion)

### Training Options

`resnet.py` options:
- `--load-tree` - Tree file to train on (uses `leopard` internally)
- `--load-model` - Load existing model weights
- `--save-model` - Save model weights after training
- `--export` - Export TorchScript model for C++ inference
- `--batch-size` - Training batch size (default: 64)
- `--epochs` - Number of training epochs (default: 100)
- `--log-level` - DEBUG, INFO, WARNING, ERROR (default: INFO)

### Low-level Tools

```bash
# Dump tree to stdout as binary SerializedNode structs
./leopard tree.qbot > training_data.bin

# Inspect first node (56 bytes per node)
./leopard tree.qbot 2>/dev/null | head -c 56 | xxd
```
