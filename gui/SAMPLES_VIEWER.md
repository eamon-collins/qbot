# Training Samples Viewer

The GUI now includes a training samples viewer mode that allows you to browse through `.qsamples` files and inspect the training data.

## Usage

```bash
cd gui
python main.py --samples path/to/file.qsamples
```

For example:
```bash
python main.py --samples ../samples/tree_0.qsamples
```

## Features

### Display
- Shows the board state for each training sample
- Displays player positions and fence counts
- Renders all walls on the board
- Shows the evaluation score (value) for the current player
- Indicates whose turn it is

### Navigation
- **Left Arrow / A**: Previous sample
- **Right Arrow / D**: Next sample
- **Home**: Jump to first sample
- **End**: Jump to last sample
- **Page Up**: Jump back 10 samples
- **Page Down**: Jump forward 10 samples
- **Q / Escape**: Quit viewer

### Stats Display (Terminal)
For each sample, the terminal displays:
- Sample number and total count
- Current player (P1 or P2)
- Value (from current player's perspective)
- Player positions (x, y) and fence counts
- Number of horizontal and vertical walls
- **Top 10 policy moves** with probabilities:
  - Move type (pawn, h_wall, v_wall)
  - Position (x, y)
  - Probability percentage

## What is a .qsamples file?

`.qsamples` files contain pre-computed training samples from self-play games. Each sample includes:
- **Board state**: Player positions, walls, fence counts
- **Policy target**: MCTS visit distribution (what moves were explored)
- **Value target**: Game outcome from current player's perspective

These files are generated during self-play (with the `--selfplay` flag in qbot) and are used for training the neural network.

## Implementation Details

The viewer directly parses the binary `.qsamples` format:
- Header (64 bytes): Magic number, version, sample count
- Samples (864 bytes each):
  - CompactState (24 bytes): Positions, fences, flags
  - Policy (836 bytes): 209 floats for action probabilities
  - Value (4 bytes): Game outcome

The state is always stored from the current player's perspective, which the viewer correctly interprets using the flags field to determine whose turn it is.
