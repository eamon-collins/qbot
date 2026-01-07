# Quoridor MCTS Bot - Project Guidelines

## Project Overview
A Monte Carlo Tree Search implementation for the board game Quoridor, designed for:
- Efficient multithreaded training/simulation
- Clean separation between engine (C++) and UI (separate process, IPC-based)
- Future neural network policy/value integration (AlphaZero-style)
- Modern C++20 standards

The code in `old_src` is my initial attempt, from when I was a bad programmer. Now that I am better, I am rewriting the codebase in `src`.
Let this code guide overarching design principles, things like I want MCTS with a neural net value function.
I want high performance, efficiently multithreaded simulation and tree building, alongside network training.
I want to be able to produce and load efficiently sized binary representations of the tree so that I can save intermediate steps, and eventually so I can load a pretrained tree to give the bot a head start against players.
For many other things, like not knowing move semantics, pool allocators, any template metaprogramming or compile time optimizations at all, and favoring a terribly complex and difficult to reason with board representation, and eschewing any code safety or good tests, do the opposite of this example.

## Architecture Principles

### Separation of Concerns
- **Engine**: Pure C++ MCTS implementation, no UI dependencies
- **UI**: Separate process communicating via IPC (stdin/stdout JSON or shared memory)
- **Training**: Parallelized self-play with checkpointing
- **Inference**: Optional neural network integration via ONNX or libtorch

### Structure Guidelines
Classes:
- Game: handles running the game, both self play and vs human. Will always be associated with a gamestate tree, and will perform operations on it/handle traversal. Will be responsible for communicating with the GUI. Responsible for administering self play and training, building the tree, and tracking the gamestate like if someone has won or not
- StateNode: The node representation. Stores the full gamestate at a particular turn. Organized into a tree where each child is a next potential move, and whose turn it is alternates as you go down the tree. Handles operations like generating valid moves and constructing its own children using those moves. The Quoridor rules are mostly encoded in this logic. Also responsible for scoring itself, and maybe for selecting the best next move during playout and other local play out functions.

### Comment Style
Cut down on comment verbosity. The code should for the most part be self documenting. When there is something a skilled human might not understand from reading the code, it can be commented. Optimization focused code and particular usage notes to avoid bugs can also be noted. Prefer 'why' over 'how'. Prefer to put information in a comment on top of the function explaining it all rather than interspersed in the function.

### Tree Structure (from papers)
Per AlphaGo Zero and MCTS papers, store edge statistics:
- `N(s,a)`: visit count
- `W(s,a)`: total action value  
- `Q(s,a)`: mean action value = W/N
- `P(s,a)`: prior probability (from policy network or uniform)

Selection uses PUCT formula:
```
U(s,a) = c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
select argmax_a [Q(s,a) + U(s,a)]
```

### Parallelization Strategy (from papers)
Use **virtual loss** for tree parallelism:
- When thread selects an edge, temporarily add virtual losses
- This discourages other threads from selecting the same path
- Remove virtual loss after backpropagation
- Atomic operations on visit counts; lock-free where possible

### Memory Management (from memory-bounded MCTS paper)
For bounded memory scenarios:
- Pre-allocate node pool
- Use LRU-based node recycling (FIFO queue of accessed nodes)
- Left-child right-sibling tree representation saves memory

## Code Style

### Modern C++
- Use `std::span` for non-owning views
- Use `std::optional` for nullable returns
- Prefer `constexpr` for compile-time constants
- full use of concepts and template metaprogramming where useful
- Use `[[nodiscard]]` for functions with important return values
- Structured bindings for tuple-like returns

### Threading
- Use `std::atomic<uint32_t>` for visit counts (not `int`)
- `std::atomic<float>` for scores where supported, else use mutex
- `std::jthread` for RAII thread management
- `std::stop_token` for cooperative cancellation

### Error Handling
- No exceptions in hot paths (use `std::expected` or error codes)
- Assertions for invariants in debug builds
- Clear preconditions documented

## File Organization

```
src/
  core/
    types.h          # Move, Position, Player types
    Game.h           # Game state (no tree logic)
    Game.cpp
  tree/
    Node.h           # TreeNode with edge statistics
    node_pool.h      # Memory-bounded node allocation
    Tree.h           # MCTS tree operations
    Tree.cpp
  search/
    mcts.h           # MCTS algorithm
    mcts.cpp
    selection.h      # UCB/PUCT selection policies
    evaluation.h     # Rollout and NN evaluation interface
  ipc/
    protocol.h       # UI communication protocol
    protocol.cpp
tests/
  test_state.cpp
  test_mcts.cpp
  test_tree.cpp
```

## Old Code
this code is my initial attempt, from when I was a bad programmer. Now that I am better, I am rewriting the codebase.
```
old_src/*
```

## Testing Strategy

Assume we are already in the conda env "qenv" which should allow us to compile and run all tests, the gui, pytorch/libtorch for compiling and training the model, and other useful things. If we need another package, ask before installing, but install with conda if possible to keep the project dependencies contained.

### Unit Tests (GoogleTest)
- State transition correctness
- Move generation (pawn moves, walls, blocking detection)
- MCTS selection convergence
- Thread safety validation

### Integration Tests
- Self-play game completion
- Memory bounds respected
- IPC protocol round-trip

### Property-Based Tests
- Random game sequences always terminate
- Wall blocking never creates unreachable goals
- Visit counts monotonically increase

## Build System

Use CMake with presets:
```cmake
cmake_minimum_required(VERSION 3.20)
project(quoridor_mcts CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Debug/Release configurations
# Optional: -fsanitize=thread for TSAN testing
```

## Performance Targets

- Node expansion: < 1μs
- Selection to leaf: < 10μs for depth 100
- Full iteration: < 50μs (excluding NN inference)
- Memory per node: ~64 bytes

## Key Differences from Original Implementation

### Issues in Original Code
1. **`std::vector<StateNode> children`**: Causes pointer invalidation on resize, thread-unsafe
2. **Full state copy per node**: Wastes memory; should store only edge transitions
3. **Python interop for visualization**: Breaks clean separation
4. **Mutex around entire backprop**: Too coarse; use per-node atomics
5. **No node pooling**: Leads to fragmentation and cache misses
6. **Score as sum not mean**: Should be W/N for proper Q-value

### Recommended Changes
1. Store children as indices into node pool, not vector of objects
2. Incrementally compute state from root when needed, or use Zobrist hashing
3. UI via separate process with JSON-over-stdout
4. Per-edge atomic statistics
5. Pre-allocated node arena with recycling
6. Proper Q = W/N calculation with FPU (first play urgency)

## References

- AlphaGo Zero paper: PUCT selection, combined policy/value network
- "Progressive Strategies for MCTS": Progressive bias, unpruning
- "Memory-Bounded MCTS": Node recycling, LRU caching
- "Real-time Enhancements": Virtual loss, tree reuse
