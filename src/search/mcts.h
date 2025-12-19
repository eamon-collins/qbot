#pragma once

/// Parallel MCTS Training System
///
/// Implements Monte Carlo Tree Search with:
/// - Virtual loss for tree parallelism (multiple threads explore different paths)
/// - Hybrid evaluation: random rollout + neural network (bootstraps toward NN)
/// - Early termination when both players out of fences
/// - Time-based checkpointing
/// - Pure tree building mode (single persistent tree)

#include "../tree/node_pool.h"
#include "../tree/StateNode.h"
#include "../util/pathfinding.h"
#include "../util/storage.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <mutex>
#include <random>
#include <stop_token>
#include <thread>
#include <vector>

// Forward declaration for optional NN inference
#ifdef ENABLE_INFERENCE
#include "../inference/inference.h"
#endif

namespace qbot {

// ============================================================================
// Configuration
// ============================================================================

/// MCTS configuration - all parameters in one place
struct MCTSConfig {
    // Selection parameters
    float c_puct = 1.5f;                       // PUCT exploration constant
    float fpu = 0.0f;                          // First play urgency for unvisited nodes
    int32_t virtual_loss_amount = 3;           // Virtual loss per selection step

    // Evaluation parameters
    int max_rollout_depth = 200;               // Max moves in random rollout

    // Threading
    int num_threads = 4;                       // Worker threads for parallel MCTS
    int checkpoint_interval_seconds = 300;     // Time between checkpoints (5 min default)

    // Paths
    std::filesystem::path checkpoint_path;     // Where to save checkpoints
    std::filesystem::path model_path;          // Optional NN model path
};

/// Result of selection phase - path from root to leaf
struct SelectionResult {
    std::vector<uint32_t> path;                // Node indices from root to leaf
    uint32_t leaf_idx{NULL_NODE};              // Final node (leaf or terminal)
    bool reached_terminal{false};              // True if hit a terminal game state
};

/// Training statistics - all atomic for thread safety
struct TrainingStats {
    std::atomic<uint64_t> total_iterations{0};  // MCTS iterations completed
    std::atomic<uint64_t> total_rollouts{0};    // Random rollouts performed
    std::atomic<uint64_t> nn_evaluations{0};    // NN evaluations performed
    std::atomic<uint32_t> max_depth{0};         // Deepest node reached
    std::chrono::steady_clock::time_point start_time;

    void reset() noexcept {
        total_iterations.store(0, std::memory_order_relaxed);
        total_rollouts.store(0, std::memory_order_relaxed);
        nn_evaluations.store(0, std::memory_order_relaxed);
        max_depth.store(0, std::memory_order_relaxed);
        start_time = std::chrono::steady_clock::now();
    }

    void update_max_depth(uint32_t depth) noexcept {
        uint32_t current = max_depth.load(std::memory_order_relaxed);
        while (depth > current &&
               !max_depth.compare_exchange_weak(current, depth,
                   std::memory_order_relaxed, std::memory_order_relaxed)) {
            // retry
        }
    }
};

// ============================================================================
// Selection Functions
// ============================================================================

/// Select the best child using PUCT formula
/// Uses existing puct_score() from StateNode.h
/// @param pool Node pool
/// @param parent_idx Parent node index
/// @param config MCTS configuration
/// @return Index of best child, or NULL_NODE if no children
[[nodiscard]] inline uint32_t select_child_puct(
    NodePool& pool,
    uint32_t parent_idx,
    const MCTSConfig& config) noexcept
{
    StateNode& parent = pool[parent_idx];
    if (!parent.has_children()) return NULL_NODE;

    uint32_t parent_visits = parent.stats.visits.load(std::memory_order_relaxed);

    uint32_t best_child = NULL_NODE;
    float best_score = -std::numeric_limits<float>::infinity();

    uint32_t child_idx = parent.first_child;
    while (child_idx != NULL_NODE) {
        StateNode& child = pool[child_idx];
        float score = puct_score(child.stats, parent_visits, config.c_puct, config.fpu);

        if (score > best_score) {
            best_score = score;
            best_child = child_idx;
        }
        child_idx = child.next_sibling;
    }

    return best_child;
}

/// Traverse from root to leaf, applying virtual loss along the path
/// Virtual loss discourages other threads from selecting the same path
/// @param pool Node pool
/// @param root_idx Starting node
/// @param config MCTS configuration
/// @return Selection result with path and leaf info
[[nodiscard]] inline SelectionResult select_to_leaf(
    NodePool& pool,
    uint32_t root_idx,
    const MCTSConfig& config) noexcept
{
    SelectionResult result;
    result.reached_terminal = false;

    uint32_t current = root_idx;
    while (current != NULL_NODE) {
        result.path.push_back(current);
        StateNode& node = pool[current];

        // Apply virtual loss as we descend
        node.stats.add_virtual_loss(config.virtual_loss_amount);

        if (node.is_terminal()) {
            result.reached_terminal = true;
            break;
        }

        if (!node.has_children()) {
            // Leaf node - needs expansion
            break;
        }

        current = select_child_puct(pool, current, config);
    }

    result.leaf_idx = result.path.empty() ? NULL_NODE : result.path.back();
    return result;
}

/// Remove virtual loss from all nodes in a path
inline void remove_virtual_loss(
    NodePool& pool,
    const std::vector<uint32_t>& path,
    int32_t amount) noexcept
{
    for (uint32_t idx : path) {
        pool[idx].stats.remove_virtual_loss(amount);
    }
}

// ============================================================================
// Evaluation Functions
// ============================================================================

/// Determine winner when both players are out of fences
/// Uses path length to goal - whoever is closer wins
/// Accounts for whose turn it is (tie goes to player about to move)
/// @param node Game state to evaluate
/// @return +1.0 for P1 win, -1.0 for P2 win
[[nodiscard]] inline float early_terminate_no_fences(const StateNode& node) noexcept {
    Pathfinder& pf = get_pathfinder();

    int p1_dist = pf.path_length(node.fences, node.p1, 8);  // P1 goal is row 8
    int p2_dist = pf.path_length(node.fences, node.p2, 0);  // P2 goal is row 0

    if (p1_dist < 0 || p2_dist < 0) {
        // Should never happen - someone is blocked
        return 0.0f;
    }

    // Account for whose turn it is
    // If it's P1's turn, P1 effectively has one less move to make
    if (node.is_p1_to_move()) {
        p1_dist--;
    } else {
        p2_dist--;
    }

    if (p1_dist < p2_dist) {
        return 1.0f;   // P1 wins
    } else if (p2_dist < p1_dist) {
        return -1.0f;  // P2 wins
    } else {
        // Tie - player whose turn it is wins (they move first)
        return node.is_p1_to_move() ? 1.0f : -1.0f;
    }
}

/// Lightweight simulation state for rollouts (copyable, no atomics)
struct SimState {
    Player p1;
    Player p2;
    FenceGrid fences;
    uint8_t flags{0};
    uint16_t ply{0};

    static constexpr uint8_t FLAG_P1_TO_MOVE = StateNode::FLAG_P1_TO_MOVE;

    explicit SimState(const StateNode& node)
        : p1(node.p1), p2(node.p2), fences(node.fences), flags(node.flags), ply(node.ply) {}

    [[nodiscard]] bool is_p1_to_move() const noexcept { return flags & FLAG_P1_TO_MOVE; }

    [[nodiscard]] int game_over() const noexcept {
        if (p1.row == 8) return 1;
        if (p2.row == 0) return -1;
        return 0;
    }

    [[nodiscard]] const Player& current_player() const noexcept {
        return is_p1_to_move() ? p1 : p2;
    }

    [[nodiscard]] const Player& opponent_player() const noexcept {
        return is_p1_to_move() ? p2 : p1;
    }

    [[nodiscard]] bool is_occupied(uint8_t row, uint8_t col) const noexcept {
        return (p1.row == row && p1.col == col) || (p2.row == row && p2.col == col);
    }

    /// Generate valid moves (simplified version for rollout)
    [[nodiscard]] std::vector<Move> generate_valid_moves() const noexcept {
        std::vector<Move> moves;
        const Player& curr = current_player();
        const Player& opp = opponent_player();
        const uint8_t r = curr.row;
        const uint8_t c = curr.col;

        // Early termination - just take shortest path
        if (p1.fences == 0 && p2.fences == 0) {
            Pathfinder& pf = get_pathfinder();
            uint8_t goal_row = is_p1_to_move() ? 8 : 0;
            auto path = pf.find_path(fences, curr, goal_row);
            if (path.size() > 1) {
                moves.push_back(Move::pawn(path[1].row, path[1].col));
            }
            return moves;
        }

        moves.reserve(140);

        // Pawn moves with jump logic
        auto can_move_up = [&](uint8_t from_row, uint8_t from_col) -> bool {
            return !fences.blocked_up(from_row, from_col);
        };
        auto can_move_down = [&](uint8_t from_row, uint8_t from_col) -> bool {
            return !fences.blocked_down(from_row, from_col);
        };
        auto can_move_left = [&](uint8_t from_row, uint8_t from_col) -> bool {
            return !fences.blocked_left(from_row, from_col);
        };
        auto can_move_right = [&](uint8_t from_row, uint8_t from_col) -> bool {
            return !fences.blocked_right(from_row, from_col);
        };

        // UP
        if (can_move_up(r, c)) {
            if (opp.row == r - 1 && opp.col == c) {
                if (can_move_up(r - 1, c)) {
                    moves.push_back(Move::pawn(r - 2, c));
                } else {
                    if (can_move_left(r - 1, c)) moves.push_back(Move::pawn(r - 1, c - 1));
                    if (can_move_right(r - 1, c)) moves.push_back(Move::pawn(r - 1, c + 1));
                }
            } else {
                moves.push_back(Move::pawn(r - 1, c));
            }
        }

        // DOWN
        if (can_move_down(r, c)) {
            if (opp.row == r + 1 && opp.col == c) {
                if (can_move_down(r + 1, c)) {
                    moves.push_back(Move::pawn(r + 2, c));
                } else {
                    if (can_move_left(r + 1, c)) moves.push_back(Move::pawn(r + 1, c - 1));
                    if (can_move_right(r + 1, c)) moves.push_back(Move::pawn(r + 1, c + 1));
                }
            } else {
                moves.push_back(Move::pawn(r + 1, c));
            }
        }

        // LEFT
        if (can_move_left(r, c)) {
            if (opp.row == r && opp.col == c - 1) {
                if (can_move_left(r, c - 1)) {
                    moves.push_back(Move::pawn(r, c - 2));
                } else {
                    if (can_move_up(r, c - 1)) moves.push_back(Move::pawn(r - 1, c - 1));
                    if (can_move_down(r, c - 1)) moves.push_back(Move::pawn(r + 1, c - 1));
                }
            } else {
                moves.push_back(Move::pawn(r, c - 1));
            }
        }

        // RIGHT
        if (can_move_right(r, c)) {
            if (opp.row == r && opp.col == c + 1) {
                if (can_move_right(r, c + 1)) {
                    moves.push_back(Move::pawn(r, c + 2));
                } else {
                    if (can_move_up(r, c + 1)) moves.push_back(Move::pawn(r - 1, c + 1));
                    if (can_move_down(r, c + 1)) moves.push_back(Move::pawn(r + 1, c + 1));
                }
            } else {
                moves.push_back(Move::pawn(r, c + 1));
            }
        }

        // Fence moves (simplified - no path validation in rollout for speed)
        if (curr.fences > 0) {
            for (uint8_t row = 0; row < 8; ++row) {
                for (uint8_t col = 0; col < 8; ++col) {
                    if (!fences.h_fence_blocked(row, col)) {
                        moves.push_back(Move::fence(row, col, true));
                    }
                    if (!fences.v_fence_blocked(row, col)) {
                        moves.push_back(Move::fence(row, col, false));
                    }
                }
            }
        }

        return moves;
    }

    /// Apply a move to this state
    void apply_move(Move move) noexcept {
        bool was_p1_turn = is_p1_to_move();
        if (move.is_pawn()) {
            if (was_p1_turn) {
                p1.row = move.row();
                p1.col = move.col();
            } else {
                p2.row = move.row();
                p2.col = move.col();
            }
        } else {
            if (move.is_horizontal()) {
                fences.place_h_fence(move.row(), move.col());
            } else {
                fences.place_v_fence(move.row(), move.col());
            }
            if (was_p1_turn) {
                p1.fences--;
            } else {
                p2.fences--;
            }
        }
        flags = was_p1_turn ? 0 : FLAG_P1_TO_MOVE;
        ply++;
    }
};

/// Early termination helper for SimState
[[nodiscard]] inline float early_terminate_simstate(const SimState& state) noexcept {
    Pathfinder& pf = get_pathfinder();

    int p1_dist = pf.path_length(state.fences, state.p1, 8);
    int p2_dist = pf.path_length(state.fences, state.p2, 0);

    if (p1_dist < 0 || p2_dist < 0) {
        return 0.0f;
    }

    if (state.is_p1_to_move()) {
        p1_dist--;
    } else {
        p2_dist--;
    }

    if (p1_dist < p2_dist) {
        return 1.0f;
    } else if (p2_dist < p1_dist) {
        return -1.0f;
    } else {
        return state.is_p1_to_move() ? 1.0f : -1.0f;
    }
}

/// Perform a random rollout from the given state to terminal or depth limit
/// Uses early termination when both players out of fences
/// @param start_node Starting game state
/// @param max_depth Maximum moves to simulate
/// @return Terminal value: +1.0 P1 wins, -1.0 P2 wins
[[nodiscard]] inline float random_rollout(
    const StateNode& start_node,
    int max_depth) noexcept
{
    thread_local std::mt19937 rng(std::random_device{}());

    // Create lightweight copy for simulation
    SimState sim(start_node);

    for (int depth = 0; depth < max_depth; ++depth) {
        // Check terminal
        int result = sim.game_over();
        if (result != 0) {
            return static_cast<float>(result);
        }

        // Early termination when fences exhausted
        if (sim.p1.fences == 0 && sim.p2.fences == 0) {
            return early_terminate_simstate(sim);
        }

        // Generate valid moves
        std::vector<Move> moves = sim.generate_valid_moves();
        if (moves.empty()) {
            return 0.0f;
        }

        // Pick random move
        std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
        Move move = moves[dist(rng)];

        // Apply move
        sim.apply_move(move);
    }

    // Reached depth limit without terminal - use heuristic
    return early_terminate_simstate(sim);
}

/// Evaluate a leaf node using hybrid strategy:
/// - Random rollout OR neural network, weighted by tree maturity
/// - More rollout when immature, more NN when mature
/// @param node Node to evaluate
/// @param config MCTS configuration
/// @param root_visits Root node visit count (for maturity estimate)
/// @param inference Optional NN inference engine
/// @return Value in [-1, 1] from P1's perspective
[[nodiscard]] inline float evaluate_leaf(
    const StateNode& node,
    const MCTSConfig& config,
    [[maybe_unused]] uint32_t root_visits,
    TrainingStats& stats,
    [[maybe_unused]] void* inference = nullptr) noexcept
{
    // Terminal nodes have known values
    if (node.is_terminal()) {
        return node.terminal_value;
    }

    // Early termination when fences exhausted (counts as rollout)
    if (node.p1.fences == 0 && node.p2.fences == 0) {
        stats.total_rollouts.fetch_add(1, std::memory_order_relaxed);
        return early_terminate_no_fences(node);
    }

#ifdef ENABLE_INFERENCE
    // Hybrid: choose between rollout and NN based on tree maturity
    if (inference != nullptr) {
        auto* model = static_cast<ModelInference*>(inference);
        if (model->is_ready()) {
            // P(use_nn) = min(0.9, sqrt(root_visits) / 1000)
            float p_nn = std::min(0.9f, std::sqrt(static_cast<float>(root_visits)) / 1000.0f);

            thread_local std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);

            if (dist(rng) < p_nn) {
                stats.nn_evaluations.fetch_add(1, std::memory_order_relaxed);
                return model->evaluate_node(&node);
            }
        }
    }
#endif

    // Default: random rollout
    stats.total_rollouts.fetch_add(1, std::memory_order_relaxed);
    return random_rollout(node, config.max_rollout_depth);
}

// ============================================================================
// MCTS Engine
// ============================================================================

/// Parallel MCTS search engine
///
/// Manages worker threads that perform MCTS iterations concurrently.
/// Uses virtual loss for tree parallelism and supports periodic checkpointing.
class MCTSEngine {
public:
    explicit MCTSEngine(MCTSConfig config = MCTSConfig{})
        : config_(std::move(config)) {}

    ~MCTSEngine() {
        stop();
    }

    // Non-copyable, non-movable
    MCTSEngine(const MCTSEngine&) = delete;
    MCTSEngine& operator=(const MCTSEngine&) = delete;
    MCTSEngine(MCTSEngine&&) = delete;
    MCTSEngine& operator=(MCTSEngine&&) = delete;

    /// Start training workers
    /// All threads run MCTS iterations from the same root
    /// @param pool Node pool (must outlive the engine while running)
    /// @param root_idx Root of the search tree
    void start_training(NodePool& pool, uint32_t root_idx);

    /// Stop all workers gracefully
    void stop();

    /// Check if training is running
    [[nodiscard]] bool is_running() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    /// Get current statistics
    [[nodiscard]] const TrainingStats& stats() const noexcept { return stats_; }

    /// Get configuration
    [[nodiscard]] const MCTSConfig& config() const noexcept { return config_; }

    /// Set optional neural network inference engine
    void set_inference(void* inference) noexcept { inference_ = inference; }

private:
    /// Single MCTS iteration: select -> expand -> evaluate -> backprop
    void mcts_iteration(NodePool& pool, uint32_t root_idx);

    /// Backpropagate value up the tree and remove virtual loss
    void backpropagate(NodePool& pool, const std::vector<uint32_t>& path,
                       float value) noexcept;

    /// Worker thread main loop
    void worker_loop(std::stop_token stop_token, int thread_id,
                     NodePool& pool, uint32_t root_idx);

    /// Checkpoint thread: periodically saves tree
    void checkpoint_loop(std::stop_token stop_token, NodePool& pool, uint32_t root_idx);

    /// Signal workers to pause for checkpointing
    void pause_workers();

    /// Resume workers after checkpoint
    void resume_workers();

    /// Print current statistics
    void print_stats(const NodePool& pool) const;

    MCTSConfig config_;
    TrainingStats stats_;
    void* inference_{nullptr};

    // Threading
    std::vector<std::jthread> workers_;
    std::jthread checkpoint_thread_;

    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};

    // Synchronization for checkpointing
    std::mutex pause_mutex_;
    std::condition_variable pause_cv_;
    std::atomic<int> workers_paused_{0};

    // Expansion mutex - prevents multiple threads expanding same node
    std::mutex expansion_mutex_;

    // Reference to pool and root for checkpointing
    NodePool* pool_ptr_{nullptr};
    uint32_t root_idx_{NULL_NODE};
};

} // namespace qbot
