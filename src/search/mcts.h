#pragma once

/// Parallel MCTS with Neural Network Evaluation
///
/// Implements Monte Carlo Tree Search with:
/// - Virtual loss for tree parallelism (multiple threads explore different paths)
/// - Neural network evaluation at leaf nodes (AlphaZero-style)
/// - Early termination when both players out of fences
/// - Time-based checkpointing
/// - Self-play mode for training data generation

#include "../tree/node_pool.h"
#include "../tree/StateNode.h"
#include "../util/pathfinding.h"
#include "../util/storage.h"

#include <array>
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

// Include inference for NN evaluation
#ifdef QBOT_ENABLE_INFERENCE
#include "../inference/inference.h"
#endif

namespace qbot {

// ============================================================================
// Configuration
// ============================================================================

/// Tree memory bounds configuration
/// Controls when to stop expanding and use NN evaluation instead
struct TreeBoundsConfig {
    size_t max_bytes = 40ULL * 1024 * 1024 * 1024;  // 40GB default
    float soft_limit_ratio = 0.80f;   // Start being selective about expansion
    float hard_limit_ratio = 0.95f;   // Stop expanding entirely
    uint32_t min_visits_to_expand = 8; // Min visits at soft limit to expand
    bool enable_recycling = false;    // LRU recycling when full (future)
};

/// MCTS configuration - all parameters in one place
struct MCTSConfig {
    // Selection parameters
    float c_puct = 1.5f;                       // PUCT exploration constant
    float fpu = 0.0f;                          // First play urgency for unvisited nodes
    int32_t virtual_loss_amount = 3;           // Virtual loss per selection step

    // Tree bounds
    TreeBoundsConfig bounds;                   // Memory limit configuration

    // Threading
    int num_threads = 4;                       // Worker threads for parallel MCTS
    int checkpoint_interval_seconds = 300;     // Time between checkpoints (5 min default)

    // Paths
    std::filesystem::path checkpoint_path;     // Where to save checkpoints
    std::filesystem::path model_path;          // Optional NN model path
};

/// Result of expansion decision
enum class ExpansionDecision {
    Expand,          // Expand the node normally
    UseNNEvaluation, // Skip expansion, use NN to evaluate directly
    AlreadyExpanded, // Node was already expanded
    Terminal         // Node is terminal, no expansion needed
};

/// Result of selection phase - path from root to leaf
struct SelectionResult {
    std::vector<uint32_t> path;                // Node indices from root to leaf
    uint32_t leaf_idx{NULL_NODE};              // Final node (leaf or terminal)
    bool reached_terminal{false};              // True if hit a terminal game state
    ExpansionDecision expansion{ExpansionDecision::Expand}; // What to do with leaf
};

/// Training statistics - all atomic for thread safety
struct TrainingStats {
    std::atomic<uint64_t> total_iterations{0};  // MCTS iterations completed
    std::atomic<uint64_t> nn_evaluations{0};    // NN evaluations performed
    std::atomic<uint64_t> skipped_expansions{0}; // Expansions skipped due to memory limit
    std::atomic<uint32_t> max_depth{0};         // Deepest node reached
    std::chrono::steady_clock::time_point start_time;

    void reset() noexcept {
        total_iterations.store(0, std::memory_order_relaxed);
        nn_evaluations.store(0, std::memory_order_relaxed);
        skipped_expansions.store(0, std::memory_order_relaxed);
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
// Expansion Policy
// ============================================================================

/// Determine whether a node should be expanded based on memory pressure
/// Returns true if expansion is allowed, false if should use NN evaluation instead
///
/// Policy:
/// - Below soft limit: always expand
/// - Between soft and hard limit: expand only if visit count exceeds threshold
/// - Above hard limit: never expand (use NN evaluation)
///
/// The visit threshold scales linearly with memory pressure, making expansion
/// progressively more selective as the tree approaches capacity.
[[nodiscard]] inline bool should_expand(
    const NodePool& pool,
    const StateNode& node,
    const TreeBoundsConfig& bounds) noexcept
{
    size_t current_bytes = pool.memory_usage_bytes();
    float utilization = static_cast<float>(current_bytes) / static_cast<float>(bounds.max_bytes);

    // Below soft limit: always expand
    if (utilization < bounds.soft_limit_ratio) {
        return true;
    }

    // Above hard limit: never expand
    if (utilization >= bounds.hard_limit_ratio) {
        return false;
    }

    // In the soft-to-hard zone: scale threshold based on pressure
    // pressure goes from 0.0 (at soft limit) to 1.0 (at hard limit)
    float pressure = (utilization - bounds.soft_limit_ratio) /
                     (bounds.hard_limit_ratio - bounds.soft_limit_ratio);

    // Threshold increases as we approach hard limit
    // At soft limit: threshold = min_visits_to_expand
    // At hard limit: threshold = min_visits_to_expand * 5
    uint32_t threshold = static_cast<uint32_t>(
        bounds.min_visits_to_expand * (1.0f + pressure * 4.0f));

    uint32_t visits = node.stats.visits.load(std::memory_order_relaxed);
    return visits >= threshold;
}

/// Decide what to do with a leaf node
[[nodiscard]] inline ExpansionDecision decide_expansion(
    const NodePool& pool,
    const StateNode& node,
    const TreeBoundsConfig& bounds) noexcept
{
    if (node.is_terminal()) {
        return ExpansionDecision::Terminal;
    }

    if (node.is_expanded()) {
        return ExpansionDecision::AlreadyExpanded;
    }

    if (should_expand(pool, node, bounds)) {
        return ExpansionDecision::Expand;
    }

    return ExpansionDecision::UseNNEvaluation;
}

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
/// Also determines whether the leaf should be expanded based on memory bounds
/// @param pool Node pool
/// @param root_idx Starting node
/// @param config MCTS configuration
/// @return Selection result with path, leaf info, and expansion decision
[[nodiscard]] inline SelectionResult select_to_leaf(
    NodePool& pool,
    uint32_t root_idx,
    const MCTSConfig& config) noexcept
{
    SelectionResult result;
    result.reached_terminal = false;
    result.expansion = ExpansionDecision::Expand;

    uint32_t current = root_idx;
    while (current != NULL_NODE) {
        result.path.push_back(current);
        StateNode& node = pool[current];

        // Apply virtual loss as we descend
        node.stats.add_virtual_loss(config.virtual_loss_amount);

        if (node.is_terminal()) {
            result.reached_terminal = true;
            result.expansion = ExpansionDecision::Terminal;
            break;
        }

        if (!node.has_children()) {
            // Leaf node - decide whether to expand based on memory pressure
            result.expansion = decide_expansion(pool, node, config.bounds);
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


/// Evaluate a leaf node
/// Uses neural network if available, otherwise uses heuristic (path distance)
/// @param node Node to evaluate
/// @param stats Training stats (updated with evaluation count)
/// @param inference Optional NN inference engine
/// @return Value in [-1, 1] from P1's perspective
[[nodiscard]] inline float evaluate_leaf(
    const StateNode& node,
    TrainingStats& stats,
    [[maybe_unused]] void* inference = nullptr) noexcept
{
    // Terminal nodes have known values
    if (node.is_terminal()) {
        return node.terminal_value;
    }

    // Early termination when fences exhausted - use path distance heuristic
    if (node.p1.fences == 0 && node.p2.fences == 0) {
        return early_terminate_no_fences(node);
    }

#ifdef QBOT_ENABLE_INFERENCE
    // Use neural network evaluation if available
    if (inference != nullptr) {
        auto* model = static_cast<ModelInference*>(inference);
        if (model->is_ready()) {
            stats.nn_evaluations.fetch_add(1, std::memory_order_relaxed);
            return model->evaluate_node(&node);
        }
    }
#endif

    // No NN available - use path distance heuristic
    return early_terminate_no_fences(node);
}

// ============================================================================
// Self-Play Support
// ============================================================================

/// Configuration for self-play game generation
struct SelfPlayConfig {
    int simulations_per_move = 800;       // MCTS iterations per move
    float temperature = 1.0f;              // Softmax temperature for move selection
    int temperature_drop_ply = 30;         // After this ply, use temperature → 0
    bool stochastic = true;                // True = sample from policy, False = argmax
};

/// Compute policy distribution from child Q-values
/// π_i = exp(Q_i / τ) / Σ exp(Q_j / τ)  (softmax)
/// @param pool Node pool
/// @param parent_idx Parent node index
/// @param temperature Softmax temperature (higher = more uniform)
/// @return Vector of (move, probability) pairs
[[nodiscard]] inline std::vector<std::pair<Move, float>> compute_policy_from_q(
    NodePool& pool,
    uint32_t parent_idx,
    float temperature) noexcept
{
    std::vector<std::pair<Move, float>> policy;
    if (!pool[parent_idx].has_children()) return policy;

    // First pass: collect Q-values and find max for numerical stability
    float max_q = -std::numeric_limits<float>::infinity();
    uint32_t child = pool[parent_idx].first_child;
    while (child != NULL_NODE) {
        float q = pool[child].stats.Q();
        max_q = std::max(max_q, q);
        policy.push_back({pool[child].move, q});
        child = pool[child].next_sibling;
    }

    if (policy.empty()) return policy;

    // Handle temperature = 0 (deterministic)
    if (temperature <= 0.0f) {
        // Find argmax
        size_t best_idx = 0;
        float best_q = policy[0].second;
        for (size_t i = 1; i < policy.size(); ++i) {
            if (policy[i].second > best_q) {
                best_q = policy[i].second;
                best_idx = i;
            }
        }
        // Set probability 1.0 for best, 0.0 for others
        for (size_t i = 0; i < policy.size(); ++i) {
            policy[i].second = (i == best_idx) ? 1.0f : 0.0f;
        }
        return policy;
    }

    // Second pass: softmax with temperature
    float sum = 0.0f;
    for (auto& [move, q] : policy) {
        q = std::exp((q - max_q) / temperature);  // Subtract max for stability
        sum += q;
    }

    // Normalize
    if (sum > 0.0f) {
        for (auto& [move, prob] : policy) {
            prob /= sum;
        }
    }

    return policy;
}

/// Select a move from the policy distribution
/// @param policy Vector of (move, probability) pairs
/// @param stochastic If true, sample from distribution; if false, take argmax
/// @return Selected move
[[nodiscard]] inline Move select_move_from_policy(
    const std::vector<std::pair<Move, float>>& policy,
    bool stochastic) noexcept
{
    if (policy.empty()) return Move{};

    if (!stochastic) {
        // Deterministic: return move with highest probability
        auto best = std::max_element(policy.begin(), policy.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        return best->first;
    }

    // Stochastic: sample from distribution
    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);

    float cumulative = 0.0f;
    for (const auto& [move, prob] : policy) {
        cumulative += prob;
        if (r <= cumulative) {
            return move;
        }
    }

    // Fallback (shouldn't happen with proper normalization)
    return policy.back().first;
}

/// Advance the root to a child after a move is played (tree reuse)
/// @param pool Node pool
/// @param current_root Current root node index
/// @param played_move The move that was played
/// @return New root index, or NULL_NODE if move not found
[[nodiscard]] inline uint32_t advance_root(
    NodePool& pool,
    uint32_t current_root,
    Move played_move) noexcept
{
    if (current_root == NULL_NODE) return NULL_NODE;

    StateNode& root = pool[current_root];

    // Find child corresponding to played move
    uint32_t child = root.first_child;
    while (child != NULL_NODE) {
        if (pool[child].move == played_move) {
            // Found it - detach from parent
            pool[child].parent = NULL_NODE;
            return child;
        }
        child = pool[child].next_sibling;
    }

    // Move not found in tree
    return NULL_NODE;
}

/// Result of a single self-play game
struct SelfPlayResult {
    int winner{0};      // +1 = P1 won, -1 = P2 won, 0 = draw/error
    int num_moves{0};   // Total moves in the game
};

// ============================================================================
// Self-Play Engine
// ============================================================================

#ifdef QBOT_ENABLE_INFERENCE

/// Self-play engine for generating training data
/// Plays complete games using MCTS + NN evaluation, accumulating statistics
/// in a persistent tree that can be dumped for training.
class SelfPlayEngine {
public:
    explicit SelfPlayEngine(SelfPlayConfig config = SelfPlayConfig{})
        : config_(std::move(config)) {}

    /// Play a single self-play game, accumulating stats in the tree
    /// @param pool Node pool (persistent across games)
    /// @param root_idx Root node index
    /// @param model Neural network for evaluation
    /// @return Game result
    SelfPlayResult self_play(NodePool& pool, uint32_t root_idx, ModelInference& model);

    /// Get configuration
    [[nodiscard]] const SelfPlayConfig& config() const noexcept { return config_; }
    [[nodiscard]] SelfPlayConfig& config() noexcept { return config_; }

private:
    /// Run MCTS iterations from a position
    void run_mcts_iterations(NodePool& pool, uint32_t root_idx, ModelInference& model, int iterations);

    /// Single MCTS iteration: select -> expand -> evaluate -> backprop
    void mcts_iteration(NodePool& pool, uint32_t root_idx, ModelInference& model);

    /// Expand a node and set priors using batch NN evaluation
    void expand_with_nn_priors(NodePool& pool, uint32_t node_idx, ModelInference& model);

    /// Backpropagate value up a path
    void backpropagate(NodePool& pool, const std::vector<uint32_t>& path, float value);

    SelfPlayConfig config_;

    // Striped mutexes for expansion (same pattern as MCTSEngine)
    static constexpr size_t NUM_EXPANSION_MUTEXES = 64;
    std::array<std::mutex, NUM_EXPANSION_MUTEXES> expansion_mutexes_;
};

#endif // QBOT_ENABLE_INFERENCE

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

    // Striped expansion mutexes - allows parallel expansion of different nodes
    // while preventing races on the same node. Index = node_idx % array_size
    static constexpr size_t NUM_EXPANSION_MUTEXES = 256;
    std::array<std::mutex, NUM_EXPANSION_MUTEXES> expansion_mutexes_;

    // Reference to pool and root for checkpointing
    NodePool* pool_ptr_{nullptr};
    uint32_t root_idx_{NULL_NODE};
};

} // namespace qbot
