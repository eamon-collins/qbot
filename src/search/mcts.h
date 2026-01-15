#pragma once

/// Parallel MCTS with Neural Network Evaluation
///
/// Implements Monte Carlo Tree Search with:
/// - Virtual loss for tree parallelism (multiple threads explore different paths)
/// - Neural network evaluation at leaf nodes (AlphaZero-style)
/// - Early termination when both players out of fences
/// - Time-based checkpointing
/// - Self-play mode for training data generation
/// - Training sample collection with MCTS visit distributions

#include "../tree/node_pool.h"
#include "../tree/StateNode.h"
#include "../util/pathfinding.h"
#include "../util/storage.h"
#include "../util/training_samples.h"

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
#include "../inference/inference_server.h"
#endif

namespace qbot {

// ============================================================================
// Inference Type Traits (for unifying ModelInference and InferenceServer)
// ============================================================================

#ifdef QBOT_ENABLE_INFERENCE

/// Type trait to distinguish direct model inference from async server
template<typename T>
struct is_inference_server : std::false_type {};

template<>
struct is_inference_server<InferenceServer> : std::true_type {};

template<typename T>
inline constexpr bool is_inference_server_v = is_inference_server<T>::value;

/// Concept for any inference provider (ModelInference or InferenceServer)
template<typename T>
concept InferenceProvider = std::same_as<T, ModelInference> || std::same_as<T, InferenceServer>;

#endif

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
    return true;
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
/// @return +1 if current player wins, -1 if current player loses (relative perspective)
[[nodiscard]] inline int early_terminate_no_fences(const StateNode& node) noexcept {
    Pathfinder& pf = get_pathfinder();

    int p1_dist = pf.path_length(node.fences, node.p1, 8);  // P1 goal is row 8
    int p2_dist = pf.path_length(node.fences, node.p2, 0);  // P2 goal is row 0

    if (p1_dist < 0 || p2_dist < 0) {
        // Should never happen - someone is blocked
        return 0;
    }

    int curr_dist = node.is_p1_to_move() ? p1_dist : p2_dist;
    int opp_dist = node.is_p1_to_move() ? p2_dist : p1_dist;
    // node.print_node();
    // std::cout << "earlyterm rel " << ((curr_dist <= opp_dist) ? 1 : -1) << " p1move " << node.is_p1_to_move() << " p1: " << p1_dist << " p2: " << p2_dist << std::endl;

    // current player moves first, they win if their distance to
    // goal is less than OR equal to the opponent's distance
    return (curr_dist <= opp_dist) ? 1 : -1;
}


/// Evaluate a leaf node
/// @param node Node to evaluate
/// @param stats Training stats (updated with evaluation count)
/// @param inference Optional NN inference engine
/// @return Value in relative perspective (+1 = current player winning, -1 = losing)
[[nodiscard]] inline float evaluate_leaf(
    const StateNode& node,
    TrainingStats& stats,
    [[maybe_unused]] void* inference = nullptr) noexcept
{
    // Terminal nodes have known values
    if (node.is_terminal()) {
        return node.terminal_value;
    }

    // DISABLED for training: Early termination when fences exhausted distorts
    // incentives by encouraging players to spend all fences quickly.
    // We want the model to learn full endgame dynamics.
    // Kept for reference - may re-enable for arena/inference mode.
    //
    // if (node.p1.fences == 0 && node.p2.fences == 0) {
    //     return early_terminate_no_fences(node);
    // }

#ifdef QBOT_ENABLE_INFERENCE
    // Use neural network evaluation if available
    if (inference != nullptr) {
        auto* model = static_cast<ModelInference*>(inference);
        if (model->is_ready()) {
            stats.nn_evaluations.fetch_add(1, std::memory_order_relaxed);
            return model->evaluate_node(&node).value;
        }
    }
#endif

    // No NN available - use path distance heuristic
    // this now returns just +/- 1, so this doesn't make sense, but should always have inference
    return std::clamp(early_terminate_no_fences(node), -10, 10) / 10.0 * .8;
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
    bool progressive_expansion = false;    // True = create children on demand, False = batch expand
    float c_puct = 1.5f;                   // PUCT exploration constant (for progressive mode)
    float fpu = 0.0f;                      // First play urgency (for progressive mode)
    int max_moves_per_game = 60;           // After this many moves, declare a draw and assign partial points to closer player
    float max_draw_reward = 0.5;           // On a draw, this is maximum reward we give the closest player 
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
    int draw_score{0}; //if draw, accumulate relative distance metric
    int num_moves{0};   // Total moves in the game
    bool error{false};      // track errors different than draws (draw is too many moves without a win, error is invalid state)
};

// ============================================================================
// Self-Play Engine
// ============================================================================

#ifdef QBOT_ENABLE_INFERENCE

/// Statistics for multi-threaded self-play (thread-safe accumulator)
struct MultiGameStats {
    std::atomic<int> p1_wins{0};
    std::atomic<int> p2_wins{0};
    std::atomic<int> draws{0};
    std::atomic<int> errors{0};
    std::atomic<int> total_moves{0};
    std::atomic<int> draw_score{0}; //if draw, accumulate relative distance metric
    std::atomic<int> games_completed{0};

    void add_result(const SelfPlayResult& result) noexcept {
        if (result.error) errors.fetch_add(1, std::memory_order_relaxed);
        else if (result.winner == 1) p1_wins.fetch_add(1, std::memory_order_relaxed);
        else if (result.winner == -1) p2_wins.fetch_add(1, std::memory_order_relaxed);
        else {
            draws.fetch_add(1, std::memory_order_relaxed);
            draw_score.fetch_add(result.draw_score, std::memory_order_relaxed);
        }
        total_moves.fetch_add(result.num_moves, std::memory_order_relaxed);
        games_completed.fetch_add(1, std::memory_order_relaxed);
    }

    void reset() noexcept {
        p1_wins.store(0, std::memory_order_relaxed);
        p2_wins.store(0, std::memory_order_relaxed);
        draws.store(0, std::memory_order_relaxed);
        total_moves.store(0, std::memory_order_relaxed);
        errors.store(0, std::memory_order_relaxed);
        draw_score.store(0, std::memory_order_relaxed);
        games_completed.store(0, std::memory_order_relaxed);
    }

    // Non-copyable due to atomics
    MultiGameStats() = default;
    MultiGameStats(const MultiGameStats&) = delete;
    MultiGameStats& operator=(const MultiGameStats&) = delete;
};

/// Self-play engine for generating training data
/// Plays complete games using MCTS + NN evaluation, accumulating statistics
/// in a persistent tree that can be dumped for training.
///
/// Supports multi-threaded self-play with shared inference server for
/// better GPU utilization.
class SelfPlayEngine {
public:
    explicit SelfPlayEngine(SelfPlayConfig config = SelfPlayConfig{})
        : config_(std::move(config)) {}

    /// Play a single self-play game, accumulating stats in the tree
    /// @param pool Node pool (persistent across games)
    /// @param root_idx Root node index
    /// @param model Neural network for evaluation
    /// @param collector Optional training sample collector (if nullptr, no samples collected)
    /// @return Game result
    SelfPlayResult self_play(NodePool& pool, uint32_t root_idx, ModelInference& model,
                             TrainingSampleCollector* collector = nullptr);

    /// Play a single self-play game using InferenceServer (for multi-threaded use)
    /// @param pool Node pool (shared across workers)
    /// @param root_idx Root node index
    /// @param server Inference server for batched GPU evaluation
    /// @param collector Optional training sample collector (if nullptr, no samples collected)
    /// @return Game result
    SelfPlayResult self_play(NodePool& pool, uint32_t root_idx, InferenceServer& server,
                             TrainingSampleCollector* collector = nullptr);

    /// Run multiple self-play games in parallel
    /// Supports memory-bounded operation: when pool reaches 80% of memory limit,
    /// workers pause, samples are extracted, pool is reset, and games continue.
    /// @param pool Node pool (shared across all games)
    /// @param root_idx Root node index (will be updated on pool reset)
    /// @param server Inference server for batched GPU evaluation
    /// @param num_games Total games to play
    /// @param num_workers Number of worker threads
    /// @param stats Output statistics (must be pre-allocated)
    /// @param bounds Memory bounds configuration (triggers reset at soft_limit_ratio)
    /// @param collector Training sample collector to accumulate samples across pool resets
    /// @param samples_file Path for intermediate qsamples saves (empty to disable)
    /// @param checkpoint_callback Called periodically with stats (optional)
    /// @param checkpoint_interval_games Games between checkpoint callbacks
    void run_multi_game(
        NodePool& pool,
        uint32_t& root_idx,
        InferenceServer& server,
        int num_games,
        int num_workers,
        MultiGameStats& stats,
        const TreeBoundsConfig& bounds,
        TrainingSampleCollector* collector,
        const std::filesystem::path& samples_file,
        std::function<void(const MultiGameStats&, const NodePool&)> checkpoint_callback = nullptr,
        int checkpoint_interval_games = 10);

    /// Play a single arena game between two models
    /// Uses shared pool but does NOT cache NN values (different models produce different values)
    /// @param pool Node pool (shared across games)
    /// @param root_idx Root node index
    /// @param server_p1 Inference server for player 1's model
    /// @param server_p2 Inference server for player 2's model
    /// @return Game result (winner from P1's perspective: +1 P1 wins, -1 P2 wins, 0 error)
    SelfPlayResult arena_game(NodePool& pool, uint32_t root_idx,
                              InferenceServer& server_p1, InferenceServer& server_p2);

    /// Run multiple arena games in parallel
    /// Uses shared pool but does NOT cache NN values (different models produce different values)
    /// @param pool Node pool (shared across all games)
    /// @param root_idx Root node index
    /// @param server_p1 Inference server for player 1's model (candidate in even games)
    /// @param server_p2 Inference server for player 2's model (current in even games)
    /// @param num_games Total games to play
    /// @param num_workers Number of worker threads
    /// @param stats Output statistics
    /// @param checkpoint_callback Called periodically with stats (optional)
    void run_multi_arena(
        NodePool& pool,
        uint32_t root_idx,
        InferenceServer& server_p1,
        InferenceServer& server_p2,
        int num_games,
        int num_workers,
        MultiGameStats& stats,
        std::function<void(const MultiGameStats&, const NodePool&)> checkpoint_callback = nullptr,
        int checkpoint_interval_games = 10);

    /// Get configuration
    [[nodiscard]] const SelfPlayConfig& config() const noexcept { return config_; }
    [[nodiscard]] SelfPlayConfig& config() noexcept { return config_; }

private:
    /// Unified self-play implementation for any inference provider
    template<InferenceProvider Inference>
    SelfPlayResult self_play_impl(NodePool& pool, uint32_t root_idx, Inference& inference,
                                   TrainingSampleCollector* collector);

    /// Unified MCTS iterations for any inference provider
    template<InferenceProvider Inference>
    void run_mcts_iterations(NodePool& pool, uint32_t root_idx, Inference& inference, int iterations);

    /// Unified expansion with NN priors for any inference provider
    template<InferenceProvider Inference>
    void expand_with_nn_priors(NodePool& pool, uint32_t node_idx, Inference& inference);

    /// Refresh priors on an already-expanded node's children using the given server
    void refresh_priors(NodePool& pool, uint32_t node_idx, InferenceServer& server);

    /// Backpropagate value up a path
    void backpropagate(NodePool& pool, const std::vector<uint32_t>& path, float value);

    // === Progressive expansion methods ===
    // Disabled by default - define QBOT_ENABLE_PROGRESSIVE to re-enable.
#ifdef QBOT_ENABLE_PROGRESSIVE
    /// Compute valid action mask and policy priors without creating children
    /// Thread-safe, only computed once per node
    void compute_priors_progressive(NodePool& pool, uint32_t node_idx, InferenceServer& server);

    /// Select best action using PUCT over all valid actions (including unmade children)
    /// Creates the child if it doesn't exist yet
    /// @return Child index if selection/creation succeeded, NULL_NODE otherwise
    uint32_t select_and_expand_progressive(NodePool& pool, uint32_t node_idx,
                                           float c_puct = 1.5f, float fpu = 0.0f);
#endif

    /// Worker thread main loop for multi-game self-play
    void worker_loop(
        std::stop_token stop_token,
        int worker_id,
        NodePool& pool,
        uint32_t root_idx,
        InferenceServer& server,
        std::atomic<int>& games_remaining,
        MultiGameStats& stats);

    /// Run MCTS iterations for arena with two servers (routes to correct model per player)
    void run_arena_mcts_iterations(
        NodePool& pool, uint32_t root_idx,
        InferenceServer& server_p1, InferenceServer& server_p2,
        int iterations);

    /// Worker thread main loop for multi-game arena
    void arena_worker_loop(
        std::stop_token stop_token,
        int worker_id,
        NodePool& pool,
        uint32_t root_idx,
        InferenceServer& server_p1,
        InferenceServer& server_p2,
        std::atomic<int>& games_remaining,
        std::atomic<int>& game_counter,
        MultiGameStats& stats);

    SelfPlayConfig config_;

    // Striped mutexes for expansion - 256 for better parallelism with many workers
    static constexpr size_t NUM_EXPANSION_MUTEXES = 256;
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
