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
#include "../inference/inference.h"
#include "../inference/inference_server.h"

namespace qbot {

// ============================================================================
// Inference Type Traits (for unifying ModelInference and InferenceServer)
// ============================================================================


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

// ============================================================================
// Configuration
// ============================================================================

/// Tree memory bounds configuration
/// Controls when to stop expanding and use NN evaluation instead
struct TreeBoundsConfig {
    size_t max_bytes = 40ULL * 1024 * 1024 * 1024;  // 40GB default
    float soft_limit_ratio = 0.80f;   // Start being selective about expansion
    // size_t soft_limit_bytes = 30ULL * 1024 * 1024 * 1024;  // 30GB default
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


// ============================================================================
// Evaluation Functions
// ============================================================================

/// Determine winner when both players are out of fences
/// Uses path length to goal - whoever is closer wins
/// @return +1 if current player wins, -1 if current player loses (relative perspective)
[[nodiscard]] inline int early_terminate_no_fences(const StateNode& node) noexcept {
    Pathfinder& pf = get_pathfinder();

    auto p1_path = pf.find_path(node.fences, node.p1, 8);
    auto p2_path = pf.find_path(node.fences, node.p2, 0);

    if (p1_path.empty() || p2_path.empty()) {
        // Should never happen - someone is blocked
        return 0;
    }

    // Path length is path.size() - 1 (path includes starting position)
    int p1_dist = static_cast<int>(p1_path.size()) - 1;
    int p2_dist = static_cast<int>(p2_path.size()) - 1;

    // Check for jump opportunities when distances are close
    int diff = p1_dist - p2_dist;
    if (diff >= -1 && diff <= 1) {
        //since check_jump returns 0 for no jump, 1 for p1 jump, -1 for p2 jump
        p1_dist -= check_jump_advantage(p1_path, p2_path, node.is_p1_to_move());
    }

    int curr_dist = node.is_p1_to_move() ? p1_dist : p2_dist;
    int opp_dist = node.is_p1_to_move() ? p2_dist : p1_dist;
    // current player moves first, they win if their distance to
    // goal is less than OR equal to the opponent's distance
    return (curr_dist <= opp_dist) ? 1 : -1;
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
    int max_moves_per_game = 100;           // After this many moves, declare a draw and assign partial points to closer player
    float max_draw_reward = 0.0;           // On a draw, this is maximum reward we give the closest player 
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

//adds dirichlet noise as in AGZ
//only add to the root node of a search tree in selfplay to ensure we see a wide variety of games
//in this context root node of a search tree is the node we are deciding on the move for rn, not the absolute root ie starting pos
static void add_dirichlet_noise(NodePool& pool, uint32_t node_idx, float alpha, float epsilon) {
    StateNode& node = pool[node_idx];
    if (!node.has_children()) return;

    std::vector<uint32_t> children;
    uint32_t child = node.first_child;
    while (child != NULL_NODE) {
        children.push_back(child);
        child = pool[child].next_sibling;
    }

    if (children.empty()) return;

    // 2. Generate Dirichlet noise
    // We use Gamma(alpha, 1) and normalize to get Dirichlet distribution
    static thread_local std::mt19937 gen(std::random_device{}());
    std::gamma_distribution<float> d(alpha, 1.0f);

    std::vector<float> noise;
    noise.reserve(children.size());
    float sum_noise = 0.0f;

    for (size_t i = 0; i < children.size(); ++i) {
        float n = d(gen);
        // Avoid extremely small values for numerical stability
        if (n < 1e-6f) n = 1e-6f; 
        noise.push_back(n);
        sum_noise += n;
    }

    // 3. Apply mixed priors: P(a) = (1 - epsilon) * policy(a) + epsilon * noise(a)
    for (size_t i = 0; i < children.size(); ++i) {
        StateNode& c = pool[children[i]];
        float noise_prob = noise[i] / sum_noise;

        // stats.prior holds the NN policy probability
        // We modify it in-place. This is safe because this is the root of a 
        // new search and we are the only thread accessing it (in self-play worker).
        c.stats.prior = (1.0f - epsilon) * c.stats.prior + epsilon * noise_prob;
    }
}

[[nodiscard]] inline std::vector<std::pair<Move, float>> compute_policy_from_visits(
    NodePool& pool,
    uint32_t parent_idx,
    float temperature) noexcept
{
    std::vector<std::pair<Move, float>> policy;
    if (!pool[parent_idx].has_children()) return policy;

    uint32_t child = pool[parent_idx].first_child;
    while (child != NULL_NODE) {
        uint32_t visits = pool[child].stats.visits.load(std::memory_order_relaxed);
        policy.push_back({pool[child].move, static_cast<float>(visits)});
        child = pool[child].next_sibling;
    }

    if (policy.empty()) return policy;

    // Temperature = 0 means deterministic (argmax)
    if (temperature <= 1e-6f) {
        size_t best_idx = 0;
        float best_visits = policy[0].second;
        int tie_count = 1;
        for (size_t i = 1; i < policy.size(); ++i) {
            float visits = policy[i].second;
            if (visits > best_visits) {
                // strict improvement: reset ties
                best_visits = visits;
                best_idx = i;
                tie_count = 1;
            } 
            else if (visits == best_visits) {
                tie_count++;
                thread_local std::mt19937 rng(std::random_device{}());
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                // replace with probability 1/n
                if (dist(rng) < (1.0f / tie_count)) {
                    best_idx = i;
                }
            }
        }
        for (size_t i = 0; i < policy.size(); ++i) {
            policy[i].second = (i == best_idx) ? 1.0f : 0.0f;
        }
        return policy;
    }

    // π(a) ∝ N(a)^(1/τ)
    float inv_temp = 1.0f / temperature;
    float sum = 0.0f;
    for (auto& [move, visits] : policy) {
        visits = std::pow(visits, inv_temp);
        sum += visits;
    }

    //normalize
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
            [](const auto& a, const auto& b) { 
                return a.second < b.second;
            });
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

inline void prune_siblings(NodePool& pool, uint32_t parent_idx, uint32_t keep_child_idx);
inline void prune_siblings_collect(NodePool& pool, uint32_t parent_idx, uint32_t keep_child_idx, std::vector<uint32_t>& freed_nodes);
inline void collect_subtree_nodes(NodePool& pool, uint32_t root_idx, std::vector<uint32_t>& out);
inline void apply_policy_to_children(
    NodePool& pool,
    uint32_t node_idx,
    const StateNode& node,
    const std::array<float, NUM_ACTIONS>& policy_logits);

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

/// Statistics for multi-threaded self-play (thread-safe accumulator)
struct MultiGameStats {
    std::atomic<int> p1_wins{0};
    std::atomic<int> p2_wins{0};
    std::atomic<int> draws{0};
    std::atomic<int> errors{0};
    std::atomic<int> total_moves{0};
    std::atomic<int> draw_score{0}; //if draw, accumulate relative distance metric
    std::atomic<int> games_completed{0};
    std::chrono::steady_clock::time_point start_time;

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
        start_time = std::chrono::steady_clock::now();
    }

    [[nodiscard]] double games_per_sec() const noexcept {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        if (elapsed == 0) return 0.0;
        return static_cast<double>(games_completed.load(std::memory_order_relaxed)) / elapsed;
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


    /// Run multiple self-play games in parallel with per-thread node pools
    /// Each worker thread creates its own NodePool, eliminating contention.
    /// @param server Inference server for batched GPU evaluation
    /// @param num_games Total games to play
    /// @param num_workers Number of worker threads
    /// @param games_per_worker Number of games each worker runs concurrently
    /// @param stats Output statistics (must be pre-allocated)
    /// @param collector Training sample collector to accumulate samples
    /// @param samples_file Path for intermediate qsamples saves (empty to disable)
    /// @param checkpoint_callback Called periodically with stats (optional)
    /// @param checkpoint_interval_games Games between checkpoint callbacks
    /// @param pool_config_per_thread Node pool configuration for each thread
    void run_multi_game(
        InferenceServer& server,
        int num_games,
        int num_workers,
        int games_per_worker,
        MultiGameStats& stats,
        TrainingSampleCollector* collector,
        const std::filesystem::path& samples_file,
        std::function<void(const MultiGameStats&)> checkpoint_callback = nullptr,
        int checkpoint_interval_games = 10,
        NodePool::Config pool_config_per_thread = NodePool::Config{});

    // struct GameContext {
    //     uint32_t current_node;
    //     std::vector<uint32_t> game_path;
    //     std::vector<uint32_t> sample_positions;
    //     int num_moves{0};
    //     bool active{true};
    //     bool needs_expansion{false};
    //     bool needs_mcts{false};
    //     int mcts_iterations_done{0};
    // };
    struct PerGameContext {
        uint32_t root_idx{NULL_NODE};           // This game's tree root (changes each move)
        uint32_t original_root{NULL_NODE};      // The very first root (for final cleanup)
        std::vector<uint32_t> game_path;        // Path of played moves for training samples

        // Store samples as we go (state + policy), outcome added at game end
        struct PendingSample {
            CompactState state;
            std::array<float, NUM_ACTIONS> policy;
        };
        std::vector<PendingSample> pending_samples;

        int num_moves{0};
        bool active{false};
        int mcts_iterations_done{0};

        void reset() {
            root_idx = NULL_NODE;
            original_root = NULL_NODE;
            game_path.clear();
            pending_samples.clear();
            num_moves = 0;
            active = false;
            mcts_iterations_done = 0;
        }
    };

    struct DeferredPrune {
        uint32_t parent_idx;
        uint32_t keep_child_idx;
    };

    struct MultiGameWorkerSync {
        std::atomic<bool>& pause_requested;
        std::atomic<bool>& draining; //waiting for games to finish
        std::atomic<int>& workers_paused;
        std::atomic<int>& total_active_games;
        std::atomic<uint32_t>& current_root;
        std::mutex& pause_mutex;
        std::condition_variable& pause_cv;
        std::condition_variable& resume_cv;
    };

    //can run multiple games per worker thread
    template<InferenceProvider Inference>
    void run_multi_game_worker(
        std::stop_token stop_token,
        NodePool& pool,
        Inference& inference,
        int games_per_worker,
        std::atomic<int>& games_remaining,
        MultiGameStats& stats,
        MultiGameWorkerSync& sync,
        TrainingSampleCollector* collector = nullptr);

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
    static constexpr size_t NUM_EXPANSION_MUTEXES = 1024;
    std::array<std::mutex, NUM_EXPANSION_MUTEXES> expansion_mutexes_;
};

// ============================================================================
// Competitive Play Engine
// ============================================================================

/// Configuration for competitive play (human vs bot, bot vs bot evaluation)
struct CompEngineConfig {
    int num_simulations = 800;      // MCTS iterations per move
    float c_puct = 1.5f;            // PUCT exploration constant
    float fpu = 0.0f;               // First play urgency
    int num_threads = 1;            // Worker threads (future: parallel search)
    int eval_batch_size = 64;       // Batch size for NN evaluation
};

/// Performs fixed-simulation search from current position and returns
/// plans for adding extra chunks of search during opponent's turn
/// also should add pruning
/// the strongest move (highest visit count, temperature=0).
/// 
class CompEngine {
public:
    explicit CompEngine(CompEngineConfig config = CompEngineConfig{})
        : config_(std::move(config)) {}

    /// Search for best move using MCTS with NN evaluation
    /// 
    /// @param pool Node pool containing the game tree
    /// @param root_idx Current position to search from
    /// @param inference NN inference provider
    /// @return Best move by visit count, or invalid Move if terminal/no legal moves
    template<InferenceProvider Inference>
    [[nodiscard]] Move search(NodePool& pool, uint32_t root_idx, Inference& inference);

    /// Search with explicit simulation count override
    template<InferenceProvider Inference>
    [[nodiscard]] Move search(NodePool& pool, uint32_t root_idx, Inference& inference, 
                              int num_simulations);

    /// Get/set configuration
    [[nodiscard]] const CompEngineConfig& config() const noexcept { return config_; }
    [[nodiscard]] CompEngineConfig& config() noexcept { return config_; }

private:
    /// Expand node and set NN-derived priors on children
    template<InferenceProvider Inference>
    void expand_with_nn_priors(NodePool& pool, uint32_t node_idx, Inference& inference);

    /// Backpropagate value up selection path (alternating perspective)
    static void backpropagate(NodePool& pool, const std::vector<uint32_t>& path, float value);

    /// Run single-threaded MCTS iterations
    template<InferenceProvider Inference>
    void run_iterations(NodePool& pool, uint32_t root_idx, Inference& inference, int iterations);

    CompEngineConfig config_;

    // Striped mutexes for thread-safe expansion (future parallel search)
    static constexpr size_t NUM_EXPANSION_MUTEXES = 64;
    std::array<std::mutex, NUM_EXPANSION_MUTEXES> expansion_mutexes_;
};


} // namespace qbot
