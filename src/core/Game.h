#pragma once

#include "../tree/node_pool.h"
#include "../util/gui_client.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace qbot {

// Forward declaration for optional model inference
class ModelInference;

/// Configuration for game/training sessions
struct GameConfig {
    size_t pool_capacity = 1'000'000;  // Node pool size
    int num_threads = 4;               // Worker threads for parallel MCTS
    float c_puct = 1.5f;               // PUCT exploration constant
    int simulations_per_move = 800;    // MCTS iterations before selecting move
};

/// Game controller for running Quoridor games
///
/// Responsibilities:
/// - Managing the game tree and node pool
/// - Running self-play and training sessions
/// - Handling human vs bot games
/// - Coordinating MCTS simulations (single and multi-threaded)
/// - Communicating with GUI (via IPC)
class Game {
public:
    using Config = GameConfig;

    /// Construct a new game with a fresh pool
    /// @param config Game configuration
    explicit Game(Config config = Config{});

    /// Construct a game with an existing pool (takes ownership)
    /// @param pool Existing node pool
    /// @param root Root node index
    Game(std::unique_ptr<NodePool> pool, uint32_t root);

    ~Game();

    /// Build a random game tree from a root node (for testing)
    /// Expands nodes randomly, controlled by branching_factor which determines
    /// probability of breadth-first (sibling) vs depth-first (child) expansion.
    /// @param root_idx Index of the subtree root node
    /// @param branching_factor Probability [0,1] of expanding breadth vs depth
    /// @param time_limit Maximum seconds to run
    /// @param node_limit Maximum nodes to create
    /// @return Number of nodes created
    size_t build_tree(uint32_t root_idx, float branching_factor,
                      std::time_t time_limit, size_t node_limit);

    /// Access the node pool
    [[nodiscard]] NodePool& pool() noexcept { return *pool_; }
    [[nodiscard]] const NodePool& pool() const noexcept { return *pool_; }

    /// Get/set root node index
    [[nodiscard]] uint32_t root() const noexcept { return root_; }
    void set_root(uint32_t idx) noexcept { root_ = idx; }

    /// Connect to GUI server for visualization
    /// @param config GUI connection configuration
    /// @return true if connected successfully
    bool connect_gui(const GUIClient::Config& config);

    /// Disconnect from GUI server
    void disconnect_gui();

    /// Check if GUI is connected
    [[nodiscard]] bool has_gui() const noexcept;

    /// Visualize a state node using GUI if connected, otherwise print to stdout
    /// @param node The state to visualize
    /// @param score Optional evaluation score to display
    void visualize_state(const StateNode& node, float score = 0.0f);

    /// Visualize current root node
    void visualize_root(float score = 0.0f);

    /// Get the GUI client (may be nullptr)
    [[nodiscard]] GUIClient* gui() noexcept { return gui_.get(); }

    /// Set the model for position evaluation (non-owning pointer)
    /// @param model Pointer to ModelInference, or nullptr to disable
    void set_model(ModelInference* model) noexcept { model_ = model; }

    /// Check if a model is set
    [[nodiscard]] bool has_model() const noexcept { return model_ != nullptr; }

    /// Select the best move from a node
    /// If a model is set, evaluates all children and picks the best move for
    /// the current player (max score for P1, min score for P2).
    /// Otherwise falls back to Q-value based selection.
    /// @param node_idx Index of the node to select from
    /// @return Best move, or invalid Move if node has no children
    [[nodiscard]] Move select_best_move(uint32_t node_idx);

    /// Select the best move using Q-values only (static version)
    /// Chooses randomly among moves with the highest Q-value.
    /// @param pool Node pool containing the tree
    /// @param node_idx Index of the node to select from
    /// @return Best move, or invalid Move if node has no children
    [[nodiscard]] static Move select_best_move_by_q(NodePool& pool, uint32_t node_idx);

    // Non-copyable, non-movable (owns resources)
    Game(const Game&) = delete;
    Game& operator=(const Game&) = delete;
    Game(Game&&) = delete;
    Game& operator=(Game&&) = delete;

private:
    Config config_;
    std::unique_ptr<NodePool> pool_;
    uint32_t root_{NULL_NODE};

    // Optional GUI connection
    std::unique_ptr<GUIClient> gui_;

    // Optional model for position evaluation (non-owning)
    ModelInference* model_{nullptr};

    // Threading synchronization
    std::mutex tree_mutex_;
    std::atomic<bool> stop_flag_{false};
    std::atomic<int> active_threads_{0};
};

} // namespace qbot
