#pragma once

#include <torch/torch.h>
#include <torch/script.h>

#include <deque>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

namespace qbot {

// Forward declaration
struct StateNode;

/// Action space constants (must match Python resnet.py)
/// Action encoding:
///   0-80:    pawn move to square (row * 9 + col)
///   81-144:  horizontal wall at (row * 8 + col)
///   145-208: vertical wall at (row * 8 + col)
inline constexpr int NUM_PAWN_ACTIONS = 81;   // 9x9 board destinations
inline constexpr int NUM_H_WALL_ACTIONS = 64; // 8x8 horizontal walls
inline constexpr int NUM_V_WALL_ACTIONS = 64; // 8x8 vertical walls
inline constexpr int NUM_WALL_ACTIONS = NUM_H_WALL_ACTIONS + NUM_V_WALL_ACTIONS;  // 128 total
inline constexpr int NUM_ACTIONS = NUM_PAWN_ACTIONS + NUM_WALL_ACTIONS;  // 209 total

// Forward declaration for Move (defined in StateNode.h)
struct Move;

/// Convert a Move to action index for the policy head
/// Pawn moves: row * 9 + col (indices 0-80)
/// Horizontal walls: 81 + row * 8 + col (indices 81-144)
/// Vertical walls: 145 + row * 8 + col (indices 145-208)
/// Returns -1 for invalid moves
[[nodiscard]] int move_to_action_index(const Move& move) noexcept;

/// Convert an action index to a Move
/// Returns an invalid Move if index is out of range
[[nodiscard]] Move action_index_to_move(int action_index) noexcept;

/// Result of neural network evaluation
struct EvalResult {
    float value;                              // Position value from current player's perspective
    std::array<float, NUM_ACTIONS> policy;    // Policy logits for all actions
};

//utility functions to flip the board and policy 180 degrees (and back) for p2 for model
inline uint64_t reverse_bits(uint64_t n) {
    n = ((n >> 1) & 0x5555555555555555ULL) | ((n & 0x5555555555555555ULL) << 1);
    n = ((n >> 2) & 0x3333333333333333ULL) | ((n & 0x3333333333333333ULL) << 2);
    n = ((n >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((n & 0x0F0F0F0F0F0F0F0FULL) << 4);
    n = ((n >> 8) & 0x00FF00FF00FF00FFULL) | ((n & 0x00FF00FF00FF00FFULL) << 8);
    n = ((n >> 16) & 0x0000FFFF0000FFFFULL) | ((n & 0x0000FFFF0000FFFFULL) << 16);
    return (n >> 32) | (n << 32);
}
[[nodiscard]] constexpr int flip_action_index(int idx) noexcept {
    if (idx < 81)  return 80 - idx;           // Pawn moves [0, 80]
    if (idx < 145) return 225 - idx;          // H-walls [81, 144]
    return 353 - idx;                         // V-walls [145, 208]
}

/// Neural network inference for position evaluation
///
/// Converts game states to tensor representation and runs batched inference
/// using a TorchScript model. Supports both immediate single-node evaluation
/// and queued batch processing for efficiency.
///
/// Input tensor format (current-player-perspective):
///   - unified tensor: [6, 9, 9]
///     [0] Current player's pawn position (one-hot)
///     [1] Opponent's pawn position (one-hot)
///     [2] Horizontal walls (padded 8x8 -> 9x9)
///     [3] Vertical walls (padded 8x8 -> 9x9)
///     [4] Current player's fences remaining / 10 (constant plane)
///     [5] Opponent's fences remaining / 10 (constant plane)
///
/// Output format:
///   - policy_logits: [209] raw logits for all actions
///   - value: scalar in [-1, 1] from current player's perspective
class ModelInference {
public:
    /// Callback type for async batch processing
    /// Called with (node_index, value, policy) for each evaluated node
    using EvalCallback = std::function<void(uint32_t, const EvalResult&)>;

    /// Construct inference engine with a TorchScript model
    /// @param model_path Path to the .pt TorchScript model file
    /// @param batch_size Maximum batch size for inference
    /// @param use_cuda Whether to use CUDA if available
    ModelInference(const std::string& model_path, int batch_size = 16, bool use_cuda = true);

    /// Default constructor for deferred initialization
    ModelInference();

    /// Check if model is loaded and ready
    [[nodiscard]] bool is_ready() const noexcept { return model_loaded_; }

    /// Queue a node for batch evaluation
    /// @param node Pointer to the node to evaluate
    /// @param node_idx Index of the node in the pool (for callback identification)
    void queue_for_evaluation(const StateNode* node, uint32_t node_idx);

    /// Process any remaining nodes in the queue
    /// @param callback Called for each evaluated node with (index, result)
    void flush_queue(const EvalCallback& callback);

    /// Evaluate a single node immediately (bypasses queue)
    /// @param node Node to evaluate
    /// @return Model's evaluation result (value and policy)
    [[nodiscard]] EvalResult evaluate_node(const StateNode* node);

    /// Evaluate multiple nodes in a batch (for efficient GPU utilization)
    /// Used for computing priors: evaluate all children at once
    /// @param nodes Vector of node pointers to evaluate
    /// @return Vector of evaluation results, one per node
    [[nodiscard]] std::vector<EvalResult> evaluate_batch(const std::vector<const StateNode*>& nodes);

    /// Evaluate and return only values (legacy interface, ignores policy)
    [[nodiscard]] std::vector<float> evaluate_batch_values(const std::vector<const StateNode*>& nodes);

    /// Get current queue size
    [[nodiscard]] size_t queue_size() const noexcept { return evaluation_queue_.size(); }

    /// Print diagnostic information about torch/CUDA availability
    static void print_diagnostics();

private:
    /// Fill unified tensor for a single node at the given batch index
    /// Handles current-player-perspective transformation
    void fill_unified_tensor(float* tensor_ptr, size_t tensor_stride,
                             int batch_idx, const StateNode* node) const;

    /// Process a batch of queued nodes
    /// @param callback Called for each evaluated node
    void process_batch(const EvalCallback& callback);

    /// Ensure pre-allocated buffers can hold at least `size` nodes
    void ensure_buffer_capacity(int size);

    torch::jit::script::Module model_;
    torch::Device device_;
    int batch_size_;
    bool model_loaded_{false};

    /// Queue entries: (node pointer, node index)
    std::deque<std::pair<const StateNode*, uint32_t>> evaluation_queue_;

    /// Pre-allocated tensor buffers (CPU pinned memory for fast GPU transfer)
    int buffer_capacity_{0};
    torch::Tensor unified_buffer_;      // [capacity, 6, 9, 9]
    torch::Tensor value_output_buffer_; // [capacity] - pinned CPU buffer for GPU->CPU transfer
    torch::Tensor policy_output_buffer_; // [capacity, NUM_ACTIONS] - policy outputs
};

} // namespace qbot
