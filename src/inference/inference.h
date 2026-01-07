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

/// Neural network inference for position evaluation
///
/// Converts game states to tensor representation and runs batched inference
/// using a TorchScript model. Supports both immediate single-node evaluation
/// and queued batch processing for efficiency.
///
/// Tensor format:
///   - pawn_tensor: [2, 9, 9] - one-hot encoded player positions
///   - wall_tensor: [2, 8, 8] - horizontal and vertical fence placements
///   - meta_tensor: [2] - remaining fence counts for each player
class ModelInference {
public:
    /// Callback type for async batch processing
    /// Called with (node_index, value) for each evaluated node
    using EvalCallback = std::function<void(uint32_t, float)>;

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
    /// @param callback Called for each evaluated node with (index, value)
    void flush_queue(const EvalCallback& callback);

    /// Evaluate a single node immediately (bypasses queue)
    /// @param node Node to evaluate
    /// @return Model's value estimate for the position
    [[nodiscard]] float evaluate_node(const StateNode* node);

    /// Evaluate multiple nodes in a batch (for efficient GPU utilization)
    /// Used for computing priors: evaluate all children at once
    /// @param nodes Vector of node pointers to evaluate
    /// @return Vector of value estimates, one per node
    [[nodiscard]] std::vector<float> evaluate_batch(const std::vector<const StateNode*>& nodes);

    /// Get current queue size
    [[nodiscard]] size_t queue_size() const noexcept { return evaluation_queue_.size(); }

    /// Print diagnostic information about torch/CUDA availability
    static void print_diagnostics();

private:
    /// Convert a StateNode to tensor inputs
    /// @return Tuple of (pawn_tensor[2,9,9], wall_tensor[2,8,8], meta_tensor[2])
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    state_to_tensors(const StateNode* node) const;

    /// Process a batch of queued nodes
    /// @param callback Called for each evaluated node
    void process_batch(const EvalCallback& callback);

    torch::jit::script::Module model_;
    torch::Device device_;
    int batch_size_;
    bool model_loaded_{false};

    /// Queue entries: (node pointer, node index)
    std::deque<std::pair<const StateNode*, uint32_t>> evaluation_queue_;
};

} // namespace qbot
