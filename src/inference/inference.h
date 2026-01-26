#pragma once

#include <torch/torch.h>
#include <torch/script.h>

#include <deque>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

namespace qbot {

struct StateNode;

inline constexpr int NUM_PAWN_ACTIONS = 81;
inline constexpr int NUM_H_WALL_ACTIONS = 64;
inline constexpr int NUM_V_WALL_ACTIONS = 64;
inline constexpr int NUM_WALL_ACTIONS = NUM_H_WALL_ACTIONS + NUM_V_WALL_ACTIONS;
inline constexpr int NUM_ACTIONS = NUM_PAWN_ACTIONS + NUM_WALL_ACTIONS;

struct Move;

[[nodiscard]] int move_to_action_index(const Move& move) noexcept;
[[nodiscard]] Move action_index_to_move(int action_index) noexcept;

struct EvalResult {
    float value;
    std::array<float, NUM_ACTIONS> policy;
};

inline uint64_t reverse_bits(uint64_t n) {
    n = ((n >> 1) & 0x5555555555555555ULL) | ((n & 0x5555555555555555ULL) << 1);
    n = ((n >> 2) & 0x3333333333333333ULL) | ((n & 0x3333333333333333ULL) << 2);
    n = ((n >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((n & 0x0F0F0F0F0F0F0F0FULL) << 4);
    n = ((n >> 8) & 0x00FF00FF00FF00FFULL) | ((n & 0x00FF00FF00FF00FFULL) << 8);
    n = ((n >> 16) & 0x0000FFFF0000FFFFULL) | ((n & 0x0000FFFF0000FFFFULL) << 16);
    return (n >> 32) | (n << 32);
}

[[nodiscard]] constexpr int flip_action_index(int idx) noexcept {
    if (idx < 81)  return 80 - idx;
    if (idx < 145) return 225 - idx;
    return 353 - idx;
}

class ModelInference {
public:
    using EvalCallback = std::function<void(uint32_t, const EvalResult&)>;

    ModelInference(const std::string& model_path, int batch_size = 16, bool use_cuda = true);
    ModelInference();

    [[nodiscard]] bool is_ready() const noexcept { return model_loaded_; }

    void queue_for_evaluation(const StateNode* node, uint32_t node_idx);
    void flush_queue(const EvalCallback& callback);

    [[nodiscard]] EvalResult evaluate_node(const StateNode* node);
    [[nodiscard]] std::vector<EvalResult> evaluate_batch(const std::vector<const StateNode*>& nodes);
    [[nodiscard]] std::vector<float> evaluate_batch_values(const std::vector<const StateNode*>& nodes);

    [[nodiscard]] size_t queue_size() const noexcept { return evaluation_queue_.size(); }

private:
    template <typename T>
    void fill_input_tensor(T* tensor_ptr, size_t tensor_stride,
                           int batch_idx, const StateNode* node) const;
    void process_batch(const EvalCallback& callback);
    void ensure_buffer_capacity(int size);

    torch::jit::script::Module model_;
    torch::Device device_;
    int batch_size_;
    bool model_loaded_{false};
    bool use_fp16_{false};

    std::deque<std::pair<const StateNode*, uint32_t>> evaluation_queue_;

    int buffer_capacity_{0};
    torch::Tensor unified_buffer_;       // [capacity, 6, 9, 9] FP16 pinned
    torch::Tensor value_output_buffer_;  // [capacity] FP32 pinned
    torch::Tensor policy_output_buffer_; // [capacity, NUM_ACTIONS] FP32 pinned
};

} // namespace qbot
