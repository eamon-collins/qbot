#include "inference.h"
#include "../tree/StateNode.h"
#include "../util/timer.h"

#include <iostream>

namespace qbot {

int move_to_action_index(const Move& move) noexcept {
    if (!move.is_valid()) return -1;

    if (move.is_pawn()) {
        return move.row() * 9 + move.col();
    } else {
        int base = move.is_horizontal() ? NUM_PAWN_ACTIONS : (NUM_PAWN_ACTIONS + NUM_H_WALL_ACTIONS);
        return base + move.row() * 8 + move.col();
    }
}

Move action_index_to_move(int action_index) noexcept {
    if (action_index < 0 || action_index >= NUM_ACTIONS) {
        return Move{};
    }

    if (action_index < NUM_PAWN_ACTIONS) {
        uint8_t row = static_cast<uint8_t>(action_index / 9);
        uint8_t col = static_cast<uint8_t>(action_index % 9);
        return Move::pawn(row, col);
    } else if (action_index < NUM_PAWN_ACTIONS + NUM_H_WALL_ACTIONS) {
        int wall_idx = action_index - NUM_PAWN_ACTIONS;
        uint8_t row = static_cast<uint8_t>(wall_idx / 8);
        uint8_t col = static_cast<uint8_t>(wall_idx % 8);
        return Move::fence(row, col, true);
    } else {
        int wall_idx = action_index - NUM_PAWN_ACTIONS - NUM_H_WALL_ACTIONS;
        uint8_t row = static_cast<uint8_t>(wall_idx / 8);
        uint8_t col = static_cast<uint8_t>(wall_idx % 8);
        return Move::fence(row, col, false);
    }
}

ModelInference::ModelInference(const std::string& model_path, int batch_size, bool use_cuda)
    : device_(use_cuda && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
    , batch_size_(batch_size)
{
    try {
        model_ = torch::jit::load(model_path, device_);
        model_.eval();

        if (device_.is_cuda()) {
            model_ = torch::jit::freeze(model_);
            model_ = torch::jit::optimize_for_inference(model_);
        }

        model_loaded_ = true;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model from " << model_path << ": " << e.what() << std::endl;
        throw;
    }
}

ModelInference::ModelInference()
    : device_(torch::kCPU)
    , batch_size_(0)
    , model_loaded_(false)
{}

void ModelInference::fill_input_tensor(at::Half* tensor_ptr, size_t tensor_stride,
                                        int batch_idx, const StateNode* node) const {
    at::Half* base = tensor_ptr + batch_idx * tensor_stride;

    bool is_p1_turn = node->is_p1_to_move();
    uint8_t my_row, my_col, my_fences;
    uint8_t opp_row, opp_col, opp_fences;
    uint64_t h_fences, v_fences;

    if (is_p1_turn) {
        my_row = node->p1.row;
        my_col = node->p1.col;
        my_fences = node->p1.fences;
        opp_row = node->p2.row;
        opp_col = node->p2.col;
        opp_fences = node->p2.fences;
        h_fences = node->fences.horizontal;
        v_fences = node->fences.vertical;
    } else {
        my_row = 8 - node->p2.row;
        my_col = 8 - node->p2.col;
        my_fences = node->p2.fences;
        opp_row = 8 - node->p1.row;
        opp_col = 8 - node->p1.col;
        opp_fences = node->p1.fences;
        h_fences = reverse_bits(node->fences.horizontal);
        v_fences = reverse_bits(node->fences.vertical);
    }

    base[0 * 81 + my_row * 9 + my_col] = at::Half(1.0f);
    base[1 * 81 + opp_row * 9 + opp_col] = at::Half(1.0f);

    for (int r = 0; r < 8; ++r) {
        uint64_t h_row = (h_fences >> (r * 8)) & 0xFF;
        uint64_t v_row = (v_fences >> (r * 8)) & 0xFF;
        for (int c = 0; c < 8; ++c) {
            if ((h_row >> c) & 1) base[2 * 81 + r * 9 + c] = at::Half(1.0f);
            if ((v_row >> c) & 1) base[3 * 81 + r * 9 + c] = at::Half(1.0f);
        }
    }

    at::Half my_fences_norm = at::Half(static_cast<float>(my_fences) * 0.1f);
    at::Half opp_fences_norm = at::Half(static_cast<float>(opp_fences) * 0.1f);
    std::fill_n(base + 4 * 81, 81, my_fences_norm);
    std::fill_n(base + 5 * 81, 81, opp_fences_norm);
}

void ModelInference::queue_for_evaluation(const StateNode* node, uint32_t node_idx) {
    evaluation_queue_.emplace_back(node, node_idx);
}

void ModelInference::flush_queue(const EvalCallback& callback) {
    while (!evaluation_queue_.empty()) {
        process_batch(callback);
    }
}

void ModelInference::process_batch(const EvalCallback& callback) {
    if (evaluation_queue_.empty()) return;

    int current_batch_size = std::min(static_cast<int>(evaluation_queue_.size()), batch_size_);

    auto batch_tensor = torch::zeros({current_batch_size, 6, 9, 9}, 
        torch::TensorOptions().dtype(torch::kHalf).pinned_memory(device_.is_cuda()));

    std::vector<uint32_t> batch_indices;
    batch_indices.reserve(current_batch_size);

    at::Half* tensor_ptr = batch_tensor.data_ptr<at::Half>();
    constexpr size_t tensor_stride = 6 * 9 * 9;

    for (int i = 0; i < current_batch_size; ++i) {
        auto [node, idx] = evaluation_queue_.front();
        evaluation_queue_.pop_front();
        fill_input_tensor(tensor_ptr, tensor_stride, i, node);
        batch_indices.push_back(idx);
    }

    batch_tensor = batch_tensor.to(device_, /*non_blocking=*/true);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(batch_tensor);

    torch::NoGradGuard no_grad;
    auto output = model_.forward(inputs);

    auto output_tuple = output.toTuple();
    auto policy_output = output_tuple->elements()[0].toTensor().to(torch::kCPU, torch::kFloat32);
    auto value_output = output_tuple->elements()[1].toTensor().to(torch::kCPU, torch::kFloat32);

    const float* policy_ptr = policy_output.data_ptr<float>();
    const float* value_ptr = value_output.data_ptr<float>();

    for (int i = 0; i < current_batch_size; ++i) {
        EvalResult result;
        result.value = std::isnan(value_ptr[i]) ? 0.0f : value_ptr[i];
        std::memcpy(result.policy.data(), policy_ptr + i * NUM_ACTIONS, NUM_ACTIONS * sizeof(float));
        callback(batch_indices[i], result);
    }
}

EvalResult ModelInference::evaluate_node(const StateNode* node) {
    auto batch_tensor = torch::zeros({1, 6, 9, 9}, 
        torch::TensorOptions().dtype(torch::kHalf).pinned_memory(device_.is_cuda()));

    at::Half* tensor_ptr = batch_tensor.data_ptr<at::Half>();
    constexpr size_t tensor_stride = 6 * 9 * 9;
    fill_input_tensor(tensor_ptr, tensor_stride, 0, node);

    batch_tensor = batch_tensor.to(device_, /*non_blocking=*/true);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(batch_tensor);

    torch::NoGradGuard no_grad;
    auto output = model_.forward(inputs);

    auto output_tuple = output.toTuple();
    auto policy_output = output_tuple->elements()[0].toTensor().to(torch::kCPU, torch::kFloat32);
    auto value_output = output_tuple->elements()[1].toTensor().to(torch::kCPU, torch::kFloat32);

    EvalResult result;
    float value = value_output.data_ptr<float>()[0];
    result.value = std::isnan(value) ? 0.0f : value;
    std::memcpy(result.policy.data(), policy_output.data_ptr<float>(), NUM_ACTIONS * sizeof(float));

    return result;
}

void ModelInference::ensure_buffer_capacity(int size) {
    if (size <= buffer_capacity_) return;

    int new_capacity = 256;
    while (new_capacity < size) new_capacity *= 2;

    unified_buffer_ = torch::zeros({new_capacity, 6, 9, 9}, 
        torch::TensorOptions().dtype(torch::kHalf).pinned_memory(device_.is_cuda()));
    value_output_buffer_ = torch::empty({new_capacity}, 
        torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(device_.is_cuda()));
    policy_output_buffer_ = torch::empty({new_capacity, NUM_ACTIONS}, 
        torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(device_.is_cuda()));

    buffer_capacity_ = new_capacity;
}

std::vector<EvalResult> ModelInference::evaluate_batch(const std::vector<const StateNode*>& nodes) {
    auto& timers = get_timers();
    std::vector<EvalResult> results;
    if (nodes.empty()) return results;

    const int batch_size = static_cast<int>(nodes.size());
    results.resize(batch_size);

    {
        ensure_buffer_capacity(batch_size);
    }

    {

        at::Half* tensor_ptr = unified_buffer_.data_ptr<at::Half>();
        constexpr size_t tensor_stride = 6 * 9 * 9;

        std::memset(tensor_ptr, 0, batch_size * tensor_stride * sizeof(at::Half));

        for (int i = 0; i < batch_size; ++i) {
            fill_input_tensor(tensor_ptr, tensor_stride, i, nodes[i]);
        }
    }

    torch::Tensor batch_tensor;
    {
        batch_tensor = unified_buffer_.slice(0, 0, batch_size).to(device_, /*non_blocking=*/true);
    }

    if (device_.is_cuda()) {
        torch::cuda::synchronize();
    }

    torch::Tensor policy_output, value_output;
    {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(batch_tensor);

        torch::NoGradGuard no_grad;
        auto output = model_.forward(inputs);

        auto output_tuple = output.toTuple();
        policy_output = output_tuple->elements()[0].toTensor();
        value_output = output_tuple->elements()[1].toTensor();
    }

    if (device_.is_cuda()) {
        torch::cuda::synchronize();
    }

    {
        value_output_buffer_.slice(0, 0, batch_size).copy_(
            value_output.flatten().to(torch::kFloat32), /*non_blocking=*/true);
        policy_output_buffer_.slice(0, 0, batch_size).copy_(
            policy_output.to(torch::kFloat32), /*non_blocking=*/true);
    }

    if (device_.is_cuda()) {
        torch::cuda::synchronize();
    }

    const float* value_ptr = value_output_buffer_.data_ptr<float>();
    const float* policy_ptr = policy_output_buffer_.data_ptr<float>();

    for (int i = 0; i < batch_size; ++i) {
        results[i].value = std::isnan(value_ptr[i]) ? 0.0f : value_ptr[i];
        std::memcpy(results[i].policy.data(), policy_ptr + i * NUM_ACTIONS, NUM_ACTIONS * sizeof(float));
    }

    return results;
}

std::vector<float> ModelInference::evaluate_batch_values(const std::vector<const StateNode*>& nodes) {
    auto results = evaluate_batch(nodes);
    std::vector<float> values;
    values.reserve(results.size());
    for (const auto& r : results) {
        values.push_back(r.value);
    }
    return values;
}

} // namespace qbot
