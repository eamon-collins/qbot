#include "inference.h"
#include "../tree/StateNode.h"
#include "../util/timer.h"

#include <iostream>

namespace qbot {

int move_to_action_index(const Move& move) noexcept {
    if (!move.is_valid()) return -1;

    if (move.is_pawn()) {
        // Pawn moves: row * 9 + col (indices 0-80)
        return move.row() * 9 + move.col();
    } else {
        // Fence moves
        int base = move.is_horizontal() ? NUM_PAWN_ACTIONS : (NUM_PAWN_ACTIONS + NUM_H_WALL_ACTIONS);
        return base + move.row() * 8 + move.col();
    }
}

Move action_index_to_move(int action_index) noexcept {
    if (action_index < 0 || action_index >= NUM_ACTIONS) {
        return Move{};  // Invalid move
    }

    if (action_index < NUM_PAWN_ACTIONS) {
        // Pawn move
        uint8_t row = static_cast<uint8_t>(action_index / 9);
        uint8_t col = static_cast<uint8_t>(action_index % 9);
        return Move::pawn(row, col);
    } else if (action_index < NUM_PAWN_ACTIONS + NUM_H_WALL_ACTIONS) {
        // Horizontal wall
        int wall_idx = action_index - NUM_PAWN_ACTIONS;
        uint8_t row = static_cast<uint8_t>(wall_idx / 8);
        uint8_t col = static_cast<uint8_t>(wall_idx % 8);
        return Move::fence(row, col, true);
    } else {
        // Vertical wall
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
        model_ = torch::jit::load(model_path, torch::kCPU);
        model_.to(device_);
        model_.eval();
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

void ModelInference::fill_unified_tensor(float* tensor_ptr, size_t tensor_stride,
                                          int batch_idx, const StateNode* node) const {
    // Tensor layout: [batch, 6, 9, 9]
    // tensor_stride = 6 * 9 * 9 = 486
    float* base = tensor_ptr + batch_idx * tensor_stride;

    // Determine current player and opponent based on whose turn it is
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
        // P2 Perspective (Relative/Flipped 180)
        // Flip coordinates: 8 - row
        my_row = 8 - node->p2.row;
        my_col = 8 - node->p2.col;
        my_fences = node->p2.fences;
        
        opp_row = 8 - node->p1.row;
        opp_col = 8 - node->p1.col;
        opp_fences = node->p1.fences;
        
        // Flip fences (bit reversal performs the spatial 180 flip for the grid)
        h_fences = reverse_bits(node->fences.horizontal);
        v_fences = reverse_bits(node->fences.vertical);
    }
    // Channel 0: Current player's pawn position (one-hot)
    // Index = channel * 81 + row * 9 + col
    base[0 * 81 + my_row * 9 + my_col] = 1.0f;

    // Channel 1: Opponent's pawn position (one-hot)
    base[1 * 81 + opp_row * 9 + opp_col] = 1.0f;

    // Channel 2: Horizontal walls (8x8 grid padded to 9x9)
    // Channel 3: Vertical walls (8x8 grid padded to 9x9)
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            // Check bits from the (potentially flipped) integers
            if ((h_fences >> (r * 8 + c)) & 1) {
                base[2 * 81 + r * 9 + c] = 1.0f;
            }
            if ((v_fences >> (r * 8 + c)) & 1) {
                base[3 * 81 + r * 9 + c] = 1.0f;
            }
        }
    }

    // Channel 4: Current player's fences remaining / 10 (constant plane)
    float my_fences_norm = static_cast<float>(my_fences) / 10.0f;
    for (int i = 0; i < 81; ++i) {
        base[4 * 81 + i] = my_fences_norm;
    }

    // Channel 5: Opponent's fences remaining / 10 (constant plane)
    float opp_fences_norm = static_cast<float>(opp_fences) / 10.0f;
    for (int i = 0; i < 81; ++i) {
        base[5 * 81 + i] = opp_fences_norm;
    }
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

    // Prepare unified batch tensor
    auto batch_tensor = torch::zeros({current_batch_size, 6, 9, 9}, torch::kFloat32);

    // Track which nodes are in this batch
    std::vector<uint32_t> batch_indices;
    batch_indices.reserve(current_batch_size);

    // Fill batch tensor
    float* tensor_ptr = batch_tensor.data_ptr<float>();
    constexpr size_t tensor_stride = 6 * 9 * 9;

    for (int i = 0; i < current_batch_size; ++i) {
        auto [node, idx] = evaluation_queue_.front();
        evaluation_queue_.pop_front();

        // Zero this entry (tensor is already zeroed, but be safe for reuse)
        std::memset(tensor_ptr + i * tensor_stride, 0, tensor_stride * sizeof(float));
        fill_unified_tensor(tensor_ptr, tensor_stride, i, node);

        batch_indices.push_back(idx);
    }

    // Move tensor to device
    batch_tensor = batch_tensor.to(device_);

    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(batch_tensor);

    torch::NoGradGuard no_grad;
    auto output = model_.forward(inputs);

    // Handle tuple output (policy_logits, value)
    auto output_tuple = output.toTuple();
    auto policy_output = output_tuple->elements()[0].toTensor().to(torch::kCPU);
    auto value_output = output_tuple->elements()[1].toTensor().to(torch::kCPU);

    // Call callback for each result
    for (int i = 0; i < current_batch_size; ++i) {
        EvalResult result;

        // Extract value (squeeze if needed)
        float value = value_output[i].item<float>();
        result.value = std::isnan(value) ? 0.0f : value;

        // Extract policy logits
        auto policy_slice = policy_output[i];
        for (int a = 0; a < NUM_ACTIONS; ++a) {
            result.policy[a] = policy_slice[a].item<float>();
        }

        callback(batch_indices[i], result);
    }
}

EvalResult ModelInference::evaluate_node(const StateNode* node) {
    // Create single-item batch
    auto options = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(device_.is_cuda());
    auto batch_tensor = torch::zeros({1, 6, 9, 9}, options);

    // Fill tensor
    float* tensor_ptr = batch_tensor.data_ptr<float>();
    constexpr size_t tensor_stride = 6 * 9 * 9;
    fill_unified_tensor(tensor_ptr, tensor_stride, 0, node);

    // Move to device
    batch_tensor = batch_tensor.to(device_, /*non_blocking=*/true);

    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(batch_tensor);

    torch::NoGradGuard no_grad;
    auto output = model_.forward(inputs);

    // Handle tuple output (policy_logits, value)
    auto output_tuple = output.toTuple();
    auto policy_output = output_tuple->elements()[0].toTensor().to(torch::kCPU);
    auto value_output = output_tuple->elements()[1].toTensor().to(torch::kCPU);

    EvalResult result;
    float value = value_output[0].item<float>();
    result.value = std::isnan(value) ? 0.0f : value;

    for (int a = 0; a < NUM_ACTIONS; ++a) {
        result.policy[a] = policy_output[0][a].item<float>();
    }

    return result;
}

void ModelInference::ensure_buffer_capacity(int size) {
    if (size <= buffer_capacity_) return;

    // Round up to power of 2 for efficiency
    int new_capacity = 256;
    while (new_capacity < size) new_capacity *= 2;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(device_.is_cuda());
    unified_buffer_ = torch::zeros({new_capacity, 6, 9, 9}, options);
    value_output_buffer_ = torch::empty({new_capacity}, options);
    policy_output_buffer_ = torch::empty({new_capacity, NUM_ACTIONS}, options);
    buffer_capacity_ = new_capacity;
}

std::vector<EvalResult> ModelInference::evaluate_batch(const std::vector<const StateNode*>& nodes) {
    auto& timers = get_timers();
    std::vector<EvalResult> results;
    if (nodes.empty()) return results;

    const int batch_size = static_cast<int>(nodes.size());
    results.resize(batch_size);

    // Ensure we have enough buffer space
    {
        ScopedTimer t(timers.tensor_alloc);
        ensure_buffer_capacity(batch_size);
    }

    // Zero and fill only the portion we need
    {
        ScopedTimer t(timers.tensor_fill);

        float* tensor_ptr = unified_buffer_.data_ptr<float>();
        constexpr size_t tensor_stride = 6 * 9 * 9;

        // Zero the buffer for this batch
        std::memset(tensor_ptr, 0, batch_size * tensor_stride * sizeof(float));

        // Fill the tensors
        for (int i = 0; i < batch_size; ++i) {
            fill_unified_tensor(tensor_ptr, tensor_stride, i, nodes[i]);
        }
    }

    // Slice buffer to actual batch size and move to GPU (async)
    torch::Tensor batch_tensor;
    {
        ScopedTimer t(timers.tensor_to_gpu);
        batch_tensor = unified_buffer_.slice(0, 0, batch_size).to(device_, /*non_blocking=*/true);
    }

    // Sync to measure actual upload time
    if (device_.is_cuda()) {
        ScopedTimer t(timers.gpu_sync_upload);
        torch::cuda::synchronize();
    }

    // Run inference (async launch)
    torch::Tensor policy_output, value_output;
    {
        ScopedTimer t(timers.model_forward);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(batch_tensor);

        torch::NoGradGuard no_grad;
        auto output = model_.forward(inputs);

        // Handle tuple output (policy_logits, value)
        auto output_tuple = output.toTuple();
        policy_output = output_tuple->elements()[0].toTensor();
        value_output = output_tuple->elements()[1].toTensor();
    }

    // Sync to measure actual forward time
    if (device_.is_cuda()) {
        ScopedTimer t(timers.gpu_sync_forward);
        torch::cuda::synchronize();
    }

    // Copy outputs to pinned CPU buffers (async)
    {
        ScopedTimer t(timers.tensor_to_cpu);
        value_output_buffer_.slice(0, 0, batch_size).copy_(value_output.flatten(), /*non_blocking=*/true);
        policy_output_buffer_.slice(0, 0, batch_size).copy_(policy_output, /*non_blocking=*/true);
    }

    // Sync to measure actual download time
    if (device_.is_cuda()) {
        ScopedTimer t(timers.gpu_sync_download);
        torch::cuda::synchronize();
    }

    // Extract results
    const float* value_ptr = value_output_buffer_.data_ptr<float>();
    const float* policy_ptr = policy_output_buffer_.data_ptr<float>();

    for (int i = 0; i < batch_size; ++i) {
        float value = value_ptr[i];
        results[i].value = std::isnan(value) ? 0.0f : value;

        const float* policy_row = policy_ptr + i * NUM_ACTIONS;
        for (int a = 0; a < NUM_ACTIONS; ++a) {
            results[i].policy[a] = policy_row[a];
        }
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

void ModelInference::print_diagnostics() {
    std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;
    std::cout << "CUDA available: " << (torch::cuda::is_available() ? "yes" : "no") << std::endl;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    }
}

} // namespace qbot
