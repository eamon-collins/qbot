#include "inference.h"
#include "../tree/StateNode.h"
#include "../util/timer.h"

#include <iostream>

namespace qbot {

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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ModelInference::state_to_tensors(const StateNode* node) const {
    // Pawn positions: 2 channels (P1, P2), 9x9 board
    auto pawn_tensor = torch::zeros({2, 9, 9}, torch::kFloat32);
    pawn_tensor[0][node->p1.row][node->p1.col] = 1.0f;
    pawn_tensor[1][node->p2.row][node->p2.col] = 1.0f;

    // Wall positions: 2 channels (horizontal, vertical), 8x8 intersections
    // Our FenceGrid stores fences at intersection points directly
    auto wall_tensor = torch::zeros({2, 8, 8}, torch::kFloat32);

    // Extract fence positions from the bitboard representation
    const FenceGrid& fences = node->fences;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            if (fences.has_h_fence(r, c)) {
                wall_tensor[0][r][c] = 1.0f;
            }
            if (fences.has_v_fence(r, c)) {
                wall_tensor[1][r][c] = 1.0f;
            }
        }
    }

    // Meta information: remaining fences for each player + turn indicator
    auto meta_tensor = torch::zeros({3}, torch::kFloat32);
    meta_tensor[0] = static_cast<float>(node->p1.fences);
    meta_tensor[1] = static_cast<float>(node->p2.fences);
    meta_tensor[2] = node->is_p1_to_move() ? 1.0f : 0.0f;  // Turn indicator

    return {pawn_tensor, wall_tensor, meta_tensor};
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

    // Prepare batch tensors
    auto batch_pawn = torch::zeros({current_batch_size, 2, 9, 9}, torch::kFloat32);
    auto batch_wall = torch::zeros({current_batch_size, 2, 8, 8}, torch::kFloat32);
    auto batch_meta = torch::zeros({current_batch_size, 3}, torch::kFloat32);

    // Track which nodes are in this batch
    std::vector<uint32_t> batch_indices;
    batch_indices.reserve(current_batch_size);

    // Fill batch tensors
    for (int i = 0; i < current_batch_size; ++i) {
        auto [node, idx] = evaluation_queue_.front();
        evaluation_queue_.pop_front();

        auto [pawn, wall, meta] = state_to_tensors(node);
        batch_pawn[i] = pawn;
        batch_wall[i] = wall;
        batch_meta[i] = meta;

        batch_indices.push_back(idx);
    }

    // Move tensors to device
    batch_pawn = batch_pawn.to(device_);
    batch_wall = batch_wall.to(device_);
    batch_meta = batch_meta.to(device_);

    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(batch_pawn);
    inputs.push_back(batch_wall);
    inputs.push_back(batch_meta);

    torch::NoGradGuard no_grad;
    auto output = model_.forward(inputs).toTensor();

    // Move output back to CPU
    output = output.to(torch::kCPU);

    // Call callback for each result
    for (int i = 0; i < current_batch_size; ++i) {
        float value = output[i].item<float>();

        // Handle NaN values
        if (std::isnan(value)) {
            value = 0.0f;
        }

        callback(batch_indices[i], value);
    }
}

float ModelInference::evaluate_node(const StateNode* node) {
    // Reuse batch path for consistency and to benefit from same optimizations
    // Single-node eval is inherently slow on GPU due to poor utilization
    auto options = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(device_.is_cuda());
    auto batch_pawn = torch::zeros({1, 2, 9, 9}, options);
    auto batch_wall = torch::zeros({1, 2, 8, 8}, options);
    auto batch_meta = torch::zeros({1, 3}, options);

    // Fill directly via accessors
    auto pawn_acc = batch_pawn.accessor<float, 4>();
    auto wall_acc = batch_wall.accessor<float, 4>();
    auto meta_acc = batch_meta.accessor<float, 2>();

    pawn_acc[0][0][node->p1.row][node->p1.col] = 1.0f;
    pawn_acc[0][1][node->p2.row][node->p2.col] = 1.0f;

    const FenceGrid& fences = node->fences;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            if (fences.has_h_fence(r, c)) {
                wall_acc[0][0][r][c] = 1.0f;
            }
            if (fences.has_v_fence(r, c)) {
                wall_acc[0][1][r][c] = 1.0f;
            }
        }
    }

    meta_acc[0][0] = static_cast<float>(node->p1.fences);
    meta_acc[0][1] = static_cast<float>(node->p2.fences);
    meta_acc[0][2] = node->is_p1_to_move() ? 1.0f : 0.0f;

    // Move to device
    batch_pawn = batch_pawn.to(device_, /*non_blocking=*/true);
    batch_wall = batch_wall.to(device_, /*non_blocking=*/true);
    batch_meta = batch_meta.to(device_, /*non_blocking=*/true);

    std::vector<torch::jit::IValue> inputs;
    inputs.reserve(3);
    inputs.push_back(batch_pawn);
    inputs.push_back(batch_wall);
    inputs.push_back(batch_meta);

    torch::NoGradGuard no_grad;
    auto output = model_.forward(inputs).toTensor();

    float value = output.to(torch::kCPU).data_ptr<float>()[0];
    return std::isnan(value) ? 0.0f : value;
}

void ModelInference::ensure_buffer_capacity(int size) {
    if (size <= buffer_capacity_) return;

    // Round up to power of 2 for efficiency
    int new_capacity = 256;
    while (new_capacity < size) new_capacity *= 2;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(device_.is_cuda());
    pawn_buffer_ = torch::zeros({new_capacity, 2, 9, 9}, options);
    wall_buffer_ = torch::zeros({new_capacity, 2, 8, 8}, options);
    meta_buffer_ = torch::zeros({new_capacity, 3}, options);
    buffer_capacity_ = new_capacity;
}

std::vector<float> ModelInference::evaluate_batch(const std::vector<const StateNode*>& nodes) {
    auto& timers = get_timers();
    std::vector<float> results;
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

        // Get raw pointers for fast zeroing and filling
        float* pawn_ptr = pawn_buffer_.data_ptr<float>();
        float* wall_ptr = wall_buffer_.data_ptr<float>();
        float* meta_ptr = meta_buffer_.data_ptr<float>();

        // Zero the buffers for this batch (only the part we'll use)
        const size_t pawn_stride = 2 * 9 * 9;
        const size_t wall_stride = 2 * 8 * 8;
        const size_t meta_stride = 3;

        std::memset(pawn_ptr, 0, batch_size * pawn_stride * sizeof(float));
        std::memset(wall_ptr, 0, batch_size * wall_stride * sizeof(float));
        std::memset(meta_ptr, 0, batch_size * meta_stride * sizeof(float));

        // Fill the tensors
        for (int i = 0; i < batch_size; ++i) {
            const StateNode* node = nodes[i];

            // Pawn positions: buffer[i][channel][row][col]
            // Index = i * (2*9*9) + channel * (9*9) + row * 9 + col
            float* pawn_base = pawn_ptr + i * pawn_stride;
            pawn_base[0 * 81 + node->p1.row * 9 + node->p1.col] = 1.0f;  // P1
            pawn_base[1 * 81 + node->p2.row * 9 + node->p2.col] = 1.0f;  // P2

            // Wall positions: buffer[i][channel][row][col]
            float* wall_base = wall_ptr + i * wall_stride;
            const FenceGrid& fences = node->fences;
            for (int r = 0; r < 8; ++r) {
                for (int c = 0; c < 8; ++c) {
                    if (fences.has_h_fence(r, c)) {
                        wall_base[0 * 64 + r * 8 + c] = 1.0f;
                    }
                    if (fences.has_v_fence(r, c)) {
                        wall_base[1 * 64 + r * 8 + c] = 1.0f;
                    }
                }
            }

            // Meta: [p1_fences, p2_fences, is_p1_turn]
            float* meta_base = meta_ptr + i * meta_stride;
            meta_base[0] = static_cast<float>(node->p1.fences);
            meta_base[1] = static_cast<float>(node->p2.fences);
            meta_base[2] = node->is_p1_to_move() ? 1.0f : 0.0f;
        }
    }

    // Slice buffers to actual batch size and move to GPU
    torch::Tensor batch_pawn, batch_wall, batch_meta;
    {
        ScopedTimer t(timers.tensor_to_gpu);
        batch_pawn = pawn_buffer_.slice(0, 0, batch_size).to(device_, /*non_blocking=*/true);
        batch_wall = wall_buffer_.slice(0, 0, batch_size).to(device_, /*non_blocking=*/true);
        batch_meta = meta_buffer_.slice(0, 0, batch_size).to(device_, /*non_blocking=*/true);
    }

    // Run inference
    torch::Tensor output;
    {
        ScopedTimer t(timers.model_forward);
        std::vector<torch::jit::IValue> inputs;
        inputs.reserve(3);
        inputs.push_back(batch_pawn);
        inputs.push_back(batch_wall);
        inputs.push_back(batch_meta);

        torch::NoGradGuard no_grad;
        output = model_.forward(inputs).toTensor();
    }

    // Move output to CPU
    {
        ScopedTimer t(timers.tensor_to_cpu);
        output = output.to(torch::kCPU).contiguous();
    }

    const float* output_ptr = output.data_ptr<float>();
    for (int i = 0; i < batch_size; ++i) {
        float value = output_ptr[i];
        results[i] = std::isnan(value) ? 0.0f : value;
    }

    return results;
}

void ModelInference::print_diagnostics() {
    std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;
    std::cout << "CUDA available: " << (torch::cuda::is_available() ? "yes" : "no") << std::endl;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    }
}

} // namespace qbot
