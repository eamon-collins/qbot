#include "inference.h"
#include "../tree/StateNode.h"

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
    auto [pawn, wall, meta] = state_to_tensors(node);

    // Add batch dimension and move to device
    pawn = pawn.unsqueeze(0).to(device_);
    wall = wall.unsqueeze(0).to(device_);
    meta = meta.unsqueeze(0).to(device_);

    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(pawn);
    inputs.push_back(wall);
    inputs.push_back(meta);

    torch::NoGradGuard no_grad;
    auto output = model_.forward(inputs).toTensor();

    float value = output.to(torch::kCPU).item<float>();

    if (std::isnan(value)) {
        value = 0.0f;
    }

    return value;
}

std::vector<float> ModelInference::evaluate_batch(const std::vector<const StateNode*>& nodes) {
    std::vector<float> results;
    if (nodes.empty()) return results;

    const int batch_size = static_cast<int>(nodes.size());
    results.reserve(batch_size);

    // Prepare batch tensors
    auto batch_pawn = torch::zeros({batch_size, 2, 9, 9}, torch::kFloat32);
    auto batch_wall = torch::zeros({batch_size, 2, 8, 8}, torch::kFloat32);
    auto batch_meta = torch::zeros({batch_size, 3}, torch::kFloat32);

    // Fill batch tensors
    for (int i = 0; i < batch_size; ++i) {
        auto [pawn, wall, meta] = state_to_tensors(nodes[i]);
        batch_pawn[i] = pawn;
        batch_wall[i] = wall;
        batch_meta[i] = meta;
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

    // Extract values
    for (int i = 0; i < batch_size; ++i) {
        float value = output[i].item<float>();
        if (std::isnan(value)) {
            value = 0.0f;
        }
        results.push_back(value);
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
