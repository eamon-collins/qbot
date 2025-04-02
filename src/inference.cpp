#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <deque>
#include <memory>
#include "Tree.h"

class ModelInference {
private:
    torch::jit::script::Module model;
    torch::Device device;
    int batch_size;
    std::deque<StateNode*> evaluation_queue;
    
public:
    ModelInference(const std::string& model_path, int batch_size = 16) 
        : batch_size(batch_size), device(torch::kCUDA, 0) {
        try {
            // Load the TorchScript model
            model = torch::jit::load(model_path);
            model.to(device);
            std::cout << "Model loaded successfully to CUDA device\n";
        }
        catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            throw;
        }
    }

    // Convert a single StateNode to tensor inputs
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> state_to_tensors(const StateNode* node) {
        // Pawn state - 2x9x9
        auto pawn_tensor = torch::zeros({2, 9, 9}, torch::kFloat32);
        
        // Set player positions
        pawn_tensor[0][node->p1.row][node->p1.col] = 1.0;
        pawn_tensor[1][node->p2.row][node->p2.col] = 1.0;
        
        // Wall state - 2x8x8
        auto wall_tensor = torch::zeros({2, 8, 8}, torch::kFloat32);
        
        // Convert gamestate to wall tensor
        for (int i = 0; i < 8; i++) {
            // Horizontal walls
            for (int j = 0; j < 8; j++) {
                int row = 2*i + 1;  // Odd rows contain horizontal walls
                if (node->gamestate[row][j] && node->gamestate[row][j+1]) {
                    wall_tensor[0][i][j] = 1.0;
                }
            }
            
            // Vertical walls
            for (int j = 0; j < 8; j++) {
                int row = 2*i;  // Even rows contain vertical walls
                if (node->gamestate[row][j] && node->gamestate[row+2][j]) {
                    wall_tensor[1][i][j] = 1.0;
                }
            }
        }
        
        // Meta state - remaining fences for each player
        auto meta_tensor = torch::zeros({2}, torch::kFloat32);
        meta_tensor[0] = node->p1.numFences;
        meta_tensor[1] = node->p2.numFences;
        
        return {pawn_tensor, wall_tensor, meta_tensor};
    }

    // Queue a node for batch evaluation
    void queue_for_evaluation(StateNode* node) {
        evaluation_queue.push_back(node);
        
        // Process a batch if we have enough nodes
        if (evaluation_queue.size() >= batch_size) {
            process_batch();
        }
    }
    
    // Process any remaining nodes in the queue
    void flush_queue() {
        if (!evaluation_queue.empty()) {
            process_batch();
        }
    }
    
    // Process a batch of nodes
    void process_batch() {
        // Determine batch size for this run
        int current_batch_size = std::min(static_cast<int>(evaluation_queue.size()), batch_size);
        
        // Prepare batch tensors
        auto batch_pawn = torch::zeros({current_batch_size, 2, 9, 9}, torch::kFloat32);
        auto batch_wall = torch::zeros({current_batch_size, 2, 8, 8}, torch::kFloat32);
        auto batch_meta = torch::zeros({current_batch_size, 2}, torch::kFloat32);
        
        // Copy data to batch tensors
        for (int i = 0; i < current_batch_size; i++) {
            StateNode* node = evaluation_queue.front();
            evaluation_queue.pop_front();
            
            auto [pawn_tensor, wall_tensor, meta_tensor] = state_to_tensors(node);
            
            batch_pawn[i] = pawn_tensor;
            batch_wall[i] = wall_tensor;
            batch_meta[i] = meta_tensor;
        }
        
        // Move tensors to GPU
        batch_pawn = batch_pawn.to(device);
        batch_wall = batch_wall.to(device);
        batch_meta = batch_meta.to(device);
        
        // Create a vector of inputs for the model
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(batch_pawn);
        inputs.push_back(batch_wall);
        inputs.push_back(batch_meta);
        
        // Forward pass
        torch::NoGradGuard no_grad;
        auto output = model.forward(inputs).toTensor();
        
        // Move output back to CPU for processing
        output = output.to(torch::kCPU);
        
        // Update nodes with model predictions
        for (int i = 0; i < current_batch_size; i++) {
            StateNode* node = evaluation_queue.front();
            evaluation_queue.pop_front();
            
            // Extract the value from the model output
            float value = output[i].item<float>();
            
            // Update the node's score
            node->score = value;
            
            // If you want to avoid NaN values or handle other edge cases
            if (std::isnan(value)) {
                node->score = 0.0;
            }
        }
    }
    
    // Evaluate a single node immediately
    float evaluate_node(StateNode* node) {
        auto [pawn_tensor, wall_tensor, meta_tensor] = state_to_tensors(node);
        
        // Move tensors to GPU
        pawn_tensor = pawn_tensor.unsqueeze(0).to(device);
        wall_tensor = wall_tensor.unsqueeze(0).to(device);
        meta_tensor = meta_tensor.unsqueeze(0).to(device);
        
        // Create inputs
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(pawn_tensor);
        inputs.push_back(wall_tensor);
        inputs.push_back(meta_tensor);
        
        // Forward pass
        torch::NoGradGuard no_grad;
        auto output = model.forward(inputs).toTensor();
        
        // Move output back to CPU and extract value
        float value = output.to(torch::kCPU).item<float>();
        
        return value;
    }
};

#ifdef INFERENCE_MAIN
// Simple main function to test the inference functionality
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./inference <model_path>\n";
        return 1;
    }
    
    std::string model_path = argv[1];
    
    try {
        std::cout << "Initializing ModelInference with model: " << model_path << std::endl;
        ModelInference inference(model_path);
        
        // Create a test node
        StateNode test_node(true);  // true means player 1's turn
        
        // Evaluate the node
        float value = inference.evaluate_node(&test_node);
        std::cout << "Model evaluation of initial state: " << value << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
#endif
