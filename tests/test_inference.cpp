#include "inference/inference.h"
#include "core/Game.h"
#include "tree/StateNode.h"

#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using namespace qbot;

namespace {
bool is_single_test_run() {
    const auto* filter = ::testing::GTEST_FLAG(filter).c_str();
    return filter && std::string(filter).find('*') == std::string::npos
           && std::string(filter) != "";
}

const char* DEFAULT_MODEL_PATH = "/home/eamon/repos/qbot/model/tree.pt";
const char* TEST_MODEL_PATH = "/tmp/test_model.pt";

// Create a fresh model with current architecture for testing
bool create_test_model() {
    // Python script to create and export a fresh model with new unified input format
    const char* script = R"(
import sys
sys.path.insert(0, '/home/eamon/repos/qbot/train')
import torch
from resnet import QuoridorNet

model = QuoridorNet()
model.eval()

# Create example input with unified 6-channel format (current-player-perspective)
example_input = torch.zeros(1, 6, 9, 9)

traced = torch.jit.trace(model, example_input)
traced.save('/tmp/test_model.pt')
print('Model saved successfully')
)";

    std::string cmd = "python3 -c \"" + std::string(script) + "\" 2>&1";
    int result = std::system(cmd.c_str());
    return result == 0;
}

// Check if a model file is compatible with current input dimensions
bool is_model_compatible(const std::string& path) {
    try {
        ModelInference inference(path, 1, false);
        if (!inference.is_ready()) return false;

        // Create a dummy node and try to evaluate
        StateNode dummy;
        dummy.init_root(true);
        inference.evaluate_node(&dummy);
        return true;
    } catch (...) {
        return false;
    }
}
}

class InferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        GameConfig config;
        config.pool_capacity = 10'000;
        game_ = std::make_unique<Game>(config);
    }

    uint32_t create_root() {
        uint32_t root_idx = game_->pool().allocate();
        EXPECT_NE(root_idx, NULL_NODE);
        game_->pool()[root_idx].init_root(true);
        game_->set_root(root_idx);
        return root_idx;
    }

    // Find all leaf nodes in the tree
    std::vector<uint32_t> find_leaf_nodes(uint32_t root_idx) {
        std::vector<uint32_t> leaves;
        std::vector<uint32_t> stack;
        stack.push_back(root_idx);

        while (!stack.empty()) {
            uint32_t idx = stack.back();
            stack.pop_back();

            const StateNode& node = game_->pool()[idx];

            if (!node.has_children()) {
                leaves.push_back(idx);
            } else {
                uint32_t child = node.first_child;
                while (child != NULL_NODE) {
                    stack.push_back(child);
                    child = game_->pool()[child].next_sibling;
                }
            }
        }
        return leaves;
    }

    std::unique_ptr<Game> game_;
};

TEST_F(InferenceTest, EvaluateRandomLeafNode) {
    const bool verbose = is_single_test_run();

    // Determine which model to use
    std::string model_path;
    std::ifstream default_model(DEFAULT_MODEL_PATH);
    if (default_model.good() && is_model_compatible(DEFAULT_MODEL_PATH)) {
        model_path = DEFAULT_MODEL_PATH;
        if (verbose) {
            std::cout << "Using default model: " << DEFAULT_MODEL_PATH << std::endl;
        }
    } else {
        // Create a fresh test model with current architecture
        if (verbose) {
            std::cout << "Default model missing or incompatible, creating test model..." << std::endl;
        }
        if (!create_test_model()) {
            GTEST_SKIP() << "Failed to create test model (requires Python with PyTorch)";
            return;
        }
        model_path = TEST_MODEL_PATH;
    }

    // Build a small tree
    uint32_t root = create_root();
    size_t created = game_->build_tree(root, 0.3f, 5, 500);

    if (verbose) {
        std::cout << "\n========== Inference Test ==========" << std::endl;
        std::cout << "Built tree with " << created << " nodes" << std::endl;
    }

    ASSERT_GT(created, 0) << "Should create some nodes";

    // Find leaf nodes
    auto leaves = find_leaf_nodes(root);
    ASSERT_GT(leaves.size(), 0) << "Should have leaf nodes";

    if (verbose) {
        std::cout << "Found " << leaves.size() << " leaf nodes" << std::endl;
    }

    // Pick a random leaf
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<size_t> dist(0, leaves.size() - 1);
    uint32_t leaf_idx = leaves[dist(rng)];

    const StateNode& leaf = game_->pool()[leaf_idx];

    if (verbose) {
        std::cout << "\n--- Selected leaf node (index " << leaf_idx << ") ---" << std::endl;
        leaf.print_node();
    }

    // Load model and evaluate
    try {
        ModelInference inference(model_path, 16, true);

        if (verbose) {
            std::cout << "\n--- Model Diagnostics ---" << std::endl;
            ModelInference::print_diagnostics();
        }

        ASSERT_TRUE(inference.is_ready()) << "Model should be loaded";

        EvalResult result = inference.evaluate_node(&leaf);

        if (verbose) {
            std::cout << "\n--- Evaluation Result ---" << std::endl;
            std::cout << "Value: " << result.value << std::endl;
            std::cout << "Policy (first 10 actions): ";
            for (int i = 0; i < 10; ++i) {
                std::cout << result.policy[i] << " ";
            }
            std::cout << "..." << std::endl;
            std::cout << "========================================\n" << std::endl;
        }

        // Value should be a valid number (not NaN or Inf)
        EXPECT_FALSE(std::isnan(result.value)) << "Value should not be NaN";
        EXPECT_FALSE(std::isinf(result.value)) << "Value should not be Inf";

        // Policy should have valid values
        for (int i = 0; i < NUM_ACTIONS; ++i) {
            EXPECT_FALSE(std::isnan(result.policy[i])) << "Policy[" << i << "] should not be NaN";
        }

    } catch (const c10::Error& e) {
        GTEST_SKIP() << "Failed to load model: " << e.what();
    }
}

TEST_F(InferenceTest, BatchEvaluation) {
    const bool verbose = is_single_test_run();

    // Determine which model to use
    std::string model_path;
    std::ifstream default_model(DEFAULT_MODEL_PATH);
    if (default_model.good() && is_model_compatible(DEFAULT_MODEL_PATH)) {
        model_path = DEFAULT_MODEL_PATH;
    } else {
        // Create a fresh test model with current architecture
        if (!create_test_model()) {
            GTEST_SKIP() << "Failed to create test model (requires Python with PyTorch)";
            return;
        }
        model_path = TEST_MODEL_PATH;
    }

    // Build a tree
    uint32_t root = create_root();
    game_->build_tree(root, 0.5f, 5, 1000);

    auto leaves = find_leaf_nodes(root);
    ASSERT_GT(leaves.size(), 0);

    try {
        ModelInference inference(model_path, 8, true);
        ASSERT_TRUE(inference.is_ready());

        // Queue several nodes for batch evaluation
        size_t num_to_eval = std::min(leaves.size(), size_t{20});
        std::vector<uint32_t> queued_indices;

        for (size_t i = 0; i < num_to_eval; ++i) {
            inference.queue_for_evaluation(&game_->pool()[leaves[i]], leaves[i]);
            queued_indices.push_back(leaves[i]);
        }

        EXPECT_EQ(inference.queue_size(), num_to_eval);

        // Collect results
        std::vector<std::pair<uint32_t, EvalResult>> results;
        inference.flush_queue([&results](uint32_t idx, const EvalResult& result) {
            results.emplace_back(idx, result);
        });

        EXPECT_EQ(results.size(), num_to_eval);
        EXPECT_EQ(inference.queue_size(), 0);

        if (verbose) {
            std::cout << "\n========== Batch Evaluation Test ==========" << std::endl;
            std::cout << "Evaluated " << results.size() << " nodes in batch" << std::endl;
            std::cout << "\nResults:" << std::endl;
            for (const auto& [idx, result] : results) {
                std::cout << "  Node " << idx << ": value=" << result.value << std::endl;
            }
            std::cout << "==========================================\n" << std::endl;
        }

        // All values should be valid
        for (const auto& [idx, result] : results) {
            EXPECT_FALSE(std::isnan(result.value)) << "Value for node " << idx << " should not be NaN";
            EXPECT_FALSE(std::isinf(result.value)) << "Value for node " << idx << " should not be Inf";
        }

    } catch (const c10::Error& e) {
        GTEST_SKIP() << "Failed to load model: " << e.what();
    }
}
