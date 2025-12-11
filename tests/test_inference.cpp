#include "inference/inference.h"
#include "core/Game.h"
#include "tree/StateNode.h"

#include <gtest/gtest.h>
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

    // Check if model file exists
    std::ifstream model_file(DEFAULT_MODEL_PATH);
    if (!model_file.good()) {
        GTEST_SKIP() << "Model file not found: " << DEFAULT_MODEL_PATH;
        return;
    }
    model_file.close();

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
        ModelInference inference(DEFAULT_MODEL_PATH, 16, true);

        if (verbose) {
            std::cout << "\n--- Model Diagnostics ---" << std::endl;
            ModelInference::print_diagnostics();
        }

        ASSERT_TRUE(inference.is_ready()) << "Model should be loaded";

        float score = inference.evaluate_node(&leaf);

        if (verbose) {
            std::cout << "\n--- Evaluation Result ---" << std::endl;
            std::cout << "Score: " << score << std::endl;
            std::cout << "========================================\n" << std::endl;
        }

        // Score should be a valid number (not NaN or Inf)
        EXPECT_FALSE(std::isnan(score)) << "Score should not be NaN";
        EXPECT_FALSE(std::isinf(score)) << "Score should not be Inf";

    } catch (const c10::Error& e) {
        GTEST_SKIP() << "Failed to load model: " << e.what();
    }
}

TEST_F(InferenceTest, BatchEvaluation) {
    const bool verbose = is_single_test_run();

    std::ifstream model_file(DEFAULT_MODEL_PATH);
    if (!model_file.good()) {
        GTEST_SKIP() << "Model file not found: " << DEFAULT_MODEL_PATH;
        return;
    }
    model_file.close();

    // Build a tree
    uint32_t root = create_root();
    game_->build_tree(root, 0.5f, 5, 1000);

    auto leaves = find_leaf_nodes(root);
    ASSERT_GT(leaves.size(), 0);

    try {
        ModelInference inference(DEFAULT_MODEL_PATH, 8, true);
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
        std::vector<std::pair<uint32_t, float>> results;
        inference.flush_queue([&results](uint32_t idx, float value) {
            results.emplace_back(idx, value);
        });

        EXPECT_EQ(results.size(), num_to_eval);
        EXPECT_EQ(inference.queue_size(), 0);

        if (verbose) {
            std::cout << "\n========== Batch Evaluation Test ==========" << std::endl;
            std::cout << "Evaluated " << results.size() << " nodes in batch" << std::endl;
            std::cout << "\nResults:" << std::endl;
            for (const auto& [idx, score] : results) {
                std::cout << "  Node " << idx << ": " << score << std::endl;
            }
            std::cout << "==========================================\n" << std::endl;
        }

        // All scores should be valid
        for (const auto& [idx, score] : results) {
            EXPECT_FALSE(std::isnan(score)) << "Score for node " << idx << " should not be NaN";
            EXPECT_FALSE(std::isinf(score)) << "Score for node " << idx << " should not be Inf";
        }

    } catch (const c10::Error& e) {
        GTEST_SKIP() << "Failed to load model: " << e.what();
    }
}
