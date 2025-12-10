#include "core/Game.h"
#include "tree/StateNode.h"

#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <queue>

using namespace qbot;

class GameTest : public ::testing::Test {
protected:
    void SetUp() override {
        GameConfig config;
        config.pool_capacity = 200'000;
        game_ = std::make_unique<Game>(config);
    }

    // Helper to create and initialize a root node
    uint32_t create_root() {
        uint32_t root_idx = game_->pool().allocate();
        EXPECT_NE(root_idx, NULL_NODE);
        game_->pool()[root_idx].init_root(true);
        game_->set_root(root_idx);
        return root_idx;
    }

    // Helper to count all nodes in tree via BFS
    size_t count_tree_nodes(uint32_t root_idx) {
        if (root_idx == NULL_NODE) return 0;

        size_t count = 0;
        std::queue<uint32_t> queue;
        queue.push(root_idx);

        while (!queue.empty()) {
            uint32_t idx = queue.front();
            queue.pop();
            count++;

            uint32_t child = game_->pool()[idx].first_child;
            while (child != NULL_NODE) {
                queue.push(child);
                child = game_->pool()[child].next_sibling;
            }
        }
        return count;
    }

    // Helper to find max depth of tree
    size_t max_tree_depth(uint32_t root_idx) {
        if (root_idx == NULL_NODE) return 0;

        std::queue<std::pair<uint32_t, size_t>> queue;
        queue.push({root_idx, 1});
        size_t max_depth = 0;

        while (!queue.empty()) {
            auto [idx, depth] = queue.front();
            queue.pop();
            max_depth = std::max(max_depth, depth);

            uint32_t child = game_->pool()[idx].first_child;
            while (child != NULL_NODE) {
                queue.push({child, depth + 1});
                child = game_->pool()[child].next_sibling;
            }
        }
        return max_depth;
    }

    std::unique_ptr<Game> game_;
};

TEST_F(GameTest, BuildTreeCreatesNodes) {
    uint32_t root = create_root();

    size_t created = game_->build_tree(root, 0.5f, 10, 100);

    EXPECT_GT(created, 0) << "Should create at least some nodes";
    // Node limit is approximate - may overshoot by one expansion batch
    EXPECT_LE(created, 250) << "Should roughly respect node limit";

    // Tree should have root + created nodes
    size_t total = count_tree_nodes(root);
    EXPECT_EQ(total, created + 1) << "Tree node count should match created + root";
}

TEST_F(GameTest, BuildTreeRespectsNodeLimit) {
    uint32_t root = create_root();

    size_t limit = 500;
    size_t created = game_->build_tree(root, 0.5f, 60, limit);

    // Allow overshoot since generate_valid_children creates a batch at once
    EXPECT_LE(created, limit + 150) << "Should roughly respect node limit";
    EXPECT_GE(created, limit - 150) << "Should get close to node limit";
}

TEST_F(GameTest, BuildTreeRespectsTimeLimit) {
    uint32_t root = create_root();

    auto start = std::chrono::steady_clock::now();
    game_->build_tree(root, 0.5f, 1, 1'000'000);  // 1 second limit, huge node limit
    auto elapsed = std::chrono::steady_clock::now() - start;

    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    EXPECT_LE(seconds, 3) << "Should stop within reasonable time after limit";
}

TEST_F(GameTest, BuildTreeWithNullRoot) {
    size_t created = game_->build_tree(NULL_NODE, 0.5f, 10, 100);
    EXPECT_EQ(created, 0) << "Should return 0 for null root";
}

TEST_F(GameTest, BuildTreeHighBranchingFactor) {
    // High branching factor = more breadth-first, shallower tree
    uint32_t root = create_root();

    size_t created = game_->build_tree(root, 0.9f, 10, 500);
    size_t depth = max_tree_depth(root);

    EXPECT_GT(created, 0);
    // With high branching factor, tree should be relatively shallow
    // (though randomness means we can't be too strict)
    std::cout << "High branching (0.9): " << created << " nodes, depth " << depth << std::endl;
}

TEST_F(GameTest, BuildTreeLowBranchingFactor) {
    // Low branching factor = more depth-first, deeper tree
    uint32_t root = create_root();

    size_t created = game_->build_tree(root, 0.1f, 10, 500);
    size_t depth = max_tree_depth(root);

    EXPECT_GT(created, 0);
    std::cout << "Low branching (0.1): " << created << " nodes, depth " << depth << std::endl;
}

TEST_F(GameTest, BuildTreeNodesAreValid) {
    uint32_t root = create_root();

    game_->build_tree(root, 0.5f, 10, 200);

    // BFS through tree and verify all nodes have valid state
    std::queue<uint32_t> queue;
    queue.push(root);

    while (!queue.empty()) {
        uint32_t idx = queue.front();
        queue.pop();

        const StateNode& node = game_->pool()[idx];

        // Check player positions are valid
        EXPECT_LT(node.p1.row, BOARD_SIZE);
        EXPECT_LT(node.p1.col, BOARD_SIZE);
        EXPECT_LT(node.p2.row, BOARD_SIZE);
        EXPECT_LT(node.p2.col, BOARD_SIZE);

        // Check fence counts are valid
        EXPECT_LE(node.p1.fences, NUM_FENCES);
        EXPECT_LE(node.p2.fences, NUM_FENCES);

        // Traverse children
        uint32_t child = node.first_child;
        while (child != NULL_NODE) {
            // Verify parent pointer
            EXPECT_EQ(game_->pool()[child].parent, idx);
            queue.push(child);
            child = game_->pool()[child].next_sibling;
        }
    }
}

TEST_F(GameTest, BuildTreeTerminalNodesNotExpanded) {
    uint32_t root = create_root();

    game_->build_tree(root, 0.5f, 10, 1000);

    // Check that terminal nodes have no children
    std::queue<uint32_t> queue;
    queue.push(root);

    int terminal_count = 0;
    while (!queue.empty()) {
        uint32_t idx = queue.front();
        queue.pop();

        const StateNode& node = game_->pool()[idx];

        if (node.is_terminal()) {
            terminal_count++;
            EXPECT_EQ(node.first_child, NULL_NODE)
                << "Terminal node should have no children";
        }

        uint32_t child = node.first_child;
        while (child != NULL_NODE) {
            queue.push(child);
            child = game_->pool()[child].next_sibling;
        }
    }

    std::cout << "Found " << terminal_count << " terminal nodes" << std::endl;
}

TEST_F(GameTest, BenchmarkBuildTree) {
    uint32_t root = create_root();

    constexpr float branching_factor = 0.5f;
    constexpr std::time_t time_limit_sec = 3;
    constexpr size_t node_limit = 5'000'000;

    auto start = std::chrono::high_resolution_clock::now();
    size_t created = game_->build_tree(root, branching_factor, time_limit_sec, node_limit);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double nodes_per_sec = (created * 1'000'000.0) / duration_us;
    double us_per_node = static_cast<double>(duration_us) / created;

    std::cout << "\n========== BENCHMARK: build_tree ==========" << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  branching_factor: " << branching_factor << std::endl;
    std::cout << "  time_limit:       " << time_limit_sec << " sec" << std::endl;
    std::cout << "  node_limit:       " << node_limit << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "  Nodes created:    " << created << std::endl;
    std::cout << "  Time elapsed:     " << duration_ms << " ms" << std::endl;
    std::cout << "  Throughput:       " << static_cast<int>(nodes_per_sec) << " nodes/sec" << std::endl;
    std::cout << "  Per-node time:    " << us_per_node << " µs/node" << std::endl;
    std::cout << "============================================\n" << std::endl;

    EXPECT_GT(created, 0) << "Should create nodes";

    // Verify tree integrity
    size_t counted = count_tree_nodes(root);
    EXPECT_EQ(counted, created + 1) << "Tree count should match";
}
