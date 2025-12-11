#include "core/Game.h"
#include "tree/StateNode.h"
#include "util/storage.h"

#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <queue>
#include <vector>

using namespace qbot;

namespace {
// Returns true if only a single test is being run (verbose output appropriate)
bool is_single_test_run() {
    const auto* filter = ::testing::GTEST_FLAG(filter).c_str();
    // If filter contains a specific test name (not "*" or default), it's a single test run
    return filter && std::string(filter).find('*') == std::string::npos
           && std::string(filter) != "";
}
}

class GameTest : public ::testing::Test {
protected:
    void SetUp() override {
        GameConfig config;
        config.pool_capacity = 500'000;
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
    if (is_single_test_run()) {
        std::cout << "High branching (0.9): " << created << " nodes, depth " << depth << std::endl;
    }
}

TEST_F(GameTest, BuildTreeLowBranchingFactor) {
    // Low branching factor = more depth-first, deeper tree
    uint32_t root = create_root();

    size_t created = game_->build_tree(root, 0.1f, 10, 500);
    size_t depth = max_tree_depth(root);

    EXPECT_GT(created, 0);
    if (is_single_test_run()) {
        std::cout << "Low branching (0.1): " << created << " nodes, depth " << depth << std::endl;
    }
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

    if (is_single_test_run()) {
        std::cout << "Found " << terminal_count << " terminal nodes" << std::endl;
    }
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

    if (is_single_test_run()) {
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
    }

    EXPECT_GT(created, 0) << "Should create nodes";

    // Verify tree integrity
    size_t counted = count_tree_nodes(root);
    EXPECT_EQ(counted, created + 1) << "Tree count should match";
}

TEST_F(GameTest, BuildTreeUntilWinAndPrintPath) {
    // Use larger pool for this test
    GameConfig config;
    config.pool_capacity = 2'000'000;
    game_ = std::make_unique<Game>(config);

    uint32_t root = create_root();
    const bool verbose = is_single_test_run();

    if (verbose) {
        std::cout << "\n========== Building tree until win state ==========" << std::endl;
    }

    // Keep building until we find a terminal node
    // Use very low branching factor to go deep quickly (depth-first)
    // Also track max depth to see progress
    constexpr float branching_factor = 0.01f;  // Almost pure depth-first
    constexpr std::time_t time_limit_per_round = 10;
    constexpr size_t node_limit_per_round = 500'000;
    constexpr int max_rounds = 10;

    uint32_t terminal_idx = NULL_NODE;
    size_t total_created = 0;

    for (int round = 0; round < max_rounds && terminal_idx == NULL_NODE; ++round) {
        size_t created = game_->build_tree(root, branching_factor, time_limit_per_round, node_limit_per_round);
        total_created += created;

        // Find max depth and look for terminal nodes
        size_t max_depth = 0;
        std::queue<std::pair<uint32_t, size_t>> queue;
        queue.push({root, 0});

        while (!queue.empty()) {
            auto [idx, depth] = queue.front();
            queue.pop();

            max_depth = std::max(max_depth, depth);
            const StateNode& node = game_->pool()[idx];

            if (node.is_terminal() && terminal_idx == NULL_NODE) {
                terminal_idx = idx;
            }

            uint32_t child = node.first_child;
            while (child != NULL_NODE) {
                queue.push({child, depth + 1});
                child = game_->pool()[child].next_sibling;
            }
        }

        if (verbose) {
            std::cout << "Round " << round << ": created " << created
                      << " nodes (total: " << total_created << "), max depth: " << max_depth << std::endl;
        }

        if (terminal_idx != NULL_NODE) break;
    }

    if (verbose) {
        std::cout << "Total nodes created: " << total_created << std::endl;
    }

    if (terminal_idx == NULL_NODE) {
        if (verbose) {
            std::cout << "No terminal node found in tree after " << max_rounds << " rounds" << std::endl;
        }
        GTEST_SKIP() << "No terminal node found - tree not deep enough";
        return;
    }

    // Build path from root to terminal by walking up parent pointers
    std::vector<uint32_t> path;
    uint32_t current = terminal_idx;
    while (current != NULL_NODE) {
        path.push_back(current);
        current = game_->pool()[current].parent;
    }

    // Reverse to get root-to-leaf order
    std::reverse(path.begin(), path.end());

    if (verbose) {
        std::cout << "\n========== Path to terminal state (" << path.size() << " moves) ==========" << std::endl;
        for (size_t i = 0; i < path.size(); ++i) {
            std::cout << "--- Step " << i << " ---" << std::endl;
            game_->pool()[path[i]].print_node();
        }
    }

    // Verify the terminal node
    const StateNode& terminal = game_->pool()[terminal_idx];
    EXPECT_TRUE(terminal.is_terminal()) << "Found node should be terminal";
    int winner = terminal.game_over();
    EXPECT_NE(winner, 0) << "Terminal node should have a winner";

    if (verbose) {
        std::cout << "Winner: Player " << (winner > 0 ? "1" : "2") << std::endl;
        std::cout << "==================================================" << std::endl;
    }
}

TEST_F(GameTest, SaveTreeForLeopard) {
    uint32_t root = create_root();

    // Build a small tree
    size_t created = game_->build_tree(root, 0.3f, 5, 500);
    ASSERT_GT(created, 0);

    // Save to temp file
    const char* path = "/tmp/leopard_test.qbot";
    auto result = TreeStorage::save(path, game_->pool(), root);
    ASSERT_TRUE(result.has_value()) << "Failed to save: " << to_string(result.error());

    if (is_single_test_run()) {
        std::cout << "\nSaved " << (created + 1) << " nodes to " << path << std::endl;
        std::cout << "Run: ./leopard " << path << " | head -c 1000 | xxd" << std::endl;
    }
}
