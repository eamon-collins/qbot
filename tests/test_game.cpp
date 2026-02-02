#include "core/Game.h"
#include "tree/StateNode.h"
#include "search/mcts.h"
#include "util/storage.h"
#include "util/gui_client.h"
#include "util/pathfinding.h"

#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <queue>
#include <vector>
#include <random>

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

        // Try to connect to GUI for visualization
        GUIClient gui;
        GUIClient::Config gui_config;
        gui_config.connect_timeout_ms = 50;  // Short timeout - fails fast if no GUI

        if (gui.connect(gui_config)) {
            std::cout << "Connected to GUI on port " << gui_config.port << " - visualizing path..." << std::endl;
            gui.send_start("Human", "Bot");

            for (size_t i = 0; i < path.size(); ++i) {
                const StateNode& node = game_->pool()[path[i]];
                int current_player = node.is_p1_to_move() ? 0 : 1;
                float q_value = node.stats.Q();
                gui.send_gamestate(node, current_player, q_value);
            }
            std::cout << "Visualization complete." << std::endl;
        } else {
            // Fall back to printing
            for (size_t i = 0; i < path.size(); ++i) {
                std::cout << "--- Step " << i << " ---" << std::endl;
                game_->pool()[path[i]].print_node();
            }
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
    // Use larger pool to reach terminal states
    GameConfig config;
    config.pool_capacity = 2'000'000;
    game_ = std::make_unique<Game>(config);

    uint32_t root = create_root();

    // Build tree with very low branching factor (almost pure depth-first)
    // to reach terminal states quickly. Quoridor games typically need 30-80 moves.
    size_t total_created = 0;
    size_t terminal_count = 0;
    constexpr float branching_factor = 0.01f;  // Almost pure depth-first
    constexpr size_t nodes_per_round = 100'000;
    constexpr int max_rounds = 20;

    for (int round = 0; round < max_rounds; ++round) {
        size_t created = game_->build_tree(root, branching_factor, 5, nodes_per_round);
        total_created += created;

        // Count terminal nodes
        terminal_count = 0;
        std::deque<uint32_t> queue;
        queue.push_back(root);
        while (!queue.empty()) {
            uint32_t idx = queue.front();
            queue.pop_front();
            const StateNode& node = game_->pool()[idx];
            if (node.is_terminal()) terminal_count++;
            uint32_t child = node.first_child;
            while (child != NULL_NODE) {
                queue.push_back(child);
                child = game_->pool()[child].next_sibling;
            }
        }
        if (terminal_count >= 3) break;  // Have enough terminals
    }

    ASSERT_GT(total_created, 0);

    // Save to temp file
    const char* path = "/tmp/leopard_test.qbot";
    auto result = TreeStorage::save(path, game_->pool(), root);
    ASSERT_TRUE(result.has_value()) << "Failed to save: " << to_string(result.error());

    if (is_single_test_run()) {
        std::cout << "\nSaved " << game_->pool().allocated() << " nodes to " << path << std::endl;
        std::cout << "Found " << terminal_count << " terminal nodes (completed games)" << std::endl;
        std::cout << "Run: ./leopard " << path << std::endl;
    }
}

TEST_F(GameTest, PathfindingOptimizationCorrectness) {
    // ========== 1. Test check_reachability_fast vs A* (Fuzz Test) ==========
    // We compare the new fast BFS against the existing trusted Pathfinder::can_reach
    // on thousands of random board states to ensure exact parity.

    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<int> row_dist(0, 8);
    std::uniform_int_distribution<int> col_dist(0, 8);
    std::uniform_int_distribution<int> bool_dist(0, 1);

    int matches = 0;
    int mismatches = 0;

    Pathfinder& pf = get_pathfinder(); // Existing A* pathfinder

    for (int i = 0; i < 5000; ++i) {
        FenceGrid fences;

        // Randomly place 10-20 fences to create complex mazes
        int num_fences = std::uniform_int_distribution<int>(10, 20)(rng);
        for (int f = 0; f < num_fences; ++f) {
            int r = std::uniform_int_distribution<int>(0, 7)(rng);
            int c = std::uniform_int_distribution<int>(0, 7)(rng);
            bool horz = bool_dist(rng);

            if (horz) {
                if (!fences.h_fence_blocked(r, c)) fences.place_h_fence(r, c);
            } else {
                if (!fences.v_fence_blocked(r, c)) fences.place_v_fence(r, c);
            }
        }

        // Random player position
        Player p;
        p.row = row_dist(rng);
        p.col = col_dist(rng);
        p.fences = 5;

        // Test P1 goal (row 8)
        bool expected_p1 = pf.can_reach(fences, p, 8);
        bool actual_p1 = check_reachability_fast(fences, p, 8);

        if (expected_p1 != actual_p1) {
            mismatches++;
            // Print failure case for debugging
            std::cout << "Mismatch P1 (Target Row 8): Expected " << expected_p1 
                      << " Got " << actual_p1 << " at Start(" << (int)p.row << "," << (int)p.col << ")" << std::endl;
        } else {
            matches++;
        }

        // Test P2 goal (row 0)
        bool expected_p2 = pf.can_reach(fences, p, 0);
        bool actual_p2 = check_reachability_fast(fences, p, 0);

        if (expected_p2 != actual_p2) {
            mismatches++;
            std::cout << "Mismatch P2 (Target Row 0): Expected " << expected_p2 
                      << " Got " << actual_p2 << " at Start(" << (int)p.row << "," << (int)p.col << ")" << std::endl;
        } else {
            matches++;
        }
    }

    EXPECT_EQ(mismatches, 0) << "Fast reachability BFS disagreed with A* on random boards";
    EXPECT_EQ(matches, 10000) << "Should have tested 5000 boards * 2 players";


    // ========== 2. Test compute_path_blockers ==========
    // Verify that the bitmasks correctly identify which fences block a specific path.
    // Path: (0,0) -> (0,1) -> (1,1)
    // Move 1: Horizontal (0,0)->(0,1). Blocked by Vertical fences at (0,0) and invalid(-1,0).
    // Move 2: Vertical (0,1)->(1,1). Blocked by Horizontal fences at (0,1) and (0,0).

    std::vector<Coord> path = {{0,0}, {0,1}, {1,1}};
    auto [h_mask, v_mask] = compute_path_blockers(path);

    // Expected Vertical Mask (blocks horizontal moves):
    // The move (0,0)->(0,1) crosses the vertical edge at col 0.
    // This edge is blocked by V-Fence at (0,0).
    // Index = row*8 + col = 0*8 + 0 = 0.
    uint64_t expected_v = (1ULL << 0);

    // Expected Horizontal Mask (blocks vertical moves):
    // The move (0,1)->(1,1) crosses the horizontal edge at row 0.
    // This edge is blocked by H-Fence at (0,1) AND H-Fence at (0,0).
    // Index 1 = 0*8 + 1 = 1.
    // Index 2 = 0*8 + 0 = 0.
    uint64_t expected_h = (1ULL << 1) | (1ULL << 0);

    EXPECT_EQ(v_mask, expected_v) << "Vertical blocker mask incorrect for simple corner path";
    EXPECT_EQ(h_mask, expected_h) << "Horizontal blocker mask incorrect for simple corner path";

    // Case 3: Verify bitmask mapping for middle of board
    // Path: (4,4) -> (5,4). Down move crossing Row 4.
    // Blocked by H-Fences at (4,4) and (4,3).
    // (4,4) index = 4*8 + 4 = 36.
    // (4,3) index = 4*8 + 3 = 35.
    path = {{4,4}, {5,4}};
    auto [h_mid, v_mid] = compute_path_blockers(path);

    uint64_t expected_h_mid = (1ULL << 36) | (1ULL << 35);
    EXPECT_EQ(h_mid, expected_h_mid) << "Horizontal blocker mask failed for middle board move";
    EXPECT_EQ(v_mid, 0ULL) << "Should be no vertical blockers for a vertical move";
}

// TEST_F(GameTest, RandomPlayWithPathfinding) {
//     const bool verbose = is_single_test_run();
//
//     // Create initial game state
//     uint32_t root_idx = create_root();
//     StateNode* root = &game_->pool()[root_idx];
//
//     std::random_device rd;
//     std::mt19937 gen(rd());
//
//     int attempts = 0;
//     while (attempts < 4) { //break below after too many attempts
//         uint32_t current_idx = root_idx;
//         StateNode* current = root;
//
//         int move_count = 0;
//         const int max_moves = 200;  // Safety limit
//
//         if (verbose) {
//             std::cout << "\n========== Random Play Until All Fences Spent ==========" << std::endl;
//         }
//
//         // Play randomly until both players have no fences left
//         while (move_count < max_moves) {
//             // Check if both players are out of fences
//             if (current->p1.fences == 0 && current->p2.fences == 0) {
//                 if (verbose) {
//                     std::cout << "\nBoth players out of fences after " << move_count << " moves!" << std::endl;
//                 }
//                 break;
//             }
//
//             // Check if game is already over
//             if (current->is_terminal()) {
//                 if (verbose) {
//                     std::cout << "\nGame ended (terminal state) after " << move_count << " moves" << std::endl;
//                 }
//                 break;
//             }
//
//             // Generate valid moves
//             auto valid_moves = current->generate_valid_moves();
//             if (valid_moves.empty()) {
//                 if (verbose) {
//                     std::cout << "\nNo valid moves available!" << std::endl;
//                 }
//                 break;
//             }
//
//             // Pick a random move
//             std::uniform_int_distribution<> dis(0, valid_moves.size() - 1);
//             Move chosen_move = valid_moves[dis(gen)];
//
//             // Create child node with this move
//             uint32_t new_idx = game_->pool().allocate();
//             if (new_idx == NULL_NODE) {
//                 if (verbose) {
//                     std::cout << "\nFailed to allocate node" << std::endl;
//                 }
//                 break;
//             }
//
//             game_->pool()[new_idx].init_from_parent(*current, chosen_move, current_idx);
//             current_idx = new_idx;
//             current = &game_->pool()[current_idx];
//             move_count++;
//         }
//
//         //Pathfinding tests
//         Pathfinder& pf = get_pathfinder();
//         bool both_can_reach = pf.check_paths(*current);
//         attempts++;
//         std::cout << both_can_reach << " att " << attempts << std::endl;
//         if (!both_can_reach && attempts >= 4) {
//             EXPECT_TRUE(both_can_reach) << " One or both players are blocked from their goal after 4 random attempts";
//             break;
//         } else if (!both_can_reach) {
//             continue;
//         }
//
//         auto p1_path = pf.find_path(current->fences, current->p1, 8);
//         auto p2_path = pf.find_path(current->fences, current->p2, 0);
//         // Use path_length method as well
//         int p1_length = pf.path_length(current->fences, current->p1, 8);
//         int p2_length = pf.path_length(current->fences, current->p2, 0);
//
//         // Verify consistency
//         if (p1_length >= 0 && !p1_path.empty()) {
//             EXPECT_EQ(p1_length, static_cast<int>(p1_path.size() )-1)
//                 << "path_length() and find_path() should return same length for P1";
//         }
//         if (p2_length >= 0 && !p2_path.empty()) {
//             EXPECT_EQ(p2_length, static_cast<int>(p2_path.size())-1)
//                 << "path_length() and find_path() should return same length for P2";
//         }
//
//         //check early term
//         int relative_winner = early_terminate_no_fences(*current);
//         int absolute_winner = current->is_p1_to_move() ? relative_winner : -relative_winner;
//
//         if (current->is_p1_to_move()) {
//             EXPECT_EQ(absolute_winner == 1, p1_length <= p2_length) << " if p1 wins on-turn, path equal or shorter";
//             EXPECT_EQ(absolute_winner == -1, p2_length < p1_length) << " if p2 wins off-turn, path shorter";
//         } else {
//             EXPECT_EQ(absolute_winner == 1, p1_length < p2_length) << " if p1 wins off-turn, path shorter";
//             EXPECT_EQ(absolute_winner == -1, p2_length <= p1_length) << " if p2 wins on-turn, path equal or shorter";
//         }
//
//         if (verbose) {
//             std::cout << "\n========== Final Game State ==========" << std::endl;
//             current->print_node();
//             std::cout << "\n========== Pathfinding Results ==========" << std::endl;
//
//             // P1's path to goal (row 8)
//             std::cout << "P1 path to goal (row 8): " << p1_path.size() - 1 << " moves" << std::endl;
//             if (!p1_path.empty()) {
//                 std::cout << "  Path: ";
//                 for (size_t i = 0; i < p1_path.size(); ++i) {
//                     if (i > 0) std::cout << " → ";
//                     std::cout << "(" << (int)p1_path[i].col << "," << (int)p1_path[i].row << ")";
//                 }
//                 std::cout << std::endl;
//             } else {
//                 std::cout << "  BLOCKED - No path to goal!" << std::endl;
//             }
//
//             // P2's path to goal (row 0)
//             std::cout << "P2 path to goal (row 0): " << p2_path.size() - 1 << " moves" << std::endl;
//             if (!p2_path.empty()) {
//                 std::cout << "  Path: ";
//                 for (size_t i = 0; i < p2_path.size(); ++i) {
//                     if (i > 0) std::cout << " → ";
//                     std::cout << "(" << (int)p2_path[i].col << "," << (int)p2_path[i].row << ")";
//                 }
//                 std::cout << std::endl;
//             } else {
//                 std::cout << "  BLOCKED - No path to goal!" << std::endl;
//             }
//
//             //early termination checks
//             std::cout << "Terminal_value: " << relative_winner << std::endl;
//             std::cout << "Winner: " << (absolute_winner == 1 ? "P1" : "P2") << std::endl;
//         }
//
//         // Basic validation
//         EXPECT_GT(move_count, 0) << "Should have made at least one move";
//         EXPECT_LE(current->p1.fences, NUM_FENCES);
//         EXPECT_LE(current->p2.fences, NUM_FENCES);
//         break;
//     }
// }
