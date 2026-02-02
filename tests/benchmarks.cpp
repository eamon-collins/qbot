// Benchmarks - built with full optimizations, NOT registered with ctest
// Run directly: ./benchmarks

#include "core/Game.h"
#include "tree/StateNode.h"
#include "inference/inference.h"
#include "search/mcts.h"
#include "tree/node_pool.h"

#include <gtest/gtest.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <vector>

using namespace qbot;

// ============================================================================
// Game Benchmarks (always available)
// ============================================================================

class GameBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        GameConfig config;
        config.pool_capacity = 5'000'000;
        game_ = std::make_unique<Game>(config);
    }

    uint32_t create_root() {
        uint32_t root_idx = game_->pool().allocate();
        EXPECT_NE(root_idx, NULL_NODE);
        game_->pool()[root_idx].init_root(true);
        game_->set_root(root_idx);
        return root_idx;
    }

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

    std::unique_ptr<Game> game_;
};

TEST_F(GameBenchmark, BuildTree) {
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

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║              Game Tree Build Benchmark                    ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Parameters:                                              ║\n";
    std::cout << "║    branching_factor: " << std::left << std::setw(36) << branching_factor << "║\n";
    std::cout << "║    time_limit:       " << std::left << std::setw(33) << std::to_string(time_limit_sec) + " sec" << "║\n";
    std::cout << "║    node_limit:       " << std::left << std::setw(36) << node_limit << "║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Results:                                                 ║\n";
    std::cout << "║    Nodes created:    " << std::left << std::setw(36) << created << "║\n";
    std::cout << "║    Time elapsed:     " << std::left << std::setw(33) << std::to_string(duration_ms) + " ms" << "║\n";
    std::cout << "║    Throughput:       " << std::left << std::setw(30) << std::to_string(static_cast<int>(nodes_per_sec)) + " nodes/sec" << "║\n";
    std::ostringstream per_node_str;
    per_node_str << std::fixed << std::setprecision(2) << us_per_node << " us/node";
    std::cout << "║    Per-node time:    " << std::left << std::setw(36) << per_node_str.str() << "║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    EXPECT_GT(created, 0) << "Should create nodes";

    size_t counted = count_tree_nodes(root);
    EXPECT_EQ(counted, created + 1) << "Tree count should match";
}


// ============================================================================
// Inference Benchmarks
// ============================================================================


namespace {

const char* DEFAULT_MODEL_PATH = "/home/eamon/repos/qbot/model/test_model.pt";
const char* TEST_MODEL_PATH = "/tmp/test_model.pt";

bool create_test_model() {
    // Create model with new unified input format (6 channels, current-player-perspective)
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

bool is_model_compatible(const std::string& path) {
    try {
        ModelInference inference(path, 1, false);
        if (!inference.is_ready()) return false;

        StateNode dummy;
        dummy.init_root(true);
        inference.evaluate_node(&dummy);
        return true;
    } catch (...) {
        return false;
    }
}

std::string get_model_path() {
    std::ifstream default_model(DEFAULT_MODEL_PATH);
    if (default_model.good() ) { //&& is_model_compatible(DEFAULT_MODEL_PATH)) {
        return DEFAULT_MODEL_PATH;
    }
    if (create_test_model()) {
        return TEST_MODEL_PATH;
    }
    return "";
}

}  // namespace

class InferenceBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        GameConfig config;
        config.pool_capacity = 10'000;
        game_ = std::make_unique<Game>(config);
    }

    std::unique_ptr<Game> game_;
};

TEST_F(InferenceBenchmark, BatchSizes) {
    std::string model_path = get_model_path();
    if (model_path.empty()) {
        GTEST_SKIP() << "No model available (requires Python with PyTorch)";
        return;
    }

    try {
        ModelInference inference(model_path, 512, true);
        ASSERT_TRUE(inference.is_ready());

        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                    Inference Batch Size Benchmark                     ║\n";
        std::cout << "╠═══════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Model: " << std::left << std::setw(62) << model_path.substr(model_path.find_last_of('/') + 1) << "║\n";
        std::cout << "║  Device: " << std::left << std::setw(61) << (torch::cuda::is_available() ? "CUDA" : "CPU") << "║\n";
        std::cout << "╠═════════════╦═══════════════╦═══════════════╦═════════════════════════╣\n";
        std::cout << "║ Batch Size  ║   Total (ms)  ║  Per Node(us) ║     Throughput (n/s)    ║\n";
        std::cout << "╠═════════════╬═══════════════╬═══════════════╬═════════════════════════╣\n";

        std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

        constexpr int WARMUP_RUNS = 3;
        constexpr int MEASURE_RUNS = 10;

        for (int batch_size : batch_sizes) {
            std::vector<StateNode> nodes(batch_size);
            std::vector<const StateNode*> node_ptrs;
            node_ptrs.reserve(batch_size);

            std::mt19937 rng(42);
            for (int i = 0; i < batch_size; ++i) {
                nodes[i].init_root(true);
                nodes[i].p1.row = rng() % 9;
                nodes[i].p1.col = rng() % 9;
                nodes[i].p2.row = rng() % 9;
                nodes[i].p2.col = rng() % 9;
                nodes[i].p1.fences = rng() % 11;
                nodes[i].p2.fences = rng() % 11;
                node_ptrs.push_back(&nodes[i]);
            }

            for (int w = 0; w < WARMUP_RUNS; ++w) {
                inference.evaluate_batch(node_ptrs);
            }

            double total_time_us = 0.0;
            for (int r = 0; r < MEASURE_RUNS; ++r) {
                auto start = std::chrono::high_resolution_clock::now();
                auto results = inference.evaluate_batch(node_ptrs);
                auto end = std::chrono::high_resolution_clock::now();

                auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                total_time_us += static_cast<double>(duration_us);

                ASSERT_EQ(results.size(), static_cast<size_t>(batch_size));
            }

            double avg_time_us = total_time_us / MEASURE_RUNS;
            double avg_time_ms = avg_time_us / 1000.0;
            double per_node_us = avg_time_us / batch_size;
            double throughput = 1'000'000.0 / per_node_us;

            std::cout << "║ " << std::right << std::setw(10) << batch_size << "  ║ "
                      << std::right << std::setw(12) << std::fixed << std::setprecision(3) << avg_time_ms << "  ║ "
                      << std::right << std::setw(12) << std::fixed << std::setprecision(2) << per_node_us << "  ║ "
                      << std::right << std::setw(22) << std::fixed << std::setprecision(0) << throughput << "   ║\n";
        }

        std::cout << "╚═════════════╩═══════════════╩═══════════════╩═════════════════════════╝\n";
        std::cout << "\n";

    } catch (const c10::Error& e) {
        GTEST_SKIP() << "Failed to load model: " << e.what();
    }
}

TEST_F(InferenceBenchmark, SelfPlayThroughput) {
    std::string model_path = get_model_path();
    if (model_path.empty()) {
        GTEST_SKIP() << "No model available";
        return;
    }

    // --- CONFIGURATION ---
    const int num_workers = 1;        // Keep to 1 to analyze serialization first
    const int batch_size = 256;       // Match your InferenceServer config
    // We want enough games to fill the batch, but since we are 
    // simulating "infinite" search, we just need enough to keep the pipe full.
    const int games_per_worker = 100;

    // High simulations per move reduces the overhead of "Game Setup/Teardown"
    // and focuses the benchmark on the MCTS/Inference loop.
    const int simulations_per_move = 800;
    const int num_moves_to_measure = 10;

    // Total MCTS iterations = workers * games_per_worker * moves * sims
    const size_t total_expected_iterations =
        static_cast<size_t>(num_workers) * games_per_worker * num_moves_to_measure * simulations_per_move;

    // --- SETUP ---
    InferenceServerConfig server_config;
    server_config.batch_size = batch_size;
    server_config.max_wait_ms = 1.0;

    InferenceServer server(model_path, server_config);
    server.start();

    SelfPlayConfig sp_config;
    sp_config.simulations_per_move = simulations_per_move;
    // CRITICAL: Limit game to 1 move. This effectively resets the board 
    // after every move, ensuring we benchmark the SAME state complexity repeatedly.
    sp_config.max_moves_per_game = num_moves_to_measure; 
    sp_config.max_draw_reward = 0.0f;

    SelfPlayEngine engine(sp_config);

    // Use a large pool to avoid allocations affecting the bench
    NodePool::Config config;
    config.initial_capacity = 20'000'000;
    NodePool pool(config);
    StateNode::set_pool(&pool);

    // Create a dummy root
    uint32_t root_idx = pool.allocate();
    pool[root_idx].init_root(true);

    MultiGameStats stats;
    TreeBoundsConfig bounds;
    TrainingSampleCollector collector; // Dummy collector

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║            Self-Play Throughput Benchmark                 ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Workers:       " << std::left << std::setw(33) << num_workers << " ║\n";
    std::cout << "║  Games/Worker:  " << std::left << std::setw(33) << games_per_worker << " ║\n";
    std::cout << "║  Sims/Move:     " << std::left << std::setw(33) << simulations_per_move << " ║\n";
    std::cout << "║  Total Target:  " << std::left << std::setw(33) << total_expected_iterations << " ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n";

    // --- BENCHMARK RUN ---
    auto start = std::chrono::high_resolution_clock::now();

    int total_games_to_play = num_workers * games_per_worker;

    engine.run_multi_game(
        pool, root_idx, server, 
        total_games_to_play, 
        num_workers, 
        games_per_worker, 
        stats, 
        bounds, 
        &collector, 
        "" // No file output
    );

    auto end = std::chrono::high_resolution_clock::now();
    server.stop();

    // --- RESULTS ---
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double seconds = duration_ms / 1000.0;

    // We can calculate actual iterations from stats if needed, or estimate
    size_t actual_moves = stats.total_moves.load();
    size_t actual_iterations = actual_moves * simulations_per_move;

    double ips = actual_iterations / seconds;

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Results:                                                 ║\n";
    std::cout << "║    Time:        " << std::left << std::setw(33) << std::to_string(seconds) + " s" << " ║\n";
    std::cout << "║    Moves:       " << std::left << std::setw(33) << actual_moves << " ║\n";
    std::cout << "║    Iterations:  " << std::left << std::setw(33) << actual_iterations << " ║\n";
    std::cout << "║    Speed:       " << std::left << std::setw(30) << std::to_string((int)ips) + " iter/sec" << " ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n";
}

