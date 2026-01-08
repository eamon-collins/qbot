#include "inference/inference.h"
#include "core/Game.h"
#include "tree/StateNode.h"

#include <gtest/gtest.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace qbot;

namespace {

const char* DEFAULT_MODEL_PATH = "/home/eamon/repos/qbot/model/tree.pt";
const char* TEST_MODEL_PATH = "/tmp/test_model.pt";

bool create_test_model() {
    const char* script = R"(
import sys
sys.path.insert(0, '/home/eamon/repos/qbot/train')
import torch
from resnet import QuoridorValueNet

model = QuoridorValueNet()
model.eval()

example_pawn = torch.zeros(1, 2, 9, 9)
example_wall = torch.zeros(1, 2, 8, 8)
example_meta = torch.zeros(1, 3)

traced = torch.jit.trace(model, (example_pawn, example_wall, example_meta))
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
    if (default_model.good() && is_model_compatible(DEFAULT_MODEL_PATH)) {
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

// Benchmark test for measuring inference latency at various batch sizes
TEST_F(InferenceBenchmark, BatchSizes) {
    std::string model_path = get_model_path();
    if (model_path.empty()) {
        GTEST_SKIP() << "No model available (requires Python with PyTorch)";
        return;
    }

    try {
        ModelInference inference(model_path, 512, true);  // Max batch size
        ASSERT_TRUE(inference.is_ready());

        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                    Inference Batch Size Benchmark                     ║\n";
        std::cout << "╠═══════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Model: " << std::left << std::setw(62) << model_path.substr(model_path.find_last_of('/') + 1) << "║\n";
        std::cout << "║  Device: " << std::left << std::setw(61) << (torch::cuda::is_available() ? "CUDA" : "CPU") << "║\n";
        std::cout << "╠═════════════╦═══════════════╦═══════════════╦═════════════════════════╣\n";
        std::cout << "║ Batch Size  ║   Total (ms)  ║  Per Node(µs) ║     Throughput (n/s)    ║\n";
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

// Extended benchmark with submission latency measurement (queue -> result)
TEST_F(InferenceBenchmark, QueueLatency) {
    std::string model_path = get_model_path();
    if (model_path.empty()) {
        GTEST_SKIP() << "No model available (requires Python with PyTorch)";
        return;
    }

    try {
        std::vector<int> internal_batch_sizes = {8, 16, 32, 64, 128};
        std::vector<int> submission_counts = {1, 10, 50, 100, 200};

        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                      Queue Submission Latency Benchmark                      ║\n";
        std::cout << "║   Measures time from queue_for_evaluation() to receiving callback result     ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";

        for (int internal_batch : internal_batch_sizes) {
            ModelInference inference(model_path, internal_batch, true);
            ASSERT_TRUE(inference.is_ready());

            std::cout << "║  Internal batch size: " << std::left << std::setw(55) << internal_batch << "║\n";
            std::cout << "╠════════════════╦════════════════╦════════════════╦════════════════════════╣\n";
            std::cout << "║  Nodes Queued  ║  Total (ms)    ║  Per Node (µs) ║  Throughput (n/s)      ║\n";
            std::cout << "╠════════════════╬════════════════╬════════════════╬════════════════════════╣\n";

            for (int num_nodes : submission_counts) {
                std::vector<StateNode> nodes(num_nodes);
                std::mt19937 rng(123);
                for (int i = 0; i < num_nodes; ++i) {
                    nodes[i].init_root(true);
                    nodes[i].p1.row = rng() % 9;
                    nodes[i].p1.col = rng() % 9;
                    nodes[i].p2.row = rng() % 9;
                    nodes[i].p2.col = rng() % 9;
                }

                constexpr int RUNS = 5;
                double total_time_us = 0.0;
                int total_results = 0;

                for (int r = 0; r < RUNS; ++r) {
                    auto start = std::chrono::high_resolution_clock::now();

                    for (int i = 0; i < num_nodes; ++i) {
                        inference.queue_for_evaluation(&nodes[i], static_cast<uint32_t>(i));
                    }

                    std::vector<float> results;
                    results.reserve(num_nodes);
                    inference.flush_queue([&results](uint32_t /*idx*/, float value) {
                        results.push_back(value);
                    });

                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                    total_time_us += static_cast<double>(duration_us);
                    total_results += static_cast<int>(results.size());
                }

                ASSERT_EQ(total_results, num_nodes * RUNS);

                double avg_time_us = total_time_us / RUNS;
                double avg_time_ms = avg_time_us / 1000.0;
                double per_node_us = avg_time_us / num_nodes;
                double throughput = 1'000'000.0 / per_node_us;

                std::cout << "║ " << std::right << std::setw(13) << num_nodes << "  ║ "
                          << std::right << std::setw(13) << std::fixed << std::setprecision(3) << avg_time_ms << "  ║ "
                          << std::right << std::setw(13) << std::fixed << std::setprecision(2) << per_node_us << "  ║ "
                          << std::right << std::setw(21) << std::fixed << std::setprecision(0) << throughput << "   ║\n";
            }

            std::cout << "╠════════════════╩════════════════╩════════════════╩════════════════════════╣\n";
        }

        std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";

    } catch (const c10::Error& e) {
        GTEST_SKIP() << "Failed to load model: " << e.what();
    }
}
