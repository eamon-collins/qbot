#pragma once

/// Async Inference Server for Multi-Threaded Self-Play
///
/// Provides a thread-safe interface for GPU inference:
/// - Worker threads submit evaluation requests
/// - Dedicated inference thread batches requests and runs GPU inference
/// - Results returned via futures for async/await pattern
///
/// This amortizes GPU kernel launch overhead across many requests and
/// allows CPU threads to do useful work while waiting for results.

#include "inference.h"
#include "../tree/StateNode.h"
#include "../util/timer.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace qbot {

/// Single evaluation request (value only, for backwards compatibility)
struct EvalRequest {
    const StateNode* node;
    std::promise<float> promise;
};

/// Single evaluation request with full result (policy + value)
struct FullEvalRequest {
    const StateNode* node;
    std::promise<EvalResult> promise;
};

/// Batch evaluation request (for computing priors on children)
struct BatchEvalRequest {
    std::vector<const StateNode*> nodes;
    std::promise<std::vector<float>> promise;
};

/// Configuration for inference server
struct InferenceServerConfig {
    int batch_size = 128;           // Target batch size before processing
    double max_wait_ms = 1.0;            // Max time to wait for batch to fill
    bool enable_batching = true;    // If false, process requests immediately
};

/// Thread-safe inference server
///
/// Usage:
///   InferenceServer server(model_path);
///   server.start();
///
///   // From worker threads:
///   auto future = server.submit(node);
///   float value = future.get();  // Blocks until result ready
///
///   server.stop();
class InferenceServer {
public:
    /// Construct server with model path
    InferenceServer(const std::string& model_path,
                    InferenceServerConfig config = InferenceServerConfig{});

    ~InferenceServer();

    // Non-copyable, non-movable
    InferenceServer(const InferenceServer&) = delete;
    InferenceServer& operator=(const InferenceServer&) = delete;

    /// Start the inference thread
    void start();

    /// Stop the inference thread (processes remaining requests first)
    void stop();

    /// Check if server is running
    [[nodiscard]] bool is_running() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    /// Submit a single node for evaluation (value only)
    /// Returns a future that will contain the value estimate
    [[nodiscard]] std::future<float> submit(const StateNode* node);

    /// Submit a single node for full evaluation (policy + value)
    /// Returns a future that will contain the complete EvalResult
    [[nodiscard]] std::future<EvalResult> submit_full(const StateNode* node);

    /// Submit a batch of nodes for evaluation (e.g., all children for priors)
    /// Returns a future that will contain all value estimates
    [[nodiscard]] std::future<std::vector<float>> submit_batch(
        std::vector<const StateNode*> nodes);

    /// Get statistics
    [[nodiscard]] size_t total_requests() const noexcept {
        return total_requests_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] size_t total_batches() const noexcept {
        return total_batches_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] size_t current_queue_size() const noexcept {
        std::lock_guard lock(queue_mutex_);
        return single_queue_.size() + batch_queue_.size();
    }

private:
    /// Main loop for inference thread
    void inference_loop();

    /// Process all pending requests
    void process_pending();

    ModelInference model_;
    InferenceServerConfig config_;

    // Inference thread
    std::thread inference_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};

    // Request queues (protected by queue_mutex_)
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<EvalRequest> single_queue_;
    std::deque<FullEvalRequest> full_eval_queue_;  // For policy + value requests
    std::deque<BatchEvalRequest> batch_queue_;

    // Statistics
    std::atomic<size_t> total_requests_{0};
    std::atomic<size_t> total_batches_{0};
};

} // namespace qbot
