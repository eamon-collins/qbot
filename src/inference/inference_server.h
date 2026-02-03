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

/// Single evaluation request with full result (policy + value)
struct EvalRequest {
    const StateNode* node;
    std::promise<EvalResult> promise;
};

/// Configuration for inference server
struct InferenceServerConfig {
    int batch_size = 128;           // Target batch size before processing
    double max_wait_ms = 1.0;            // Max time to wait for batch to fill
    bool enable_batching = true;    // If false, process requests immediately
    bool verbose = true;    // If true, talk about the inferences we're doing
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

    void flush();

    /// Check if server is running
    [[nodiscard]] bool is_running() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    /// Submit a single node for full evaluation (policy + value)
    /// Returns a future that will contain the complete EvalResult
    [[nodiscard]] std::future<EvalResult> submit(const StateNode* node);

    [[nodiscard]] std::vector<std::future<EvalResult>> submit_batch(
        const std::vector<const StateNode*>& nodes);

    /// Get statistics
    [[nodiscard]] size_t total_requests() const noexcept {
        return total_requests_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] size_t total_batches() const noexcept {
        return total_batches_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] size_t current_queue_size() const noexcept {
        std::lock_guard lock(queue_mutex_);
        return eval_queue_.size();
    }
    const SelfPlayTimers& get_inference_timers() const {
        return server_timers_;
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
    std::deque<EvalRequest> eval_queue_;  // For policy + value requests

    // Statistics
    SelfPlayTimers server_timers_;
    void print_stats();
    std::atomic<size_t> total_requests_{0}; //actually counts total nodes requested
    std::atomic<size_t> batch_requests_{0}; //counts each submit_batch as 1
    std::atomic<size_t> single_requests_{0};//counts each submit as 1
    std::atomic<size_t> total_batches_{0}; //counts actual gpu submissions
    std::atomic<size_t> total_batch_size_sum_{0};  // Sum of all batch sizes for averaging
    std::atomic<size_t> total_flushes_{0};
    std::atomic<size_t> total_batch_triggers_{0};
    std::atomic<size_t> total_time_triggers_{0};
    std::chrono::steady_clock::time_point start_time_;

public:
    /// Get average batch size
    [[nodiscard]] double avg_batch_size() const noexcept {
        size_t batches = total_batches_.load(std::memory_order_relaxed);
        if (batches == 0) return 0.0;
        size_t sum = total_batch_size_sum_.load(std::memory_order_relaxed);
        return static_cast<double>(sum) / batches;
    }
};

} // namespace qbot
