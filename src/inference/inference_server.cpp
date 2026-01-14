#include "inference_server.h"

#include <chrono>
#include <iomanip>
#include <iostream>

namespace qbot {

InferenceServer::InferenceServer(const std::string& model_path,
                                 InferenceServerConfig config)
    : model_(model_path, config.batch_size, true)
    , config_(config)
{}

InferenceServer::~InferenceServer() {
    stop();
}

void InferenceServer::start() {
    if (running_.exchange(true, std::memory_order_acq_rel)) {
        return;  // Already running
    }

    stop_requested_.store(false, std::memory_order_release);
    inference_thread_ = std::thread([this] { inference_loop(); });

    std::cout << "[InferenceServer] Started with batch_size=" << config_.batch_size
              << ", max_wait=" << config_.max_wait_ms << "ms\n";
}

void InferenceServer::stop() {
    if (!running_.load(std::memory_order_acquire)) {
        return;  // Not running
    }

    // Signal stop
    stop_requested_.store(true, std::memory_order_release);

    // Wake up inference thread
    queue_cv_.notify_all();

    // Wait for thread to finish
    if (inference_thread_.joinable()) {
        inference_thread_.join();
    }

    running_.store(false, std::memory_order_release);

    size_t requests = total_requests_.load();
    size_t batches = total_batches_.load();
    double avg_batch = batches > 0 ? static_cast<double>(requests) / batches : 0.0;
    std::cout << "[InferenceServer] Stopped. Total requests: " << requests
              << ", batches: " << batches
              << ", avg batch size: " << std::fixed << std::setprecision(1) << avg_batch << "\n";
}

std::future<float> InferenceServer::submit(const StateNode* node) {
    std::promise<float> promise;
    auto future = promise.get_future();

    {
        std::lock_guard lock(queue_mutex_);
        single_queue_.push_back({node, std::move(promise)});
    }

    total_requests_.fetch_add(1, std::memory_order_relaxed);
    queue_cv_.notify_one();

    return future;
}

std::future<EvalResult> InferenceServer::submit_full(const StateNode* node) {
    std::promise<EvalResult> promise;
    auto future = promise.get_future();

    {
        std::lock_guard lock(queue_mutex_);
        full_eval_queue_.push_back({node, std::move(promise)});
    }

    total_requests_.fetch_add(1, std::memory_order_relaxed);
    queue_cv_.notify_one();

    return future;
}

std::future<std::vector<float>> InferenceServer::submit_batch(
    std::vector<const StateNode*> nodes)
{
    std::promise<std::vector<float>> promise;
    auto future = promise.get_future();

    size_t count = nodes.size();

    {
        std::lock_guard lock(queue_mutex_);
        batch_queue_.push_back({std::move(nodes), std::move(promise)});
    }

    total_requests_.fetch_add(count, std::memory_order_relaxed);
    queue_cv_.notify_one();

    return future;
}

void InferenceServer::inference_loop() {
    auto& timers = get_timers();

    while (!stop_requested_.load(std::memory_order_acquire)) {
        // Wait for requests or timeout
        {
            std::unique_lock lock(queue_mutex_);

            auto deadline = std::chrono::steady_clock::now() +
                           std::chrono::microseconds(static_cast<long long>(config_.max_wait_ms * 1000));

            queue_cv_.wait_until(lock, deadline, [this] {
                return !single_queue_.empty() ||
                       !full_eval_queue_.empty() ||
                       !batch_queue_.empty() ||
                       stop_requested_.load(std::memory_order_acquire);
            });
        }

        // Process pending requests
        process_pending();
    }

    // Process any remaining requests before exiting
    process_pending();
}

void InferenceServer::process_pending() {
    auto& timers = get_timers();

    // Collect all pending requests
    std::vector<EvalRequest> singles;
    std::vector<FullEvalRequest> full_evals;
    std::vector<BatchEvalRequest> batches;

    {
        std::lock_guard lock(queue_mutex_);

        // Take all single requests (up to batch size)
        while (!single_queue_.empty() &&
               singles.size() < static_cast<size_t>(config_.batch_size)) {
            singles.push_back(std::move(single_queue_.front()));
            single_queue_.pop_front();
        }

        // Take all full eval requests (up to batch size)
        while (!full_eval_queue_.empty() &&
               full_evals.size() < static_cast<size_t>(config_.batch_size)) {
            full_evals.push_back(std::move(full_eval_queue_.front()));
            full_eval_queue_.pop_front();
        }

        // Take all batch requests
        while (!batch_queue_.empty()) {
            batches.push_back(std::move(batch_queue_.front()));
            batch_queue_.pop_front();
        }
    }

    // Process full eval requests first (they need complete EvalResult with policy)
    if (!full_evals.empty()) {
        std::vector<const StateNode*> full_nodes;
        full_nodes.reserve(full_evals.size());
        for (const auto& req : full_evals) {
            full_nodes.push_back(req.node);
        }

        std::vector<EvalResult> full_results;
        {
            ScopedTimer t(timers.nn_batch_eval);
            full_results = model_.evaluate_batch(full_nodes);
        }

        for (size_t i = 0; i < full_evals.size(); ++i) {
            full_evals[i].promise.set_value(std::move(full_results[i]));
        }

        total_batches_.fetch_add(1, std::memory_order_relaxed);
    }

    // Process value-only requests (singles and batches)
    // Combine singles and batches into one mega-batch for better GPU utilization
    std::vector<const StateNode*> all_nodes;
    size_t singles_count = 0;

    if (!singles.empty()) {
        singles_count = singles.size();
        all_nodes.reserve(singles.size() + batches.size() * 100);  // estimate
        for (const auto& req : singles) {
            all_nodes.push_back(req.node);
        }
    }

    // Track where each batch's results start
    std::vector<size_t> batch_offsets;
    for (auto& batch_req : batches) {
        batch_offsets.push_back(all_nodes.size());
        for (const auto* node : batch_req.nodes) {
            all_nodes.push_back(node);
        }
    }

    if (all_nodes.empty()) return;

    // Single GPU call for everything
    std::vector<float> all_values;
    {
        ScopedTimer t(timers.nn_batch_eval);
        all_values = model_.evaluate_batch_values(all_nodes);
    }

    // Distribute results to singles
    for (size_t i = 0; i < singles_count; ++i) {
        singles[i].promise.set_value(all_values[i]);
    }

    // Distribute results to batches
    for (size_t b = 0; b < batches.size(); ++b) {
        size_t start = batch_offsets[b];
        size_t count = batches[b].nodes.size();
        std::vector<float> batch_values(all_values.begin() + start,
                                         all_values.begin() + start + count);
        batches[b].promise.set_value(std::move(batch_values));
    }

    total_batches_.fetch_add(1, std::memory_order_relaxed);
}

} // namespace qbot
