#include "inference_server.h"

#include <chrono>
#include <iomanip>
#include <iostream>

namespace qbot {

InferenceServer::InferenceServer(const std::string& model_path,
                                 InferenceServerConfig config)
    : model_(model_path, config.batch_size, true)
    , config_(config)
{
    // torch::set_num_threads(1);
    // torch::set_num_interop_threads(1);
}

InferenceServer::~InferenceServer() {
    stop();
}

void InferenceServer::start() {
    if (running_.exchange(true, std::memory_order_acq_rel)) {
        return;  // Already running
    }
    start_time_ = std::chrono::steady_clock::now();

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
    size_t batch_requests = batch_requests_.load();
    size_t batches = total_batches_.load();
    double avg_batch = batches > 0 ? static_cast<double>(requests) / batches : 0.0;
    std::cout << "[InferenceServer] Stopped. Total requests: " << requests
              << ", batched requests: " << batch_requests
              << ", GPU batches: " << batches
              << ", avg batch size: " << std::fixed << std::setprecision(1) << avg_batch << std::endl;
}

void InferenceServer::flush() {
    queue_cv_.notify_all();
}

std::future<EvalResult> InferenceServer::submit(const StateNode* node) {
    std::promise<EvalResult> promise;
    auto future = promise.get_future();

    {
        std::lock_guard lock(queue_mutex_);
        eval_queue_.push_back({node, std::move(promise)});
    }

    total_requests_.fetch_add(1, std::memory_order_relaxed);
    single_requests_.fetch_add(1, std::memory_order_relaxed);
    queue_cv_.notify_one();

    return future;
}

std::vector<std::future<EvalResult>> InferenceServer::submit_batch(
    const std::vector<const StateNode*>& nodes) 
{
    if (nodes.empty()) return {};

    // prepare everything OUTSIDE the lock
    // should not block inference thread
    std::vector<std::future<EvalResult>> futures;
    futures.reserve(nodes.size());

    std::vector<EvalRequest> requests;
    requests.reserve(nodes.size());

    for (const auto* node : nodes) {
        std::promise<EvalResult> p;
        futures.push_back(p.get_future());
        requests.push_back({node, std::move(p)});
    }

    // lock once and move data
    {
        std::lock_guard lock(queue_mutex_);

        // Move all requests to the main queue
        // std::deque::insert with move iterators is efficient
        eval_queue_.insert(
            eval_queue_.end(),
            std::make_move_iterator(requests.begin()),
            std::make_move_iterator(requests.end())
        );
    }

    total_requests_.fetch_add(nodes.size(), std::memory_order_relaxed);
    batch_requests_.fetch_add(1, std::memory_order_relaxed);
    queue_cv_.notify_one(); 

    return futures;
}

void InferenceServer::inference_loop() {
    auto& timers = get_timers();
    while (!stop_requested_.load(std::memory_order_acquire)) {
        // Wait for requests or timeout
        {
            std::unique_lock lock(queue_mutex_);

            auto deadline = std::chrono::steady_clock::now() +
                           std::chrono::microseconds(static_cast<long long>(config_.max_wait_ms * 1000));

            bool batch_trigger = queue_cv_.wait_until(lock, deadline, [this] {
                return eval_queue_.size() >= config_.batch_size || 
                       stop_requested_.load(std::memory_order_acquire);
            });
            if (!batch_trigger) {
                total_time_triggers_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        // Process pending requests
        process_pending();
    }

    // Process any remaining requests before exiting
    process_pending();
}

void InferenceServer::process_pending() {
    auto& timers = get_timers();

    // use a local deque and swap it with the member queue to reduce lock contention
    std::deque<EvalRequest> local_requests;
    {
        std::lock_guard lock(queue_mutex_);

        if (eval_queue_.empty()) {
            return;
        }
        local_requests.swap(eval_queue_);
    }

    std::vector<const StateNode*> nodes;
    nodes.reserve(local_requests.size());

    for (const auto& req : local_requests) {
        nodes.push_back(req.node);
    }

    //run inference
    std::vector<EvalResult> results;
    {
        ScopedTimer t(timers.nn_inference);
        // Process the entire accumulated batch at once
        results = model_.evaluate_batch(nodes);
    }

    // distribute results
    for (size_t i = 0; i < local_requests.size(); ++i) {
        local_requests[i].promise.set_value(std::move(results[i]));
    }

    total_batches_.fetch_add(1, std::memory_order_relaxed);
    total_batch_size_sum_.fetch_add(local_requests.size(), std::memory_order_relaxed);

    if (config_.verbose) {
        size_t batches = total_batches_.load(std::memory_order_relaxed);
        constexpr size_t STATS_INTERVAL = 100000; // Print every N GPU evaluations
        if (batches > 0 && batches % STATS_INTERVAL == 0) {
            print_stats();
        }
    }
}

void InferenceServer::print_stats() {
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = now - start_time_;
    double elapsed = diff.count();

    if (elapsed > 0) {
        size_t requests = total_requests_.load(std::memory_order_relaxed);
        size_t batches = total_batches_.load(std::memory_order_relaxed);
        size_t client_submissions = batch_requests_.load(std::memory_order_relaxed);
        size_t single_submissions = single_requests_.load(std::memory_order_relaxed);
        
        double nodes_per_sec = static_cast<double>(requests) / elapsed;
        double gpu_batches_per_sec = static_cast<double>(batches) / elapsed;
        double avg_batch_size = static_cast<double>(requests) / batches;
        float nodes_per_submit = client_submissions > 0 ? ((requests - single_submissions) / static_cast<float>(client_submissions) ) : 0.0f; //submit_batch() avg size
        float time_trig = 100 * total_time_triggers_.load(std::memory_order_relaxed) / static_cast<float>(batches);

        std::cout << "[InferenceServer] " << std::fixed << std::setprecision(1)
                  << "Nodes/s_b(): " << nodes_per_submit << " | "
                  << "submit_batch(): " << client_submissions << " | "
                  << "GPU Batches: " << batches << " | "
                  << "Time trig: " << time_trig << "% | "
                  << "Avg Batch: " << avg_batch_size << " | "
                  << "Tput: " << std::setprecision(0) << nodes_per_sec << " nodes/s | "
                  << "GPU: " << std::setprecision(1) << gpu_batches_per_sec << " batches/s" 
                  << std::endl;
    }
}

} // namespace qbot
