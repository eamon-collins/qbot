#include "mcts.h"

#include <iomanip>
#include <iostream>
#include <sstream>

namespace qbot {

// ============================================================================
// MCTSEngine Implementation
// ============================================================================

void MCTSEngine::start_training(NodePool& pool, uint32_t root_idx) {
    if (running_.exchange(true, std::memory_order_acq_rel)) {
        return;  // Already running
    }

    pool_ptr_ = &pool;
    root_idx_ = root_idx;
    stats_.reset();

    std::cout << "Starting MCTS training with " << config_.num_threads << " threads\n";
    std::cout << "  c_puct: " << config_.c_puct << "\n";
    std::cout << "  virtual_loss: " << config_.virtual_loss_amount << "\n";
    std::cout << "  memory limit: " << (config_.bounds.max_bytes / (1024ULL * 1024 * 1024)) << " GB\n";
    std::cout << "  soft limit: " << (config_.bounds.soft_limit_ratio * 100) << "%"
              << ", hard limit: " << (config_.bounds.hard_limit_ratio * 100) << "%\n";
    std::cout << "  min visits to expand: " << config_.bounds.min_visits_to_expand << "\n";
    std::cout << "  checkpoint interval: " << config_.checkpoint_interval_seconds << "s\n";
    if (!config_.checkpoint_path.empty()) {
        std::cout << "  checkpoint path: " << config_.checkpoint_path << "\n";
    }
    std::cout << "\n";

    // Launch worker threads
    for (int i = 0; i < config_.num_threads; ++i) {
        workers_.emplace_back([this, &pool, root_idx, i](std::stop_token st) {
            worker_loop(st, i, pool, root_idx);
        });
    }

    // Launch checkpoint thread if path is set
    if (!config_.checkpoint_path.empty() && config_.checkpoint_interval_seconds > 0) {
        checkpoint_thread_ = std::jthread([this, &pool, root_idx](std::stop_token st) {
            checkpoint_loop(st, pool, root_idx);
        });
    }
}

void MCTSEngine::stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) {
        return;  // Not running
    }

    std::cout << "\nStopping MCTS training...\n";

    // Request stop on all threads (jthread handles this automatically on destruction)
    // Clear workers - this requests stop and joins
    workers_.clear();
    checkpoint_thread_ = std::jthread{};

    // Print final stats
    print_stats(*pool_ptr_);

    pool_ptr_ = nullptr;
    root_idx_ = NULL_NODE;
}

void MCTSEngine::mcts_iteration(NodePool& pool, uint32_t root_idx) {
    // SELECTION: traverse to leaf, applying virtual loss
    SelectionResult selection = select_to_leaf(pool, root_idx, config_);

    if (selection.path.empty()) {
        return;  // Something went wrong
    }

    float value;
    uint32_t eval_node_idx = selection.leaf_idx;

    switch (selection.expansion) {
        case ExpansionDecision::Terminal:
            // Terminal node - use known value
            value = pool[selection.leaf_idx].terminal_value;
            break;

        case ExpansionDecision::UseNNEvaluation:
            // Memory pressure too high - skip expansion, evaluate directly
            stats_.skipped_expansions.fetch_add(1, std::memory_order_relaxed);
            {
                uint32_t root_visits = pool[root_idx].stats.visits.load(std::memory_order_relaxed);
                value = evaluate_leaf(pool[eval_node_idx], config_, root_visits, stats_, inference_);
            }
            break;

        case ExpansionDecision::Expand:
        case ExpansionDecision::AlreadyExpanded:
        default: {
            StateNode& leaf = pool[selection.leaf_idx];

            // EXPANSION: generate children if not already expanded
            if (!leaf.is_expanded()) {
                // Use striped mutex - allows parallel expansion of different nodes
                size_t mutex_idx = selection.leaf_idx % NUM_EXPANSION_MUTEXES;
                std::lock_guard lock(expansion_mutexes_[mutex_idx]);
                if (!leaf.is_expanded()) {
                    leaf.generate_valid_children(pool, selection.leaf_idx);
                }
            }

            // If expansion created children, select one for evaluation
            if (leaf.has_children()) {
                uint32_t child = select_child_puct(pool, selection.leaf_idx, config_);
                if (child != NULL_NODE) {
                    // Add virtual loss to selected child and add to path
                    pool[child].stats.add_virtual_loss(config_.virtual_loss_amount);
                    selection.path.push_back(child);
                    eval_node_idx = child;
                }
            }

            // EVALUATION
            uint32_t root_visits = pool[root_idx].stats.visits.load(std::memory_order_relaxed);
            value = evaluate_leaf(pool[eval_node_idx], config_, root_visits, stats_, inference_);
            break;
        }
    }

    // Update max depth statistic
    stats_.update_max_depth(static_cast<uint32_t>(selection.path.size()));

    // BACKPROPAGATION: update stats and remove virtual loss
    backpropagate(pool, selection.path, value);

    stats_.total_iterations.fetch_add(1, std::memory_order_relaxed);
}

void MCTSEngine::backpropagate(NodePool& pool, const std::vector<uint32_t>& path,
                                float value) noexcept {
    // Propagate value up the tree
    // Value is from P1's perspective (+1 = P1 wins)
    // Each node's value is from the perspective of the player to move at that node

    for (size_t i = path.size(); i > 0; --i) {
        uint32_t idx = path[i - 1];
        StateNode& node = pool[idx];

        // Value from this node's perspective
        // If P1 to move at this node, P1 wants high value -> use value as-is
        // If P2 to move at this node, P2 wants low value -> negate
        float node_value = node.is_p1_to_move() ? value : -value;

        // Atomic update
        node.stats.update(node_value);

        // Remove virtual loss
        node.stats.remove_virtual_loss(config_.virtual_loss_amount);
    }
}

void MCTSEngine::worker_loop(std::stop_token stop_token, int thread_id,
                              NodePool& pool, uint32_t root_idx) {
    (void)thread_id;  // Could use for logging

    while (!stop_token.stop_requested()) {
        // Check for checkpoint pause
        if (paused_.load(std::memory_order_acquire)) {
            workers_paused_.fetch_add(1, std::memory_order_release);

            // Wait for resume
            std::unique_lock lock(pause_mutex_);
            pause_cv_.wait(lock, [this, &stop_token] {
                return !paused_.load(std::memory_order_acquire) ||
                       stop_token.stop_requested();
            });

            workers_paused_.fetch_sub(1, std::memory_order_release);

            if (stop_token.stop_requested()) {
                break;
            }
            continue;
        }

        // Run one MCTS iteration
        mcts_iteration(pool, root_idx);
    }
}

void MCTSEngine::checkpoint_loop(std::stop_token stop_token, NodePool& pool,
                                  uint32_t root_idx) {
    while (!stop_token.stop_requested()) {
        // Sleep for checkpoint interval
        for (int i = 0; i < config_.checkpoint_interval_seconds && !stop_token.stop_requested(); ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        if (stop_token.stop_requested()) {
            break;
        }

        // Pause workers
        pause_workers();

        // Print stats
        print_stats(pool);

        // Save checkpoint
        if (!config_.checkpoint_path.empty()) {
            auto result = TreeStorage::save(config_.checkpoint_path, pool, root_idx);
            if (result) {
                auto file_size = std::filesystem::file_size(config_.checkpoint_path);
                std::cout << "  Saved checkpoint: " << file_size / 1024 << " KB\n";
            } else {
                std::cerr << "  Checkpoint failed: " << to_string(result.error()) << "\n";
            }
        }

        std::cout << std::flush;

        // Resume workers
        resume_workers();
    }
}

void MCTSEngine::pause_workers() {
    paused_.store(true, std::memory_order_release);

    // Wait for all workers to acknowledge pause
    int expected = static_cast<int>(workers_.size());
    while (workers_paused_.load(std::memory_order_acquire) < expected) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void MCTSEngine::resume_workers() {
    {
        std::lock_guard lock(pause_mutex_);
        paused_.store(false, std::memory_order_release);
    }
    pause_cv_.notify_all();

    // Wait for all workers to resume
    while (workers_paused_.load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void MCTSEngine::print_stats(const NodePool& pool) const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - stats_.start_time).count();

    uint64_t iterations = stats_.total_iterations.load(std::memory_order_relaxed);
    uint64_t rollouts = stats_.total_rollouts.load(std::memory_order_relaxed);
    uint64_t nn_evals = stats_.nn_evaluations.load(std::memory_order_relaxed);
    uint64_t skipped = stats_.skipped_expansions.load(std::memory_order_relaxed);
    uint32_t depth = stats_.max_depth.load(std::memory_order_relaxed);

    double iter_per_sec = elapsed > 0 ? static_cast<double>(iterations) / elapsed : 0;

    // Memory usage
    size_t mem_used_mb = pool.memory_usage_bytes() / (1024 * 1024);
    size_t mem_limit_mb = config_.bounds.max_bytes / (1024 * 1024);
    float mem_pct = 100.0f * static_cast<float>(pool.memory_usage_bytes()) /
                    static_cast<float>(config_.bounds.max_bytes);

    // Format time as HH:MM:SS
    int hours = static_cast<int>(elapsed / 3600);
    int mins = static_cast<int>((elapsed % 3600) / 60);
    int secs = static_cast<int>(elapsed % 60);

    std::ostringstream ss;
    ss << "[" << std::setfill('0')
       << std::setw(2) << hours << ":"
       << std::setw(2) << mins << ":"
       << std::setw(2) << secs << "] ";

    ss << "Iter: " << iterations
       << ", Nodes: " << pool.allocated()
       << ", Mem: " << mem_used_mb << "/" << mem_limit_mb << "MB"
       << " (" << std::fixed << std::setprecision(1) << mem_pct << "%)"
       << ", Depth: " << depth
       << ", Rate: " << std::setprecision(0) << iter_per_sec << "/s";

    if (skipped > 0) {
        ss << ", Skipped: " << skipped;
    }
    if (nn_evals > 0) {
        ss << ", NN: " << nn_evals;
    }
    ss << ", Rollouts: " << rollouts;

    std::cout << ss.str() << "\n";
}

} // namespace qbot
