#include "mcts.h"

#ifdef QBOT_ENABLE_INFERENCE
#include "../inference/inference.h"
#endif

#include "../util/timer.h"

#include <algorithm>
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
            value = evaluate_leaf(pool[eval_node_idx], stats_, inference_);
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
                    leaf.generate_valid_children();
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
            value = evaluate_leaf(pool[eval_node_idx], stats_, inference_);
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

    std::cout << ss.str() << "\n";
}

// ============================================================================
// SelfPlayEngine Implementation
// ============================================================================

#ifdef QBOT_ENABLE_INFERENCE

// Public wrappers that delegate to the template implementation
SelfPlayResult SelfPlayEngine::self_play(NodePool& pool, uint32_t root_idx, ModelInference& model,
                                          TrainingSampleCollector* collector) {
    return self_play_impl(pool, root_idx, model, collector);
}

SelfPlayResult SelfPlayEngine::self_play(NodePool& pool, uint32_t root_idx, InferenceServer& server,
                                          TrainingSampleCollector* collector) {
    return self_play_impl(pool, root_idx, server, collector);
}

template<InferenceProvider Inference>
SelfPlayResult SelfPlayEngine::self_play_impl(NodePool& pool, uint32_t root_idx, Inference& inference,
                                               TrainingSampleCollector* collector) {
    SelfPlayResult result;
    auto& timers = get_timers();

    std::vector<uint32_t> game_path;
    game_path.reserve(config_.max_moves_per_game + 1);

    std::vector<uint32_t> sample_positions;
    if (collector) {
        sample_positions.reserve(config_.max_moves_per_game + 1);
    }

    uint32_t current = root_idx;
    game_path.push_back(current);
    pool[current].set_on_game_path();

    while (current != NULL_NODE) {
        StateNode& node = pool[current];

        // Check terminal conditions
        if (node.is_terminal()) {
            // Convert from relative to absolute perspective
            int relative_value = static_cast<int>(node.terminal_value);
            result.winner = node.is_p1_to_move() ? relative_value : -relative_value;
            break;
        }
        if (node.p1.row == 8) {
            result.winner = 1;  // P1 won
            //terminal val always negative as person whose turn it would be just lost
            node.set_terminal(-1.0f);
            break;
        }
        if (node.p2.row == 0) {
            result.winner = -1;  // P2 won
            node.set_terminal(-1.0f);
            break;
        }
        if (node.p1.fences == 0 && node.p2.fences == 0) {
            int relative_winner = early_terminate_no_fences(node);
            result.winner = node.is_p1_to_move() ? relative_winner : -relative_winner;
            node.set_terminal(static_cast<float>(relative_winner));
            // std::cout << "selfplay earlyterm rel " << relative_winner << " abs " << result.winner << " p1move: " << node.is_p1_to_move() << std::endl;
            break;
        }

        // Expansion
#ifdef QBOT_ENABLE_PROGRESSIVE
        if constexpr (is_inference_server_v<Inference>) {
            if (config_.progressive_expansion) {
                if (!node.priors_set.load(std::memory_order_acquire)) {
                    ScopedTimer t(timers.expansion);
                    compute_priors_progressive(pool, current, inference);
                }
            } else if (!node.is_expanded()) {
                ScopedTimer t(timers.expansion);
                expand_with_nn_priors(pool, current, inference);
            }
        } else
#endif
        {
            if (!node.is_expanded()) {
                ScopedTimer t(timers.expansion);
                expand_with_nn_priors(pool, current, inference);
            }
        }

        // Check for valid moves
#ifdef QBOT_ENABLE_PROGRESSIVE
        bool has_valid_moves = (is_inference_server_v<Inference> && config_.progressive_expansion)
            ? node.valid_moves_computed.load(std::memory_order_acquire)
            : node.has_children();
#else
        bool has_valid_moves = node.has_children();
#endif
        if (!has_valid_moves) {
            result.error = true;
            break;
        }

        // Run MCTS iterations
        {
            ScopedTimer t(timers.mcts_iterations);
            run_mcts_iterations(pool, current, inference, config_.simulations_per_move);
        }

        if (collector) {
            sample_positions.push_back(current);
        }

        // Compute and select move
        float temp = (result.num_moves < config_.temperature_drop_ply)
                     ? config_.temperature : 0.0f;
        std::vector<std::pair<Move, float>> policy;
        {
            ScopedTimer t(timers.policy_compute);
            policy = compute_policy_from_q(pool, current, temp);
        }

        if (policy.empty()) {
            result.error = true;
            break;
        }

        Move selected_move;
        {
            ScopedTimer t(timers.move_selection);
            selected_move = select_move_from_policy(policy, config_.stochastic && temp > 0);
        }

        uint32_t next = NULL_NODE;
        {
            ScopedTimer t(timers.child_lookup);
            uint32_t child = node.first_child;
            while (child != NULL_NODE) {
                if (pool[child].move == selected_move) {
                    next = child;
                    break;
                }
                child = pool[child].next_sibling;
            }
        }

        if (next == NULL_NODE) {
            result.error = true;
            break;
        }

        current = next;
        game_path.push_back(current);
        pool[current].set_on_game_path();
        result.num_moves++;


        // Early termination for long games (scaled draw value)
        if (result.num_moves >= config_.max_moves_per_game) {
            int relative_draw = early_terminate_no_fences(pool[current]);
            int absolute_draw = pool[current].is_p1_to_move() ? relative_draw : -relative_draw;
            float game_value = absolute_draw * config_.max_draw_reward;
            result.winner = 0;
            result.draw_score = game_value;
            for (size_t i = game_path.size(); i > 0; --i) {
                uint32_t idx = game_path[i - 1];
                StateNode& n = pool[idx];
                //converts back from absolute to relative as we go up
                float node_value = n.is_p1_to_move() ? game_value : -game_value;
                n.stats.update(node_value);
            }
            if (collector) {
                for (uint32_t pos : sample_positions) {
                    collector->add_sample(pool, pos, game_value);
                }
            }
            return result;
        }
    }

    // Backpropagate game result
    if (result.winner != 0) {
        ScopedTimer t(timers.backprop);
        float value = static_cast<float>(result.winner);
        for (size_t i = game_path.size(); i > 0; --i) {
            uint32_t idx = game_path[i - 1];
            StateNode& n = pool[idx];
            //conversion back to relative
            float node_value = n.is_p1_to_move() ? value : -value;
            n.stats.update(node_value);
        }
    }

    if (collector && result.winner != 0) {
        float game_outcome = static_cast<float>(result.winner);
        for (uint32_t pos : sample_positions) {
            collector->add_sample(pool, pos, game_outcome);
        }
    }

    return result;
}

template<InferenceProvider Inference>
void SelfPlayEngine::run_mcts_iterations(NodePool& pool, uint32_t root_idx,
                                          Inference& inference, int iterations) {
    auto& timers = get_timers();
    constexpr int EVAL_BATCH_SIZE = 64;

    // PendingEval structure differs: InferenceServer uses futures, ModelInference uses batch eval
    struct PendingEval {
        std::vector<uint32_t> path;
        uint32_t eval_node_idx;
        std::conditional_t<is_inference_server_v<Inference>, std::future<float>, char> future{};
    };
    std::vector<PendingEval> pending;
    pending.reserve(EVAL_BATCH_SIZE);

    auto flush_pending = [&]() {
        if (pending.empty()) return;

        if constexpr (is_inference_server_v<Inference>) {
            // InferenceServer: wait for futures
            for (auto& p : pending) {
                float value = p.future.get();  // Relative perspective
                pool[p.eval_node_idx].stats.set_nn_value(value);
                backpropagate(pool, p.path, value);
            }
        } else {
            // ModelInference: batch evaluate
            std::vector<const StateNode*> nodes_to_eval;
            std::vector<size_t> needs_eval_idx;
            nodes_to_eval.reserve(pending.size());
            needs_eval_idx.reserve(pending.size());

            for (size_t i = 0; i < pending.size(); ++i) {
                StateNode& node = pool[pending[i].eval_node_idx];
                if (node.stats.has_nn_value()) {
                    backpropagate(pool, pending[i].path, node.stats.get_nn_value());
                } else {
                    nodes_to_eval.push_back(&node);
                    needs_eval_idx.push_back(i);
                }
            }

            if (!nodes_to_eval.empty()) {
                std::vector<float> values;
                {
                    ScopedTimer t(timers.nn_single_eval);
                    values = inference.evaluate_batch_values(nodes_to_eval);
                }
                for (size_t j = 0; j < needs_eval_idx.size(); ++j) {
                    size_t i = needs_eval_idx[j];
                    pool[pending[i].eval_node_idx].stats.set_nn_value(values[j]);
                    backpropagate(pool, pending[i].path, values[j]);
                }
            }
        }
        pending.clear();
    };

    MCTSConfig dummy_config;
    dummy_config.c_puct = 1.5f;
    dummy_config.fpu = 0.0f;

    for (int i = 0; i < iterations; ++i) {
        ScopedTimer t(timers.single_mcts);

        std::vector<uint32_t> path;
        path.reserve(64);
        uint32_t current = root_idx;

#ifdef QBOT_ENABLE_PROGRESSIVE
        if constexpr (is_inference_server_v<Inference>) {
            if (config_.progressive_expansion) {
                // Progressive mode: traverse using priors, creating children on demand
                while (current != NULL_NODE) {
                    path.push_back(current);
                    StateNode& node = pool[current];

                    if (node.is_terminal()) break;

                    if (!node.priors_set.load(std::memory_order_acquire)) {
                        ScopedTimer t2(timers.expansion);
                        compute_priors_progressive(pool, current, inference);
                    }

                    uint32_t next = select_and_expand_progressive(pool, current,
                                                                   config_.c_puct, config_.fpu);
                    if (next == NULL_NODE) break;
                    current = next;
                }
            } else {
                // Batch expansion mode
                while (current != NULL_NODE) {
                    path.push_back(current);
                    StateNode& node = pool[current];
                    if (node.is_terminal()) break;
                    if (!node.has_children()) break;
                    current = select_child_puct(pool, current, dummy_config);
                }
            }
        } else
#endif
        {
            // Standard batch expansion (ModelInference or non-progressive InferenceServer)
            while (current != NULL_NODE) {
                path.push_back(current);
                StateNode& node = pool[current];
                if (node.is_terminal()) break;
                if (!node.has_children()) break;
                current = select_child_puct(pool, current, dummy_config);
            }
        }

        if (path.empty()) continue;

        uint32_t leaf_idx = path.back();
        StateNode& leaf = pool[leaf_idx];

        if (leaf.is_terminal()) {
            backpropagate(pool, path, leaf.terminal_value);
            continue;
        }

        // Expansion (skip in progressive mode - already handled)
#ifdef QBOT_ENABLE_PROGRESSIVE
        bool skip_expansion = is_inference_server_v<Inference> && config_.progressive_expansion;
#else
        constexpr bool skip_expansion = false;
#endif
        if (!skip_expansion && !leaf.is_expanded()) {
            ScopedTimer t2(timers.expansion);
            expand_with_nn_priors(pool, leaf_idx, inference);
        }

        uint32_t eval_node_idx = leaf_idx;
        if (!skip_expansion && leaf.has_children()) {
            uint32_t child = select_child_puct(pool, leaf_idx, dummy_config);
            if (child != NULL_NODE) {
                path.push_back(child);
                eval_node_idx = child;
            }
        }

        // Queue for evaluation
        StateNode& eval_node = pool[eval_node_idx];
        if (eval_node.stats.has_nn_value()) {
            backpropagate(pool, path, eval_node.stats.get_nn_value());
        } else {
            if constexpr (is_inference_server_v<Inference>) {
                auto future = inference.submit(&eval_node);
                pending.push_back({std::move(path), eval_node_idx, std::move(future)});
            } else {
                pending.push_back({std::move(path), eval_node_idx, {}});
            }

            if (pending.size() >= EVAL_BATCH_SIZE) {
                flush_pending();
            }
        }
    }

    flush_pending();
}

template<InferenceProvider Inference>
void SelfPlayEngine::expand_with_nn_priors(NodePool& pool, uint32_t node_idx, Inference& inference) {
    auto& timers = get_timers();
    StateNode& node = pool[node_idx];

    size_t mutex_idx = node_idx % NUM_EXPANSION_MUTEXES;
    std::lock_guard lock(expansion_mutexes_[mutex_idx]);

    if (node.is_expanded()) return;

    // Get policy from NN - method differs by inference type
    EvalResult parent_eval;
    {
        ScopedTimer t(timers.nn_single_eval);
        if constexpr (is_inference_server_v<Inference>) {
            auto future = inference.submit_full(&node);
            parent_eval = future.get();
        } else {
            parent_eval = inference.evaluate_node(&node);
        }
    }

    // Cache NN value (already in relative perspective)
    node.stats.set_nn_value(parent_eval.value);

    {
        ScopedTimer t(timers.generate_children);
        node.generate_valid_children();
    }

    if (!node.has_children()) return;

    // Collect policy logits and apply softmax
    std::vector<std::pair<uint32_t, int>> child_actions;
    float max_logit = -std::numeric_limits<float>::infinity();

    uint32_t child = node.first_child;
    while (child != NULL_NODE) {
        int action_idx = move_to_action_index(pool[child].move);
        if (action_idx >= 0 && action_idx < NUM_ACTIONS) {
            child_actions.emplace_back(child, action_idx);
            max_logit = std::max(max_logit, parent_eval.policy[action_idx]);
        }
        child = pool[child].next_sibling;
    }

    if (child_actions.empty()) return;

    float sum_exp = 0.0f;
    std::vector<float> exp_logits(child_actions.size());
    for (size_t i = 0; i < child_actions.size(); ++i) {
        int action_idx = child_actions[i].second;
        exp_logits[i] = std::exp(parent_eval.policy[action_idx] - max_logit);
        sum_exp += exp_logits[i];
    }

    for (size_t i = 0; i < child_actions.size(); ++i) {
        uint32_t child_idx = child_actions[i].first;
        pool[child_idx].stats.prior = exp_logits[i] / sum_exp;
    }
}

/// Backpropagate value up path (value is relative to last node in path)
void SelfPlayEngine::backpropagate(NodePool& pool, const std::vector<uint32_t>& path, float value) {
    float node_value = value;
    for (size_t i = path.size(); i > 0; --i) {
        uint32_t idx = path[i - 1];
        StateNode& node = pool[idx];
        node.stats.update(node_value);
        node_value = -node_value;  // Negate as we go up (alternating players)
    }
}

// ============================================================================
// Progressive Expansion Methods
// Disabled by default - define QBOT_ENABLE_PROGRESSIVE to re-enable.
// ============================================================================

#ifdef QBOT_ENABLE_PROGRESSIVE

void SelfPlayEngine::compute_priors_progressive(NodePool& pool, uint32_t node_idx, InferenceServer& server) {
    StateNode& node = pool[node_idx];

    // Compute valid action mask (with pathfinding) - thread-safe, only done once
    node.compute_valid_action_mask();

    // Check if priors already set
    if (node.priors_set.load(std::memory_order_acquire)) {
        return;
    }

    // Submit GPU request OUTSIDE mutex - allows other threads to submit while we wait
    // This enables batching in the inference server
    auto future = server.submit_full(&node);
    EvalResult eval = future.get();  // Block here, but other threads can submit requests

    // Now acquire mutex just for the short critical section
    size_t mutex_idx = node_idx % NUM_EXPANSION_MUTEXES;
    std::lock_guard lock(expansion_mutexes_[mutex_idx]);

    // Double-check after acquiring lock - another thread might have finished first
    if (node.priors_set.load(std::memory_order_acquire)) {
        return;
    }

    // Cache NN value (already in relative perspective)
    node.stats.set_nn_value(eval.value);

    // Compute softmax over valid actions only
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < StateNode::NUM_ACTIONS; ++i) {
        if (node.is_action_valid(i)) {
            max_logit = std::max(max_logit, eval.policy[i]);
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < StateNode::NUM_ACTIONS; ++i) {
        if (node.is_action_valid(i)) {
            node.policy_priors[i] = std::exp(eval.policy[i] - max_logit);
            sum_exp += node.policy_priors[i];
        } else {
            node.policy_priors[i] = 0.0f;
        }
    }

    // Normalize
    if (sum_exp > 0.0f) {
        for (int i = 0; i < StateNode::NUM_ACTIONS; ++i) {
            node.policy_priors[i] /= sum_exp;
        }
    }

    node.priors_set.store(true, std::memory_order_release);
}

uint32_t SelfPlayEngine::select_and_expand_progressive(NodePool& pool, uint32_t node_idx,
                                                        float c_puct, float fpu) {
    StateNode& node = pool[node_idx];

    // Ensure valid moves and priors are computed
    if (!node.priors_set.load(std::memory_order_acquire)) {
        return NULL_NODE;
    }

    // Build action_idx -> child_idx map in O(num_children) instead of O(num_actions * num_children)
    std::array<uint32_t, StateNode::NUM_ACTIONS> action_to_child;
    action_to_child.fill(NULL_NODE);

    uint32_t child = node.first_child;
    while (child != NULL_NODE) {
        int action_idx = move_to_action_index(pool[child].move);
        if (action_idx >= 0 && action_idx < StateNode::NUM_ACTIONS) {
            action_to_child[action_idx] = child;
        }
        child = pool[child].next_sibling;
    }

    uint32_t parent_visits = node.stats.visits.load(std::memory_order_relaxed);
    float sqrt_parent = std::sqrt(static_cast<float>(parent_visits));

    float best_score = -std::numeric_limits<float>::infinity();
    int best_action = -1;
    uint32_t best_child = NULL_NODE;

    // Iterate over valid actions with O(1) child lookup
    for (int action_idx = 0; action_idx < StateNode::NUM_ACTIONS; ++action_idx) {
        if (!node.is_action_valid(action_idx)) continue;

        float prior = node.policy_priors[action_idx];
        uint32_t child_idx = action_to_child[action_idx];

        float q, n_with_vl;
        if (child_idx != NULL_NODE) {
            q = pool[child_idx].stats.Q(fpu);
            n_with_vl = static_cast<float>(pool[child_idx].stats.N_with_virtual());
        } else {
            q = fpu;
            n_with_vl = 0.0f;
        }

        float u = c_puct * prior * sqrt_parent / (1.0f + n_with_vl);
        float score = q + u;

        if (score > best_score) {
            best_score = score;
            best_action = action_idx;
            best_child = child_idx;
        }
    }

    // If best action has no child yet, create it
    if (best_child == NULL_NODE && best_action >= 0) {
        best_child = node.add_child_for_action(best_action);
    }

    return best_child;
}

#endif // QBOT_ENABLE_PROGRESSIVE

void SelfPlayEngine::refresh_priors(NodePool& pool, uint32_t node_idx, InferenceServer& server) {
    StateNode& node = pool[node_idx];

    if (!node.has_children()) return;

    // Evaluate the node to get fresh policy logits
    auto future = server.submit_full(&node);
    EvalResult eval = future.get();

    // Collect policy logits for valid moves and apply softmax
    std::vector<std::pair<uint32_t, int>> child_actions;
    float max_logit = -std::numeric_limits<float>::infinity();

    uint32_t child = node.first_child;
    while (child != NULL_NODE) {
        int action_idx = move_to_action_index(pool[child].move);
        if (action_idx >= 0 && action_idx < NUM_ACTIONS) {
            child_actions.emplace_back(child, action_idx);
            max_logit = std::max(max_logit, eval.policy[action_idx]);
        }
        child = pool[child].next_sibling;
    }

    if (child_actions.empty()) return;

    // Compute softmax over valid moves only
    float sum_exp = 0.0f;
    std::vector<float> exp_logits(child_actions.size());
    for (size_t i = 0; i < child_actions.size(); ++i) {
        int action_idx = child_actions[i].second;
        exp_logits[i] = std::exp(eval.policy[action_idx] - max_logit);
        sum_exp += exp_logits[i];
    }

    // Update priors on children
    for (size_t i = 0; i < child_actions.size(); ++i) {
        uint32_t child_idx = child_actions[i].first;
        pool[child_idx].stats.prior = exp_logits[i] / sum_exp;
    }
}

void SelfPlayEngine::run_multi_game(
    NodePool& pool,
    uint32_t& root_idx,
    InferenceServer& server,
    int num_games,
    int num_workers,
    MultiGameStats& stats,
    const TreeBoundsConfig& bounds,
    TrainingSampleCollector* collector,
    const std::filesystem::path& samples_file,
    std::function<void(const MultiGameStats&, const NodePool&)> checkpoint_callback,
    int checkpoint_interval_games)
{
    stats.reset();
    std::atomic<int> games_remaining{num_games};

    // Synchronization for memory-bounded pool reset
    std::atomic<bool> pause_requested{false};
    std::atomic<int> workers_paused{0};
    std::mutex pause_mutex;
    std::condition_variable pause_cv;
    std::condition_variable resume_cv;
    std::atomic<bool> paused{false};

    // Shared root index that workers can read (updated on pool reset)
    std::atomic<uint32_t> current_root{root_idx};

    int pool_reset_count = 0;

    std::cout << "[SelfPlayEngine] Starting " << num_games << " games with "
              << num_workers << " workers\n";
    std::cout << "[SelfPlayEngine] Memory limit: " << (bounds.max_bytes / (1024*1024*1024)) << " GB, "
              << "soft limit: " << (bounds.soft_limit_ratio * 100) << "%\n";

    auto start_time = std::chrono::steady_clock::now();

    // Launch worker threads
    std::vector<std::jthread> workers;
    workers.reserve(num_workers);

    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back([this, &pool, &current_root, &server, &games_remaining, &stats,
                              &pause_requested, &workers_paused, &pause_mutex, &pause_cv,
                              &resume_cv, &paused, i](std::stop_token st) {
            while (!st.stop_requested()) {
                // Check if pause is requested - if so, wait
                if (pause_requested.load(std::memory_order_acquire)) {
                    workers_paused.fetch_add(1, std::memory_order_relaxed);
                    {
                        std::unique_lock lock(pause_mutex);
                        pause_cv.notify_all();  // Signal main thread we're paused
                        resume_cv.wait(lock, [&]() {
                            return !paused.load(std::memory_order_acquire) || st.stop_requested();
                        });
                    }
                    workers_paused.fetch_sub(1, std::memory_order_relaxed);
                    if (st.stop_requested()) break;
                }

                // Claim a game to play
                int remaining = games_remaining.fetch_sub(1, std::memory_order_relaxed);
                if (remaining <= 0) {
                    games_remaining.fetch_add(1, std::memory_order_relaxed);
                    break;
                }

                // Play one game using current root
                uint32_t root = current_root.load(std::memory_order_acquire);
                SelfPlayResult result = self_play(pool, root, server);
                stats.add_result(result);
            }
        });
    }

    // Monitor progress, call checkpoints, and handle memory-bounded resets
    int last_checkpoint = 0;
    while (games_remaining.load(std::memory_order_relaxed) > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Check memory usage
        size_t current_bytes = pool.memory_usage_bytes();
        float utilization = static_cast<float>(current_bytes) / static_cast<float>(bounds.max_bytes);

        if (utilization >= bounds.soft_limit_ratio && collector != nullptr) {
            std::cout << "\n[SelfPlayEngine] Memory at " << std::fixed << std::setprecision(1)
                      << (utilization * 100) << "% - initiating pool reset..." << std::endl;

            // Request workers to pause
            paused.store(true, std::memory_order_release);
            pause_requested.store(true, std::memory_order_release);

            // Wait for all workers to pause (with timeout)
            {
                std::unique_lock lock(pause_mutex);
                auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                pause_cv.wait_until(lock, deadline, [&]() {
                    return workers_paused.load(std::memory_order_relaxed) >= num_workers ||
                           games_remaining.load(std::memory_order_relaxed) <= 0;
                });
            }

            int paused_count = workers_paused.load(std::memory_order_relaxed);
            std::cout << "[SelfPlayEngine] " << paused_count << "/" << num_workers << " workers paused" << std::endl;

            // Extract training samples from current tree
            uint32_t old_root = current_root.load(std::memory_order_relaxed);
            auto tree_samples = extract_samples_from_tree(pool, old_root);

            for (auto& sample : tree_samples) {
                collector->add_sample_direct(std::move(sample));
            }

            // Save intermediate qsamples file
            if (!samples_file.empty() && collector->size() > 0) {
                auto result = TrainingSampleStorage::save(samples_file, collector->samples());
                if (!result) {
                    std::cerr << "[SelfPlayEngine] Warning: Failed to save intermediate samples\n";
                }
            }

            pool.clear();

            // Reinitialize root node
            uint32_t new_root = pool.allocate();
            pool[new_root].init_root(true);
            current_root.store(new_root, std::memory_order_release);
            root_idx = new_root;

            ++pool_reset_count;
            std::cout << "[SelfPlayEngine] Pool reset #" << pool_reset_count
                      << " complete" << std::endl;

            // Resume workers
            pause_requested.store(false, std::memory_order_release);
            paused.store(false, std::memory_order_release);
            resume_cv.notify_all();
        }

        // Regular checkpoint callback
        int completed = stats.games_completed.load(std::memory_order_relaxed);
        if (checkpoint_callback &&
            completed - last_checkpoint >= checkpoint_interval_games) {
            checkpoint_callback(stats, pool);
            last_checkpoint = completed;
        }
    }

    // Signal stop and wait for workers
    for (auto& w : workers) {
        w.request_stop();
    }
    resume_cv.notify_all();  // Wake any paused workers
    workers.clear();

    auto end_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    int completed = stats.games_completed.load(std::memory_order_relaxed);
    auto games_per_sec = elapsed > 0 ? static_cast<double>(completed) / elapsed : 0;

    std::cout << "[SelfPlayEngine] Completed " << completed << " games in "
              << elapsed << "s (" << std::fixed << std::setprecision(1)
              << games_per_sec << " games/s)";
    if (pool_reset_count > 0) {
        std::cout << " with " << pool_reset_count << " pool resets";
    }
    std::cout << "\n";
}

void SelfPlayEngine::worker_loop(
    std::stop_token stop_token,
    int worker_id,
    NodePool& pool,
    uint32_t root_idx,
    InferenceServer& server,
    std::atomic<int>& games_remaining,
    MultiGameStats& stats)
{
    (void)worker_id;  // Could use for logging

    while (!stop_token.stop_requested()) {
        // Claim a game to play
        int remaining = games_remaining.fetch_sub(1, std::memory_order_relaxed);
        if (remaining <= 0) {
            // No more games - restore the count and exit
            games_remaining.fetch_add(1, std::memory_order_relaxed);
            break;
        }

        // Play one game
        SelfPlayResult result = self_play(pool, root_idx, server);
        stats.add_result(result);
    }
}

// ============================================================================
// Arena Implementation (two models playing against each other)
// ============================================================================

/// Helper to get the correct server based on whose turn it is
inline InferenceServer& get_server_for_player(
    const StateNode& node,
    InferenceServer& server_p1,
    InferenceServer& server_p2) {
    return node.is_p1_to_move() ? server_p1 : server_p2;
}

/// Run MCTS iterations for arena mode with two servers
void SelfPlayEngine::run_arena_mcts_iterations(
    NodePool& pool, uint32_t root_idx,
    InferenceServer& server_p1, InferenceServer& server_p2,
    int iterations)
{
    auto& timers = get_timers();
    constexpr int EVAL_BATCH_SIZE = 32;  // Smaller batches since we split between two servers

    struct PendingEval {
        std::vector<uint32_t> path;
        uint32_t eval_node_idx;
        std::future<float> future;
    };
    std::vector<PendingEval> pending;
    pending.reserve(EVAL_BATCH_SIZE);

    auto flush_pending = [&]() {
        if (pending.empty()) return;

        for (auto& p : pending) {
            float leaf_value = p.future.get();  // Relative perspective at eval_node
            // Don't cache NN values in arena - different models produce different values
            if (p.eval_node_idx != p.path.back()) {
                leaf_value = -leaf_value;
                std::cout << "ERROR last nod in path isnt eval node" << std::endl;
            }
            backpropagate(pool, p.path, leaf_value);
        }
        pending.clear();
    };

    MCTSConfig dummy_config;
    dummy_config.c_puct = 1.5f;
    dummy_config.fpu = 0.0f;

    for (int i = 0; i < iterations; ++i) {
        ScopedTimer t(timers.single_mcts);

        std::vector<uint32_t> path;
        path.reserve(64);

        uint32_t current = root_idx;
        while (current != NULL_NODE) {
            path.push_back(current);
            StateNode& node = pool[current];

            if (node.is_terminal()) break;
            if (!node.has_children()) break;

            current = select_child_puct(pool, current, dummy_config);
        }

        if (path.empty()) continue;

        uint32_t leaf_idx = path.back();
        StateNode& leaf = pool[leaf_idx];

        if (leaf.is_terminal()) {
            backpropagate(pool, path, leaf.terminal_value);
            continue;
        }

        if (!leaf.is_expanded()) {
            ScopedTimer t2(timers.expansion);
            // Expand using the server for the player at this position
            InferenceServer& server = get_server_for_player(leaf, server_p1, server_p2);
            expand_with_nn_priors(pool, leaf_idx, server);
        }

        uint32_t eval_node_idx = leaf_idx;
        if (leaf.has_children()) {
            uint32_t child = select_child_puct(pool, leaf_idx, dummy_config);
            if (child != NULL_NODE) {
                path.push_back(child);
                eval_node_idx = child;
            }
        }

        // In arena, always evaluate fresh - don't use cached values since
        // different models produce different values for the same position
        StateNode& eval_node = pool[eval_node_idx];
        InferenceServer& server = get_server_for_player(eval_node, server_p1, server_p2);
        auto future = server.submit(&eval_node);
        pending.push_back({std::move(path), eval_node_idx, std::move(future)});

        if (pending.size() >= EVAL_BATCH_SIZE) {
            flush_pending();
        }
    }

    flush_pending();
}

SelfPlayResult SelfPlayEngine::arena_game(
    NodePool& pool, uint32_t root_idx,
    InferenceServer& server_p1, InferenceServer& server_p2)
{
    auto& timers = get_timers();
    SelfPlayResult result;
    result.winner = 0;
    result.num_moves = 0;

    std::vector<uint32_t> game_path;
    game_path.reserve(config_.max_moves_per_game + 1);
    game_path.push_back(root_idx);
    pool[root_idx].set_on_game_path();

    uint32_t current = root_idx;

    while (current != NULL_NODE) {
        StateNode& node = pool[current];

        // Check terminal
        if (node.is_terminal()) {
            int relative_value = static_cast<int>(node.terminal_value);
            result.winner = node.is_p1_to_move() ? relative_value : -relative_value;
            break;
        }
        if (node.p1.row == 8) {
            result.winner = 1;
            node.set_terminal(-1.0f);
            break;
        }
        if (node.p2.row == 0) {
            result.winner = -1;
            node.set_terminal(-1.0f);
            break;
        }
        if (node.p1.fences == 0 && node.p2.fences == 0) {
            int relative_winner = early_terminate_no_fences(node);
            result.winner = node.is_p1_to_move() ? relative_winner : -relative_winner;
            node.set_terminal(static_cast<float>(relative_winner));
            break;
        }

        // Refresh priors on existing children using the model for the player whose turn it is
        // This is critical for arena: priors may have been set by a different model in a previous game
        InferenceServer& current_server = node.is_p1_to_move() ? server_p1 : server_p2;
        if (node.has_children()) {
            refresh_priors(pool, current, current_server);
        }

        // Run MCTS with both servers
        {
            ScopedTimer t(timers.mcts_iterations);
            run_arena_mcts_iterations(pool, current, server_p1, server_p2, config_.simulations_per_move);
        }

        // Use temperature for exploration early, then drop to deterministic
        float temp = (result.num_moves < config_.temperature_drop_ply)
                     ? config_.temperature : 0.0f;
        bool stochastic = (temp > 0.0f);

        std::vector<std::pair<Move, float>> policy;
        {
            ScopedTimer t(timers.policy_compute);
            policy = compute_policy_from_q(pool, current, temp);
        }

        if (policy.empty()) {
            result.error = true;;
            break;
        }

        Move selected_move;
        {
            ScopedTimer t(timers.move_selection);
            selected_move = select_move_from_policy(policy, stochastic);
        }

        uint32_t next = NULL_NODE;
        {
            ScopedTimer t(timers.child_lookup);
            uint32_t child = node.first_child;
            while (child != NULL_NODE) {
                if (pool[child].move == selected_move) {
                    next = child;
                    break;
                }
                child = pool[child].next_sibling;
            }
        }

        if (next == NULL_NODE) {
            result.error = true;
            break;
        }

        current = next;
        game_path.push_back(current);
        pool[current].set_on_game_path();
        result.num_moves++;

        // Early termination for long games
        if (result.num_moves >= config_.max_moves_per_game) {
            int relative_draw = early_terminate_no_fences(pool[current]);
            int absolute_draw = pool[current].is_p1_to_move() ? relative_draw : -relative_draw;
            float game_value = absolute_draw * config_.max_draw_reward;
            result.winner = 0;
            result.draw_score = absolute_draw;
            for (size_t i = game_path.size(); i > 0; --i) {
                uint32_t idx = game_path[i - 1];
                StateNode& n = pool[idx];
                //conversion back to relative
                float node_value = n.is_p1_to_move() ? game_value : -game_value;
                n.stats.update(node_value);
            }
            return result;
        }
    }

    // Backpropagate final result
    if (! result.error) {
        ScopedTimer t(timers.backprop);
        float value = static_cast<float>(result.winner);
        for (size_t i = game_path.size(); i > 0; --i) {
            uint32_t idx = game_path[i - 1];
            StateNode& node = pool[idx];
            float node_value = node.is_p1_to_move() ? value : -value;
            node.stats.update(node_value);
        }
    }

    return result;
}

void SelfPlayEngine::arena_worker_loop(
    std::stop_token stop_token,
    int worker_id,
    NodePool& pool,
    uint32_t root_idx,
    InferenceServer& server_p1,
    InferenceServer& server_p2,
    std::atomic<int>& games_remaining,
    std::atomic<int>& game_counter,
    MultiGameStats& stats)
{
    (void)worker_id;

    while (!stop_token.stop_requested()) {
        int remaining = games_remaining.fetch_sub(1, std::memory_order_relaxed);
        if (remaining <= 0) {
            games_remaining.fetch_add(1, std::memory_order_relaxed);
            break;
        }

        // Alternate which model is P1 based on game number
        int game_num = game_counter.fetch_add(1, std::memory_order_relaxed);
        bool swap_players = (game_num % 2 == 1);

        SelfPlayResult result;
        if (swap_players) {
            // Swap: server_p2 plays as P1, server_p1 plays as P2
            result = arena_game(pool, root_idx, server_p2, server_p1);
            // Invert winner since we swapped the models
            result.winner = -result.winner;
        } else {
            result = arena_game(pool, root_idx, server_p1, server_p2);
        }

        stats.add_result(result);
    }
}

void SelfPlayEngine::run_multi_arena(
    NodePool& pool,
    uint32_t root_idx,
    InferenceServer& server_p1,
    InferenceServer& server_p2,
    int num_games,
    int num_workers,
    MultiGameStats& stats,
    std::function<void(const MultiGameStats&, const NodePool&)> checkpoint_callback,
    int checkpoint_interval_games)
{
    stats.reset();
    std::atomic<int> games_remaining{num_games};
    std::atomic<int> game_counter{0};

    std::cout << "[SelfPlayEngine] Starting arena: " << num_games << " games with "
              << num_workers << " workers\n";

    auto start_time = std::chrono::steady_clock::now();

    // Launch worker threads
    std::vector<std::jthread> workers;
    workers.reserve(num_workers);

    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back([this, &pool, root_idx, &server_p1, &server_p2,
                              &games_remaining, &game_counter, &stats, i](std::stop_token st) {
            arena_worker_loop(st, i, pool, root_idx, server_p1, server_p2,
                              games_remaining, game_counter, stats);
        });
    }

    // Monitor progress
    int last_checkpoint = 0;
    while (stats.games_completed.load(std::memory_order_relaxed) < num_games) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        int completed = stats.games_completed.load(std::memory_order_relaxed);
        if (checkpoint_callback &&
            completed - last_checkpoint >= checkpoint_interval_games) {
            checkpoint_callback(stats, pool);
            last_checkpoint = completed;
        }
    }

    // Wait for all workers to finish
    workers.clear();

    auto end_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    int completed = stats.games_completed.load(std::memory_order_relaxed);
    auto games_per_sec = elapsed > 0 ? static_cast<double>(completed) / elapsed : 0;

    std::cout << "[SelfPlayEngine] Arena completed " << completed << " games in "
              << elapsed << "s (" << std::fixed << std::setprecision(1)
              << games_per_sec << " games/s)\n";
}

// Explicit template instantiations for both inference types
template SelfPlayResult SelfPlayEngine::self_play_impl<ModelInference>(
    NodePool&, uint32_t, ModelInference&, TrainingSampleCollector*);
template SelfPlayResult SelfPlayEngine::self_play_impl<InferenceServer>(
    NodePool&, uint32_t, InferenceServer&, TrainingSampleCollector*);

template void SelfPlayEngine::run_mcts_iterations<ModelInference>(
    NodePool&, uint32_t, ModelInference&, int);
template void SelfPlayEngine::run_mcts_iterations<InferenceServer>(
    NodePool&, uint32_t, InferenceServer&, int);

template void SelfPlayEngine::expand_with_nn_priors<ModelInference>(
    NodePool&, uint32_t, ModelInference&);
template void SelfPlayEngine::expand_with_nn_priors<InferenceServer>(
    NodePool&, uint32_t, InferenceServer&);

#endif // QBOT_ENABLE_INFERENCE

} // namespace qbot
