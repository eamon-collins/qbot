#include "mcts.h"

#include "../inference/inference.h"

#include "../util/timer.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace qbot {


// ============================================================================
// SelfPlayEngine Implementation
// ============================================================================


template<InferenceProvider Inference>
void SelfPlayEngine::run_mcts_iterations(NodePool& pool, uint32_t root_idx,
                                          Inference& inference, int iterations) {
    auto& timers = get_timers();
    constexpr int EVAL_BATCH_SIZE = 128;

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
                values = inference.evaluate_batch_values(nodes_to_eval);
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
        std::vector<uint32_t> path;
        path.reserve(64);
        uint32_t current = root_idx;

        // Standard batch expansion (ModelInference or non-progressive InferenceServer)
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

        // Expansion
        if (!leaf.is_expanded()) {
            expand_with_nn_priors(pool, leaf_idx, inference);
        }

        uint32_t eval_node_idx = leaf_idx;
        if (leaf.has_children()) {
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
    if constexpr (is_inference_server_v<Inference>) {
        auto future = inference.submit_full(&node);
        parent_eval = future.get();
    } else {
        parent_eval = inference.evaluate_node(&node);
    }

    // Cache NN value (already in relative perspective)
    node.stats.set_nn_value(parent_eval.value);

    node.generate_valid_children();

    if (!node.has_children()) return;

    // Collect policy logits and apply softmax
    std::vector<std::pair<uint32_t, int>> child_actions;
    float max_logit = -std::numeric_limits<float>::infinity();
    bool flip_policy = !node.is_p1_to_move(); 

    uint32_t child = node.first_child;
    while (child != NULL_NODE) {
        int action_idx = move_to_action_index(pool[child].move);
        action_idx = flip_policy ? flip_action_index(action_idx) : action_idx;
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

/// @param game_path Path from root to terminal state (actual game played)
/// @param winner Absolute winner: +1 = P1 wins, -1 = P2 wins, 0 = draw
void SelfPlayEngine::backpropagate_game_outcome(NodePool& pool, const std::vector<uint32_t>& game_path, int winner) {
    if (winner == 0) {
        // Draw - don't record wins or losses
        return;
    }
    
    for (uint32_t idx : game_path) {
        StateNode& node = pool[idx];
        // Convert absolute winner to relative (from current player's perspective)
        // winner = +1 means P1 won, winner = -1 means P2 won
        // If it's P1's turn at this node, P1 winning means +1 (win)
        // If it's P2's turn at this node, P1 winning means -1 (loss)
        float relative_value = node.is_p1_to_move() 
            ? static_cast<float>(winner) 
            : static_cast<float>(-winner);
        
        node.stats.record_game_outcome(relative_value);
    }
}

void SelfPlayEngine::refresh_priors(NodePool& pool, uint32_t node_idx, InferenceServer& server) {
    StateNode& node = pool[node_idx];

    if (!node.has_children()) return;

    // Evaluate the node to get fresh policy logits
    auto future = server.submit_full(&node);
    EvalResult eval = future.get();

    // Collect policy logits for valid moves and apply softmax
    std::vector<std::pair<uint32_t, int>> child_actions;
    float max_logit = -std::numeric_limits<float>::infinity();
    bool flip_policy = !node.is_p1_to_move(); 

    uint32_t child = node.first_child;
    while (child != NULL_NODE) {
        int action_idx = move_to_action_index(pool[child].move);
        action_idx = flip_policy ? flip_action_index(action_idx) : action_idx;
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
    int games_per_worker,
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
    std::atomic<bool> draining{false};
    std::atomic<int> workers_paused{0};
    std::atomic<int> total_active_games{0};
    std::mutex pause_mutex;
    std::condition_variable pause_cv;
    std::condition_variable resume_cv;
    std::atomic<bool> paused{false};
    // Shared root index that workers can read (updated on pool reset)
    std::atomic<uint32_t> current_root{root_idx};

    MultiGameWorkerSync sync{
        pause_requested, draining, workers_paused, total_active_games, 
        current_root, pause_mutex, pause_cv, resume_cv
    };

    int pool_reset_count = 0;

    //220000 is approx constant for # bytes per game per simulationpermove.
    size_t soft_limit_bytes = bounds.max_bytes - (220000ULL * num_workers * games_per_worker * config_.simulations_per_move);
    std::cout << "[SelfPlayEngine] Starting " << num_games << " games with "
              << num_workers << " workers\n";
    std::cout << "[SelfPlayEngine] Memory limit: " << (bounds.max_bytes / (1024*1024*1024)) << " GB, "
              << "soft limit: " << (soft_limit_bytes / static_cast<double>(1024*1024*1024)) << " GB\n";

    auto start_time = std::chrono::steady_clock::now();

    // LAUNCH WORKER THREADS
    std::vector<std::jthread> workers;
    workers.reserve(num_workers);

    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back([this, &pool, &server, &games_remaining, &stats,
                              &sync, collector, games_per_worker](std::stop_token st) {
            run_multi_game_worker(st, pool, server, games_per_worker,
                                  games_remaining, stats, sync, collector);
        });
    }

    // Monitor progress, call checkpoints, and handle memory-bounded resets
    int last_checkpoint = 0;
    while (stats.games_completed.load(std::memory_order_relaxed) < num_games) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Check memory usage
        size_t current_bytes = pool.memory_usage_bytes();

        if (current_bytes >= soft_limit_bytes && collector != nullptr) {
            std::cout << "\n[SelfPlayEngine] Memory at " << std::fixed << std::setprecision(1)
                      << (current_bytes / 1073741824) << " GB - initiating pool reset..." << std::endl;

            // request workers to pause
            draining.store(true, std::memory_order_release);

            // wait for all active games to finish
            auto drain_start = std::chrono::steady_clock::now();
            while (total_active_games.load(std::memory_order_relaxed) > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                auto elapsed = std::chrono::steady_clock::now() - drain_start;
                if (elapsed > std::chrono::seconds(2400)) {
                    std::cerr << "[SelfPlayEngine] Drain timeout after 40 minutes, forcing reset\n";
                    break;
                }
            }

            //they should be paused already basically from above
            {
                std::unique_lock lock(pause_mutex);
                pause_cv.wait_for(lock, std::chrono::seconds(10), [&]() {
                    return workers_paused.load(std::memory_order_relaxed) >= num_workers;
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

            if (stats.games_completed.load(std::memory_order_relaxed) < num_games) {
                //if we have more to go, clear tree, if we're done, leave it so it can be saved
                pool.clear();

                // Reinitialize root node
                uint32_t new_root = pool.allocate();
                pool[new_root].init_root(true);
                current_root.store(new_root, std::memory_order_release);
                root_idx = new_root;

                ++pool_reset_count;
                std::cout << "[SelfPlayEngine] Pool reset #" << pool_reset_count
                          << " complete" << std::endl;
            }

            // Resume workers
            {
                std::lock_guard lock(pause_mutex);
                draining.store(false, std::memory_order_release);
            }
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
// In mcts.cpp:

template<InferenceProvider Inference>
void SelfPlayEngine::run_multi_game_worker(
    std::stop_token stop_token,
    NodePool& pool,
    Inference& inference,
    int games_per_worker,
    std::atomic<int>& games_remaining,
    MultiGameStats& stats,
    MultiGameWorkerSync& sync,
    TrainingSampleCollector* collector)
{
    auto& timers = get_timers();
    std::vector<GameContext> games(games_per_worker);
    int active_games = 0;

    auto check_pause = [&]() -> bool {
        if (stop_token.stop_requested()) {
            return true;
        }

        if (sync.pause_requested.load(std::memory_order_acquire)) {
            sync.workers_paused.fetch_add(1, std::memory_order_relaxed);
            {
                std::unique_lock lock(sync.pause_mutex);
                sync.pause_cv.notify_all();  // Tell main thread we're paused
                sync.resume_cv.wait(lock, [&]() {
                    return !sync.pause_requested.load(std::memory_order_acquire) ||
                           stop_token.stop_requested();
                });
            }
            sync.workers_paused.fetch_sub(1, std::memory_order_relaxed);

            if (stop_token.stop_requested()) return true;

            // Invalidate all games - pool was reset
            std::cerr << "invalidating games\n";
            for (auto& g : games) {
                if (g.active) active_games--;
                g.active = false;
            }
            return true;  // Pause happened, reclaim games
        }
        return false;
    };

    auto try_claim_game = [&](GameContext& g) -> bool {
        // Don't claim new games while draining
        if (sync.draining.load(std::memory_order_acquire)) {
            return false;
        }

        int remaining = games_remaining.fetch_sub(1, std::memory_order_relaxed);
        if (remaining <= 0) {
            games_remaining.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        uint32_t root = sync.current_root.load(std::memory_order_acquire);
        g.current_node = root;
        g.game_path.clear();
        g.game_path.push_back(root);
        g.sample_positions.clear();
        g.num_moves = 0;
        g.active = true;
        g.needs_expansion = false;
        g.needs_mcts = false;
        g.mcts_iterations_done = 0;
        pool[root].set_on_game_path();

        sync.total_active_games.fetch_add(1, std::memory_order_relaxed);
        return true;
    };

    // auto finish_game = [&](GameContext& g, SelfPlayResult result) {
    //     if (result.winner != 0 && !result.error) {
    //         float value = static_cast<float>(result.winner);
    //         for (size_t i = g.game_path.size(); i > 0; --i) {
    //             uint32_t idx = g.game_path[i - 1];
    //             StateNode& n = pool[idx];
    //             float node_value = n.is_p1_to_move() ? value : -value;
    //             n.stats.update(node_value);
    //         }
    //     }
    //
    //     if (collector && result.winner != 0 && !result.error) {
    //         float game_outcome = static_cast<float>(result.winner);
    //     }
    //
    //     stats.add_result(result);
    //     g.active = false;
    //     active_games--;
    //     sync.total_active_games.fetch_sub(1, std::memory_order_relaxed);
    // };
    
    auto finish_game = [&](GameContext& g, SelfPlayResult result) {
        // Backpropagate actual game outcome for training targets
        if (result.winner != 0 && !result.error) {
            backpropagate_game_outcome(pool, g.game_path, result.winner);
        }

        if (result.winner != 0 && !result.error) {
            float value = static_cast<float>(result.winner);
            for (size_t i = g.game_path.size(); i > 0; --i) {
                uint32_t idx = g.game_path[i - 1];
                StateNode& n = pool[idx];
                float node_value = n.is_p1_to_move() ? value : -value;
                n.stats.update(node_value);
            }
        }

        stats.add_result(result);
        g.active = false;
        active_games--;
        sync.total_active_games.fetch_sub(1, std::memory_order_relaxed);
    };

    if (check_pause()) return;

    for (auto& g : games) {
        if (try_claim_game(g)) {
            active_games++;
        }
    }

    struct PendingEval {
        int game_idx;
        std::vector<uint32_t> path;
        uint32_t eval_node_idx;
        std::future<float> future;
    };

    struct PendingExpansion {
        int game_idx;
        uint32_t node_idx;
        std::future<EvalResult> future;
    };

    MCTSConfig mcts_config;
    mcts_config.c_puct = 1.5f;
    mcts_config.fpu = 0.0f;

    while (! stop_token.stop_requested()) {
        if (active_games == 0) {
            if (sync.draining.load(std::memory_order_acquire)) {
                // Enter pause state, wait for drain to complete
                sync.workers_paused.fetch_add(1, std::memory_order_relaxed);
                {
                    std::unique_lock lock(sync.pause_mutex);
                    sync.pause_cv.notify_all();
                    sync.resume_cv.wait(lock, [&]() {
                        return !sync.draining.load(std::memory_order_acquire) ||
                               stop_token.stop_requested();
                    });
                }
                sync.workers_paused.fetch_sub(1, std::memory_order_relaxed);

                if (stop_token.stop_requested()) break;

                // Reclaim games after drain complete
                for (auto& g : games) {
                    if (!g.active && try_claim_game(g)) {
                        active_games++;
                    }
                }
                if (active_games == 0) break;  // No more games to claim
                continue;
            } else {
                // Not draining, try to claim more games
                for (auto& g : games) {
                    if (!g.active && try_claim_game(g)) {
                        active_games++;
                    }
                }
                if (active_games == 0) break;  // No more games available
                continue;
            }
        }

        std::vector<PendingExpansion> pending_expansions;

        // Phase 1: Check terminals and submit expansions
        for (int gi = 0; gi < games_per_worker; ++gi) {
            GameContext& g = games[gi];
            if (!g.active) continue;

            StateNode& node = pool[g.current_node];

            if (node.is_terminal()) {
                int rel = static_cast<int>(node.terminal_value);
                SelfPlayResult result;
                result.winner = node.is_p1_to_move() ? rel : -rel;
                result.num_moves = g.num_moves;
                finish_game(g, result);
                if (try_claim_game(g)) active_games++;
                continue;
            }

            if (node.p1.row == 8) {
                node.set_terminal(-1.0f);
                SelfPlayResult result;
                result.winner = 1;
                result.num_moves = g.num_moves;
                finish_game(g, result);
                if (try_claim_game(g)) active_games++;
                continue;
            }

            if (node.p2.row == 0) {
                node.set_terminal(-1.0f);
                SelfPlayResult result;
                result.winner = -1;
                result.num_moves = g.num_moves;
                finish_game(g, result);
                if (try_claim_game(g)) active_games++;
                continue;
            }

            if (node.p1.fences == 0 && node.p2.fences == 0) {
                int rel = early_terminate_no_fences(node);
                node.set_terminal(static_cast<float>(rel));
                SelfPlayResult result;
                result.winner = node.is_p1_to_move() ? rel : -rel;
                result.num_moves = g.num_moves;
                finish_game(g, result);
                if (try_claim_game(g)) active_games++;
                continue;
            }

            if (g.num_moves >= config_.max_moves_per_game) {
                int rel = early_terminate_no_fences(node);
                int abs = node.is_p1_to_move() ? rel : -rel;
                SelfPlayResult result;
                result.winner = 0;
                result.draw_score = abs * config_.max_draw_reward;
                result.num_moves = g.num_moves;

                float game_value = result.draw_score;
                for (size_t i = g.game_path.size(); i > 0; --i) {
                    StateNode& n = pool[g.game_path[i - 1]];
                    float nv = n.is_p1_to_move() ? game_value : -game_value;
                    n.stats.update(nv);
                }
                stats.add_result(result);
                g.active = false;
                active_games--;
                sync.total_active_games.fetch_sub(1, std::memory_order_relaxed);
                if (try_claim_game(g)) active_games++;
                continue;
            }

            if (!node.is_expanded()) {
                g.needs_expansion = true;
                if constexpr (is_inference_server_v<Inference>) {
                    auto future = inference.submit_full(&node);
                    pending_expansions.push_back({gi, g.current_node, std::move(future)});
                }
            } else {
                g.needs_expansion = false;
                g.needs_mcts = true;
                g.mcts_iterations_done = 0;
            }
        }

        // Force flush for expansions - don't wait for full batch
        if (!pending_expansions.empty()) {
            if constexpr (is_inference_server_v<Inference>) {
                inference.flush();
            }
        }

        for (auto& pe : pending_expansions) {
            GameContext& g = games[pe.game_idx];
            StateNode& node = pool[pe.node_idx];

            EvalResult eval = pe.future.get();

            size_t mutex_idx = pe.node_idx % NUM_EXPANSION_MUTEXES;
            std::lock_guard lock(expansion_mutexes_[mutex_idx]);

            if (!node.is_expanded()) {
                node.stats.set_nn_value(eval.value);
                node.generate_valid_children();

                if (node.has_children()) {
                    bool flip = !node.is_p1_to_move();
                    std::vector<std::pair<uint32_t, int>> child_actions;
                    float max_logit = -std::numeric_limits<float>::infinity();

                    uint32_t child = node.first_child;
                    while (child != NULL_NODE) {
                        int action_idx = move_to_action_index(pool[child].move);
                        action_idx = flip ? flip_action_index(action_idx) : action_idx;
                        if (action_idx >= 0 && action_idx < NUM_ACTIONS) {
                            child_actions.emplace_back(child, action_idx);
                            max_logit = std::max(max_logit, eval.policy[action_idx]);
                        }
                        child = pool[child].next_sibling;
                    }

                    float sum_exp = 0.0f;
                    std::vector<float> exp_logits(child_actions.size());
                    for (size_t i = 0; i < child_actions.size(); ++i) {
                        exp_logits[i] = std::exp(eval.policy[child_actions[i].second] - max_logit);
                        sum_exp += exp_logits[i];
                    }
                    for (size_t i = 0; i < child_actions.size(); ++i) {
                        pool[child_actions[i].first].stats.prior = exp_logits[i] / sum_exp;
                    }
                }
            }

            g.needs_expansion = false;
            g.needs_mcts = node.has_children();
            g.mcts_iterations_done = 0;
        }

        // Phase 2: MCTS iterations
        std::vector<PendingEval> pending_evals;
        int iterations_since_pause_check = 0;
        constexpr int PAUSE_CHECK_INTERVAL = 100;
        constexpr int MAX_PENDING_PER_FLUSH = 256;  // Flush when we have this many

        auto flush_evals = [&]() {
            if (pending_evals.empty()) return;

            if constexpr (is_inference_server_v<Inference>) {
                inference.flush();
            }

            for (auto& pe : pending_evals) {
                float value = pe.future.get();
                pool[pe.eval_node_idx].stats.set_nn_value(value);
                backpropagate(pool, pe.path, value);
                games[pe.game_idx].mcts_iterations_done++;
            }
            pending_evals.clear();
        };

        while (true) {
            if (++iterations_since_pause_check >= PAUSE_CHECK_INTERVAL) {
                iterations_since_pause_check = 0;
                if (check_pause()) break;
            }

            // Check if all MCTS is done
            bool any_needs_mcts = false;
            for (int gi = 0; gi < games_per_worker; ++gi) {
                GameContext& g = games[gi];
                if (g.active && g.needs_mcts && 
                    g.mcts_iterations_done < config_.simulations_per_move) {
                    any_needs_mcts = true;
                    break;
                }
            }
            if (!any_needs_mcts) {
                flush_evals();
                break;
            }

            // Collect one iteration from each game that needs it
            for (int gi = 0; gi < games_per_worker; ++gi) {
                GameContext& g = games[gi];
                if (!g.active || !g.needs_mcts) continue;
                if (g.mcts_iterations_done >= config_.simulations_per_move) continue;

                ScopedTimer timer(timers.mcts_core);

                std::vector<uint32_t> path;
                path.reserve(64);
                uint32_t current = g.current_node;

                while (current != NULL_NODE) {
                    path.push_back(current);
                    StateNode& n = pool[current];
                    if (n.is_terminal()) break;
                    if (!n.has_children()) break;
                    current = select_child_puct(pool, current, mcts_config);
                }

                if (path.empty()) continue;

                uint32_t leaf_idx = path.back();
                StateNode& leaf = pool[leaf_idx];

                if (leaf.is_terminal()) {
                    backpropagate(pool, path, leaf.terminal_value);
                    g.mcts_iterations_done++;
                    continue;
                }

                if (!leaf.is_expanded()) {
                    size_t mutex_idx = leaf_idx % NUM_EXPANSION_MUTEXES;
                    std::lock_guard lock(expansion_mutexes_[mutex_idx]);
                    if (!leaf.is_expanded()) {
                        leaf.generate_valid_children();
                    }
                }

                uint32_t eval_node_idx = leaf_idx;
                if (leaf.has_children()) {
                    uint32_t child = select_child_puct(pool, leaf_idx, mcts_config);
                    if (child != NULL_NODE) {
                        path.push_back(child);
                        eval_node_idx = child;
                    }
                }

                StateNode& eval_node = pool[eval_node_idx];
                if (eval_node.stats.has_nn_value()) {
                    backpropagate(pool, path, eval_node.stats.get_nn_value());
                    g.mcts_iterations_done++;
                } else {
                    if constexpr (is_inference_server_v<Inference>) {
                        auto future = inference.submit(&eval_node);
                        pending_evals.push_back({gi, std::move(path), eval_node_idx, std::move(future)});
                    }
                }
            }

            // Flush if we have enough pending, OR if we collected from all games
            if (pending_evals.size() >= MAX_PENDING_PER_FLUSH || 
                pending_evals.size() >= static_cast<size_t>(active_games)) {
                flush_evals();
            }
        }

        if (sync.pause_requested.load(std::memory_order_acquire)) {
            continue;
        }

        // Phase 3: Make moves
        for (int gi = 0; gi < games_per_worker; ++gi) {
            GameContext& g = games[gi];
            if (!g.active || !g.needs_mcts) continue;
            if (g.mcts_iterations_done < config_.simulations_per_move) continue;

            if (collector) {
                g.sample_positions.push_back(g.current_node);
            }

            float temp = (g.num_moves < config_.temperature_drop_ply) 
                       ? config_.temperature : 0.0f;
            auto policy = compute_policy_from_visits(pool, g.current_node, temp);

            if (policy.empty()) {
                SelfPlayResult result;
                result.error = true;
                result.num_moves = g.num_moves;
                finish_game(g, result);
                if (try_claim_game(g)) active_games++;
                continue;
            }

            Move selected = select_move_from_policy(policy, config_.stochastic && temp > 0);

            uint32_t next = NULL_NODE;
            StateNode& node = pool[g.current_node];
            uint32_t child = node.first_child;
            while (child != NULL_NODE) {
                if (pool[child].move == selected) {
                    next = child;
                    break;
                }
                child = pool[child].next_sibling;
            }

            if (next == NULL_NODE) {
                SelfPlayResult result;
                result.error = true;
                result.num_moves = g.num_moves;
                finish_game(g, result);
                if (try_claim_game(g)) active_games++;
                continue;
            }

            g.current_node = next;
            g.game_path.push_back(next);
            pool[next].set_on_game_path();
            g.num_moves++;
            g.needs_mcts = false;
        }
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

    // Allocate a fresh root for each player's search
    // We maintain the "true" game state by walking down the tree
    uint32_t game_state_idx = root_idx;

    while (result.num_moves < config_.max_moves_per_game) {
        StateNode& game_state = pool[game_state_idx];

        // Check terminal conditions
        if (game_state.is_terminal()) {
            int relative_value = static_cast<int>(game_state.terminal_value);
            result.winner = game_state.is_p1_to_move() ? relative_value : -relative_value;
            break;
        }
        if (game_state.p1.row == 8) { result.winner = 1; break; }
        if (game_state.p2.row == 0) { result.winner = -1; break; }
        if (game_state.p1.fences == 0 && game_state.p2.fences == 0) {
            int rel = early_terminate_no_fences(game_state);
            result.winner = game_state.is_p1_to_move() ? rel : -rel;
            break;
        }

        // Create fresh search root by copying current game state
        uint32_t search_root = pool.allocate();
        StateNode& search_node = pool[search_root];

        // Copy game state (but not tree structure)
        search_node.p1 = game_state.p1;
        search_node.p2 = game_state.p2;
        search_node.fences = game_state.fences;
        search_node.ply = game_state.ply;
        search_node.flags = game_state.flags & StateNode::FLAG_P1_TO_MOVE;  // Only preserve turn
        search_node.first_child = NULL_NODE;
        search_node.next_sibling = NULL_NODE;
        search_node.parent = NULL_NODE;
        search_node.move = Move{};
        search_node.terminal_value = 0.0f;
        search_node.stats.reset();

        // Get the current player's server
        InferenceServer& server = search_node.is_p1_to_move() ? server_p1 : server_p2;

        // Expand and run MCTS with ONLY this player's model
        expand_with_nn_priors(pool, search_root, server);
        run_mcts_iterations(pool, search_root, server, config_.simulations_per_move);

        // Select move using visit counts
        float temp = (result.num_moves < config_.temperature_drop_ply)
                     ? config_.temperature : 0.0f;
        auto policy = compute_policy_from_visits(pool, search_root, temp);

        if (policy.empty()) { 
            result.error = true; 
            break; 
        }

        Move selected = select_move_from_policy(policy, temp > 0.0f);

        // Find or create the child in the GAME state tree (not search tree)
        // This maintains a record of the actual game played
        if (!game_state.is_expanded()) {
            // Need to expand to find the child - use current player's server
            expand_with_nn_priors(pool, game_state_idx, server);
        }

        uint32_t next = NULL_NODE;
        uint32_t child = game_state.first_child;
        while (child != NULL_NODE) {
            if (pool[child].move == selected) {
                next = child;
                break;
            }
            child = pool[child].next_sibling;
        }

        if (next == NULL_NODE) {
            result.error = true;
            break;
        }

        game_state_idx = next;
        result.num_moves++;

        //check if too many moves, declare draw
        if (result.num_moves >= config_.max_moves_per_game) {
            int relative_draw = early_terminate_no_fences(pool[game_state_idx]);
            int absolute_draw = pool[game_state_idx].is_p1_to_move() ? relative_draw : -relative_draw;
            float game_value = absolute_draw * config_.max_draw_reward;
            result.winner = 0;
            result.draw_score = absolute_draw;
            return result;
        }

        //delete search tree and continue, will create a new one next iter for the other model.
        pool.deallocate_subtree(search_root);
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
            result.draw_score = -result.draw_score;
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
template void SelfPlayEngine::run_mcts_iterations<ModelInference>(
    NodePool&, uint32_t, ModelInference&, int);
template void SelfPlayEngine::run_mcts_iterations<InferenceServer>(
    NodePool&, uint32_t, InferenceServer&, int);

template void SelfPlayEngine::expand_with_nn_priors<ModelInference>(
    NodePool&, uint32_t, ModelInference&);
template void SelfPlayEngine::expand_with_nn_priors<InferenceServer>(
    NodePool&, uint32_t, InferenceServer&);

template void SelfPlayEngine::run_multi_game_worker<InferenceServer>(
    std::stop_token, NodePool&, InferenceServer&, int,
    std::atomic<int>&, MultiGameStats&, MultiGameWorkerSync&, TrainingSampleCollector*);

} // namespace qbot
