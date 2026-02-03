#include "mcts.h"

#include "../inference/inference.h"

#include "../util/timer.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace qbot {
constexpr bool batch_requests = false;

// ============================================================================
// SelfPlayEngine Implementation
// ============================================================================

template<InferenceProvider Inference>
void SelfPlayEngine::run_mcts_iterations(NodePool& pool, uint32_t root_idx,
                                          Inference& inference, int iterations) {
    MCTSConfig dummy_config;
    dummy_config.c_puct = 1.5f;
    dummy_config.fpu = 0.0f;

    // Serial MCTS loop
    for (int i = 0; i < iterations; ++i) {
        std::vector<uint32_t> path;
        path.reserve(64);
        uint32_t current = root_idx;

        // Selection: Traverse until terminal or leaf (unexpanded or no children)
        while (current != NULL_NODE) {
            path.push_back(current);
            StateNode& node = pool[current];

            if (node.is_terminal()) break;
            if (!node.is_expanded()) break;
            if (!node.has_children()) break; // Expanded but no moves (should be terminal-ish)

            current = select_child_puct(pool, current, dummy_config);
        }

        if (path.empty()) continue;

        uint32_t leaf_idx = path.back();
        StateNode& leaf = pool[leaf_idx];

        // Terminal or already evaluated leaf
        if (leaf.is_terminal()) {
            backpropagate(pool, path, leaf.terminal_value);
            continue;
        }

        // If not expanded, expand (evaluates, sets priors) and backpropagate the new value
        if (!leaf.is_expanded()) {
            //this fn submits to inference server, generates children in the meantime, and waits for response
            //then assigns the policy to generated children and returns
            expand_with_nn_priors(pool, leaf_idx, inference);

            // If expansion set a value (it should have), backpropagate it
            if (leaf.stats.has_nn_value()) {
                backpropagate(pool, path, leaf.stats.get_nn_value());
            } else if (leaf.has_children()) {
                std::cerr << "Error, leaf has children but inference failed to produce value?" << std::endl;
            } else {
                std::cerr << "Error, leaf has no children but is not terminal and failed to gen valid children" << std::endl;
            }
        } else {
            // We hit an expanded node that has no children (stalemate/blocked)
            // or we somehow re-selected a leaf that was just expanded but not advanced?
            // Just backprop its value again.
            if (leaf.stats.has_nn_value()) {
                backpropagate(pool, path, leaf.stats.get_nn_value());
            }
        }
    }
}

template<InferenceProvider Inference>
void SelfPlayEngine::expand_with_nn_priors(NodePool& pool, uint32_t node_idx, Inference& inference) {
    StateNode& node = pool[node_idx];

    if (node.is_expanded()) return;
    // size_t mutex_idx = node_idx % NUM_EXPANSION_MUTEXES;
    // std::lock_guard lock(expansion_mutexes_[mutex_idx]);

    // Get policy from NN - method differs by inference type
    EvalResult parent_eval;
    if constexpr (is_inference_server_v<Inference>) {
        auto future = inference.submit(&node);
        node.generate_valid_children();
        parent_eval = future.get();
    } else {
        parent_eval = inference.evaluate_node(&node);
        node.generate_valid_children();
    }

    node.stats.set_nn_value(parent_eval.value);

    if (!node.has_children()) return;

    apply_policy_to_children(pool, node_idx, node, parent_eval.policy);
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
    auto future = server.submit(&node);
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
    size_t soft_limit_bytes = bounds.max_bytes - (10000ULL * num_workers * games_per_worker * config_.simulations_per_move);
    std::cout << "[SelfPlayEngine] Starting " << num_games << " games with "
              << num_workers << " workers\n";
    std::cout << "[SelfPlayEngine] Memory limit: " << (bounds.max_bytes / (1024*1024*1024)) << " GB, "
              << "soft limit: " << (soft_limit_bytes / static_cast<double>(1024*1024*1024)) << " GB" << std::endl;

    auto start_time = std::chrono::steady_clock::now();
    std::mutex timer_mutex;
    SelfPlayTimers global_timers;

    // LAUNCH WORKER THREADS
    std::vector<std::jthread> workers;
    workers.reserve(num_workers);

    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back([this, &pool, &server, &games_remaining, &stats,
                              &sync, collector, games_per_worker,
                              &timer_mutex, &global_timers](std::stop_token st) {
            run_multi_game_worker(st, pool, server, games_per_worker,
                                  games_remaining, stats, sync, collector);
            std::lock_guard<std::mutex> lock(timer_mutex);
            global_timers.merge(get_timers());
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

            //save at checkpoints, overwrites each time with no double samples
            if (!samples_file.empty() && collector->size() > 0) {
                auto result = TrainingSampleStorage::save(samples_file, collector->samples());
                if (!result) {
                    std::cerr << "[SelfPlayEngine] Warning: Failed to save intermediate samples\n";
                }
            }
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
    //merge in inference server timers and print all
    global_timers.merge(server.get_inference_timers());
    global_timers.print();
}

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
    std::vector<PerGameContext> games(games_per_worker);
    int active_games = 0;

    // Deferred work queues
    std::vector<DeferredPrune> deferred_prunes;
    std::vector<uint32_t> deferred_tree_roots;  // Trees to fully delete
    std::vector<uint32_t> nodes_to_free;        // Batch of nodes to return to pool

    deferred_prunes.reserve(64);
    deferred_tree_roots.reserve(16);
    nodes_to_free.reserve(4096);

    // Batch return nodes to pool - single CAS to splice free list
    auto flush_freed_nodes = [&]() {
        if (nodes_to_free.empty()) return;

        // Build a local free list: nodes_to_free[0] -> [1] -> ... -> [n-1] -> old_head
        // Then CAS head to point to [0]

        // Clear node state and link into local chain
        for (size_t i = 0; i < nodes_to_free.size(); ++i) {
            StateNode& node = pool[nodes_to_free[i]];
            node.first_child = NULL_NODE;
            node.parent = NULL_NODE;
            // Link to next in our batch (or will link to old head)
            node.next_sibling = (i + 1 < nodes_to_free.size()) ? nodes_to_free[i + 1] : NULL_NODE;
        }

        // Splice into global free list with single CAS
        uint32_t batch_head = nodes_to_free[0];
        uint32_t batch_tail = nodes_to_free.back();
        size_t count = nodes_to_free.size();

        pool.batch_deallocate(batch_head, batch_tail, count);
        nodes_to_free.clear();
    };

    // Execute deferred CPU work
    auto do_deferred_work = [&]() {
        // Collect nodes from prunes
        for (auto& dp : deferred_prunes) {
            prune_siblings_collect(pool, dp.parent_idx, dp.keep_child_idx, nodes_to_free);
        }
        deferred_prunes.clear();

        // Collect nodes from tree deletions
        for (uint32_t root : deferred_tree_roots) {
            collect_subtree_nodes(pool, root, nodes_to_free);
        }
        deferred_tree_roots.clear();

        // Batch return all collected nodes
        flush_freed_nodes();
    };

    auto check_pause = [&]() -> bool {
        if (stop_token.stop_requested()) return true;

        if (sync.pause_requested.load(std::memory_order_acquire)) {
            do_deferred_work();

            sync.workers_paused.fetch_add(1, std::memory_order_relaxed);
            {
                std::unique_lock lock(sync.pause_mutex);
                sync.pause_cv.notify_all();
                sync.resume_cv.wait(lock, [&]() {
                    return !sync.pause_requested.load(std::memory_order_acquire) ||
                           stop_token.stop_requested();
                });
            }
            sync.workers_paused.fetch_sub(1, std::memory_order_relaxed);

            if (stop_token.stop_requested()) return true;

            for (auto& g : games) {
                if (g.active) active_games--;
                g.reset();
            }
            return true;
        }
        return false;
    };

    auto create_game_root = [&](PerGameContext& g) -> bool {
        if (sync.draining.load(std::memory_order_acquire)) return false;

        int remaining = games_remaining.fetch_sub(1, std::memory_order_relaxed);
        if (remaining <= 0) {
            games_remaining.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        uint32_t root = pool.allocate();
        if (root == NULL_NODE) {
            games_remaining.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        pool[root].init_root(true);
        pool[root].set_on_game_path();

        g.root_idx = root;
        g.original_root = root;
        g.game_path.clear();
        g.game_path.push_back(root);
        g.num_moves = 0;
        g.active = true;
        g.mcts_iterations_done = 0;

        sync.total_active_games.fetch_add(1, std::memory_order_relaxed);
        return true;
    };

    auto finish_game = [&](PerGameContext& g, SelfPlayResult result) {
        if (collector && !result.error && result.winner != 0) {
            for (size_t i = 0; i < g.pending_samples.size(); ++i) {
                auto& pending = g.pending_samples[i];

                bool was_p1_turn = (pending.state.flags & StateNode::FLAG_P1_TO_MOVE) != 0;
                float outcome = was_p1_turn 
                    ? static_cast<float>(result.winner) 
                    : static_cast<float>(-result.winner);

                TrainingSample sample;
                sample.state = pending.state;
                sample.policy = pending.policy;

                if (outcome > 0) {
                    sample.wins = 1;
                    sample.losses = 0;
                    sample.value = 1.0f;
                } else {
                    sample.wins = 0;
                    sample.losses = 1;
                    sample.value = -1.0f;
                }

                collector->add_sample_direct(std::move(sample));
            }
        }

        if (g.original_root != NULL_NODE) {
            deferred_tree_roots.push_back(g.original_root);
        }

        stats.add_result(result);
        g.reset();
        active_games--;
        sync.total_active_games.fetch_sub(1, std::memory_order_relaxed);
    };

    MCTSConfig mcts_config;
    mcts_config.c_puct = config_.c_puct;
    mcts_config.fpu = config_.fpu;

    struct PendingExpansion {
        int game_idx;
        // std::vector<uint32_t> path;
        uint32_t leaf_idx;
        std::future<EvalResult> future;
        bool children_generated{false};  // Track if we already generated children
    };

    if (check_pause()) return;

    for (auto& g : games) {
        if (create_game_root(g)) active_games++;
    }

    while (!stop_token.stop_requested()) {
        if (active_games == 0) {
            do_deferred_work();

            if (sync.draining.load(std::memory_order_acquire)) {
                std::cerr << "[Worker] All games finished, entering pause (draining)\n";
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
                std::cerr << "[Worker] Resumed from pause\n";

                if (stop_token.stop_requested()) break;

                for (auto& g : games) {
                    if (!g.active && create_game_root(g)) active_games++;
                }
                if (active_games == 0) break;
                continue;
            } else {
                for (auto& g : games) {
                    if (!g.active && create_game_root(g)) active_games++;
                }
                if (active_games == 0) break;
                continue;
            }
        }

        // ============================================================
        // Phase 1: Check terminals and expand roots if needed
        // ============================================================
        std::vector<PendingExpansion> pending_root_expansions;

        for (int gi = 0; gi < games_per_worker; ++gi) {
            PerGameContext& g = games[gi];
            if (!g.active) continue;

            StateNode& node = pool[g.root_idx];

            if (node.is_terminal()) {
                int rel = static_cast<int>(node.terminal_value);
                SelfPlayResult result;
                result.winner = node.is_p1_to_move() ? rel : -rel;
                result.num_moves = g.num_moves;
                finish_game(g, result);
                if (create_game_root(g)) active_games++;
                continue;
            }

            if (node.p1.row == 8) {
                node.set_terminal(-1.0f);
                SelfPlayResult result;
                result.winner = 1;
                result.num_moves = g.num_moves;
                finish_game(g, result);
                if (create_game_root(g)) active_games++;
                continue;
            }

            if (node.p2.row == 0) {
                node.set_terminal(-1.0f);
                SelfPlayResult result;
                result.winner = -1;
                result.num_moves = g.num_moves;
                finish_game(g, result);
                if (create_game_root(g)) active_games++;
                continue;
            }

            if (node.p1.fences == 0 && node.p2.fences == 0) {
                int rel = early_terminate_no_fences(node);
                node.set_terminal(static_cast<float>(rel));
                SelfPlayResult result;
                result.winner = node.is_p1_to_move() ? rel : -rel;
                result.num_moves = g.num_moves;
                finish_game(g, result);
                if (create_game_root(g)) active_games++;
                continue;
            }

            if (g.num_moves >= config_.max_moves_per_game) {
                int rel = early_terminate_no_fences(node);
                int abs = node.is_p1_to_move() ? rel : -rel;
                SelfPlayResult result;
                result.winner = 0;
                result.draw_score = abs * config_.max_draw_reward;
                result.num_moves = g.num_moves;
                finish_game(g, result);
                if (create_game_root(g)) active_games++;
                continue;
            }

            if (!node.is_expanded()) {
                if constexpr (is_inference_server_v<Inference>) {
                    auto future = inference.submit(&node);
                    pending_root_expansions.push_back({gi, g.root_idx, std::move(future), false});
                }
            } else {
                g.mcts_iterations_done = 0;
                //only apply to root of search, and only when this player has fences left
                if (node.is_p1_to_move() ? node.p1.fences > 0 : node.p2.fences > 0) {
                    add_dirichlet_noise(pool, node.self_index, 0.08f, 0.25f);
                }
            }
        }

        if (!pending_root_expansions.empty()) {
            if constexpr (is_inference_server_v<Inference>) {
                // inference.flush();
            }

            // Generate children while GPU processes - this is the expensive part!
            for (auto& pe : pending_root_expansions) {
                //says leaf_idx but is root of search here
                StateNode& node = pool[pe.leaf_idx];
                if (!node.is_expanded()) {
                    node.generate_valid_children();
                    pe.children_generated = true;
                }
            }

            // Also do deferred prune/delete work
            // do_deferred_work();

            // Now get results and apply priors
            for (auto& pe : pending_root_expansions) {
                PerGameContext& g = games[pe.game_idx];
                StateNode& node = pool[pe.leaf_idx];

                EvalResult eval = pe.future.get();

                node.stats.set_nn_value(eval.value);
                if (!node.is_expanded() && !pe.children_generated) {
                    node.generate_valid_children();
                }
                node.set_expanded();

                if (node.has_children()) {
                    apply_policy_to_children(pool, pe.leaf_idx, node, eval.policy);
                }
                //only apply to root of search, and only when this player has fences left
                if (node.is_p1_to_move() ? node.p1.fences > 0 : node.p2.fences > 0) {
                    add_dirichlet_noise(pool, pe.leaf_idx, 0.08f, 0.25f);
                }

                g.mcts_iterations_done = 0;
            }
        }

        if (check_pause()) continue;

        // ============================================================
        // Phase 2: MCTS simulations with pipelined child generation
        // ============================================================
        std::vector<PendingExpansion> pending_expansions;
        constexpr int BATCH_TARGET = 256;
        // int BATCH_TARGET = games_per_worker * 0.8f;
        int iterations_since_pause_check = 0;
        constexpr int PAUSE_CHECK_INTERVAL = 1000;

        //heap stuff here so it isn't in tight loop
        //path_pool keeps the path each simulation takes, cleared at the start of the sim.
        //these paths aren't that long, don't reserve the inner vector as it will grow to natural length quickly
        std::vector<std::vector<uint32_t>> path_pool;
        path_pool.reserve(games_per_worker);
        for (int gi = 0; gi < games_per_worker; ++gi) {
            path_pool.push_back(std::vector<uint32_t>());
        }

        while (true) {
            if (++iterations_since_pause_check >= PAUSE_CHECK_INTERVAL) {
                iterations_since_pause_check = 0;
                if (check_pause()) break;
            }

            bool any_needs_mcts = false;
            for (int gi = 0; gi < games_per_worker; ++gi) {
                PerGameContext& g = games[gi];
                if (g.active && g.mcts_iterations_done < config_.simulations_per_move) {
                    any_needs_mcts = true;
                    break;
                }
            }

            if (!any_needs_mcts) {
                // Flush remaining
                if (!pending_expansions.empty()) {
                    if constexpr (is_inference_server_v<Inference>) {
                        // inference.flush();
                    }

                    // Generate children while waiting
                    for (auto& pe : pending_expansions) {
                        StateNode& leaf = pool[pe.leaf_idx];
                        if (!leaf.is_expanded()) {
                            leaf.generate_valid_children();
                            pe.children_generated = true;
                        }
                    }
                    do_deferred_work();

                    for (auto& pe : pending_expansions) {
                        PerGameContext& g = games[pe.game_idx];
                        StateNode& leaf = pool[pe.leaf_idx];
                        EvalResult eval = pe.future.get();

                        leaf.stats.set_nn_value(eval.value);
                        if (!leaf.is_expanded() && !pe.children_generated) {
                            leaf.generate_valid_children();
                        }
                        leaf.set_expanded();

                        if (leaf.has_children()) {
                            apply_policy_to_children(pool, pe.leaf_idx, leaf, eval.policy);
                        }
                        backpropagate(pool, path_pool[pe.game_idx], eval.value);
                        g.mcts_iterations_done++;
                    }
                    pending_expansions.clear();
                }
                break;
            }

            // Collect simulations
            for (int gi = 0; gi < games_per_worker; ++gi) {
                PerGameContext& g = games[gi];
                if (!g.active) continue;
                if (g.mcts_iterations_done >= config_.simulations_per_move) continue;

                std::vector<uint32_t>& path = path_pool[gi];
                path.clear();
                // path.reserve(64);
                uint32_t current = g.root_idx;

                while (current != NULL_NODE) {
                    path.push_back(current);
                    StateNode& n = pool[current];

                    if (n.is_terminal()) break;
                    if (!n.is_expanded()) break;
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

                if (leaf.is_expanded()) {
                    if (leaf.stats.has_nn_value()) {
                        backpropagate(pool, path, leaf.stats.get_nn_value());
                        g.mcts_iterations_done++;
                    }
                    continue;
                }

                if constexpr (is_inference_server_v<Inference>) {
                    if constexpr (batch_requests) {
                        //fill the future field when submit as batch
                        pending_expansions.push_back({gi, leaf_idx, {}, false});
                    } else {
                        auto future = inference.submit(&leaf);
                        pending_expansions.push_back({gi, leaf_idx, std::move(future), false});
                    }
                }
            }

            // Process batch
            if (pending_expansions.size() >= BATCH_TARGET ||
                pending_expansions.size() >= static_cast<size_t>(active_games)) {
            // if (pending_expansions.size() > 0) {
            //consider other possible submit triggers
                if (batch_requests) {
                    //get list of node references from pending expansions. bit annoying
                    std::vector<const StateNode*> batch_nodes(pending_expansions.size());
                    std::transform(pending_expansions.begin(), pending_expansions.end(), batch_nodes.begin(),
                       [&](const auto& pe) { return &pool[pe.leaf_idx]; });

                    //submit for inference, pass futures back, should be same order
                    auto futures = inference.submit_batch(batch_nodes);
                    for (size_t i = 0; i < futures.size(); ++i) {
                        pending_expansions[i].future = std::move(futures[i]);
                    }
                }

                // generate children while GPU computes
                // overlaps expensive CPU work with GPU inference
                for (auto& pe : pending_expansions) {
                    StateNode& leaf = pool[pe.leaf_idx];
                    if (!leaf.is_expanded()) {
                        leaf.generate_valid_children();
                        pe.children_generated = true;
                    }
                }

                // Also do deferred cleanup work
                // do_deferred_work();

                // Now collect results - GPU should be done or nearly done
                for (auto& pe : pending_expansions) {
                    PerGameContext& g = games[pe.game_idx];
                    StateNode& leaf = pool[pe.leaf_idx];

                    EvalResult eval = pe.future.get();

                    leaf.stats.set_nn_value(eval.value);
                    if (!leaf.is_expanded() && !pe.children_generated) {
                        leaf.generate_valid_children();
                    }
                    leaf.set_expanded();

                    if (leaf.has_children()) {
                        apply_policy_to_children(pool, pe.leaf_idx, leaf, eval.policy);
                    }

                    backpropagate(pool, path_pool[pe.game_idx], eval.value);
                    // path_pool[pe.game_idx].clear();
                    g.mcts_iterations_done++;
                }
                pending_expansions.clear();
            }
        }

        if (check_pause()) continue;

        // ============================================================
        // Phase 3: Select moves, defer pruning, advance games
        // ============================================================
        for (int gi = 0; gi < games_per_worker; ++gi) {
            PerGameContext& g = games[gi];
            if (!g.active) continue;
            if (g.mcts_iterations_done < config_.simulations_per_move) continue;

            StateNode& root = pool[g.root_idx];

            if (collector && root.has_children()) {
                PerGameContext::PendingSample pending;
                pending.state = extract_compact_state(root);
                pending.policy = extract_visit_distribution(pool, g.root_idx);
                g.pending_samples.push_back(std::move(pending));
            }

            float temp = (g.num_moves < config_.temperature_drop_ply)
                       ? config_.temperature : 0.0f;
            auto policy = compute_policy_from_visits(pool, g.root_idx, temp);

            if (policy.empty()) {
                SelfPlayResult result;
                result.error = true;
                result.num_moves = g.num_moves;
                finish_game(g, result);
                if (create_game_root(g)) active_games++;
                continue;
            }

            Move selected = select_move_from_policy(policy, config_.stochastic && temp > 0);

            uint32_t next = NULL_NODE;
            uint32_t child = root.first_child;
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
                if (create_game_root(g)) active_games++;
                continue;
            }

            // Defer pruning
            deferred_prunes.push_back({g.root_idx, next});

            g.game_path.push_back(next);
            pool[next].set_on_game_path();
            g.root_idx = next;
            g.num_moves++;
            g.mcts_iterations_done = 0;
        }
        //test out doing it immediately
        do_deferred_work();
    }

    // Final cleanup
    do_deferred_work();
    for (auto& g : games) {
        if (g.active && g.original_root != NULL_NODE) {
            pool.deallocate_subtree(g.original_root);
            sync.total_active_games.fetch_sub(1, std::memory_order_relaxed);
        }
    }
}

inline void prune_siblings(NodePool& pool, uint32_t parent_idx, uint32_t keep_child_idx) {
    StateNode& parent = pool[parent_idx];

    uint32_t child = parent.first_child;
    while (child != NULL_NODE) {
        uint32_t next_sib = pool[child].next_sibling;

        if (child != keep_child_idx) {
            pool.deallocate_subtree(child);
        }

        child = next_sib;
    }

    // Update parent to only have the kept child
    parent.first_child = keep_child_idx;
    pool[keep_child_idx].next_sibling = NULL_NODE;
}

/// Collect nodes to free into a vector without returning to pool yet
inline void collect_subtree_nodes(NodePool& pool, uint32_t root_idx, std::vector<uint32_t>& out) {
    if (root_idx == NULL_NODE) return;

    size_t start = out.size();
    out.push_back(root_idx);

    while (start < out.size()) {
        uint32_t idx = out[start++];
        uint32_t child = pool[idx].first_child;
        while (child != NULL_NODE) {
            out.push_back(child);
            child = pool[child].next_sibling;
        }
    }
}

/// Prune siblings, collecting freed nodes instead of deallocating immediately
inline void prune_siblings_collect(NodePool& pool, uint32_t parent_idx, uint32_t keep_child_idx, 
                                    std::vector<uint32_t>& freed_nodes) {
    StateNode& parent = pool[parent_idx];

    uint32_t child = parent.first_child;
    while (child != NULL_NODE) {
        uint32_t next_sib = pool[child].next_sibling;
        if (child != keep_child_idx) {
            collect_subtree_nodes(pool, child, freed_nodes);
        }
        child = next_sib;
    }

    parent.first_child = keep_child_idx;
    pool[keep_child_idx].next_sibling = NULL_NODE;
}

inline void apply_policy_to_children(
    NodePool& pool,
    uint32_t node_idx,
    const StateNode& node,
    const std::array<float, NUM_ACTIONS>& policy_logits)
{
    bool flip_policy = !node.is_p1_to_move();

    constexpr int MAX_MOVES = 256;
    uint32_t child_indices[MAX_MOVES];
    float child_logits[MAX_MOVES]; 
    int count = 0;

    float max_logit = -std::numeric_limits<float>::infinity();

    uint32_t child = node.first_child;
    while (child != NULL_NODE) {
        int action_idx = move_to_action_index(pool[child].move);
        if (flip_policy) {
            action_idx = flip_action_index(action_idx);
        }

        if (action_idx >= 0 && action_idx < NUM_ACTIONS) {
            float val = policy_logits[action_idx];
            if (count < MAX_MOVES) {
                child_indices[count] = child;
                child_logits[count] = val;
                count++;
            }

            max_logit = std::max(max_logit, val);
        }
        child = pool[child].next_sibling;
    }

    if (count == 0) return;

    float sum_exp = 0.0f;

    // Reuse child_logits array to store the exponentiated values
    // to avoid a third array or second pass re-calculation.
    for (int i = 0; i < count; ++i) {
        float e = std::exp(child_logits[i] - max_logit);
        child_logits[i] = e; // Store exp value back in place
        sum_exp += e;
    }

    float scale = 1.0f / sum_exp;
    for (int i = 0; i < count; ++i) {
        pool[child_indices[i]].stats.prior = child_logits[i] * scale;
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

// ============================================================================
// CompEngine Implementation
// ============================================================================

template<InferenceProvider Inference>
void CompEngine::expand_with_nn_priors(NodePool& pool, uint32_t node_idx, Inference& inference) {
    StateNode& node = pool[node_idx];

    size_t mutex_idx = node_idx % NUM_EXPANSION_MUTEXES;
    std::lock_guard lock(expansion_mutexes_[mutex_idx]);

    if (node.is_expanded()) return;

    EvalResult eval;
    if constexpr (is_inference_server_v<Inference>) {
        eval = inference.submit(&node).get();
    } else {
        eval = inference.evaluate_node(&node);
    }

    node.stats.set_nn_value(eval.value);
    node.generate_valid_children();

    if (!node.has_children()) return;

    apply_policy_to_children(pool, node_idx, node, eval.policy);
}

void CompEngine::backpropagate(NodePool& pool, const std::vector<uint32_t>& path, float value) {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        pool[*it].stats.update(value);
        value = -value;
    }
}

template<InferenceProvider Inference>
void CompEngine::run_iterations(NodePool& pool, uint32_t root_idx, 
                                 Inference& inference, int iterations) {
    MCTSConfig mcts_config;
    mcts_config.c_puct = config_.c_puct;
    mcts_config.fpu = config_.fpu;

    // Serial loop - no pending queue, expansion is blocking
    for (int i = 0; i < iterations; ++i) {
        std::vector<uint32_t> path;
        path.reserve(64);
        uint32_t cur = root_idx;

        // Selection: descend tree via PUCT
        while (cur != NULL_NODE) {
            path.push_back(cur);
            StateNode& n = pool[cur];

            if (n.is_terminal()) break;
            if (!n.is_expanded()) break;
            if (!n.has_children()) break; // Expanded but no moves

            cur = select_child_puct(pool, cur, mcts_config);
        }

        if (path.empty()) continue;

        uint32_t leaf = path.back();
        StateNode& leaf_node = pool[leaf];

        // Terminal: backprop known value
        if (leaf_node.is_terminal()) {
            backpropagate(pool, path, leaf_node.terminal_value);
            continue;
        }

        // Expansion
        if (!leaf_node.is_expanded()) {
            expand_with_nn_priors(pool, leaf, inference);
            // After expansion, we have the node's value (from NN). Backpropagate it immediately.
            if (leaf_node.stats.has_nn_value()) {
                backpropagate(pool, path, leaf_node.stats.get_nn_value());
            }
        } else {
            // Node was already expanded (but had no children or re-hit?)
            // If it has a value, backprop it.
            if (leaf_node.stats.has_nn_value()) {
                backpropagate(pool, path, leaf_node.stats.get_nn_value());
            }
        }
    }
}

template<InferenceProvider Inference>
Move CompEngine::search(NodePool& pool, uint32_t root_idx, Inference& inference) {
    return search(pool, root_idx, inference, config_.num_simulations);
}

template<InferenceProvider Inference>
Move CompEngine::search(NodePool& pool, uint32_t root_idx, Inference& inference, 
                        int num_simulations) {
    if (root_idx == NULL_NODE) return Move{};

    StateNode& root = pool[root_idx];
    if (root.is_terminal()) return Move{};

    // Ensure root is expanded with priors
    expand_with_nn_priors(pool, root_idx, inference);
    if (!root.has_children()) return Move{};

    //Run MCTS (future: split across num_threads)
    run_iterations(pool, root_idx, inference, num_simulations);

    // Select move with highest visit count (temperature = 0, deterministic)
    return select_move_from_policy(compute_policy_from_visits(pool, root_idx, 0.0f), false);
}

// Explicit instantiations
// competitive engine
template void CompEngine::expand_with_nn_priors<ModelInference>(NodePool&, uint32_t, ModelInference&);
template void CompEngine::expand_with_nn_priors<InferenceServer>(NodePool&, uint32_t, InferenceServer&);
template void CompEngine::run_iterations<ModelInference>(NodePool&, uint32_t, ModelInference&, int);
template void CompEngine::run_iterations<InferenceServer>(NodePool&, uint32_t, InferenceServer&, int);
template Move CompEngine::search<ModelInference>(NodePool&, uint32_t, ModelInference&);
template Move CompEngine::search<InferenceServer>(NodePool&, uint32_t, InferenceServer&);
template Move CompEngine::search<ModelInference>(NodePool&, uint32_t, ModelInference&, int);
template Move CompEngine::search<InferenceServer>(NodePool&, uint32_t, InferenceServer&, int);

//selfplay engine
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
