#include "Global.h"
#include "Game.h"
#include "storage.h"
#include "utility.h"
#include <chrono>

void Game::parallel_self_play(const std::string& checkpoint_file, int num_threads, int games_per_checkpoint) {
    std::vector<std::thread> workers;
    std::atomic<int> games_completed{0};
    std::atomic<int> total_simulations{0};

    // Statistics tracking
    std::atomic<int> p1_wins{0};
    std::atomic<int> p2_wins{0};
    std::atomic<int> total_moves{0};

    auto start_time = std::chrono::steady_clock::now();

    // Launch worker threads
    for (int i = 0; i < num_threads; i++) {
        workers.emplace_back([this, i, &games_completed, &total_simulations, 
                             &p1_wins, &p2_wins, &total_moves, 
                             checkpoint_file, games_per_checkpoint]() {
            self_play_worker(i, games_completed, total_simulations, 
                           p1_wins, p2_wins, total_moves,
                           checkpoint_file, games_per_checkpoint);
        });
    }

    // Main thread monitors progress and handles checkpointing
    while (!stop_training) {
        std::this_thread::sleep_for(std::chrono::seconds(30));

        int games = games_completed.load();
        if (games > 0 && games % games_per_checkpoint == 0) {
            // Pause workers during checkpoint
            active_threads = 0;
            while (active_threads.load() != num_threads) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            // Save checkpoint
            int nodes_saved = save_tree(root, checkpoint_file);

            // Output statistics
            auto current_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::minutes>(current_time - start_time);

            std::cout << "\n=== Checkpoint at " << games << " games ===\n";
            std::cout << "Time elapsed: " << duration.count() << " minutes\n";
            std::cout << "Total simulations: " << total_simulations.load() << "\n";
            std::cout << "P1 win rate: " << (100.0 * p1_wins / games) << "%\n";
            std::cout << "P2 win rate: " << (100.0 * p2_wins / games) << "%\n";
            std::cout << "Avg game length: " << (1.0 * total_moves / games) << " moves\n";
            std::cout << "Saved " << nodes_saved << " nodes\n";
            output_tree_stats(root);

            // Resume workers
            active_threads = num_threads;
        }
    }

    // Clean up
    for (auto& worker : workers) {
        worker.join();
    }
}

void Game::self_play_worker(int thread_id, 
                           std::atomic<int>& games_completed,
                           std::atomic<int>& total_simulations,
                           std::atomic<int>& p1_wins,
                           std::atomic<int>& p2_wins,
                           std::atomic<int>& total_moves,
                           const std::string& checkpoint_file,
                           int games_per_checkpoint) {

    while (!stop_training) {
        // Check if we should pause for checkpointing
        if (active_threads.load() == 0) {
            active_threads.fetch_add(1);
            while (active_threads.load() != num_threads) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        // Play one complete game
        play_single_game(thread_id, total_simulations, p1_wins, p2_wins, total_moves);
        games_completed.fetch_add(1);
    }
}

void Game::play_single_game(int thread_id, 
                           std::atomic<int>& total_simulations,
                           std::atomic<int>& p1_wins,
                           std::atomic<int>& p2_wins,
                           std::atomic<int>& total_moves) {
    StateNode* current = root;
    int moves_in_game = 0;

    while (!current->game_over()) {
        // Run MCTS from current position
        const int SIMULATIONS_PER_MOVE = 800;

        for (int i = 0; i < SIMULATIONS_PER_MOVE; i++) {
            parallel_mcts_iteration(current, thread_id);
            total_simulations.fetch_add(1);
        }

        // Select best move based on visit counts
        StateNode* best_child = select_best_child(current);
        if (!best_child) break;

        current = best_child;
        moves_in_game++;
    }

    // Update statistics
    total_moves.fetch_add(moves_in_game);
    int winner = current->game_over();
    if (winner == 1) p1_wins.fetch_add(1);
    else if (winner == -1) p2_wins.fetch_add(1);
}

void Game::parallel_mcts_iteration(StateNode* root, int thread_id) {
    std::vector<StateNode*> path;
    StateNode* current = root;

    // Track nodes we've applied virtual loss to
    std::vector<StateNode*> virtual_loss_nodes;

    // SELECTION PHASE with virtual loss
    while (current->children.size() > 0 && !current->game_over()) {
        path.push_back(current);

        // Apply virtual loss to encourage exploration diversity
        apply_virtual_loss(current, virtual_loss_nodes);

        // Select best child considering virtual loss
        StateNode* best = nullptr;
        double best_ucb = -std::numeric_limits<double>::infinity();

        for (auto& child : current->children) {
            double ucb = child.UCB(current->turn);
            if (ucb > best_ucb) {
                best_ucb = ucb;
                best = &child;
            }
        }

        if (!best) break;
        current = best;
    }

    // EXPANSION PHASE
    if (!current->game_over() && current->visits > 0) {
        // Only one thread should expand at a time
        bool should_expand = false;
        {
            std::lock_guard<std::mutex> lock(tree_mutex);
            if (current->children.empty()) {
                current->generate_valid_children();
                should_expand = true;
            }
        }

        if (should_expand && !current->children.empty()) {
            // Select random unexplored child
            std::vector<StateNode*> unexplored;
            for (auto& child : current->children) {
                if (child.visits == 0) {
                    unexplored.push_back(&child);
                }
            }

            if (!unexplored.empty()) {
                std::uniform_int_distribution<> dist(0, unexplored.size() - 1);
                current = unexplored[dist(get_rng())];
                path.push_back(current);
            }
        }
    }

    // EVALUATION PHASE
    double value;
    if (current->game_over()) {
        value = current->game_over();  // 1 for p1 win, -1 for p2 win
    } else if (model_loaded && thread_id % 4 == 0) {  // Use NN for 25% of evaluations
        value = model.evaluate_node(current);
    } else {
        // Traditional simulation
        StateNode* terminal = current->play_out();
        value = terminal->game_over();
        if (!value) {
            value = pathfinding(terminal) > 0 ? 1 : -1;
        }
    }

    // BACKPROPAGATION PHASE lock score/visits
	{
		std::lock_guard<std::mutex> lock(backprop_mutex);
		for (auto* node : path) {
			node->visits += 1;
			node->score += value;
		}
	}

    remove_virtual_loss(virtual_loss_nodes);
}

void Game::apply_virtual_loss(StateNode* node, std::vector<StateNode*>& virtual_loss_nodes) {
    // Apply virtual loss to discourage other threads from taking same path
	std::lock_guard<std::mutex> lock(backprop_mutex);
    for (int i = 0; i < VIRTUAL_LOSS_COUNT; i++) {
        node->visits += 1;
		node->score += VIRTUAL_LOSS_VALUE;
    }
    virtual_loss_nodes.push_back(node);
}

void Game::remove_virtual_loss(const std::vector<StateNode*>& virtual_loss_nodes) {
	std::lock_guard<std::mutex> lock(backprop_mutex);
    for (auto* node : virtual_loss_nodes) {
        // Remove the virtual visits and losses
        node->visits -= VIRTUAL_LOSS_COUNT;
		node->score -= VIRTUAL_LOSS_COUNT * VIRTUAL_LOSS_VALUE;
    }
}

StateNode* Game::select_best_child(StateNode* node) {
    if (node->children.empty()) return nullptr;

    StateNode* best = nullptr;
    int max_visits = -1;

    // Temperature-based selection for exploration
    float temperature = (node->ply < 30) ? 1.0f : 0.1f;

    if (temperature > 0.1f) {
        // Probabilistic selection based on visit counts
        std::vector<float> probs;
        float sum = 0;

        for (auto& child : node->children) {
            float prob = std::pow(child.visits, 1.0f / temperature);
            probs.push_back(prob);
            sum += prob;
        }

        // Normalize and sample
        for (auto& p : probs) p /= sum;
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        int selected = dist(get_rng());

        return &(node->children[selected]);
    } else {
        // Deterministic selection (most visited)
        for (auto& child : node->children) {
            if (child.visits > max_visits) {
                max_visits = child.visits;
                best = &child;
            }
        }
        return best;
    }
}
