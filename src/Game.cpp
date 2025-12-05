
#include "Global.h"
#include "Game.h"
#include "storage.h"
#include "utility.h"
#include <chrono>


#ifndef NOVIZ
#include "Python.h"
#endif

using namespace std::chrono;

void Game::run_game(){
	int depth = 1;
	std::time_t time = std::time(0);
	bool gameOver = false;

	StateNode* currState = this->root;
	//currState->generate_valid_children();
	//build_tree(currState, depth, time);

#ifndef NOVIZ
	//make it so all threads can use python interpreter. mutex to govern GIL
	Py_Initialize();
	    PyRun_SimpleString(R"(
import sys
import os
sys.path.append(os.path.join(sys.prefix, 'lib', 'python3.12', 'site-packages'))
		)");
	// Build the name object
	PyObject* sysPath = PySys_GetObject("path");
	PyList_Append(sysPath, PyUnicode_FromString("Quoridor-Online/quoridor/client"));
	PyList_Append(sysPath, PyUnicode_FromString("Quoridor-Online"));
#endif

	Move player_move;
	while(!gameOver){
		StateNode* next_state;
		if(!currState->turn){
			if(currState->p1.numFences == 0 && currState->p2.numFences == 0){
				vector<Move> testpath;
				int diff = pathfinding(currState, testpath, true);
			}
			int ret = currState->get_best_move();
			next_state = &(currState->children[ret]);
			if (next_state != nullptr){
				next_state->print_node();
				if (next_state->game_over()){
					std::cout << "!!! qbot Wins !!!" << std::endl;
					gameOver = true;
					break;
				}
				player_move = get_player_move(next_state);	
			}else{
				std::cout << "best_node is nullptr";
				return;
			}
		} else { //should only be run if we start game with player move
			next_state = currState;
			next_state->print_node();
			player_move = get_player_move(next_state);
		}

		//player_move = Move(viz_retval);
		std::cout << player_move << "\n";

		//find the move in the tree, create it if it doesn't exist yet.
		bool move_exists = false;
		for (auto &child : next_state->children){
			if (player_move == child.move){
				currState = &child;
				move_exists = true;
			}
		}
		if (!move_exists){
			move_exists = test_and_add_move(next_state, player_move);
			for (auto &child : next_state->children){
				if (player_move == child.move){
					currState = &child;
				}
			}
		}
		//std::cout << "AFTER playermove";
		currState->print_node();

		if (!move_exists){
			std::cout << "Could not produce player move state\n";
			return;
		} else if (currState->game_over()) {
			//check win condition
			std::cout << "!!! Player1 Wins !!!" << std::endl;
			gameOver = true;
		}

	}
#ifndef NOVIZ
	Py_Finalize();
#endif
}

Move Game::get_player_move(StateNode* currState){
	Move player_move;
	std::string viz_retval;
	time_point<system_clock> start, end;
	while( true ){ //input loop, gathers input from player until it receives a valid move
		start = system_clock::now();
		viz_retval = currState->visualize_gamestate();
		end = system_clock::now();
		duration<double> length = end - start;
		if (length.count() < .2){//trys to detect error state so it doesnt open python instances on loop
			std::cout << length.count() << " too short, loop detected, terminating" << std::endl;
			//TERMINATE PROGRAM
			//If this is short it means the python is erroring out and returning too quick
			std::terminate();
			return Move();
		}

		if (viz_retval == "error"){
			std::cout << "Viz gamestate returned error" << std::endl;
			break;
		}
		try {
			player_move = Move(viz_retval);
			//basic checking done in Move constructor, but we need gamestate specific
			std::vector<Move> valid_moves;
			currState->generate_valid_moves(valid_moves);
			for (auto move : valid_moves){
				if (move.type == 'p')
					cout << move  << std::endl;
				if (move == player_move){//gen_valid_moves doesn't check pathfinding so we still need that
					// vector<Move> path; // just to debug pathfinding
					// pathfinding(currState, path, true);
					int pf = pathfinding(currState, move);
					cout << "Relative length to goal " << pf << std::endl;
					if (player_move.type == 'f' &&  pf == -999){
						std::cout << "Attempted move blocks a player from their goal" << std::endl;	
					}else{
						return player_move;
					}
					break;
				}
			}
			std::cout << "Attempted move doesn't appear to be possible" << std::endl;
		}catch(InvalidMoveException& e ){
			std::cout << e.what() << std::endl;
			std::cout << "Enter a valid move" << std::endl;
		}
	}
	return Move();//bad state, should have returned earlier

}

void Game::self_play(const int timeout) {
	bool gameOver = false;
	StateNode* nextState, *currState = this->root;
	std::time_t startTime = std::time(0);
	std::time_t currTime = startTime;


	while(!(currTime - startTime > timeout)) {
		while(!gameOver) {
			int ret = currState->get_best_move();
			nextState = &(currState->children[ret]);
			if (nextState != nullptr){
				nextState->print_node();
			}else{
				std::cout << "best_node is nullptr";
				return;
			}

			currState = nextState;
			// currState->print_node();
			gameOver = currState->game_over();
		}

		gameOver = false;
		currTime = std::time(0);
	}
}

void Game::self_play(const std::string& checkpoint_file, const int games_per_checkpoint) {
    StateNode* currState = nullptr;
    int games_played = 0;
    std::time_t startTime = std::time(0);

    while (true) { // Run indefinitely
        // Start new game from root
        currState = this->root;
        bool gameOver = false;

        // Play a single game
        while (!gameOver) {
            int ret = currState->get_best_move();
            StateNode* nextState = &(currState->children[ret]);

            if (nextState != nullptr) {
                nextState->print_node(); // Keep this for debugging/monitoring
                gameOver = nextState->game_over();
                currState = nextState;
            } else {
                std::cout << "best_node is nullptr" << std::endl;
                break;
            }
        }

        games_played++;

        // Checkpoint the tree periodically
        if (games_played % games_per_checkpoint == 0) {
            int nodes_saved = save_tree(root, checkpoint_file);
            std::cout << "Checkpoint after " << games_played << " games. Saved " 
                     << nodes_saved << " nodes to " << checkpoint_file << std::endl;

            // Output some statistics
            std::time_t currentTime = std::time(0);
            double minutes = difftime(currentTime, startTime) / 60.0;
            std::cout << "Training stats after " << minutes << " minutes:" << std::endl;
            std::cout << "Games played: " << games_played << std::endl;
            std::cout << "Games per hour: " << (float)(games_played) / minutes << std::endl;
            // output_tree_stats(root);
        }

        // Optional: Break if some training target is met
        // if (some_condition) break;
    }
}

void Game::train_alpha(const std::string& checkpoint_file, int iterations_before_training) {
    if (!model_loaded) {
        std::cerr << "Error: No model loaded for AlphaGo Zero training" << std::endl;
        return;
    }

    std::time_t start_time = std::time(0);
    int total_iterations = 0;
    int games_played = 0;

    output_tree_stats(root);

    while (total_iterations < iterations_before_training) {
        StateNode* current = root;
        std::vector<StateNode*> game_path;

        while (!current->game_over()) {
            game_path.push_back(current);

            // Ensure children exist
            if (current->children.empty()) {
                current->generate_valid_children();
            }

            // Run MCTS simulations from this position
            const int SIMULATIONS_PER_MOVE = 800;
            for (int sim = 0; sim < SIMULATIONS_PER_MOVE; sim++) {
                perform_mcts_simulation(current);
                total_iterations++;

                // Check if we should stop and train
                if (total_iterations >= iterations_before_training) {
                    break;
                }
            }

            // Select move based on visit counts
            current = select_move_by_visits(current, games_played);

            if (total_iterations >= iterations_before_training) {
                break;
            }
        }

        // Backpropagate actual game outcome through the game path
        if (current->game_over()) {
            float game_outcome = current->game_over();

            // Update all nodes in the game path with true outcome
            for (auto* node : game_path) {
                node->visits++;
                node->score += game_outcome;
            }
        }

        games_played++;

        // Periodic logging
        if (games_played % 100 == 0) {
            std::time_t current_time = std::time(0);
            double minutes = difftime(current_time, start_time) / 60.0;

            std::cout << "Progress: " << games_played << " games, " 
                     << total_iterations << " iterations in " 
                     << minutes << " minutes" << std::endl;
            output_tree_stats(root);
        }
    }

    // Save the enhanced tree for training
    std::cout << "\nReached training checkpoint after " << total_iterations << " iterations" << std::endl;
	output_tree_stats(root);

    int nodes_saved = save_tree(root, checkpoint_file);
    std::cout << "Saved " << nodes_saved << " nodes to " << checkpoint_file << std::endl;
    std::cout << "Ready for external model training." << std::endl;
}

// Helper function for MCTS simulation with value-only model
void Game::perform_mcts_simulation(StateNode* root) {
    StateNode* current = root;
    std::vector<StateNode*> path;

    // Selection phase - traverse tree using UCB
    while (!current->children.empty()) {
        path.push_back(current);

        // Find best child using UCB (no prior probabilities)
        StateNode* best_child = nullptr;
        double best_ucb = -std::numeric_limits<double>::infinity();

        for (auto& child : current->children) {
            double ucb = child.UCB(current->turn);
            if (ucb > best_ucb) {
                best_ucb = ucb;
                best_child = &child;
            }
        }

        if (!best_child) break;
        current = best_child;
    }

    // Expansion phase
    if (!current->game_over() && current->visits > 0) {
        // Only expand if we've visited this node before
        if (current->children.empty()) {
            current->generate_valid_children();
        }

        // Select a random unexplored child for evaluation
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

    // Evaluation phase - use neural network
    double value = model.evaluate_node(current);

    // Backup phase - propagate value up the tree
    for (auto* node : path) {
        node->visits++;
        node->score += value;
    }
}

// Helper to select moves during self-play
StateNode* Game::select_move_by_visits(StateNode* node, int game_number) {
    if (node->children.empty()) return node;

    // Temperature for exploration (explore more in early moves)
    float temperature = (node->ply < 30) ? 1.0f : 0.1f;

    if (temperature > 0.1f && game_number % 10 != 0) {  // Explore 90% of games
        // Sample proportional to visit counts
        std::vector<float> visit_probs;
        float sum = 0;

        for (const auto& child : node->children) {
            float prob = std::pow(child.visits, 1.0f / temperature);
            visit_probs.push_back(prob);
            sum += prob;
        }

        // Normalize
        for (auto& p : visit_probs) p /= sum;

        // Sample
        std::discrete_distribution<> dist(visit_probs.begin(), visit_probs.end());
        int selected = dist(get_rng());

        return &(node->children[selected]);
    } else {
        // Select deterministically (most visited)
        StateNode* best = nullptr;
        int max_visits = -1;

        for (auto& child : node->children) {
            if (child.visits > max_visits) {
                max_visits = child.visits;
                best = &child;
            }
        }

        return best ? best : &(node->children[0]);
	}
}

void Game::better_self_play(const std::string& checkpoint_file, const int games_per_checkpoint) {
    int games_played = 0;
    std::time_t startTime = std::time(0);
    double total_moves = 0;
    int player1_wins = 0, player2_wins = 0;
    StateNode* best_root = this->root;

    while (true) { // Run continuously until stopped externally
        // Start a new game from initial position
        StateNode* curr_state = best_root;
        bool game_over = false;
        int moves_in_game = 0;

        // Play a single game until termination
        while (!game_over) {
            // For each move, run MCTS to determine the best action
            // This is where AlphaGo Zero would use its neural network for evaluation and prior probabilities

            // Generate all valid children if they don't exist yet
            if (curr_state->children.size() == 0) {
                curr_state->generate_valid_children();
            }

            // Set a fixed number of MCTS simulations per move
            const int num_simulations = 100; // AlphaGo Zero used 1600 simulations per move

            // Run multiple MCTS simulations from current state
            for (int i = 0; i < num_simulations; i++) {
                // Select, expand, simulate, and backpropagate in the tree
                StateNode* leaf = curr_state;

                // SELECTION - find a leaf node using UCB
                while (leaf->children.size() > 0 && !leaf->game_over()) {
                    // Find child with highest UCB value
                    StateNode* best_child = nullptr;
                    double best_ucb = -std::numeric_limits<double>::infinity();

                    for (auto& child : leaf->children) {
                        double ucb = child.UCB(leaf->turn);
                        if (ucb > best_ucb) {
                            best_ucb = ucb;
                            best_child = &child;
                        }
                    }

                    if (best_child == nullptr) break; // Safety check
                    leaf = best_child;
                }

                // EXPANSION - if leaf is not terminal and has no children
                if (!leaf->game_over() && leaf->children.size() == 0) {
					leaf->generate_valid_children();
                    // leaf->generate_random_child();

                    // If children were created, select one of them
                    if (leaf->children.size() > 0) {
                        std::uniform_int_distribution<> dist(0, leaf->children.size() - 1);
                        leaf = &(leaf->children[dist(get_rng())]);
                    }
                }

                // SIMULATION - play out from the leaf node
                leaf->play_out();
            }

            StateNode* best_node = nullptr;
			int mod = curr_state->turn ? 1 : -1;
			double best_score = -std::numeric_limits<double>::infinity();

            for (auto& child : curr_state->children) {
                if (mod * child.score > best_score) {
                    best_score = child.score;
                    best_node = &child;
                }
            }

            if (best_node == nullptr) {
                std::cerr << "Error: No valid moves found during self-play\n";
                break;
            }

            // Make the selected move
            curr_state = best_node;
            moves_in_game++;

            game_over = curr_state->game_over();
        }

        // Game has finished
        games_played++;
        total_moves += moves_in_game;

        // Record win statistics
        if (curr_state->game_over() == 1) {
            player1_wins++;
        } else if (curr_state->game_over() == -1) {
            player2_wins++;
        }

        // Checkpoint the tree periodically
        if (games_played % games_per_checkpoint == 0) {
            int nodes_saved = save_tree(root, checkpoint_file);

            // Output statistics
            std::time_t currentTime = std::time(0);
            double minutes = difftime(currentTime, startTime) / 60.0;

            std::cout << "=== Training Progress Report ===\n";
            std::cout << "Games played: " << games_played << "\n";
            std::cout << "Time elapsed: " << minutes << " minutes\n";
            std::cout << "Games per hour: " << (games_played / (minutes / 60.0)) << "\n";
            std::cout << "Average game length: " << (total_moves / games_played) << " moves\n";
            std::cout << "Player 1 win rate: " << (100.0 * player1_wins / games_played) << "%\n";
            std::cout << "Player 2 win rate: " << (100.0 * player2_wins / games_played) << "%\n";
            std::cout << "Saved " << nodes_saved << " nodes to " << checkpoint_file << "\n";
            std::cout << "==============================\n";
        }
    }
}

//recursive function for building the state tree
void build_tree(StateNode* currState, int depth, std::time_t starttime){
	std::time_t time = std::time(0) - starttime;
	if (depth >= MAXDEPTH || time > MAXTIME)
		return;

	//places a list of valid, evaluated moves in currState->children
	currState->generate_valid_children();

	//removes moves I don't think are good
	//currState->prune_children();

	std::vector<StateNode>::iterator it;
	for (it = currState->children.begin(); it != currState->children.end(); it++) {
		//must be careful when passing pointer to element in vector, if vector resizes this pointer will become invalid.
		//should be okay here but in case any unexplained phenoms happen, check this
		//this comment was prophetic^ too bad i didn't remember it was here til i changed the structure to fix this lol
		//it should be good now as list doesn't change element positions
		build_tree(&(*it), depth + 1, starttime);
	}	
}

//receives opposing move and sets the resulting gamestate as this->root
bool Game::receive_opposing_move(){
	return false;
}

//outputs move
bool Game::play_move(){
	return false;
}

Game::Game(StateNode* start, int num_threads, std::string model_file){
	this->root = start;
	this->ply = start->ply;
	this->num_threads = num_threads;

	if (!model_file.empty()) {
        this->model = ModelInference(model_file);
		this->model_loaded = true;
	}
}


//
//
//PARALLEL ATTEMPT
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
        const int SIMULATIONS_PER_MOVE = 100;

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
