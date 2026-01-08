/// QuoridorMain.cpp
/// Main entry point for the Quoridor MCTS bot
///
/// Supports:
/// - Interactive play against human
/// - Self-play training
/// - Tree loading/saving
/// - Neural network model loading

#include "../tree/node_pool.h"
#include "../util/storage.h"
#include "../util/gui_client.h"
#include "../util/pathfinding.h"
#include "../util/timer.h"
#include "../search/mcts.h"
#include "Game.h"

#ifdef QBOT_ENABLE_INFERENCE
#include "../inference/inference.h"
#endif

#include <csignal>

#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

namespace po = boost::program_options;

namespace qbot {

// Global pointer for signal handler to stop training gracefully
static MCTSEngine* g_mcts_engine = nullptr;

void signal_handler(int signum) {
    if (signum == SIGINT || signum == SIGTERM) {
        std::cout << "\nReceived signal " << signum << ", stopping...\n";
        if (g_mcts_engine) {
            g_mcts_engine->stop();
        }
    }
}

enum class RunMode {
    Interactive,    // Play against human
    Train,          // Tree building (rollout-based)
    SelfPlay,       // AlphaZero-style self-play with NN
    Arena,          // Model vs model evaluation
};

/// Application configuration parsed from command line
struct Config {
    RunMode mode = RunMode::Interactive;
    int player = 1;                              // Which player the bot plays as (1 or 2)
    int num_threads = 4;                         // Number of threads for MCTS
    std::string save_file = "tree.qbot";         // File to save tree to
    std::string load_file;                       // File to load tree from
    std::string model_file;                      // Neural network model file
    bool verbose = false;                        // Verbose output
    int games_per_checkpoint = 100;              // Games between checkpoints in training

    // Training-specific options
    int training_iterations = 10000;             // Total training iterations
    int simulations_per_move = 800;              // MCTS simulations per move

    // Self-play specific options
    int num_games = 1000;                        // Number of self-play games
    int batch_size = 256;                        // Inference batch size for GPU
    float temperature = 1.0f;                    // Softmax temperature
    int temperature_drop_ply = 30;               // Ply to drop temperature to 0

    // Arena specific options
    std::string candidate_model;                 // Candidate model to evaluate
    std::string best_model_path = "model/current_best.pt";  // Path to save best model
    int arena_games = 100;                       // Number of arena games
    float win_threshold = 0.55f;                 // Threshold to replace best model

    [[nodiscard]] static std::optional<Config> from_args(int argc, char* argv[]);
    void print(std::ostream& os = std::cout) const;
};

std::optional<Config> Config::from_args(int argc, char* argv[]) {
    Config config;

    po::options_description desc("Quoridor MCTS Bot - Play and train a Quoridor AI");
    desc.add_options()
        ("help,h", "Show this help message")
        ("player,p", po::value<int>(&config.player)->default_value(1),
            "Player number (1 or 2, bot plays as this player)")
        ("threads,t", po::value<int>(&config.num_threads)->default_value(4),
            "Number of threads for MCTS")
        ("save,s", po::value<std::string>(&config.save_file)->default_value("tree.qbot"),
            "File to save tree/checkpoints to")
        ("load,l", po::value<std::string>(&config.load_file)->default_value(""),
            "Load tree from file")
        ("model,m", po::value<std::string>(&config.model_file)->default_value(""),
            "Load neural network model from file")
        ("train,b", po::bool_switch(),
            "Run in tree-building mode (rollout-based)")
        ("selfplay", po::bool_switch(),
            "Run in self-play mode (NN-only, complete games)")
        ("arena", po::bool_switch(),
            "Run arena mode: evaluate candidate vs current model")
        ("candidate", po::value<std::string>(&config.candidate_model)->default_value(""),
            "Candidate model path (for arena mode)")
        ("best-model", po::value<std::string>(&config.best_model_path)->default_value("model/current_best.pt"),
            "Path to save best model (for arena mode)")
        ("arena-games", po::value<int>(&config.arena_games)->default_value(100),
            "Number of games for arena evaluation")
        ("win-threshold", po::value<float>(&config.win_threshold)->default_value(0.55f),
            "Win rate threshold to replace best model")
        ("verbose,v", po::bool_switch(&config.verbose),
            "Enable verbose output")
        ("games,g", po::value<int>(&config.num_games)->default_value(1000),
            "Number of self-play games")
        ("batch-size,B", po::value<int>(&config.batch_size)->default_value(256),
            "Inference batch size for GPU (higher = better GPU utilization)")
        ("iterations,i", po::value<int>(&config.training_iterations)->default_value(10000),
            "Training iterations (tree-building mode)")
        ("simulations,n", po::value<int>(&config.simulations_per_move)->default_value(800),
            "MCTS simulations per move")
        ("temperature", po::value<float>(&config.temperature)->default_value(1.0f),
            "Softmax temperature for move selection")
        ("temp-drop", po::value<int>(&config.temperature_drop_ply)->default_value(30),
            "Ply to drop temperature to 0");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << "\n\n" << desc << "\n";
        return std::nullopt;
    }

    if (vm.count("help")) {
        std::cout << "Usage: quoridor [OPTIONS]\n\n" << desc << "\n";
        return std::nullopt;
    }

    if (vm["arena"].as<bool>()) {
        config.mode = RunMode::Arena;
    } else if (vm["selfplay"].as<bool>()) {
        config.mode = RunMode::SelfPlay;
    } else if (vm["train"].as<bool>()) {
        config.mode = RunMode::Train;
    } else {
        config.mode = RunMode::Interactive;
    }

    // Validate
    if (config.player != 1 && config.player != 2) {
        std::cerr << "Error: Player must be 1 or 2\n";
        return std::nullopt;
    }
    if (config.num_threads < 1) {
        std::cerr << "Error: Thread count must be at least 1\n";
        return std::nullopt;
    }

    return config;
}

void Config::print(std::ostream& os) const {
    os << "Configuration:\n";
    const char* mode_str = "Interactive";
    if (mode == RunMode::Train) mode_str = "Tree-building";
    else if (mode == RunMode::SelfPlay) mode_str = "Self-play";
    else if (mode == RunMode::Arena) mode_str = "Arena";
    os << "  Mode:           " << mode_str << "\n";
    os << "  Player:         " << player << "\n";
    os << "  Threads:        " << num_threads << "\n";
    os << "  Save file:      " << save_file << "\n";
    if (!load_file.empty()) {
        os << "  Load file:      " << load_file << "\n";
    }
    if (!model_file.empty()) {
        os << "  Model file:     " << model_file << "\n";
    }
    os << "  Verbose:        " << (verbose ? "yes" : "no") << "\n";
    if (mode == RunMode::Train) {
        os << "  Iterations:     " << training_iterations << "\n";
        os << "  Sims/move:      " << simulations_per_move << "\n";
    }
    if (mode == RunMode::SelfPlay) {
        os << "  Games:          " << num_games << "\n";
        os << "  Sims/move:      " << simulations_per_move << "\n";
        os << "  Temperature:    " << temperature << "\n";
        os << "  Temp drop ply:  " << temperature_drop_ply << "\n";
    }
    if (mode == RunMode::Arena) {
        os << "  Current model:  " << model_file << "\n";
        os << "  Candidate:      " << candidate_model << "\n";
        os << "  Best model:     " << best_model_path << "\n";
        os << "  Arena games:    " << arena_games << "\n";
        os << "  Win threshold:  " << (win_threshold * 100) << "%\n";
        os << "  Sims/move:      " << simulations_per_move << "\n";
    }
}

// ============================================================================
// Tree Loading/Saving
// ============================================================================

/// Load or create a new game tree
[[nodiscard]] std::pair<std::unique_ptr<NodePool>, uint32_t>
initialize_tree(const Config& config) {
    if (!config.load_file.empty() && std::filesystem::exists(config.load_file)) {
        std::cout << "Loading tree from: " << config.load_file << "\n";

        auto start = std::chrono::steady_clock::now();
        auto result = TreeStorage::load(config.load_file);
        auto end = std::chrono::steady_clock::now();
        auto sec = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

        if (result.has_value()) {
            auto& loaded = *result;
            std::cout << "  Loaded " << loaded.pool->allocated() << " nodes in " << sec << " s\n";
            return {std::move(loaded.pool), loaded.root};
        } else {
            std::cerr << "Warning: Failed to load tree: "
                      << to_string(result.error()) << "\n";
            std::cerr << "  Starting with fresh tree\n";
        }
    }

    // Create new tree with root node
    auto pool = std::make_unique<NodePool>();
    bool p1_starts = (config.player == 1);
    uint32_t root = pool->allocate(Move{}, NULL_NODE, p1_starts);

    std::cout << "Created new tree (player " << config.player << " to move first)\n";
    return {std::move(pool), root};
}

/// Save the current tree
bool save_tree(const NodePool& pool, uint32_t root, const std::string& path) {
    std::cout << "Saving tree to: " << path << "\n";

    auto start = std::chrono::steady_clock::now();
    auto result = TreeStorage::save(path, pool, root);
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

    if (result.has_value()) {
        auto file_size = std::filesystem::file_size(path);
        std::cout << "  Saved " << pool.allocated() << " nodes ("
                  << file_size / 1024 << " KB) in " << ms << " s\n";
        return true;
    } else {
        std::cerr << "Error saving tree: " << to_string(result.error()) << "\n";
        return false;
    }
}

/// Save only nodes that were part of actual games (pruned)
bool save_tree_pruned(const NodePool& pool, uint32_t root, const std::string& path) {
    std::cout << "Saving pruned tree to: " << path << "\n";

    auto start = std::chrono::steady_clock::now();
    auto result = TreeStorage::save_pruned(path, pool, root);
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

    if (result.has_value()) {
        auto file_size = std::filesystem::file_size(path);
        std::cout << "  Pruned " << pool.allocated() << " -> " << result.value()
                  << " nodes (" << file_size / 1024 << " KB) in " << ms << " s\n";
        return true;
    } else {
        std::cerr << "Error saving tree: " << to_string(result.error()) << "\n";
        return false;
    }
}

// ============================================================================
// Game Modes
// ============================================================================

/// Run interactive game against human
int run_interactive(const Config& config,
                    std::unique_ptr<NodePool> pool,
                    uint32_t root) {
    // Create Game instance from the pool
    Game game(std::move(pool), root);

    // Connect to GUI
    GUIClient gui;
    GUIClient::Config gui_config;
    gui_config.host = "localhost";
    gui_config.port = 8765;
    gui_config.connect_timeout_ms = 5000;
    gui_config.read_timeout_ms = 60000;  // 60 seconds for human to make a move

    if (!gui.connect(gui_config)) {
        std::cerr << "Failed to connect to GUI: " << gui.last_error() << "\n";
        std::cerr << "Please ensure the GUI server is running.\n";
        return 1;
    }
    std::cout << "Connected to GUI!\n\n";

#ifdef QBOT_ENABLE_INFERENCE
    // Load model for position evaluation if specified
    std::unique_ptr<ModelInference> model;
    if (!config.model_file.empty()) {
        std::cout << "Loading model from: " << config.model_file << "\n";
        model = std::make_unique<ModelInference>(config.model_file);
        if (model->is_ready()) {
            std::cout << "Model loaded successfully!\n\n";
            game.set_model(model.get());
        } else {
            std::cerr << "Warning: Failed to load model, using Q-values instead\n\n";
            model.reset();
        }
    }
#endif

    // Initialize the root node with starting game state
    StateNode& root_node = game.pool()[root];
    root_node.init_root(true);  // P1 (human) starts

    gui.send_start("Human", "Bot");

    uint32_t current_idx = root;
    bool game_over = false;

    while (!game_over) {
        StateNode& current = game.pool()[current_idx];

        // Evaluate current position - use model if available, otherwise Q-value
        float score = current.stats.Q(0.0f);
#ifdef QBOT_ENABLE_INFERENCE
        if (model && model->is_ready()) {
            score = model->evaluate_node(&current).value;
        }
#endif
        gui.send_gamestate(current, current.is_p1_to_move() ? 0 : 1, score);

        // Check for terminal state
        int result = current.game_over();
        if (result != 0) {
            game_over = true;
            if (result == 1) {
                std::cout << "\n*** HUMAN WINS! ***\n\n";
            } else {
                std::cout << "\n*** BOT WINS! ***\n\n";
            }
            break;
        }

        if (current.is_p1_to_move()) {
            bool valid_move = false;
            while (!valid_move) {
                auto gui_move_opt = gui.request_move(0);
                if (!gui_move_opt) {
                    std::cerr << "Failed to get move from GUI: " << gui.last_error() << "\n";
                    return 1;
                }

                auto gui_move = *gui_move_opt;
                if (gui_move.type == GUIClient::GUIMove::Type::Quit) {
                    std::cout << "Quit received from GUI.\n";
                    return 0;
                }

                Move move = GUIClient::to_engine_move(gui_move);

                // Validate the move
                if (!current.is_move_valid(move)) {
                    std::cout << "ILLEGAL MOVE! ";
                    if (move.is_pawn()) {
                        std::cout << "Pawn move to (" << static_cast<int>(move.row())
                                  << ", " << static_cast<int>(move.col()) << ")";
                    } else {
                        std::cout << (move.is_horizontal() ? "H" : "V") << " fence at ("
                                  << static_cast<int>(move.row()) << ", "
                                  << static_cast<int>(move.col()) << ")";
                    }
                    std::cout << " is not legal. Try again.\n";

                    // Re-send gamestate to GUI so it can reset and let user try again
                    gui.send_gamestate(current, 0, score);
                    continue;
                }

                if (move.is_fence()) {
                    Pathfinder& pf = get_pathfinder();
                    if (!pf.check_paths_with_fence(current, move)) {
                        std::cout << "ILLEGAL MOVE! Fence would block a player's path to goal. Try again.\n";
                        gui.send_gamestate(current, 0, score);
                        continue;
                    }
                }

                valid_move = true;
                move.print("Human");

                // Find or create the child node for this move
                uint32_t next_idx = current.find_or_create_child(move);
                if (next_idx == NULL_NODE) {
                    std::cerr << "Error: Failed to create child node for move\n";
                    return 1;
                }
                current_idx = next_idx;
            }
        } else {
            // Bot's turn - Game::select_best_move handles model if set
            Move move = game.select_best_move(current_idx);
            if (!move.is_valid()) {
                std::cerr << "Error: Bot has no valid moves\n";
                return 1;
            }

            move.print("Bot");

            uint32_t next_idx = current.find_or_create_child(move);
            if (next_idx == NULL_NODE) {
                std::cerr << "Error: Failed to create child node for bot's move\n";
                return 1;
            }
            current_idx = next_idx;
        }
    }

    // Send final state to GUI
    StateNode& final_state = game.pool()[current_idx];
    gui.send_gamestate(final_state, -1, final_state.terminal_value);

    // Optionally save tree
    if (!config.save_file.empty() && config.verbose) {
        save_tree(game.pool(), root, config.save_file);
    }

    return 0;
}

/// Run MCTS tree building / training
int run_training(const Config& config,
                 std::unique_ptr<NodePool> pool,
                 uint32_t root) {
    std::cout << "\n=== MCTS Training Mode ===\n";

    // Initialize the root node only if it's a fresh tree (not loaded)
    StateNode& root_node = (*pool)[root];
    if (!root_node.is_expanded() && !root_node.has_children()) {
        root_node.init_root(true);  // P1 starts
    }

    // Configure MCTS engine
    MCTSConfig mcts_config;
    mcts_config.num_threads = config.num_threads;
    mcts_config.checkpoint_interval_seconds = 300;  // 5 minutes
    mcts_config.checkpoint_path = config.save_file;

    if (!config.model_file.empty()) {
        mcts_config.model_path = config.model_file;
    }

    // Create engine
    MCTSEngine engine(mcts_config);

    // Set up signal handler for graceful shutdown
    g_mcts_engine = &engine;
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Start training
    std::cout << "Starting MCTS tree building...\n";
    std::cout << "Press Ctrl+C to stop and save.\n\n";

    engine.start_training(*pool, root);

    // Wait for training to complete (runs until signal)
    while (engine.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Clean up signal handler
    g_mcts_engine = nullptr;
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);

    // Final save
    std::cout << "\nSaving final tree...\n";
    if (!config.save_file.empty()) {
        save_tree(*pool, root, config.save_file);
    }

    std::cout << "Training complete!\n";
    return 0;
}

#ifdef QBOT_ENABLE_INFERENCE

/// Play a single game between two models
/// @param model_p1 Model playing as Player 1
/// @param model_p2 Model playing as Player 2
/// @param simulations MCTS simulations per move
/// @return 1 if P1 wins, -1 if P2 wins, 0 if draw
int play_arena_game(ModelInference& model_p1, ModelInference& model_p2, int simulations) {
    // Create fresh tree for this game
    NodePool pool;
    uint32_t root = pool.allocate(Move{}, NULL_NODE, true);
    pool[root].init_root(true);  // P1 starts

    // Configure for deterministic play (temperature=0)
    SelfPlayConfig sp_config;
    sp_config.simulations_per_move = simulations;
    sp_config.temperature = 0.0f;  // Deterministic
    sp_config.stochastic = false;

    uint32_t current = root;
    int num_moves = 0;

    while (current != NULL_NODE && num_moves < 500) {
        StateNode& node = pool[current];

        // Check for terminal
        if (node.is_terminal()) {
            return (node.terminal_value > 0) ? 1 : -1;
        }
        if (node.p1.row == 8) return 1;
        if (node.p2.row == 0) return -1;

        // Select which model to use based on whose turn it is
        ModelInference& current_model = node.is_p1_to_move() ? model_p1 : model_p2;

        // Expand if needed
        if (!node.is_expanded()) {
            // Get policy from parent's perspective using current player's model
            EvalResult parent_eval = current_model.evaluate_node(&node);

            node.generate_valid_children();
            if (!node.has_children()) return 0;  // Error

            // Set priors from parent's policy logits (softmax over legal moves only)
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

            if (!child_actions.empty()) {
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
        }

        // Run MCTS iterations
        for (int i = 0; i < simulations; ++i) {
            // Selection
            std::vector<uint32_t> path;
            uint32_t cur = current;
            while (cur != NULL_NODE) {
                path.push_back(cur);
                StateNode& n = pool[cur];
                if (n.is_terminal() || !n.has_children()) break;

                // Select best child
                uint32_t best_child = NULL_NODE;
                float best_score = -std::numeric_limits<float>::infinity();
                uint32_t parent_visits = n.stats.visits.load(std::memory_order_relaxed);

                uint32_t c = n.first_child;
                while (c != NULL_NODE) {
                    float score = puct_score(pool[c].stats, parent_visits, 1.5f, 0.0f);
                    if (score > best_score) {
                        best_score = score;
                        best_child = c;
                    }
                    c = pool[c].next_sibling;
                }
                cur = best_child;
            }

            if (path.empty()) continue;

            // Evaluate leaf
            uint32_t leaf = path.back();
            StateNode& leaf_node = pool[leaf];
            float p1_value;  // Value from P1's absolute perspective
            if (leaf_node.is_terminal()) {
                // terminal_value is already from P1's perspective
                p1_value = leaf_node.terminal_value;
            } else {
                // Evaluate with the model for the player at this position
                ModelInference& eval_model = leaf_node.is_p1_to_move() ? model_p1 : model_p2;
                float leaf_value = eval_model.evaluate_node(&leaf_node).value;
                // leaf_value is from current player's perspective, convert to P1's
                p1_value = leaf_node.is_p1_to_move() ? leaf_value : -leaf_value;
            }

            // Backpropagate using P1's absolute perspective
            for (size_t j = path.size(); j > 0; --j) {
                StateNode& n = pool[path[j-1]];
                // Convert P1's perspective to this node's current player's perspective
                float node_value = n.is_p1_to_move() ? p1_value : -p1_value;
                n.stats.update(node_value);
            }
        }

        // Select best move (deterministic - highest visit count)
        uint32_t best_child = NULL_NODE;
        uint32_t best_visits = 0;
        uint32_t child = node.first_child;
        while (child != NULL_NODE) {
            uint32_t visits = pool[child].stats.visits.load(std::memory_order_relaxed);
            if (visits > best_visits) {
                best_visits = visits;
                best_child = child;
            }
            child = pool[child].next_sibling;
        }

        if (best_child == NULL_NODE) return 0;  // Error

        current = best_child;
        num_moves++;
    }

    return 0;  // Draw (too many moves)
}

/// Run arena evaluation: candidate model vs current model
int run_arena(const Config& config) {
    std::cout << "\n=== Arena Mode ===\n";

    // Validate inputs
    if (config.model_file.empty()) {
        std::cerr << "Error: Current model required (-m)\n";
        return 1;
    }
    if (config.candidate_model.empty()) {
        std::cerr << "Error: Candidate model required (--candidate)\n";
        return 1;
    }

    // Load both models
    std::cout << "Loading current model: " << config.model_file << "\n";
    ModelInference current_model(config.model_file);
    if (!current_model.is_ready()) {
        std::cerr << "Error: Failed to load current model\n";
        return 1;
    }

    std::cout << "Loading candidate model: " << config.candidate_model << "\n";
    ModelInference candidate_model(config.candidate_model);
    if (!candidate_model.is_ready()) {
        std::cerr << "Error: Failed to load candidate model\n";
        return 1;
    }

    std::cout << "\nStarting arena evaluation...\n";
    std::cout << "  Games: " << config.arena_games << "\n";
    std::cout << "  Sims/move: " << config.simulations_per_move << "\n";
    std::cout << "  Win threshold: " << (config.win_threshold * 100) << "%\n\n";

    int candidate_wins = 0;
    int current_wins = 0;
    int draws = 0;

    auto start_time = std::chrono::steady_clock::now();

    // Play games, alternating which model is P1/P2
    for (int game = 0; game < config.arena_games; ++game) {
        bool candidate_is_p1 = (game % 2 == 0);
        int result;

        if (candidate_is_p1) {
            result = play_arena_game(candidate_model, current_model, config.simulations_per_move);
            // result > 0 means P1 (candidate) won
            if (result > 0) ++candidate_wins;
            else if (result < 0) ++current_wins;
            else ++draws;
        } else {
            result = play_arena_game(current_model, candidate_model, config.simulations_per_move);
            // result > 0 means P1 (current) won
            if (result > 0) ++current_wins;
            else if (result < 0) ++candidate_wins;
            else ++draws;
        }

        // Progress update
        if ((game + 1) % 10 == 0 || game == config.arena_games - 1) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            double games_per_sec = elapsed > 0 ? static_cast<double>(game + 1) / elapsed : 0;

            float candidate_rate = static_cast<float>(candidate_wins) / (game + 1 - draws);
            std::cout << "[" << (game + 1) << "/" << config.arena_games << "] "
                      << "Candidate: " << candidate_wins << ", Current: " << current_wins
                      << ", Draws: " << draws
                      << " | Win rate: " << std::fixed << std::setprecision(1)
                      << (candidate_rate * 100) << "%"
                      << " | " << std::setprecision(2) << games_per_sec << " g/s\n";
        }
    }

    // Calculate final win rate (excluding draws)
    int decisive_games = candidate_wins + current_wins;
    float candidate_win_rate = decisive_games > 0
        ? static_cast<float>(candidate_wins) / decisive_games
        : 0.0f;

    std::cout << "\n=== Arena Results ===\n";
    std::cout << "  Candidate wins: " << candidate_wins << "\n";
    std::cout << "  Current wins:   " << current_wins << "\n";
    std::cout << "  Draws:          " << draws << "\n";
    std::cout << "  Win rate:       " << std::fixed << std::setprecision(1)
              << (candidate_win_rate * 100) << "%\n";

    // Check if candidate should replace current
    if (candidate_win_rate >= config.win_threshold) {
        std::cout << "\nCandidate wins! (" << (candidate_win_rate * 100)
                  << "% >= " << (config.win_threshold * 100) << "%)\n";
        std::cout << "Promoting candidate to: " << config.best_model_path << "\n";

        // Create directory if needed
        std::filesystem::path best_path(config.best_model_path);
        if (best_path.has_parent_path()) {
            std::filesystem::create_directories(best_path.parent_path());
        }

        // Copy candidate to best model path
        try {
            std::filesystem::copy_file(config.candidate_model, config.best_model_path,
                                       std::filesystem::copy_options::overwrite_existing);
            std::cout << "Model promoted successfully!\n";
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error copying model: " << e.what() << "\n";
            return 1;
        }
    } else {
        std::cout << "\nCurrent model retained. (" << (candidate_win_rate * 100)
                  << "% < " << (config.win_threshold * 100) << "%)\n";
    }

    return 0;
}

/// Run AlphaZero-style self-play with NN evaluation
int run_selfplay(const Config& config,
                 std::unique_ptr<NodePool> pool,
                 uint32_t root) {
    std::cout << "\n=== Self-Play Mode (NN-only) ===\n";

    // Model is required for self-play
    if (config.model_file.empty()) {
        std::cerr << "Error: Self-play mode requires a model file (-m)\n";
        return 1;
    }

    // Initialize the root node only if it's a fresh tree
    StateNode& root_node = (*pool)[root];
    if (!root_node.is_expanded() && !root_node.has_children()) {
        root_node.init_root(true);  // P1 starts
    }

    // Configure self-play engine
    SelfPlayConfig sp_config;
    sp_config.simulations_per_move = config.simulations_per_move;
    sp_config.temperature = config.temperature;
    sp_config.temperature_drop_ply = config.temperature_drop_ply;
    sp_config.stochastic = true;

    SelfPlayEngine engine(sp_config);

    // Determine samples file path (same as save_file but with .qsamples extension)
    std::string samples_file;
    if (!config.save_file.empty()) {
        std::filesystem::path p(config.save_file);
        samples_file = (p.parent_path() / p.stem()).string() + ".qsamples";
    }

    // Create training sample collector
    TrainingSampleCollector collector;
    collector.reserve(config.num_games * 100);  // ~100 moves per game estimate

    std::cout << "Starting self-play...\n";
    std::cout << "  Games:       " << config.num_games << "\n";
    std::cout << "  Threads:     " << config.num_threads << "\n";
    std::cout << "  Sims/move:   " << config.simulations_per_move << "\n";
    std::cout << "  Temperature: " << config.temperature << " (drops to 0 at ply "
              << config.temperature_drop_ply << ")\n";
    std::cout << "  Tree file:   " << config.save_file << "\n";
    std::cout << "  Samples:     " << samples_file << "\n\n";

    // Use multi-threaded path if threads > 1
    if (config.num_threads > 1) {
        // Create inference server for batched GPU access
        InferenceServerConfig server_config;
        server_config.batch_size = config.batch_size;
        server_config.max_wait_ms = 0.1;
        InferenceServer server(config.model_file, server_config);
        server.start();

        auto checkpoint_callback = [&config, &pool, root](const MultiGameStats& stats, const NodePool& p) {
            int completed = stats.games_completed.load(std::memory_order_relaxed);
            int p1 = stats.p1_wins.load(std::memory_order_relaxed);
            int p2 = stats.p2_wins.load(std::memory_order_relaxed);
            int d = stats.draws.load(std::memory_order_relaxed);
            int moves = stats.total_moves.load(std::memory_order_relaxed);

            std::cout << "[" << completed << "/" << config.num_games << "] "
                      << "P1: " << p1 << ", P2: " << p2 << ", Draw: " << d
                      << " | Nodes: " << p.allocated()
                      << " | Avg moves: " << (completed > 0 ? moves / completed : 0) << "\n";

            // Checkpoint save
            if (!config.save_file.empty() && completed % config.games_per_checkpoint == 0) {
                save_tree(const_cast<NodePool&>(p), root, config.save_file);
            }
        };

        MultiGameStats stats;
        engine.run_multi_game(
            *pool, root, server,
            config.num_games, config.num_threads,
            stats, checkpoint_callback, 10);

        // Capture server stats before stopping
        size_t total_requests = server.total_requests();
        size_t total_batches = server.total_batches();
        double avg_batch_size = total_batches > 0
            ? static_cast<double>(total_requests) / total_batches : 0.0;

        server.stop();

        // Final save (pruned to only game-path nodes)
        if (!config.save_file.empty()) {
            save_tree_pruned(*pool, root, config.save_file);
        }

        // Extract training samples from the completed game tree
        auto tree_samples = extract_samples_from_tree(*pool, root);
        for (auto& sample : tree_samples) {
            collector.add_sample_direct(std::move(sample));
        }

        int p1_wins = stats.p1_wins.load(std::memory_order_relaxed);
        int p2_wins = stats.p2_wins.load(std::memory_order_relaxed);
        int draws = stats.draws.load(std::memory_order_relaxed);
        int total_moves = stats.total_moves.load(std::memory_order_relaxed);

        std::cout << "\nSelf-play complete!\n";
        std::cout << "  P1 wins: " << p1_wins << " (" << (100.0 * p1_wins / config.num_games) << "%)\n";
        std::cout << "  P2 wins: " << p2_wins << " (" << (100.0 * p2_wins / config.num_games) << "%)\n";
        std::cout << "  Draws:   " << draws << "\n";
        std::cout << "  Avg moves per game: " << (total_moves / config.num_games) << "\n";
        std::cout << "  Samples: " << collector.size() << "\n";
        std::cout << "  Avg batch size: " << std::fixed << std::setprecision(1) << avg_batch_size
                  << " (" << total_requests << " requests / " << total_batches << " batches)\n";

    } else {
        // Single-threaded path (original implementation)
        std::cout << "Loading model from: " << config.model_file << "\n";
        ModelInference model(config.model_file);
        if (!model.is_ready()) {
            std::cerr << "Error: Failed to load model\n";
            return 1;
        }
        std::cout << "Model loaded successfully!\n\n";

        int p1_wins = 0, p2_wins = 0, draws = 0;
        int total_moves = 0;
        auto start_time = std::chrono::steady_clock::now();

        for (int game = 0; game < config.num_games; ++game) {
            SelfPlayResult result = engine.self_play(*pool, root, model, &collector);

            if (result.winner == 1) ++p1_wins;
            else if (result.winner == -1) ++p2_wins;
            else ++draws;
            total_moves += result.num_moves;

            if ((game + 1) % 10 == 0 || game == config.num_games - 1) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                double games_per_sec = elapsed > 0 ? static_cast<double>(game + 1) / elapsed : 0;

                std::cout << "[" << (game + 1) << "/" << config.num_games << "] "
                          << "P1: " << p1_wins << ", P2: " << p2_wins << ", Draw: " << draws
                          << " | Nodes: " << pool->allocated()
                          << " | Samples: " << collector.size()
                          << " | Avg moves: " << (total_moves / (game + 1))
                          << " | " << std::fixed << std::setprecision(1) << games_per_sec << " g/s\n";
            }

            if ((game + 1) % config.games_per_checkpoint == 0 && !config.save_file.empty()) {
                save_tree(*pool, root, config.save_file);
            }
        }

        // Final save (pruned to only game-path nodes)
        if (!config.save_file.empty()) {
            save_tree_pruned(*pool, root, config.save_file);
        }

        std::cout << "\nSelf-play complete!\n";
        std::cout << "  P1 wins: " << p1_wins << " (" << (100.0 * p1_wins / config.num_games) << "%)\n";
        std::cout << "  P2 wins: " << p2_wins << " (" << (100.0 * p2_wins / config.num_games) << "%)\n";
        std::cout << "  Draws:   " << draws << "\n";
        std::cout << "  Avg moves per game: " << (total_moves / config.num_games) << "\n";
    }

    // Save training samples
    if (!samples_file.empty() && collector.size() > 0) {
        auto result = TrainingSampleStorage::save(samples_file, collector.samples());
        if (!result) {
            std::cerr << "Warning: Failed to save training samples: "
                      << to_string(result.error()) << "\n";
        } else {
            std::cout << "Saved " << collector.size() << " training samples to " << samples_file << "\n";
        }
    }

    // Print timing breakdown
    get_timers().print();

    return 0;
}
#endif // QBOT_ENABLE_INFERENCE

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "Quoridor MCTS Bot\n";
    std::cout << "================\n\n";

    // Parse command line arguments
    auto config_opt = Config::from_args(argc, argv);
    if (!config_opt) {
        return 1;  // Error or help was printed
    }
    const Config& config = *config_opt;

    if (config.verbose) {
        config.print();
        std::cout << "\n";
    }

    // Initialize tree (load from file or create new)
    auto [pool, root] = initialize_tree(config);

    // Run appropriate mode
    switch (config.mode) {
        case RunMode::Interactive:
            return run_interactive(config, std::move(pool), root);

        case RunMode::Train:
            return run_training(config, std::move(pool), root);

        case RunMode::SelfPlay:
#ifdef QBOT_ENABLE_INFERENCE
            return run_selfplay(config, std::move(pool), root);
#else
            std::cerr << "Error: Self-play mode requires QBOT_ENABLE_INFERENCE\n";
            std::cerr << "Rebuild with: cmake -DENABLE_INFERENCE=ON ..\n";
            return 1;
#endif

        case RunMode::Arena:
#ifdef QBOT_ENABLE_INFERENCE
            return run_arena(config);
#else
            std::cerr << "Error: Arena mode requires QBOT_ENABLE_INFERENCE\n";
            std::cerr << "Rebuild with: cmake -DENABLE_INFERENCE=ON ..\n";
            return 1;
#endif
    }

    return 0;
}

} // namespace qbot

// Global main entry point
int main(int argc, char* argv[]) {
    return qbot::main(argc, argv);
}
