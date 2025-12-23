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
#include "../search/mcts.h"
#include "Game.h"

#include <csignal>

#include <boost/program_options.hpp>
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
    Train,          // Self-play training
    // Future modes:
    // Analyze,     // Analyze a position
    // Benchmark,   // Performance benchmarking
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
            "Run in training mode (self-play)")
        ("verbose,v", po::bool_switch(&config.verbose),
            "Enable verbose output")
        ("games,g", po::value<int>(&config.games_per_checkpoint)->default_value(100),
            "Games per checkpoint in training")
        ("iterations,i", po::value<int>(&config.training_iterations)->default_value(10000),
            "Training iterations")
        ("simulations,n", po::value<int>(&config.simulations_per_move)->default_value(800),
            "MCTS simulations per move");

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

    config.mode = vm["train"].as<bool>() ? RunMode::Train : RunMode::Interactive;

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
    os << "  Mode:           " << (mode == RunMode::Train ? "Training" : "Interactive") << "\n";
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
        os << "  Games/checkpoint: " << games_per_checkpoint << "\n";
        os << "  Iterations:     " << training_iterations << "\n";
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

        auto result = TreeStorage::load(config.load_file);
        if (result.has_value()) {
            auto& loaded = *result;
            std::cout << "  Loaded " << loaded.pool->allocated() << " nodes\n";
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

    auto result = TreeStorage::save(path, pool, root);
    if (result.has_value()) {
        auto file_size = std::filesystem::file_size(path);
        std::cout << "  Saved " << pool.allocated() << " nodes ("
                  << file_size / 1024 << " KB)\n";
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

    // Initialize the root node with starting game state
    StateNode& root_node = (*pool)[root];
    root_node.init_root(true);  // P1 (human) starts

    gui.send_start("Human", "Bot");

    uint32_t current_idx = root;
    bool game_over = false;

    while (!game_over) {
        StateNode& current = (*pool)[current_idx];

        float score = current.stats.Q(0.0f);
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
                uint32_t next_idx = current.find_or_create_child(*pool, current_idx, move);
                if (next_idx == NULL_NODE) {
                    std::cerr << "Error: Failed to create child node for move\n";
                    return 1;
                }
                current_idx = next_idx;
            }
        } else {
            // Bot's turn
            Move move = Game::select_best_move(*pool, current_idx);
            if (!move.is_valid()) {
                std::cerr << "Error: Bot has no valid moves\n";
                return 1;
            }

            move.print("Bot");

            uint32_t next_idx = current.find_or_create_child(*pool, current_idx, move);
            if (next_idx == NULL_NODE) {
                std::cerr << "Error: Failed to create child node for bot's move\n";
                return 1;
            }
            current_idx = next_idx;
        }
    }

    // Send final state to GUI
    StateNode& final_state = (*pool)[current_idx];
    gui.send_gamestate(final_state, -1, final_state.terminal_value);

    // Optionally save tree
    if (!config.save_file.empty() && config.verbose) {
        save_tree(*pool, root, config.save_file);
    }

    return 0;
}

/// Run MCTS tree building / training
int run_training(const Config& config,
                 std::unique_ptr<NodePool> pool,
                 uint32_t root) {
    std::cout << "\n=== MCTS Training Mode ===\n";

    // Initialize the root node if it's a fresh tree
    StateNode& root_node = (*pool)[root];
    if (!root_node.move.is_valid() && root_node.ply == 0) {
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
    }

    return 0;
}

} // namespace qbot

// Global main entry point
int main(int argc, char* argv[]) {
    return qbot::main(argc, argv);
}
