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

#include <boost/program_options.hpp>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

namespace po = boost::program_options;

namespace qbot {

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
int run_interactive([[maybe_unused]] const Config& config,
                    [[maybe_unused]] std::unique_ptr<NodePool> pool,
                    [[maybe_unused]] uint32_t root) {
    std::cout << "\n=== Interactive Mode ===\n";

    // TODO: Implement interactive game loop
    // This would involve:
    // 1. Display current board state
    // 2. If human's turn: get move from stdin
    // 3. If bot's turn: run MCTS and select best move
    // 4. Apply move and update tree
    // 5. Check for game over
    // 6. Loop until game ends

    /*
    Game game(config);
    game.set_tree(std::move(pool), root);

    if (!config.model_file.empty()) {
        game.load_model(config.model_file);
    }

    while (!game.is_over()) {
        game.display_board();

        if (game.is_human_turn()) {
            Move move = game.get_human_move();
            game.apply_move(move);
        } else {
            Move move = game.search_best_move(config.simulations_per_move);
            std::cout << "Bot plays: " << move << "\n";
            game.apply_move(move);
        }
    }

    game.display_result();

    if (!config.save_file.empty()) {
        save_tree(*game.pool(), game.root(), config.save_file);
    }
    */

    std::cout << "Interactive mode not yet implemented.\n";
    std::cout << "The game engine components need to be built first.\n";

    return 0;
}

/// Run self-play training
int run_training([[maybe_unused]] const Config& config,
                 [[maybe_unused]] std::unique_ptr<NodePool> pool,
                 [[maybe_unused]] uint32_t root) {
    std::cout << "\n=== Training Mode ===\n";
    std::cout << "Games per checkpoint: " << config.games_per_checkpoint << "\n";
    std::cout << "Total iterations:     " << config.training_iterations << "\n";
    std::cout << "Simulations per move: " << config.simulations_per_move << "\n";
    std::cout << "Threads:              " << config.num_threads << "\n\n";

    // TODO: Implement self-play training loop
    // This would involve:
    // 1. Initialize game state
    // 2. For each game:
    //    a. Play game using MCTS (both sides)
    //    b. Collect training data (states, policies, outcomes)
    //    c. Update visit counts in tree
    // 3. Periodically:
    //    a. Save tree checkpoint
    //    b. Train neural network on collected data (if using NN)
    //    c. Report statistics

    /*
    Trainer trainer(config);
    trainer.set_tree(std::move(pool), root);

    if (!config.model_file.empty()) {
        trainer.load_model(config.model_file);
    }

    int games_completed = 0;
    int checkpoint_num = 0;

    while (games_completed < config.training_iterations) {
        // Play a batch of games
        auto stats = trainer.play_games(config.games_per_checkpoint, config.num_threads);

        games_completed += config.games_per_checkpoint;
        checkpoint_num++;

        std::cout << "Checkpoint " << checkpoint_num << ":\n";
        std::cout << "  Games:      " << games_completed << "\n";
        std::cout << "  P1 wins:    " << stats.p1_wins << "\n";
        std::cout << "  P2 wins:    " << stats.p2_wins << "\n";
        std::cout << "  Avg length: " << stats.avg_game_length << " moves\n";
        std::cout << "  Tree size:  " << trainer.pool()->allocated() << " nodes\n";

        // Save checkpoint
        std::string checkpoint_path = config.save_file + "." + std::to_string(checkpoint_num);
        save_tree(*trainer.pool(), trainer.root(), checkpoint_path);

        // Also save to main file
        save_tree(*trainer.pool(), trainer.root(), config.save_file);
    }

    std::cout << "\nTraining complete!\n";
    std::cout << "Final tree size: " << trainer.pool()->allocated() << " nodes\n";
    */

    std::cout << "Training mode not yet implemented.\n";
    std::cout << "The MCTS search and game engine components need to be built first.\n";

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
