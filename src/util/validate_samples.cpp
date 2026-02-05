/// validate_samples.cpp
/// Validates pathfinding for all states in .qsamples files
///
/// Reads .qsamples files and checks that each pawn can reach their goal
/// Only checks the last recorded state in each game (works backwards,
/// skipping until the previous state was a starting state).

#include "../tree/StateNode.h"
#include "../util/pathfinding.h"
#include "../util/training_samples.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace qbot {

/// Check if a CompactState is a starting state (both players at start with 10 fences)
[[nodiscard]] bool is_start_state(const CompactState& state) noexcept {
    // In absolute coordinates (before flipping)
    // P1 starts at (0, 4), P2 at (8, 4), both with 10 fences

    // Check the flag to see whose turn it is
    bool is_p1_turn = (state.flags & 0x04) != 0;

    uint8_t abs_p1_row, abs_p1_col, abs_p2_row, abs_p2_col;

    if (is_p1_turn) {
        abs_p1_row = state.p1_row;
        abs_p1_col = state.p1_col;
        abs_p2_row = state.p2_row;
        abs_p2_col = state.p2_col;
    } else {
        // Unflip 180 degrees
        abs_p1_row = 8 - state.p1_row;
        abs_p1_col = 8 - state.p1_col;
        abs_p2_row = 8 - state.p2_row;
        abs_p2_col = 8 - state.p2_col;
    }

    return (abs_p1_row == 0 && abs_p1_col == 4 && state.p1_fences == 10 &&
            abs_p2_row == 8 && abs_p2_col == 4 && state.p2_fences == 10);
}

/// Convert CompactState to StateNode for pathfinding
[[nodiscard]] StateNode compact_to_node(const CompactState& state) noexcept {
    StateNode node;

    // Restore absolute coordinates
    bool is_p1_turn = (state.flags & 0x04) != 0;

    if (is_p1_turn) {
        node.p1.row = state.p1_row;
        node.p1.col = state.p1_col;
        node.p2.row = state.p2_row;
        node.p2.col = state.p2_col;
        node.fences.horizontal = state.fences_horizontal;
        node.fences.vertical = state.fences_vertical;
    } else {
        // Unflip 180 degrees
        node.p1.row = 8 - state.p1_row;
        node.p1.col = 8 - state.p1_col;
        node.p2.row = 8 - state.p2_row;
        node.p2.col = 8 - state.p2_col;

        // Reverse fence bits
        node.fences.horizontal = reverse_bits(state.fences_horizontal);
        node.fences.vertical = reverse_bits(state.fences_vertical);
    }

    node.p1.fences = state.p1_fences;
    node.p2.fences = state.p2_fences;
    node.flags = state.flags;

    return node;
}

/// Validate a single .qsamples file
struct ValidationResult {
    size_t total_games = 0;
    size_t states_checked = 0;
    size_t p1_blocked = 0;
    size_t p2_blocked = 0;
    size_t both_blocked = 0;
};

[[nodiscard]] ValidationResult validate_samples_file(const std::filesystem::path& path, bool verbose) {
    ValidationResult result;

    if (verbose) {
        std::cout << "Loading samples from " << path << "...\n";
    }

    auto samples_result = TrainingSampleStorage::load(path);
    if (!samples_result) {
        std::cerr << "Error loading file: " << to_string(samples_result.error()) << "\n";
        return result;
    }

    const auto& samples = *samples_result;
    if (verbose) {
        std::cout << "Loaded " << samples.size() << " samples\n";
    }

    if (samples.empty()) {
        return result;
    }

    // Thread-local pathfinder for reuse
    Pathfinder& pathfinder = get_pathfinder();

    // Work backwards through samples, only checking last state of each game
    bool in_game = false;

    for (size_t i = samples.size(); i > 0; --i) {
        const auto& sample = samples[i - 1];

        if (!in_game) {
            // Check if this is an end state (look ahead to see if next is start or doesn't exist)
            bool is_end_state = false;

            if (i == samples.size()) {
                // Last sample in file
                is_end_state = true;
            } else if (is_start_state(samples[i].state)) {
                // Next sample is a start state, so this is an end state
                is_end_state = true;
            }

            if (is_end_state) {
                in_game = true;
                result.total_games++;

                // Check pathfinding for this state
                StateNode node = compact_to_node(sample.state);

                bool p1_can_reach = pathfinder.can_reach(node.fences, node.p1, 8);
                bool p2_can_reach = pathfinder.can_reach(node.fences, node.p2, 0);

                result.states_checked++;

                if (!p1_can_reach && !p2_can_reach) {
                    result.both_blocked++;
                    if (verbose) {
                        std::cout << "VIOLATION at sample " << (i - 1) << ": BOTH players blocked!\n";
                        std::cout << "  P1 at (" << static_cast<int>(node.p1.row) << ", "
                                  << static_cast<int>(node.p1.col) << "), fences: "
                                  << static_cast<int>(node.p1.fences) << "\n";
                        std::cout << "  P2 at (" << static_cast<int>(node.p2.row) << ", "
                                  << static_cast<int>(node.p2.col) << "), fences: "
                                  << static_cast<int>(node.p2.fences) << "\n";
                    }
                } else if (!p1_can_reach) {
                    result.p1_blocked++;
                    if (verbose) {
                        std::cout << "VIOLATION at sample " << (i - 1) << ": P1 blocked!\n";
                        std::cout << "  P1 at (" << static_cast<int>(node.p1.row) << ", "
                                  << static_cast<int>(node.p1.col) << "), fences: "
                                  << static_cast<int>(node.p1.fences) << "\n";
                        std::cout << "  P2 at (" << static_cast<int>(node.p2.row) << ", "
                                  << static_cast<int>(node.p2.col) << "), fences: "
                                  << static_cast<int>(node.p2.fences) << "\n";
                    }
                } else if (!p2_can_reach) {
                    result.p2_blocked++;
                    if (verbose) {
                        std::cout << "VIOLATION at sample " << (i - 1) << ": P2 blocked!\n";
                        std::cout << "  P1 at (" << static_cast<int>(node.p1.row) << ", "
                                  << static_cast<int>(node.p1.col) << "), fences: "
                                  << static_cast<int>(node.p1.fences) << "\n";
                        std::cout << "  P2 at (" << static_cast<int>(node.p2.row) << ", "
                                  << static_cast<int>(node.p2.col) << "), fences: "
                                  << static_cast<int>(node.p2.fences) << "\n";
                    }
                }
            }
        } else {
            // In a game, check if we reached the start
            if (is_start_state(sample.state)) {
                in_game = false;
            }
        }
    }

    return result;
}

} // namespace qbot

int main(int argc, char* argv[]) {
    using namespace qbot;

    bool verbose = false;
    std::vector<std::string> files;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [-v|--verbose] <file1.qsamples> [file2.qsamples ...]\n";
            std::cout << "\nValidates pathfinding in .qsamples files.\n";
            std::cout << "Checks that each pawn can reach their goal at the last state of each game.\n";
            std::cout << "\nOptions:\n";
            std::cout << "  -v, --verbose    Show detailed output for each violation\n";
            std::cout << "  -h, --help       Show this help message\n";
            return 0;
        } else {
            files.push_back(arg);
        }
    }

    if (files.empty()) {
        std::cerr << "Usage: " << argv[0] << " [-v|--verbose] <file1.qsamples> [file2.qsamples ...]\n";
        std::cerr << "Use --help for more information\n";
        return 1;
    }

    ValidationResult total_result;

    int retcode = 0;
    for (const auto& file : files) {
        std::filesystem::path path(file);

        if (!std::filesystem::exists(path)) {
            std::cerr << "File not found: " << path << "\n";
            continue;
        }

        if (verbose) {
            std::cout << "\n" << std::string(80, '=') << "\n";
        }

        auto result = validate_samples_file(path, verbose);

        total_result.total_games += result.total_games;
        total_result.states_checked += result.states_checked;
        total_result.p1_blocked += result.p1_blocked;
        total_result.p2_blocked += result.p2_blocked;
        total_result.both_blocked += result.both_blocked;

        size_t total_violations = result.p1_blocked + result.p2_blocked + result.both_blocked;

        // Only output if verbose or if violations found
        if (verbose || total_violations > 0) {
            if (verbose) {
                std::cout << "\nResults for " << path.filename() << ":\n";
                std::cout << "  Games found: " << result.total_games << "\n";
                std::cout << "  States checked: " << result.states_checked << "\n";
                std::cout << "  P1 blocked: " << result.p1_blocked << "\n";
                std::cout << "  P2 blocked: " << result.p2_blocked << "\n";
                std::cout << "  Both blocked: " << result.both_blocked << "\n";
            }

            if (total_violations > 0) {
                std::cout << path.filename() << ": " << total_violations
                          << " violation" << (total_violations != 1 ? "s" : "")
                          << " (P1: " << result.p1_blocked
                          << ", P2: " << result.p2_blocked
                          << ", both: " << result.both_blocked << ")\n";
                retcode = 1;
            } else if (verbose) {
                std::cout << "  All pathfinding checks passed!\n";
            }
        }
    }

    if (files.size() > 1) {
        size_t total_violations = total_result.p1_blocked + total_result.p2_blocked + total_result.both_blocked;

        if (verbose || total_violations > 0) {
            if (verbose) {
                std::cout << "\n" << std::string(80, '=') << "\n";
                std::cout << "TOTAL RESULTS:\n";
                std::cout << "  Games found: " << total_result.total_games << "\n";
                std::cout << "  States checked: " << total_result.states_checked << "\n";
                std::cout << "  P1 blocked: " << total_result.p1_blocked << "\n";
                std::cout << "  P2 blocked: " << total_result.p2_blocked << "\n";
                std::cout << "  Both blocked: " << total_result.both_blocked << "\n";
            }

            if (total_violations > 0) {
                std::cout << "\nTOTAL: " << total_violations
                          << " violation" << (total_violations != 1 ? "s" : "")
                          << " across " << files.size() << " files\n";
                return 1;
            } else if (verbose) {
                std::cout << "  All pathfinding checks passed!\n";
            }
        }
    }

    return retcode;
}
