/// leopard - Tree file to training data converter
///
/// AlphaZero-style training data extraction:
/// Finds all terminal nodes (completed games) and traces back to root,
/// outputting each position labeled with the actual game outcome.
///
/// Output format: SerializedNode structs where terminal_value contains
/// the game outcome (z = +1 if P1 won, -1 if P2 won) for ALL nodes,
/// not just terminal ones.

#include "storage.h"
#include "../tree/StateNode.h"

#include <cstdio>
#include <deque>
#include <iostream>
#include <unordered_set>
#include <vector>

using namespace qbot;

/// Serialize a node with a specific outcome value
void output_node(const StateNode& node, float outcome) {
    SerializedNode serialized;
    serialized.first_child = node.first_child;
    serialized.next_sibling = node.next_sibling;
    serialized.parent = node.parent;
    serialized.p1_row = node.p1.row;
    serialized.p1_col = node.p1.col;
    serialized.p1_fences = node.p1.fences;
    serialized.p2_row = node.p2.row;
    serialized.p2_col = node.p2.col;
    serialized.p2_fences = node.p2.fences;
    serialized.move_data = node.move.data;
    serialized.flags = node.flags;
    serialized.reserved = 0;
    serialized.ply = node.ply;
    serialized.fences_horizontal = node.fences.horizontal;
    serialized.fences_vertical = node.fences.vertical;
    serialized.visits = node.stats.visits.load(std::memory_order_relaxed);
    serialized.total_value = node.stats.total_value.load(std::memory_order_relaxed);
    serialized.prior = node.stats.prior;
    // Store the GAME OUTCOME, not the node's terminal status
    serialized.terminal_value = outcome;

    std::fwrite(&serialized, sizeof(SerializedNode), 1, stdout);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: leopard <tree_file>\n";
        std::cerr << "\n";
        std::cerr << "Extracts AlphaZero-style training data from a tree file.\n";
        std::cerr << "Finds completed games (terminal nodes) and outputs all positions\n";
        std::cerr << "along each game path, labeled with the actual game outcome.\n";
        std::cerr << "\n";
        std::cerr << "Output: Binary SerializedNode structs to stdout.\n";
        std::cerr << "        terminal_value field contains game outcome z ∈ {-1, +1}\n";
        return 1;
    }

    auto result = TreeStorage::load(argv[1]);
    if (!result) {
        std::cerr << "Failed to load tree from " << argv[1] << ": "
                  << to_string(result.error()) << "\n";
        return 1;
    }

    auto& [pool, root, timestamp] = *result;

    if (root == NULL_NODE) {
        std::cerr << "Tree has no root node\n";
        return 1;
    }

    // Step 1: Find all terminal nodes via BFS
    std::vector<uint32_t> terminal_nodes;
    std::deque<uint32_t> queue;
    queue.push_back(root);

    while (!queue.empty()) {
        uint32_t idx = queue.front();
        queue.pop_front();

        const StateNode& node = (*pool)[idx];

        if (node.is_terminal()) {
            terminal_nodes.push_back(idx);
        }

        // Add children to queue
        uint32_t child = node.first_child;
        while (child != NULL_NODE) {
            queue.push_back(child);
            child = (*pool)[child].next_sibling;
        }
    }

    std::cerr << "Found " << terminal_nodes.size() << " terminal nodes (completed games)\n";

    if (terminal_nodes.empty()) {
        std::cerr << "No completed games in tree - nothing to output\n";
        return 0;
    }

    // Step 2: For each terminal, trace back to root and output training samples
    // Track which nodes we've already output to avoid duplicates
    // (same position can be reached via different game paths)
    std::unordered_set<uint32_t> output_nodes;
    size_t sample_count = 0;
    size_t game_count = 0;

    for (uint32_t terminal_idx : terminal_nodes) {
        const StateNode& terminal = (*pool)[terminal_idx];
        float outcome = terminal.terminal_value;  // z = +1 (P1 wins) or -1 (P2 wins)

        // Walk back to root
        uint32_t current = terminal_idx;
        while (current != NULL_NODE) {
            // Only output each node once (first game path wins)
            if (output_nodes.find(current) == output_nodes.end()) {
                output_node((*pool)[current], outcome);
                output_nodes.insert(current);
                ++sample_count;
            }

            current = (*pool)[current].parent;
        }
        ++game_count;
    }

    std::cerr << "Wrote " << sample_count << " training samples from "
              << game_count << " games ("
              << (sample_count * sizeof(SerializedNode)) << " bytes)\n";

    return 0;
}
