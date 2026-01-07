/// leopard - Tree file to training data converter
///
/// Extracts training data from MCTS tree using accumulated Q-values.
/// Each node's Q-value represents the average outcome from that position,
/// based on all simulations that passed through it.
///
/// Output format: SerializedNode structs where terminal_value contains
/// the Q-value from P1's perspective (positive = P1 advantage).

#include "storage.h"
#include "../tree/StateNode.h"

#include <cstdio>
#include <deque>
#include <iostream>
#include <vector>

using namespace qbot;

// Minimum visits required for a node to be included in training data
constexpr uint32_t MIN_VISITS_FOR_TRAINING = 5;

/// Serialize a node with its Q-value (from P1's perspective)
void output_node(const StateNode& node, float value) {
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
    // Store Q-value from P1's perspective
    serialized.terminal_value = value;

    std::fwrite(&serialized, sizeof(SerializedNode), 1, stdout);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: leopard <tree_file>\n";
        std::cerr << "\n";
        std::cerr << "Extracts training data from MCTS tree using Q-values.\n";
        std::cerr << "Outputs all nodes with sufficient visits, labeled with their\n";
        std::cerr << "accumulated Q-value (from P1's perspective).\n";
        std::cerr << "\n";
        std::cerr << "Output: Binary SerializedNode structs to stdout.\n";
        std::cerr << "        terminal_value field contains Q ∈ [-1, +1] from P1's perspective\n";
        std::cerr << "        (positive = P1 advantage, negative = P2 advantage)\n";
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

    // BFS through tree, output all nodes with sufficient visits
    std::deque<uint32_t> queue;
    queue.push_back(root);

    size_t sample_count = 0;
    size_t terminal_count = 0;
    size_t skipped_low_visits = 0;

    while (!queue.empty()) {
        uint32_t idx = queue.front();
        queue.pop_front();

        const StateNode& node = (*pool)[idx];
        uint32_t visits = node.stats.visits.load(std::memory_order_relaxed);

        // Check if node has enough visits to be useful training data
        if (visits >= MIN_VISITS_FOR_TRAINING) {
            float q_value;

            if (node.is_terminal()) {
                // Terminal nodes have known exact values
                q_value = node.terminal_value;
                ++terminal_count;
            } else {
                // Get Q-value: stored Q is from perspective of player to move
                // We need to convert to P1's perspective for consistent training
                float stored_q = node.stats.Q(0.0f);

                // If P1 to move: stored Q is already from P1's perspective
                // If P2 to move: stored Q is from P2's perspective, negate for P1's
                q_value = node.is_p1_to_move() ? stored_q : -stored_q;
            }

            output_node(node, q_value);
            ++sample_count;
        } else {
            ++skipped_low_visits;
        }

        // Add children to queue
        uint32_t child = node.first_child;
        while (child != NULL_NODE) {
            queue.push_back(child);
            child = (*pool)[child].next_sibling;
        }
    }

    std::cerr << "Found " << terminal_count << " terminal nodes (completed games)\n";
    std::cerr << "Wrote " << sample_count << " training samples ("
              << (sample_count * sizeof(SerializedNode)) << " bytes)\n";
    std::cerr << "Skipped " << skipped_low_visits << " nodes with < "
              << MIN_VISITS_FOR_TRAINING << " visits\n";

    return 0;
}
