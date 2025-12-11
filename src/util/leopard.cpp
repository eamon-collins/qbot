/// leopard - Tree file to stdout converter for training
///
/// Reads a binary tree file and outputs all nodes as raw SerializedNode
/// structs to stdout for consumption by Python training scripts.

#include "storage.h"
#include "../tree/StateNode.h"

#include <cstdio>
#include <deque>
#include <iostream>

using namespace qbot;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: leopard <tree_file>\n";
        std::cerr << "Outputs tree nodes as binary SerializedNode structs to stdout\n";
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

    // Process nodes in breadth-first order
    std::deque<uint32_t> queue;
    queue.push_back(root);
    size_t node_count = 0;

    while (!queue.empty()) {
        uint32_t idx = queue.front();
        queue.pop_front();

        const StateNode& node = (*pool)[idx];

        // Serialize the node
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
        serialized.terminal_value = node.terminal_value;

        // Write raw bytes to stdout
        std::fwrite(&serialized, sizeof(SerializedNode), 1, stdout);
        ++node_count;

        // Add children to queue
        uint32_t child = node.first_child;
        while (child != NULL_NODE) {
            queue.push_back(child);
            child = (*pool)[child].next_sibling;
        }
    }

    std::cerr << "Wrote " << node_count << " nodes ("
              << (node_count * sizeof(SerializedNode)) << " bytes)\n";

    return 0;
}
