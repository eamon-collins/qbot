#include "Tree.h" 
#include "storage.h"
#include <cstring>
#include <iostream>

int num_threads = 1;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: leopard <tree_file>\n";
        return 1;
    }

    StateNode* root = load_tree(argv[1], false);
    if (!root) {
        std::cerr << "Failed to load tree from " << argv[1] << "\n";
        return 1;
    }

    // Process nodes in breadth-first order using a queue
    std::deque<StateNode*> queue;
    queue.push_back(root);

    while (!queue.empty()) {
        StateNode* node = queue.front();
        queue.pop_front();

        std::cout << *node;

        // Add children to queue
        for (auto& child : node->children) {
            queue.push_back(&child);
        }
    }

    delete root;
    return 0;
}
