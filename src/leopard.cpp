#include "Tree.h" 
#include "storage.h"
#include <cstring>
#include <iostream>

int num_threads = 1;

struct BinaryNode {
    Move move;
    Player p1;
    Player p2;
    bool turn;
    bool gamestate[2*NUMROWS - 1][NUMCOLS];
    double score;
    int visits;
    int ply;
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: leopard <tree_file>\n";
        return 1;
    }

    StateNode* root = load_tree(argv[1]);
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

        // Convert to binary format
        BinaryNode bnode;
        bnode.move = node->move;
        bnode.p1 = node->p1;
        bnode.p2 = node->p2;
        bnode.turn = node->turn;
        memcpy(bnode.gamestate, node->gamestate, sizeof(bnode.gamestate));
        bnode.score = node->score; 
        bnode.visits = node->visits;
        bnode.ply = node->ply;

        // Write binary node to stdout
        std::cout.write(reinterpret_cast<char*>(&bnode), sizeof(BinaryNode));

        // Add children to queue
        for (auto& child : node->children) {
            queue.push_back(&child);
        }
    }

    delete root;
    return 0;
}
