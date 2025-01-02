#include <queue>
#include <vector>
#include <unordered_set>
#include <limits>

// Define helper structures for A* search
struct Position {
    int x, y;

    bool operator==(const Position& other) const {
        return x == other.x && y == other.y;
    }

    struct Hash {
        size_t operator()(const Position& pos) const {
            return std::hash<int>()(pos.x) ^ (std::hash<int>()(pos.y) << 1);
        }
    };
};

struct Node {
    Position pos;
    int g; // Cost from start
    int h; // Heuristic cost to goal
    int f; // Total cost (g + h)
    Node* parent;

    bool operator>(const Node& other) const {
        return f > other.f;
    }
};

// Heuristic function: Manhattan distance
int heuristic(const Position& start, const Position& goal) {
    return std::abs(start.x - goal.x) + std::abs(start.y - goal.y);
}

// Valid move generator considering Quoridor rules
std::vector<Position> generate_pawn_moves(const Position& current, const StateNode* state) {
    std::vector<Position> moves;

    // Define directions for standard moves
    const static std::vector<std::pair<int, int>> directions = {
        {0, 1},   // North
        {0, -1},  // South
        {1, 0},   // East
        {-1, 0}   // West
    };

    // Check each direction for valid moves
    for (const auto& [dx, dy] : directions) {
        Position neighbor = {current.x + dx, current.y + dy};

        // Check if move is within the board boundaries
        if (neighbor.x < 0 || neighbor.x >= 9 || neighbor.y < 0 || neighbor.y >= 9) {
            continue;
        }

        // Check if there is a fence blocking the move
        if (state->is_fence_between(current, neighbor)) {
            continue;
        }

        // Check if neighbor square is occupied by an opponent's pawn
        if (neighbor.x == state->p1.x && neighbor.y == state->p1.y || 
            neighbor.x == state->p2.x && neighbor.y == state->p2.y) {

            // Attempt jump over the pawn
            Position jump = {neighbor.x + dx, neighbor.y + dy};
            if (jump.x >= 0 && jump.x < 9 && jump.y >= 0 && jump.y < 9 && 
                !state->is_fence_between(neighbor, jump)) {
                moves.push_back(jump);
                continue;
            }

            // Diagonal moves if jump is blocked
            for (const auto& [ddx, ddy] : directions) {
                if ((ddx == dx && ddy == dy) || (ddx == -dx && ddy == -dy)) {
                    continue; // Skip backtracking and original directions
                }
                Position diagonal = {neighbor.x + ddx, neighbor.y + ddy};
                if (diagonal.x >= 0 && diagonal.x < 9 && diagonal.y >= 0 && diagonal.y < 9 &&
                    !state->is_fence_between(neighbor, diagonal)) {
                    moves.push_back(diagonal);
                }
            }

            continue;
        }

        moves.push_back(neighbor);
    }

    return moves;
    return moves;
}

// A* search for shortest path
int astar(const StateNode* state, const Position& start, const std::vector<int>& goal_rows) {
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open_list;
    std::unordered_set<Position, Position::Hash> closed_list;

    open_list.push({start, 0, heuristic(start, {start.x, goal_rows[0]}), 0, nullptr});

    while (!open_list.empty()) {
        Node current = open_list.top();
        open_list.pop();

        // Check if goal reached
        for (int goal : goal_rows) {
            if (current.pos.y == goal) return current.g;
        }

        // Add to closed list
        closed_list.insert(current.pos);

        // Expand neighbors
        for (const Position& neighbor : generate_valid_moves(current.pos, state)) {
            if (closed_list.count(neighbor)) continue;

            int g_cost = current.g + 1;
            int h_cost = heuristic(neighbor, {neighbor.x, goal_rows[0]});
            int f_cost = g_cost + h_cost;

            open_list.push({neighbor, g_cost, h_cost, f_cost, nullptr});
        }
    }

    // Return large penalty if path not found
    return std::numeric_limits<int>::max();
}

// Main pathfinding function
int pathfinding(StateNode* state, Move move) {
    Position p1_start = {state->p1.x, state->p1.y};
    Position p2_start = {state->p2.x, state->p2.y};
    std::vector<int> p1_goal_rows = {0};      // Replace with actual goal rows for P1
    std::vector<int> p2_goal_rows = {8};     // Replace with actual goal rows for P2

    // Find shortest paths
    int p1_dist = astar(state, p1_start, p1_goal_rows);
    int p2_dist = astar(state, p2_start, p2_goal_rows);

    // Check for blockages
    if (p1_dist == std::numeric_limits<int>::max() || p2_dist == std::numeric_limits<int>::max()) {
        return -999;
    }

    // Return difference
    return p1_dist - p2_dist;
}
