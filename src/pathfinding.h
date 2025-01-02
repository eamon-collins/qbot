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
	std::vector<Move> vmoves;
	Player currPlayer = this->turn ? this->p1 : this->p2;
    Player otherPlayer = this->turn ? this->p2 : this->p1;

    // PAWN MOVES
    // Vertical moves
    for (int dy = -1; dy <= 1; dy += 2) { // dy = -1 for up, +1 for down
        int newRow = currPlayer.row + dy;
        if (newRow >= 0 && newRow < NUMROWS) {
            if (!this->gamestate[2 * currPlayer.row + dy][currPlayer.col] && 
                (newRow != otherPlayer.row || currPlayer.col != otherPlayer.col)) {
                // Valid move to an unoccupied adjacent square
                vmoves.emplace_back('p', newRow, currPlayer.col, false);
            } else if (newRow == otherPlayer.row && currPlayer.col == otherPlayer.col) {
                // Check jump over opponent
                int jumpRow = currPlayer.row + 2 * dy;
                if (jumpRow >= 0 && jumpRow < NUMROWS &&
                    !this->gamestate[2 * otherPlayer.row + dy][otherPlayer.col]) {
                    vmoves.emplace_back('p', jumpRow, currPlayer.col, false);
                } else {
                    // Diagonal moves around opponent
                    if (currPlayer.col + 1 < NUMCOLS && 
                        !this->gamestate[2 * currPlayer.row + dy][currPlayer.col + 1]) {
                        vmoves.emplace_back('p', currPlayer.row + dy, currPlayer.col + 1, false);
                    }
                    if (currPlayer.col - 1 >= 0 && 
                        !this->gamestate[2 * currPlayer.row + dy][currPlayer.col - 1]) {
                        vmoves.emplace_back('p', currPlayer.row + dy, currPlayer.col - 1, false);
                    }
                }
            }
        }
    }

    // Horizontal moves
    for (int dx = -1; dx <= 1; dx += 2) { // dx = -1 for left, +1 for right
        int newCol = currPlayer.col + dx;
        if (newCol >= 0 && newCol < NUMCOLS) {
            if (!this->gamestate[2 * currPlayer.row][currPlayer.col + (dx == 1 ? 0 : -1)] && 
                (currPlayer.row != otherPlayer.row || newCol != otherPlayer.col)) {
                // Valid move to an unoccupied adjacent square
                vmoves.emplace_back('p', currPlayer.row, newCol, false);
            } else if (currPlayer.row == otherPlayer.row && newCol == otherPlayer.col) {
                // Check jump over opponent
                int jumpCol = currPlayer.col + 2 * dx;
                if (jumpCol >= 0 && jumpCol < NUMCOLS &&
                    !this->gamestate[2 * currPlayer.row][currPlayer.col + (dx == 1 ? 1 : -2)]) {
                    vmoves.emplace_back('p', currPlayer.row, jumpCol, false);
                } else {
                    // Diagonal moves around opponent
                    if (currPlayer.row + 1 < NUMROWS &&
                        !this->gamestate[2 * currPlayer.row + 1][currPlayer.col + dx]) {
                        vmoves.emplace_back('p', currPlayer.row + 1, currPlayer.col + dx, false);
                    }
                    if (currPlayer.row - 1 >= 0 &&
                        !this->gamestate[2 * currPlayer.row - 1][currPlayer.col + dx]) {
                        vmoves.emplace_back('p', currPlayer.row - 1, currPlayer.col + dx, false);
                    }
                }
            }
        }
    }

    return vmoves;
}

int astar(StateNode* state, Player player, int goalRow) {
    std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> open_list;
    std::unordered_set<int> closed_list;

    auto hash_position = [](int row, int col) { return row * NUMCOLS + col; };

    // Initialize start node
    open_list.push({player.row, player.col, 0, heuristic(player.row, player.col, goalRow), 0});

    while (!open_list.empty()) {
        AStarNode current = open_list.top();
        open_list.pop();

        // If we reach the goal, return the cost
        if (current.row == goalRow) {
            return current.g;
        }

        // Mark this node as visited
        int currentHash = hash_position(current.row, current.col);
        if (closed_list.count(currentHash)) continue;
        closed_list.insert(currentHash);

        // Generate valid moves from the current position
        std::vector<Move> moves = state->generate_pawn_moves(moves);
        

        for (const Move& move : moves) {
            if (move.type != 'p') continue;  // Only process pawn moves
            int neighborRow = move.row;
            int neighborCol = move.col;
            int neighborHash = hash_position(neighborRow, neighborCol);

            if (closed_list.count(neighborHash)) continue;

            int gCost = current.g + 1;
            int hCost = heuristic(neighborRow, neighborCol, goalRow);
            int fCost = gCost + hCost;

            open_list.push({neighborRow, neighborCol, gCost, hCost, fCost});
        }
    }

    // Return a high cost if no path exists
    return std::numeric_limits<int>::max();
}

// Main pathfinding function
int pathfinding(StateNode* state, Move move) {
	bool gamestate[2*NUMROWS - 1][NUMCOLS];
	//copy over the state in question's gamestate so we don't affect the state itself
	memcpy(gamestate, state->gamestate, (2*NUMROWS-1)*NUMCOLS*sizeof(bool));

	//apply the proposed move
	if (move.type == 'f'){
		if (move.horizontal){
			state->gamestate[move.row][move.col] = true;
			state->gamestate[move.row][move.col+1] = true;
		}else{
			state->gamestate[move.row][move.col] = true;
			state->gamestate[move.row+2][move.col] = true;
		}
	}
	Player p1 = state->p1;
    Player p2 = state->p2;
    int p1_goal = 0;     // P1 must reach row 0
    int p2_goal = NUMROWS - 1;  // P2 must reach the last row

    // Calculate shortest paths for both players
    int p1_distance = astar(state, p1, p1_goal);
    int p2_distance = astar(state, p2, p2_goal);

	//Restore gamestate
	state->gamestate = gamestate;

    // Check for blockages
    if (p1_dist == std::numeric_limits<int>::max() || p2_dist == std::numeric_limits<int>::max()) {
        return -999;
    }

    // Return difference
    return p1_dist - p2_dist;
}
