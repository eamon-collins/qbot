#include "pathfinding.h"

#include <algorithm>
#include <limits>
#include <cstring>

namespace qbot {

Pathfinder::Pathfinder() {
    reset();
}

void Pathfinder::reset() noexcept {
    for (auto& row : nodes_) {
        for (auto& node : row) {
            node.g_cost = std::numeric_limits<uint8_t>::max();
            node.f_cost = std::numeric_limits<uint8_t>::max();
            node.parent_row = 0;
            node.parent_col = 0;
            node.in_open = false;
            node.in_closed = false;
        }
    }
    open_count_ = 0;
}

bool Pathfinder::can_reach(const FenceGrid& fences, const Player& player, uint8_t goal_row) noexcept {
    reset();

    const uint8_t start_row = player.row;
    const uint8_t start_col = player.col;

    // Initialize start node
    nodes_[start_row][start_col].g_cost = 0;
    nodes_[start_row][start_col].f_cost = heuristic(start_row, goal_row);
    nodes_[start_row][start_col].in_open = true;
    open_set_[0] = {start_row, start_col};
    open_count_ = 1;

    while (open_count_ > 0) {
        // Find node with lowest f_cost in open set
        size_t best_idx = 0;
        uint8_t best_f = nodes_[open_set_[0].row][open_set_[0].col].f_cost;
        for (size_t i = 1; i < open_count_; ++i) {
            uint8_t f = nodes_[open_set_[i].row][open_set_[i].col].f_cost;
            if (f < best_f) {
                best_f = f;
                best_idx = i;
            }
        }

        Coord current = open_set_[best_idx];

        // Check if we reached the goal row
        if (current.row == goal_row) {
            return true;
        }

        // Move from open to closed
        nodes_[current.row][current.col].in_open = false;
        nodes_[current.row][current.col].in_closed = true;

        // Remove from open set by swapping with last
        open_set_[best_idx] = open_set_[--open_count_];

        // Process neighbors
        std::array<Coord, 4> neighbors;
        int neighbor_count = get_neighbors(fences, current.row, current.col, neighbors);

        uint8_t current_g = nodes_[current.row][current.col].g_cost;

        for (int i = 0; i < neighbor_count; ++i) {
            Coord neighbor = neighbors[i];
            Node& n_node = nodes_[neighbor.row][neighbor.col];

            if (n_node.in_closed) {
                continue;
            }

            uint8_t tentative_g = current_g + 1;

            if (!n_node.in_open) {
                // New node
                n_node.g_cost = tentative_g;
                n_node.f_cost = tentative_g + heuristic(neighbor.row, goal_row);
                n_node.parent_row = current.row;
                n_node.parent_col = current.col;
                n_node.in_open = true;
                open_set_[open_count_++] = neighbor;
            } else if (tentative_g < n_node.g_cost) {
                // Better path found
                n_node.g_cost = tentative_g;
                n_node.f_cost = tentative_g + heuristic(neighbor.row, goal_row);
                n_node.parent_row = current.row;
                n_node.parent_col = current.col;
            }
        }
    }

    return false;
}

bool Pathfinder::check_paths(const StateNode& state) noexcept {
    // P1 needs row 8, P2 needs row 0
    return can_reach(state.fences, state.p1, 8) && can_reach(state.fences, state.p2, 0);
}

bool Pathfinder::check_paths_with_fence(const StateNode& state, Move fence_move) noexcept {
    // Copy fence grid and apply the move
    FenceGrid temp_fences = state.fences;

    if (fence_move.is_horizontal()) {
        temp_fences.place_h_fence(fence_move.row(), fence_move.col());
    } else {
        temp_fences.place_v_fence(fence_move.row(), fence_move.col());
    }

    return can_reach(temp_fences, state.p1, 8) && can_reach(temp_fences, state.p2, 0);
}

bool Pathfinder::check_player_path_with_fence(const StateNode& state, Move fence_move, bool player1) noexcept {
    // Copy fence grid and apply the move
    FenceGrid temp_fences = state.fences;

    if (fence_move.is_horizontal()) {
        temp_fences.place_h_fence(fence_move.row(), fence_move.col());
    } else {
        temp_fences.place_v_fence(fence_move.row(), fence_move.col());
    }
    if (player1) {
        return can_reach(temp_fences, state.p1, 8);
    } else {
        return can_reach(temp_fences, state.p2, 0);
    }
}

std::vector<Coord> Pathfinder::find_path(const StateNode& state) noexcept {
    const Player& player = state.is_p1_to_move() ? state.p1 : state.p2;
    uint8_t goal_row = state.is_p1_to_move() ? 8 : 0;
    return find_path(state.fences, player, goal_row);
}

std::vector<Coord> Pathfinder::find_path(const FenceGrid& fences, const Player& player, uint8_t goal_row) noexcept {
    reset();

    const uint8_t start_row = player.row;
    const uint8_t start_col = player.col;

    // Initialize start node
    nodes_[start_row][start_col].g_cost = 0;
    nodes_[start_row][start_col].f_cost = heuristic(start_row, goal_row);
    nodes_[start_row][start_col].in_open = true;
    open_set_[0] = {start_row, start_col};
    open_count_ = 1;

    Coord goal_coord{0, 0};
    bool found = false;

    while (open_count_ > 0) {
        // Find node with lowest f_cost
        size_t best_idx = 0;
        uint8_t best_f = nodes_[open_set_[0].row][open_set_[0].col].f_cost;
        for (size_t i = 1; i < open_count_; ++i) {
            uint8_t f = nodes_[open_set_[i].row][open_set_[i].col].f_cost;
            if (f < best_f) {
                best_f = f;
                best_idx = i;
            }
        }

        Coord current = open_set_[best_idx];

        if (current.row == goal_row) {
            goal_coord = current;
            found = true;
            break;
        }

        nodes_[current.row][current.col].in_open = false;
        nodes_[current.row][current.col].in_closed = true;
        open_set_[best_idx] = open_set_[--open_count_];

        std::array<Coord, 4> neighbors;
        int neighbor_count = get_neighbors(fences, current.row, current.col, neighbors);
        uint8_t current_g = nodes_[current.row][current.col].g_cost;

        for (int i = 0; i < neighbor_count; ++i) {
            Coord neighbor = neighbors[i];
            Node& n_node = nodes_[neighbor.row][neighbor.col];

            if (n_node.in_closed) {
                continue;
            }

            uint8_t tentative_g = current_g + 1;

            if (!n_node.in_open) {
                n_node.g_cost = tentative_g;
                n_node.f_cost = tentative_g + heuristic(neighbor.row, goal_row);
                n_node.parent_row = current.row;
                n_node.parent_col = current.col;
                n_node.in_open = true;
                open_set_[open_count_++] = neighbor;
            } else if (tentative_g < n_node.g_cost) {
                n_node.g_cost = tentative_g;
                n_node.f_cost = tentative_g + heuristic(neighbor.row, goal_row);
                n_node.parent_row = current.row;
                n_node.parent_col = current.col;
            }
        }
    }

    if (!found) {
        return {};
    }

    // Reconstruct path
    std::vector<Coord> path;
    Coord current = goal_coord;
    while (!(current.row == start_row && current.col == start_col)) {
        path.push_back(current);
        Node& n = nodes_[current.row][current.col];
        current = {n.parent_row, n.parent_col};
    }
    path.push_back({start_row, start_col});

    std::reverse(path.begin(), path.end());
    return path;
}

int Pathfinder::path_length(const FenceGrid& fences, const Player& player, uint8_t goal_row) noexcept {
    reset();

    const uint8_t start_row = player.row;
    const uint8_t start_col = player.col;

    nodes_[start_row][start_col].g_cost = 0;
    nodes_[start_row][start_col].f_cost = heuristic(start_row, goal_row);
    nodes_[start_row][start_col].in_open = true;
    open_set_[0] = {start_row, start_col};
    open_count_ = 1;

    while (open_count_ > 0) {
        size_t best_idx = 0;
        uint8_t best_f = nodes_[open_set_[0].row][open_set_[0].col].f_cost;
        for (size_t i = 1; i < open_count_; ++i) {
            uint8_t f = nodes_[open_set_[i].row][open_set_[i].col].f_cost;
            if (f < best_f) {
                best_f = f;
                best_idx = i;
            }
        }

        Coord current = open_set_[best_idx];

        if (current.row == goal_row) {
            return nodes_[current.row][current.col].g_cost;
        }

        nodes_[current.row][current.col].in_open = false;
        nodes_[current.row][current.col].in_closed = true;
        open_set_[best_idx] = open_set_[--open_count_];

        std::array<Coord, 4> neighbors;
        int neighbor_count = get_neighbors(fences, current.row, current.col, neighbors);
        uint8_t current_g = nodes_[current.row][current.col].g_cost;

        for (int i = 0; i < neighbor_count; ++i) {
            Coord neighbor = neighbors[i];
            Node& n_node = nodes_[neighbor.row][neighbor.col];

            if (n_node.in_closed) {
                continue;
            }

            uint8_t tentative_g = current_g + 1;

            if (!n_node.in_open) {
                n_node.g_cost = tentative_g;
                n_node.f_cost = tentative_g + heuristic(neighbor.row, goal_row);
                n_node.parent_row = current.row;
                n_node.parent_col = current.col;
                n_node.in_open = true;
                open_set_[open_count_++] = neighbor;
            } else if (tentative_g < n_node.g_cost) {
                n_node.g_cost = tentative_g;
                n_node.f_cost = tentative_g + heuristic(neighbor.row, goal_row);
                n_node.parent_row = current.row;
                n_node.parent_col = current.col;
            }
        }
    }

    return -1;
}

Pathfinder& get_pathfinder() noexcept {
    thread_local Pathfinder pf;
    return pf;
}

int check_jump_advantage(
    const std::vector<Coord>& p1_path,
    const std::vector<Coord>& p2_path,
    bool p1_moves_first) noexcept
{
    if (p1_path.size() < 2 || p2_path.size() < 2) {
        return 0;
    }

    size_t p1_idx = 0;
    size_t p2_idx = 0;

    while (p1_idx + 1 < p1_path.size() || p2_idx + 1 < p2_path.size()) {
        if (p1_moves_first) {
            // P1 moves
            if (p1_idx + 1 < p1_path.size()) {
                Coord p1_next = p1_path[p1_idx + 1];
                Coord p2_current = p2_path[p2_idx];
                if (p1_next.row == p2_current.row && p1_next.col == p2_current.col) {
                    return 1;  // P1 jumps P2
                }
                p1_idx++;
            }

            // P2 moves
            if (p2_idx + 1 < p2_path.size()) {
                Coord p2_next = p2_path[p2_idx + 1];
                Coord p1_current = p1_path[p1_idx];
                if (p2_next.row == p1_current.row && p2_next.col == p1_current.col) {
                    return -1;  // P2 jumps P1
                }
                p2_idx++;
            }
        } else {
            // P2 moves first
            if (p2_idx + 1 < p2_path.size()) {
                Coord p2_next = p2_path[p2_idx + 1];
                Coord p1_current = p1_path[p1_idx];
                if (p2_next.row == p1_current.row && p2_next.col == p1_current.col) {
                    return -1;  // P2 jumps P1
                }
                p2_idx++;
            }

            // P1 moves
            if (p1_idx + 1 < p1_path.size()) {
                Coord p1_next = p1_path[p1_idx + 1];
                Coord p2_current = p2_path[p2_idx];
                if (p1_next.row == p2_current.row && p1_next.col == p2_current.col) {
                    return 1;  // P1 jumps P2
                }
                p1_idx++;
            }
        }
    }

    return 0;
}

// Returns a pair of {horizontal_blockers, vertical_blockers} bitmasks
// Bit 'i' corresponds to a fence at index 'i' (row*8 + col).
std::pair<uint64_t, uint64_t> compute_path_blockers(const std::vector<Coord>& path) {
    uint64_t h_blockers = 0;
    uint64_t v_blockers = 0;

    if (path.size() < 2) return {0, 0};

    for (size_t i = 0; i < path.size() - 1; ++i) {
        uint8_t r1 = path[i].row;
        uint8_t c1 = path[i].col;
        uint8_t r2 = path[i+1].row;
        uint8_t c2 = path[i+1].col;

        if (r1 == r2) { 
            // Horizontal move: blocked by VERTICAL fences
            // A move from (r,c) to (r,c+1) is blocked by V-fence at (r, c) or (r-1, c)
            uint8_t fence_col = std::min(c1, c2);
            // V-fence at row r
            if (r1 < 8) v_blockers |= (1ULL << (r1 * 8 + fence_col));
            // V-fence at row r-1
            if (r1 > 0) v_blockers |= (1ULL << ((r1 - 1) * 8 + fence_col));
        } else {
            // Vertical move: blocked by HORIZONTAL fences
            // A move from (r,c) to (r+1,c) is blocked by H-fence at (r, c) or (r, c-1)
            uint8_t fence_row = std::min(r1, r2);
            // H-fence at col c
            if (c1 < 8) h_blockers |= (1ULL << (fence_row * 8 + c1));
            // H-fence at col c-1
            if (c1 > 0) h_blockers |= (1ULL << (fence_row * 8 + (c1 - 1)));
        }
    }
    return {h_blockers, v_blockers};
}

using Bitboard = unsigned __int128;

static constexpr Bitboard ROW_0_MASK = 0x1FF; // First 9 bits
static constexpr Bitboard ROW_8_MASK = ROW_0_MASK << (8 * 9);
static constexpr Bitboard COL_0_MASK = [](){
    Bitboard b = 0;
    for(int r=0; r<9; ++r) b |= ((Bitboard)1 << (r*9));
    return b;
}();
static constexpr Bitboard COL_8_MASK = COL_0_MASK << 8;

// Full board mask: all 81 valid positions (bits 0-80)
static constexpr Bitboard BOARD_MASK = [](){
    Bitboard b = 0;
    for(int i=0; i<81; ++i) b |= ((Bitboard)1 << i);
    return b;
}();

// Precompute connectivity masks for a FenceGrid
// Returns {blocked_down_mask, blocked_right_mask}
static std::pair<Bitboard, Bitboard> build_wall_masks(const FenceGrid& fences) {
    Bitboard down = 0;
    Bitboard right = 0;

    // FIX: Iterate through ALL rows (0 to 8), not just to 7.
    // Row 8 needs to be checked for horizontal (Right) blocks.
    for (int r = 0; r < 9; ++r) {
        for (int c = 0; c < 9; ++c) {
            int idx = r * 9 + c;

            // Check Down: (r,c) -> (r+1,c)
            // Only possible if we are not in the last row (r < 8)
            // Use fences.blocked_down() for consistency with Pathfinder
            if (r < 8 && fences.blocked_down(r, c)) {
                down |= ((Bitboard)1 << idx);
            }

            // Check Right: (r,c) -> (r,c+1)
            // Only possible if we are not in the last column (c < 8)
            // Use fences.blocked_right() for consistency with Pathfinder
            if (c < 8 && fences.blocked_right(r, c)) {
                right |= ((Bitboard)1 << idx);
            }
        }
    }
    return {down, right};
}

bool check_reachability_fast(const FenceGrid& fences, const Player& player, uint8_t goal_row) noexcept {
    // 1. Build masks
    auto [b_down, b_right] = build_wall_masks(fences);

    // 2. Setup Bitboard BFS
    Bitboard reach = (Bitboard)1 << (player.row * 9 + player.col);
    Bitboard visited = 0;
    Bitboard goal_mask = (goal_row == 0) ? ROW_0_MASK : ROW_8_MASK;

    // 3. Flood fill
    while (reach) {
        // If any reaching bit hits the goal row, we are done
        if (reach & goal_mask) return true;

        visited |= reach;
        Bitboard next = 0;

        // Move Right: Shift Left 1
        // blocked by COL_0 wrap-around AND b_right at current position (i)
        next |= (reach << 1) & ~COL_0_MASK & ~(b_right << 1); 

        // Move Left: Shift Right 1
        // blocked by COL_8 wrap-around AND b_right at target position (i-1)
        next |= (reach >> 1) & ~COL_8_MASK & ~b_right;

        // Move Down: Shift Left 9
        // blocked by b_down at current position (i)
        next |= (reach << 9) & ~(b_down << 9);

        // Move Up: Shift Right 9
        // blocked by b_down at target position (i-9)
        next |= (reach >> 9) & ~b_down;

        // Only keep valid board positions (prevents row overflow) and unvisited squares
        reach = next & BOARD_MASK & ~visited;
    }

    return false;
}
} // namespace qbot
