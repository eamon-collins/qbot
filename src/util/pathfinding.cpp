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
static constexpr Bitboard ROW_8_MASK = ROW_0_MASK << (8 * 9); // Bits 72-80
static constexpr Bitboard COL_0_MASK = [](){
    Bitboard b = 0;
    for(int r=0; r<9; ++r) b |= ((Bitboard)1 << (r*9));
    return b;
}();
static constexpr Bitboard COL_8_MASK = COL_0_MASK << 8;

// Helper to find the index of the first set bit (Least Significant Bit)
// Needed to pick a specific tile when backtracking.
static inline int get_lsb_index(Bitboard bb) {
    uint64_t lo = static_cast<uint64_t>(bb);
    if (lo) return __builtin_ctzll(lo);
    return 64 + __builtin_ctzll(static_cast<uint64_t>(bb >> 64));
}

static std::pair<Bitboard, Bitboard> build_wall_masks_fast(const FenceGrid& fences) {
    Bitboard b_down = 0;
    Bitboard b_right = 0;

    uint64_t h = fences.horizontal;
    uint64_t v = fences.vertical;

    for (int r = 0; r < 9; ++r) {
        int shift = r * 9;

        // --- Build Right Blockers ---
        // Blocked if V-fence at (r,c) OR (r-1,c)
        // Note: v-fence at row 8 (index 7) doesn't exist (v is 8x8), 
        // but row 8 board checks row 7 v-fences via v_prev.
        uint8_t v_curr = (r < 8) ? ((v >> (r * 8)) & 0xFF) : 0;
        uint8_t v_prev = (r > 0) ? ((v >> ((r - 1) * 8)) & 0xFF) : 0;

        b_right |= ((Bitboard)(v_curr | v_prev) << shift);

        // --- Build Down Blockers ---
        if (r < 8) {
            // Standard rows 0-7: Blocked by H-fence at (r,c) or (r,c-1)
            uint8_t h_row = (h >> (r * 8)) & 0xFF;
            // (h_row << 1) aligns "left" fence to current col
            // Cast to Bitboard needed before shift to avoid overflow if h_row | ... exceeds 64 bits (unlikely here but good practice)
            // or simply to match types. 
            // Note: (h_row | (h_row << 1)) produces a 9-bit pattern (cols 0-8).
            b_down |= ((Bitboard)(h_row | (h_row << 1)) << shift);
        } else {
            // FIX: Row 8 (Bottom Edge)
            // Must explicitly block going down from the last row to prevent 
            // the bitboard from shifting into "ghost" rows and tunneling.
            b_down |= ((Bitboard)0x1FF << shift);
        }
    }

    return {b_down, b_right};
}

std::vector<Coord> find_path_bitboard(const FenceGrid& fences, const Player& player, uint8_t goal_row) noexcept {
    // 1. Setup
    auto [b_down, b_right] = build_wall_masks_fast(fences);

    // Max path length is 81. Use stack array for speed.
    // history[i] stores the "frontier" (newly reached nodes) at step i.
    std::array<Bitboard, 82> history;

    int start_idx = player.row * 9 + player.col;
    Bitboard current_frontier = (Bitboard)1 << start_idx;
    Bitboard visited = current_frontier;
    Bitboard goal_mask = (goal_row == 0) ? ROW_0_MASK : (ROW_0_MASK << (8 * 9));

    history[0] = current_frontier;
    int depth = 0;
    bool found = false;

    // 2. Forward Search (BFS)
    // We limit loop to 81 because no shortest path can be longer than the number of squares.
    for (depth = 0; depth < 81; ++depth) {
        if (current_frontier & goal_mask) {
            found = true;
            break;
        }

        Bitboard next = 0;

        // Move Right (i -> i+1): Blocked by b_right at i
        next |= (current_frontier << 1) & ~COL_0_MASK & ~(b_right << 1); 

        // Move Left (i -> i-1): Blocked by b_right at i-1
        next |= (current_frontier >> 1) & ~COL_8_MASK & ~b_right;

        // Move Down (i -> i+9): Blocked by b_down at i
        next |= (current_frontier << 9) & ~(b_down << 9);

        // Move Up (i -> i-9): Blocked by b_down at i-9
        next |= (current_frontier >> 9) & ~b_down;

        // Only keep strictly NEW nodes
        current_frontier = next & ~visited;

        if (current_frontier == 0) return {}; // No path found

        visited |= current_frontier;
        history[depth + 1] = current_frontier;
    }

    if (!found) return {};

    // 3. Backtrack to reconstruct path
    // Pick specific goal node (first bit set in the intersection)
    int curr_idx = get_lsb_index(current_frontier & goal_mask);

    std::vector<Coord> path;
    path.reserve(depth + 1);

    // Add goal
    path.push_back({static_cast<uint8_t>(curr_idx / 9), static_cast<uint8_t>(curr_idx % 9)});

    // Walk back from depth-1 to 0
    for (int d = depth - 1; d >= 0; --d) {
        Bitboard prev_frontier = history[d];

        // We need to find a neighbor 'prev' in 'prev_frontier' that can reach 'curr_idx'
        // We check all 4 potential predecessors.

        // Potential Prev: UP (curr - 9) -> Moved Down to get here
        if (curr_idx >= 9) {
            int prev = curr_idx - 9;
            // Check if prev was in the frontier AND move Down was valid (not blocked by b_down at prev)
            if ((prev_frontier & ((Bitboard)1 << prev)) && !((b_down >> prev) & 1)) {
                curr_idx = prev;
                goto push_node;
            }
        }

        // Potential Prev: DOWN (curr + 9) -> Moved Up to get here
        if (curr_idx < 72) {
            int prev = curr_idx + 9;
            // Check if prev was in frontier AND move Up was valid (not blocked by b_down at curr)
            if ((prev_frontier & ((Bitboard)1 << prev)) && !((b_down >> curr_idx) & 1)) {
                curr_idx = prev;
                goto push_node;
            }
        }

        // Potential Prev: LEFT (curr - 1) -> Moved Right to get here
        if ((curr_idx % 9) > 0) {
            int prev = curr_idx - 1;
            // Check if prev was in frontier AND move Right was valid (not blocked by b_right at prev)
            if ((prev_frontier & ((Bitboard)1 << prev)) && !((b_right >> prev) & 1)) {
                curr_idx = prev;
                goto push_node;
            }
        }

        // Potential Prev: RIGHT (curr + 1) -> Moved Left to get here
        if ((curr_idx % 9) < 8) {
            int prev = curr_idx + 1;
            // Check if prev was in frontier AND move Left was valid (not blocked by b_right at curr)
            if ((prev_frontier & ((Bitboard)1 << prev)) && !((b_right >> curr_idx) & 1)) {
                curr_idx = prev;
                goto push_node;
            }
        }

        push_node:
        path.push_back({static_cast<uint8_t>(curr_idx / 9), static_cast<uint8_t>(curr_idx % 9)});
    }

    // Path is currently Goal -> Start. Reverse it.
    std::reverse(path.begin(), path.end());
    return path;
}

bool check_reachability_fast(const FenceGrid& fences, const Player& player, uint8_t goal_row) noexcept {
    auto [b_down, b_right] = build_wall_masks_fast(fences);

    Bitboard reach = (Bitboard)1 << (player.row * 9 + player.col);
    Bitboard visited = 0;
    Bitboard goal_mask = (goal_row == 0) ? ROW_0_MASK : ROW_8_MASK;

    while (reach) {
        if (reach & goal_mask) return true;

        visited |= reach;
        Bitboard next = 0;

        // Move Right: shift left 1, mask out col 0 wrap-around and right-walls
        next |= (reach << 1) & ~COL_0_MASK & ~(b_right << 1); 

        // Move Left: shift right 1, mask out col 8 wrap-around and right-walls (at dest)
        next |= (reach >> 1) & ~COL_8_MASK & ~b_right;

        // Move Down: shift left 9, mask out down-walls
        // The fix in build_wall_masks_fast ensures b_down prevents shifting off-board here
        next |= (reach << 9) & ~(b_down << 9);

        // Move Up: shift right 9, mask out down-walls (at dest)
        next |= (reach >> 9) & ~b_down;

        reach = next & ~visited;
    }

    return false;
}
} // namespace qbot
