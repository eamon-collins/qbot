#include "pathfinding.h"

#include <algorithm>
#include <limits>

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

struct FastNode {
    uint32_t last_seen_id = 0;
    uint8_t g_cost;
    uint8_t f_cost;
    bool in_open;
    bool in_closed;
};

struct FastContext {
    std::array<FastNode, 81> nodes;
    std::array<uint8_t, 81> open_set; // Stores indices 0-80
    uint32_t current_id = 0;
};

bool check_reachability_fast(const FenceGrid& fences, const Player& player, uint8_t goal_row) noexcept {
    static thread_local FastContext ctx;

    // Increment search generation. If 0 (overflow), reset strictly (rare).
    ctx.current_id++;
    if (ctx.current_id == 0) {
        ctx.nodes.fill(FastNode{});
        ctx.current_id = 1;
    }

    const uint32_t search_id = ctx.current_id;
    uint8_t open_count = 0;

    // Helper to add to open set
    auto add_open = [&](uint8_t idx, uint8_t g, uint8_t h) {
        FastNode& n = ctx.nodes[idx];
        n.g_cost = g;
        n.f_cost = g + h;
        n.last_seen_id = search_id;
        n.in_open = true;
        n.in_closed = false;
        ctx.open_set[open_count++] = idx;
    };

    // Initialize start
    uint8_t start_idx = player.row * 9 + player.col;
    uint8_t start_h = (player.row > goal_row) ? (player.row - goal_row) : (goal_row - player.row);
    add_open(start_idx, 0, start_h);

    while (open_count > 0) {
        // Linear scan for lowest f_cost (faster than heap for N < 20)
        int best_pos = 0;
        uint8_t best_f = ctx.nodes[ctx.open_set[0]].f_cost;

        for (int i = 1; i < open_count; ++i) {
            uint8_t f = ctx.nodes[ctx.open_set[i]].f_cost;
            if (f < best_f) {
                best_f = f;
                best_pos = i;
            }
        }

        uint8_t curr_idx = ctx.open_set[best_pos];

        // Remove from open (unordered remove: swap with last)
        ctx.nodes[curr_idx].in_open = false;
        ctx.nodes[curr_idx].in_closed = true;
        open_count--;
        ctx.open_set[best_pos] = ctx.open_set[open_count];

        // Check Goal
        uint8_t r = curr_idx / 9;
        if (r == goal_row) return true;

        uint8_t c = curr_idx % 9;
        uint8_t next_g = ctx.nodes[curr_idx].g_cost + 1;

        // Process neighbors
        auto check = [&](uint8_t n_idx, uint8_t n_r) {
            FastNode& n = ctx.nodes[n_idx];

            // If strictly new node
            if (n.last_seen_id != search_id) {
                uint8_t h = (n_r > goal_row) ? (n_r - goal_row) : (goal_row - n_r);
                add_open(n_idx, next_g, h);
                return;
            }

            // If already visited (Manhattan is consistent, so closed nodes are optimal)
            if (n.in_closed) return;

            // If in open but we found a better path (rare in unit grids, but possible)
            if (n.in_open && next_g < n.g_cost) {
                n.g_cost = next_g;
                uint8_t h = (n_r > goal_row) ? (n_r - goal_row) : (goal_row - n_r);
                n.f_cost = next_g + h;
                // Already in open_set array, just updated cost
            }
        };

        if (r > 0 && !fences.blocked_up(r, c))    check(curr_idx - 9, r - 1);
        if (r < 8 && !fences.blocked_down(r, c))  check(curr_idx + 9, r + 1);
        if (c > 0 && !fences.blocked_left(r, c))  check(curr_idx - 1, r);
        if (c < 8 && !fences.blocked_right(r, c)) check(curr_idx + 1, r);
    }

    return false;
}

} // namespace qbot
