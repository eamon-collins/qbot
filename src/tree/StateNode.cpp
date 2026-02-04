#include "StateNode.h"
#include "node_pool.h"
#include "../util/pathfinding.h"
#include "../util/timer.h"

#include <iostream>
#include <sstream>

namespace qbot {

// Thread-local pool pointer definition
thread_local NodePool* StateNode::pool_ = nullptr;

void StateNode::print_node() const noexcept {
    std::ostringstream ss;

    // Print move info
    ss << "Move: ";
    if (!move.is_valid()) {
        ss << "(root)";
    } else if (move.is_pawn()) {
        ss << "pawn to (" << static_cast<int>(move.row()) << "," << static_cast<int>(move.col()) << ")";
    } else {
        ss << "fence " << (move.is_horizontal() ? "H" : "V")
           << " at (" << static_cast<int>(move.row()) << "," << static_cast<int>(move.col()) << ")";
    }
    ss << "\n";

    // Print stats
    uint32_t v = stats.visits.load(std::memory_order_relaxed);
    float q = stats.Q();
    ss << "Visits: " << v << "  Q: " << q
       << "  Turn: P" << (is_p1_to_move() ? "1" : "2") << "\n";

    // Print P1/P2 info
    ss << "P1: (" << static_cast<int>(p1.row) << "," << static_cast<int>(p1.col)
       << ") fences=" << static_cast<int>(p1.fences) << "   "
       << "P2: (" << static_cast<int>(p2.row) << "," << static_cast<int>(p2.col)
       << ") fences=" << static_cast<int>(p2.fences) << "\n";

    if (is_terminal()) {
        ss << "TERMINAL: " << (!is_p1_to_move() ? "P1 wins" : "P2 wins") << "\n";
    }

    // Draw board: rows go from 8 (top) down to 0 (bottom)
    // Interleave fence rows between square rows
    for (int row = 8; row >= 0; --row) {
        // Draw square row
        for (int col = 0; col < 9; ++col) {
            // Draw cell
            if (p1.row == row && p1.col == col) {
                ss << 'H';  // Player 1 (Human in vs-human games)
            } else if (p2.row == row && p2.col == col) {
                ss << 'B';  // Player 2 (Bot in vs-human games)
            } else {
                ss << '.';  // Empty square
            }

            // Draw vertical fence to the right (if any)
            if (col < 8) {
                // Vertical fence at intersection (r, col) blocks between col and col+1
                // Check both fence segments that could block this edge
                bool blocked = false;
                if (row > 0 && fences.has_v_fence(row - 1, col)) blocked = true;
                if (row < 8 && fences.has_v_fence(row, col)) blocked = true;
                ss << (blocked ? '|' : ' ');
            }
        }
        ss << "\n";

        // Draw horizontal fence row (between this row and row-1)
        if (row > 0) {
            for (int col = 0; col < 9; ++col) {
                // Horizontal fence at intersection (row-1, c) blocks between row-1 and row
                bool blocked = false;
                if (col > 0 && fences.has_h_fence(row - 1, col - 1)) blocked = true;
                if (col < 8 && fences.has_h_fence(row - 1, col)) blocked = true;

                ss << (blocked ? '-' : ' ');

                // Gap between horizontal segments (fence intersections)
                if (col < 8) {
                    ss << ' ';
                }
            }
            ss << "\n";
        }
    }
    ss << "\n";

    std::cout << ss.str();
}

std::vector<Move> StateNode::generate_valid_moves(size_t* out_fence_count) const noexcept {
    std::vector<Move> moves;

    const Player& curr = current_player();
    const Player& opp = opponent_player();
    const uint8_t r = curr.row;
    const uint8_t c = curr.col;

    // // When both players are out of fences, the game is deterministic -
    // // each player should just race to their goal via shortest path.
    // // This significantly speeds up endgame by reducing branching factor to 1.
    // if (p1.fences == 0 && p2.fences == 0) {
    //     if (out_fence_count) *out_fence_count = 0;
    //
    //     Pathfinder& pf = get_pathfinder();
    //     uint8_t goal_row = is_p1_to_move() ? 8 : 0;
    //     auto path = pf.find_path(fences, curr, goal_row);
    //
    //     if (path.size() > 1) {
    //         moves.push_back(Move::pawn(path[1].row, path[1].col));
    //     }
    //     return moves;
    // }

    moves.reserve(140);  // Rough upper bound: ~4 pawn + ~128 fence moves max

    // ========== PAWN MOVES ==========
    // Helper lambdas to check if movement is possible (using FenceGrid methods)
    auto can_move_up = [&](uint8_t from_row, uint8_t from_col) -> bool {
        return !fences.blocked_up(from_row, from_col);
    };

    auto can_move_down = [&](uint8_t from_row, uint8_t from_col) -> bool {
        return !fences.blocked_down(from_row, from_col);
    };

    auto can_move_left = [&](uint8_t from_row, uint8_t from_col) -> bool {
        return !fences.blocked_left(from_row, from_col);
    };

    auto can_move_right = [&](uint8_t from_row, uint8_t from_col) -> bool {
        return !fences.blocked_right(from_row, from_col);
    };

    // Check each direction for basic moves and jump moves
    // UP (row - 1)
    if (can_move_up(r, c)) {
        if (opp.row == r - 1 && opp.col == c) {
            // Opponent directly above - try to jump
            if (can_move_up(r - 1, c)) {
                // Straight jump over opponent
                moves.push_back(Move::pawn(r - 2, c));
            } else {
                // Wall behind opponent or board edge - diagonal jumps allowed
                if (can_move_left(r - 1, c)) {
                    moves.push_back(Move::pawn(r - 1, c - 1));
                }
                if (can_move_right(r - 1, c)) {
                    moves.push_back(Move::pawn(r - 1, c + 1));
                }
            }
        } else {
            // Normal move up
            moves.push_back(Move::pawn(r - 1, c));
        }
    }

    // DOWN (row + 1)
    if (can_move_down(r, c)) {
        if (opp.row == r + 1 && opp.col == c) {
            // Opponent directly below - try to jump
            if (can_move_down(r + 1, c)) {
                // Straight jump over opponent
                moves.push_back(Move::pawn(r + 2, c));
            } else {
                // Wall behind opponent or board edge - diagonal jumps allowed
                if (can_move_left(r + 1, c)) {
                    moves.push_back(Move::pawn(r + 1, c - 1));
                }
                if (can_move_right(r + 1, c)) {
                    moves.push_back(Move::pawn(r + 1, c + 1));
                }
            }
        } else {
            // Normal move down
            moves.push_back(Move::pawn(r + 1, c));
        }
    }

    // LEFT (col - 1)
    if (can_move_left(r, c)) {
        if (opp.row == r && opp.col == c - 1) {
            // Opponent directly left - try to jump
            if (can_move_left(r, c - 1)) {
                // Straight jump over opponent
                moves.push_back(Move::pawn(r, c - 2));
            } else {
                // Wall behind opponent or board edge - diagonal jumps allowed
                if (can_move_up(r, c - 1)) {
                    moves.push_back(Move::pawn(r - 1, c - 1));
                }
                if (can_move_down(r, c - 1)) {
                    moves.push_back(Move::pawn(r + 1, c - 1));
                }
            }
        } else {
            // Normal move left
            moves.push_back(Move::pawn(r, c - 1));
        }
    }

    // RIGHT (col + 1)
    if (can_move_right(r, c)) {
        if (opp.row == r && opp.col == c + 1) {
            // Opponent directly right - try to jump
            if (can_move_right(r, c + 1)) {
                // Straight jump over opponent
                moves.push_back(Move::pawn(r, c + 2));
            } else {
                // Wall behind opponent or board edge - diagonal jumps allowed
                if (can_move_up(r, c + 1)) {
                    moves.push_back(Move::pawn(r - 1, c + 1));
                }
                if (can_move_down(r, c + 1)) {
                    moves.push_back(Move::pawn(r + 1, c + 1));
                }
            }
        } else {
            // Normal move right
            moves.push_back(Move::pawn(r, c + 1));
        }
    }

    size_t pawn_move_count = moves.size();

    // ========== FENCE MOVES ==========
    // Only if current player has fences remaining
    if (curr.fences > 0) {
        // Horizontal fences: intersection at (row, col), fence spans to (row, col+1)
        // Fence blocks movement between rows row and row+1 at columns col and col+1
        for (uint8_t row = 0; row < 8; ++row) {
            for (uint8_t col = 0; col < 8; ++col) {
                if (!fences.h_fence_blocked(row, col)) {
                    moves.push_back(Move::fence(row, col, true));
                }
            }
        }

        // Vertical fences: intersection at (row, col), fence spans to (row+1, col)
        // Fence blocks movement between cols col and col+1 at rows row and row+1
        for (uint8_t row = 0; row < 8; ++row) {
            for (uint8_t col = 0; col < 8; ++col) {
                if (!fences.v_fence_blocked(row, col)) {
                    moves.push_back(Move::fence(row, col, false));
                }
            }
        }
    }

    if (out_fence_count) {
        *out_fence_count = moves.size() - pawn_move_count;
    }

    return moves;
}

[[nodiscard]] inline bool path_intersects_fence(const std::vector<Coord>& path, Move fence) noexcept {
    // A path of 0 or 1 node has no edges to cut
    if (path.size() < 2) return false;

    // Unpack fence data once to avoid repeated bit-shifting accessors in the loop
    const uint8_t f_r = fence.row();
    const uint8_t f_c = fence.col();

    if (fence.is_horizontal()) {
        // CASE: Horizontal Fence at (f_r, f_c)
        // Blocks: VERTICAL movement between rows [f_r] and [f_r+1]
        // Spans:  Columns [f_c] and [f_c+1]

        for (size_t i = 0; i < path.size() - 1; ++i) {
            // Optimization: Filter by Column first (cheapest check)
            // The fence only exists at columns f_c and f_c+1.
            // If the path step isn't on these columns, it can't be blocked.
            const uint8_t u_c = path[i].col;
            if (u_c != f_c && u_c != f_c + 1) continue;

            // Filter by Movement Type
            // Horizontal fence only blocks vertical moves (col must not change)
            if (u_c != path[i+1].col) continue;

            // Exact Crossing Check
            // We are blocked if we step from f_r to f_r+1 (or vice versa).
            // min(row1, row2) must equal f_r.
            const uint8_t u_r = path[i].row;
            const uint8_t v_r = path[i+1].row;

            if (std::min(u_r, v_r) == f_r) {
                return true; 
            }
        }
    } else {
        // CASE: Vertical Fence at (f_r, f_c)
        // Blocks: HORIZONTAL movement between cols [f_c] and [f_c+1]
        // Spans:  Rows [f_r] and [f_r+1]

        for (size_t i = 0; i < path.size() - 1; ++i) {
            // Optimization: Filter by Row first
            // The fence only exists at rows f_r and f_r+1.
            const uint8_t u_r = path[i].row;
            if (u_r != f_r && u_r != f_r + 1) continue;

            // Filter by Movement Type
            // Vertical fence only blocks horizontal moves (row must not change)
            if (u_r != path[i+1].row) continue;

            // Exact Crossing Check
            // We are blocked if we step from f_c to f_c+1 (or vice versa).
            // min(col1, col2) must equal f_c.
            const uint8_t u_c = path[i].col;
            const uint8_t v_c = path[i+1].col;

            if (std::min(u_c, v_c) == f_c) {
                return true;
            }
        }
    }

    return false;
}


size_t StateNode::generate_valid_children() noexcept {
    if (is_terminal() || is_expanded()) return 0;

    auto& timers = get_timers();
    ScopedTimer timer_whole(timers.setup_gen);

    std::vector<Move> moves = generate_valid_moves();
    if (moves.empty()) return 0;

    NodePool& p = pool();

    Move valid_moves[256]; 
    int valid_count = 0;

    {
        ScopedTimer timer(timers.pathfinding);
        Pathfinder& pf = get_pathfinder();

        std::vector<Coord> p1_path = pf.find_path(fences, p1, 8);
        std::vector<Coord> p2_path = pf.find_path(fences, p2, 0);
        auto [p1_h_block, p1_v_block] = compute_path_blockers(p1_path);
        auto [p2_h_block, p2_v_block] = compute_path_blockers(p2_path);

        for (const Move& m : moves) {
            bool is_valid = true;

            if (m.is_fence()) {
                uint8_t fr = m.row();
                uint8_t fc = m.col();
                uint8_t f_idx = fr * 8 + fc;
                uint64_t mask = (1ULL << f_idx);

                bool cuts_p1 = false;
                bool cuts_p2 = false;

                if (m.is_horizontal()) {
                    if (p1_h_block & mask) cuts_p1 = true;
                    if (p2_h_block & mask) cuts_p2 = true;
                } else {
                    if (p1_v_block & mask) cuts_p1 = true;
                    if (p2_v_block & mask) cuts_p2 = true;
                }

                if (cuts_p1 || cuts_p2) {
                    FenceGrid tmp_fences = fences;
                    if (m.is_horizontal()) tmp_fences.place_h_fence(fr, fc);
                    else tmp_fences.place_v_fence(fr, fc);

                    if (cuts_p1 && cuts_p2 &&
                            (!check_reachability_fast(tmp_fences, p1, 8) ||
                            !check_reachability_fast(tmp_fences, p2, 0))) {
                        is_valid = false;
                    } else if (cuts_p1 && !check_reachability_fast(tmp_fences, p1, 8)) {
                        is_valid = false;
                    } else if (cuts_p2 && !check_reachability_fast(tmp_fences, p2, 0)) {
                        is_valid = false;
                    }
                }
            }

            if (is_valid) {
                valid_moves[valid_count++] = m;
            }
        }
    }

    size_t child_count = 0;
    uint32_t child_indices[256];
    {
        // ScopedTimer timer(timers.allocation); 

        //maybe should batch allocate
        for (int i = 0; i < valid_count; ++i) {
            uint32_t child_idx = p.allocate();
            if (child_idx == NULL_NODE) break;

            p[child_idx].init_from_parent(*this, valid_moves[i], self_index);
            child_indices[child_count++] = child_idx;
        }
    }

    if (child_count == 0) return 0;

    float uniform_prior = 1.0f / static_cast<float>(child_count);
    for (size_t i = 0; i + 1 < child_count; ++i) {
        p[child_indices[i]].next_sibling = child_indices[i + 1];
        p[child_indices[i]].stats.prior = uniform_prior;
    }
    p[child_indices[child_count - 1]].next_sibling = NULL_NODE;
    p[child_indices[child_count - 1]].stats.prior = uniform_prior;

    first_child = child_indices[0];
    set_expanded();

    return child_count;
}

uint32_t StateNode::find_or_create_child(Move move) noexcept {
    NodePool& p = pool();

    // First check if move exists among existing children
    uint32_t child = first_child;
    while (child != NULL_NODE) {
        if (p[child].move == move) {
            return child;
        }
        child = p[child].next_sibling;
    }

    // If not expanded yet, expand and then find the child
    if (!is_expanded()) {
        generate_valid_children();
    }

    // Now search again
    child = first_child;
    while (child != NULL_NODE) {
        if (p[child].move == move) {
            return child;
        }
        child = p[child].next_sibling;
    }

    return NULL_NODE;
}

} // namespace qbot
