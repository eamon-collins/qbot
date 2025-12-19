#include "StateNode.h"
#include "node_pool.h"
#include "../util/pathfinding.h"

#include <iostream>
#include <sstream>

namespace qbot {

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
    ss << "Visits: " << v << "  Q: " << q << "  Ply: " << ply
       << "  Turn: P" << (is_p1_to_move() ? "1" : "2") << "\n";

    // Print P1/P2 info
    ss << "P1: (" << static_cast<int>(p1.row) << "," << static_cast<int>(p1.col)
       << ") fences=" << static_cast<int>(p1.fences) << "   "
       << "P2: (" << static_cast<int>(p2.row) << "," << static_cast<int>(p2.col)
       << ") fences=" << static_cast<int>(p2.fences) << "\n";

    if (is_terminal()) {
        ss << "TERMINAL: " << (terminal_value > 0 ? "P1 wins" : "P2 wins") << "\n";
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

    // Optimization: when both players are out of fences, the game is deterministic.
    // Each player should just race to their goal via shortest path.
    if (p1.fences == 0 && p2.fences == 0) {
        if (out_fence_count) *out_fence_count = 0;

        Pathfinder& pf = get_pathfinder();
        uint8_t goal_row = is_p1_to_move() ? 8 : 0;
        auto path = pf.find_path(fences, curr, goal_row);

        if (path.size() > 1) {
            moves.push_back(Move::pawn(path[1].row, path[1].col));
        }
        return moves;
    }

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

size_t StateNode::generate_valid_children(NodePool& pool, uint32_t my_index) noexcept {
    if (is_terminal() || is_expanded()) {
        return 0;
    }

    std::vector<Move> moves = generate_valid_moves();
    if (moves.empty()) {
        return 0;
    }

    // Track children indices for linking
    std::vector<uint32_t> child_indices;
    child_indices.reserve(moves.size());

    Pathfinder& pf = get_pathfinder();

    for (const Move& move : moves) {
        // For fence moves, validate that it doesn't block either player
        if (move.is_fence() && !pf.check_paths_with_fence(*this, move)) {
            continue;
        }

        uint32_t child_idx = pool.allocate();
        if (child_idx == NULL_NODE) {
            // Allocation failed - rollback all allocated children
            for (uint32_t idx : child_indices) {
                pool.deallocate(idx);
            }
            return 0;
        }

        // Initialize child from this parent's state with move applied
        pool[child_idx].init_from_parent(*this, move, my_index);
        child_indices.push_back(child_idx);
    }

    if (child_indices.empty()) {
        return 0;
    }

    // Link children as siblings (left-child right-sibling representation)
    for (size_t i = 0; i + 1 < child_indices.size(); ++i) {
        pool[child_indices[i]].next_sibling = child_indices[i + 1];
    }
    pool[child_indices.back()].next_sibling = NULL_NODE;

    // Set first child and mark as expanded
    first_child = child_indices.front();
    set_expanded();

    return child_indices.size();
}

} // namespace qbot
