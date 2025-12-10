#include "StateNode.h"
#include "node_pool.h"

#include <array>
#include <queue>

namespace qbot {

std::vector<Move> StateNode::generate_valid_moves(size_t* out_fence_count) const noexcept {
    std::vector<Move> moves;
    moves.reserve(140);  // Rough upper bound: ~4 pawn + ~128 fence moves max

    const Player& curr = current_player();
    const Player& opp = opponent_player();
    const uint8_t r = curr.row;
    const uint8_t c = curr.col;

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
        // Horizontal fences: intersection row in [0,7], col in [0,6]
        // Fence at (row, col) blocks movement between rows row and row+1 at columns col and col+1
        for (uint8_t row = 0; row < 8; ++row) {
            for (uint8_t col = 0; col < 7; ++col) {
                if (!fences.h_fence_blocked(row, col)) {
                    moves.push_back(Move::fence(row, col, true));
                }
            }
        }

        // Vertical fences: intersection row in [0,6], col in [0,7]
        // Fence at (row, col) blocks movement between cols col and col+1 at rows row and row+1
        for (uint8_t row = 0; row < 7; ++row) {
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

bool StateNode::can_reach_goal(bool check_p1) const noexcept {
    const Player& player = check_p1 ? p1 : p2;
    const uint8_t goal_row = check_p1 ? 8 : 0;

    // BFS to find if player can reach goal row
    // visited[row][col] tracks visited squares
    std::array<std::array<bool, BOARD_SIZE>, BOARD_SIZE> visited{};
    std::queue<std::pair<uint8_t, uint8_t>> queue;

    queue.push({player.row, player.col});
    visited[player.row][player.col] = true;

    while (!queue.empty()) {
        auto [r, c] = queue.front();
        queue.pop();

        if (r == goal_row) {
            return true;
        }

        // Try all four directions
        // Up
        if (!fences.blocked_up(r, c) && !visited[r - 1][c]) {
            visited[r - 1][c] = true;
            queue.push({static_cast<uint8_t>(r - 1), c});
        }
        // Down
        if (!fences.blocked_down(r, c) && !visited[r + 1][c]) {
            visited[r + 1][c] = true;
            queue.push({static_cast<uint8_t>(r + 1), c});
        }
        // Left
        if (!fences.blocked_left(r, c) && !visited[r][c - 1]) {
            visited[r][c - 1] = true;
            queue.push({r, static_cast<uint8_t>(c - 1)});
        }
        // Right
        if (!fences.blocked_right(r, c) && !visited[r][c + 1]) {
            visited[r][c + 1] = true;
            queue.push({r, static_cast<uint8_t>(c + 1)});
        }
    }

    return false;
}

bool StateNode::is_fence_move_valid(Move fence_move) const noexcept {
    assert(fence_move.is_fence());

    // Copy just the fence grid and apply the move
    FenceGrid temp_fences = fences;

    if (fence_move.is_horizontal()) {
        temp_fences.place_h_fence(fence_move.row(), fence_move.col());
    } else {
        temp_fences.place_v_fence(fence_move.row(), fence_move.col());
    }

    // Check if both players can still reach their goals with the new fence
    // P1 needs to reach row 8, P2 needs to reach row 0
    auto can_reach = [&temp_fences](const Player& player, uint8_t goal_row) -> bool {
        std::array<std::array<bool, BOARD_SIZE>, BOARD_SIZE> visited{};
        std::queue<std::pair<uint8_t, uint8_t>> queue;

        queue.push({player.row, player.col});
        visited[player.row][player.col] = true;

        while (!queue.empty()) {
            auto [r, c] = queue.front();
            queue.pop();

            if (r == goal_row) {
                return true;
            }

            // Up
            if (!temp_fences.blocked_up(r, c) && !visited[r - 1][c]) {
                visited[r - 1][c] = true;
                queue.push({static_cast<uint8_t>(r - 1), c});
            }
            // Down
            if (!temp_fences.blocked_down(r, c) && !visited[r + 1][c]) {
                visited[r + 1][c] = true;
                queue.push({static_cast<uint8_t>(r + 1), c});
            }
            // Left
            if (!temp_fences.blocked_left(r, c) && !visited[r][c - 1]) {
                visited[r][c - 1] = true;
                queue.push({r, static_cast<uint8_t>(c - 1)});
            }
            // Right
            if (!temp_fences.blocked_right(r, c) && !visited[r][c + 1]) {
                visited[r][c + 1] = true;
                queue.push({r, static_cast<uint8_t>(c + 1)});
            }
        }
        return false;
    };

    return can_reach(p1, 8) && can_reach(p2, 0);
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

    for (const Move& move : moves) {
        // For fence moves, validate that it doesn't block either player
        if (move.is_fence() && !is_fence_move_valid(move)) {
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
