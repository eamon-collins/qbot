#pragma once

#include "../tree/StateNode.h"

#include <array>
#include <cstdint>
#include <vector>

namespace qbot {

/// Coordinate on the board
struct Coord {
    uint8_t row;
    uint8_t col;

    constexpr bool operator==(const Coord& o) const noexcept {
        return row == o.row && col == o.col;
    }
};

/// A* pathfinder optimized for Quoridor board
///
/// Retains internal buffers to avoid repeated allocations.
/// Thread-local instances recommended for concurrent use.
class Pathfinder {
public:
    Pathfinder();

    /// Check if both players can reach their goals
    /// @param state Game state to check
    /// @return true if neither player is blocked
    [[nodiscard]] bool check_paths(const StateNode& state) noexcept;

    /// Check if both players can reach their goals with a hypothetical fence
    /// @param state Current game state
    /// @param fence_move Fence move to test (must be a fence move)
    /// @return true if neither player would be blocked
    [[nodiscard]] bool check_paths_with_fence(const StateNode& state, Move fence_move) noexcept;
    [[nodiscard]] bool check_player_path_with_fence(const StateNode& state, Move fence_move, bool player1) noexcept;

    /// Check if a specific player can reach their goal
    /// @param fences Fence grid to use
    /// @param player Player to check
    /// @param goal_row Target row (8 for P1, 0 for P2)
    /// @return true if player can reach goal
    [[nodiscard]] bool can_reach(const FenceGrid& fences, const Player& player, uint8_t goal_row) noexcept;

    /// Find shortest path for current player to their goal using A*
    /// @param state Game state
    /// @return Path as list of coordinates (empty if no path exists)
    [[nodiscard]] std::vector<Coord> find_path(const StateNode& state) noexcept;

    /// Find shortest path for a specific player
    /// @param fences Fence grid
    /// @param player Player to pathfind for
    /// @param goal_row Target row
    /// @return Path as list of coordinates (empty if no path exists)
    [[nodiscard]] std::vector<Coord> find_path(const FenceGrid& fences, const Player& player, uint8_t goal_row) noexcept;

    /// Get the path length (distance) without computing full path
    /// Uses A* but stops at goal without reconstructing path
    /// @return Distance to goal, or -1 if unreachable
    [[nodiscard]] int path_length(const FenceGrid& fences, const Player& player, uint8_t goal_row) noexcept;

private:
    /// A* node state
    struct Node {
        uint8_t g_cost;     // Cost from start
        uint8_t f_cost;     // g + heuristic
        uint8_t parent_row;
        uint8_t parent_col;
        bool in_open;
        bool in_closed;
        // uint32_t generated_id{0};
    };

    /// Heuristic: Manhattan distance to goal row (admissible for grid)
    [[nodiscard]] static constexpr uint8_t heuristic(uint8_t row, uint8_t goal_row) noexcept {
        return row > goal_row ? row - goal_row : goal_row - row;
    }

    /// Reset internal state for new search
    void reset() noexcept;

    /// Get neighbors of a cell respecting fences
    /// Returns count of valid neighbors, fills neighbors array
    [[nodiscard]] inline int get_neighbors(const FenceGrid fences, uint8_t row, uint8_t col,
                                  std::array<Coord, 4>& neighbors) const noexcept {
        int count = 0;

        // Up
        if (!fences.blocked_up(row, col)) {
            neighbors[count++] = {static_cast<uint8_t>(row - 1), col};
        }
        // Down
        if (!fences.blocked_down(row, col)) {
            neighbors[count++] = {static_cast<uint8_t>(row + 1), col};
        }
        // Left
        if (!fences.blocked_left(row, col)) {
            neighbors[count++] = {row, static_cast<uint8_t>(col - 1)};
        }
        // Right
        if (!fences.blocked_right(row, col)) {
            neighbors[count++] = {row, static_cast<uint8_t>(col + 1)};
        }

        return count;
    }

    // Reusable buffers
    std::array<std::array<Node, BOARD_SIZE>, BOARD_SIZE> nodes_;

    // Open set as a simple array-based priority queue
    // For 9x9 board, linear scan is competitive with heap
    std::array<Coord, BOARD_SIZE * BOARD_SIZE> open_set_;
    size_t open_count_;
    uint32_t search_id_{0};
};

/// Thread-local pathfinder instance for convenient access
Pathfinder& get_pathfinder() noexcept;

//checks for if one player can jump the other on the path.
// returns 0 for no jump, 1 for p1 jumps, -1 for p2 jumps
[[nodiscard]] int check_jump_advantage(
    const std::vector<Coord>& p1_path,
    const std::vector<Coord>& p2_path,
    bool p1_moves_first) noexcept;

} // namespace qbot
