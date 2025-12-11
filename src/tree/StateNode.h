#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <cassert>
#include <vector>

namespace qbot {

/// Board dimensions - Quoridor is played on a 9x9 grid
inline constexpr uint8_t BOARD_SIZE = 9;
inline constexpr uint8_t NUM_FENCES = 10;  // Each player starts with 10 fences

/// Sentinel value indicating no node / invalid index
inline constexpr uint32_t NULL_NODE = std::numeric_limits<uint32_t>::max();

/// Compact move representation for Quoridor (fits in 16 bits)
/// Pawn moves: type=0, row/col in [0,8]
/// Fence moves: type=1, row in [0,7], col in [0,7], horizontal bit
struct Move {
    uint16_t data{0xFFFF};  // 0xFFFF = invalid/unset move

    constexpr Move() = default;

    /// Create a pawn move to (row, col)
    [[nodiscard]] static constexpr Move pawn(uint8_t row, uint8_t col) noexcept {
        assert(row < 9 && col < 9);
        Move m;
        m.data = static_cast<uint16_t>((row << 4) | col);
        return m;
    }

    /// Create a fence move at (row, col), horizontal or vertical
    [[nodiscard]] static constexpr Move fence(uint8_t row, uint8_t col, bool horizontal) noexcept {
        assert(row < 8 && col < 8);
        Move m;
        m.data = static_cast<uint16_t>(0x8000 | (horizontal ? 0x4000 : 0) | (row << 4) | col);
        return m;
    }

    [[nodiscard]] constexpr bool is_pawn() const noexcept { return (data & 0x8000) == 0; }
    [[nodiscard]] constexpr bool is_fence() const noexcept { return (data & 0x8000) != 0; }
    [[nodiscard]] constexpr bool is_horizontal() const noexcept { return (data & 0x4000) != 0; }
    [[nodiscard]] constexpr bool is_valid() const noexcept { return data != 0xFFFF; }
    [[nodiscard]] constexpr uint8_t row() const noexcept { return (data >> 4) & 0xF; }
    [[nodiscard]] constexpr uint8_t col() const noexcept { return data & 0xF; }

    constexpr bool operator==(const Move& other) const noexcept { return data == other.data; }
    constexpr bool operator!=(const Move& other) const noexcept { return data != other.data; }
};

static_assert(sizeof(Move) == 2, "Move should be 2 bytes");

/// Compact player state (4 bytes)
struct Player {
    uint8_t row;        // [0,8] - pawn row position
    uint8_t col;        // [0,8] - pawn column position
    uint8_t fences;     // remaining fences [0,10]
    uint8_t padding{0};

    constexpr Player() : row(0), col(4), fences(NUM_FENCES) {}
    constexpr Player(uint8_t r, uint8_t c, uint8_t f) : row(r), col(c), fences(f) {}
};

static_assert(sizeof(Player) == 4, "Player should be 4 bytes");

/// Fence grid representation using bitsets
///
/// Board layout (9x9 squares with 8x8 fence intersection points):
///
///   Squares are indexed [0,8] x [0,8]
///   Fence intersection points are indexed [0,7] x [0,7]
///   Each fence covers 2 squares worth of edge
///
/// Horizontal fence at (r, c): blocks vertical movement between rows r and r+1
///   at columns c and c+1. Stored as segment at (r, c).
///
/// Vertical fence at (r, c): blocks horizontal movement between columns c and c+1
///   at rows r and r+1. Stored as segment at (r, c).
///
/// A fence occupies 2 consecutive edge segments.
struct FenceGrid {
    uint64_t horizontal{0};  // Bit r*8+c = horizontal fence placed at intersection (r,c)
    uint64_t vertical{0};    // Bit r*8+c = vertical fence placed at intersection (r,c)

    constexpr FenceGrid() = default;

    /// Check if a horizontal fence was placed at intersection (r, c)
    /// r in [0,7], c in [0,7]
    [[nodiscard]] constexpr bool has_h_fence(uint8_t r, uint8_t c) const noexcept {
        assert(r < 8 && c < 8);
        return (horizontal >> (r * 8 + c)) & 1;
    }

    /// Check if a vertical fence was placed at intersection (r, c)
    /// r in [0,7], c in [0,7]
    [[nodiscard]] constexpr bool has_v_fence(uint8_t r, uint8_t c) const noexcept {
        assert(r < 8 && c < 8);
        return (vertical >> (r * 8 + c)) & 1;
    }

    /// Place a horizontal fence at intersection (r, c)
    /// Blocks movement between rows r and r+1 at columns c and c+1
    constexpr void place_h_fence(uint8_t r, uint8_t c) noexcept {
        assert(r < 8 && c < 7);  // c < 7 because fence spans c and c+1
        horizontal |= (1ULL << (r * 8 + c));
    }

    /// Place a vertical fence at intersection (r, c)
    /// Blocks movement between columns c and c+1 at rows r and r+1
    constexpr void place_v_fence(uint8_t r, uint8_t c) noexcept {
        assert(r < 7 && c < 8);  // r < 7 because fence spans r and r+1
        vertical |= (1ULL << (r * 8 + c));
    }

    /// Check if horizontal fence placement at (r, c) is blocked
    [[nodiscard]] constexpr bool h_fence_blocked(uint8_t r, uint8_t c) const noexcept {
        assert(r < 8 && c < 7);
        // Blocked if there's already a horizontal fence at c or c-1 on the same row
        if (has_h_fence(r, c)) return true;
        if (c > 0 && has_h_fence(r, c - 1)) return true;
        if (c < 6 && has_h_fence(r, c + 1)) return true;
        // Blocked if a vertical fence crosses at the midpoint
        // Vertical fence at (r-1, c) or (r, c) would cross
        if (r > 0 && has_v_fence(r - 1, c)) return true;
        if (has_v_fence(r, c)) return true;
        return false;
    }

    /// Check if vertical fence placement at (r, c) is blocked
    [[nodiscard]] constexpr bool v_fence_blocked(uint8_t r, uint8_t c) const noexcept {
        assert(r < 7 && c < 8);
        // Blocked if there's already a vertical fence at r or r-1 on the same column
        if (has_v_fence(r, c)) return true;
        if (r > 0 && has_v_fence(r - 1, c)) return true;
        if (r < 6 && has_v_fence(r + 1, c)) return true;
        // Blocked if a horizontal fence crosses at the midpoint
        // Horizontal fence at (r, c-1) or (r, c) would cross
        if (c > 0 && has_h_fence(r, c - 1)) return true;
        if (has_h_fence(r, c)) return true;
        return false;
    }

    /// Check if there's a wall blocking movement from (row,col) going up (to row-1)
    /// A horizontal fence at intersection (row-1, c) blocks if c <= col < c+2
    [[nodiscard]] constexpr bool blocked_up(uint8_t row, uint8_t col) const noexcept {
        if (row == 0) return true;  // Board edge
        // Check for horizontal fence that spans this edge
        // Fence at (row-1, c) blocks columns c and c+1
        if (col > 0 && has_h_fence(row - 1, col - 1)) return true;
        if (col < 8 && has_h_fence(row - 1, col)) return true;
        return false;
    }

    /// Check if there's a wall blocking movement from (row,col) going down (to row+1)
    [[nodiscard]] constexpr bool blocked_down(uint8_t row, uint8_t col) const noexcept {
        if (row >= 8) return true;  // Board edge
        // Fence at (row, c) blocks columns c and c+1
        if (col > 0 && has_h_fence(row, col - 1)) return true;
        if (col < 8 && has_h_fence(row, col)) return true;
        return false;
    }

    /// Check if there's a wall blocking movement from (row,col) going left (to col-1)
    [[nodiscard]] constexpr bool blocked_left(uint8_t row, uint8_t col) const noexcept {
        if (col == 0) return true;  // Board edge
        // Fence at (r, col-1) blocks rows r and r+1
        if (row > 0 && has_v_fence(row - 1, col - 1)) return true;
        if (row < 8 && has_v_fence(row, col - 1)) return true;
        return false;
    }

    /// Check if there's a wall blocking movement from (row,col) going right (to col+1)
    [[nodiscard]] constexpr bool blocked_right(uint8_t row, uint8_t col) const noexcept {
        if (col >= 8) return true;  // Board edge
        // Fence at (r, col) blocks rows r and r+1
        if (row > 0 && has_v_fence(row - 1, col)) return true;
        if (row < 8 && has_v_fence(row, col)) return true;
        return false;
    }
};

static_assert(sizeof(FenceGrid) == 16, "FenceGrid should be 16 bytes");

/// Edge statistics stored at child nodes (represents the edge from parent to this node)
/// Using atomic operations for lock-free concurrent MCTS
///
/// Per AlphaGo Zero:
///   N(s,a) = visit count
///   W(s,a) = total action value
///   Q(s,a) = mean action value = W/N
///   P(s,a) = prior probability (from policy network or uniform)
struct EdgeStats {
    std::atomic<uint32_t> visits{0};       // N(s,a): visit count
    std::atomic<float> total_value{0.0f};  // W(s,a): sum of values from backpropagation
    std::atomic<int32_t> virtual_loss{0};  // Temporary penalty for tree parallelism
    float prior{0.0f};                     // P(s,a): policy prior (set once, read-only after)

    EdgeStats() = default;

    // Non-copyable due to atomics, but we can construct from values
    EdgeStats(const EdgeStats&) = delete;
    EdgeStats& operator=(const EdgeStats&) = delete;

    /// Move constructor for initialization
    EdgeStats(EdgeStats&& other) noexcept
        : visits(other.visits.load(std::memory_order_relaxed))
        , total_value(other.total_value.load(std::memory_order_relaxed))
        , virtual_loss(other.virtual_loss.load(std::memory_order_relaxed))
        , prior(other.prior) {}

    /// Get Q(s,a) = W(s,a) / N(s,a) with first-play urgency
    [[nodiscard]] float Q(float fpu = 0.0f) const noexcept {
        uint32_t n = visits.load(std::memory_order_relaxed);
        if (n == 0) return fpu;
        return total_value.load(std::memory_order_relaxed) / static_cast<float>(n);
    }

    /// Get N(s,a) including virtual loss
    [[nodiscard]] uint32_t N_with_virtual() const noexcept {
        return visits.load(std::memory_order_relaxed)
             + static_cast<uint32_t>(virtual_loss.load(std::memory_order_relaxed));
    }

    /// Apply virtual loss (called during selection)
    void add_virtual_loss(int32_t amount = 1) noexcept {
        virtual_loss.fetch_add(amount, std::memory_order_relaxed);
    }

    /// Remove virtual loss (called after backpropagation)
    void remove_virtual_loss(int32_t amount = 1) noexcept {
        virtual_loss.fetch_sub(amount, std::memory_order_relaxed);
    }

    /// Update statistics during backpropagation (atomic)
    void update(float value) noexcept {
        visits.fetch_add(1, std::memory_order_relaxed);
        // Atomic float addition via CAS loop
        float expected = total_value.load(std::memory_order_relaxed);
        while (!total_value.compare_exchange_weak(
            expected, expected + value,
            std::memory_order_relaxed, std::memory_order_relaxed)) {
            // expected is updated on failure
        }
    }
};

/// Tree node using left-child right-sibling representation
/// Children are stored as indices into the node pool, not as pointers
/// This allows for memory-bounded node allocation with recycling
///
/// Contains the full game state for fast move generation and evaluation
/// from any node without tree traversal.
struct alignas(64) StateNode {
    // === Tree structure (indices into node pool) ===
    uint32_t first_child{NULL_NODE};   // Left-child (first child)
    uint32_t next_sibling{NULL_NODE};  // Right-sibling (next child of same parent)
    uint32_t parent{NULL_NODE};        // Parent node

    // === Game state ===
    Player p1;           // Player 1 state (starts at row 0, goal row 8)
    Player p2;           // Player 2 state (starts at row 8, goal row 0)
    FenceGrid fences;    // All placed fences

    // The move that led to this node (edge from parent)
    Move move;

    // Edge statistics for this node (represents edge from parent)
    EdgeStats stats;

    // State flags
    uint8_t flags{0};
    static constexpr uint8_t FLAG_EXPANDED   = 0x01;  // Children have been generated
    static constexpr uint8_t FLAG_TERMINAL   = 0x02;  // Game over at this node
    static constexpr uint8_t FLAG_P1_TO_MOVE = 0x04;  // Player 1's turn (else P2)

    // Terminal value (only valid if FLAG_TERMINAL is set)
    // +1.0 = P1 wins, -1.0 = P2 wins, 0.0 = draw (if applicable)
    float terminal_value{0.0f};

    // Ply counter (number of moves made to reach this state)
    uint16_t ply{0};

    StateNode() = default;

    /// Initialize as root node (starting position)
    void init_root(bool p1_starts = true) noexcept {
        first_child = NULL_NODE;
        next_sibling = NULL_NODE;
        parent = NULL_NODE;
        move = Move{};  // Invalid/unset for root

        // Starting positions: P1 at (0, 4), P2 at (8, 4)
        p1 = Player(0, 4, NUM_FENCES);
        p2 = Player(8, 4, NUM_FENCES);
        fences = FenceGrid{};

        flags = p1_starts ? FLAG_P1_TO_MOVE : 0;
        terminal_value = 0.0f;
        ply = 0;

        // Reset stats
        stats.visits.store(1, std::memory_order_relaxed);  // Init to 1 to avoid div by 0
        stats.total_value.store(0.0f, std::memory_order_relaxed);
        stats.virtual_loss.store(0, std::memory_order_relaxed);
        stats.prior = 0.0f;
    }

    /// Initialize as child node from parent state with the given move applied
    void init_from_parent(const StateNode& parent_node, Move m, uint32_t parent_idx) noexcept {
        first_child = NULL_NODE;
        next_sibling = NULL_NODE;
        parent = parent_idx;
        move = m;

        // Copy parent state
        p1 = parent_node.p1;
        p2 = parent_node.p2;
        fences = parent_node.fences;
        ply = parent_node.ply + 1;

        // Apply move
        bool was_p1_turn = parent_node.is_p1_to_move();
        if (m.is_pawn()) {
            if (was_p1_turn) {
                p1.row = m.row();
                p1.col = m.col();
            } else {
                p2.row = m.row();
                p2.col = m.col();
            }
        } else {  // Fence move
            if (m.is_horizontal()) {
                fences.place_h_fence(m.row(), m.col());
            } else {
                fences.place_v_fence(m.row(), m.col());
            }
            if (was_p1_turn) {
                p1.fences--;
            } else {
                p2.fences--;
            }
        }

        // Switch turns
        flags = was_p1_turn ? 0 : FLAG_P1_TO_MOVE;
        terminal_value = 0.0f;

        // Check for terminal state
        if (p1.row == 8) {
            set_terminal(1.0f);   // P1 wins
        } else if (p2.row == 0) {
            set_terminal(-1.0f);  // P2 wins
        }

        // Reset stats
        stats.visits.store(0, std::memory_order_relaxed);
        stats.total_value.store(0.0f, std::memory_order_relaxed);
        stats.virtual_loss.store(0, std::memory_order_relaxed);
        stats.prior = 0.0f;
    }

    // Legacy init for backward compatibility with node pool
    void init(Move m, uint32_t parent_idx, bool p1_to_move) noexcept {
        first_child = NULL_NODE;
        next_sibling = NULL_NODE;
        parent = parent_idx;
        move = m;
        flags = p1_to_move ? FLAG_P1_TO_MOVE : 0;
        terminal_value = 0.0f;
        ply = 0;
        // Reset stats
        stats.visits.store(0, std::memory_order_relaxed);
        stats.total_value.store(0.0f, std::memory_order_relaxed);
        stats.virtual_loss.store(0, std::memory_order_relaxed);
        stats.prior = 0.0f;
    }

    [[nodiscard]] bool is_expanded() const noexcept { return flags & FLAG_EXPANDED; }
    [[nodiscard]] bool is_terminal() const noexcept { return flags & FLAG_TERMINAL; }
    [[nodiscard]] bool is_p1_to_move() const noexcept { return flags & FLAG_P1_TO_MOVE; }
    [[nodiscard]] bool has_children() const noexcept { return first_child != NULL_NODE; }

    void set_expanded() noexcept { flags |= FLAG_EXPANDED; }
    void set_terminal(float value) noexcept {
        flags |= FLAG_TERMINAL;
        terminal_value = value;
    }

    /// Get current player reference
    [[nodiscard]] const Player& current_player() const noexcept {
        return is_p1_to_move() ? p1 : p2;
    }

    /// Get opponent player reference
    [[nodiscard]] const Player& opponent_player() const noexcept {
        return is_p1_to_move() ? p2 : p1;
    }

    /// Check if a square is occupied by either player
    [[nodiscard]] bool is_occupied(uint8_t row, uint8_t col) const noexcept {
        return (p1.row == row && p1.col == col) || (p2.row == row && p2.col == col);
    }

    /// Check if the game is over
    /// Returns: 0 = not over, 1 = P1 wins, -1 = P2 wins
    [[nodiscard]] int game_over() const noexcept {
        if (p1.row == 8) return 1;
        if (p2.row == 0) return -1;
        return 0;
    }

    /// Print a visual representation of the game state to stdout
    void print_node() const noexcept;

    /// Generate all valid moves from this position
    /// Returns vector of valid moves. Pawn moves come first, then fence moves.
    /// @param out_fence_count Optional pointer to receive count of fence moves
    [[nodiscard]] std::vector<Move> generate_valid_moves(size_t* out_fence_count = nullptr) const noexcept;

    /// Generate all valid child nodes from this position
    /// Allocates children from the pool and links them using left-child right-sibling.
    /// Validates fence moves with pathfinding to ensure no player gets blocked.
    /// @param pool Node pool to allocate children from
    /// @param my_index This node's index in the pool
    /// @return Number of children generated, or 0 if terminal/already expanded/allocation failed
    size_t generate_valid_children(class NodePool& pool, uint32_t my_index) noexcept;
};

// Note: With full game state, size is larger than 64 bytes
// This is acceptable for training speed - we trade memory for avoiding tree traversal
static_assert(sizeof(StateNode) >= 64, "StateNode includes full game state");

/// PUCT selection formula from AlphaGo Zero
/// U(s,a) = c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
/// Select argmax_a [Q(s,a) + U(s,a)]
[[nodiscard]] inline float puct_score(
    const EdgeStats& edge,
    uint32_t parent_visits,
    float c_puct = 1.5f,
    float fpu = 0.0f) noexcept
{
    uint32_t n = edge.N_with_virtual();
    float q = edge.Q(fpu);
    float u = c_puct * edge.prior * std::sqrt(static_cast<float>(parent_visits))
            / (1.0f + static_cast<float>(n));

    // Account for virtual loss in Q calculation
    int32_t vl = edge.virtual_loss.load(std::memory_order_relaxed);
    if (vl > 0 && n > 0) {
        // Virtual loss reduces Q value
        q = (edge.total_value.load(std::memory_order_relaxed) - static_cast<float>(vl))
          / static_cast<float>(n);
    }

    return q + u;
}

} // namespace qbot
