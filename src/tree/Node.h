#pragma once

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cassert>

namespace qbot {

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
/// Memory layout optimized for cache efficiency (~64 bytes target)
struct alignas(64) TreeNode {
    // Tree structure (indices into node pool)
    uint32_t first_child{NULL_NODE};   // Left-child (first child)
    uint32_t next_sibling{NULL_NODE};  // Right-sibling (next child of same parent)
    uint32_t parent{NULL_NODE};        // Parent node

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

    TreeNode() = default;

    // Initialize for a specific move
    void init(Move m, uint32_t parent_idx, bool p1_to_move) noexcept {
        first_child = NULL_NODE;
        next_sibling = NULL_NODE;
        parent = parent_idx;
        move = m;
        flags = p1_to_move ? FLAG_P1_TO_MOVE : 0;
        terminal_value = 0.0f;
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
};

// Verify size constraints (64-byte alignment, actual data ~36 bytes)
static_assert(sizeof(TreeNode) == 64, "TreeNode should be exactly 64 bytes (one cache line)");

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
