#pragma once

/// Training Sample Storage for AlphaZero-style training
///
/// Stores training samples with:
/// - Board state (compact representation)
/// - MCTS visit distribution (policy target)
/// - Game outcome (value target)
///
/// File format (.qsamples):
///   Header (64 bytes)
///   TrainingSample[] (variable size based on sample_count)

#include "../tree/StateNode.h"
#include "../tree/node_pool.h"
#include "../inference/inference.h"

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <string_view>
#include <vector>

namespace qbot {

/// Error codes for training sample operations
enum class TrainingSampleError {
    FileNotFound,
    PermissionDenied,
    InvalidFormat,
    VersionMismatch,
    CorruptedData,
    IoError,
    NoSamples,
};

[[nodiscard]] constexpr std::string_view to_string(TrainingSampleError err) noexcept {
    switch (err) {
        case TrainingSampleError::FileNotFound:     return "File not found";
        case TrainingSampleError::PermissionDenied: return "Permission denied";
        case TrainingSampleError::InvalidFormat:    return "Invalid file format";
        case TrainingSampleError::VersionMismatch:  return "Version mismatch";
        case TrainingSampleError::CorruptedData:    return "Corrupted data";
        case TrainingSampleError::IoError:          return "I/O error";
        case TrainingSampleError::NoSamples:        return "No samples";
    }
    return "Unknown error";
}

/// File header for training samples format
struct TrainingSampleHeader {
    static constexpr uint32_t MAGIC = 0x51534D50;  // "QSMP" in little-endian
    static constexpr uint16_t CURRENT_VERSION = 1;

    uint32_t magic = MAGIC;
    uint16_t version = CURRENT_VERSION;
    uint16_t flags = 0;
    uint32_t sample_count = 0;
    uint32_t reserved1 = 0;
    uint64_t timestamp = 0;

    // Reserved for future use
    uint8_t reserved[40] = {};

    [[nodiscard]] bool is_valid() const noexcept {
        return magic == MAGIC && version <= CURRENT_VERSION;
    }
};

static_assert(sizeof(TrainingSampleHeader) == 64, "Header should be 64 bytes");

/// Compact state representation for training samples
/// Contains just enough info to reconstruct input tensor
struct CompactState {
    // Player positions (4 bytes)
    uint8_t p1_row;
    uint8_t p1_col;
    uint8_t p2_row;
    uint8_t p2_col;

    // Fence counts (2 bytes)
    uint8_t p1_fences;
    uint8_t p2_fences;

    // Flags (1 byte) - includes whose turn it is
    uint8_t flags;

    // Reserved for alignment (1 byte)
    uint8_t reserved;

    // Fence grids (16 bytes)
    uint64_t fences_horizontal;
    uint64_t fences_vertical;
};

static_assert(sizeof(CompactState) == 24, "CompactState should be 24 bytes");

/// Single training sample with state, policy target, and value target
/// Policy is stored as visit counts (will be normalized during training)
struct TrainingSample {
    // Board state (24 bytes)
    CompactState state;

    // MCTS visit distribution - 209 actions (836 bytes as floats)
    // Stored as proportions (0.0-1.0), already normalized
    std::array<float, NUM_ACTIONS> policy;

    // Game outcome from current player's perspective (4 bytes)
    // +1.0 = current player won, -1.0 = current player lost, 0.0 = draw
    float value;
};

static_assert(sizeof(TrainingSample) == 864, "TrainingSample should be 864 bytes");

/// Extract compact state from a StateNode
[[nodiscard]] inline CompactState extract_compact_state(const StateNode& node) noexcept {
    CompactState state;
    state.p1_row = node.p1.row;
    state.p1_col = node.p1.col;
    state.p2_row = node.p2.row;
    state.p2_col = node.p2.col;
    state.p1_fences = node.p1.fences;
    state.p2_fences = node.p2.fences;
    state.flags = static_cast<uint8_t>(node.flags);
    state.reserved = 0;
    state.fences_horizontal = node.fences.horizontal;
    state.fences_vertical = node.fences.vertical;
    return state;
}

/// Extract visit distribution from a node's children
/// Returns normalized probabilities (sum to 1.0)
/// @param pool Node pool
/// @param node_idx Index of parent node
/// @return Policy distribution over 209 actions
[[nodiscard]] std::array<float, NUM_ACTIONS> extract_visit_distribution(
    const NodePool& pool, uint32_t node_idx) noexcept;

/// Training sample storage operations
class TrainingSampleStorage {
public:
    /// Save training samples to a file
    /// @param path Output file path
    /// @param samples Vector of training samples
    /// @return Success or error
    [[nodiscard]] static std::expected<void, TrainingSampleError> save(
        const std::filesystem::path& path,
        const std::vector<TrainingSample>& samples);

    /// Load training samples from a file
    /// @param path Input file path
    /// @return Vector of samples or error
    [[nodiscard]] static std::expected<std::vector<TrainingSample>, TrainingSampleError> load(
        const std::filesystem::path& path);

    /// Read just the header from a file (for inspection)
    [[nodiscard]] static std::expected<TrainingSampleHeader, TrainingSampleError> read_header(
        const std::filesystem::path& path);

    /// Estimate file size for given sample count
    [[nodiscard]] static constexpr size_t estimate_file_size(size_t sample_count) noexcept {
        return sizeof(TrainingSampleHeader) + sample_count * sizeof(TrainingSample);
    }
};

/// Extract training samples from a pruned game tree
/// Only extracts from nodes marked as on_game_path (part of a completed game with
/// win/loss backpropagated). This ensures training data only includes positions
/// with known outcomes, not speculative tree exploration.
/// @param pool Node pool containing the tree
/// @param root_idx Root node index
/// @return Vector of extracted training samples
[[nodiscard]] std::vector<TrainingSample> extract_samples_from_tree(
    const NodePool& pool, uint32_t root_idx);

/// Collector for gathering training samples during self-play
/// Thread-safe for use with multi-threaded self-play
class TrainingSampleCollector {
public:
    /// Reserve space for expected number of samples
    void reserve(size_t count);

    /// Add a sample from a position after MCTS search
    /// @param pool Node pool
    /// @param node_idx Index of the position node
    /// @param game_outcome Final game outcome (+1 P1 win, -1 P2 win, 0 draw)
    void add_sample(const NodePool& pool, uint32_t node_idx, float game_outcome);

    /// Add a pre-constructed sample directly
    void add_sample_direct(TrainingSample sample);

    /// Get collected samples
    [[nodiscard]] const std::vector<TrainingSample>& samples() const noexcept { return samples_; }

    /// Move out samples (clears collector)
    [[nodiscard]] std::vector<TrainingSample> take_samples() noexcept;

    /// Get current sample count
    [[nodiscard]] size_t size() const noexcept { return samples_.size(); }

    /// Clear all samples
    void clear() noexcept { samples_.clear(); }

private:
    std::vector<TrainingSample> samples_;
    mutable std::mutex mutex_;
};

} // namespace qbot
