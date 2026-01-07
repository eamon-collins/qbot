#pragma once

#include "../tree/node_pool.h"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace qbot {

/// Error codes for storage operations
enum class StorageError {
    FileNotFound,
    PermissionDenied,
    InvalidFormat,
    VersionMismatch,
    CorruptedData,
    ChecksumMismatch,  // Deprecated, no longer used
    InsufficientMemory,
    IoError,
    EmptyTree,
};

[[nodiscard]] constexpr std::string_view to_string(StorageError err) noexcept {
    switch (err) {
        case StorageError::FileNotFound:       return "File not found";
        case StorageError::PermissionDenied:   return "Permission denied";
        case StorageError::InvalidFormat:      return "Invalid file format";
        case StorageError::VersionMismatch:    return "Version mismatch";
        case StorageError::CorruptedData:      return "Corrupted data";
        case StorageError::ChecksumMismatch:   return "Checksum mismatch";
        case StorageError::InsufficientMemory: return "Insufficient memory";
        case StorageError::IoError:            return "I/O error";
        case StorageError::EmptyTree:          return "Empty tree";
    }
    return "Unknown error";
}

/// File header for tree storage format
/// All multi-byte values are little-endian
struct TreeFileHeader {
    static constexpr uint32_t MAGIC = 0x51424F54;  // "QBOT" in little-endian
    static constexpr uint16_t CURRENT_VERSION = 1;

    uint32_t magic = MAGIC;
    uint16_t version = CURRENT_VERSION;
    uint16_t flags = 0;
    uint32_t node_count = 0;
    uint32_t root_index = NULL_NODE;
    uint32_t reserved1 = 0;  // Reserved (was checksum, removed for performance)
    uint64_t timestamp = 0; // Unix timestamp when saved

    // Reserved for future use
    uint8_t reserved[32] = {};

    [[nodiscard]] bool is_valid() const noexcept {
        return magic == MAGIC && version <= CURRENT_VERSION;
    }
};

static_assert(sizeof(TreeFileHeader) == 64, "Header should be 64 bytes");

/// Compact serialized node format (without atomics or padding)
/// This is the on-disk representation of StateNode
struct SerializedNode {
    // Tree structure
    uint32_t first_child;
    uint32_t next_sibling;
    uint32_t parent;

    // Game state
    uint8_t p1_row;
    uint8_t p1_col;
    uint8_t p1_fences;
    uint8_t p2_row;
    uint8_t p2_col;
    uint8_t p2_fences;

    // Move and flags
    uint16_t move_data;
    uint8_t flags;
    uint8_t reserved;
    uint16_t ply;

    // Fence grid (128 bits = 16 bytes)
    uint64_t fences_horizontal;
    uint64_t fences_vertical;

    // Statistics
    uint32_t visits;
    float total_value;
    float prior;
    float terminal_value;
};

static_assert(sizeof(SerializedNode) == 56, "SerializedNode should be 56 bytes");

/// Result of loading a tree
struct LoadedTree {
    std::unique_ptr<NodePool> pool;
    uint32_t root;
    uint64_t timestamp;
};

/// Tree storage operations
class TreeStorage {
public:
    /// Save a tree to a file
    /// @param path Output file path
    /// @param pool Node pool containing the tree
    /// @param root Root node index
    /// @return Success or error
    [[nodiscard]] static std::expected<void, StorageError> save(
        const std::filesystem::path& path,
        const NodePool& pool,
        uint32_t root);

    /// Load a tree from a file
    /// @param path Input file path
    /// @return Loaded tree or error
    [[nodiscard]] static std::expected<LoadedTree, StorageError> load(
        const std::filesystem::path& path);

    /// Read just the header from a file (for inspection)
    [[nodiscard]] static std::expected<TreeFileHeader, StorageError> read_header(
        const std::filesystem::path& path);

    /// Estimate file size for a tree with given node count
    [[nodiscard]] static constexpr size_t estimate_file_size(size_t node_count) noexcept {
        return sizeof(TreeFileHeader) + node_count * sizeof(SerializedNode);
    }

private:
    /// Convert StateNode to SerializedNode
    static SerializedNode serialize_node(const StateNode& node) noexcept;

    /// Convert SerializedNode to StateNode (in-place initialization)
    static void deserialize_node(const SerializedNode& src, StateNode& dst) noexcept;
};

} // namespace qbot
