#include "storage.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace qbot {

SerializedNode TreeStorage::serialize_node(const StateNode& node) noexcept {
    SerializedNode sn;
    // Tree structure
    sn.first_child = node.first_child;
    sn.next_sibling = node.next_sibling;
    sn.parent = node.parent;

    // Game state
    sn.p1_row = node.p1.row;
    sn.p1_col = node.p1.col;
    sn.p1_fences = node.p1.fences;
    sn.p2_row = node.p2.row;
    sn.p2_col = node.p2.col;
    sn.p2_fences = node.p2.fences;

    // Move and flags
    sn.move_data = node.move.data;
    sn.flags = node.flags;
    sn.reserved = 0;
    sn.ply = node.ply;

    // Fence grid
    sn.fences_horizontal = node.fences.horizontal;
    sn.fences_vertical = node.fences.vertical;

    // Statistics
    sn.visits = node.stats.visits.load(std::memory_order_relaxed);
    sn.total_value = node.stats.total_value.load(std::memory_order_relaxed);
    sn.prior = node.stats.prior;
    sn.terminal_value = node.terminal_value;
    return sn;
}

void TreeStorage::deserialize_node(const SerializedNode& src, StateNode& dst) noexcept {
    // Tree structure
    dst.first_child = src.first_child;
    dst.next_sibling = src.next_sibling;
    dst.parent = src.parent;

    // Game state
    dst.p1.row = src.p1_row;
    dst.p1.col = src.p1_col;
    dst.p1.fences = src.p1_fences;
    dst.p2.row = src.p2_row;
    dst.p2.col = src.p2_col;
    dst.p2.fences = src.p2_fences;

    // Move and flags
    dst.move.data = src.move_data;
    dst.flags = src.flags;
    dst.ply = src.ply;

    // Fence grid
    dst.fences.horizontal = src.fences_horizontal;
    dst.fences.vertical = src.fences_vertical;

    // Statistics
    dst.terminal_value = src.terminal_value;
    dst.stats.visits.store(src.visits, std::memory_order_relaxed);
    dst.stats.total_value.store(src.total_value, std::memory_order_relaxed);
    dst.stats.virtual_loss.store(0, std::memory_order_relaxed);
    dst.stats.prior = src.prior;
}

std::expected<void, StorageError> TreeStorage::save(
    const std::filesystem::path& path,
    const NodePool& pool,
    uint32_t root)
{
    if (root == NULL_NODE) {
        return std::unexpected(StorageError::EmptyTree);
    }

    std::printf("TreeStorage::save - pool.allocated()=%zu, root=%u\n",
                pool.allocated(), root);

    // Pass 1: BFS to collect reachable nodes and build sparse remap
    std::vector<uint32_t> reachable;
    std::unordered_map<uint32_t, uint32_t> remap;  // old_idx -> new_idx

    {
        std::queue<uint32_t> queue;
        queue.push(root);

        while (!queue.empty()) {
            uint32_t idx = queue.front();
            queue.pop();

            uint32_t new_idx = static_cast<uint32_t>(reachable.size());
            reachable.push_back(idx);
            remap[idx] = new_idx;

            const StateNode& node = pool[idx];
            uint32_t child = node.first_child;
            while (child != NULL_NODE) {
                queue.push(child);
                child = pool[child].next_sibling;
            }
        }
    }

    if (reachable.empty()) {
        return std::unexpected(StorageError::EmptyTree);
    }

    std::printf("TreeStorage::save - BFS found %zu reachable nodes\n", reachable.size());

    // Debug: check for orphaned nodes
    if (reachable.size() < pool.allocated()) {
        std::unordered_set<uint32_t> reachable_set(reachable.begin(), reachable.end());
        size_t printed = 0;
        size_t parent_null = 0;
        size_t parent_not_reachable = 0;
        size_t parent_missing_child = 0;

        for (size_t i = 0; i < pool.allocated(); ++i) {
            if (reachable_set.find(static_cast<uint32_t>(i)) == reachable_set.end()) {
                const StateNode& orphan = pool[static_cast<uint32_t>(i)];
                uint32_t parent_idx = orphan.parent;

                if (parent_idx == NULL_NODE) {
                    parent_null++;
                    if (printed < 3) {
                        std::printf("  Orphan %zu: parent=NULL\n", i);
                        printed++;
                    }
                } else if (reachable_set.find(parent_idx) == reachable_set.end()) {
                    parent_not_reachable++;
                    if (printed < 3) {
                        std::printf("  Orphan %zu: parent=%u (also orphaned)\n", i, parent_idx);
                        printed++;
                    }
                } else {
                    parent_missing_child++;
                    if (printed < 3) {
                        const StateNode& p = pool[parent_idx];
                        std::printf("  Orphan %zu: parent=%u is reachable, parent.first_child=%u\n",
                                    i, parent_idx, p.first_child);
                        printed++;
                    }
                }
            }
        }
        std::printf("  Orphan breakdown: parent_null=%zu, parent_orphaned=%zu, parent_missing_link=%zu\n",
                    parent_null, parent_not_reachable, parent_missing_child);
    }

    // Open file and write header
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        if (!std::filesystem::exists(path.parent_path())) {
            return std::unexpected(StorageError::FileNotFound);
        }
        return std::unexpected(StorageError::PermissionDenied);
    }

    TreeFileHeader header;
    header.node_count = static_cast<uint32_t>(reachable.size());
    header.root_index = 0;  // Root is always first after compaction
    header.reserved1 = 0;   // No checksum
    header.timestamp = static_cast<uint64_t>(
        std::chrono::system_clock::now().time_since_epoch().count());

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!file) {
        return std::unexpected(StorageError::IoError);
    }

    // Pass 2: Stream nodes directly to file
    auto remap_index = [&remap](uint32_t idx) -> uint32_t {
        if (idx == NULL_NODE) return NULL_NODE;
        auto it = remap.find(idx);
        return (it != remap.end()) ? it->second : NULL_NODE;
    };

    for (uint32_t old_idx : reachable) {
        SerializedNode sn = serialize_node(pool[old_idx]);

        // Remap indices
        sn.first_child = remap_index(sn.first_child);
        sn.next_sibling = remap_index(sn.next_sibling);
        sn.parent = remap_index(sn.parent);

        file.write(reinterpret_cast<const char*>(&sn), sizeof(sn));
        if (!file) {
            return std::unexpected(StorageError::IoError);
        }
    }

    file.close();
    if (!file) {
        return std::unexpected(StorageError::IoError);
    }

    return {};
}

std::expected<LoadedTree, StorageError> TreeStorage::load(
    const std::filesystem::path& path)
{
    // Open file
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (!std::filesystem::exists(path)) {
            return std::unexpected(StorageError::FileNotFound);
        }
        return std::unexpected(StorageError::PermissionDenied);
    }

    // Read header
    TreeFileHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file) {
        return std::unexpected(StorageError::IoError);
    }

    // Validate header
    if (header.magic != TreeFileHeader::MAGIC) {
        return std::unexpected(StorageError::InvalidFormat);
    }
    if (header.version > TreeFileHeader::CURRENT_VERSION) {
        return std::unexpected(StorageError::VersionMismatch);
    }
    if (header.node_count == 0) {
        return std::unexpected(StorageError::EmptyTree);
    }

    // Create node pool with exact capacity needed
    NodePool::Config config;
    config.initial_capacity = header.node_count;
    config.enable_lru = false;  // Disable LRU for loaded trees initially

    LoadedTree result{
        .pool = std::make_unique<NodePool>(config),
        .root = header.root_index,
        .timestamp = header.timestamp,
    };

    // Stream read: read one node at a time directly into pool
    SerializedNode sn;
    for (uint32_t i = 0; i < header.node_count; ++i) {
        file.read(reinterpret_cast<char*>(&sn), sizeof(sn));
        if (!file) {
            return std::unexpected(StorageError::IoError);
        }

        uint32_t idx = result.pool->allocate();
        if (idx == NULL_NODE) {
            return std::unexpected(StorageError::InsufficientMemory);
        }
        // Indices should be sequential since pool is fresh
        if (idx != i) {
            return std::unexpected(StorageError::CorruptedData);
        }
        deserialize_node(sn, (*result.pool)[idx]);
    }

    return result;
}

std::expected<TreeFileHeader, StorageError> TreeStorage::read_header(
    const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (!std::filesystem::exists(path)) {
            return std::unexpected(StorageError::FileNotFound);
        }
        return std::unexpected(StorageError::PermissionDenied);
    }

    TreeFileHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file) {
        return std::unexpected(StorageError::IoError);
    }

    if (!header.is_valid()) {
        return std::unexpected(StorageError::InvalidFormat);
    }

    return header;
}

} // namespace qbot
