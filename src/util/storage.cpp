#include "storage.h"

#include <chrono>
#include <cstring>
#include <queue>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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

    // Pass 1: BFS to collect reachable nodes and build dense remap vector
    // Using vector instead of unordered_map for O(1) lookup without hashing
    std::vector<uint32_t> reachable;
    std::vector<uint32_t> remap(pool.capacity(), NULL_NODE);

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

    const size_t node_count = reachable.size();
    const size_t file_size = sizeof(TreeFileHeader) + node_count * sizeof(SerializedNode);

    // Open file with POSIX for mmap
    int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        if (!std::filesystem::exists(path.parent_path())) {
            return std::unexpected(StorageError::FileNotFound);
        }
        return std::unexpected(StorageError::PermissionDenied);
    }

    // Pre-allocate file to exact size (avoids fragmentation, required for mmap)
    if (ftruncate(fd, static_cast<off_t>(file_size)) != 0) {
        ::close(fd);
        return std::unexpected(StorageError::IoError);
    }

    // mmap the file for direct memory writes
    void* mapped = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        ::close(fd);
        return std::unexpected(StorageError::IoError);
    }

    // Hint kernel about sequential write pattern
    madvise(mapped, file_size, MADV_SEQUENTIAL);

    // Write header directly to mapped memory
    auto* header = static_cast<TreeFileHeader*>(mapped);
    header->magic = TreeFileHeader::MAGIC;
    header->version = TreeFileHeader::CURRENT_VERSION;
    header->flags = 0;
    header->node_count = static_cast<uint32_t>(node_count);
    header->root_index = 0;  // Root is always first after compaction
    header->reserved1 = 0;
    header->timestamp = static_cast<uint64_t>(
        std::chrono::system_clock::now().time_since_epoch().count());
    std::memset(header->reserved, 0, sizeof(header->reserved));

    // Get pointer to node array in mapped region
    auto* nodes = reinterpret_cast<SerializedNode*>(
        static_cast<char*>(mapped) + sizeof(TreeFileHeader));

    // Remap helper - direct vector lookup, no hashing
    auto remap_index = [&remap](uint32_t idx) -> uint32_t {
        if (idx == NULL_NODE) return NULL_NODE;
        return remap[idx];
    };

    // Pass 2: Serialize directly into mapped memory
    for (size_t i = 0; i < node_count; ++i) {
        uint32_t old_idx = reachable[i];
        SerializedNode& sn = nodes[i];

        // Serialize node
        const StateNode& node = pool[old_idx];
        sn.first_child = remap_index(node.first_child);
        sn.next_sibling = remap_index(node.next_sibling);
        sn.parent = remap_index(node.parent);
        sn.p1_row = node.p1.row;
        sn.p1_col = node.p1.col;
        sn.p1_fences = node.p1.fences;
        sn.p2_row = node.p2.row;
        sn.p2_col = node.p2.col;
        sn.p2_fences = node.p2.fences;
        sn.move_data = node.move.data;
        sn.flags = node.flags;
        sn.reserved = 0;
        sn.ply = node.ply;
        sn.fences_horizontal = node.fences.horizontal;
        sn.fences_vertical = node.fences.vertical;
        sn.visits = node.stats.visits.load(std::memory_order_relaxed);
        sn.total_value = node.stats.total_value.load(std::memory_order_relaxed);
        sn.prior = node.stats.prior;
        sn.terminal_value = node.terminal_value;
    }

    // Sync and unmap
    if (msync(mapped, file_size, MS_SYNC) != 0) {
        munmap(mapped, file_size);
        ::close(fd);
        return std::unexpected(StorageError::IoError);
    }

    munmap(mapped, file_size);
    ::close(fd);

    return {};
}

std::expected<size_t, StorageError> TreeStorage::save_pruned(
    const std::filesystem::path& path,
    const NodePool& pool,
    uint32_t root)
{
    if (root == NULL_NODE) {
        return std::unexpected(StorageError::EmptyTree);
    }

    // Pass 1: BFS collecting only nodes on game path
    std::vector<uint32_t> reachable;
    std::vector<uint32_t> remap(pool.capacity(), NULL_NODE);

    {
        std::queue<uint32_t> queue;
        queue.push(root);

        while (!queue.empty()) {
            uint32_t idx = queue.front();
            queue.pop();

            // Skip if already visited
            if (remap[idx] != NULL_NODE) continue;

            const StateNode& node = pool[idx];

            // Only include nodes that are on game path
            if (!node.is_on_game_path()) continue;

            uint32_t new_idx = static_cast<uint32_t>(reachable.size());
            reachable.push_back(idx);
            remap[idx] = new_idx;

            // Only follow children that are also on game path
            uint32_t child = node.first_child;
            while (child != NULL_NODE) {
                if (pool[child].is_on_game_path()) {
                    queue.push(child);
                }
                child = pool[child].next_sibling;
            }
        }
    }

    if (reachable.empty()) {
        return std::unexpected(StorageError::EmptyTree);
    }

    const size_t node_count = reachable.size();
    const size_t file_size = sizeof(TreeFileHeader) + node_count * sizeof(SerializedNode);

    int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        if (!std::filesystem::exists(path.parent_path())) {
            return std::unexpected(StorageError::FileNotFound);
        }
        return std::unexpected(StorageError::PermissionDenied);
    }

    if (ftruncate(fd, static_cast<off_t>(file_size)) != 0) {
        ::close(fd);
        return std::unexpected(StorageError::IoError);
    }

    void* mapped = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        ::close(fd);
        return std::unexpected(StorageError::IoError);
    }

    madvise(mapped, file_size, MADV_SEQUENTIAL);

    auto* header = static_cast<TreeFileHeader*>(mapped);
    header->magic = TreeFileHeader::MAGIC;
    header->version = TreeFileHeader::CURRENT_VERSION;
    header->flags = 0;
    header->node_count = static_cast<uint32_t>(node_count);
    header->root_index = 0;
    header->reserved1 = 0;
    header->timestamp = static_cast<uint64_t>(
        std::chrono::system_clock::now().time_since_epoch().count());
    std::memset(header->reserved, 0, sizeof(header->reserved));

    auto* nodes = reinterpret_cast<SerializedNode*>(
        static_cast<char*>(mapped) + sizeof(TreeFileHeader));

    // Remap helper - only remap if target is on game path
    auto remap_index = [&remap, &pool](uint32_t idx) -> uint32_t {
        if (idx == NULL_NODE) return NULL_NODE;
        // Only return remapped index if node is on game path
        if (!pool[idx].is_on_game_path()) return NULL_NODE;
        return remap[idx];
    };

    // Remap sibling - find next sibling that is on game path
    auto remap_sibling = [&pool, &remap](uint32_t idx) -> uint32_t {
        while (idx != NULL_NODE) {
            if (pool[idx].is_on_game_path() && remap[idx] != NULL_NODE) {
                return remap[idx];
            }
            idx = pool[idx].next_sibling;
        }
        return NULL_NODE;
    };

    // Remap first_child - find first child that is on game path
    auto remap_first_child = [&pool, &remap](const StateNode& node) -> uint32_t {
        uint32_t child = node.first_child;
        while (child != NULL_NODE) {
            if (pool[child].is_on_game_path() && remap[child] != NULL_NODE) {
                return remap[child];
            }
            child = pool[child].next_sibling;
        }
        return NULL_NODE;
    };

    // Pass 2: Serialize
    for (size_t i = 0; i < node_count; ++i) {
        uint32_t old_idx = reachable[i];
        SerializedNode& sn = nodes[i];
        const StateNode& node = pool[old_idx];

        sn.first_child = remap_first_child(node);
        sn.next_sibling = remap_sibling(node.next_sibling);
        sn.parent = remap_index(node.parent);
        sn.p1_row = node.p1.row;
        sn.p1_col = node.p1.col;
        sn.p1_fences = node.p1.fences;
        sn.p2_row = node.p2.row;
        sn.p2_col = node.p2.col;
        sn.p2_fences = node.p2.fences;
        sn.move_data = node.move.data;
        sn.flags = node.flags;
        sn.reserved = 0;
        sn.ply = node.ply;
        sn.fences_horizontal = node.fences.horizontal;
        sn.fences_vertical = node.fences.vertical;
        sn.visits = node.stats.visits.load(std::memory_order_relaxed);
        sn.total_value = node.stats.total_value.load(std::memory_order_relaxed);
        sn.prior = node.stats.prior;
        sn.terminal_value = node.terminal_value;
    }

    if (msync(mapped, file_size, MS_SYNC) != 0) {
        munmap(mapped, file_size);
        ::close(fd);
        return std::unexpected(StorageError::IoError);
    }

    munmap(mapped, file_size);
    ::close(fd);

    return node_count;
}

std::expected<LoadedTree, StorageError> TreeStorage::load(
    const std::filesystem::path& path)
{
    // Open file with POSIX for mmap
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        if (!std::filesystem::exists(path)) {
            return std::unexpected(StorageError::FileNotFound);
        }
        return std::unexpected(StorageError::PermissionDenied);
    }

    // Get file size
    struct stat st;
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        return std::unexpected(StorageError::IoError);
    }
    const size_t file_size = static_cast<size_t>(st.st_size);

    if (file_size < sizeof(TreeFileHeader)) {
        ::close(fd);
        return std::unexpected(StorageError::InvalidFormat);
    }

    // mmap the entire file read-only
    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        ::close(fd);
        return std::unexpected(StorageError::IoError);
    }

    // Hint kernel about sequential read pattern
    madvise(mapped, file_size, MADV_SEQUENTIAL);

    // Read header from mapped memory
    const auto* header = static_cast<const TreeFileHeader*>(mapped);

    // Validate header
    if (header->magic != TreeFileHeader::MAGIC) {
        munmap(mapped, file_size);
        ::close(fd);
        return std::unexpected(StorageError::InvalidFormat);
    }
    if (header->version > TreeFileHeader::CURRENT_VERSION) {
        munmap(mapped, file_size);
        ::close(fd);
        return std::unexpected(StorageError::VersionMismatch);
    }
    if (header->node_count == 0) {
        munmap(mapped, file_size);
        ::close(fd);
        return std::unexpected(StorageError::EmptyTree);
    }

    // Validate file size matches expected
    const size_t expected_size = sizeof(TreeFileHeader) +
                                  header->node_count * sizeof(SerializedNode);
    if (file_size < expected_size) {
        munmap(mapped, file_size);
        ::close(fd);
        return std::unexpected(StorageError::CorruptedData);
    }

    // Create node pool with some headroom for continued training
    NodePool::Config config;
    config.initial_capacity = header->node_count + 10'000'000;  // 10M headroom
    config.chunk_size = 10'000'000;  // 10M growth chunks
    // config.enable_lru = false;

    LoadedTree result{
        .pool = std::make_unique<NodePool>(config),
        .root = header->root_index,
        .timestamp = header->timestamp,
    };

    // Get pointer to node array in mapped region
    const auto* nodes = reinterpret_cast<const SerializedNode*>(
        static_cast<const char*>(mapped) + sizeof(TreeFileHeader));

    // Bulk allocate and deserialize
    const uint32_t node_count = header->node_count;
    for (uint32_t i = 0; i < node_count; ++i) {
        uint32_t idx = result.pool->allocate();
        if (idx == NULL_NODE) {
            munmap(mapped, file_size);
            ::close(fd);
            return std::unexpected(StorageError::InsufficientMemory);
        }
        // Indices should be sequential since pool is fresh
        if (idx != i) {
            munmap(mapped, file_size);
            ::close(fd);
            return std::unexpected(StorageError::CorruptedData);
        }
        deserialize_node(nodes[i], (*result.pool)[idx]);
    }

    munmap(mapped, file_size);
    ::close(fd);

    return result;
}

std::expected<TreeFileHeader, StorageError> TreeStorage::read_header(
    const std::filesystem::path& path)
{
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        if (!std::filesystem::exists(path)) {
            return std::unexpected(StorageError::FileNotFound);
        }
        return std::unexpected(StorageError::PermissionDenied);
    }

    TreeFileHeader header;
    ssize_t bytes_read = ::read(fd, &header, sizeof(header));
    ::close(fd);

    if (bytes_read != sizeof(header)) {
        return std::unexpected(StorageError::IoError);
    }

    if (!header.is_valid()) {
        return std::unexpected(StorageError::InvalidFormat);
    }

    return header;
}

} // namespace qbot
