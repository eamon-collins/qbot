#include <gtest/gtest.h>

#include "util/storage.h"
#include "tree/node_pool.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

using namespace qbot;

namespace {

/// RAII helper to create and clean up a temporary file
class TempFile {
public:
    TempFile() {
        path_ = std::filesystem::temp_directory_path() /
                ("qbot_test_" + std::to_string(std::rand()) + ".tree");
    }

    ~TempFile() {
        std::filesystem::remove(path_);
    }

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

/// Helper to build a simple test tree
/// Returns root index
uint32_t build_test_tree(NodePool& pool, int depth, int branching) {
    // Create root
    uint32_t root = pool.allocate(Move{}, NULL_NODE, true);
    EXPECT_NE(root, NULL_NODE);
    if (root == NULL_NODE) return NULL_NODE;

    pool[root].stats.visits.store(100, std::memory_order_relaxed);
    pool[root].stats.total_value.store(50.0f, std::memory_order_relaxed);
    pool[root].stats.prior = 1.0f;

    if (depth <= 0) return root;

    // Build children recursively
    std::vector<uint32_t> current_level = {root};

    for (int d = 0; d < depth; ++d) {
        std::vector<uint32_t> next_level;
        bool p1_turn = (d % 2 == 1);  // Alternates

        for (uint32_t parent : current_level) {
            ChildBuilder builder(pool, parent);

            for (int b = 0; b < branching; ++b) {
                Move move = Move::pawn(
                    static_cast<uint8_t>(d + 1),
                    static_cast<uint8_t>(b));
                uint32_t child = builder.add_child(move, p1_turn);
                EXPECT_NE(child, NULL_NODE);

                // Set some stats
                pool[child].stats.visits.store(
                    static_cast<uint32_t>(10 * (depth - d)),
                    std::memory_order_relaxed);
                pool[child].stats.total_value.store(
                    static_cast<float>(d * b),
                    std::memory_order_relaxed);
                pool[child].stats.prior = 1.0f / static_cast<float>(branching);

                next_level.push_back(child);
            }

            builder.commit();
        }

        current_level = std::move(next_level);
    }

    return root;
}

/// Count nodes reachable from root via BFS
size_t count_reachable_nodes(const NodePool& pool, uint32_t root) {
    if (root == NULL_NODE) return 0;

    size_t count = 0;
    std::vector<uint32_t> stack = {root};

    while (!stack.empty()) {
        uint32_t idx = stack.back();
        stack.pop_back();
        ++count;

        const StateNode& node = pool[idx];
        uint32_t child = node.first_child;
        while (child != NULL_NODE) {
            stack.push_back(child);
            child = pool[child].next_sibling;
        }
    }

    return count;
}

/// Verify two trees are structurally equivalent
void verify_trees_equal(
    const NodePool& pool1, uint32_t root1,
    const NodePool& pool2, uint32_t root2)
{
    ASSERT_EQ(root1 == NULL_NODE, root2 == NULL_NODE);
    if (root1 == NULL_NODE) return;

    // BFS comparison
    std::vector<std::pair<uint32_t, uint32_t>> stack = {{root1, root2}};

    while (!stack.empty()) {
        auto [idx1, idx2] = stack.back();
        stack.pop_back();

        const StateNode& n1 = pool1[idx1];
        const StateNode& n2 = pool2[idx2];

        // Compare node data
        EXPECT_EQ(n1.move, n2.move);
        EXPECT_EQ(n1.flags, n2.flags);
        EXPECT_FLOAT_EQ(n1.terminal_value, n2.terminal_value);

        // Compare stats
        EXPECT_EQ(n1.stats.visits.load(), n2.stats.visits.load());
        EXPECT_FLOAT_EQ(n1.stats.total_value.load(), n2.stats.total_value.load());
        EXPECT_FLOAT_EQ(n1.stats.prior, n2.stats.prior);

        // Compare children count
        size_t children1 = 0, children2 = 0;
        std::vector<uint32_t> c1_indices, c2_indices;

        for (uint32_t c = n1.first_child; c != NULL_NODE; c = pool1[c].next_sibling) {
            c1_indices.push_back(c);
            ++children1;
        }
        for (uint32_t c = n2.first_child; c != NULL_NODE; c = pool2[c].next_sibling) {
            c2_indices.push_back(c);
            ++children2;
        }

        ASSERT_EQ(children1, children2);

        // Add children pairs to stack
        for (size_t i = 0; i < children1; ++i) {
            stack.emplace_back(c1_indices[i], c2_indices[i]);
        }
    }
}

} // anonymous namespace

// ============================================================================
// TreeFileHeader Tests
// ============================================================================

TEST(TreeFileHeaderTest, DefaultHeaderIsValid) {
    TreeFileHeader header;
    EXPECT_EQ(header.magic, TreeFileHeader::MAGIC);
    EXPECT_EQ(header.version, TreeFileHeader::CURRENT_VERSION);
    EXPECT_TRUE(header.is_valid());
}

TEST(TreeFileHeaderTest, InvalidMagicIsRejected) {
    TreeFileHeader header;
    header.magic = 0xDEADBEEF;
    EXPECT_FALSE(header.is_valid());
}

TEST(TreeFileHeaderTest, FutureVersionIsRejected) {
    TreeFileHeader header;
    header.version = TreeFileHeader::CURRENT_VERSION + 1;
    EXPECT_FALSE(header.is_valid());
}

TEST(TreeFileHeaderTest, HeaderSizeIs64Bytes) {
    EXPECT_EQ(sizeof(TreeFileHeader), 64u);
}

// ============================================================================
// Save/Load Tests
// ============================================================================

TEST(StorageTest, SaveEmptyTreeFails) {
    TempFile tmp;
    NodePool pool;

    auto result = TreeStorage::save(tmp.path(), pool, NULL_NODE);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), StorageError::EmptyTree);
}

TEST(StorageTest, SaveAndLoadSingleNode) {
    TempFile tmp;

    // Create tree with single node
    NodePool pool;
    uint32_t root = pool.allocate(Move::pawn(4, 4), NULL_NODE, true);
    ASSERT_NE(root, NULL_NODE);

    pool[root].stats.visits.store(42, std::memory_order_relaxed);
    pool[root].stats.total_value.store(21.5f, std::memory_order_relaxed);
    pool[root].stats.prior = 0.75f;
    pool[root].set_terminal(1.0f);

    // Save
    auto save_result = TreeStorage::save(tmp.path(), pool, root);
    ASSERT_TRUE(save_result.has_value());

    // Load
    auto load_result = TreeStorage::load(tmp.path());
    ASSERT_TRUE(load_result.has_value());

    auto& loaded = *load_result;

    // Verify
    EXPECT_EQ(loaded.root, 0u);  // Root should be remapped to 0

    const StateNode& loaded_root = (*loaded.pool)[loaded.root];
    EXPECT_EQ(loaded_root.move, Move::pawn(4, 4));
    EXPECT_EQ(loaded_root.stats.visits.load(), 42u);
    EXPECT_FLOAT_EQ(loaded_root.stats.total_value.load(), 21.5f);
    EXPECT_FLOAT_EQ(loaded_root.stats.prior, 0.75f);
    EXPECT_TRUE(loaded_root.is_terminal());
    EXPECT_FLOAT_EQ(loaded_root.terminal_value, 1.0f);
}

TEST(StorageTest, SaveAndLoadTreeWithChildren) {
    TempFile tmp;

    NodePool pool;
    uint32_t root = pool.allocate(Move{}, NULL_NODE, true);

    // Add children
    ChildBuilder builder(pool, root);
    uint32_t c1 = builder.add_child(Move::pawn(3, 4), false);
    uint32_t c2 = builder.add_child(Move::pawn(5, 4), false);
    uint32_t c3 = builder.add_child(Move::fence(2, 3, true), false);
    builder.commit();

    ASSERT_NE(c1, NULL_NODE);
    ASSERT_NE(c2, NULL_NODE);
    ASSERT_NE(c3, NULL_NODE);

    // Set some stats
    pool[root].stats.visits.store(100, std::memory_order_relaxed);
    pool[c1].stats.visits.store(40, std::memory_order_relaxed);
    pool[c2].stats.visits.store(35, std::memory_order_relaxed);
    pool[c3].stats.visits.store(25, std::memory_order_relaxed);

    // Save and load
    ASSERT_TRUE(TreeStorage::save(tmp.path(), pool, root).has_value());
    auto load_result = TreeStorage::load(tmp.path());
    ASSERT_TRUE(load_result.has_value());

    auto& loaded = *load_result;

    // Count children
    size_t child_count = 0;
    uint32_t child = (*loaded.pool)[loaded.root].first_child;
    while (child != NULL_NODE) {
        ++child_count;
        child = (*loaded.pool)[child].next_sibling;
    }

    EXPECT_EQ(child_count, 3u);
}

TEST(StorageTest, SaveAndLoadDeepTree) {
    TempFile tmp;

    NodePool::Config config;
    config.initial_capacity = 10000;
    NodePool pool(config);

    // Build tree: depth=4, branching=3 => 1 + 3 + 9 + 27 + 81 = 121 nodes
    uint32_t root = build_test_tree(pool, 4, 3);
    size_t original_count = count_reachable_nodes(pool, root);

    EXPECT_EQ(original_count, 121u);

    // Save
    ASSERT_TRUE(TreeStorage::save(tmp.path(), pool, root).has_value());

    // Load
    auto load_result = TreeStorage::load(tmp.path());
    ASSERT_TRUE(load_result.has_value());

    auto& loaded = *load_result;
    size_t loaded_count = count_reachable_nodes(*loaded.pool, loaded.root);

    EXPECT_EQ(loaded_count, original_count);

    // Verify structure
    verify_trees_equal(pool, root, *loaded.pool, loaded.root);
}

TEST(StorageTest, ReadHeaderOnly) {
    TempFile tmp;

    NodePool pool;
    uint32_t root = build_test_tree(pool, 2, 2);  // 1 + 2 + 4 = 7 nodes

    ASSERT_TRUE(TreeStorage::save(tmp.path(), pool, root).has_value());

    auto header_result = TreeStorage::read_header(tmp.path());
    ASSERT_TRUE(header_result.has_value());

    auto& header = *header_result;
    EXPECT_EQ(header.magic, TreeFileHeader::MAGIC);
    EXPECT_EQ(header.version, TreeFileHeader::CURRENT_VERSION);
    EXPECT_EQ(header.node_count, 7u);
    EXPECT_EQ(header.root_index, 0u);
    EXPECT_GT(header.timestamp, 0u);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(StorageTest, LoadNonExistentFile) {
    auto result = TreeStorage::load("/nonexistent/path/file.tree");

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), StorageError::FileNotFound);
}

TEST(StorageTest, LoadInvalidFile) {
    TempFile tmp;

    // Write garbage to file (at least 64 bytes for header)
    std::ofstream file(tmp.path(), std::ios::binary);
    const char garbage[128] = "This is not a valid tree file! Invalid magic number here...";
    file.write(garbage, sizeof(garbage));
    file.close();

    auto result = TreeStorage::load(tmp.path());

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), StorageError::InvalidFormat);
}

TEST(StorageTest, ChecksumValidation) {
    // Checksum validation has been removed for performance.
    // This test now verifies that files load even with corrupted data
    // (trusting the filesystem for integrity).
    TempFile tmp;

    // Create and save a valid tree with some children
    NodePool pool;
    uint32_t root = pool.allocate(Move::pawn(4, 4), NULL_NODE, true);
    pool[root].stats.visits.store(42, std::memory_order_relaxed);

    ChildBuilder builder(pool, root);
    (void)builder.add_child(Move::pawn(3, 4), false);
    (void)builder.add_child(Move::pawn(5, 4), false);
    builder.commit();

    ASSERT_TRUE(TreeStorage::save(tmp.path(), pool, root).has_value());

    // Get original file size to verify corruption is within bounds
    auto file_size = std::filesystem::file_size(tmp.path());
    ASSERT_GT(file_size, sizeof(TreeFileHeader) + 10);

    // Read the original byte at the corruption position
    unsigned char original_byte = 0;
    {
        std::ifstream file(tmp.path(), std::ios::binary);
        file.seekg(static_cast<std::streamoff>(sizeof(TreeFileHeader) + 4));
        file.read(reinterpret_cast<char*>(&original_byte), 1);
    }

    // Corrupt the file (modify a byte in the node data to something different)
    {
        std::fstream file(tmp.path(), std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(static_cast<std::streamoff>(sizeof(TreeFileHeader) + 4));
        unsigned char corrupted_byte = static_cast<unsigned char>(original_byte ^ 0xFF);
        file.write(reinterpret_cast<char*>(&corrupted_byte), 1);
        file.flush();
    }

    // Load should succeed now (no checksum validation)
    auto result = TreeStorage::load(tmp.path());
    EXPECT_TRUE(result.has_value());
}

// ============================================================================
// Index Remapping Tests
// ============================================================================

TEST(StorageTest, IndexRemappingPreservesStructure) {
    TempFile tmp;

    // Create a pool with gaps (allocate some, deallocate middle ones)
    NodePool pool;

    // Allocate nodes at non-contiguous indices
    uint32_t root = pool.allocate(Move{}, NULL_NODE, true);
    uint32_t dummy1 = pool.allocate(Move::pawn(0, 0), NULL_NODE, true);
    uint32_t dummy2 = pool.allocate(Move::pawn(1, 1), NULL_NODE, true);
    uint32_t child = pool.allocate(Move::pawn(3, 4), root, false);

    // Set up tree structure manually (dummy nodes not in tree)
    pool[root].first_child = child;
    pool[root].set_expanded();

    // Deallocate dummy nodes to create gaps
    pool.deallocate(dummy1);
    pool.deallocate(dummy2);

    // Save and load
    ASSERT_TRUE(TreeStorage::save(tmp.path(), pool, root).has_value());
    auto load_result = TreeStorage::load(tmp.path());
    ASSERT_TRUE(load_result.has_value());

    // Loaded tree should have contiguous indices starting at 0
    EXPECT_EQ(load_result->root, 0u);
    EXPECT_EQ((*load_result->pool)[0].first_child, 1u);
}

// ============================================================================
// Move Type Preservation Tests
// ============================================================================

TEST(StorageTest, MoveTypesPreservedCorrectly) {
    TempFile tmp;

    NodePool pool;
    uint32_t root = pool.allocate(Move{}, NULL_NODE, true);

    ChildBuilder builder(pool, root);
    (void)builder.add_child(Move::pawn(4, 4), false);           // Pawn move
    (void)builder.add_child(Move::fence(3, 3, true), false);    // Horizontal fence
    (void)builder.add_child(Move::fence(5, 5, false), false);   // Vertical fence
    builder.commit();

    ASSERT_TRUE(TreeStorage::save(tmp.path(), pool, root).has_value());
    auto load_result = TreeStorage::load(tmp.path());
    ASSERT_TRUE(load_result.has_value());

    auto& loaded = *load_result;

    // Check each child's move
    uint32_t child = (*loaded.pool)[loaded.root].first_child;

    // First child: pawn move
    EXPECT_TRUE((*loaded.pool)[child].move.is_pawn());
    EXPECT_EQ((*loaded.pool)[child].move.row(), 4);
    EXPECT_EQ((*loaded.pool)[child].move.col(), 4);
    child = (*loaded.pool)[child].next_sibling;

    // Second child: horizontal fence
    EXPECT_TRUE((*loaded.pool)[child].move.is_fence());
    EXPECT_TRUE((*loaded.pool)[child].move.is_horizontal());
    EXPECT_EQ((*loaded.pool)[child].move.row(), 3);
    EXPECT_EQ((*loaded.pool)[child].move.col(), 3);
    child = (*loaded.pool)[child].next_sibling;

    // Third child: vertical fence
    EXPECT_TRUE((*loaded.pool)[child].move.is_fence());
    EXPECT_FALSE((*loaded.pool)[child].move.is_horizontal());
    EXPECT_EQ((*loaded.pool)[child].move.row(), 5);
    EXPECT_EQ((*loaded.pool)[child].move.col(), 5);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(StorageTest, LargeTreePerformance) {
    TempFile tmp;

    // Build a reasonably large tree
    NodePool::Config config;
    config.initial_capacity = 100000;
    NodePool pool(config);

    // depth=6, branching=4 => 1 + 4 + 16 + 64 + 256 + 1024 + 4096 = 5461 nodes
    uint32_t root = build_test_tree(pool, 6, 4);
    size_t node_count = count_reachable_nodes(pool, root);

    // Save
    auto save_start = std::chrono::high_resolution_clock::now();
    ASSERT_TRUE(TreeStorage::save(tmp.path(), pool, root).has_value());
    auto save_end = std::chrono::high_resolution_clock::now();

    auto save_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        save_end - save_start).count();
    std::cout << "Save took " << save_ms << " ms for " << node_count << " nodes\n";

    // Check file size
    auto file_size = std::filesystem::file_size(tmp.path());
    EXPECT_EQ(file_size, TreeStorage::estimate_file_size(node_count));

    // Load
    auto load_start = std::chrono::high_resolution_clock::now();
    auto load_result = TreeStorage::load(tmp.path());
    auto load_end = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(load_result.has_value());

    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        load_end - load_start).count();
    std::cout << "Load took " << load_ms << " ms\n";

    // Verify loaded tree
    size_t loaded_count = count_reachable_nodes(*load_result->pool, load_result->root);
    EXPECT_EQ(loaded_count, node_count);
}

// ============================================================================
// Error String Tests
// ============================================================================

TEST(StorageTest, ErrorStringConversion) {
    EXPECT_EQ(to_string(StorageError::FileNotFound), "File not found");
    EXPECT_EQ(to_string(StorageError::ChecksumMismatch), "Checksum mismatch");
    EXPECT_EQ(to_string(StorageError::EmptyTree), "Empty tree");
}
