#pragma once

#include "Node.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace qbot {

/// Memory-bounded node pool with LRU-based recycling
///
/// Design principles from memory-bounded MCTS paper:
/// - Pre-allocated arena of nodes for cache efficiency
/// - FIFO-based recycling when pool is exhausted
/// - Lock-free allocation via atomic free list
/// - Nodes are accessed by index, not pointer (stable across reallocations)
///
/// Thread safety:
/// - allocate() is thread-safe via atomic CAS on free list
/// - recycle() requires external synchronization (typically done during tree reuse)
/// - Access to individual nodes must be synchronized by the caller
class NodePool {
public:
    /// Configuration for node pool
    struct Config {
        size_t capacity = 1'000'000;       // Maximum number of nodes
        size_t recycle_batch = 1000;       // Nodes to recycle when pool exhausted
        bool enable_lru = true;            // Enable LRU tracking for recycling
    };

    NodePool() : NodePool(Config{}) {}

    explicit NodePool(Config config)
        : config_(config)
        , nodes_(config.capacity)
        , free_head_(0)
        , allocated_count_(0)
    {
        // Initialize free list: each node points to the next
        for (size_t i = 0; i < config.capacity - 1; ++i) {
            nodes_[i].next_sibling = static_cast<uint32_t>(i + 1);
        }
        nodes_[config.capacity - 1].next_sibling = NULL_NODE;

        if (config.enable_lru) {
            lru_queue_.reserve(config.capacity);
        }
    }

    // Non-copyable, non-movable (nodes are referenced by index)
    NodePool(const NodePool&) = delete;
    NodePool& operator=(const NodePool&) = delete;
    NodePool(NodePool&&) = delete;
    NodePool& operator=(NodePool&&) = delete;

    /// Allocate a node from the pool
    /// Returns NULL_NODE if pool is exhausted and recycling fails
    [[nodiscard]] uint32_t allocate() noexcept {
        uint32_t idx = try_allocate();
        if (idx != NULL_NODE) {
            return idx;
        }

        // Pool exhausted - try recycling if enabled
        if (config_.enable_lru) {
            std::lock_guard lock(recycle_mutex_);
            // Double-check after acquiring lock
            idx = try_allocate();
            if (idx != NULL_NODE) {
                return idx;
            }

            // Recycle oldest nodes
            if (recycle_lru_batch()) {
                return try_allocate();
            }
        }

        return NULL_NODE;
    }

    /// Allocate a node and initialize it
    [[nodiscard]] uint32_t allocate(Move move, uint32_t parent_idx, bool p1_to_move) noexcept {
        uint32_t idx = allocate();
        if (idx != NULL_NODE) {
            nodes_[idx].init(move, parent_idx, p1_to_move);
            if (config_.enable_lru) {
                touch(idx);
            }
        }
        return idx;
    }

    /// Return a node to the pool
    /// The caller must ensure the node is not referenced elsewhere
    void deallocate(uint32_t idx) noexcept {
        assert(idx < config_.capacity);

        // Push onto free list using CAS
        uint32_t expected = free_head_.load(std::memory_order_relaxed);
        do {
            nodes_[idx].next_sibling = expected;
        } while (!free_head_.compare_exchange_weak(
            expected, idx,
            std::memory_order_release, std::memory_order_relaxed));

        allocated_count_.fetch_sub(1, std::memory_order_relaxed);
    }

    /// Access a node by index
    [[nodiscard]] TreeNode& operator[](uint32_t idx) noexcept {
        assert(idx < config_.capacity);
        return nodes_[idx];
    }

    [[nodiscard]] const TreeNode& operator[](uint32_t idx) const noexcept {
        assert(idx < config_.capacity);
        return nodes_[idx];
    }

    /// Get a node, returning nullptr if invalid index
    [[nodiscard]] TreeNode* get(uint32_t idx) noexcept {
        if (idx >= config_.capacity) return nullptr;
        return &nodes_[idx];
    }

    [[nodiscard]] const TreeNode* get(uint32_t idx) const noexcept {
        if (idx >= config_.capacity) return nullptr;
        return &nodes_[idx];
    }

    /// Mark node as recently used (for LRU tracking)
    void touch(uint32_t idx) noexcept {
        if (!config_.enable_lru) return;

        std::lock_guard lock(lru_mutex_);
        // Simple append - actual LRU would need more sophisticated tracking
        // This is a FIFO approximation which is sufficient for MCTS
        lru_queue_.push_back(idx);
    }

    /// Pool statistics
    [[nodiscard]] size_t capacity() const noexcept { return config_.capacity; }
    [[nodiscard]] size_t allocated() const noexcept {
        return allocated_count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] size_t available() const noexcept {
        return config_.capacity - allocated();
    }
    [[nodiscard]] float utilization() const noexcept {
        return static_cast<float>(allocated()) / static_cast<float>(config_.capacity);
    }

    /// Clear all nodes and reset the pool
    void clear() noexcept {
        std::lock_guard lock1(recycle_mutex_);
        std::lock_guard lock2(lru_mutex_);

        // Rebuild free list
        for (size_t i = 0; i < config_.capacity - 1; ++i) {
            nodes_[i].next_sibling = static_cast<uint32_t>(i + 1);
        }
        nodes_[config_.capacity - 1].next_sibling = NULL_NODE;

        free_head_.store(0, std::memory_order_relaxed);
        allocated_count_.store(0, std::memory_order_relaxed);
        lru_queue_.clear();
    }

    /// Iterator support for traversing allocated nodes
    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = TreeNode;
        using difference_type = std::ptrdiff_t;
        using pointer = TreeNode*;
        using reference = TreeNode&;

        Iterator(NodePool* pool, uint32_t idx) : pool_(pool), idx_(idx) {
            advance_to_valid();
        }

        reference operator*() { return (*pool_)[idx_]; }
        pointer operator->() { return &(*pool_)[idx_]; }

        Iterator& operator++() {
            ++idx_;
            advance_to_valid();
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const Iterator& other) const { return idx_ == other.idx_; }
        bool operator!=(const Iterator& other) const { return idx_ != other.idx_; }

    private:
        void advance_to_valid() {
            // Skip nodes that are in the free list
            // This is O(n) and mainly for debugging - production code should
            // maintain a separate allocated list
            while (idx_ < pool_->config_.capacity && !is_allocated()) {
                ++idx_;
            }
        }

        bool is_allocated() const {
            // A node is allocated if its move is valid or it has stats
            return (*pool_)[idx_].move.is_valid() ||
                   (*pool_)[idx_].stats.visits.load(std::memory_order_relaxed) > 0;
        }

        NodePool* pool_;
        uint32_t idx_;
    };

    Iterator begin() { return Iterator(this, 0); }
    Iterator end() { return Iterator(this, static_cast<uint32_t>(config_.capacity)); }

private:
    /// Try to allocate from free list (lock-free)
    [[nodiscard]] uint32_t try_allocate() noexcept {
        uint32_t idx = free_head_.load(std::memory_order_relaxed);
        while (idx != NULL_NODE) {
            uint32_t next = nodes_[idx].next_sibling;
            if (free_head_.compare_exchange_weak(
                idx, next,
                std::memory_order_acquire, std::memory_order_relaxed)) {
                allocated_count_.fetch_add(1, std::memory_order_relaxed);
                return idx;
            }
            // idx is updated on CAS failure
        }
        return NULL_NODE;
    }

    /// Recycle a batch of LRU nodes
    /// Returns true if nodes were recycled
    /// Must be called with recycle_mutex_ held
    bool recycle_lru_batch() noexcept {
        if (lru_queue_.empty()) {
            return false;
        }

        size_t to_recycle = std::min(config_.recycle_batch, lru_queue_.size());
        size_t recycled = 0;

        for (size_t i = 0; i < to_recycle && recycled < config_.recycle_batch; ++i) {
            uint32_t idx = lru_queue_[i];

            // Only recycle if node is leaf and has low visit count
            TreeNode& node = nodes_[idx];
            if (!node.has_children() &&
                node.stats.visits.load(std::memory_order_relaxed) < 10) {

                // Unlink from parent
                if (node.parent != NULL_NODE) {
                    unlink_from_parent(idx);
                }

                deallocate(idx);
                ++recycled;
            }
        }

        // Remove processed entries from front of queue
        if (to_recycle > 0) {
            lru_queue_.erase(lru_queue_.begin(),
                            lru_queue_.begin() + static_cast<ptrdiff_t>(to_recycle));
        }

        return recycled > 0;
    }

    /// Unlink a node from its parent's child list
    void unlink_from_parent(uint32_t idx) noexcept {
        TreeNode& node = nodes_[idx];
        if (node.parent == NULL_NODE) return;

        TreeNode& parent = nodes_[node.parent];

        if (parent.first_child == idx) {
            // Node is first child
            parent.first_child = node.next_sibling;
        } else {
            // Find previous sibling
            uint32_t prev = parent.first_child;
            while (prev != NULL_NODE && nodes_[prev].next_sibling != idx) {
                prev = nodes_[prev].next_sibling;
            }
            if (prev != NULL_NODE) {
                nodes_[prev].next_sibling = node.next_sibling;
            }
        }

        node.parent = NULL_NODE;
        node.next_sibling = NULL_NODE;
    }

    Config config_;
    std::vector<TreeNode> nodes_;

    // Lock-free free list
    std::atomic<uint32_t> free_head_;
    std::atomic<size_t> allocated_count_;

    // LRU tracking (FIFO approximation)
    std::mutex lru_mutex_;
    std::vector<uint32_t> lru_queue_;

    // Recycling synchronization
    std::mutex recycle_mutex_;
};

/// RAII helper for allocating nodes
class ScopedNode {
public:
    ScopedNode(NodePool& pool, Move move, uint32_t parent, bool p1_to_move)
        : pool_(pool)
        , idx_(pool.allocate(move, parent, p1_to_move))
    {}

    ~ScopedNode() {
        if (idx_ != NULL_NODE && !released_) {
            pool_.deallocate(idx_);
        }
    }

    // Non-copyable
    ScopedNode(const ScopedNode&) = delete;
    ScopedNode& operator=(const ScopedNode&) = delete;

    // Movable
    ScopedNode(ScopedNode&& other) noexcept
        : pool_(other.pool_), idx_(other.idx_), released_(other.released_) {
        other.released_ = true;
    }

    [[nodiscard]] uint32_t index() const noexcept { return idx_; }
    [[nodiscard]] bool valid() const noexcept { return idx_ != NULL_NODE; }
    [[nodiscard]] TreeNode& node() { return pool_[idx_]; }

    /// Release ownership (prevent deallocation on destruction)
    uint32_t release() noexcept {
        released_ = true;
        return idx_;
    }

private:
    NodePool& pool_;
    uint32_t idx_;
    bool released_{false};
};

/// Helper to allocate multiple children atomically
class ChildBuilder {
public:
    explicit ChildBuilder(NodePool& pool, uint32_t parent_idx)
        : pool_(pool), parent_idx_(parent_idx) {}

    /// Add a child node, returns the child index or NULL_NODE on failure
    [[nodiscard]] uint32_t add_child(Move move, bool p1_to_move) {
        uint32_t idx = pool_.allocate(move, parent_idx_, p1_to_move);
        if (idx == NULL_NODE) {
            return NULL_NODE;
        }

        children_.push_back(idx);
        return idx;
    }

    /// Commit all children to the parent node
    /// Must be called after all children are added
    void commit() noexcept {
        if (children_.empty()) return;

        TreeNode& parent = pool_[parent_idx_];

        // Link children as siblings
        for (size_t i = 0; i < children_.size() - 1; ++i) {
            pool_[children_[i]].next_sibling = children_[i + 1];
        }
        pool_[children_.back()].next_sibling = NULL_NODE;

        // Set first child
        parent.first_child = children_.front();
        parent.set_expanded();

        committed_ = true;
    }

    /// Rollback - deallocate all children (called on error or destruction without commit)
    void rollback() noexcept {
        if (committed_) return;

        for (uint32_t idx : children_) {
            pool_.deallocate(idx);
        }
        children_.clear();
    }

    ~ChildBuilder() {
        if (!committed_) {
            rollback();
        }
    }

    [[nodiscard]] std::span<const uint32_t> children() const noexcept {
        return children_;
    }

    [[nodiscard]] size_t size() const noexcept { return children_.size(); }

private:
    NodePool& pool_;
    uint32_t parent_idx_;
    std::vector<uint32_t> children_;
    bool committed_{false};
};

} // namespace qbot
