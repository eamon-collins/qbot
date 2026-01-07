#pragma once

#include "StateNode.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace qbot {

/// Chunked node pool with automatic growth
///
/// Uses chunked allocation to avoid copies during resize:
/// - Nodes are stored in fixed-size chunks
/// - When pool is exhausted, a new chunk is allocated (no copying)
/// - Indices remain stable across growth
/// - Each chunk is cache-efficient for traversal
///
/// Thread safety:
/// - allocate() is thread-safe via atomic CAS on free list
/// - grow() uses a mutex to ensure only one thread grows at a time
/// - Access to individual nodes must be synchronized by the caller
class NodePool {
public:
    /// Configuration for node pool
    struct Config {
        size_t initial_capacity = 10'000'000;  // Initial number of nodes (10M)
        size_t chunk_size = 10'000'000;        // Nodes per chunk (10M)
        size_t recycle_batch = 1000;           // Nodes to recycle when pool exhausted
        bool enable_lru = true;                // Enable LRU tracking for recycling
    };

    NodePool() : NodePool(Config{}) {}

    explicit NodePool(Config config)
        : config_(config)
        , chunk_size_(config.chunk_size)
        , free_head_(0)  // Will point to index 0 after initialization
        , allocated_count_(0)
        , total_capacity_(0)
    {
        // Set static pool pointer for StateNode access
        StateNode::set_pool(this);

        // Calculate initial number of chunks
        size_t initial_chunks = (config.initial_capacity + config.chunk_size - 1) / config.chunk_size;
        if (initial_chunks == 0) initial_chunks = 1;

        // Allocate all initial chunks
        for (size_t i = 0; i < initial_chunks; ++i) {
            chunks_.push_back(std::make_unique<StateNode[]>(chunk_size_));
        }

        // Build free list in FORWARD order (0 -> 1 -> 2 -> ... -> N-1 -> NULL)
        // This ensures sequential allocation which is required for save/load
        size_t total = initial_chunks * chunk_size_;
        for (size_t i = 0; i + 1 < total; ++i) {
            node_at(static_cast<uint32_t>(i)).next_sibling = static_cast<uint32_t>(i + 1);
        }
        node_at(static_cast<uint32_t>(total - 1)).next_sibling = NULL_NODE;

        total_capacity_.store(total, std::memory_order_relaxed);
        free_head_.store(0, std::memory_order_relaxed);

        if (config.enable_lru) {
            lru_queue_.reserve(config.initial_capacity);
        }
    }

    // Non-copyable, non-movable (nodes are referenced by index)
    NodePool(const NodePool&) = delete;
    NodePool& operator=(const NodePool&) = delete;
    NodePool(NodePool&&) = delete;
    NodePool& operator=(NodePool&&) = delete;

    /// Allocate a node from the pool
    /// Automatically grows the pool if exhausted
    /// Sets the node's self_index automatically
    [[nodiscard]] uint32_t allocate() noexcept {
        uint32_t idx = try_allocate();
        if (idx != NULL_NODE) {
            node_at(idx).self_index = idx;
            return idx;
        }

        // Pool exhausted - grow it
        {
            std::lock_guard lock(grow_mutex_);
            // Double-check after acquiring lock
            idx = try_allocate();
            if (idx != NULL_NODE) {
                node_at(idx).self_index = idx;
                return idx;
            }

            // Grow the pool by adding a new chunk
            grow();
            idx = try_allocate();
            if (idx != NULL_NODE) {
                node_at(idx).self_index = idx;
            }
            return idx;
        }
    }

    /// Allocate a node and initialize it
    [[nodiscard]] uint32_t allocate(Move move, uint32_t parent_idx, bool p1_to_move) noexcept {
        uint32_t idx = allocate();
        if (idx != NULL_NODE) {
            node_at(idx).init(move, parent_idx, p1_to_move);
            if (config_.enable_lru) {
                touch(idx);
            }
        }
        return idx;
    }

    /// Return a node to the pool
    /// The caller must ensure the node is not referenced elsewhere
    void deallocate(uint32_t idx) noexcept {
        assert(idx < total_capacity_.load(std::memory_order_relaxed));

        // Push onto free list using CAS
        uint32_t expected = free_head_.load(std::memory_order_relaxed);
        do {
            node_at(idx).next_sibling = expected;
        } while (!free_head_.compare_exchange_weak(
            expected, idx,
            std::memory_order_release, std::memory_order_relaxed));

        allocated_count_.fetch_sub(1, std::memory_order_relaxed);
    }

    /// Access a node by index (chunked indexing)
    [[nodiscard]] StateNode& operator[](uint32_t idx) noexcept {
        return node_at(idx);
    }

    [[nodiscard]] const StateNode& operator[](uint32_t idx) const noexcept {
        return node_at(idx);
    }

    /// Get a node, returning nullptr if invalid index
    [[nodiscard]] StateNode* get(uint32_t idx) noexcept {
        if (idx >= total_capacity_.load(std::memory_order_relaxed)) return nullptr;
        return &node_at(idx);
    }

    [[nodiscard]] const StateNode* get(uint32_t idx) const noexcept {
        if (idx >= total_capacity_.load(std::memory_order_relaxed)) return nullptr;
        return &node_at(idx);
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
    [[nodiscard]] size_t capacity() const noexcept {
        return total_capacity_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] size_t allocated() const noexcept {
        return allocated_count_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] size_t available() const noexcept {
        return capacity() - allocated();
    }
    [[nodiscard]] float utilization() const noexcept {
        return static_cast<float>(allocated()) / static_cast<float>(capacity());
    }
    [[nodiscard]] size_t num_chunks() const noexcept {
        std::lock_guard lock(grow_mutex_);
        return chunks_.size();
    }

    /// Memory usage in bytes (allocated nodes * node size)
    [[nodiscard]] size_t memory_usage_bytes() const noexcept {
        return allocated() * sizeof(StateNode);
    }

    /// Total memory capacity in bytes
    [[nodiscard]] size_t memory_capacity_bytes() const noexcept {
        return capacity() * sizeof(StateNode);
    }

    /// Clear all nodes and reset the pool
    void clear() noexcept {
        std::lock_guard lock1(grow_mutex_);
        std::lock_guard lock2(lru_mutex_);

        // Rebuild free list across all chunks
        size_t total = total_capacity_.load(std::memory_order_relaxed);
        for (size_t i = 0; i < total - 1; ++i) {
            node_at(static_cast<uint32_t>(i)).next_sibling = static_cast<uint32_t>(i + 1);
        }
        node_at(static_cast<uint32_t>(total - 1)).next_sibling = NULL_NODE;

        free_head_.store(0, std::memory_order_relaxed);
        allocated_count_.store(0, std::memory_order_relaxed);
        lru_queue_.clear();
    }

    /// Iterator support for traversing allocated nodes
    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = StateNode;
        using difference_type = std::ptrdiff_t;
        using pointer = StateNode*;
        using reference = StateNode&;

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
            while (idx_ < pool_->capacity() && !is_allocated()) {
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
    Iterator end() { return Iterator(this, static_cast<uint32_t>(capacity())); }

private:
    /// Access node by index using chunked storage
    [[nodiscard]] StateNode& node_at(uint32_t idx) noexcept {
        size_t chunk_idx = idx / chunk_size_;
        size_t offset = idx % chunk_size_;
        return chunks_[chunk_idx][offset];
    }

    [[nodiscard]] const StateNode& node_at(uint32_t idx) const noexcept {
        size_t chunk_idx = idx / chunk_size_;
        size_t offset = idx % chunk_size_;
        return chunks_[chunk_idx][offset];
    }

    /// Try to allocate from free list (lock-free)
    [[nodiscard]] uint32_t try_allocate() noexcept {
        uint32_t idx = free_head_.load(std::memory_order_relaxed);
        while (idx != NULL_NODE) {
            uint32_t next = node_at(idx).next_sibling;
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

    /// Grow the pool by adding a new chunk
    /// Must be called with grow_mutex_ held
    void grow() noexcept {
        auto start = std::chrono::high_resolution_clock::now();

        add_chunk_unlocked();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "[NodePool] Grew to " << chunks_.size() << " chunks ("
                  << total_capacity_.load(std::memory_order_relaxed) << " nodes) in "
                  << duration_us << " us (no copy needed)\n";
    }

    /// Add a new chunk without holding the grow mutex
    /// Called during construction and from grow()
    void add_chunk_unlocked() noexcept {
        size_t old_capacity = total_capacity_.load(std::memory_order_relaxed);

        // Allocate new chunk
        chunks_.push_back(std::make_unique<StateNode[]>(chunk_size_));

        // Initialize free list for the new chunk
        uint32_t chunk_start = static_cast<uint32_t>(old_capacity);
        uint32_t chunk_end = static_cast<uint32_t>(old_capacity + chunk_size_ - 1);

        for (uint32_t i = chunk_start; i < chunk_end; ++i) {
            node_at(i).next_sibling = i + 1;
        }

        // Link new chunk's last node to existing free list head
        uint32_t old_head = free_head_.load(std::memory_order_relaxed);
        node_at(chunk_end).next_sibling = old_head;

        // Update capacity before updating free head (so other threads see valid indices)
        total_capacity_.store(old_capacity + chunk_size_, std::memory_order_release);

        // Point free head to start of new chunk
        free_head_.store(chunk_start, std::memory_order_release);
    }

    /// Recycle a batch of LRU nodes
    /// Returns true if nodes were recycled
    /// Must be called with grow_mutex_ held
    bool recycle_lru_batch() noexcept {
        if (lru_queue_.empty()) {
            return false;
        }

        size_t to_recycle = std::min(config_.recycle_batch, lru_queue_.size());
        size_t recycled = 0;

        for (size_t i = 0; i < to_recycle && recycled < config_.recycle_batch; ++i) {
            uint32_t idx = lru_queue_[i];

            // Only recycle if node is leaf and has low visit count
            StateNode& node = node_at(idx);
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
        StateNode& node = node_at(idx);
        if (node.parent == NULL_NODE) return;

        StateNode& parent = node_at(node.parent);

        if (parent.first_child == idx) {
            // Node is first child
            parent.first_child = node.next_sibling;
        } else {
            // Find previous sibling
            uint32_t prev = parent.first_child;
            while (prev != NULL_NODE && node_at(prev).next_sibling != idx) {
                prev = node_at(prev).next_sibling;
            }
            if (prev != NULL_NODE) {
                node_at(prev).next_sibling = node.next_sibling;
            }
        }

        node.parent = NULL_NODE;
        node.next_sibling = NULL_NODE;
    }

    Config config_;
    size_t chunk_size_;

    // Chunked storage - each chunk is a fixed-size array of nodes
    std::vector<std::unique_ptr<StateNode[]>> chunks_;

    // Lock-free free list
    std::atomic<uint32_t> free_head_;
    std::atomic<size_t> allocated_count_;
    std::atomic<size_t> total_capacity_;

    // LRU tracking (FIFO approximation)
    std::mutex lru_mutex_;
    std::vector<uint32_t> lru_queue_;

    // Growth synchronization
    mutable std::mutex grow_mutex_;
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
    [[nodiscard]] StateNode& node() { return pool_[idx_]; }

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
/// Test only workflows
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

        StateNode& parent = pool_[parent_idx_];

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
