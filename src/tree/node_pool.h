// src/tree/node_pool.h
#pragma once

#include "StateNode.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <vector>
#include <thread>

namespace qbot {

/// High-performance Thread-Safe Node Pool
///
/// Features:
/// 1. TLAB (Thread-Local Allocation Buffer): Zero-contention allocation.
/// 2. Batch Recycling: Spills/Refills entire chains of nodes with 1 CAS.
/// 3. Memory Stability: Reuses hot memory before requesting new pages.
class NodePool {
public:
    static constexpr size_t MAX_CHUNKS = 4096; 

    struct Config {
        size_t initial_capacity = 180'000'000;
        size_t chunk_size = 20'000'000;
        size_t batch_size = 8192;
        size_t local_cache_limit = 1'000'000;
    };

    NodePool() : NodePool(Config{}) {}

    explicit NodePool(Config config)
        : config_(config)
        , chunk_size_(config.chunk_size)
        , fresh_index_(0)
        , total_capacity_(0)
    {
        // NOTE: Pool must be explicitly bound to thread via bind_to_thread()
        // This allows multiple pools to coexist (per-thread pools)

        for (auto& ptr : chunk_ptrs_) {
            ptr.store(nullptr, std::memory_order_relaxed);
        }

        size_t initial_chunks = (config.initial_capacity + config.chunk_size - 1) / config.chunk_size;
        if (initial_chunks == 0) initial_chunks = 1;
        if (initial_chunks > MAX_CHUNKS) initial_chunks = MAX_CHUNKS;

        chunk_storage_.reserve(MAX_CHUNKS);

        for (size_t i = 0; i < initial_chunks; ++i) {
            auto chunk = std::make_unique<StateNode[]>(chunk_size_);
            StateNode* raw_ptr = chunk.get();
            chunk_storage_.push_back(std::move(chunk));
            chunk_ptrs_[i].store(raw_ptr, std::memory_order_relaxed);
        }

        total_capacity_ = initial_chunks * chunk_size_;
    }

    /// Bind this pool to the current thread
    /// Must be called before any StateNode operations on this thread
    void bind_to_thread() noexcept {
        StateNode::set_pool(this);
    }

    NodePool(const NodePool&) = delete;
    NodePool& operator=(const NodePool&) = delete;

    [[nodiscard]] uint32_t allocate() noexcept {
        ThreadCache& cache = get_thread_cache();

        // 1. Thread-local recycle bin (Hot L1 Cache)
        if (!cache.recycled.empty()) {
            uint32_t idx = cache.recycled.back();
            cache.recycled.pop_back();
            node_at(idx).self_index = idx;
            return idx;
        }

        // 2. Thread-local bump pointer (Zero Contention)
        if (cache.bump_next < cache.bump_end) {
            uint32_t idx = cache.bump_next++;
            node_at(idx).self_index = idx;
            return idx;
        }

        // 3. Refill TLAB (Global Atomic Op)
        return refill_and_allocate(cache);
    }

    [[nodiscard]] uint32_t allocate(Move move, uint32_t parent_idx, bool p1_to_move) noexcept {
        uint32_t idx = allocate();
        if (idx != NULL_NODE) {
            node_at(idx).init(move, parent_idx, p1_to_move);
        }
        return idx;
    }

    void deallocate(uint32_t idx) noexcept {
        ThreadCache& cache = get_thread_cache();
        cache.recycled.push_back(idx);

        // No need to spill - we're the only thread using this pool
        // Just keep recycling locally
    }

    // Recursive deallocate
    void deallocate_subtree(uint32_t root_idx) noexcept {
        if (root_idx == NULL_NODE) return;

        // Iterative traversal to avoid stack overflow
        std::vector<uint32_t> to_delete;
        to_delete.reserve(128);
        to_delete.push_back(root_idx);

        size_t i = 0;
        while (i < to_delete.size()) {
            uint32_t idx = to_delete[i++];
            uint32_t child = node_at(idx).first_child;
            while (child != NULL_NODE) {
                to_delete.push_back(child);
                child = node_at(child).next_sibling;
            }
        }

        // Recycle all nodes
        for (auto it = to_delete.rbegin(); it != to_delete.rend(); ++it) {
            deallocate(*it);
        }
    }

    void batch_deallocate(uint32_t head, uint32_t tail, size_t count) noexcept {
        (void)tail; 
        uint32_t current = head;
        while (current != NULL_NODE && count > 0) {
            uint32_t next = node_at(current).next_sibling;
            deallocate(current);
            current = next;
            count--;
        }
    }

    // Minimal logic for maximum performance
    [[nodiscard]] inline StateNode& operator[](uint32_t idx) noexcept { return node_at(idx); }
    [[nodiscard]] inline const StateNode& operator[](uint32_t idx) const noexcept { return node_at(idx); }

    // Statistics
    [[nodiscard]] size_t capacity() const noexcept { return total_capacity_; }
    [[nodiscard]] size_t allocated() const noexcept {
        // Per-thread pool: allocated = fresh_index (high water mark)
        // Recycled nodes are still "allocated" (owned by this pool)
        return fresh_index_;
    }
    [[nodiscard]] size_t available() const noexcept { return capacity() - allocated(); }
    [[nodiscard]] float utilization() const noexcept {
        size_t cap = capacity(); return cap > 0 ? (float)allocated() / cap : 0.0f;
    }
    [[nodiscard]] size_t num_chunks() const noexcept {
        std::lock_guard lock(grow_mutex_);
        return chunk_storage_.size();
    }
    [[nodiscard]] size_t memory_usage_bytes() const noexcept { return allocated() * sizeof(StateNode); }

    void clear() noexcept {
        std::lock_guard lock(grow_mutex_);
        fresh_index_ = 0;
        // Clear thread cache as well
        ThreadCache& cache = get_thread_cache();
        cache.recycled.clear();
        cache.bump_next = 0;
        cache.bump_end = 0;
    }

private:
    struct ThreadCache {
        NodePool* owner_pool = nullptr; 
        uint32_t bump_next = 0;
        uint32_t bump_end = 0;
        std::vector<uint32_t> recycled;
    };

    static ThreadCache& get_thread_cache() {
        static thread_local ThreadCache cache;
        return cache;
    }

    // Inlined, minimal logic. 
    // Uses relaxed memory order because index validity is guaranteed by the allocation barrier (ACQUIRE).
    [[nodiscard]] inline StateNode& node_at(uint32_t idx) noexcept {
        return chunk_ptrs_[idx / chunk_size_].load(std::memory_order_relaxed)[idx % chunk_size_];
    }

    [[nodiscard]] inline const StateNode& node_at(uint32_t idx) const noexcept {
        return chunk_ptrs_[idx / chunk_size_].load(std::memory_order_relaxed)[idx % chunk_size_];
    }

    uint32_t refill_and_allocate(ThreadCache& cache) noexcept {
        if (cache.owner_pool != this) {
            cache = ThreadCache{};
            cache.owner_pool = this;
            cache.recycled.reserve(config_.local_cache_limit);
        }

        // Per-thread pool: no CAS needed! Just increment.
        uint32_t current_fresh = fresh_index_;
        size_t current_cap = total_capacity_;
        uint32_t batch = config_.batch_size;

        // Grow if needed
        if (current_fresh + batch > current_cap) {
            grow_with_lock();
            current_cap = total_capacity_;
        }

        // Claim batch (no contention, just assign)
        uint32_t result = current_fresh;
        fresh_index_ = current_fresh + batch;

        node_at(result).self_index = result;
        cache.bump_next = current_fresh + 1;
        cache.bump_end = current_fresh + batch;

        return result;
    }

    // NOTE: Global batch coordination removed for per-thread pools
    // Each pool is owned by a single thread, so no CAS/sharing needed

    void grow_with_lock() {
        std::lock_guard lock(grow_mutex_);

        size_t current_cap = total_capacity_;
        uint32_t current_fresh = fresh_index_;

        if (current_fresh + config_.batch_size <= current_cap) return;

        if (chunk_storage_.size() >= MAX_CHUNKS) {
            std::cerr << "[NodePool] Fatal: OOM - MAX_CHUNKS reached" << std::endl;
            std::terminate();
        }

        size_t current_chunks = chunk_storage_.size();
        auto chunk = std::make_unique<StateNode[]>(chunk_size_);
        StateNode* raw_ptr = chunk.get();

        chunk_storage_.push_back(std::move(chunk));
        chunk_ptrs_[current_chunks].store(raw_ptr, std::memory_order_release);

        size_t new_cap = (current_chunks + 1) * chunk_size_;
        total_capacity_ = new_cap;

        std::cout << "[NodePool] Grew to " << current_chunks + 1 << " chunks ("
                  << new_cap << " nodes)" << std::endl;
    }

    Config config_;
    size_t chunk_size_;

    std::vector<std::unique_ptr<StateNode[]>> chunk_storage_;
    std::atomic<StateNode*> chunk_ptrs_[MAX_CHUNKS];  // Keep atomic for safe reading across chunks

    // Per-thread pool: no contention, use simple variables
    uint32_t fresh_index_;
    size_t total_capacity_;
    mutable std::mutex grow_mutex_;  // Still need for grow (reallocates chunks)
};

} // namespace qbot
