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
        , allocated_count_(0)
        , total_capacity_(0)
        , global_batch_head_(NULL_NODE) // Points to head of the batch stack
    {
        StateNode::set_pool(this);

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

        total_capacity_.store(initial_chunks * chunk_size_, std::memory_order_relaxed);
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

        // Spill if too full
        if (cache.recycled.size() >= config_.local_cache_limit) {
            spill_to_global(cache);
        }
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
    [[nodiscard]] size_t capacity() const noexcept { return total_capacity_.load(std::memory_order_relaxed); }
    [[nodiscard]] size_t allocated() const noexcept { return allocated_count_.load(std::memory_order_relaxed); }
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
        fresh_index_.store(0, std::memory_order_relaxed);
        global_batch_head_.store(NULL_NODE, std::memory_order_relaxed);
        allocated_count_.store(0, std::memory_order_relaxed);
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

        // 1. Try to recycle a batch from global free list FIRST
        // This keeps memory usage stable and reuses hot cache lines.
        if (refill_from_recycled_batch(cache)) {
            uint32_t idx = cache.recycled.back();
            cache.recycled.pop_back();
            node_at(idx).self_index = idx;
            return idx;
        }

        // 2. If no recycled batches, fallback to fresh allocation
        while (true) {
            uint32_t current_fresh = fresh_index_.load(std::memory_order_relaxed);
            // ACQUIRE capacity to ensure chunk pointers are visible
            size_t current_cap = total_capacity_.load(std::memory_order_acquire);
            uint32_t batch = config_.batch_size;

            if (current_fresh + batch <= current_cap) {
                // Try to claim batch. If failed, loop again (don't increment blindly)
                if (fresh_index_.compare_exchange_weak(current_fresh, current_fresh + batch, 
                                                     std::memory_order_relaxed, 
                                                     std::memory_order_relaxed)) {
                    allocated_count_.fetch_add(batch, std::memory_order_relaxed);
                    uint32_t result = current_fresh;
                    node_at(result).self_index = result;
                    cache.bump_next = current_fresh + 1;
                    cache.bump_end = current_fresh + batch;
                    return result;
                }
                continue; 
            }

            // OOM - Grow
            grow_with_lock();
        }
    }

    // Pop a whole chain of nodes from global stack
    bool refill_from_recycled_batch(ThreadCache& cache) {
        uint32_t batch_head = global_batch_head_.load(std::memory_order_relaxed);
        while (batch_head != NULL_NODE) {
            // The first node of the batch contains the pointer to the next batch in its 'parent' field
            // (We repurpose 'parent' while the node is in the free list to save space)
            uint32_t next_batch = node_at(batch_head).parent;

            if (global_batch_head_.compare_exchange_weak(batch_head, next_batch, 
                                                       std::memory_order_acquire, 
                                                       std::memory_order_relaxed)) {
                // Success! We grabbed the batch. Walk it and push to local.
                uint32_t curr = batch_head;
                int count = 0;
                while (curr != NULL_NODE) {
                    cache.recycled.push_back(curr);
                    curr = node_at(curr).next_sibling;
                    count++;
                }
                allocated_count_.fetch_add(count, std::memory_order_relaxed);
                return true;
            }
        }
        return false;
    }

    // Push a chain of nodes to global stack
    void spill_to_global(ThreadCache& cache) {
        size_t count = cache.recycled.size() / 2;
        if (count == 0) return;

        // 1. Link the nodes into a chain: head -> ... -> tail -> NULL
        uint32_t head = cache.recycled.back();
        uint32_t curr = head;
        cache.recycled.pop_back();

        for (size_t i = 1; i < count; ++i) {
            uint32_t next = cache.recycled.back();
            cache.recycled.pop_back();
            node_at(curr).next_sibling = next;
            curr = next;
        }
        node_at(curr).next_sibling = NULL_NODE; // Terminate chain

        // 2. Push chain to global stack
        uint32_t old_batch_head = global_batch_head_.load(std::memory_order_relaxed);
        do {
            // Repurpose 'parent' to point to the next batch
            node_at(head).parent = old_batch_head;
        } while (!global_batch_head_.compare_exchange_weak(old_batch_head, head,
                                                         std::memory_order_release,
                                                         std::memory_order_relaxed));

        allocated_count_.fetch_sub(count, std::memory_order_relaxed);
    }

    void grow_with_lock() {
        std::lock_guard lock(grow_mutex_);

        size_t current_cap = total_capacity_.load(std::memory_order_relaxed);
        uint32_t current_fresh = fresh_index_.load(std::memory_order_relaxed);

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
        total_capacity_.store(new_cap, std::memory_order_release);

        std::cout << "[NodePool] Grew to " << current_chunks + 1 << " chunks (" 
                  << new_cap << " nodes)" << std::endl;
    }

    Config config_;
    size_t chunk_size_;

    std::vector<std::unique_ptr<StateNode[]>> chunk_storage_;
    std::atomic<StateNode*> chunk_ptrs_[MAX_CHUNKS];

    std::atomic<uint32_t> fresh_index_;

    // Points to the first node of the most recently spilled batch
    std::atomic<uint32_t> global_batch_head_;

    std::atomic<size_t> allocated_count_;
    std::atomic<size_t> total_capacity_;
    mutable std::mutex grow_mutex_;
};

} // namespace qbot
