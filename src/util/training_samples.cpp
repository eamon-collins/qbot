#include "training_samples.h"

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>

namespace qbot {

std::array<float, NUM_ACTIONS> extract_visit_distribution(
    const NodePool& pool, uint32_t node_idx) noexcept
{
    std::array<float, NUM_ACTIONS> policy{};  // Zero-initialized

    const StateNode& parent = pool[node_idx];
    if (!parent.has_children()) {
        // No children - return uniform over all actions as fallback
        float uniform = 1.0f / NUM_ACTIONS;
        for (auto& p : policy) p = uniform;
        return policy;
    }

    // First pass: sum total visits across all children
    uint32_t total_visits = 0;
    uint32_t child = parent.first_child;
    while (child != NULL_NODE) {
        total_visits += pool[child].stats.visits.load(std::memory_order_relaxed);
        child = pool[child].next_sibling;
    }

    if (total_visits == 0) {
        // No visits recorded - use priors as fallback
        child = parent.first_child;
        while (child != NULL_NODE) {
            int action_idx = move_to_action_index(pool[child].move);
            if (action_idx >= 0 && action_idx < NUM_ACTIONS) {
                policy[action_idx] = pool[child].stats.prior;
            }
            child = pool[child].next_sibling;
        }
        return policy;
    }

    // Second pass: convert visit counts to probabilities
    child = parent.first_child;
    while (child != NULL_NODE) {
        int action_idx = move_to_action_index(pool[child].move);
        if (action_idx >= 0 && action_idx < NUM_ACTIONS) {
            uint32_t visits = pool[child].stats.visits.load(std::memory_order_relaxed);
            policy[action_idx] = static_cast<float>(visits) / static_cast<float>(total_visits);
        }
        child = pool[child].next_sibling;
    }

    return policy;
}

// ============================================================================
// TrainingSampleStorage Implementation
// ============================================================================

std::expected<void, TrainingSampleError> TrainingSampleStorage::save(
    const std::filesystem::path& path,
    const std::vector<TrainingSample>& samples)
{
    if (samples.empty()) {
        return std::unexpected(TrainingSampleError::NoSamples);
    }

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return std::unexpected(TrainingSampleError::IoError);
    }

    // Write header
    TrainingSampleHeader header;
    header.sample_count = static_cast<uint32_t>(samples.size());
    header.timestamp = static_cast<uint64_t>(
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!file) {
        return std::unexpected(TrainingSampleError::IoError);
    }

    // Write samples
    file.write(reinterpret_cast<const char*>(samples.data()),
               static_cast<std::streamsize>(samples.size() * sizeof(TrainingSample)));
    if (!file) {
        return std::unexpected(TrainingSampleError::IoError);
    }

    std::cout << "[TrainingSampleStorage] Saved " << samples.size() << " samples to " << path << "\n";
    return {};
}

std::expected<std::vector<TrainingSample>, TrainingSampleError> TrainingSampleStorage::load(
    const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path)) {
        return std::unexpected(TrainingSampleError::FileNotFound);
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::unexpected(TrainingSampleError::IoError);
    }

    // Read header
    TrainingSampleHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file) {
        return std::unexpected(TrainingSampleError::IoError);
    }

    if (!header.is_valid()) {
        return std::unexpected(TrainingSampleError::InvalidFormat);
    }

    if (header.version > TrainingSampleHeader::CURRENT_VERSION) {
        return std::unexpected(TrainingSampleError::VersionMismatch);
    }

    // Read samples
    std::vector<TrainingSample> samples(header.sample_count);
    file.read(reinterpret_cast<char*>(samples.data()),
              static_cast<std::streamsize>(header.sample_count * sizeof(TrainingSample)));
    if (!file) {
        return std::unexpected(TrainingSampleError::IoError);
    }

    std::cout << "[TrainingSampleStorage] Loaded " << samples.size() << " samples from " << path << "\n";
    return samples;
}

std::expected<TrainingSampleHeader, TrainingSampleError> TrainingSampleStorage::read_header(
    const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path)) {
        return std::unexpected(TrainingSampleError::FileNotFound);
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::unexpected(TrainingSampleError::IoError);
    }

    TrainingSampleHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file) {
        return std::unexpected(TrainingSampleError::IoError);
    }

    if (!header.is_valid()) {
        return std::unexpected(TrainingSampleError::InvalidFormat);
    }

    return header;
}

// ============================================================================
// TrainingSampleCollector Implementation
// ============================================================================

void TrainingSampleCollector::reserve(size_t count) {
    std::lock_guard lock(mutex_);
    samples_.reserve(count);
}

void TrainingSampleCollector::add_sample(const NodePool& pool, uint32_t node_idx, float game_outcome) {
    const StateNode& node = pool[node_idx];

    TrainingSample sample;

    // Extract compact state
    sample.state = extract_compact_state(node);

    // Extract MCTS visit distribution as policy target
    sample.policy = extract_visit_distribution(pool, node_idx);

    // Value target: game outcome from current player's perspective
    // game_outcome is from P1's perspective (+1 = P1 wins)
    // We need it from current player's perspective
    sample.value = node.is_p1_to_move() ? game_outcome : -game_outcome;

    std::lock_guard lock(mutex_);
    samples_.push_back(sample);
}

void TrainingSampleCollector::add_sample_direct(TrainingSample sample) {
    std::lock_guard lock(mutex_);
    samples_.push_back(std::move(sample));
}

std::vector<TrainingSample> TrainingSampleCollector::take_samples() noexcept {
    std::lock_guard lock(mutex_);
    return std::move(samples_);
}

// ============================================================================
// Tree Sample Extraction
// ============================================================================

namespace {

/// Extract samples from a subtree via DFS
/// Only extracts from nodes marked as on_game_path (part of a completed game)
void extract_samples_dfs(const NodePool& pool, uint32_t node_idx,
                         std::vector<TrainingSample>& samples) {
    const StateNode& node = pool[node_idx];

    // Only extract from nodes that were on an actual game path (had win/loss backpropagated)
    if (!node.is_on_game_path()) {
        return;
    }

    // Skip terminal nodes (no policy target - they have no children to form a distribution)
    if (node.is_terminal()) {
        return;
    }

    // Skip nodes without children (no policy target)
    if (!node.has_children()) {
        return;
    }

    // Check if this node has any visit counts (was used in search)
    uint32_t total_visits = 0;
    uint32_t child = node.first_child;
    while (child != NULL_NODE) {
        total_visits += pool[child].stats.visits.load(std::memory_order_relaxed);
        child = pool[child].next_sibling;
    }

    // Only extract sample if the node was actually searched
    if (total_visits > 0) {
        TrainingSample sample;
        sample.state = extract_compact_state(node);
        sample.policy = extract_visit_distribution(pool, node_idx);

        //should already be from current player's perspective and backpropagated sum of lower game path nodes.
        sample.value = node.stats.Q();

        samples.push_back(sample);
    }

    // Recurse to children that are on the game path
    child = node.first_child;
    while (child != NULL_NODE) {
        if (pool[child].is_on_game_path()) {
            extract_samples_dfs(pool, child, samples);
        }
        child = pool[child].next_sibling;
    }
}

} // anonymous namespace

std::vector<TrainingSample> extract_samples_from_tree(
    const NodePool& pool, uint32_t root_idx)
{
    std::vector<TrainingSample> samples;
    // samples.reserve(pool.allocated() / 200);  // Rough estimate

    extract_samples_dfs(pool, root_idx, samples);

    std::cout << "[extract_samples_from_tree] Extracted " << samples.size()
              << " samples from tree with " << pool.allocated() << " nodes" << std::endl;

    return samples;
}

} // namespace qbot
