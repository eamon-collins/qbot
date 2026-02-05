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

    // If it's P2's turn, the board state in the sample is flipped 180°.
    // The policy indices must also be flipped to match the network's relative perspective.
    const bool flip_perspective = !parent.is_p1_to_move();

    // First pass: sum total visits across all children
    uint32_t total_visits = 0;
    uint32_t child = parent.first_child;
    while (child != NULL_NODE) {
        total_visits += pool[child].stats.visits.load(std::memory_order_relaxed);
        child = pool[child].next_sibling;
    }

    // Populate policy (visits if available, otherwise priors)
    const float scale = (total_visits > 0) ? (1.0f / static_cast<float>(total_visits)) : 1.0f;
    const bool use_visits = (total_visits > 0);

    child = parent.first_child;
    while (child != NULL_NODE) {
        const StateNode& child_node = pool[child];
        int action_idx = move_to_action_index(child_node.move);

        if (action_idx >= 0 && action_idx < NUM_ACTIONS) {
            // Use visits count if we searched, otherwise rely on the existing priors
            float value = use_visits 
                        ? (static_cast<float>(child_node.stats.visits.load(std::memory_order_relaxed)) * scale)
                        : child_node.stats.prior;

            if (flip_perspective) {
                action_idx = flip_action_index(action_idx);
            }

            policy[action_idx] = value;
        }
        child = child_node.next_sibling;
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


void TrainingSampleCollector::add_sample_direct(TrainingSample sample) {
    std::lock_guard lock(mutex_);
    samples_.push_back(std::move(sample));
}

std::vector<TrainingSample> TrainingSampleCollector::take_samples() noexcept {
    std::lock_guard lock(mutex_);
    return std::move(samples_);
}

} // namespace qbot
