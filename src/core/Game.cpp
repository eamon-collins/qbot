#include "Game.h"

#include <chrono>
#include <random>

namespace qbot {

Game::Game(Config config)
    : config_(config)
    , pool_(std::make_unique<NodePool>(NodePool::Config{.capacity = config.pool_capacity}))
    , root_(NULL_NODE)
{}

Game::~Game() = default;

size_t Game::build_tree(uint32_t root_idx, float branching_factor,
                        std::time_t time_limit, size_t node_limit) {
    if (root_idx == NULL_NODE) {
        return 0;
    }

    auto start_time = std::chrono::steady_clock::now();
    auto deadline = start_time + std::chrono::seconds(time_limit);

    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    size_t nodes_created = 0;
    std::vector<uint32_t> frontier;
    frontier.push_back(root_idx);

    while (!frontier.empty() && nodes_created < node_limit) {
        if (std::chrono::steady_clock::now() >= deadline) {
            break;
        }

        // Pick a random node from the frontier
        std::uniform_int_distribution<size_t> idx_dist(0, frontier.size() - 1);
        size_t pick = idx_dist(rng);
        uint32_t current_idx = frontier[pick];

        StateNode& current = (*pool_)[current_idx];

        // Skip terminal nodes
        if (current.is_terminal()) {
            frontier.erase(frontier.begin() + static_cast<ptrdiff_t>(pick));
            continue;
        }

        // Expand if not already expanded
        if (!current.is_expanded()) {
            size_t children_created = current.generate_valid_children(*pool_, current_idx);
            nodes_created += children_created;

            if (children_created == 0) {
                frontier.erase(frontier.begin() + static_cast<ptrdiff_t>(pick));
                continue;
            }
        }

        // Decide whether to explore siblings (breadth) or go deeper (depth)
        // branching_factor controls this: higher = more breadth, lower = more depth
        bool go_deeper = dist(rng) >= branching_factor;

        if (go_deeper) {
            // Remove current from frontier and add one of its children
            frontier.erase(frontier.begin() + static_cast<ptrdiff_t>(pick));

            // Collect children and pick one randomly
            std::vector<uint32_t> children;
            uint32_t child = current.first_child;
            while (child != NULL_NODE) {
                children.push_back(child);
                child = (*pool_)[child].next_sibling;
            }

            if (!children.empty()) {
                std::uniform_int_distribution<size_t> child_dist(0, children.size() - 1);
                frontier.push_back(children[child_dist(rng)]);
            }
        } else {
            // Add all children to frontier (breadth expansion)
            frontier.erase(frontier.begin() + static_cast<ptrdiff_t>(pick));

            uint32_t child = current.first_child;
            while (child != NULL_NODE) {
                frontier.push_back(child);
                child = (*pool_)[child].next_sibling;
            }
        }
    }

    return nodes_created;
}

} // namespace qbot
