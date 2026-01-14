#include "Game.h"

#ifdef QBOT_ENABLE_INFERENCE
#include "../inference/inference.h"
#endif

#include <chrono>
#include <cmath>
#include <limits>
#include <random>

namespace qbot {

Game::Game(Config config)
    : config_(config)
    , pool_(std::make_unique<NodePool>(NodePool::Config{.initial_capacity = config.pool_capacity}))
    , root_(NULL_NODE)
{}

Game::Game(std::unique_ptr<NodePool> pool, uint32_t root)
    : config_()
    , pool_(std::move(pool))
    , root_(root)
{}

Game::~Game() {
    disconnect_gui();
}

bool Game::connect_gui(const GUIClient::Config& config) {
    if (!gui_) {
        gui_ = std::make_unique<GUIClient>();
    }
    return gui_->connect(config);
}

void Game::disconnect_gui() {
    if (gui_) {
        gui_->disconnect();
    }
}

bool Game::has_gui() const noexcept {
    return gui_ && gui_->is_connected();
}

void Game::visualize_state(const StateNode& node, float score) {
    qbot::visualize_state(node, gui_.get(), -1, score);
}

void Game::visualize_root(float score) {
    if (root_ != NULL_NODE) {
        visualize_state((*pool_)[root_], score);
    }
}

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
            size_t children_created = current.generate_valid_children();
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

Move Game::select_best_move_by_q(NodePool& pool, uint32_t node_idx) {
    StateNode& current = pool[node_idx];

    // Ensure children are generated
    if (!current.is_expanded()) {
        current.generate_valid_children();
    }

    // Collect all children with their scores
    struct ScoredChild {
        Move move;
        float score;
    };
    std::vector<ScoredChild> children;

    uint32_t child_idx = current.first_child;
    while (child_idx != NULL_NODE) {
        StateNode& child = pool[child_idx];
        float q = child.stats.Q(0.0f);
        children.push_back({child.move, q});
        child_idx = child.next_sibling;
    }

    if (children.empty()) {
        return Move{};
    }

    thread_local std::mt19937 rng(std::random_device{}());

    // Find best score
    float best_score = -std::numeric_limits<float>::infinity();
    for (const auto& c : children) {
        if (c.score > best_score) {
            best_score = c.score;
        }
    }

    // Collect all with best score (within epsilon)
    std::vector<Move> best_moves;
    for (const auto& c : children) {
        if (std::abs(c.score - best_score) < 1e-6f) {
            best_moves.push_back(c.move);
        }
    }

    std::uniform_int_distribution<size_t> dist(0, best_moves.size() - 1);
    return best_moves[dist(rng)];
}

Move Game::select_best_move(uint32_t node_idx) {
#ifdef QBOT_ENABLE_INFERENCE
    if (model_ && model_->is_ready()) {
        StateNode& current = (*pool_)[node_idx];

        // Ensure children are generated
        if (!current.is_expanded()) {
            current.generate_valid_children();
        }

        if (current.first_child == NULL_NODE) {
            return Move{};
        }

        // P1 wants to maximize score (closer to +1), P2 wants to minimize (closer to -1)
        bool maximize = current.is_p1_to_move();
        Move best_move;
        float best_score = maximize ? -std::numeric_limits<float>::infinity()
                                    : std::numeric_limits<float>::infinity();

        uint32_t child_idx = current.first_child;
        while (child_idx != NULL_NODE) {
            StateNode& child = (*pool_)[child_idx];
            auto result = model_->evaluate_node(&child);
            float score = result.value;

            bool dominated = maximize ? (score > best_score) : (score < best_score);
            if (dominated) {
                best_score = score;
                best_move = child.move;
            }

            child_idx = child.next_sibling;
        }

        return best_move;
    }
#endif
    // Fall back to Q-value selection
    return select_best_move_by_q(*pool_, node_idx);
}

} // namespace qbot
