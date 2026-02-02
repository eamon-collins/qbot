#pragma once

/// Simple profiling timers for self-play performance analysis
/// Usage:
///   ProfileTimers timers;
///   { ScopedTimer t(timers.expansion); do_expansion(); }
///   timers.print();

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <iomanip>

namespace qbot {

/// Accumulator for a single timed section
struct TimerAccumulator {
    std::atomic<uint64_t> total_ns{0};
    std::atomic<uint64_t> count{0};

    void add(uint64_t ns) noexcept {
        total_ns.fetch_add(ns, std::memory_order_relaxed);
        count.fetch_add(1, std::memory_order_relaxed);
    }

    void reset() noexcept {
        total_ns.store(0, std::memory_order_relaxed);
        count.store(0, std::memory_order_relaxed);
    }

    [[nodiscard]] double total_ms() const noexcept {
        return static_cast<double>(total_ns.load(std::memory_order_relaxed)) / 1e6;
    }

    [[nodiscard]] double avg_us() const noexcept {
        uint64_t c = count.load(std::memory_order_relaxed);
        if (c == 0) return 0.0;
        return static_cast<double>(total_ns.load(std::memory_order_relaxed)) / (c * 1000.0);
    }
};

/// RAII timer that adds elapsed time to an accumulator on destruction
class ScopedTimer {
public:
    explicit ScopedTimer(TimerAccumulator& acc) noexcept
        : acc_(acc), start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
        acc_.add(static_cast<uint64_t>(ns));
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    TimerAccumulator& acc_;
    std::chrono::high_resolution_clock::time_point start_;
};

/// All timers for self-play profiling
struct SelfPlayTimers {
    TimerAccumulator nn_inference;     // Total NN inference time (including batching, GPU transfers, etc.)
    TimerAccumulator mcts_core;        // MCTS core logic (selection, expansion, backprop) excluding NN
    TimerAccumulator pathfinding;      // Pathfinding for wall validation
    TimerAccumulator setup_gen;      // gen valid moves and some setup
    TimerAccumulator allocation;      // allocating and initializing nodes
    TimerAccumulator linking;      // linking child nodes

    void reset() noexcept {
        nn_inference.reset();
        mcts_core.reset();
        pathfinding.reset();
        setup_gen.reset();
        allocation.reset();
        linking.reset();
    }

    void print() const {
        auto print_line = [](const char* name, const TimerAccumulator& t) {
            std::cout << "  " << std::left << std::setw(20) << name
                      << std::right << std::setw(10) << std::fixed << std::setprecision(1)
                      << t.total_ms() << " ms"
                      << std::setw(12) << t.count.load(std::memory_order_relaxed) << " calls"
                      << std::setw(10) << std::setprecision(2) << t.avg_us() << " us/call\n";
        };

        std::cout << "\n=== Self-Play Timing Breakdown ===\n";
        print_line("NN Inference", nn_inference);
        print_line("MCTS Core", mcts_core);
        print_line("Pathfinding", pathfinding);
        print_line("setup_gen", setup_gen);
        print_line("allocation", allocation);
        print_line("linking", linking);
        std::cout << "==================================\n\n";
    }
};

/// Global timers instance (thread-safe due to atomics)
inline SelfPlayTimers& get_timers() {
    static SelfPlayTimers timers;
    return timers;
}

} // namespace qbot
