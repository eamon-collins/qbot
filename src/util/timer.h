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
    TimerAccumulator expansion;        // expand_with_nn_priors
    TimerAccumulator mcts_iterations;  // run_mcts_iterations (all iterations for one move)
    TimerAccumulator single_mcts;      // single mcts_iteration
    TimerAccumulator policy_compute;   // compute_policy_from_q
    TimerAccumulator move_selection;   // select_move_from_policy
    TimerAccumulator child_lookup;     // finding child for selected move
    TimerAccumulator backprop;         // backpropagation
    TimerAccumulator nn_batch_eval;    // NN batch evaluation specifically
    TimerAccumulator nn_single_eval;   // NN single node evaluation
    TimerAccumulator generate_children;// generate_valid_children

    // Detailed inference breakdown
    TimerAccumulator tensor_alloc;     // torch::zeros calls
    TimerAccumulator tensor_fill;      // filling tensor data
    TimerAccumulator tensor_to_gpu;    // .to(device_) transfer
    TimerAccumulator model_forward;    // model_.forward()
    TimerAccumulator tensor_to_cpu;    // output.to(kCPU)

    void reset() noexcept {
        expansion.reset();
        mcts_iterations.reset();
        single_mcts.reset();
        policy_compute.reset();
        move_selection.reset();
        child_lookup.reset();
        backprop.reset();
        nn_batch_eval.reset();
        nn_single_eval.reset();
        generate_children.reset();
        tensor_alloc.reset();
        tensor_fill.reset();
        tensor_to_gpu.reset();
        model_forward.reset();
        tensor_to_cpu.reset();
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
        print_line("mcts_iterations", mcts_iterations);
        print_line("  single_mcts", single_mcts);
        print_line("expansion", expansion);
        print_line("  generate_children", generate_children);
        print_line("  nn_batch_eval", nn_batch_eval);
        print_line("nn_single_eval", nn_single_eval);
        print_line("policy_compute", policy_compute);
        print_line("move_selection", move_selection);
        print_line("child_lookup", child_lookup);
        print_line("backprop", backprop);
        std::cout << "--- Inference Detail ---\n";
        print_line("  tensor_alloc", tensor_alloc);
        print_line("  tensor_fill", tensor_fill);
        print_line("  tensor_to_gpu", tensor_to_gpu);
        print_line("  model_forward", model_forward);
        print_line("  tensor_to_cpu", tensor_to_cpu);
        std::cout << "==================================\n\n";
    }
};

/// Global timers instance (thread-safe due to atomics)
inline SelfPlayTimers& get_timers() {
    static SelfPlayTimers timers;
    return timers;
}

} // namespace qbot
