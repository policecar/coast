#ifndef COAST_SIM_THREAD_CONTROLLER_H
#define COAST_SIM_THREAD_CONTROLLER_H

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cstdint>

namespace sim {

class env;

/**
 * Simulation execution mode.
 */
enum class sim_mode : std::uint8_t {
    paused,     ///< No stepping, waits for single_step commands
    throttled,  ///< Runs at target steps/second with sleep-based pacing
    unlimited   ///< Runs as fast as possible (CPU-bound)
};

/**
 * Controls simulation execution in a dedicated thread.
 *
 * The simulation thread runs independently of the GUI thread, with
 * lock-free communication for mode control and status queries.
 *
 * Thread model:
 * - GUI thread: calls set_mode(), set_steps_per_second(), single_step()
 * - Sim thread: runs process()/swap_io() loop, respects mode settings
 *
 * Data flow for visualization is handled through hooks registered on sim::env,
 * which run on the simulation thread and can copy data to GUI-owned buffers.
 */
class sim_thread_controller {
public:
    /**
     * Construct controller and start simulation thread.
     * Thread starts in paused mode.
     * @param sim_env Reference to simulation environment (must outlive controller)
     */
    explicit sim_thread_controller(env& sim_env);

    /**
     * Stop thread and wait for it to finish.
     * Uses std::jthread's automatic stop_token mechanism.
     */
    ~sim_thread_controller();

    // Non-copyable, non-movable (owns thread)
    sim_thread_controller(const sim_thread_controller&) = delete;
    sim_thread_controller& operator=(const sim_thread_controller&) = delete;
    sim_thread_controller(sim_thread_controller&&) = delete;
    sim_thread_controller& operator=(sim_thread_controller&&) = delete;

    // --- GUI thread interface: mode control ---

    /**
     * Set the simulation mode.
     * Takes effect on next iteration of simulation loop.
     */
    void set_mode(sim_mode mode);

    /**
     * Set target steps per second for throttled mode.
     * @param rate Target rate (must be positive)
     */
    void set_steps_per_second(double rate);

    /**
     * Request a single simulation step.
     * Only effective when in paused mode.
     */
    void single_step();

    // --- GUI thread interface: status queries (lock-free reads) ---

    /**
     * Get current simulation mode.
     */
    [[nodiscard]] sim_mode get_mode() const;

    /**
     * Get total number of simulation steps executed.
     */
    [[nodiscard]] std::uint64_t get_step_count() const;

    /**
     * Get measured steps per second (smoothed).
     */
    [[nodiscard]] double get_actual_sps() const;

    /**
     * Get target steps per second setting.
     */
    [[nodiscard]] double get_target_sps() const;

private:
    env& sim_env_;
    std::jthread thread_;

    // Atomics for lock-free control
    std::atomic<sim_mode> mode_{sim_mode::paused};
    std::atomic<double> target_sps_{1000.0};
    std::atomic<std::uint64_t> step_count_{0};
    std::atomic<double> actual_sps_{0.0};
    std::atomic<bool> single_step_pending_{false};

    // For waking thread when paused
    std::mutex wake_mutex_;
    std::condition_variable wake_cv_;

    /**
     * Main simulation thread function.
     * Runs until stop is requested via stop_token.
     */
    void thread_main(std::stop_token st);

    /**
     * Execute one simulation step (process + swap_io).
     */
    void do_step();
};

} // namespace sim

#endif // COAST_SIM_THREAD_CONTROLLER_H
