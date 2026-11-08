#ifndef COAST_SIM_THREAD_CONTROLLER_H
#define COAST_SIM_THREAD_CONTROLLER_H

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cstdint>
#include <cassert>
#include <deque>
#include <functional>
#include <future>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "core/step_error.h"

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

    /**
     * Run `fn` on the simulation thread at the next quiescent boundary, and
     * block the caller until it has completed; returns fn's result (or rethrows
     * its exception).
     *
     * The boundary is the coherent post-swap point: right after a do_step()
     * (process + swap_io) fully completes, or — if the sim is not stepping —
     * while parked in the paused wait (an enqueue wakes it). Because `fn`
     * executes ON the sim thread at that boundary it may freely touch state the
     * sim thread owns: io_buffer read pages, the wgpu queue (readbacks /
     * uploads), entity save_state/load_state, and the step counter. No step runs
     * for fn's duration, so the count is frozen while it executes; the
     * controller's mode is untouched.
     *
     * This is the single quiescence seam for all non-sim-thread access to sim
     * state (state save/load/checkpoint, set_step_count): rather than parking the
     * thread from outside and doing the work on the caller's thread (off the
     * wgpu-queue owner), the work runs on the correct thread and the caller
     * simply waits for the result.
     *
     * Multiple producers may call concurrently (MPSC); tasks run one at a time
     * in FIFO order. Must NOT be called on the sim thread itself (self-deadlock;
     * asserted). If the controller is shutting down the call does not block — it
     * throws std::runtime_error rather than wait for a thread that will no
     * longer drain (the stop-escape).
     */
    template <class F>
    auto run_quiesced(F&& fn) -> std::invoke_result_t<F&>
    {
        using result_t = std::invoke_result_t<F&>;

        // Only non-sim threads may block on the sim thread draining the task.
        assert(std::this_thread::get_id() !=
               sim_thread_id_.load(std::memory_order_acquire) &&
               "run_quiesced must not be called on the sim thread");

        auto prom    = std::make_shared<std::promise<result_t>>();
        auto fut     = prom->get_future();
        auto wrapped = std::make_shared<std::decay_t<F>>(std::forward<F>(fn));

        // One queue entry. reject=false runs fn and fulfils the promise (on the
        // sim thread, at the boundary). reject=true fulfils it with the stopping
        // exception WITHOUT running fn (teardown path). Both shared_ptr captures
        // keep the entry copyable for std::function.
        auto job = [wrapped, prom](bool reject) {
            if (reject) {
                prom->set_exception(std::make_exception_ptr(std::runtime_error(
                    "sim_thread_controller stopping; quiesced task not run")));
                return;
            }
            try {
                if constexpr (std::is_void_v<result_t>) { (*wrapped)(); prom->set_value(); }
                else                                    { prom->set_value((*wrapped)()); }
            } catch (...) {
                prom->set_exception(std::current_exception());
            }
        };

        bool rejected = false;
        {
            std::lock_guard lock(wake_mutex_);
            if (quiesce_stopping_ || thread_.get_stop_token().stop_requested()) {
                rejected = true;
            } else {
                quiesced_queue_.push_back(std::move(job));
                quiesced_pending_.store(true, std::memory_order_release);
                // Notify UNDER the lock (as set_mode()/single_step() do): the
                // controller's teardown acquires wake_mutex_, which gives it a
                // happens-before edge to this cv access — so wake_cv_ is never
                // destroyed concurrently with a requester's notify, even though
                // the requester runs on a different thread than the owner.
                wake_cv_.notify_all();   // wake a parked sim thread to drain
            }
        }
        if (rejected) {
            // Reject outside the lock; unblocks the caller via fut.get() below.
            prom->set_exception(std::make_exception_ptr(std::runtime_error(
                "sim_thread_controller stopping; quiesced task not run")));
        }
        return fut.get();
    }

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
     * Set the step count (for restoring from checkpoint).
     *
     * Call inside a run_quiesced() task (sim thread, atomic with the state apply)
     * or at startup before stepping — never from another thread mid-run, which
     * races the per-step counter update. Does not require paused mode.
     */
    void set_step_count(std::uint64_t count);

    /**
     * Set a step limit - simulation will auto-pause when reaching this step.
     * Set to 0 to disable (default).
     */
    void set_step_limit(std::uint64_t limit);

    /**
     * Get measured steps per second (smoothed).
     */
    [[nodiscard]] double get_actual_sps() const;

    /**
     * Get target steps per second setting.
     */
    [[nodiscard]] double get_target_sps() const;

    // --- Hot-path fault surfacing ---

    /// A recorded per-step fault: the hot-path error plus the step it occurred on
    /// (the controller stamps the step, since step_error itself stays lean).
    struct step_fault {
        core::step_error error;
        std::uint64_t    step;
    };

    /**
     * Halt the run from outside the step loop, sticky like a per-step fault, and
     * record `e` as the fault at the current step (the first fault wins). For a
     * runtime state load that failed after applying part of the checkpoint, so
     * the sim cannot step on a partly loaded env. Call it on the sim thread
     * inside the run_quiesced() task that failed, so no step runs in between.
     */
    void halt(const core::step_error& e);

    /**
     * True once a per-step process() fault, or halt(), has halted the
     * simulation. Sticky: the controller forces paused mode and will not resume
     * stepping. Lock-free.
     */
    [[nodiscard]] bool is_halted() const {
        return halted_.load(std::memory_order_acquire);
    }

    /**
     * The first recorded fault (sticky), or nullopt if the run has not faulted.
     * Used by the headless supervisor (→ stderr + non-zero exit) and the GUI.
     */
    [[nodiscard]] std::optional<step_fault> get_fault() const;

    /**
     * Number of run_quiesced() tasks queued and not yet taken by the sim thread
     * (a task being run has already left the queue). Read-only observation
     * point, so a test can establish that a requester is queued instead of
     * sleeping on the assumption that it is.
     */
    [[nodiscard]] std::size_t queued_quiesced_tasks() const {
        std::lock_guard lock(wake_mutex_);
        return quiesced_queue_.size();
    }

private:
    env& sim_env_;

    // Atomics for lock-free control
    std::atomic<sim_mode> mode_{sim_mode::paused};
    std::atomic<double> target_sps_{1000.0};
    std::atomic<std::uint64_t> step_count_{0};
    std::atomic<double> actual_sps_{0.0};
    std::atomic<bool> single_step_pending_{false};
    std::atomic<std::uint64_t> step_limit_{0};  // 0 = no limit

    // For waking the thread when paused, and for quiesced-task signaling
    mutable std::mutex wake_mutex_;   // mutable: queued_quiesced_tasks() reads under it
    std::condition_variable wake_cv_;

    // Quiesced-task seam (run_quiesced). The MPSC queue is guarded by
    // wake_mutex_ (so the park predicate and the queue share one lock); a parked
    // thread is woken via wake_cv_ when a task is enqueued. quiesced_pending_ is
    // a lock-free fast-path flag so the hot stepping loop checks "any task?"
    // without taking wake_mutex_ every iteration; it only locks to drain when
    // the flag is set. quiesce_stopping_ latches at teardown so late callers
    // reject instead of enqueuing into a thread that will no longer drain.
    // sim_thread_id_ is published once at thread_main entry for the
    // self-deadlock assert in run_quiesced.
    std::deque<std::function<void(bool)>> quiesced_queue_;
    bool                                  quiesce_stopping_ = false;
    std::atomic<bool>                     quiesced_pending_{false};
    std::atomic<std::thread::id>          sim_thread_id_;   // value-inits to "no thread"

    // The simulation thread. Default-constructed (non-joinable) here and started
    // explicitly as the last statement of the constructor body, then joined
    // explicitly in the destructor body (see sim_thread_controller.cpp). Because
    // both the start and the join are sequenced by explicit statements rather
    // than by member initialization/teardown, correctness no longer depends on
    // this being the last-declared member — reordering members cannot reintroduce
    // the startup/shutdown race of issue #52. Do not start the thread in a member
    // initializer.
    std::jthread thread_;

    /**
     * Main simulation thread function.
     * Runs until stop is requested via stop_token.
     */
    void thread_main(std::stop_token st);

    /**
     * Execute one simulation step (process + swap_io).
     */
    void do_step();

    /**
     * Record the first hot-path fault and halt the simulation (sticky). Sets
     * halted_ and forces paused mode; thread_main keeps re-forcing paused so an
     * external set_mode() cannot resume a faulted run.
     */
    void record_fault(const core::step_error& e, std::uint64_t step);

    // Hot-path fault state. do_step() writes it via record_fault(); the supervisor
    // and GUI read it via get_fault()/is_halted().
    std::atomic<bool>        halted_{false};
    mutable std::mutex       fault_mutex_;
    std::optional<step_fault> fault_;

    /**
     * Run all queued quiesced tasks (reject=false). Called on the sim thread at
     * the quiescent boundary. Swaps the queue out under wake_mutex_ then runs
     * tasks with the lock released, so a task may freely re-enter the controller
     * and so a slow task does not block enqueues.
     */
    void drain_quiesced();
};

} // namespace sim

#endif // COAST_SIM_THREAD_CONTROLLER_H
