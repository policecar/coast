#include "sim_core/sim_thread_controller.h"
#include "sim_core/sim_env.h"
#include "probe/probe_instance.h"
#include "diag/diag.h"

#include <chrono>

namespace sim {

sim_thread_controller::sim_thread_controller(env& sim_env)
    : sim_env_(sim_env)
{
    // Two-phase construction: thread_ is default-constructed (non-joinable) as
    // a member, then started here as the LAST statement of the constructor
    // body. By the time the body runs, every atomic, wake_mutex_ and wake_cv_
    // is fully constructed, so thread_main can safely touch them the instant it
    // launches. Sequencing the start with an explicit statement (rather than a
    // member initializer) means this guarantee no longer depends on thread_
    // being the last-declared member (issue #52).
    thread_ = std::jthread([this](std::stop_token st) { thread_main(st); });
}

sim_thread_controller::~sim_thread_controller()
{
    // Request stop FIRST, so any run_quiesced that races teardown observes
    // stop_requested() under wake_mutex_ and rejects itself rather than
    // enqueuing into a thread that will no longer drain.
    thread_.request_stop();

    // Latch the stopping state and orphan any already-queued quiesced tasks.
    std::deque<std::function<void(bool)>> orphaned;
    {
        std::lock_guard lock(wake_mutex_);
        quiesce_stopping_ = true;
        orphaned.swap(quiesced_queue_);
        quiesced_pending_.store(false, std::memory_order_release);
    }
    wake_cv_.notify_all();   // wake the parked sim thread so it observes stop

    // Reject orphaned tasks BEFORE joining. A requester may be blocked in
    // run_quiesced behind a task that itself holds the sim thread, so the join
    // below could only complete once that task finishes — rejecting first
    // unblocks such requesters without deadlocking teardown. reject=true fulfils
    // each promise with the stopping exception without running the task body.
    for (auto& job : orphaned) job(true);

    // Explicit join (issue #52 pattern): join HERE, in the destructor body,
    // while wake_mutex_/wake_cv_ are still alive — so shutdown is correct
    // regardless of member declaration order, rather than relying on jthread's
    // implicit join during member teardown.
    if (thread_.joinable()) {
        thread_.join();
    }
}

void sim_thread_controller::set_mode(sim_mode mode)
{
    // A single-step request belongs to the paused state it was made in. Drop it
    // on any mode change so it cannot fire on a later, unrelated pause (#201).
    single_step_pending_.store(false, std::memory_order_release);
    mode_.store(mode, std::memory_order_release);

    // Wake thread if transitioning from paused
    if (mode != sim_mode::paused) {
        std::lock_guard lock(wake_mutex_);
        wake_cv_.notify_one();
    }
}

void sim_thread_controller::set_steps_per_second(double rate)
{
    if (rate > 0.0) {
        target_sps_.store(rate, std::memory_order_release);
    }
}

void sim_thread_controller::single_step()
{
    single_step_pending_.store(true, std::memory_order_release);

    // Wake thread to process the single step
    std::lock_guard lock(wake_mutex_);
    wake_cv_.notify_one();
}

sim_mode sim_thread_controller::get_mode() const
{
    return mode_.load(std::memory_order_acquire);
}

std::uint64_t sim_thread_controller::get_step_count() const
{
    return step_count_.load(std::memory_order_acquire);
}

double sim_thread_controller::get_actual_sps() const
{
    return actual_sps_.load(std::memory_order_acquire);
}

double sim_thread_controller::get_target_sps() const
{
    return target_sps_.load(std::memory_order_acquire);
}

void sim_thread_controller::set_step_count(std::uint64_t count)
{
    step_count_.store(count, std::memory_order_release);
}

void sim_thread_controller::set_step_limit(std::uint64_t limit)
{
    step_limit_.store(limit, std::memory_order_release);
}

void sim_thread_controller::do_step()
{
    // A halted run advances no further (the fault is sticky); guards against a
    // single_step request re-running the faulting step.
    if (halted_.load(std::memory_order_acquire))
        return;

    // The step limit is checked where the step happens. The top-of-loop limit
    // check and the dispatch below are separate atomic reads, so a single_step()
    // or set_mode() landing between them would otherwise step past it (#201).
    if (const auto limit = step_limit_.load(std::memory_order_acquire);
        limit > 0 && step_count_.load(std::memory_order_relaxed) >= limit)
        return;

    // Set the current step for probe recording before processing
    std::uint64_t current_step = step_count_.load(std::memory_order_relaxed);
    probe::set_current_step(current_step);

    // Hot-path faults come back by value, not as exceptions: a violated invariant
    // (or a GPU device fault in swap_io) halts the run cleanly instead of
    // unwinding out of this sim thread. An entity's throw is already converted to
    // such a fault in env::dispatch_cpu_entities(); this catch is the backstop for
    // the rest of the step (hooks, swap_io, the GPU host loop), which would
    // otherwise reach std::terminate (#198).
    try {
        if (auto r = sim_env_.process(); !r) {
            record_fault(r.error(), current_step);
            return;
        }
        if (auto r = sim_env_.swap_io(); !r) {
            record_fault(r.error(), current_step);
            return;
        }
    } catch (...) {
        COAST_ERROR("sim", "step {} threw: {}", current_step, core::current_exception_what());
        record_fault(core::step_error{core::error_kind::exception,
                                      core::step_error::no_entity, COAST_SITE},
                     current_step);
        return;
    }
    step_count_.fetch_add(1, std::memory_order_relaxed);
}

void sim_thread_controller::record_fault(const core::step_error& e,
                                         std::uint64_t step)
{
    {
        std::lock_guard lock(fault_mutex_);
        if (!fault_)                       // sticky: the first fault wins
            fault_ = step_fault{e, step};
    }
    halted_.store(true, std::memory_order_release);
    mode_.store(sim_mode::paused, std::memory_order_release);
}

void sim_thread_controller::halt(const core::step_error& e)
{
    record_fault(e, step_count_.load(std::memory_order_acquire));
}

std::optional<sim_thread_controller::step_fault>
sim_thread_controller::get_fault() const
{
    std::lock_guard lock(fault_mutex_);
    return fault_;
}

void sim_thread_controller::drain_quiesced()
{
    // Swap the queue out under the lock, then run with the lock released: a task
    // may re-enter the controller, and a slow task must not block producers.
    std::deque<std::function<void(bool)>> jobs;
    {
        std::lock_guard lock(wake_mutex_);
        jobs.swap(quiesced_queue_);
        quiesced_pending_.store(false, std::memory_order_release);
    }
    for (auto& job : jobs) job(false);   // run on the sim thread + fulfil promise
}

void sim_thread_controller::thread_main(std::stop_token st)
{
    using clock = std::chrono::steady_clock;
    using namespace std::chrono_literals;

    // Publish our identity for run_quiesced's self-deadlock assert.
    sim_thread_id_.store(std::this_thread::get_id(), std::memory_order_release);

    // For SPS measurement
    auto sps_measure_start = clock::now();
    std::uint64_t sps_measure_steps = 0;
    constexpr auto sps_measure_interval = 500ms;

    // For throttled mode timing
    auto last_step_time = clock::now();

    while (!st.stop_requested()) {
        // Run any queued quiesced tasks at the coherent post-swap boundary: the
        // previous iteration's do_step (process + swap_io) has fully completed
        // and no new one has begun, so the read pages and the wgpu queue are
        // quiescent. Gated by a lock-free flag so the hot stepping loop never
        // takes wake_mutex_ when idle. In throttled mode a task enqueued mid-pace
        // waits at most one step period (until this next boundary); saves are not
        // latency-critical.
        if (quiesced_pending_.load(std::memory_order_acquire)) {
            drain_quiesced();
        }

        // Check step limit and auto-pause if reached
        const std::uint64_t limit = step_limit_.load(std::memory_order_acquire);
        if (limit > 0 && step_count_.load(std::memory_order_acquire) >= limit) {
            // Drop any pending single step too: it is void at the limit (#201).
            // do_step() enforces the limit; this keeps the paused branch from
            // counting a no-op step in the SPS measurement.
            single_step_pending_.store(false, std::memory_order_release);
            mode_.store(sim_mode::paused, std::memory_order_release);
        }

        // A faulted run stays paused regardless of any external set_mode().
        if (halted_.load(std::memory_order_acquire)) {
            mode_.store(sim_mode::paused, std::memory_order_release);
        }

        const sim_mode current_mode = mode_.load(std::memory_order_acquire);

        switch (current_mode) {
        case sim_mode::paused: {
            // Check for single step request
            if (single_step_pending_.exchange(false, std::memory_order_acquire)) {
                do_step();
                ++sps_measure_steps;
            } else {
                // Park and wait for a mode change, single_step, a queued quiesced
                // task, or stop. A pending quiesced task wakes the thread so it
                // drains at the top-of-loop boundary, then re-parks (the task
                // advances no sim time).
                std::unique_lock lock(wake_mutex_);
                wake_cv_.wait_for(lock, 100ms, [&] {
                    if (st.stop_requested()) return true;
                    if (quiesced_pending_.load(std::memory_order_acquire))
                        return true;
                    return mode_.load(std::memory_order_acquire) != sim_mode::paused ||
                           single_step_pending_.load(std::memory_order_acquire);
                });
            }
            break;
        }

        case sim_mode::throttled: {
            const double target_rate = target_sps_.load(std::memory_order_acquire);
            const auto step_duration = std::chrono::duration<double>(1.0 / target_rate);

            do_step();
            ++sps_measure_steps;

            // Calculate time to sleep
            const auto now = clock::now();
            const auto elapsed = now - last_step_time;
            last_step_time = now;

            if (elapsed < step_duration) {
                const auto sleep_time = step_duration - elapsed;
                std::this_thread::sleep_for(sleep_time);
                last_step_time = clock::now();
            }
            break;
        }

        case sim_mode::unlimited: {
            // Run as fast as possible
            do_step();
            ++sps_measure_steps;

            // Yield occasionally to prevent complete CPU starvation
            // (every 1000 steps, yield to allow other threads)
            if ((step_count_.load(std::memory_order_relaxed) % 1000) == 0) {
                std::this_thread::yield();
            }
            break;
        }
        }

        // Update SPS measurement periodically
        const auto now = clock::now();
        if (now - sps_measure_start >= sps_measure_interval) {
            const auto interval_seconds =
                std::chrono::duration<double>(now - sps_measure_start).count();
            const double measured_sps =
                static_cast<double>(sps_measure_steps) / interval_seconds;

            // Exponential moving average for smoothing
            const double current = actual_sps_.load(std::memory_order_relaxed);
            const double smoothed = (current == 0.0) ? measured_sps
                                                      : (0.7 * measured_sps + 0.3 * current);
            actual_sps_.store(smoothed, std::memory_order_release);

            sps_measure_start = now;
            sps_measure_steps = 0;
        }
    }
}

} // namespace sim
