#include "sim_thread_controller.h"
#include "sim_env.h"

#include <chrono>

namespace sim {

sim_thread_controller::sim_thread_controller(env& sim_env)
    : sim_env_(sim_env)
    , thread_([this](std::stop_token st) { thread_main(st); })
{
}

sim_thread_controller::~sim_thread_controller()
{
    // Request stop (jthread does this automatically, but be explicit)
    thread_.request_stop();

    // Wake the thread if it's waiting on the condition variable
    {
        std::lock_guard lock(wake_mutex_);
    }
    wake_cv_.notify_all();

    // jthread destructor will join automatically
}

void sim_thread_controller::set_mode(sim_mode mode)
{
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

void sim_thread_controller::do_step()
{
    sim_env_.process();
    sim_env_.swap_io();
    step_count_.fetch_add(1, std::memory_order_relaxed);
}

void sim_thread_controller::thread_main(std::stop_token st)
{
    using clock = std::chrono::steady_clock;
    using namespace std::chrono_literals;

    // For SPS measurement
    auto sps_measure_start = clock::now();
    std::uint64_t sps_measure_steps = 0;
    constexpr auto sps_measure_interval = 500ms;

    // For throttled mode timing
    auto last_step_time = clock::now();

    while (!st.stop_requested()) {
        const sim_mode current_mode = mode_.load(std::memory_order_acquire);

        switch (current_mode) {
        case sim_mode::paused: {
            // Check for single step request
            if (single_step_pending_.exchange(false, std::memory_order_acquire)) {
                do_step();
                ++sps_measure_steps;
            } else {
                // Wait for mode change or single_step
                std::unique_lock lock(wake_mutex_);
                wake_cv_.wait_for(lock, 100ms, [&] {
                    return st.stop_requested() ||
                           mode_.load(std::memory_order_acquire) != sim_mode::paused ||
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
