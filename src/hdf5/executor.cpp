//
// hdf5::executor — single-thread funnel for all HDF5 library calls.
//
#include "hdf5/executor.h"
#include "diag/diag.h"

#include <hdf5.h>

#include <exception>
#include <utility>

namespace hdf5 {

executor::executor(std::size_t queue_cap)
    : queue_cap_(queue_cap)
{
    // The worker is started in the constructor body, after all queue
    // state above is fully initialised, to avoid the worker observing
    // partial state.
    worker_ = std::thread(&executor::worker_main_, this);
}

executor::~executor()
{
    {
        std::lock_guard<std::mutex> lk(q_mtx_);
        stop_.store(true, std::memory_order_release);
    }
    q_not_empty_.notify_all();
    q_not_full_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void executor::submit(std::function<void()> job)
{
    if (!job) return;

    std::unique_lock<std::mutex> lk(q_mtx_);
    q_not_full_.wait(lk, [&] {
        return queue_.size() < queue_cap_ || stop_.load(std::memory_order_acquire);
    });
    // If we are shutting down, drop the job silently. We do NOT execute
    // post-stop work; that would extend shutdown indefinitely if a
    // misbehaving producer keeps submitting. Sync callers see this as
    // a never-completing future, which is correct: do not call submit_sync
    // during executor teardown.
    if (stop_.load(std::memory_order_acquire)) {
        return;
    }
    queue_.emplace_back(std::move(job));
    lk.unlock();
    q_not_empty_.notify_one();
}

// worker_main_ is a std::thread entry function: an exception leaving it calls
// std::terminate. The COAST_* macros are non-throwing by construction (they route
// through detail::emit_guarded, which swallows the std::format bad_alloc the
// pre-diag fprintf path could not raise), so the diagnostics below are safe here
// unwrapped.
void executor::worker_main_()
{
    // Force libhdf5 to complete its lazy global init exactly once, on
    // this thread, before any other code can submit work. Replaces the
    // ad-hoc H5open() workaround previously sitting in
    // observe_registry::attach().
    if (H5open() < 0) {
        COAST_ERROR("hdf5", "H5open() failed; HDF5 I/O will not work");
    }

    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lk(q_mtx_);
            q_not_empty_.wait(lk, [&] {
                return !queue_.empty() || stop_.load(std::memory_order_acquire);
            });
            if (queue_.empty() && stop_.load(std::memory_order_acquire)) {
                break;
            }
            job = std::move(queue_.front());
            queue_.pop_front();
        }
        q_not_full_.notify_one();

        try {
            job();
        } catch (const std::exception& e) {
            error_count_.fetch_add(1, std::memory_order_acq_rel);
            COAST_ERROR("hdf5", "async job threw std::exception: {}", e.what());
        } catch (...) {
            error_count_.fetch_add(1, std::memory_order_acq_rel);
            COAST_ERROR("hdf5", "async job threw unknown exception");
        }
    }

    // libhdf5 has no public "shutdown" call; H5close() is documented as
    // releasing all resources, but the standard practice is to leave the
    // library running until process exit. Letting the worker thread
    // simply return is correct and matches what every well-behaved HDF5
    // application does at teardown.
}

} // namespace hdf5
