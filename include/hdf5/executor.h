//
// hdf5::executor — single-thread funnel for all HDF5 library calls.
//
// The linked libhdf5 is built with Threadsafety: OFF, which means its
// contract is "exactly one thread inside the library at a time, process
// wide" — not per-file. Two threads operating on different files still
// race on libhdf5's global ID table, error stack, type registry and lazy
// init path.
//
// This class encodes that contract: it owns the one worker thread that
// is ever allowed to enter libhdf5. Every other piece of code submits
// work to that thread.
//
// Ownership: one instance, constructed in main() before any HDF5 user is
// constructed, passed by non-owning reference into the subsystems that
// need it (logging_manager, observe_registry, ...). Destruction drains
// the queue and joins the worker — see ~executor.
//
#ifndef COAST_HDF5_EXECUTOR_H
#define COAST_HDF5_EXECUTOR_H

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

namespace hdf5 {

class executor {
public:
    // Default queue cap matches the previous per-writer observe queue cap.
    // Tune later if benchmarks show producer-side blocking matters.
    static constexpr std::size_t default_queue_cap = 1024;

    explicit executor(std::size_t queue_cap = default_queue_cap);
    ~executor();

    executor(const executor&)            = delete;
    executor& operator=(const executor&) = delete;
    executor(executor&&)                 = delete;
    executor& operator=(executor&&)      = delete;

    // Fire-and-forget submission. Blocks the caller only when the queue
    // is full (back-pressure). Exceptions thrown by the job are logged
    // via COAST_ERROR (coast::diag) and counted via error_count(); the
    // worker thread is not torn down by them.
    void submit(std::function<void()> job);

    // Blocking submission. Runs the job on the worker thread and returns
    // its result (or rethrows its exception) to the caller. Used by
    // state_logger::save_state / load_state and by anything that needs
    // an hid_t handed back at construction time.
    //
    // Implementation note on the shared_ptr wrapping:
    //
    //   `submit()` takes a std::function<void()>, which requires its
    //   target to be copy-constructible. The lambda we hand it has to
    //   own two things that may be move-only:
    //
    //     1. The user's functor `fn`. Lambdas with reference captures
    //        are copyable, but any user functor that itself holds a
    //        std::unique_ptr, a std::promise, or a move-only RAII handle
    //        will not be — and we cannot constrain callers to "stay
    //        copyable", because that contradicts the whole point of
    //        modern C++ value semantics.
    //
    //     2. The std::promise<result_t> the worker fulfils. std::promise
    //        is always move-only, so a lambda that owns it directly is
    //        itself move-only.
    //
    //   The cleanest fix would be to replace std::function with a
    //   move-only function type. That is a non-trivial primitive (small-
    //   buffer optimisation, allocator awareness, exception safety) and
    //   one std::function<void()> across the codebase is more familiar.
    //   Instead we pay one heap allocation per submit_sync: both the
    //   user functor and the promise are wrapped in std::shared_ptr,
    //   which is copyable. The inner lambda therefore captures two
    //   shared_ptrs (copyable) and becomes std::function-compatible.
    //
    //   The executor invokes each job at most once, so the shared state
    //   is touched by exactly one consumer (the worker thread) and the
    //   producer side only holds the future, not the promise — no race.
    //   submit_sync is not on a hot path (opens, closes, creates,
    //   save_state's whole body), so the allocation cost does not
    //   matter. Streaming hot paths use submit() directly with their
    //   own pre-bundled byte buffers.
    template <class F>
    auto submit_sync(F&& fn) -> std::invoke_result_t<F&>
    {
        using result_t = std::invoke_result_t<F&>;
        auto prom = std::make_shared<std::promise<result_t>>();
        auto fut  = prom->get_future();

        auto wrapped_fn = std::make_shared<std::decay_t<F>>(std::forward<F>(fn));
        submit([wrapped_fn, prom] {
            try {
                if constexpr (std::is_void_v<result_t>) {
                    (*wrapped_fn)();
                    prom->set_value();
                } else {
                    prom->set_value((*wrapped_fn)());
                }
            } catch (...) {
                prom->set_exception(std::current_exception());
            }
        });

        return fut.get();
    }

    // Diagnostic counter of async-job exceptions. Sync jobs propagate to
    // the caller via the future and do not bump this counter.
    [[nodiscard]] std::uint64_t error_count() const noexcept {
        return error_count_.load(std::memory_order_acquire);
    }

private:
    void worker_main_();

    std::size_t queue_cap_;

    std::mutex                              q_mtx_;
    std::condition_variable                 q_not_empty_;
    std::condition_variable                 q_not_full_;
    std::deque<std::function<void()>>       queue_;
    std::atomic<bool>                       stop_{false};
    std::atomic<std::uint64_t>              error_count_{0};

    // The worker thread is started last so that all members above are
    // fully constructed before it begins observing them.
    std::thread worker_;
};

} // namespace hdf5

#endif // COAST_HDF5_EXECUTOR_H
