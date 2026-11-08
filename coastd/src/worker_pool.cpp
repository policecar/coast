#include "daemon/worker_pool.h"

#include <exception>
#include <string>
#include <utility>

#include "daemon/event_queue.h"
#include "diag/diag.h"

namespace coastd {

worker_pool::worker_pool(int n, event_queue& out) : out_(out) {
    if (n < 1) n = 1;
    threads_.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        threads_.emplace_back([this] { worker_loop(); });
}

worker_pool::~worker_pool() { shutdown(); }

void worker_pool::submit(task_key key, task t) {
    {
        std::lock_guard lk(m_);
        if (stop_) return;  // shutting down: refuse new work
        tasks_.emplace_back(std::move(key), std::move(t));
    }
    cv_.notify_one();
}

void worker_pool::shutdown() {
    {
        std::lock_guard lk(m_);
        if (stop_) return;
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& th : threads_)
        if (th.joinable()) th.join();
    threads_.clear();
}

void worker_pool::worker_loop() {
    for (;;) {
        task_key key;
        task     t;
        {
            std::unique_lock lk(m_);
            cv_.wait(lk, [&] { return stop_ || !tasks_.empty(); });
            // On shutdown, drop queued-but-unstarted tasks (per the header
            // contract): a task already running is past this lock and finishes,
            // but pulling the backlog here would run it AFTER core::run() has
            // returned and closed the event_queue — its result is dropped anyway,
            // so running it only adds shutdown latency (a queued build is minutes)
            // against systemd's stop timeout.
            if (stop_) return;
            key = std::move(tasks_.front().first);
            t   = std::move(tasks_.front().second);
            tasks_.pop_front();
        }
        // A task must return a worker_result even on failure; this catch is the
        // backstop so a stray throw cannot take down a worker thread — and, just as
        // importantly, cannot LOSE the result. The core clears its in-flight marker
        // for a (kind, key) only when the matching worker_result arrives, so a
        // dropped result leaks that marker permanently: with build_concurrency == 1
        // that means no build ever starts again, for the life of the daemon, behind
        // one log line. Synthesize the failure from the key instead. The core then
        // takes an ordinary build/resolve failure — retried with backoff — which is
        // the truthful description of what happened.
        auto fail_result = [&key](const std::string& why) {
            worker_result r;
            r.kind    = key.kind;
            r.id      = key.id;
            r.sha     = key.sha;
            r.ok      = false;
            r.message = why;
            return r;
        };
        try {
            out_.push(t());
        } catch (const std::exception& e) {
            COAST_ERROR("coastd.worker", "task threw: {} — reporting it as a failed task "
                        "so the core clears its in-flight marker", e.what());
            try {
                out_.push(fail_result(std::string("task threw: ") + e.what()));
            } catch (...) {
                COAST_ERROR("coastd.worker", "could not even report the failure; this task's "
                            "in-flight marker will not clear until the daemon restarts");
            }
        } catch (...) {
            COAST_ERROR("coastd.worker", "task threw a non-exception — reporting it as a "
                        "failed task so the core clears its in-flight marker");
            try {
                out_.push(fail_result("task threw a non-exception"));
            } catch (...) {
                COAST_ERROR("coastd.worker", "could not even report the failure; this task's "
                            "in-flight marker will not clear until the daemon restarts");
            }
        }
    }
}

} // namespace coastd
