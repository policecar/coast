#ifndef COAST_COASTD_EVENT_QUEUE_H
#define COAST_COASTD_EVENT_QUEUE_H

// -----------------------------------------------------------------------------
// coastd::event_queue — the single boundary into the core.
//
// Multi-producer (connection threads, workers, the liveness reaper),
// single-consumer (the core loop). A plain mutex + condition_variable + deque:
// coastd's event rate is a handful per second, so lock-free machinery would be
// unearned complexity (IDD non-goal: raw throughput).
//
// The consumer calls pop_until(deadline): it blocks until an event arrives OR
// the deadline passes, whichever first. That gives the core its two wakeup
// sources at once — an external event wakes it NOW, and the ~1 Hz scan deadline
// is the backstop that keeps reality-derived work (cool-off GC, a crashed child)
// moving even when no event arrives.
// -----------------------------------------------------------------------------

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <utility>

#include "daemon/events.h"

namespace coastd {

class event_queue {
public:
    // Enqueue an event and wake the consumer. Thread-safe; never blocks the
    // producer. A no-op after close() (a late worker result is simply dropped).
    void push(event e) {
        {
            std::lock_guard lk(m_);
            if (closed_) return;
            q_.push_back(std::move(e));
        }
        cv_.notify_one();
    }

    // Wait for an event until `deadline`. On success sets `out` and returns true.
    // Returns false if the deadline passed with no event, or the queue is closed
    // and drained — the caller distinguishes via is_closed() if it needs to.
    // The deadline is a STEADY_CLOCK point (not wall-clock): this is only a ~1 Hz
    // backstop, and a backward NTP/manual clock step must not stretch it (which
    // would stall cool-off GC and crashed-child relaunch dispatch). Wall-clock
    // reasoning stays in reality.now, used for mtime comparisons.
    bool pop_until(std::chrono::steady_clock::time_point deadline, event& out) {
        std::unique_lock lk(m_);
        cv_.wait_until(lk, deadline, [&] { return closed_ || !q_.empty(); });
        if (q_.empty()) return false;   // timed out, or closed and empty
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    // Wake every waiter, refuse further pushes, and DROP any pending events. The
    // drop matters for a client_command: it carries a shared promise the core
    // thread would have fulfilled, and a connection thread is blocked in
    // future::get() on it. If the core thread has already exited (its last-resort
    // backstop fired) it will never drain the queue, so leaving the command
    // enqueued would keep its promise alive and wedge that connection — and then
    // comms.stop()'s join — forever. Dropping the command destroys the promise;
    // the future then throws broken_promise, which handle_line turns into an error
    // reply and the connection unblocks. Other pending events are best-effort
    // (reality is re-derived at next boot), so dropping them is harmless.
    void close() {
        {
            std::lock_guard lk(m_);
            closed_ = true;
            q_.clear();
        }
        cv_.notify_all();
    }

    [[nodiscard]] bool is_closed() const {
        std::lock_guard lk(m_);
        return closed_;
    }

private:
    mutable std::mutex      m_;
    std::condition_variable cv_;
    std::deque<event>       q_;
    bool                    closed_ = false;
};

} // namespace coastd

#endif // COAST_COASTD_EVENT_QUEUE_H
