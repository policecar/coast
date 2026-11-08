#ifndef COAST_COASTD_WORKER_POOL_H
#define COAST_COASTD_WORKER_POOL_H

// -----------------------------------------------------------------------------
// coastd::worker_pool — bounded async workers for finite, independent,
// cancellable jobs (resolve a ref, build a worktree, gc a worktree, git commit).
//
// A task is a callable returning a worker_result; the pool runs it on a worker
// thread and pushes that result onto the core's event_queue. Tasks are the edge:
// they touch the world (git, cmake, the filesystem) but reach the core only via
// the queue. A task MUST return a worker_result even on failure (never throw) so
// the core always learns the task finished and can clear its in-flight marker.
//
// Every task is submitted WITH its identifying key, because the backstop needs it.
// The core erases resolve_in_flight_/build_in_flight_/gc_in_flight_ only when the
// matching worker_result arrives, so a task that threw used to leak its key
// forever: with build_concurrency == 1, one lost build result stopped ALL builds
// for the daemon's lifetime, evidenced by a single log line. A std::function
// carrying no key left the backstop nothing to report. With the key the pool
// synthesizes an ok=false result, the core clears the marker, and the work is
// simply retried.
//
// "Exactly one build in flight" (IDD §6E) is NOT enforced here — the pool is
// generic; the dispatcher gates build submission. The pool just runs what it is
// given, up to `n` concurrently.
// -----------------------------------------------------------------------------

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "daemon/events.h"

namespace coastd {

class event_queue;

class worker_pool {
public:
    using task = std::function<worker_result()>;

    // What the task is working on — enough to synthesize its failure result.
    // Mirrors the identifying fields of worker_result.
    struct task_key {
        task_kind kind{};
        job_id    id;    // resolve target; empty for sha-keyed tasks
        sha_t     sha;   // the build/gc subject SHA
    };

    // Spawn `n` worker threads (n >= 1) that post results onto `out`.
    worker_pool(int n, event_queue& out);
    ~worker_pool();

    worker_pool(const worker_pool&)            = delete;
    worker_pool& operator=(const worker_pool&) = delete;

    // Enqueue a task under `key`. Thread-safe; returns immediately.
    void submit(task_key key, task t);

    // Stop accepting tasks, drain nothing new, and join all workers. Idempotent;
    // also called by the destructor. In-flight tasks finish; queued-but-unstarted
    // tasks are dropped.
    void shutdown();

private:
    void worker_loop();

    event_queue&             out_;
    std::mutex               m_;
    std::condition_variable  cv_;
    std::deque<std::pair<task_key, task>> tasks_;
    bool                     stop_ = false;
    std::vector<std::thread> threads_;
};

} // namespace coastd

#endif // COAST_COASTD_WORKER_POOL_H
