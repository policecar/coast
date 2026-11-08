#ifndef COAST_COASTD_BUILDER_H
#define COAST_COASTD_BUILDER_H

// -----------------------------------------------------------------------------
// coastd::builder — turns a git ref into a runnable `coast` binary.
//
// It produces WORKER TASKS (callables returning worker_result); the core submits
// them to the worker_pool and learns the outcome via the event queue. Each task
// catches its own failures and returns a worker_result — it never throws.
//
//   resolve  : refresh remote-tracking refs (`git fetch`, throttled + best-effort)
//              then `git rev-parse` the symbolic ref to a full SHA (frozen at
//              submit). See daemon_config's fetch_* block for why the fetch and
//              its remote-first preference exist.
//   build    : `git worktree add <sha>` + `git submodule update --init
//              --recursive` (Demonstrate spike 02: worktrees don't populate
//              submodules → configure fails without it) + `cmake` configure +
//              build of the `coast` target. Incremental: an existing worktree is
//              reused and cmake decides staleness.
//   gc       : remove an idle worktree (`git worktree remove --force`, then a
//              filesystem sweep as a fallback).
//
// A worktree is keyed by its resolved SHA under worktrees_root/<sha>; the coast
// binary lands at worktrees_root/<sha>/build/coast.
// -----------------------------------------------------------------------------

#include <mutex>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "daemon/types.h"
#include "daemon/worker_pool.h"

namespace coastd {

// At most one `git fetch` per fetch_min_interval across the concurrent resolve tasks
// of a burst of submissions. The mutex is held for the whole fetch, so a resolve that
// arrives meanwhile WAITS for it instead of looking up its ref before the fetch has
// brought it in (builder.cpp, maybe_fetch).
struct fetch_gate {
    std::mutex   m;
    std::int64_t last_ms = 0;   // epoch-ms of the last fetch ATTEMPT; 0 = never
};

class builder {
public:
    explicit builder(daemon_config cfg);

    // Worker tasks. They capture what they need by value, so they are safe to run
    // after the builder is gone.
    [[nodiscard]] worker_pool::task resolve_task(job_id id, std::string ref_symbolic) const;
    [[nodiscard]] worker_pool::task build_task(sha_t sha) const;
    [[nodiscard]] worker_pool::task gc_task(sha_t sha) const;

    [[nodiscard]] std::filesystem::path worktree_dir(const sha_t& sha) const;
    [[nodiscard]] std::filesystem::path coast_binary(const sha_t& sha) const;

    // Build-completion sentinel: <worktree>/.coastd-ready, written atomically as
    // build_task's LAST step. Its presence — NOT the worktree directory's, which
    // `git worktree add` creates seconds into a minutes-long build — is what marks
    // a SHA runnable. rescan() reports readiness from this file, so a job is never
    // spawned against a half-built (or restart-interrupted) worktree.
    [[nodiscard]] std::filesystem::path ready_marker(const sha_t& sha) const;

private:
    daemon_config cfg_;

    // The fetch throttle, shared with every resolve task the builder hands out. A
    // shared_ptr (not a plain member) because a task captures it BY VALUE, keeping
    // the builder's "safe to run after the builder is gone" contract intact.
    std::shared_ptr<fetch_gate> fetch_gate_;
};

} // namespace coastd

#endif // COAST_COASTD_BUILDER_H
