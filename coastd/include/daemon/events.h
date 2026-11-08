#ifndef COAST_COASTD_EVENTS_H
#define COAST_COASTD_EVENTS_H

// -----------------------------------------------------------------------------
// coastd events — the ONLY things that cross into the single-writer core.
//
// Every edge (a client connection thread, a worker, the child-liveness reaper)
// touches state solely by pushing an `event`; `event_queue::push` is the sole
// API boundary into the core. The core pops one event, folds it into the store
// via apply(), then rescans reality and reconciles.
//
// Three event kinds:
//   * client_command — a request from a client (submit/cancel/list/…). Modeled
//     as a closure over the core so new commands need no new event type: the
//     connection thread captures its own reply promise, the core just runs it on
//     the core thread (keeping the core the single writer while the client gets a
//     synchronous-feeling RPC).
//   * worker_result — a finite async task (resolve/build/gc/commit) finished.
//   * child_exited — a supervised `coast` child terminated (waitpid edge). Carries
//     the durable outcome the liveness edge read from the run dir before the
//     reaped fact was gone (a SIGKILLed child leaves no run manifest).
// -----------------------------------------------------------------------------

#include <cstdint>
#include <functional>
#include <string>
#include <variant>

#include "daemon/types.h"

namespace coastd {

class core;  // the closure in client_command mutates the core on the core thread

// Which finite async task a worker_result reports on.
enum class task_kind : std::uint8_t { resolve, build, gc, commit };

// A client request, deferred onto the core thread. `fn` mutates the store and
// fulfills its own captured reply promise. `label` is for diagnostics only.
struct client_command {
    std::function<void(core&)> fn;
    std::string                label;
};

// A finite async task completed. `ok` == false carries a human message in
// `message`; a resolve success carries the resolved SHA in `sha`.
struct worker_result {
    task_kind   kind;
    job_id      id;        // resolve target; empty for sha-keyed tasks
    sha_t       sha;       // resolve output, or the build/gc subject SHA
    bool        ok = true;
    std::string message;   // error text when !ok
};

// A supervised child terminated. `wstatus` is the raw waitpid status; the
// supervisor decodes clean-vs-interrupted. `steps_done`/`checkpoint` are read
// from the run dir at exit (authoritative even when no manifest was written).
struct child_exited {
    job_id        id;
    int           wstatus     = 0;
    std::uint64_t steps_done  = 0;   // DURABLE, resumable progress (newest checkpoint/manifest)
    std::string   checkpoint;   // newest checkpoint basename rel run_dir, or ""
    bool          fatal = false;  // coast wrote exit_reason=fatal_error -> give up, don't resume
    std::uint64_t tailed_step = 0;  // last LIVE step seen on the log tail (may exceed the last
                                    // checkpoint); lets the crash-loop guard tell "did real work
                                    // but wasn't killed before a checkpoint" from "did nothing"
                                    // WITHOUT coupling the guard to checkpoint_interval
};

using event = std::variant<client_command, worker_result, child_exited>;

} // namespace coastd

#endif // COAST_COASTD_EVENTS_H
