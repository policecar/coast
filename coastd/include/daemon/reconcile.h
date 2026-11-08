#ifndef COAST_COASTD_RECONCILE_H
#define COAST_COASTD_RECONCILE_H

// -----------------------------------------------------------------------------
// coastd::reconcile — the pure scheduling + recovery state machine.
// -----------------------------------------------------------------------------
//
//   reconcile(active_jobs, reality, config) -> actions
//
// PURE: no I/O, no globals, and no clock beyond `reality.now`. The entire
// scheduling AND recovery policy is therefore a deterministic function of its
// explicit inputs, unit-testable with zero subprocesses. Because recovery runs
// the SAME function (boot -> load store -> probe reality -> reconcile), there is
// no separate, rarely-exercised recovery path to rot.
//
// Actions are UNORDERED and reconverging: each depends only on `reality`, never
// on another action in the pass. reconcile may re-emit an action whose effect is
// already in flight (e.g. a build already running); the dispatcher is idempotent
// per (kind, key), so over-emission is harmless and a partial dispatch after a
// crash simply gets re-derived next pass.
//
// The rule table, applied per active job (first match wins):
//
//   failure_reason set + child alive      -> kill_child{SIGKILL}      (defensive)
//   failure_reason set + child dead       -> move_bucket{failed}
//   cancel_requested + child alive        -> kill_child{SIGTERM}
//   cancel_requested + child dead         -> move_bucket{cancelled}
//   (failure BEFORE cancel is deliberate: a recorded failure is the truer
//    terminal cause, and must not be relabelled `cancelled/` where nobody
//    would look for its failure_reason.)
//   ref_sha empty                         -> resolve_ref
//   worktree for ref_sha absent           -> build_worktree
//   worktree present + child alive        -> (running; no action)
//   worktree present + done (steps>=target)-> move_bucket{done}
//   worktree present + child dead + work  -> spawn candidate (see below)
//
// Then two cross-job policies over the whole set:
//   * SLOT BUDGET: spawn candidates are ordered by (priority desc, submitted_at
//     asc = FIFO) and spawn_child is emitted for the top (slots - running_count).
//     A relaunch resumes from the newest checkpoint; the supervisor derives the
//     command line — spawn_child carries only the job id.
//   * WORKTREE GC: a SHA with refcount 0 (no active job references it) whose
//     worktree has been idle (mtime) longer than cfg.worktree_cooloff -> gc_worktree.
//     refcount is DERIVED here from the active jobs, so it can never desync.
//
// `jobs` is the ACTIVE bucket only. Terminal jobs do not schedule and do not
// hold a worktree refcount, so they are irrelevant to reconcile.
// -----------------------------------------------------------------------------

#include <span>
#include <vector>

#include "daemon/types.h"

namespace coastd {

[[nodiscard]] std::vector<action> reconcile(std::span<const job>  jobs,
                                            const reality&        r,
                                            const daemon_config&  cfg);

// Backoff window (ms) before an operation with `failures` prior no-progress
// attempts may retry: relaunch_backoff_base * 2^(failures-1), clamped to
// relaunch_backoff_max. Shared by reconcile's relaunch spacing and the core's
// per-SHA build-retry spacing. `failures <= 0` -> 0. The shift is internally
// capped so it can never overflow regardless of the caller's counter.
[[nodiscard]] std::uint64_t backoff_ms(int failures, const daemon_config& cfg);

} // namespace coastd

#endif // COAST_COASTD_RECONCILE_H
