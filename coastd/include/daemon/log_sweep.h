#ifndef COAST_COASTD_LOG_SWEEP_H
#define COAST_COASTD_LOG_SWEEP_H

// -----------------------------------------------------------------------------
// coastd::sweep_logs — age-based log housekeeping, in the standard Linux shape:
// compress what is old, delete what is ancient.
//
// coastd writes two kinds of log that nothing else ever bounds:
//   <run_dir>/logs/run_<...>.log   one per LAUNCH; an --unlimited job's grows for
//                                  as long as the job runs (weeks)
//   <worktree>/coastd-build.log    per-SHA, appended across every build attempt
//
// Age is judged by MTIME, and that is also what keeps the sweep off a live file:
// a log currently being written has, by definition, a fresh mtime, so a threshold
// of hours-to-days can never select the file the supervisor is tailing. No
// coupling to job state, and no rotating a file out from under an open fd — which
// is the classic way log rotation breaks a tailer.
//
// Compression shells out to gzip(1). That is deliberate: it is the standard tool,
// present on every target system, it produces the format an operator expects to
// find, and it costs no new library dependency. gzip removes the original itself
// on success, so the operation is atomic from this code's point of view — either
// <name>.log or <name>.log.gz exists, never a half-written pair.
//
// Every step is best-effort and independently failable: a log that cannot be
// compressed is logged once and skipped, never retried into a loop, and never
// escalated to anything that could fail a job. Losing a diagnostic must not cost
// a run.
// -----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>

#include "daemon/types.h"

namespace coastd {

// What one sweep did. Returned for logging and for tests to assert on.
struct sweep_stats {
    int compressed = 0;   // .log -> .log.gz
    int deleted    = 0;   // .log.gz removed past log_delete_after
    int failed     = 0;   // steps that errored (already logged individually)
};

// Walk runs_root and worktrees_root once, applying cfg's age policy. Safe to call
// from a worker thread; touches no daemon state and takes no locks. Never throws.
sweep_stats sweep_logs(const daemon_config& cfg) noexcept;

// Append `body` under a `header` banner to an append-only log, enforcing a hard
// size ceiling: past `max_bytes` the file is restarted with a marker line rather
// than grown. Age-based policy alone cannot bound this file — a crash-looping SHA
// reaches gigabytes within a single day, far inside any sane compress-after
// window. The log is a rebuildable diagnostic, so discarding the old bytes is the
// right trade against filling the disk the jobs need.
void append_capped(const std::filesystem::path& log, std::string_view header,
                   const std::string& body, std::size_t max_bytes) noexcept;

} // namespace coastd

#endif // COAST_COASTD_LOG_SWEEP_H
