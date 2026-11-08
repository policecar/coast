#include "daemon/reconcile.h"

#include <algorithm>
#include <csignal>
#include <unordered_map>

namespace coastd {

namespace {

// child_alive lookup with a default of "dead": a job the reality scan never saw
// a live child for is treated as not running (the safe default under R1 — a job
// we cannot prove is alive is relaunched, never assumed running).
bool alive(const reality& r, const job_id& id) {
    auto it = r.child_alive.find(id);
    return it != r.child_alive.end() && it->second;
}

std::uint64_t ms_since_epoch(sys_time t) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch()).count());
}

} // namespace

// Backoff window before an operation with `failures` no-progress attempts may
// retry: base * 2^(failures-1), clamped to max. The shift is internally capped at
// 30 so it can never overflow regardless of the caller's counter (relaunch's
// counter is bounded by max_launch_failures, the build counter by
// max_build_failures; both terminal at their cap). Namespace-scope (declared in
// reconcile.h) so the core's build-retry spacing reuses the exact same policy.
std::uint64_t backoff_ms(int failures, const daemon_config& cfg) {
    if (failures <= 0) return 0;
    const auto base = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(cfg.relaunch_backoff_base).count());
    const auto cap = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(cfg.relaunch_backoff_max).count());
    const int shift = failures - 1 > 30 ? 30 : failures - 1;   // guard the shift
    const std::uint64_t w = base << shift;
    return w > cap ? cap : w;
}

std::vector<action> reconcile(std::span<const job>  jobs,
                              const reality&        r,
                              const daemon_config&  cfg) {
    std::vector<action> out;

    // Refcount is DERIVED from the active jobs here (never persisted), so it can
    // never desync across a crash. A non-terminal job "holds" its frozen ref_sha.
    std::unordered_map<sha_t, int> refcount;

    // Ready-but-idle jobs competing for a free slot; scheduled by priority below.
    std::vector<const job*> spawn_candidates;

    // ---- Per-job lifecycle decisions (first match wins) --------------------
    for (const job& j : jobs) {
        if (!j.ref_sha.empty()) ++refcount[j.ref_sha];

        const bool running = alive(r, j.id);

        // Failure is checked BEFORE cancel: if a terminal failure was already
        // recorded (build fail, crash-loop, poison), that is the true terminal
        // cause and must not be relabelled `cancelled/` by a subsequent cancel —
        // the failure_reason would then be filed where no operator looks for it.
        if (!j.failure_reason.empty()) {
            // Ensure no child survives, then bucket it. The kill is defensive: a
            // failed job should not have a live child, but reconcile never assumes.
            if (running) out.push_back(kill_child{j.id, SIGKILL});
            else         out.push_back(move_bucket{j.id, bucket::failed});
            continue;
        }
        if (j.cancel_requested) {
            if (running) out.push_back(kill_child{j.id, SIGTERM});
            else         out.push_back(move_bucket{j.id, bucket::cancelled});
            continue;
        }
        if (j.ref_sha.empty()) {
            out.push_back(resolve_ref{j.id, j.ref_symbolic});
            continue;
        }
        // COMPLETION is terminal and must be decided BEFORE build readiness, exactly
        // as failure and cancel are. It used to sit below the build gate, which made
        // filing a finished job conditional on having a worktree to run it in — and
        // those are unrelated facts. A job whose steps are done but whose worktree is
        // gone (GC'd after a crash, or worktrees_root on tmpfs across a reboot, with
        // steps_done folded back from the manifest at boot) would emit build_worktree:
        // a ~238 MB tree and a full cmake build, minutes of CPU, purely so a job that
        // finished can be moved to done/. Worse, if that build then failed
        // max_build_failures times, the core stamps failure_reason on every job on the
        // SHA — filing a SUCCESSFULLY COMPLETED job to failed/. Under disk pressure it
        // produced no action at all and the job sat in active/ indefinitely.
        //
        // steps_target == 0 is unbounded and never satisfies this. `running` is
        // required because a live child still owns the run dir; the terminal move
        // happens on the pass after it exits.
        if (!running && j.steps_target > 0 && j.steps_done >= j.steps_target) {
            out.push_back(move_bucket{j.id, bucket::done});
            continue;
        }
        if (!r.builds_ready.contains(j.ref_sha)) {
            // Not runnable yet: no worktree, a build in progress (directory exists
            // but the .coastd-ready sentinel does not), or a build interrupted by
            // a daemon restart. All three want the same action — (re)build; the
            // dispatcher dedups an already-in-flight build. This is what closes the
            // "spawn against a half-built binary" race AND makes restart-mid-build
            // resume the build instead of hard-failing the job.
            //
            // Under disk pressure a build is exactly the wrong thing to start: it
            // is the single largest allocation the daemon makes (~238 MB of tree +
            // objects) and it can fail HALFWAY, leaving a worktree that is neither
            // usable nor obviously garbage. Hold the job here instead — no strike,
            // no failure_reason, it simply waits for space like it waits for a slot.
            if (r.disk_pressure) continue;
            // Same hold for an unreadable worktrees root. `builds_ready` is empty in
            // that case not because nothing is built but because nothing could be
            // SEEN, and acting on it would rebuild trees that may already exist —
            // repeatedly, since the condition persists — until max_build_failures
            // stamps a terminal failure_reason on every job on this SHA, running
            // ones included. Waiting costs a pass; the alternative loses the queue.
            if (r.worktrees_unreadable) continue;
            out.push_back(build_worktree{j.ref_sha});
            continue;
        }
        // Build ready (sentinel present).
        if (running) continue;                       // already stepping; nothing to do
        // (completion was decided above, before the build gate)
        // Relaunch backoff: a job with prior no-progress launches waits an
        // increasing spacing (see daemon_config) before its next attempt, so a
        // transient fault does not burn all its strikes in seconds. Skipping it for
        // this pass frees the slot for a lower-priority ready job rather than idling.
        //
        // The spacing is owed on EITHER strike counter. The core keeps two, and they
        // are charged on mutually exclusive branches (core.cpp): launch_failures for
        // "no forward progress at all", no_checkpoint_launches for "ran real steps
        // but checkpointed nothing". Keying the gate on launch_failures alone left
        // the second kind unspaced — it relaunched every scan pass (~1 s) and
        // exhausted all max_launch_failures strikes in ~5 s, which is precisely the
        // burn-all-strikes-in-seconds outcome this backoff exists to prevent. Both
        // counters are capped at max_launch_failures and both reset on durable
        // progress, so max() is the honest "how many consecutive attempts achieved
        // nothing durable" and stays within backoff_ms's documented shift bound.
        //
        // A last_launch_ms in the FUTURE means the wall clock, not the job, is
        // wrong (clock stepped back, or a job file restored from a host with a
        // skewed clock — reachable across restarts now that the field persists).
        // Waiting for real time to catch up could park the job for hours, so treat
        // it as "no backoff owed": the strike caps still bound crash-looping.
        const std::uint64_t now_ms  = ms_since_epoch(r.now);
        const int           strikes = std::max(j.launch_failures, j.no_checkpoint_launches);
        if (strikes > 0 && j.last_launch_ms > 0 && j.last_launch_ms <= now_ms &&
            now_ms < j.last_launch_ms + backoff_ms(strikes, cfg))
            continue;
        // Disk pressure gates the START of new compute, never its continuation.
        // A running child is left alone deliberately: killing it would discard the
        // steps since its last checkpoint and free nothing that is not already on
        // disk, and its next checkpoint is small next to a build. Terminal moves
        // (done/failed/cancelled) and worktree GC above stay enabled — GC is the
        // one action that gives space BACK, so disabling it here would remove the
        // daemon's only self-relief.
        if (r.disk_pressure) continue;
        spawn_candidates.push_back(&j);              // eligible to (re)launch
    }

    // ---- Slot budget: launch the highest-priority ready jobs ---------------
    // The scheduler. Ordering is derived purely from the job set + reality
    // (running_count), not from action interdependence.
    int budget = cfg.slots - r.running_count;
    if (budget < 0) budget = 0;

    std::sort(spawn_candidates.begin(), spawn_candidates.end(),
              [](const job* a, const job* b) { return job_queue_less(*a, *b); });

    for (int i = 0; i < budget && i < static_cast<int>(spawn_candidates.size()); ++i)
        out.push_back(spawn_child{spawn_candidates[i]->id});

    // ---- Worktree GC: reclaim idle, unreferenced build trees ---------------
    // A SHA with refcount 0 whose worktree has been idle past the cool-off is
    // reclaimable. A new job for the same SHA during cool-off reuses the tree
    // (it is still on disk), so this is pure hygiene, never on the hot path.
    for (const sha_t& sha : r.worktrees_on_disk) {
        if (refcount[sha] > 0) continue;             // still referenced -> keep
        auto it = r.worktree_idle_since.find(sha);
        if (it == r.worktree_idle_since.end()) continue;  // unknown age -> keep (conservative)
        if (r.now - it->second > cfg.worktree_cooloff)
            out.push_back(gc_worktree{sha});
    }

    return out;
}

} // namespace coastd
