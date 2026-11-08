#ifndef COAST_COASTD_TYPES_H
#define COAST_COASTD_TYPES_H

// -----------------------------------------------------------------------------
// coastd — core value types (the shared vocabulary of the daemon)
// -----------------------------------------------------------------------------
//
// coastd is a resident per-machine service that runs COAST simulations headless:
// it builds a requested git ref natively, schedules jobs against a fixed number
// of slots, supervises the one-shot `coast` child, and auto-resumes interrupted
// jobs from their last checkpoint after a crash or reboot.
//
// The spine of the design is a single PURE function
//     reconcile(active_jobs, reality, config) -> actions
// that IS the scheduling + recovery state machine. This header defines its
// inputs (`job`, `reality`, `daemon_config`) and outputs (`action`). Keeping
// these as plain data — no methods, no I/O — is what lets the whole policy be
// unit-tested with zero subprocesses (IDD §11, the primary reliability lever).
//
// Design notes worth carrying at the type level:
//   * A `job` is an INSTANCE (one file); its lifecycle state is a `bucket`
//     (which directory it lives in), NOT a per-state type. `status` is an
//     ADVISORY copy of that phase for humans/`list`, never trusted over reality.
//   * Everything a crash could desync is DERIVED, never stored: slot
//     availability, SHA refcount, worktree presence, worktree idle-age.
//   * There is deliberately NO checkpoint-integrity state. coast writes
//     checkpoints atomic-durably (tmp -> fsync -> rename -> fsync(dir)), so the
//     newest COMPLETE checkpoint is intact by construction; recovery just picks
//     the newest one by step and cleans stale *.tmp. (The structural HDF5 check
//     the Demonstrate stage validated was the fallback for an unmodified coast,
//     which we are not shipping — see .dew design-verification.)
// -----------------------------------------------------------------------------

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "daemon/net_defaults.h"

namespace coastd {

using job_id   = std::string;   // "<UTC>_<mnemonic>_<rand>"; the filename stem IS the id
using sha_t    = std::string;   // resolved git sha, frozen at submit (worktree key + code id)
using sys_time = std::chrono::system_clock::time_point;

// Build-completion sentinel filename under a worktree. Written atomically as the
// LAST step of a successful build; its presence (not the worktree directory's)
// is what makes a SHA runnable. Shared by the builder (writer) and rescan (reader).
inline constexpr std::string_view build_ready_marker = ".coastd-ready";

// Does `s` have the shape of a resolved git SHA — hex only, 7..64 chars
// (abbreviated .. SHA-256)?
//
// ONE definition on purpose. This predicate defines the worktree namespace at BOTH
// ends: the builder gates on it before a SHA becomes a path component under
// worktrees_root (so a tampered ref_sha loaded from a job file cannot walk "../"
// into `git worktree add` or `remove_all`), and rescan gates on it when reading
// that root back (so a directory that is not ours — an operator's scratch copy, an
// editor backup — is never reported as a worktree and never handed to GC). The two
// must agree exactly: a stricter reader silently loses trees the writer created; a
// stricter writer leaves trees the reader will try, and fail, to `git worktree
// remove` on every cool-off pass forever. So it is shared, not mirrored.
inline bool is_sha_shaped(std::string_view s) noexcept {
    if (s.size() < 7 || s.size() > 64) return false;
    for (const unsigned char c : s) {
        const bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        if (!hex) return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Lifecycle bucket — DERIVED from which jobs/<bucket>/ subdirectory the file is
// in. A transition to a terminal bucket is an atomic `rename` across dirs.
// `active` is the one non-terminal bucket; the rest are terminal.
// -----------------------------------------------------------------------------
enum class bucket : std::uint8_t { active, done, failed, cancelled, quarantine };

constexpr std::string_view bucket_dir(bucket b) noexcept {
    switch (b) {
        case bucket::active:     return "active";
        case bucket::done:       return "done";
        case bucket::failed:     return "failed";
        case bucket::cancelled:  return "cancelled";
        case bucket::quarantine: return "quarantine";
    }
    return "active";
}

constexpr bool is_terminal(bucket b) noexcept { return b != bucket::active; }

// -----------------------------------------------------------------------------
// Advisory last-recorded phase within the active bucket. NEVER authoritative:
// reconcile derives what to do from `reality` (child_alive, worktrees_on_disk),
// not from this field. It exists for humans and the `list` command.
// -----------------------------------------------------------------------------
enum class job_status : std::uint8_t {
    queued,     // submitted; ref not yet resolved
    resolving,  // resolve_ref dispatched
    building,   // worktree build dispatched
    ready,      // worktree present; awaiting a slot
    running,    // child spawned
    done, failed, cancelled, quarantine
};

// The advisory job_status matching a terminal bucket, so `status` stays
// consistent with `bucket` after a move (a client must never see status
// "running" next to bucket "done"). nullopt for the non-terminal `active`
// bucket, whose status is set along the queued/resolving/building/running path.
constexpr std::optional<job_status> terminal_status_for(bucket b) noexcept {
    switch (b) {
        case bucket::done:       return job_status::done;
        case bucket::failed:     return job_status::failed;
        case bucket::cancelled:  return job_status::cancelled;
        case bucket::quarantine: return job_status::quarantine;
        case bucket::active:     return std::nullopt;
    }
    return std::nullopt;
}

// -----------------------------------------------------------------------------
// One job. Flat struct + advisory enum (chosen over a variant-of-states) so
// flat-JSON (de)serialization stays trivial and `status` stays genuinely
// advisory. `bucket` is intentionally absent — it is the file's directory.
// -----------------------------------------------------------------------------
struct job {
    int           schema_version   = 1;   // compat contract; unknown version => quarantine, never mangle
    std::string   coastd_version;         // provenance (git describe of the daemon)
    job_id        id;
    std::string   submitted_at;           // ISO-8601 Z; also the FIFO tiebreak key
    std::string   user_message;           // free-text note to other humans
    std::string   ref_symbolic;           // what the user asked for (branch/tag/sha)
    sha_t         ref_sha;                // resolved ONCE, frozen ("" = not yet resolved)
    std::string   run_dir;                // absolute; <runs_root>/<id>
    std::uint64_t steps_target     = 0;   // -n goal in coast's step space (0 = unbounded); see load_state_step
    std::uint64_t steps_done       = 0;   // cumulative across resumes
    // Initial state for the FIRST launch: the final state of job load_state_from
    // ("" = start fresh). A resume from this job's own checkpoint always wins over
    // it (plan_launch). load_state is that job's final_state.h5, resolved once at
    // submit so a restart reproduces the command line without re-resolving.
    // coast continues the loaded state's step count, so this job's checkpoints and
    // manifest count from load_state_step: do_submit stores steps_target as
    // load_state_step + the requested steps and starts steps_done at load_state_step,
    // keeping every steps_done/steps_target comparison in that one step space.
    job_id        load_state_from;
    std::string   load_state;
    std::uint64_t load_state_step  = 0;   // the source job's steps_done at submit
    std::string   checkpoint;             // newest checkpoint basename, rel run_dir ("" = none)
    int           priority         = 0;   // HIGHER runs first; ties broken by submitted_at (FIFO)
    bool          cancel_requested = false;
    job_status    status           = job_status::queued;  // advisory
    std::int64_t  pid              = 0;   // 0 = none; duplicate-guard, not reattach
    std::string   start_token;            // proc start-time; recycle-safe process identity
    int           launch_failures  = 0;   // consecutive relaunches with NO forward progress (crash-loop guard); reset on progress
    int           no_checkpoint_launches = 0;  // consecutive launches that ran REAL steps but checkpointed nothing (checkpoint_interval-too-large guard); reset on durable progress
    std::uint64_t last_launch_ms   = 0;   // epoch ms of the last spawn ATTEMPT (success or fail); drives relaunch backoff. 0 = never launched
    std::string   failure_reason;         // "" = none; non-empty => a terminal failure was recorded
    // Fields present in the job FILE that this build does not model, carried
    // through verbatim so a status update cannot delete them. The schema is
    // documented as ADDITIVE, but a parse that rebuilds the job from a fixed key
    // list plus a dump that writes only those keys silently strips anything else —
    // so an older daemon in a mixed fleet, or after a one-binary rollback, would
    // destroy a field a newer daemon added at the same schema_version. Serialized
    // JSON object text; empty when there are none. Never round-tripped into the
    // wire protocol — this is a storage concern only.
    std::string   extra_fields;
};

// The scheduler's queue order, in ONE place: priority desc, submitted_at asc
// (FIFO), id asc as the same-second total-order tiebreak. Reconcile's spawn
// ordering, both reorder forms in core, and job_store's display order must all
// agree on this — a job would otherwise show at one position in `list` and
// launch from another. It was once hand-copied at those four sites, each copy
// carrying a comment promising it matched the others.
inline bool job_queue_less(const job& a, const job& b) {
    if (a.priority != b.priority) return a.priority > b.priority;                  // higher first
    if (a.submitted_at != b.submitted_at) return a.submitted_at < b.submitted_at;  // FIFO
    return a.id < b.id;
}

// -----------------------------------------------------------------------------
// reality — the current world, gathered fresh each loop and NEVER cached. A
// derived quantity cannot desync across a crash, so recovery is just "probe
// reality at boot, then reconcile" (the same path as steady state).
// -----------------------------------------------------------------------------
struct reality {
    std::unordered_set<sha_t>           worktrees_on_disk;    // directory present (GC scope)
    std::unordered_set<sha_t>           builds_ready;         // .coastd-ready sentinel present (runnable)
    std::unordered_map<job_id, bool>    child_alive;          // pid + start_token verified
    std::unordered_map<sha_t, sys_time> worktree_idle_since;  // FS mtime, per on-disk worktree
    int      running_count = 0;                               // count of alive children
    // Free space on the daemon's roots, probed each rescan (statvfs, ~250 ns/root
    // measured — free of charge next to the directory walk above). `disk_pressure`
    // is the reconcile input (a policy answer: "may new work start?"); the byte
    // count is kept alongside it so the operator log can say HOW MUCH is left, and
    // so a bool that is always false is impossible to mistake for "never probed".
    bool          disk_pressure   = false;   // min_free < cfg.min_free_disk_bytes
    std::uint64_t disk_free_bytes = 0;       // minimum across the roots; 0 = unmeasurable
    // The worktrees scan FAILED (root exists but could not be read: permissions, a
    // stale mount, EMFILE). Without this flag an unreadable root is reported as an
    // empty `builds_ready`, which is indistinguishable from "nothing is built yet" —
    // and that reads as a call to rebuild EVERYTHING, whose repeated failure is
    // terminal for every job on the SHA. An unknown world is not an empty world:
    // reconcile holds new builds while this is set, exactly as it holds under disk
    // pressure — no strike, no failure_reason, the job simply waits.
    bool          worktrees_unreadable = false;
    sys_time now;                                             // sampled at rescan; keeps reconcile pure
};

// -----------------------------------------------------------------------------
// actions — reconcile's output. UNORDERED: an action depends only on `reality`,
// never on another action in the same pass, so a partial dispatch (crash
// mid-batch) is harmless — the next reconcile re-derives from wherever reality
// actually is. The dispatcher that executes them is idempotent per (kind, key),
// so reconcile may safely re-emit an action whose effect is already in flight.
// -----------------------------------------------------------------------------
struct resolve_ref    { job_id id; std::string ref_symbolic; };
struct build_worktree { sha_t  sha; };
struct spawn_child    { job_id id; };
struct kill_child     { job_id id; int signal; };   // SIGTERM (graceful) | SIGKILL (hard)
struct move_bucket    { job_id id; bucket to; };
struct gc_worktree    { sha_t  sha; };

// (There is deliberately no "persist this job" action. A mutated job is written by
// whichever core handler mutated it, on the core thread, at the moment it decided —
// routing that back out through reconcile would make the pure function responsible
// for state it does not own, and give the store a second write path to keep in step.)
using action = std::variant<resolve_ref, build_worktree, spawn_child,
                            kill_child, move_bucket, gc_worktree>;

// -----------------------------------------------------------------------------
// daemon_config — the fixed policy knobs. Independent, absolute roots (IDD §5.1):
// small git-versioned state vs large unversioned run artifacts vs per-SHA build
// worktrees.
// -----------------------------------------------------------------------------
struct daemon_config {
    std::filesystem::path state_root;      // <coastd_state>  git-versioned job files
    std::filesystem::path runs_root;       // <coastd_runs>   unversioned run dirs (logs, *.h5)
    std::filesystem::path worktrees_root;  // <coastd_worktrees> per-resolved-SHA build trees
    std::filesystem::path repo_root;       // the git repo coast is built from

    // Shared dataset directory exposed to every job as a `data` symlink inside its
    // run dir. A child chdirs to its run dir, so the COAST convention of a relative
    // "data/<set>/..." dataset path (which every stock config uses) resolves through
    // this link — a submitted config need not hard-code an absolute dataset path.
    // Empty (the default) disables the mechanism: no symlink is created and jobs
    // must reference datasets by absolute path, the prior behaviour. Made absolute
    // when the link is created (a relative target would resolve against the run dir,
    // never what the operator meant). See prepare_run_config in child_cmd.cpp.
    std::filesystem::path dataset_base;

    // Daemon-owned coast app_config, written at boot and passed to every child via
    // --app-config. It pins config_base to "<config_dir>/" so every app-config-
    // governed output (observe, logs, and the snapshot default) resolves FLATLY
    // under each job's run dir — the child is launched with -c <run_dir>/config.json,
    // so <config_dir> expands to that run dir. Being an EXPLICIT --app-config it also
    // has the highest discovery precedence, so it SHADOWS any inherited
    // $COAST_APP_CONFIG or stray ./coast.json that would otherwise silently redirect
    // a job's outputs out of the run dir. Absolute (the child chdirs to its run dir).
    // Empty disables the mechanism (e.g. a unit test driving plan_launch directly).
    std::filesystem::path app_config_path;

    // ---- ref freshness (git fetch) ---------------------------------------------
    // coastd resolves a submitted ref against repo_root. Without a fetch that
    // resolution sees only whatever the local clone happens to hold, so a stale
    // clone silently runs OLD CODE under a current-sounding ref name — a worse
    // failure than a clean "could not resolve", because the job succeeds and lies.
    // With fetch_enabled, resolve_task refreshes remote-tracking refs before
    // resolving, and prefers `<remote>/<ref>` over a same-named LOCAL branch for
    // plain branch names (a local branch in the daemon's clone is an accident of
    // whoever last checked something out; the remote is the source of truth on a
    // server). Explicit forms still win: `refs/heads/x`, `HEAD`, a raw SHA, a tag,
    // or anything with a revision suffix (`^`, `~`, `@{…}`) is resolved verbatim.
    // Disabling this restores the pure-local behaviour, INCLUDING local-first
    // resolution — the two are one capability.
    bool        fetch_enabled = true;
    std::string git_remote;   // "" = auto ("origin" if present, else the sole remote)

    // Bound on `git fetch` — NETWORK-bound, so it gets its own (generous) budget
    // rather than sharing git_op_timeout with the quick local rev-parse calls.
    // An overrun is NOT fatal: resolve falls back to the local clone (see below).
    std::chrono::seconds fetch_min_interval{30};  // coalescing window: skip a fetch this fresh
    std::chrono::seconds fetch_timeout{300};

    // Extra -D flags passed to the per-worktree `cmake` configure of coast (e.g.
    // "-DCOAST_GPU_BACKEND=none"). Empty = coast's own defaults. The operator
    // sets this to match the target machine (GPU backend, console on/off).
    std::vector<std::string> build_cmake_args;

    int slots               = 1;           // max concurrent running children
    int worker_pool_size    = 4;           // bounded async workers (build, gc, commit)
    int build_concurrency   = 1;           // exactly one build in flight (IDD §6E)
    int max_launch_failures = 5;           // crash-loop cap: relaunches with no progress before -> failed/
    int max_build_failures  = 3;           // consecutive build failures on a SHA before jobs on it -> failed/

    // Builder parallelism for `cmake --build -j N`. 0 = auto (half the detected
    // cores); an operator on a shared box can pin it lower.
    int build_jobs = 0;

    // ---- client-edge (comms_server) bounds -------------------------------------
    // Defaults suit a trusted-LAN lab box; tune for many-client or slow-link
    // deployments via the config file (comms_* keys). Each bound's DoS rationale
    // lives at its use site in comms_server.cpp.
    std::size_t          max_connections    = 128;        // simultaneous client connections
    // ORDERING INVARIANT: max_request_bytes MUST exceed max_config_bytes, because a
    // submit carries the config's CONTENT inside its request line (protocol.h). If
    // the line cap were the smaller of the two, a config that passes the config cap
    // would be killed by the line cap first — a legal input failing at the wrong
    // layer with the wrong message. clamp_edge_bounds() below computes the exact
    // floor, 2*max_config_bytes + 64 KiB: the 2x covers JSON string escaping
    // (measured ~8.4% on a real config; only quotes, backslashes and control chars
    // inflate) and the 64 KiB the request envelope. These DEFAULTS already satisfy
    // that floor (10 MiB > 2*4 MiB + 64 KiB = 8.06 MiB), with headroom — an earlier
    // 8 MiB default was 64 KiB short and tripped the clamp on every boot, emitting a
    // spurious warning with no config present. The clamp therefore fires only on an
    // operator override that reintroduces the trap, which is its actual purpose.
    std::size_t          max_request_bytes  = 10u << 20;  // 10 MiB — one request line
    std::size_t          max_config_bytes   = 4u << 20;   // 4 MiB — a submitted config
    std::chrono::seconds first_request_timeout{30};       // silence allowed before the first request line
    std::chrono::seconds line_assembly_timeout{30};       // total time to assemble one request line (slowloris bound)
    std::chrono::seconds send_timeout{30};                // a single reply write must make progress within this

    // Relaunch backoff. After a launch that made no durable progress a job waits
    // relaunch_backoff_base * 2^(launch_failures-1), capped at relaunch_backoff_max,
    // before its next attempt. Without this, relaunches fire every scan_period
    // (~1 s), so a purely TRANSIENT fault (fork EAGAIN under memory pressure, a
    // briefly unavailable dataset mount, a GC/build race on the SHA) burns all
    // max_launch_failures strikes in ~5 s and terminally fails a job that would
    // have succeeded moments later. Kept PURE: reconcile compares reality.now to
    // the job's last_launch_ms, so the backoff is a derived scheduling decision,
    // never persisted timer state.
    std::chrono::seconds relaunch_backoff_base{2};
    std::chrono::seconds relaunch_backoff_max{60};

    // Disk-pressure floor. Below this much free space on ANY daemon root, reconcile
    // stops STARTING new work (no build, no spawn) while letting running children
    // and worktree GC continue — GC is the only action that gives space back, and
    // killing a running child would throw away compute without freeing anything a
    // checkpoint had not already written. Sized against measurement, not taste:
    // a built worktree is ~238 MB (DESIGN_VERIFICATION, perref-worktree-refcount),
    // so 2 GiB leaves room for a build plus several checkpoints of headroom for the
    // operator to notice and act. 0 disables the check entirely.
    std::uint64_t min_free_disk_bytes = 2ULL << 30;   // 2 GiB

    // Wall-clock ceilings on the builder's subprocesses. These are BOUNDS, not
    // hang-detectors: a build's "slow" (cold FetchContent, a big submodule fetch
    // on a poor link) is indistinguishable from "hung" from the outside — CPU/
    // memory tell us nothing — so we cap total duration and treat the overrun as
    // a build failure rather than a wedge. build_timeout is per heavy step
    // (submodule update / cmake configure / cmake build), generous so a healthy
    // slow build is never killed; git_op_timeout bounds the quick local git calls
    // (resolve rev-parse, gc worktree remove/prune). An unbounded builder step is
    // the one hazard that can silently park a worker thread forever and, with
    // build_concurrency==1, block ALL future builds for the daemon's lifetime.
    std::chrono::seconds      build_timeout{60 * 60};      // per heavy build step
    std::chrono::seconds      git_op_timeout{60};          // quick local git ops (builder: resolve, gc)
    std::chrono::seconds      audit_git_timeout{20};       // job_store audit-commit git (committer thread, best-effort)

    std::chrono::seconds      worktree_cooloff{6 * 3600};  // idle worktree GC grace

    // Log housekeeping, in the standard Linux shape: compress what is old, delete
    // what is ancient. Both run logs (<run_dir>/logs/run_*.log — one per launch,
    // and an --unlimited job's can grow for weeks) and the per-SHA build log are
    // otherwise written forever with no ceiling.
    //
    // Age is judged by MTIME, which is also how a log still being written is
    // recognized: an active child's log has a fresh mtime by definition, so the
    // threshold alone keeps the sweep off any file the supervisor is tailing — no
    // coupling to job state, and no chance of rotating a file out from under an
    // open fd. 0 disables either stage.
    std::chrono::seconds      log_compress_after{7 * 24 * 3600};    // gzip logs idle this long
    std::chrono::seconds      log_delete_after{90 * 24 * 3600};     // remove .gz older than this
    std::chrono::seconds      log_sweep_period{6 * 3600};           // how often the sweep runs
    // Hard ceiling on the APPEND-ONLY build log, enforced at write time. Age alone
    // does not bound it: a SHA that crash-loops writes gigabytes within one day,
    // well inside log_compress_after. On overflow the log is restarted with a
    // marker — it is a rebuildable diagnostic, not a record of anything durable.
    std::size_t               build_log_max_bytes = 64u << 20;      // 64 MiB

    // Grace a SIGTERM'd child gets to write its final --save-on-term checkpoint
    // and exit before the supervisor escalates to SIGKILL. Applies to BOTH a job
    // CANCEL and a daemon SHUTDOWN (supervisor::shutdown SIGTERMs every child and
    // waits this long before hard-killing). Keep systemd's TimeoutStopSec above
    // it so the daemon is not itself killed mid-grace. Raise it for large models
    // whose final checkpoint write can exceed the default.
    std::chrono::seconds      term_grace{10};

    std::chrono::milliseconds scan_period{1000};           // ~1 Hz reality-scan backstop
    std::uint16_t tcp_port = default_tcp_port;
    std::uint16_t udp_port = default_udp_port;

    // ---- UDP discovery responder ------------------------------------------
    // Discovery answers an unauthenticated ~15-byte probe with a ~250 B datagram.
    // Two properties follow from that and are worth naming, because neither is
    // obvious from "it just helps clients find the daemon":
    //
    //  * It is a REFLECTOR/AMPLIFIER (~17x). The source address of a UDP probe is
    //    trivially spoofed, so an attacker can aim this daemon's replies at a third
    //    party who never contacted it. That is the one channel here that reaches a
    //    victim outside the accepted "anyone who can submit gets RCE" model.
    //  * It is the only unauthenticated channel that ever spoke about the host's
    //    filesystem layout. It no longer does — see discovery_server.cpp.
    //
    // discovery_enabled turns the responder off entirely (the TCP machine_info
    // command still answers, so a client configured with an address loses nothing;
    // only LAN auto-discovery goes away). Default ON: with the path fields removed
    // and the rate caps below in place, what remains is "a coastd exists here, at
    // this port, speaking this version" — appropriate for the trusted-lab-network
    // deployment this is built for, and the GUI's discover button depends on it.
    // Turn it OFF on any host where the port is reachable by untrusted peers.
    bool          discovery_enabled = true;
    // Interface to bind the responder to. Empty = INADDR_ANY. Setting it to a lab
    // interface's address keeps the responder off, say, a public NIC without
    // needing a firewall rule.
    std::string   discovery_bind;
    // Reply rate caps. The global cap is the amplification bound: past it the
    // daemon is useless as a reflector regardless of how fast it is probed. The
    // per-source cap bounds what any single (possibly spoofed, i.e. victim)
    // address can be made to receive. Both are far above real use — discovery is
    // user-initiated and a client probes once per refresh.
    int           discovery_max_replies_per_sec = 20;
    int           discovery_max_replies_per_source_per_sec = 2;
    std::uint64_t checkpoint_interval = 500000;            // coast --checkpoint N
};

// Enforce the client-edge ordering invariant after config load: a submit's request
// line must be able to hold a maximum-sized config plus its escaping and envelope.
// Returns true if it had to correct something (the caller logs it) — a silent fixup
// would hide an operator's mistake, and the mistake's symptom (a legal config
// rejected as an over-long line) is exactly the kind that gets misdiagnosed as a
// daemon fault. Applied to the LOADED config, so it also covers a config file that
// sets only one of the two keys.
inline bool clamp_edge_bounds(daemon_config& cfg) {
    const std::size_t required = cfg.max_config_bytes * 2 + (64u << 10);
    if (cfg.max_request_bytes >= required) return false;
    cfg.max_request_bytes = required;
    return true;
}

} // namespace coastd

#endif // COAST_COASTD_TYPES_H
