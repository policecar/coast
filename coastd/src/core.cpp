#include "daemon/core.h"

#include <csignal>
#include <sys/wait.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <system_error>
#include <thread>
#include <utility>

#include <nlohmann/json.hpp>

#include "core/app_config.h"        // coast::paths::app_config (the child's --app-config)
#include "daemon/atomic_write.h"
#include "daemon/child_cmd.h"
#include "daemon/child_proc.h"
#include "daemon/log_sweep.h"
#include "daemon/reality_probe.h"
#include "daemon/reconcile.h"
#include "daemon/version.h"
#include "diag/diag.h"
#include "sim_core/run_manifest.h"  // sim::iso8601_utc

namespace coastd {

namespace {
template <class... Ts> struct overload : Ts... { using Ts::operator()...; };
template <class... Ts> overload(Ts...) -> overload<Ts...>;

std::uint64_t now_epoch_ms() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

// Restart the worktree cool-off clock for a tree that was just USED.
//
// reality_probe derives worktree_idle_since from the worktree directory's mtime,
// and the builder writes the .coastd-ready marker into that directory LAST — so
// without this the mtime freezes at build-completion and "idle since" silently
// means "built at". A SHA referenced for longer than worktree_cooloff (6 h by
// default — routine for an --unlimited run) was then GC-eligible the instant its
// refcount hit 0, with no grace at all, forcing a full rebuild on an immediate
// resubmit. Nothing was ever at risk of being reclaimed out from under a running
// job (reconcile's refcount>0 guard is what prevents that); the cost was purely
// lost reuse, which is the entire point of the cool-off.
//
// Called at the two events that bound a reference's life: a spawn (the tree is in
// use now) and a terminal move (the reference is being released, so the idle
// clock should start HERE, not at the last spawn — otherwise a long run that was
// launched once, hours ago, is stale the moment it finishes).
//
// Best-effort and deliberately quiet at debug: failing to touch costs a premature
// rebuild, never correctness, and the tree may legitimately be gone already (a
// concurrent GC, an operator's rm -rf) by the time a terminal move lands.
void touch_worktree(const std::filesystem::path& dir) {
    if (dir.empty()) return;
    std::error_code ec;
    std::filesystem::last_write_time(dir, std::filesystem::file_time_type::clock::now(), ec);
    if (ec)
        COAST_DEBUG("coastd.core", "could not refresh the cool-off clock on {}: {}",
                    dir.string(), ec.message());
}
} // namespace

core::core(daemon_config cfg)
    : cfg_(std::move(cfg)),
      store_(cfg_),
      pool_(cfg_.worker_pool_size, queue_),
      builder_(cfg_),
      sup_(cfg_, queue_) {}

void core::write_app_config() {
    if (cfg_.app_config_path.empty()) return;  // mechanism disabled (e.g. a direct-driven test)
    // A daemon-owned coast app_config. config_base = "<config_dir>/" makes every
    // app-config-governed output (observe, logs, snapshot default) resolve flatly
    // under the child's run dir, because the child is launched with
    // -c <run_dir>/config.json so <config_dir> expands to that run dir. Written
    // authoritatively (overwrite, not never-clobber) each boot so it always reflects
    // this daemon's policy; passed to every child via --app-config, which — being the
    // highest discovery precedence — also shadows any inherited $COAST_APP_CONFIG.
    coast::paths::app_config app;
    app.paths.config_base = "<config_dir>/";
    atomic_write(cfg_.app_config_path, nlohmann::json(app).dump(2));
    COAST_INFO("coastd.core", "wrote daemon app_config {} (config_base=<config_dir>/)",
               cfg_.app_config_path.string());
}

void core::boot() {
    store_.init_store();
    store_.load_all();
    write_app_config();                       // pin per-run output locations; shadow ambient app_config
    recover_at_boot();                        // pid duplicate-guard + torn-checkpoint sweep
    store_.sweep_stale_tmp();                 // leftover job-file *.tmp from a crash mid-write
    store_.commit("boot: catch-up");          // re-record any FS state a crash left uncommitted
    // The rest of recovery IS the steady-state path: run()'s first iteration
    // rescans reality and reconciles, relaunching each interrupted job from its
    // newest checkpoint. No separate recovery code to rot.
    COAST_INFO("coastd.core", "booted: {} active job(s)", store_.active_jobs().size());
}

void core::recover_at_boot() {
    for (job j : store_.active_jobs()) {
        // pid duplicate-guard. Under systemd's cgroup reap the child is already
        // dead at boot (is_same_process -> false, no-op). Off-systemd (dev/macOS)
        // an orphan can survive daemon death; relaunching it would put two children
        // on the same run dir (the R1 nightmare). Kill the survivor first. It was
        // reparented away from us, so we can only signal it, not waitpid it.
        bool orphan_survives = false;
        if (j.pid > 0 && !j.start_token.empty() &&
            is_same_process(static_cast<pid_t>(j.pid), j.start_token)) {
            COAST_WARN("coastd.core", "boot: killing orphaned child pid {} of job {}",
                       j.pid, j.id);
            send_signal(static_cast<pid_t>(j.pid), SIGKILL);

            // CONFIRM the death; do not assume it. send_signal returns void and
            // downgrades a failed kill to a warning, so "we sent SIGKILL" is not
            // "the process is gone" — EPERM (the orphan reparented under different
            // credentials, or coastd restarted as another user) leaves it running,
            // and is_same_process correctly reports EPERM as ALIVE. A child wedged
            // in uninterruptible I/O also outlives its SIGKILL until the D-state
            // clears. Clearing the identity in either case is what puts a SECOND
            // coast on the same run dir — the exact R1 corruption this guard exists
            // to prevent, produced by the guard itself.
            //
            // Poll for a bounded window: a D-state that clears is the common
            // recoverable case. This costs boot time only when an orphan actually
            // survives (normally is_same_process is false and we never get here);
            // with N surviving orphans it is N x the bound, sequentially.
            constexpr auto confirm_window = std::chrono::seconds(5);
            const auto deadline = std::chrono::steady_clock::now() + confirm_window;
            while (is_same_process(static_cast<pid_t>(j.pid), j.start_token)) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    orphan_survives = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            if (orphan_survives)
                COAST_ERROR("coastd.core", "boot: orphaned child pid {} of job {} SURVIVED "
                            "SIGKILL after {}s — keeping its recorded identity so the spawn "
                            "guard refuses to launch a second child onto the same run dir. "
                            "This job will not run until the orphan is gone and coastd is "
                            "restarted; kill pid {} by hand if it is stuck.",
                            j.pid, j.id, confirm_window.count(), j.pid);
        }

        // Its recorded identity is now stale (killed and confirmed gone, or already
        // dead). Clear it so reconcile treats the job as not-running and relaunches
        // from the newest checkpoint. NOT cleared when the orphan survived: a
        // nonzero pid is exactly what the spawn dispatcher's skew guard tests, so
        // leaving it set is what blocks the second launch.
        bool dirty = false;
        if (!orphan_survives && (j.pid != 0 || !j.start_token.empty())) {
            j.pid = 0;
            j.start_token.clear();
            dirty = true;
        }

        // Fold any outcome the run dir already records but the store never saw. A
        // daemon crash BETWEEN a child's clean exit and its child_exited event being
        // applied leaves the job file lagging: steps_done stale, or a fatal exit
        // unrecorded. read_outcome re-derives both from the manifest + newest
        // checkpoint — the SAME source the exit path uses — so:
        //   * a job that actually FINISHED (steps_done >= target) is recognized and
        //     moved to done/ instead of being relaunched from scratch. This matters
        //     most for a bounded job with steps_target < checkpoint_interval, which
        //     writes NO periodic checkpoint: without this fold the spawn dispatcher
        //     (which only catches up from the snapshots dir) would rerun it from 0.
        //   * a fatal exit (exit_reason=fatal_error) is filed to failed/ instead of
        //     being resumed — the manifest's fatal flag is the ONLY record of it and
        //     is otherwise dropped when its child_exited event is lost.
        // The manifest is deleted before each launch, so a present one always
        // describes THIS job's last clean/fatal exit (a SIGKILLed child writes none).
        const child_outcome oc = read_outcome(j);
        if (oc.steps_done > j.steps_done) { j.steps_done = oc.steps_done; dirty = true; }
        if (!oc.checkpoint.empty() && oc.checkpoint != j.checkpoint) {
            j.checkpoint = oc.checkpoint; dirty = true;
        }
        if (oc.fatal && j.failure_reason.empty()) {
            j.failure_reason = "coast reported a fatal error"; dirty = true;
        }
        if (dirty) store_.put(j);

        // Sweep torn checkpoint *.tmp under this job's run dir. A SIGKILL mid-write
        // leaves one; coast's atomic-durable write means the newest COMPLETE
        // checkpoint (a plain *.h5) is intact, so newest_checkpoint already ignores
        // the tmp — this is pure hygiene against accumulation.
        std::error_code ec;
        const auto snaps = run_snapshots_dir(j);
        if (std::filesystem::is_directory(snaps, ec)) {
            for (const auto& de : std::filesystem::directory_iterator(snaps, ec)) {
                if (de.path().extension() != ".tmp") continue;
                std::error_code rec;
                if (std::filesystem::remove(de.path(), rec))
                    COAST_INFO("coastd.core", "boot: swept torn checkpoint {}",
                               de.path().string());
            }
        }
    }
}

void core::run() {
    sup_.start();
    // Latched flag, never re-armed here: a stop() that arrived before this thread
    // was scheduled must still be honoured (see core.h stop_requested_). The
    // sup_.shutdown() below then runs on the zero-iteration path too, which is what
    // we want — it is the same orderly child teardown the normal exit performs.
    while (!stop_requested_.load()) {
        try {
            event e;
            // STEADY clock: this is a ~1 Hz backstop, not a wall-clock schedule; a
            // backward NTP step must not stretch it (event_queue::pop_until note).
            const auto deadline = std::chrono::steady_clock::now() + cfg_.scan_period;
            if (queue_.pop_until(deadline, e)) apply(e);
            // Re-check BEFORE the dispatch phase. A stop() that lands while we are
            // parked in pop_until would otherwise still complete a full
            // rescan/reconcile/dispatch pass — which can SPAWN a child that
            // sup_.shutdown() then immediately SIGTERMs. Self-healing and lossless,
            // but pure waste, and it makes shutdown logs read as if a job had
            // started and been cancelled.
            if (stop_requested_.load()) break;

            const reality r = rescan(sup_, cfg_);
            log_disk_pressure_edge(r);
            for (const action& a : reconcile(store_.active_jobs(), r, cfg_)) dispatch(a);
            maybe_sweep_logs();
        } catch (const std::exception& ex) {
            // IDD §10: a store-write or other I/O failure inside the core loop is
            // SERIOUS but MUST NOT crash the daemon. An exception escaping this
            // thread's entry function would call std::terminate — killing the whole
            // process (and, on a persistent condition like a full disk, a systemd
            // RestartSec crash-loop). Degrade to a logged daemon-health error and
            // continue; the next iteration re-derives reality and retries. A
            // recurring error is the operator's signal, not a crash.
            COAST_ERROR("coastd.core", "core loop iteration failed (continuing): {}", ex.what());
        }
    }
    sup_.shutdown();
}

// Age-based log housekeeping, off the core thread.
//
// Scheduled here rather than in reconcile because it is pure housekeeping: it
// depends on no job state, produces no action, and must never influence what the
// scheduler decides. Throttled to cfg_.log_sweep_period — the loop runs at ~1 Hz
// and a filesystem walk per second would be absurd — and gated by an in-flight
// flag so a slow sweep (gzip on a big log) is never stacked on itself. The work
// runs on the worker pool: it shells out to gzip and walks two directory trees,
// neither of which may block the single writer.
void core::maybe_sweep_logs() {
    if (log_sweep_in_flight_) return;
    const auto now = std::chrono::steady_clock::now();
    if (last_log_sweep_ != std::chrono::steady_clock::time_point{} &&
        now - last_log_sweep_ < cfg_.log_sweep_period)
        return;
    last_log_sweep_ = now;
    log_sweep_in_flight_ = true;
    // Keyed as `commit` — the maintenance kind. The result only clears the flag;
    // nothing else in the core keys off it, and the worker pool's backstop
    // synthesizes one even if the task throws, so the flag cannot latch.
    pool_.submit({task_kind::commit, {}, {}}, [cfg = cfg_]() {
        const sweep_stats st = sweep_logs(cfg);
        if (st.compressed || st.deleted || st.failed)
            COAST_INFO("coastd.core", "log sweep: {} compressed, {} deleted, {} failed",
                       st.compressed, st.deleted, st.failed);
        return worker_result{task_kind::commit, "", "", true, "log_sweep"};
    });
}

// Report disk pressure on its EDGES only (see core.h last_disk_pressure_). Both
// directions are worth a line: the onset explains why a queue suddenly stopped
// making progress, and the recovery is what tells the operator their `rm` worked.
void core::log_disk_pressure_edge(const reality& r) {
    if (r.disk_pressure == last_disk_pressure_) return;
    last_disk_pressure_ = r.disk_pressure;

    const double free_mib = static_cast<double>(r.disk_free_bytes) / (1024.0 * 1024.0);
    const double floor_mib = static_cast<double>(cfg_.min_free_disk_bytes) / (1024.0 * 1024.0);
    if (r.disk_pressure)
        COAST_ERROR("coastd.core",
                    "DISK PRESSURE: {:.0f} MiB free on the daemon roots, below the "
                    "{:.0f} MiB floor — no new builds or job launches will start "
                    "(running jobs continue). Free space or lower min_free_disk_bytes.",
                    free_mib, floor_mib);
    else
        COAST_INFO("coastd.core", "disk pressure cleared: {:.0f} MiB free (floor {:.0f} MiB); "
                   "builds and launches resume", free_mib, floor_mib);
}

void core::stop() {
    stop_requested_.store(true);
    queue_.close();  // wake pop_until immediately
}

// ---- RPC seam ---------------------------------------------------------------

namespace {
// Run an RPC body and fulfil its promise — with the EXCEPTION when one escapes.
// These bodies write the store, and a durable write can throw (ENOSPC, a
// read-only state_root). Letting that unwind into run()'s generic catch left the
// promise unfulfilled, so the waiting connection saw std::future_error /
// "broken promise" and the client got an opaque "internal error" while the real
// cause went only to the daemon log. Capturing it here hands the client the
// actual message and keeps the loop's own error path for genuinely loop-level
// faults.
//
// (do_submit's store_.put() is safe to leave throwing: put() writes durably
// BEFORE touching the in-memory cache, so a failed submit leaves no phantom
// entry. try_put is not usable there — it refuses to create, by design.)
template <typename T, typename Fn>
void fulfil(std::promise<T>& prom, Fn&& fn) {
    try {
        prom.set_value(std::forward<Fn>(fn)());
    } catch (...) {
        prom.set_exception(std::current_exception());
    }
}
} // namespace

std::future<job_id> core::submit(job_spec spec) {
    auto prom = std::make_shared<std::promise<job_id>>();
    auto fut  = prom->get_future();
    queue_.push(client_command{
        [spec = std::move(spec), prom](core& c) { fulfil(*prom, [&] { return c.do_submit(spec); }); },
        "submit"});
    return fut;
}

std::future<bool> core::cancel(job_id id) {
    auto prom = std::make_shared<std::promise<bool>>();
    auto fut  = prom->get_future();
    queue_.push(client_command{
        [id = std::move(id), prom](core& c) { fulfil(*prom, [&] { return c.do_cancel(id); }); },
        "cancel"});
    return fut;
}

std::future<bool> core::reorder_relative(job_id id, int delta) {
    auto prom = std::make_shared<std::promise<bool>>();
    auto fut  = prom->get_future();
    queue_.push(client_command{
        [id = std::move(id), delta, prom](core& c) {
            fulfil(*prom, [&] { return c.do_reorder_relative(id, delta); });
        },
        "reorder_relative"});
    return fut;
}

std::future<bool> core::reorder(job_id id, int pos) {
    auto prom = std::make_shared<std::promise<bool>>();
    auto fut  = prom->get_future();
    queue_.push(client_command{
        [id = std::move(id), pos, prom](core& c) { fulfil(*prom, [&] { return c.do_reorder(id, pos); }); },
        "reorder"});
    return fut;
}

void core::apply_live_step(job_view& v) const {
    // progress_of() returns a value only for a supervised child that has emitted at
    // least one progress line — exactly the running-but-not-yet-checkpointed case
    // the durable steps_done cannot show. steps_done is left untouched.
    if (auto p = sup_.progress_of(v.j.id)) v.live_step = p->step;
}

std::future<std::vector<job_view>> core::list() {
    auto prom = std::make_shared<std::promise<std::vector<job_view>>>();
    auto fut  = prom->get_future();
    queue_.push(client_command{[prom](core& c) {
        fulfil(*prom, [&] {
            std::vector<job_view> out;
            for (auto& [j, b] : c.store_.all_jobs()) {
                job_view v{std::move(j), b};
                c.apply_live_step(v);
                out.push_back(std::move(v));
            }
            return out;
        });
    }, "list"});
    return fut;
}

std::future<std::optional<job_view>> core::query(job_id id) {
    auto prom = std::make_shared<std::promise<std::optional<job_view>>>();
    auto fut  = prom->get_future();
    queue_.push(client_command{[id = std::move(id), prom](core& c) {
        fulfil(*prom, [&]() -> std::optional<job_view> {
            auto j = c.store_.get(id);
            auto b = c.store_.bucket_of(id);
            if (!j || !b) return std::nullopt;
            job_view v{*j, *b};
            c.apply_live_step(v);
            return v;
        });
    }, "query"});
    return fut;
}

// ---- command handlers (core thread) -----------------------------------------

std::string core::new_job_id(const std::string& hint) const {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    ::gmtime_r(&t, &tm);
    char ts[24];
    std::strftime(ts, sizeof ts, "%Y%m%dT%H%M%SZ", &tm);

    std::string m;
    for (char ch : hint) {
        const char lc = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if ((lc >= 'a' && lc <= 'z') || (lc >= '0' && lc <= '9') || lc == '-') m += lc;
        if (m.size() >= 12) break;
    }
    if (m.empty()) m = "job";

    std::random_device rd;
    char hex[7];
    std::snprintf(hex, sizeof hex, "%06x", static_cast<unsigned>(rd() & 0xffffffu));
    return std::string(ts) + "_" + m + "_" + hex;
}

job_id core::do_submit(const job_spec& spec) {
    job j;
    j.coastd_version = std::string(coastd::version);
    // Mnemonic preference: explicit label, else the client-reported config basename
    // (wire submits), else the local config path's stem (in-process callers).
    const std::string hint =
        !spec.label.empty()       ? spec.label
      : !spec.config_name.empty() ? std::filesystem::path(spec.config_name).stem().string()
                                  : spec.config_path.stem().string();
    j.id = new_job_id(hint);
    // Collision guard: the id is timestamp(1s) + label + 24-bit random, so a burst
    // of same-label submissions in one second could in principle repeat. A repeat
    // would silently clobber an existing job's file — regenerate until unique
    // rather than overwrite (the store never silently loses a job elsewhere).
    while (store_.bucket_of(j.id)) j.id = new_job_id(hint);
    j.submitted_at   = sim::iso8601_utc(std::chrono::system_clock::now());
    j.user_message   = spec.user_message;
    j.ref_symbolic   = spec.ref_symbolic;
    j.run_dir        = (cfg_.runs_root / j.id).string();
    // steps_target and steps_done live in coast's step space, which a loaded state
    // continues (see job::load_state); 0 stays unbounded. The edge has refused an
    // overflow.
    j.steps_target   = spec.steps_target == 0 ? 0 : spec.load_state_step + spec.steps_target;
    j.steps_done     = spec.load_state_step;
    j.load_state_from = spec.load_state_from;
    j.load_state     = spec.load_state;
    j.load_state_step = spec.load_state_step;
    j.priority       = spec.priority;
    j.status         = job_status::queued;

    // Prepare the run config from the EXACT bytes the edge validated (carried in
    // spec.config_text). A wire submit always supplies them — the config travelled
    // as content, so there is no file to re-read and the submit-config TOCTOU cannot
    // exist. A DIRECT in-process caller may instead pass a local path; read it ONCE
    // here (a single read has no TOCTOU). Either way the file is read at most once.
    std::string config_text = spec.config_text;
    if (config_text.empty()) {
        std::ifstream in(spec.config_path);
        if (in) { std::ostringstream ss; ss << in.rdbuf(); config_text = ss.str(); }
    }
    if (!prepare_run_config(config_text, j, cfg_.dataset_base)) {
        const std::string src = !spec.config_path.empty() ? spec.config_path.string()
                              : !spec.config_name.empty() ? "submitted config '" + spec.config_name + "'"
                                                          : "the submitted config";
        j.failure_reason = "could not prepare run config from " + src;
    }

    store_.put(j);
    store_.commit("submit " + j.id);
    COAST_INFO("coastd.core", "submitted job {} (ref {}, target {})", j.id, j.ref_symbolic,
               j.steps_target);
    return j.id;
}

bool core::do_cancel(const job_id& id) {
    auto b = store_.bucket_of(id);
    if (!b || is_terminal(*b)) return false;  // unknown or already terminal
    auto j = store_.get(id);
    if (!j) return false;
    j->cancel_requested = true;
    store_.put(*j);
    COAST_INFO("coastd.core", "cancel requested for {}", id);
    return true;
}

// Move `id` by `delta` places within the pending queue, resolved against the
// queue AS IT IS NOW.
//
// The absolute-position form below is inherently racy for a UI: a client computes
// `pos` from a snapshot (the GUI polls at ~2 s) and by the time the request lands
// the queue may have changed — the job above may have started running, another may
// have been submitted — so the click moves a DIFFERENT job than the one the user
// pointed at. No client-side refresh closes that; only resolving on this thread,
// which is the single writer and therefore sees the authoritative order, does.
bool core::do_reorder_relative(const job_id& id, int delta) {
    auto b = store_.bucket_of(id);
    if (!b || is_terminal(*b)) return false;

    std::vector<job> pending;
    for (job& j : store_.active_jobs())
        if (j.status != job_status::running) pending.push_back(std::move(j));
    std::sort(pending.begin(), pending.end(), job_queue_less);
    const auto it = std::find_if(pending.begin(), pending.end(),
                                 [&](const job& j) { return j.id == id; });
    if (it == pending.end()) return false;   // not pending (already running?)

    const int here = static_cast<int>(std::distance(pending.begin(), it));
    const int want = std::clamp(here + delta, 0, static_cast<int>(pending.size()) - 1);
    if (want == here) return true;           // already at the end it asked for: not a failure
    return do_reorder(id, want);
}

// Move `id` to 0-based position `pos` within the PENDING queue (active jobs not
// yet running). Because integer priorities cannot express an exact mid-queue
// insert via a single value, we renumber positionally: sort the pending set by
// the scheduler's own comparator (job_queue_less), splice `id` to the clamped
// slot, then reassign priority = N .. 1 top-to-bottom. The result is strictly
// monotonic with position, so the next reconcile schedules in exactly this
// order. Running jobs are excluded (their queue rank is moot — they already
// hold a slot), so `pos` indexes the pending jobs a human actually sees.
//
// TRADE-OFF (intentional): this renumber OVERWRITES any explicit priority a job
// was submitted with — after any reorder the pending set carries positional
// priorities N..1, not the submitted values. That is the price of expressing an
// exact drag-to-position with a single integer priority; a caller that needs a
// job to always jump the queue should re-submit/re-reorder it rather than rely on
// a large submit-time priority surviving an unrelated reorder.
bool core::do_reorder(const job_id& id, int pos) {
    auto b = store_.bucket_of(id);
    if (!b || is_terminal(*b)) return false;  // unknown or already terminal

    std::vector<job> pending;
    for (job& j : store_.active_jobs())
        if (j.status != job_status::running) pending.push_back(std::move(j));

    const auto is_target = [&](const job& j) { return j.id == id; };
    if (std::none_of(pending.begin(), pending.end(), is_target))
        return false;  // already running (or otherwise not a pending job)

    std::sort(pending.begin(), pending.end(), job_queue_less);

    // Splice the target out and reinsert at the clamped position.
    auto it = std::find_if(pending.begin(), pending.end(), is_target);
    job target = std::move(*it);
    pending.erase(it);
    const int clamped = std::clamp(pos, 0, static_cast<int>(pending.size()));
    pending.insert(pending.begin() + clamped, std::move(target));

    // Reassign strictly-decreasing priorities and persist only what changed.
    const int n = static_cast<int>(pending.size());
    for (int i = 0; i < n; ++i) {
        const int new_prio = n - i;
        if (pending[i].priority != new_prio) {
            pending[i].priority = new_prio;
            store_.put(pending[i]);
        }
    }
    store_.commit("reorder " + id + " -> pos " + std::to_string(clamped));
    COAST_INFO("coastd.core", "reordered {} to pending position {}", id, clamped);
    return true;
}

// ---- event application ------------------------------------------------------

void core::apply(event& e) {
    std::visit(overload{
        [&](client_command& c) { if (c.fn) c.fn(*this); },
        [&](worker_result& w)  { on_worker_result(w); },
        [&](child_exited& c)   { on_child_exited(c); },
    }, e);
}

void core::on_worker_result(const worker_result& wr) {
    switch (wr.kind) {
        case task_kind::resolve: {
            resolve_in_flight_.erase(wr.id);
            auto j = store_.get(wr.id);
            if (!j) break;
            // If the job went terminal (a cancel while its resolve was in flight),
            // its file has already moved to cancelled/. Writing status=building or a
            // failure_reason here would put an active-phase status into a terminal
            // bucket — the inconsistency the move_bucket handler works to prevent.
            // The build branch already filters via active_jobs(); mirror that guard.
            if (auto b = store_.bucket_of(wr.id); !b || is_terminal(*b)) break;
            if (wr.ok) { j->ref_sha = wr.sha; j->status = job_status::building; }
            else       { j->failure_reason = wr.message.empty() ? "resolve failed" : wr.message; }
            store_.put(*j);
            break;
        }
        case task_kind::build: {
            build_in_flight_.erase(wr.sha);
            // Drop the per-SHA retry state once the SHA is settled (built or declared
            // unbuildable below), so neither map grows one stale entry per SHA ever
            // built over the daemon's lifetime. (A SHA abandoned mid-retry — all its
            // jobs cancelled before the cap — leaves at most one small stale entry,
            // harmless: a later job on it would just observe a spurious first-attempt
            // backoff, not incorrect scheduling.)
            const auto clear_retry_state = [&] {
                build_failures_.erase(wr.sha);
                build_last_attempt_ms_.erase(wr.sha);
            };
            if (wr.ok) {
                clear_retry_state();  // clean build clears the retry counter
                break;  // build_task wrote .coastd-ready, so next rescan sees the SHA runnable
            }
            // A build failure is often TRANSIENT (network blip during the submodule
            // fetch or cmake FetchContent). Do NOT terminally fail every job on the
            // SHA on the first strike — that would burn a job that would have built
            // moments later, the same hazard the sim relaunch backoff was added for
            // (types.h relaunch_backoff). Count the failure; dispatch(build_worktree)
            // spaces the retry by backoff. Only at max_build_failures is the SHA
            // declared unbuildable and its jobs filed to failed/.
            const int n = ++build_failures_[wr.sha];
            if (n >= cfg_.max_build_failures) {
                for (auto& j : store_.active_jobs())
                    if (j.ref_sha == wr.sha && j.failure_reason.empty()) {
                        j.failure_reason = (wr.message.empty() ? "build failed" : wr.message) +
                                           " (after " + std::to_string(n) + " attempts)";
                        store_.put(j);
                    }
                clear_retry_state();  // jobs are terminal now; drop the retry state
            } else {
                COAST_WARN("coastd.core", "build for {} failed (attempt {}/{}); retrying after "
                           "backoff: {}", wr.sha, n, cfg_.max_build_failures,
                           wr.message.empty() ? "build failed" : wr.message);
            }
            break;
        }
        case task_kind::gc:     gc_in_flight_.erase(wr.sha); break;
        case task_kind::commit:
            // The maintenance kind. Today the only such task is the log sweep, which
            // tags itself in `message`; clearing the flag here is what lets the next
            // sweep be scheduled. The worker pool synthesizes a result even when a
            // task throws, so this flag cannot latch on a failed sweep.
            if (wr.message == "log_sweep") log_sweep_in_flight_ = false;
            break;
    }
}

void core::on_child_exited(const child_exited& ce) {
    auto j = store_.get(ce.id);
    if (!j) return;

    // The child's identity is stale now, in every bucket.
    j->pid = 0;
    j->start_token.clear();
    const std::uint64_t prev = j->steps_done;
    if (ce.steps_done > j->steps_done) j->steps_done = ce.steps_done;  // monotonic
    // Only ever ADVANCE the recorded checkpoint. read_outcome derives it from the
    // newest file in the snapshots dir, so a partially-readable dir (or a pruning
    // race) could in principle report an OLDER one than the store already holds;
    // recording that would advertise a resume point behind the real one. steps_done
    // is already guarded this way — the checkpoint that names it should be too.
    if (!ce.checkpoint.empty() && ce.steps_done >= j->steps_done)
        j->checkpoint = ce.checkpoint;

    // Terminal-bucket guard (mirror on_worker_result's resolve branch). A cancel
    // or a recorded failure may have already moved this job to cancelled/ or
    // failed/ while its child_exited sat behind an unrelated event in the queue.
    // move_bucket has set the terminal status; the job is settled. Writing
    // status=ready here, or adding a launch_failures/no_checkpoint strike (which
    // could even stamp a bogus "crash-loop" failure_reason onto a clean cancel),
    // would put an active-phase field into a terminal bucket — the exact
    // inconsistency move_bucket works to prevent. Persist only the monotonic
    // pid/steps/checkpoint bookkeeping folded above and stop.
    if (auto b = store_.bucket_of(ce.id); !b || is_terminal(*b)) {
        if (!store_.try_put(*j))
            COAST_WARN("coastd.core", "job {} terminal-exit bookkeeping not durably "
                       "persisted (continuing on in-memory state)", ce.id);
        return;
    }

    const bool progressed = j->steps_done > prev;   // DURABLE (checkpointed) progress
    const bool done       = j->steps_target > 0 && j->steps_done >= j->steps_target;
    // Did the child run ANY real steps this launch, even if it was killed before
    // the next checkpoint? `prev` is the resume base, so a tailed step beyond it is
    // genuine compute. This decouples the crash-loop guard from checkpoint_interval.
    const bool did_work   = progressed || ce.tailed_step > prev;

    if (ce.fatal) {
        j->failure_reason = "coast reported a fatal error";
    } else if (progressed) {
        j->launch_failures = 0;         // durable forward progress clears the crash-loop counter
        j->no_checkpoint_launches = 0;  // and the "ran steps but never checkpointed" counter
    } else if (j->cancel_requested) {
        // The child exited because the daemon SIGTERM'd it for a cancel, not because
        // it crashed. Charge NO crash-loop / no-checkpoint strike: the pending cancel
        // will file this job to cancelled/ next reconcile. A strike here could set a
        // "crash-loop" failure_reason, and reconcile checks failure BEFORE cancel
        // (failure is the truer terminal cause) — so the job would be misfiled to
        // failed/ with a bogus reason instead of cleanly cancelled.
    } else if (!done && did_work) {
        // Ran real steps (seen on the log tail) but was killed before its first
        // checkpoint of this launch. That is NOT a crash-loop — it is a
        // checkpoint_interval too large for how often the child is interrupted, so
        // it does NOT count a launch_failures strike (which would mislabel a working
        // job `failed`). But it is not free of a liveness risk either: a job that
        // NEVER reaches a checkpoint makes zero durable progress and would resume
        // from the same base forever. Bound it on a SEPARATE counter with a distinct
        // terminal reason so a pathological "always dies just before checkpoint_interval"
        // job cannot spin indefinitely, while an occasionally-interrupted one keeps
        // running (the counter resets on any durable progress above).
        if (++j->no_checkpoint_launches >= cfg_.max_launch_failures) {
            j->failure_reason = "ran real steps but checkpointed nothing in " +
                                std::to_string(j->no_checkpoint_launches) +
                                " consecutive launches (checkpoint_interval=" +
                                std::to_string(cfg_.checkpoint_interval) + " too large?)";
        } else {
            COAST_WARN("coastd.core",
                       "job {} ran to step {} but checkpointed nothing before exit ({}/{}); "
                       "not a crash-loop strike (consider lowering checkpoint_interval={})",
                       ce.id, ce.tailed_step, j->no_checkpoint_launches,
                       cfg_.max_launch_failures, cfg_.checkpoint_interval);
        }
    } else if (!done) {
        // No forward progress at all (e.g. an immediate segfault) and not complete:
        // this launch achieved nothing. Left unchecked it would relaunch forever;
        // cap it.
        if (++j->launch_failures >= cfg_.max_launch_failures)
            j->failure_reason = "crash-loop: " + std::to_string(j->launch_failures) +
                                " consecutive launches with no forward progress";
    }
    // Keep the advisory status honest now that the child is gone: an interrupted
    // job awaiting a slot must NOT read `running` with pid 0 — that misleads `list`
    // and, worse, makes do_reorder (which filters out status==running) refuse to
    // reorder a genuinely pending job. Terminal outcomes are labelled by the
    // imminent move_bucket; here we only fix the continue-and-resume case.
    if (j->failure_reason.empty() && !done)
        j->status = job_status::ready;

    // A clean completion (progressed to >= target) leaves failure_reason empty ->
    // reconcile moves it to done; a bounded interruption relaunches to resume;
    // the cap trip sets failure_reason -> reconcile moves it to failed.
    //
    // try_put (not put): if this durable write fails (e.g. ENOSPC), the in-memory
    // job — with pid now cleared — must still be updated, or the spawn dispatcher's
    // "pid != 0 -> a child_exited is pending" skew guard would block the resume
    // forever on a stale pid. The next successful put re-persists; a crash before
    // that is healed by boot recovery.
    if (!store_.try_put(*j))
        COAST_WARN("coastd.core", "job {} exit state not durably persisted; "
                   "continuing on in-memory state (will re-persist on resume)", ce.id);
}

// ---- dispatch (the action visitor) ------------------------------------------

void core::dispatch(const action& a) {
    std::visit(overload{
        [&](const resolve_ref& x) {
            if (resolve_in_flight_.contains(x.id)) return;
            // Advisory status only — write it with the NON-throwing try_put. If the
            // durable write threw here (as store_.put would), the exception would
            // unwind BEFORE the in-flight insert + task submit below, but on a later
            // pass reconcile re-emits resolve_ref and we'd retry cleanly. The real
            // hazard is the reverse ordering: inserting into resolve_in_flight_ and
            // THEN throwing would leave the id marked in-flight forever (only a
            // worker_result erases it, and no task was ever submitted) — permanently
            // un-schedulable. So mark in-flight and submit as one non-throwing unit,
            // AFTER the advisory write.
            if (auto j = store_.get(x.id); j && j->status == job_status::queued) {
                j->status = job_status::resolving;   // advisory phase, for `list`
                store_.try_put(*j);
            }
            resolve_in_flight_.insert(x.id);
            pool_.submit({task_kind::resolve, x.id, {}},
                         builder_.resolve_task(x.id, x.ref_symbolic));
        },
        [&](const build_worktree& x) {
            if (build_in_flight_.contains(x.sha)) return;
            // Never build a SHA whose worktree is being GC'd: gc_task runs `git
            // worktree remove` + remove_all on the same path, so a concurrent build
            // would race a directory being deleted (spurious build failure ->
            // terminal job failure). The gc finishes in seconds; its worker_result
            // clears gc_in_flight_ and reconcile re-emits this build next pass.
            if (gc_in_flight_.contains(x.sha)) return;
            // Build-retry backoff: after a failed attempt, wait an increasing spacing
            // (the SAME policy as sim relaunch) before retrying, so a transient fault
            // does not spin a rebuild every scan (~1 Hz) and burn max_build_failures
            // in seconds. Skipping this pass frees the (single) build slot for another
            // SHA rather than hot-looping this one.
            if (auto it = build_failures_.find(x.sha); it != build_failures_.end() && it->second > 0) {
                const std::uint64_t last = build_last_attempt_ms_[x.sha];
                if (last > 0 && now_epoch_ms() < last + backoff_ms(it->second, cfg_)) return;
            }
            if (static_cast<int>(build_in_flight_.size()) >= cfg_.build_concurrency) return;
            build_in_flight_.insert(x.sha);
            build_last_attempt_ms_[x.sha] = now_epoch_ms();
            pool_.submit({task_kind::build, {}, x.sha}, builder_.build_task(x.sha));
        },
        [&](const spawn_child& x) {
            if (sup_.is_supervising(x.id)) return;   // supervisor already has a live child
            auto j = store_.get(x.id);
            if (!j) return;

            // Never launch against a worktree being GC'd: gc_task is deleting that
            // tree (and its coast binary). Skip until the gc worker_result clears
            // gc_in_flight_; the tree is then gone, so reconcile emits a rebuild
            // (not a spawn) next pass — never a spawn onto a half-deleted tree.
            if (!j->ref_sha.empty() && gc_in_flight_.contains(j->ref_sha)) return;

            // (1) Store-side skew guard. supervisor::poll_once erases a child from
            // live_ids the instant it reaps it, but the child_exited event that
            // clears j->pid may still be queued behind an unrelated event. In that
            // window reality says "dead" while the store still records "running with
            // pid P". Spawning here would put a SECOND child on the run dir. A
            // nonzero recorded pid means "a prior launch is still being accounted"
            // -> do not spawn; the pending child_exited clears it and reconcile then
            // decides afresh.
            if (j->pid != 0) return;

            // (2) Reconcile the store's step count with on-disk truth BEFORE the
            // launch decision. A daemon crash between a child's final checkpoint and
            // its child_exited event leaves j->steps_done lagging the newest
            // checkpoint; the resume would then compute remaining=0. (This comment
            // used to say coast reads `-n 0` as UNLIMITED and would run forever —
            // stale: `-n 0` is a parse error now, so the failure mode is a REFUSED
            // launch rather than an unbounded run. child_cmd.cpp's plan_launch
            // documents the current semantics and clamps to 1.) Catch up from disk;
            // if the job is now complete, don't spawn (reconcile moves it to done
            // next pass).
            // An unreadable snapshots dir is NOT "no checkpoints": launching on
            // that reading restarts a long run from step 0 and destroys the
            // progress it was resuming from. Skip this pass and retry — the
            // condition is usually transient (a stale mount, a permissions blip),
            // and a job that waits is recoverable where one that restarts is not.
            bool ck_scan_failed = false;
            auto ck_now = newest_checkpoint(*j, &ck_scan_failed);
            if (ck_scan_failed) {
                COAST_ERROR("coastd.core", "job {}: snapshots dir unreadable; not launching "
                            "(a launch would restart from step 0)", j->id);
                return;
            }
            // A first launch starts from load_state (plan_launch). If that file is gone
            // since submit, starting fresh instead would silently run a different
            // experiment, so record a terminal failure; reconcile files it next pass.
            // A stat that ERRORS is not "gone" (stale mount): wait, as above.
            if (!ck_now && !j->load_state.empty()) {
                std::error_code ec;
                const auto st = std::filesystem::status(j->load_state, ec);
                if (st.type() != std::filesystem::file_type::regular) {
                    if (ec && st.type() != std::filesystem::file_type::not_found) {
                        COAST_ERROR("coastd.core", "job {}: cannot stat load_state {}: {}; "
                                    "not launching", j->id, j->load_state, ec.message());
                        return;
                    }
                    j->failure_reason = "load_state_from " + j->load_state_from + ": " +
                                        j->load_state + " is missing or not a regular file at launch";
                    store_.put(*j);
                    COAST_ERROR("coastd.core", "job {}: {}", j->id, j->failure_reason);
                    return;
                }
            }
            if (auto ck = std::move(ck_now); ck && ck->step > j->steps_done) {
                j->steps_done = ck->step;
                j->checkpoint = std::filesystem::relative(ck->path, std::filesystem::path(j->run_dir))
                                    .generic_string();
                // THROWING put(), unlike the try_put()s below in this same handler —
                // and that asymmetry is deliberate, not an oversight (it was flagged
                // as one in review, which is why it is written down now).
                //
                // Those use try_put because a live child already exists: refusing to
                // continue would strand a running process, so a failed durable write
                // must degrade to in-memory state. Here nothing has been spawned yet,
                // and what we are persisting is the RESUME POINT we are about to
                // launch against. If that write fails (ENOSPC, EIO) and we spawned
                // anyway, the child would run from a step count the store does not
                // know about; a daemon restart before its first checkpoint would then
                // resume from the OLD, lower step and silently redo work — or, with
                // steps_target already met on disk, mis-file the job.
                //
                // Throwing here means the launch simply does not happen this pass. The
                // core loop catches it, logs a daemon-health error and continues (see
                // run()'s try/catch), and the next pass re-derives reality and retries.
                // Cost is the remaining actions of THIS pass being skipped, ~1 s of
                // delay; the alternative is launching against state we failed to
                // record. Fail closed.
                store_.put(*j);
            }
            if (j->steps_target > 0 && j->steps_done >= j->steps_target) return;

            // Stamp the attempt time BEFORE spawning so the relaunch-backoff clock
            // (reconcile compares reality.now to last_launch_ms) starts from the
            // launch, whether it succeeds or fails.
            j->last_launch_ms = now_epoch_ms();

            // The tree is being used right now — restart its cool-off clock (see
            // touch_worktree). Before the spawn, so a spawn that FAILS still counts
            // as a reference: the job stays active and will retry against this tree.
            if (!j->ref_sha.empty()) touch_worktree(builder_.worktree_dir(j->ref_sha));

            auto info = sup_.spawn(*j, builder_.coast_binary(j->ref_sha));
            if (info) {
                j->pid         = info->pid;
                j->start_token = info->start_token;
                j->status      = job_status::running;
                // try_put (not put): sup_.spawn() has already produced a LIVE child.
                // If the durable write threw here, the exception would unwind with
                // j->pid unpersisted (disk AND cache), so a later restart could not
                // detect the orphan and would spawn a SECOND child on the same run
                // dir (the R1 nightmare) — the exact asymmetry on_child_exited uses
                // try_put to avoid. try_put always refreshes the in-memory cache
                // (so is_supervising + the pid!=0 skew guard keep the running daemon
                // correct) and logs if the disk write failed; the next successful
                // put re-persists. (The residual crash-BETWEEN-spawn-and-write window
                // is inherent to posix_spawn and pre-existed this change.)
                if (!store_.try_put(*j))
                    COAST_WARN("coastd.core", "job {} launched but pid not durably "
                               "persisted; running on in-memory state", x.id);
            } else {
                // (3) posix_spawn failed (coast binary missing/unexecutable, EPERM,
                // resource exhaustion). No child was created, so NO child_exited will
                // ever arrive to advance the crash-loop cap — without recording the
                // failure here the job is re-selected every reconcile pass forever,
                // silently starving the queue. Treat a spawn failure exactly like a
                // no-progress launch so it caps out to failed/ after max_launch_failures.
                if (++j->launch_failures >= cfg_.max_launch_failures)
                    j->failure_reason = "spawn failed " + std::to_string(j->launch_failures) +
                                        "x (coast binary missing or not executable?)";
                store_.put(*j);
                COAST_WARN("coastd.core", "spawn failed for job {} (attempt {}/{})",
                           x.id, j->launch_failures, cfg_.max_launch_failures);
            }
        },
        [&](const kill_child& x)  { sup_.request_kill(x.id, x.signal); },
        [&](const move_bucket& x) {
            // Keep the advisory `status` consistent with the terminal bucket so a
            // client never sees status "running" next to bucket "done"/"failed"/
            // "cancelled". move_bucket only renames the file; status is content we
            // update here first, then the rename carries it across.
            if (auto j = store_.get(x.id)) {
                if (auto st = terminal_status_for(x.to); st && j->status != *st) {
                    j->status = *st;
                    store_.put(*j);
                }
                // This job is releasing its reference on the SHA. Start the cool-off
                // clock HERE so the tree gets its full grace period from the moment it
                // became (possibly) unreferenced — see touch_worktree. Doing it only
                // at spawn would leave a long single-launch run stale on completion.
                if (!j->ref_sha.empty()) touch_worktree(builder_.worktree_dir(j->ref_sha));
            }
            store_.move_bucket(x.id, x.to);
            store_.commit(x.id + " -> " + std::string(bucket_dir(x.to)));
        },
        [&](const gc_worktree& x) {
            if (gc_in_flight_.contains(x.sha)) return;
            // Never GC a SHA that is being built (or freshly referenced again): a
            // build in flight is actively writing the very tree gc_task would delete.
            // reconcile only emits gc when refcount is 0, but a new job can reference
            // the SHA on a later pass while the cool-off GC is still pending; skip
            // here and let the (now non-zero) refcount keep the tree on the next pass.
            if (build_in_flight_.contains(x.sha)) return;
            gc_in_flight_.insert(x.sha);
            pool_.submit({task_kind::gc, {}, x.sha}, builder_.gc_task(x.sha));
        },
    }, a);
}

} // namespace coastd
