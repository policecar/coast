#include "core/run_id.h"
#include "daemon/child_cmd.h"

#include <charconv>
#include <fstream>
#include <sstream>
#include <system_error>

#include <nlohmann/json.hpp>

#include "daemon/atomic_write.h"
#include "diag/diag.h"

namespace coastd {

namespace {
namespace fs = std::filesystem;

fs::path run_dir_of(const job& j)   { return {j.run_dir}; }
fs::path logs_dir(const job& j)     { return run_dir_of(j) / "logs"; }
fs::path manifest_path(const job& j){ return run_dir_of(j) / "coast_runtime.json"; }
fs::path final_state(const job& j)  { return run_dir_of(j) / "final_state.h5"; }

// Parse the trailing "_<step>.h5" of a checkpoint filename into its step number.
std::optional<std::uint64_t> step_of(const std::string& filename) {
    if (!filename.ends_with(".h5")) return std::nullopt;
    const std::string stem = filename.substr(0, filename.size() - 3);  // drop .h5
    const auto us = stem.rfind('_');
    if (us == std::string::npos) return std::nullopt;
    const std::string digits = stem.substr(us + 1);
    if (digits.empty()) return std::nullopt;
    std::uint64_t v = 0;
    const auto* end = digits.data() + digits.size();
    auto [p, ec] = std::from_chars(digits.data(), end, v);
    if (ec != std::errc{} || p != end) return std::nullopt;
    return v;
}
} // namespace

fs::path run_snapshots_dir(const job& j) { return run_dir_of(j) / "snapshots"; }
fs::path run_config_path(const job& j)   { return run_dir_of(j) / "config.json"; }
fs::path run_submitted_config_path(const job& j) { return run_dir_of(j) / "config.submitted.json"; }

// Expose the operator's shared dataset directory as a `data` symlink inside the
// run dir. The child chdirs to the run dir, so a config's conventional relative
// "data/<set>/..." path then resolves through this link. Best-effort and
// idempotent: a resume finds the link already present, and any failure only costs
// the convenience — never the run — so nothing here returns an error.
fs::path dataset_link_target(const fs::path& dataset_base) {
    if (dataset_base.empty()) return {};
    // Absolute target: a relative dataset_base would resolve against the LINK's
    // directory (the run dir), never the daemon's working dir the operator meant.
    // Absolutize against the daemon CWD, matching how a relative root resolves.
    std::error_code ec;
    fs::path target = fs::absolute(dataset_base, ec);
    return ec ? dataset_base : target;
}

static void link_dataset_base(const fs::path& dataset_base, const job& j) {
    if (dataset_base.empty()) return;   // mechanism disabled

    std::error_code ec;
    const fs::path target = dataset_link_target(dataset_base);

    if (!fs::exists(target, ec))
        COAST_WARN("coastd.cmd", "dataset_base {} does not exist; a job's relative "
                   "data paths will not resolve until it does", target.string());

    const fs::path link = run_dir_of(j) / "data";
    // Refresh a stale link (dataset_base changed, or a prior submit's leftover) but
    // never clobber a real file/dir a name collision could have put there. remove()
    // on a symlink drops the LINK, not its target.
    if (fs::is_symlink(link, ec)) fs::remove(link, ec);
    if (fs::exists(link, ec)) {
        COAST_WARN("coastd.cmd", "run dir {} already holds a non-symlink 'data' entry; "
                   "leaving it in place", run_dir_of(j).string());
        return;
    }
    fs::create_directory_symlink(target, link, ec);
    if (ec)
        COAST_WARN("coastd.cmd", "could not link dataset_base {} into run dir {}: {}",
                   target.string(), run_dir_of(j).string(), ec.message());
}

bool prepare_run_config(const std::string& config_text, const job& j,
                        const fs::path& dataset_base) {
    std::error_code ec;
    fs::create_directories(run_snapshots_dir(j), ec);
    fs::create_directories(logs_dir(j), ec);
    link_dataset_base(dataset_base, j);

    // Parse the bytes handed in (validated at the connection edge). Re-parsing the
    // SAME bytes yields the SAME DOM the path scan saw, so "validated == written"
    // holds; the source path is deliberately not re-read.
    nlohmann::json cfg;
    try {
        cfg = nlohmann::json::parse(config_text);
    } catch (const std::exception& e) {
        COAST_ERROR("coastd.cmd", "run config for job {} is not valid JSON: {}", j.id, e.what());
        return false;
    }

    // coastd owns checkpoint location: force an absolute snapshot_dir so coast
    // writes checkpoints exactly where coastd will look, independent of the
    // child CWD or any app_config (an absolute leaf wins verbatim).
    cfg["snapshot_dir"] = run_snapshots_dir(j).string();

    // A run_id of "auto" (or none: its default) becomes a fresh id here, once, so
    // every launch of this job reads the same one and its datasets continue their
    // sequences from its own checkpoints (SPEC.md S.5.14). Anything else is kept:
    // an explicit id, or a non-string, which coast refuses when the job launches.
    if (const auto it = cfg.find("run_id"); it == cfg.end() || *it == coast::k_auto_run_id)
        cfg["run_id"] = coast::new_run_id();

    // Durable, atomic write — a crash/disk-full mid-write must not leave a torn
    // config.json that the next resume would feed to `coast -c` (parse failure,
    // no re-prepare path). Same tmp->fsync->rename->fsync(dir) as the job store.
    try {
        atomic_write(run_config_path(j), cfg.dump(2));
    } catch (const std::exception& e) {
        COAST_ERROR("coastd.cmd", "cannot write {}: {}", run_config_path(j).string(), e.what());
        return false;
    }

    // Archive the submission VERBATIM, next to the config actually run. The file
    // above is not the user's file: it has snapshot_dir injected and is re-dumped
    // through nlohmann, so its key order and formatting are the serializer's, not
    // theirs. Without this copy the exact bytes that were submitted exist nowhere
    // once the client goes away — which is precisely what you want back when a run
    // months later needs explaining. Cheap (one small text file per run) and
    // strictly additive: nothing reads it, so a failure here must NOT fail the job.
    try {
        atomic_write(run_submitted_config_path(j), config_text);
    } catch (const std::exception& e) {
        COAST_WARN("coastd.cmd", "could not archive the submitted config for job {}: {} "
                                 "(the run is unaffected)", j.id, e.what());
    }
    return true;
}

std::optional<checkpoint_info> newest_checkpoint(const job& j, bool* scan_failed) {
    if (scan_failed) *scan_failed = false;
    const fs::path dir = run_snapshots_dir(j);
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        // A missing snapshots dir is the ordinary pre-first-checkpoint state; an
        // is_directory that ERRORS (permissions, I/O, a stale mount) is not.
        if (ec && scan_failed) *scan_failed = true;
        return std::nullopt;
    }

    std::optional<checkpoint_info> best;
    // A failed directory_iterator construction yields end(), so the loop below
    // simply never runs — indistinguishable from an empty directory unless the
    // error_code is read. Read it.
    //
    // The iteration must not THROW either, and a range-for cannot express that: it
    // uses the throwing operator++, and directory_entry's argless predicates throw
    // too. Both are reachable — a stale mount or an EACCES appearing mid-scan — and
    // this function's entire contract is to REPORT a failed scan through
    // scan_failed. Throwing it instead lands in supervisor::poll_once between
    // reaping a child and recording that fact (read_outcome is called there), i.e.
    // in the one window where an escape corrupts the supervisor's registry. Hence
    // the explicit increment(ec) loop.
    //
    // Entry-level probes get their OWN error_code: a single file vanishing between
    // readdir and stat (checkpoint pruning races this scan by design) means skip
    // that entry, not "the scan failed" — which would wrongly block a resume.
    fs::directory_iterator it(dir, ec);
    const fs::directory_iterator end;
    for (; !ec && it != end; it.increment(ec)) {
        std::error_code fec;
        if (!it->is_regular_file(fec)) continue;   // false on error: skip, don't fail
        const auto step = step_of(it->path().filename().string());
        if (!step) continue;
        if (!best || *step > best->step) best = checkpoint_info{it->path(), *step};
    }
    if (ec) {
        COAST_ERROR("coastd.cmd", "could not scan snapshots for job {}: {}", j.id, ec.message());
        if (scan_failed) *scan_failed = true;
        return std::nullopt;
    }
    return best;
}

launch_plan plan_launch(const job& j, const daemon_config& cfg, const fs::path& coast_bin) {
    launch_plan p;
    p.exe = coast_bin.string();
    p.cwd = run_dir_of(j);
    // One log file per LAUNCH. steps_done alone collides across a crash-loop
    // (no-progress relaunches share it) and an O_TRUNC open destroys the prior
    // attempt's log. launch_failures disambiguates the no-progress relaunches;
    // no_checkpoint_launches disambiguates the "ran real steps but checkpointed
    // nothing" relaunches, which advance NEITHER steps_done NOR launch_failures
    // (so without it those attempts would still collide). The triple is unique per
    // attempt: durable progress resets both counters and bumps steps_done.
    p.logfile = logs_dir(j) / ("run_" + std::to_string(j.steps_done) + "_" +
                               std::to_string(j.launch_failures) + "_" +
                               std::to_string(j.no_checkpoint_launches) + ".log");
    p.manifest = manifest_path(j);

    // The core's spawn dispatcher refuses to launch when the scan failed, so
    // reaching here with scan_failed is a logic error — say so rather than
    // silently planning a from-scratch run over an existing checkpoint.
    bool scan_failed = false;
    const auto resume = newest_checkpoint(j, &scan_failed);
    if (scan_failed)
        COAST_ERROR("coastd.cmd", "job {}: planning a launch despite an unreadable "
                    "snapshots dir — the resume point may be wrong", j.id);
    // coast continues a loaded state's step count, so a first launch from
    // load_state starts at load_state_step (0 for a fresh job), not at 0.
    const std::uint64_t base = resume ? resume->step : j.load_state_step;
    // Run length is EXPLICIT in coast: `-n <count>` with count >= 1, or --unlimited.
    // There is no sentinel — `-n 0` is a parse error, not "run forever" (it used to
    // be, which is exactly why it was removed). So an unbounded job (steps_target
    // == 0) must say --unlimited, and a bounded one must never compute a zero
    // budget. The core's spawn dispatcher already guarantees it never launches a
    // bounded job whose on-disk checkpoint has reached target, so reaching
    // remaining == 0 here is a logic error; clamp to 1 (a single trailing step that
    // checkpoints and exits) and log loudly rather than emit a command line coast
    // rejects — which would fail the launch, not run forever.
    const bool unbounded = (j.steps_target == 0);
    std::uint64_t remaining = j.steps_target > base ? j.steps_target - base : 0;
    if (!unbounded && remaining == 0) {
        COAST_ERROR("coastd.cmd",
                    "job {} reached plan_launch already at/past target ({} >= {}); "
                    "clamping -n to 1", j.id, base, j.steps_target);
        remaining = 1;
    }

    p.args = {
        "--headless",
        "-c", run_config_path(j).string(),
    };
    if (unbounded) {
        // --unlimited normally REFUSES --save-state/--emit-run-manifest (an
        // unlimited run ends only by being killed, so they would be lost). The
        // --save-on-term below is what makes them honest and lifts that refusal:
        // the daemon stops a child with SIGTERM, which exits through the normal
        // path. Dropping --save-on-term here would turn every unbounded launch
        // into a parse error.
        p.args.emplace_back("--unlimited");
    } else {
        p.args.emplace_back("-n");
        p.args.emplace_back(std::to_string(remaining));
    }
    p.args.insert(p.args.end(), {
        "--checkpoint", std::to_string(cfg.checkpoint_interval),
        "--save-on-term",  // graceful daemon shutdown -> final checkpoint, then exit
        "--save-state", final_state(j).string(),
        "--emit-run-manifest", manifest_path(j).string(),
    });
    // Pin output placement AND shadow any ambient $COAST_APP_CONFIG / ./coast.json by
    // handing coast the daemon-owned app_config explicitly (highest discovery
    // precedence). Absolute; the child chdirs to its run dir before reading it.
    if (!cfg.app_config_path.empty()) {
        p.args.emplace_back("--app-config");
        p.args.emplace_back(cfg.app_config_path.string());
    }
    // The job's own checkpoint wins; the submitted initial state is for the first
    // launch only (the core refuses that launch if the file has gone since submit).
    if (resume) {
        p.args.emplace_back("--load-state");
        p.args.emplace_back(resume->path.string());
    } else if (!j.load_state.empty()) {
        p.args.emplace_back("--load-state");
        p.args.emplace_back(j.load_state);
    }
    return p;
}

std::expected<initial_state, std::string>
initial_state_from(const std::string& id, const job* src, bucket b) {
    const std::string what = "load_state_from '" + id + "'";
    if (id.empty()) return std::unexpected(what + " is empty");
    if (!src) return std::unexpected(what + " names no known job");
    if (b != bucket::done)
        return std::unexpected(what + " is in " + std::string(bucket_dir(b)) +
                               "/, not done/ (only a finished job has a final state)");
    std::error_code ec;
    const fs::path canon = fs::canonical(final_state(*src), ec);
    if (ec || !fs::is_regular_file(canon, ec))
        return std::unexpected(what + " has no final_state.h5 (" + final_state(*src).string() + ")");
    return initial_state{canon, src->steps_done};
}

child_outcome read_outcome(const job& j) {
    child_outcome out;
    const auto ckpt = newest_checkpoint(j);
    if (ckpt) {
        // fs::relative touches the filesystem (weakly_canonical), so the argless
        // overload throws on a stale mount — out of a function the supervisor calls
        // immediately after reaping a child. Degrade instead: this field is
        // REPORTING only (protocol status, GUI, job file); the resume path derives
        // its own checkpoint via newest_checkpoint, so an absolute fallback here
        // costs display tidiness and nothing else.
        std::error_code rec;
        const auto rel = fs::relative(ckpt->path, run_dir_of(j), rec);
        out.checkpoint = rec ? ckpt->path.generic_string() : rel.generic_string();
        out.steps_done = ckpt->step;   // resumable point; overridden by a clean manifest below
    }

    // A clean exit writes the manifest; a SIGKILL does not. When present it is
    // authoritative for completed steps (the checkpoint may lag the final step).
    std::ifstream in(manifest_path(j));
    if (in) {
        try {
            nlohmann::json m;
            in >> m;
            // Extract into locals FIRST, commit only once the WHOLE manifest has
            // parsed. Otherwise a partially-malformed one (a wrong-typed field
            // throws mid-extraction) leaves steps_done already overwritten from a
            // document we then decided not to trust — skewing the resume point and
            // masking the failure behind a warning.
            const std::uint64_t steps = m.value("n_steps_completed", out.steps_done);
            const bool fatal          = m.value("exit_reason", std::string{}) == "fatal_error";
            out.steps_done = steps;
            out.fatal      = fatal;
        } catch (const std::exception& e) {
            COAST_WARN("coastd.cmd", "manifest {} unreadable: {}", manifest_path(j).string(), e.what());
        }
    }
    return out;
}

} // namespace coastd
