#ifndef COAST_COASTD_CHILD_CMD_H
#define COAST_COASTD_CHILD_CMD_H

// -----------------------------------------------------------------------------
// coastd::child_cmd — the coast<->coastd command contract for one job.
//
// coastd owns output locations by construction: it writes the job's config with
// snapshot_dir set to an ABSOLUTE <run_dir>/snapshots (the path resolver's
// "absolute leaf wins verbatim" rule), and passes --save-state / --load-state /
// --emit-run-manifest as absolute paths. So every output lands under the run dir
// regardless of the child's CWD, and coastd knows exactly where to find
// checkpoints — no guessing, no re-deriving coast's resolution.
//
// Run-dir layout (all under <runs_root>/<id>):
//   config.json            normalized sim config (snapshot_dir -> abs snapshots/)
//   snapshots/*.h5         checkpoints  (<stem>_<UTC>_<step>.h5; coastd owns identity by step)
//   logs/run_<n>.log       one per launch/resume; the tailed progress feed
//   coast_runtime.json     run manifest (clean-exit only)
//   final_state.h5         --save-state target (definitive end state)
// -----------------------------------------------------------------------------

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "daemon/types.h"

namespace coastd {

struct checkpoint_info {
    std::filesystem::path path;
    std::uint64_t         step = 0;
};

struct launch_plan {
    std::string              exe;      // the built coast binary
    std::vector<std::string> args;     // argv after exe
    std::filesystem::path    logfile;  // stdout+stderr sink (also the progress feed)
    std::filesystem::path    cwd;      // the run dir
    std::filesystem::path    manifest; // coast_runtime.json; deleted before launch so
                                       // "manifest present" == THIS launch exited cleanly
};

struct child_outcome {
    std::uint64_t steps_done = 0;      // manifest completed, else newest checkpoint step
    std::string   checkpoint;          // newest checkpoint basename rel run_dir, or ""
    bool          fatal = false;       // manifest exit_reason == fatal_error (give up, don't resume)
};

// Run-dir path helpers.
[[nodiscard]] std::filesystem::path run_snapshots_dir(const job& j);
[[nodiscard]] std::filesystem::path run_config_path(const job& j);
// The submission archived verbatim. NOT the config that runs — see below.
[[nodiscard]] std::filesystem::path run_submitted_config_path(const job& j);

// The target every run dir's `data` link gets for this dataset_base: absolute
// (a relative target would resolve against the run dir), against the daemon's
// working directory; verbatim if that fails. machine_info reports this same path,
// so a client can compare it with a run dir's link. "" for an unset dataset_base.
[[nodiscard]] std::filesystem::path dataset_link_target(const std::filesystem::path& dataset_base);

// Write `config_text` (the exact config bytes validated at the connection edge)
// into the run dir, producing TWO files:
//   config.json           — what coast actually runs: the parsed config with
//                           snapshot_dir rewritten to the absolute <run_dir>/
//                           snapshots, re-serialized (so key order/formatting are
//                           nlohmann's, not the submitter's).
//   config.submitted.json — the submitted bytes, verbatim, as the provenance
//                           record. Advisory: nothing reads it, and a failure to
//                           write it does not fail the submission.
// Creates run_dir + snapshots/ + logs/. Takes the bytes, not a path, so the run
// config is written from the SAME bytes that were path-scanned — nothing is ever
// re-read between validation and use (there is no submit-config TOCTOU).
//
// When dataset_base is non-empty it is also exposed as a `data` symlink inside the
// run dir (absolutized first), so a job config's conventional relative dataset
// path resolves from the child's CWD; empty leaves that off. Linking is
// best-effort and idempotent — a failure or a pre-existing link never fails the
// submission.
// Returns false on parse/write failure of config.json (a submission-time error).
[[nodiscard]] bool prepare_run_config(const std::string& config_text, const job& j,
                                      const std::filesystem::path& dataset_base = {});

// Newest checkpoint under <run_dir>/snapshots by trailing _<step>.h5, or nullopt.
//
// nullopt is AMBIGUOUS on its own: "this run has not checkpointed yet" and "the
// snapshots directory could not be read" look identical, and they call for
// opposite decisions — the first is a fresh run, the second must not be treated
// as one (a job nine hours into a ten-hour run would restart from zero). Pass
// scan_failed to tell them apart; it is set only when the directory EXISTS but
// could not be scanned, never for the ordinary not-yet-created case.
[[nodiscard]] std::optional<checkpoint_info> newest_checkpoint(const job& j,
                                                               bool* scan_failed = nullptr);

// Build the coast launch plan. If a checkpoint exists it is a RESUME
// (--load-state newest, -n = remaining); otherwise a first launch, which starts
// from j.load_state when set (-n = steps_target - load_state_step).
[[nodiscard]] launch_plan plan_launch(const job& j, const daemon_config& cfg,
                                      const std::filesystem::path& coast_bin);

// The initial state a submit's `load_state_from` names: the source job's
// <run_dir>/final_state.h5 (canonical) and the step it holds.
struct initial_state {
    std::filesystem::path path;
    std::uint64_t         step = 0;
};

// Resolve `load_state_from` = `id` against the job it names (`src` with its
// bucket; nullptr when no such job exists). Only a job in done/ qualifies, and its
// final_state.h5 must be a regular file. The step is the source's steps_done: for
// a done job that is its manifest's n_steps_completed, and the final state was
// saved at that same step (both are coast's step count at exit). Returns the
// rejection message otherwise. Pure apart from the file check.
[[nodiscard]] std::expected<initial_state, std::string>
initial_state_from(const std::string& id, const job* src, bucket b);

// Read a finished child's outcome from the run dir: manifest n_steps_completed if
// present (clean exit), else the newest checkpoint step (interrupted — a SIGKILL
// leaves no manifest). checkpoint is the newest checkpoint, the resume payload.
[[nodiscard]] child_outcome read_outcome(const job& j);

} // namespace coastd

#endif // COAST_COASTD_CHILD_CMD_H
