//
// The run manifest (--emit-run-manifest, sim_core/run_manifest.h) of a
// headless run: what is known at the start is collected on construction, the
// run adds what it learns (state files written, why it ended), and the
// destructor writes the file exactly once, on whatever path leaves
// run_headless — clean completion, early setup failure, or a caught fatal.
//
// The emit point must run while the observe registry is still attached:
// sim_instance teardown detaches and clears it after run_headless returns, so
// the object lives in run_headless's frame, not in main().
//
// Inactive when --emit-run-manifest is not given: every call is then a no-op.
//

#ifndef COAST_APP_RUN_PROVENANCE_H
#define COAST_APP_RUN_PROVENANCE_H

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "sim_core/run_manifest.h"

namespace coast     { class sim_instance; }
namespace coast::cli { struct options; }

namespace coast::app {

class run_provenance {
public:
    // Collects the run facts and each stochastic entity's effective seed. Must
    // run after build_env (the entities exist) and before the run starts.
    run_provenance(sim_instance& sim, const cli::options& opts);
    ~run_provenance();   // writes the manifest; never throws

    run_provenance(const run_provenance&)            = delete;
    run_provenance& operator=(const run_provenance&) = delete;

    // A state file written at `step` (--save-state or a checkpoint). The RAW
    // absolute path is captured now, while the CWD is the run's;
    // write_run_manifest decides the recorded frame.
    void record_savestate(const std::filesystem::path& p, std::uint64_t step);

    // Defaults to fatal_error, so a path that never sets it reads as a crash.
    void set_exit_reason(sim::exit_reason_t r) { m_.exit_reason = r; }

private:
    sim_instance&                         sim_;
    sim::run_manifest_inputs_t            m_;
    std::chrono::steady_clock::time_point t0_;
};

} // namespace coast::app

#endif // COAST_APP_RUN_PROVENANCE_H
