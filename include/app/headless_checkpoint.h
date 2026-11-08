//
// --checkpoint N for a headless run: the periodic snapshot writer.
//
// The cadence is enforced ON the sim thread by a post_swap hook (on_post_swap),
// not by run_headless's supervisory poll loop. The hook fires once per step at
// the coherent post-swap boundary (read pages flipped, GPU readbacks done — the
// boundary run_quiesced targets), so it tests the exact step count and
// snapshots there: a run that reaches step start + k*N checkpoints at EXACTLY
// that step, independent of SPS or poll timing.
//
// The state extraction (the heavy buffer snapshot) runs inline on the sim
// thread; the HDF5 write is submitted async to the executor so stepping overlaps
// the write. collect() reaps finished writes — run_headless calls it from the
// poll loop, so a failed write is reported while the run is still going (an
// --unlimited run may last days), and once more, waiting, before the manifest
// is written.
//

#ifndef COAST_APP_HEADLESS_CHECKPOINT_H
#define COAST_APP_HEADLESS_CHECKPOINT_H

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <future>
#include <mutex>
#include <string>
#include <vector>

#include "logging/logging_common.h"   // log_result

namespace sim     { class env; class sim_thread_controller; }
namespace logging { class logging_manager; }

namespace coast::app {

class headless_checkpointer {
public:
    struct written_t {
        std::filesystem::path path;
        std::uint64_t         step;
    };

    // interval >= 1. target_step 0 = unlimited (feeds the progress events only).
    headless_checkpointer(std::uint64_t               interval,
                          std::uint64_t               start_step,
                          std::uint64_t               target_step,
                          std::filesystem::path       snapshot_dir,
                          std::string                 config_name,
                          sim::env&                   env,
                          logging::logging_manager&   lm,
                          sim::sim_thread_controller& ctrl);

    [[nodiscard]] std::uint64_t                interval()     const { return interval_; }
    [[nodiscard]] const std::filesystem::path& snapshot_dir() const { return snapshot_dir_; }

    // The post_swap hook body (sim thread). Writes at every interval boundary
    // past start_step. A failed checkpoint is logged and skipped — it must not end
    // the run (a throw leaving the hook would reach do_step()'s backstop and halt
    // it as a fault, #198). Never throws.
    void on_post_swap() noexcept;

    // --save-on-term (caller thread, sim paused): one final checkpoint, extracted
    // THROUGH run_quiesced on the sim thread — extract_state reads io_buffer read
    // pages and the wgpu queue, which the sim thread owns. Skipped when the
    // periodic hook already checkpointed that step (two files for one state, two
    // manifest entries describing it). Logs a failure. Never throws.
    void write_final() noexcept;

    // Reaps the finished writes: each that succeeded is passed to `on_written`,
    // each that failed is logged. wait = false leaves unfinished writes pending;
    // wait = true blocks until every write has finished. Caller thread only.
    void collect(bool wait, const std::function<void(const written_t&)>& on_written);

private:
    // Snapshot at `completed` and submit the write. Sim thread. Throws on a
    // failed extraction or submission.
    void write(std::uint64_t completed);

    struct pending_t {
        std::future<logging::log_result> fut;
        std::filesystem::path            path;
        std::uint64_t                    step;
    };

    const std::uint64_t         interval_;
    const std::uint64_t         start_step_;
    const std::uint64_t         target_step_;
    const std::filesystem::path snapshot_dir_;
    const std::string           config_name_;
    sim::env&                   env_;
    logging::logging_manager&   lm_;
    sim::sim_thread_controller& ctrl_;

    // Highest step a checkpoint has been SUBMITTED for (write_final's dedup).
    // Stamped by write() only once submission succeeded: a step whose write
    // THREW must not count, or the dedup would suppress the final save that
    // replaces it — losing it in exactly the low-disk case it exists for.
    std::atomic<std::uint64_t> last_written_{0};

    std::mutex             mtx_;       // guards pending_ (sim thread pushes, caller reaps)
    std::vector<pending_t> pending_;
};

} // namespace coast::app

#endif // COAST_APP_HEADLESS_CHECKPOINT_H
