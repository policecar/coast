//
// Created for the COAST observe system
//

#ifndef COAST_OBSERVE_WRITER_H
#define COAST_OBSERVE_WRITER_H

#include "observe/observe_site.h"
#include "hdf5/handle.h"

#include <hdf5.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace hdf5 { class executor; }

namespace observe {

/// The run a writer writes for (SPEC.md S.5.14). Every file records `run_id`
/// (/metadata@run_id). When the run continues from a checkpoint (`from_step`, the
/// checkpoint's step count), an existing file of the same run is continued, not
/// recreated: its rows for steps from `from_step` on are dropped (those steps run
/// again) and the new rows are appended after the rest. Otherwise — no
/// continuation, no file, another run's file, or one that cannot be read — the
/// file is created afresh, as before, and the reason is logged.
struct observe_run_t {
    std::string                  run_id;
    std::optional<std::uint64_t> from_step;
};

/// Per-group HDF5 writer. Owns the group's HDF5 file handle and the
/// per-site dataset handles, all opened lazily on first commit. All
/// HDF5 work is submitted to the process-wide hdf5::executor whose
/// single worker thread is the one thread allowed to touch libhdf5 (the
/// linked build has Threadsafety: OFF — see design/hdf5-executor.md).
///
/// No worker thread, mutex, condvar or queue are owned here: the
/// executor provides all of that, shared across observe groups and
/// every other HDF5 user in the process.
class observe_writer_t {
public:
    observe_writer_t(std::filesystem::path h5_path,
                     hdf5::executor&     h5_exec,
                     observe_run_t       run = {});
    ~observe_writer_t();

    observe_writer_t(const observe_writer_t&)            = delete;
    observe_writer_t& operator=(const observe_writer_t&) = delete;

    /// Open the HDF5 file (submit_sync to the executor). Idempotent: a
    /// second call after a successful start is a no-op.
    void start();

    /// Flush pending writes, close all dataset handles, and close the
    /// HDF5 file — in that order, so children close before their parent.
    /// Idempotent.
    void stop_and_join();

    /// Sim-thread entry: copies `bytes` from the caller-supplied `value`
    /// pointer into an owned buffer and submits an append job to the
    /// executor. `bytes` must equal `site->value_bytes`. Blocks the caller
    /// only when the executor's queue is full (back-pressure).
    void enqueue(observe_site_t* site, const void* value,
                 std::size_t bytes, std::uint64_t step);

private:
    std::filesystem::path             h5_path_;
    hdf5::executor&                 h5_exec_;
    observe_run_t                     run_;
    hdf5::file_h                    h5_file_;
    std::atomic<bool>                 started_{false};
    std::atomic<bool>                 stopped_{false};

    // Executor thread: reopen `site`'s datasets in a continued file; false if the
    // file has none (or they do not fit the site, which is logged).
    bool open_existing(observe_site_t* site);

    // Sites whose datasets we have opened, for orderly shutdown. Only
    // touched on the executor's worker thread, so no synchronisation
    // is needed.
    std::vector<observe_site_t*>      opened_sites_;
};

} // namespace observe

#endif // COAST_OBSERVE_WRITER_H
