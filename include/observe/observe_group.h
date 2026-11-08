//
// Created for the COAST observe system
//

#ifndef COAST_OBSERVE_GROUP_H
#define COAST_OBSERVE_GROUP_H

#include "observe/observe_trigger.h"
#include "observe/observe_site.h"

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace observe {

class observe_writer_t;  // owned by value via unique_ptr below

/// Minimal spinlock guarding a group's `sites` map. Chosen over a mutex because
/// contention is near-zero and critical sections are a single map find/insert:
///   - the sim thread takes it only while inserting a newly-discovered site
///     (rare — after warmup every emit() resolves via the registry's site_cache_
///     and never touches this map again),
///   - the GUI thread takes it briefly, at panel cadence, to read site_count().
/// A busy spin is therefore cheaper and simpler than a mutex here.
class site_spin_lock {
public:
    void lock()   noexcept { while (flag_.test_and_set(std::memory_order_acquire)) { /* spin */ } }
    void unlock() noexcept { flag_.clear(std::memory_order_release); }
private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

using site_map_t = std::unordered_map<std::string, std::unique_ptr<observe_site_t>>;

/// One observe group: a named bundle of sites sharing trigger configuration
/// and a single output HDF5 file.
///
/// Threading model (much simpler than the old inline-OBSERVE machinery):
///   - The observation pass runs single-threaded on the sim thread, device
///     idle, after every entity's process(). It is the only writer of
///     `active_this_step` and the only reader of `ranges`/`combine`.
///   - `ranges`/`combine` may be mutated by the GUI under the registry mutex
///     (`with_group_locked`); the pass reads them under the same mutex while
///     computing `active_this_step` at the start of each pass.
///   - `prog_active`/`prog_single_pending` are driven by `observe_accessor::
///     trigger()` from within perform_observations (sim thread). Kept atomic
///     so a future GUI-thread trigger stays well-defined.
///   - `sites` is a plain std::unordered_map guarded by `sites_lock`. The sim
///     thread inserts under the lock (only on first discovery of each site); the
///     GUI panel reads the count under the same lock via site_count(). All
///     access goes through the lock — there is no implicit-concurrency map.
struct observe_group_t {
    std::string                       group_id;

    // Configuration / triggers (read by the pass under the registry mutex):
    std::vector<step_range_t>         ranges;
    combine_mode_t                    combine{combine_mode_t::OR_};

    // Programmatic trigger inputs, driven via observe_accessor::trigger():
    //   START          -> prog_active = true
    //   STOP           -> prog_active = false
    //   SINGLE_CAPTURE -> prog_single_pending = true (consumed end-of-pass)
    std::atomic<bool>                 prog_active{false};
    std::atomic<bool>                 prog_single_pending{false};

    // Whether this group commits this step. Recomputed at the start of every
    // pass from (ranges, combine, prog_active, prog_single_pending), and
    // re-evaluated live by trigger() so a following observe() in the same pass
    // sees the change. Atomic because the GUI panel may trigger() from the main
    // thread while the pass reads it on the sim thread.
    std::atomic<bool>                 active_this_step{false};

    // Sites discovered for this group. Guarded by sites_lock (see threading
    // notes above). Insert only on the sim thread via resolve_site; read the
    // count on the GUI thread via site_count(). observe_site_t objects are
    // heap-stable (held by unique_ptr), so a raw pointer returned from an insert
    // stays valid across later inserts regardless of rehash.
    site_map_t                        sites;
    mutable site_spin_lock            sites_lock;

    /// Thread-safe site count for the GUI panel. Takes sites_lock so it never
    /// races a concurrent sim-thread insert.
    [[nodiscard]] std::size_t site_count() const {
        sites_lock.lock();
        const std::size_t n = sites.size();
        sites_lock.unlock();
        return n;
    }

    // Writer (lazy; opened on first commit):
    std::unique_ptr<observe_writer_t> writer;
    std::filesystem::path             h5_path;
};

} // namespace observe

#endif // COAST_OBSERVE_GROUP_H
