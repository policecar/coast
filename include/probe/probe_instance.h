//
// Created for the COAST probe system
//

#ifndef COAST_PROBE_INSTANCE_H
#define COAST_PROBE_INSTANCE_H

#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <atomic>
#include "spsc_queue.h"
#include "probe_observer.h"

namespace core {
class entity_element;
struct structure_entry;
}

namespace probe {

/// Represents a registered probe at a specific (element, probe_name) location.
/// Each instance has exactly one producer thread (SPSC model).
struct probe_instance {
    const core::entity_element* element;
    const core::structure_entry* structure;
    std::string probe_name;
    std::string full_path;  // Cached: e.g., "neuron[42]/dendrite[3]/activation"

    // Activation state - checked on every probe() call
    std::atomic<bool> active{false};

    // Decimation: record every Nth call (1 = every call)
    std::atomic<std::uint32_t> decimation{1};
    std::atomic<std::uint32_t> call_count{0};

    // Value queue: SPSC (one sim thread produces, GUI thread consumes)
    spsc_queue<double> values;

    // Observers - only accessed from GUI thread
    std::vector<std::unique_ptr<probe_observer>> observers;

    probe_instance(const core::entity_element* elem,
                   const core::structure_entry* entry,
                   std::string_view name);

    /// Record a value (called from simulation thread).
    /// Handles decimation internally.
    void record(double value);

    /// Drain values to observers (called from GUI thread).
    void update_observers();

    /// Activate this probe (increments element's active counter).
    void activate();

    /// Deactivate and destroy all observers.
    void deactivate();

    /// Add an observer (GUI thread only).
    void add_observer(std::unique_ptr<probe_observer> observer);

    /// Remove all observers (GUI thread only).
    void clear_observers();
};

} // namespace probe

#endif // COAST_PROBE_INSTANCE_H
