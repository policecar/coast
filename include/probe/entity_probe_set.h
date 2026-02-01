//
// Created for the COAST probe system
//

#ifndef COAST_ENTITY_PROBE_SET_H
#define COAST_ENTITY_PROBE_SET_H

#include <string_view>
#include <vector>
#include <memory>
#include <cstdint>
#include <tbb/concurrent_hash_map.h>
#include "probe_instance.h"

namespace sim {
class io_entity;
}

namespace core {
class entity_element;
class structure_registry;
}

namespace probe {

/// Per-entity probe management with thread-safe registration.
/// Handles concurrent probe registration from TBB threads.
class entity_probe_set {
    sim::io_entity* owner_;
    core::structure_registry* structure_;

    // Thread-safe map for concurrent probe registration from TBB threads
    // Key: hash of (element pointer, probe name)
    tbb::concurrent_hash_map<std::size_t, std::unique_ptr<probe_instance>> instances_;

public:
    entity_probe_set();
    entity_probe_set(sim::io_entity* owner, core::structure_registry* structure);

    // Non-copyable
    entity_probe_set(const entity_probe_set&) = delete;
    entity_probe_set& operator=(const entity_probe_set&) = delete;

    // Movable
    entity_probe_set(entity_probe_set&&) = default;
    entity_probe_set& operator=(entity_probe_set&&) = default;

    /// Initialize with owner and structure (called after construction).
    void init(sim::io_entity* owner, core::structure_registry* structure);

    /// Record a probe value (called from entity_element::probe via sim threads).
    void record(const core::entity_element* element, std::string_view name, double value);

    /// Get all registered probe instances (for GUI discovery).
    [[nodiscard]] std::vector<probe_instance*> all_instances();

    /// Activate a probe with optional decimation (GUI thread).
    probe_instance* activate(const core::entity_element* element,
                             std::string_view name,
                             std::uint32_t decimation = 1);

    /// Deactivate a probe (GUI thread).
    void deactivate(probe_instance* instance);

    /// Set decimation rate for an active probe.
    static void set_decimation(probe_instance* instance, std::uint32_t decimation);

    /// Update all active observers (called from GUI thread each frame).
    void update_observers();

private:
    static std::size_t make_key(const core::entity_element* element, std::string_view name);
    probe_instance* find_instance(const core::entity_element* element, std::string_view name);
    probe_instance* get_or_create(const core::entity_element* element, std::string_view name);
};

} // namespace probe

#endif // COAST_ENTITY_PROBE_SET_H
