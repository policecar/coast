//
// Created for the COAST probe system
//

#include "probe/probe_instance.h"
#include "core/structure_registry.h"

namespace probe {

probe_instance::probe_instance(const core::entity_element* elem,
                               const core::structure_entry* entry,
                               std::string_view name)
    : element(elem)
    , structure(entry)
    , probe_name(name)
    , full_path(entry ? entry->full_probe_path(name) : std::string(name))
{}

void probe_instance::record(double value) {
    // Decimation check
    std::uint32_t dec = decimation.load(std::memory_order_relaxed);
    if (dec > 1) {
        std::uint32_t count = call_count.fetch_add(1, std::memory_order_relaxed);
        if (count % dec != 0) {
            return;
        }
    }
    values.try_push(value);
}

void probe_instance::update_observers() {
    values.drain([this](double value) {
        for (auto& obs : observers) {
            obs->push(value);
        }
    });
}

void probe_instance::deactivate() {
    active.store(false, std::memory_order_relaxed);
    observers.clear();
    values.clear();
    call_count.store(0, std::memory_order_relaxed);
}

void probe_instance::add_observer(std::unique_ptr<probe_observer> observer) {
    observers.push_back(std::move(observer));
}

void probe_instance::clear_observers() {
    observers.clear();
}

} // namespace probe
