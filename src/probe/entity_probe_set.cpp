//
// Created for the COAST probe system
//

#include "probe/entity_probe_set.h"
#include "core/entity_element.h"
#include "core/structure_registry.h"
#include <functional>

namespace probe {

entity_probe_set::entity_probe_set()
    : owner_(nullptr), structure_(nullptr) {}

entity_probe_set::entity_probe_set(sim::io_entity* owner, core::structure_registry* structure)
    : owner_(owner), structure_(structure) {}

void entity_probe_set::init(sim::io_entity* owner, core::structure_registry* structure) {
    owner_ = owner;
    structure_ = structure;
}

std::size_t entity_probe_set::make_key(const core::entity_element* element,
                                        std::string_view name) {
    std::size_t h1 = std::hash<const void*>{}(element);
    std::size_t h2 = std::hash<std::string_view>{}(name);
    return h1 ^ (h2 << 1);  // Simple hash combine
}

void entity_probe_set::record(const core::entity_element* element,
                               std::string_view name,
                               double value) {
    std::size_t key = make_key(element, name);

    // Fast path: lookup existing instance (read-only access)
    typename decltype(instances_)::const_accessor reader;
    if (instances_.find(reader, key)) {
        auto* instance = reader->second.get();

        // Check if active (single relaxed load - very fast when inactive)
        if (!instance->active.load(std::memory_order_relaxed)) {
            return;
        }

        // Record value (handles decimation internally)
        instance->record(value);
        return;
    }

    // Slow path: first time seeing this probe - register for discovery
    typename decltype(instances_)::accessor writer;
    if (instances_.insert(writer, key)) {
        const core::structure_entry* entry = structure_ ? structure_->find(element) : nullptr;
        writer->second = std::make_unique<probe_instance>(element, entry, name);
    }
    // Don't record on first registration - just make it discoverable
}

std::vector<probe_instance*> entity_probe_set::all_instances() {
    std::vector<probe_instance*> result;
    for (auto& pair : instances_) {
        result.push_back(pair.second.get());
    }
    return result;
}

probe_instance* entity_probe_set::find_instance(const core::entity_element* element,
                                                  std::string_view name) {
    std::size_t key = make_key(element, name);
    typename decltype(instances_)::const_accessor reader;
    if (instances_.find(reader, key)) {
        return reader->second.get();
    }
    return nullptr;
}

probe_instance* entity_probe_set::get_or_create(const core::entity_element* element,
                                                  std::string_view name) {
    std::size_t key = make_key(element, name);
    typename decltype(instances_)::accessor writer;
    if (!instances_.find(writer, key)) {
        if (instances_.insert(writer, key)) {
            const core::structure_entry* entry = structure_ ? structure_->find(element) : nullptr;
            writer->second = std::make_unique<probe_instance>(element, entry, name);
        }
    }
    return writer->second.get();
}

probe_instance* entity_probe_set::discover(const core::entity_element* element,
                                            std::string_view name) {
    return get_or_create(element, name);
}

probe_instance* entity_probe_set::activate(const core::entity_element* element,
                                            std::string_view name,
                                            std::uint32_t decimation) {
    auto* instance = get_or_create(element, name);
    instance->decimation.store(decimation, std::memory_order_relaxed);
    instance->call_count.store(0, std::memory_order_relaxed);
    instance->activate();
    return instance;
}

void entity_probe_set::deactivate(probe_instance* instance) {
    if (instance) {
        instance->deactivate();
    }
}

void entity_probe_set::set_decimation(probe_instance* instance, std::uint32_t decimation) {
    if (instance) {
        instance->decimation.store(decimation, std::memory_order_relaxed);
        instance->call_count.store(0, std::memory_order_relaxed);
    }
}

void entity_probe_set::update_observers() {
    for (auto& pair : instances_) {
        if (pair.second->active.load(std::memory_order_relaxed)) {
            pair.second->update_observers();
        }
    }
}

} // namespace probe
