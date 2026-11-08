//
// Created for the COAST probe system
//

#ifndef COAST_STRUCTURE_REGISTRY_H
#define COAST_STRUCTURE_REGISTRY_H

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "core/element_type.h"

namespace sim {
struct io_entity;
}

namespace core {

class entity_element;

/// Entry in the structure tree, representing one entity_element instance.
struct structure_entry {
    element_type type;
    std::size_t index;                           // Index within parent's child list
    entity_element* element;
    structure_entry* parent;
    std::vector<std::string> probe_names;        // Cached from probe_points()
    std::vector<std::unique_ptr<structure_entry>> children;

    // Pre-computed full path (e.g., "neuron[42]/dendrite[3]")
    // Computed once during build(), thread-safe to read afterwards.
    std::string cached_path;

    structure_entry(element_type t, std::size_t idx, entity_element* elem, structure_entry* par)
        : type(t), index(idx), element(elem), parent(par) {}

    /// Get the full path string for this entry.
    [[nodiscard]] const std::string& path() const { return cached_path; }

    /// Compute and set the path (called during structure building).
    void compute_path();

    /// Get the full path including a probe name (e.g., "neuron[42]/dendrite[3]/activation").
    [[nodiscard]] std::string full_probe_path(std::string_view probe_name) const;
};

/// Registry that maps entity_element pointers to their structural paths.
/// Built once during io_entity initialization, provides O(1) lookup.
class structure_registry {
    std::unique_ptr<structure_entry> root_;
    std::unordered_map<const entity_element*, structure_entry*> element_map_;

public:
    structure_registry() = default;
    ~structure_registry() = default;

    // Non-copyable
    structure_registry(const structure_registry&) = delete;
    structure_registry& operator=(const structure_registry&) = delete;

    // Movable
    structure_registry(structure_registry&&) = default;
    structure_registry& operator=(structure_registry&&) = default;

    /// Build the structure tree recursively from the root io_entity.
    /// Sets root_entity_ on all entity_elements in the tree.
    void build(sim::io_entity* root);

    /// O(1) lookup from element pointer to structure entry.
    [[nodiscard]] structure_entry* find(const entity_element* element);
    [[nodiscard]] const structure_entry* find(const entity_element* element) const;

    /// Get the root entry for GUI traversal.
    [[nodiscard]] structure_entry* root() { return root_.get(); }
    [[nodiscard]] const structure_entry* root() const { return root_.get(); }

    /// Check if the registry has been built.
    [[nodiscard]] bool empty() const { return root_ == nullptr; }

private:
    void build_recursive(structure_entry* parent_entry, entity_element* element,
                         sim::io_entity* root_entity);
};

} // namespace core

#endif // COAST_STRUCTURE_REGISTRY_H
