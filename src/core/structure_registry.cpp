//
// Created for the COAST probe system
//

#include "core/structure_registry.h"
#include "core/entity_element.h"
#include "sim_core/io_entity.h"

namespace core {

// -- structure_entry --

const std::string& structure_entry::path() const {
    if (!path_valid) {
        if (parent == nullptr) {
            // Root element: just the type name
            cached_path = type.name();
        } else {
            // Build path: parent_path/type[index]
            const std::string& parent_path = parent->path();
            cached_path = parent_path + "/" + type.name() + "[" + std::to_string(index) + "]";
        }
        path_valid = true;
    }
    return cached_path;
}

std::string structure_entry::full_probe_path(std::string_view probe_name) const {
    return path() + "/" + std::string(probe_name);
}

// -- structure_registry --

void structure_registry::build(sim::io_entity* root) {
    if (!root) return;

    // Clear any existing structure
    element_map_.clear();
    root_.reset();

    // Create root entry - cast io_entity to entity_element
    auto* root_element = static_cast<entity_element*>(root);
    root_ = std::make_unique<structure_entry>(
        root_element->type_id(),
        0,
        root_element,
        nullptr
    );

    // Register root in the map
    element_map_[root_element] = root_.get();

    // Set root_entity on the root element itself
    root_element->root_entity_ = root;

    // Cache probe names
    for (auto pname : root_element->probe_points()) {
        root_->probe_names.emplace_back(pname);
    }

    // Build recursively for all children
    build_recursive(root_.get(), root_element, root);
}

void structure_registry::build_recursive(structure_entry* parent_entry,
                                          entity_element* element,
                                          sim::io_entity* root_entity) {
    // Iterate over all child groups
    for (const auto& group : element->children()) {
        for (std::size_t i = 0; i < group.count; ++i) {
            // Get the child element (non-const access)
            auto* child = element->child_impl(group.type.info(), i);
            if (!child) continue;

            // Create entry for this child
            auto child_entry = std::make_unique<structure_entry>(
                group.type,
                i,
                child,
                parent_entry
            );

            // Register in the map
            element_map_[child] = child_entry.get();

            // Set root_entity on the child
            child->root_entity_ = root_entity;

            // Cache probe names
            for (auto pname : child->probe_points()) {
                child_entry->probe_names.emplace_back(pname);
            }

            // Recurse into grandchildren
            build_recursive(child_entry.get(), child, root_entity);

            // Add to parent's children list
            parent_entry->children.push_back(std::move(child_entry));
        }
    }
}

structure_entry* structure_registry::find(const entity_element* element) {
    auto it = element_map_.find(element);
    return it != element_map_.end() ? it->second : nullptr;
}

const structure_entry* structure_registry::find(const entity_element* element) const {
    auto it = element_map_.find(element);
    return it != element_map_.end() ? it->second : nullptr;
}

} // namespace core
