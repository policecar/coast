//
// Created for the COAST probe system
//

#ifndef COAST_ENTITY_ELEMENT_H
#define COAST_ENTITY_ELEMENT_H

#include <vector>
#include <string_view>
#include <typeinfo>
#include "element_type.h"

namespace sim {
struct io_entity;  // Forward declaration
}

namespace core {

class structure_registry;  // Forward declaration

/// Base class for all structural elements in the entity hierarchy.
/// Provides type identification, hierarchical navigation, and probing support.
class entity_element {
protected:
    sim::io_entity* root_entity_{nullptr};

    friend class structure_registry;  // Sets root_entity_ during structure building

public:
    virtual ~entity_element() = default;

    // -- Type identification --

    /// Returns the type identifier for this element.
    /// Default implementation returns a generic "unknown" type.
    /// Subclasses should override using ELEMENT_TYPE macro.
    [[nodiscard]] virtual element_type type_id() const {
        return element_type::of<entity_element>("unknown");
    }

    // -- Child structure --

    /// Describes a group of children of the same type.
    struct child_group {
        element_type type;
        std::size_t count;
    };

    /// Returns all child groups (types and counts).
    [[nodiscard]] virtual std::vector<child_group> children() const { return {}; }

    /// Type-safe child access by index.
    template<typename T>
    T* child(std::size_t index) {
        return static_cast<T*>(child_impl(typeid(T), index));
    }

    template<typename T>
    [[nodiscard]] const T* child(std::size_t index) const {
        return static_cast<const T*>(child_impl(typeid(T), index));
    }

    /// Get count of children of a specific type.
    template<typename T>
    [[nodiscard]] std::size_t child_count() const {
        return child_count_impl(typeid(T));
    }

    // -- Probing --

    /// Returns the list of available probe point names on this element.
    [[nodiscard]] virtual std::vector<std::string_view> probe_points() const { return {}; }

    /// Record a probe value. Only records if the probe is active.
    void probe(std::string_view name, double value);

    // -- Root entity access --

    /// Get the root io_entity that owns the probing infrastructure.
    [[nodiscard]] sim::io_entity* root_entity() { return root_entity_; }
    [[nodiscard]] const sim::io_entity* root_entity() const { return root_entity_; }

protected:
    /// Implementation hook for type-erased child access.
    [[nodiscard]] virtual entity_element* child_impl(const std::type_info& type, std::size_t index) {
        (void)type; (void)index;
        return nullptr;
    }

    [[nodiscard]] virtual const entity_element* child_impl(const std::type_info& type, std::size_t index) const {
        (void)type; (void)index;
        return nullptr;
    }

    /// Implementation hook for type-erased child count.
    [[nodiscard]] virtual std::size_t child_count_impl(const std::type_info& type) const {
        (void)type;
        return 0;
    }
};

} // namespace core

#endif // COAST_ENTITY_ELEMENT_H
