//
// Created for the COAST probe system
//

#ifndef COAST_ENTITY_ELEMENT_H
#define COAST_ENTITY_ELEMENT_H

#include <atomic>
#include <vector>
#include <string_view>
#include <typeinfo>
#include "element_type.h"

namespace sim {
struct io_entity;  // Forward declaration
}

namespace probe {
class entity_probe_set;
struct probe_instance;
}

namespace core {

class structure_registry;  // Forward declaration

/// Base class for all structural elements in the entity hierarchy.
/// Provides type identification, hierarchical navigation, and probing support.
class entity_element {
protected:
    sim::io_entity* root_entity_{nullptr};

    // Probe fast-path: count of active probes on this element.
    // Mutable because it's probe infrastructure, not simulation state.
    mutable std::atomic<std::uint32_t> active_probe_count_{0};

    friend class structure_registry;  // Sets root_entity_ during structure building
    friend class probe::entity_probe_set;  // Manages active_probe_count_
    friend struct probe::probe_instance;  // Decrements active_probe_count_ on deactivate

public:
    entity_element() = default;
    virtual ~entity_element() = default;

    // Non-copyable
    entity_element(const entity_element&) = delete;
    entity_element& operator=(const entity_element&) = delete;

    // Movable (atomic counter is copied, which is fine for probe metadata)
    entity_element(entity_element&& other) noexcept
        : root_entity_(other.root_entity_)
        , active_probe_count_(other.active_probe_count_.load(std::memory_order_relaxed))
    {
        other.root_entity_ = nullptr;
    }

    entity_element& operator=(entity_element&& other) noexcept {
        if (this != &other) {
            root_entity_ = other.root_entity_;
            active_probe_count_.store(
                other.active_probe_count_.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
            other.root_entity_ = nullptr;
        }
        return *this;
    }

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

    /// Record a probe value. Only records if any probe on this element is active.
    /// Inline fast-path: single atomic load when no probes are active.
    void probe(std::string_view name, double value) {
        if (active_probe_count_.load(std::memory_order_relaxed) == 0) return;
        probe_slow_path(name, value);
    }

    // -- Root entity access --

    /// Get the root io_entity that owns the probing infrastructure.
    [[nodiscard]] sim::io_entity* root_entity() { return root_entity_; }
    [[nodiscard]] const sim::io_entity* root_entity() const { return root_entity_; }

protected:
    /// Slow path for probe recording (hash map lookup).
    void probe_slow_path(std::string_view name, double value);
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
