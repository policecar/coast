//
// Created for the COAST probe system
//

#ifndef COAST_ELEMENT_TYPE_H
#define COAST_ELEMENT_TYPE_H

#include <typeinfo>
#include <cstddef>
#include <functional>

namespace core {

/// Type-safe identifier for entity_element types.
/// Uses typeid for compile-time safety with display names for GUI.
class element_type {
    const std::type_info* info_;
    const char* display_name_;

    element_type(const std::type_info* info, const char* name)
        : info_(info), display_name_(name) {}

public:
    /// Create an element_type for a specific class with a display name.
    template<typename T>
    static element_type of(const char* display_name) {
        return element_type{&typeid(T), display_name};
    }

    /// Get the underlying type_info reference.
    [[nodiscard]] const std::type_info& info() const { return *info_; }

    /// Get the human-readable display name (for GUI).
    [[nodiscard]] const char* name() const { return display_name_; }

    /// Type equality comparison using type_info.
    bool operator==(const element_type& other) const {
        return *info_ == *other.info_;
    }

    bool operator!=(const element_type& other) const {
        return !(*this == other);
    }

    /// Check if this element_type matches a specific class.
    template<typename T>
    [[nodiscard]] bool is() const { return *info_ == typeid(T); }

    /// Hash code for use in containers.
    [[nodiscard]] std::size_t hash() const { return info_->hash_code(); }
};

} // namespace core

// Specialization of std::hash for element_type
template<>
struct std::hash<core::element_type> {
    std::size_t operator()(const core::element_type& t) const noexcept {
        return t.hash();
    }
};

#endif // COAST_ELEMENT_TYPE_H
