//
// Created for the COAST probe system
//
// Helper macros to reduce boilerplate when implementing entity_element subclasses.
//

#ifndef COAST_ENTITY_ELEMENT_MACROS_H
#define COAST_ENTITY_ELEMENT_MACROS_H

#include "entity_element.h"
#include "probe_name_registry.h"

/// Implement type_id() for an entity_element subclass.
/// Usage: ELEMENT_TYPE(my_class, "display_name")
#define ELEMENT_TYPE(cls, display_name) \
    [[nodiscard]] core::element_type type_id() const override { \
        return core::element_type::of<cls>(display_name); \
    }

/// Implement children(), child_impl(), and child_count_impl() for a single child type.
/// Usage: ELEMENT_CHILDREN_1(member_vector, ChildType)
#define ELEMENT_CHILDREN_1(member, ChildType) \
    [[nodiscard]] std::vector<child_group> children() const override { \
        return {{core::element_type::of<ChildType>(#ChildType), member.size()}}; \
    } \
    [[nodiscard]] core::entity_element* child_impl(const std::type_info& type, std::size_t index) override { \
        if (type == typeid(ChildType) && index < member.size()) \
            return &member[index]; \
        return nullptr; \
    } \
    [[nodiscard]] const core::entity_element* child_impl(const std::type_info& type, std::size_t index) const override { \
        if (type == typeid(ChildType) && index < member.size()) \
            return &member[index]; \
        return nullptr; \
    } \
    [[nodiscard]] std::size_t child_count_impl(const std::type_info& type) const override { \
        if (type == typeid(ChildType)) return member.size(); \
        return 0; \
    }

/// Implement children(), child_impl(), and child_count_impl() for two child types.
/// Usage: ELEMENT_CHILDREN_2(member1, Type1, member2, Type2)
#define ELEMENT_CHILDREN_2(m1, T1, m2, T2) \
    [[nodiscard]] std::vector<child_group> children() const override { \
        return { \
            {core::element_type::of<T1>(#T1), m1.size()}, \
            {core::element_type::of<T2>(#T2), m2.size()} \
        }; \
    } \
    [[nodiscard]] core::entity_element* child_impl(const std::type_info& type, std::size_t index) override { \
        if (type == typeid(T1) && index < m1.size()) return &m1[index]; \
        if (type == typeid(T2) && index < m2.size()) return &m2[index]; \
        return nullptr; \
    } \
    [[nodiscard]] const core::entity_element* child_impl(const std::type_info& type, std::size_t index) const override { \
        if (type == typeid(T1) && index < m1.size()) return &m1[index]; \
        if (type == typeid(T2) && index < m2.size()) return &m2[index]; \
        return nullptr; \
    } \
    [[nodiscard]] std::size_t child_count_impl(const std::type_info& type) const override { \
        if (type == typeid(T1)) return m1.size(); \
        if (type == typeid(T2)) return m2.size(); \
        return 0; \
    }

/// Implement probe_points() returning a list of probe names.
/// Usage: ELEMENT_PROBES("activation", "threshold", "spike")
/// NOTE: Prefer using PROBE() macro instead, which auto-registers probe names.
#define ELEMENT_PROBES(...) \
    [[nodiscard]] std::vector<std::string_view> probe_points() const override { \
        return {__VA_ARGS__}; \
    }

/// Record a probe value and auto-register the probe name on first call.
/// Usage: PROBE("activity", activity_value);
/// The probe name is registered once per call site via static initialization,
/// making it visible in the probe browser without manual ELEMENT_PROBES declaration.
#define PROBE(name, value) \
    do { \
        static const bool registered_ = (::core::probe_name_registry::register_probe(typeid(*this), name), true); \
        (void)registered_; \
        probe(name, value); \
    } while(0)

#endif // COAST_ENTITY_ELEMENT_MACROS_H
