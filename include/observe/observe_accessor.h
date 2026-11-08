//
// Created for the COAST observe system
//
// Typed front-end handed to entity_element::perform_observations() during the
// single-threaded post-process observation pass. It is a thin wrapper over the
// observe_registry: it derives the HDF5 dtype/shape from the value's C++ type
// and forwards a type-erased record to the registry, which gates on the group's
// per-step activity and (if active) hands the value straight to the group writer.
//
// Usage inside a perform_observations() override:
//     void neuron_t::perform_observations(observe::observe_accessor& acc) {
//         if (acc.group_active("neuron"))
//             acc.observe(*this, "neuron", "activity", neuron_activity);
//     }
//
// The optional trigger() mirrors the old OBSERVE_TRIGGER capability: a group
// may be driven START / STOP / SINGLE_CAPTURE from within an override (evaluate
// the condition, then observe). It is purely opt-in — range-configured groups
// need no trigger at all.

#ifndef COAST_OBSERVE_ACCESSOR_H
#define COAST_OBSERVE_ACCESSOR_H

#include "observe/observe_dtype.h"
#include "observe/observe_trigger.h"

#include <cstddef>
#include <span>
#include <string_view>

namespace core { class entity_element; }

namespace observe {

class observe_registry;

class observe_accessor {
public:
    explicit observe_accessor(observe_registry& reg) noexcept : reg_(reg) {}

    /// Scalar value (float / double / [u]intN). Requires a dtype_traits<T>.
    template<class T>
    void observe(core::entity_element& e, std::string_view group,
                 std::string_view id, const T& value) {
        emit(e, group, id, dtype_traits<T>::h5_dtype(),
             dtype_traits<T>::shape(), &value, sizeof(T));
    }

    /// 1-D span -> per-row dataset shape [N].
    template<class T>
    void observe(core::entity_element& e, std::string_view group,
                 std::string_view id, std::span<const T> values) {
        const hsize_t shape[1] = { static_cast<hsize_t>(values.size()) };
        emit(e, group, id, dtype_traits<T>::h5_dtype(),
             std::span<const hsize_t>(shape, 1),
             values.data(), values.size_bytes());
    }

    /// N-D span + explicit shape -> per-row dataset shape [d0, d1, ...].
    /// `product(shape)` must equal `values.size()`.
    template<class T>
    void observe(core::entity_element& e, std::string_view group,
                 std::string_view id, std::span<const T> values,
                 std::span<const std::size_t> shape) {
        constexpr std::size_t max_rank = 8;
        hsize_t dims[max_rank];
        const std::size_t n = shape.size() < max_rank ? shape.size() : max_rank;
        for (std::size_t i = 0; i < n; ++i) dims[i] = static_cast<hsize_t>(shape[i]);
        emit(e, group, id, dtype_traits<T>::h5_dtype(),
             std::span<const hsize_t>(dims, n),
             values.data(), values.size_bytes());
    }

    /// True iff `group` commits this step (range OR trigger, combine-aware).
    /// Cheap — use it to skip expensive value computation (e.g. GPU readback).
    [[nodiscard]] bool group_active(std::string_view group) const;

    /// Optional programmatic trigger; the mode is applied only when cond is true.
    void trigger(std::string_view group, trigger_mode_t mode, bool cond);

private:
    void emit(core::entity_element& e, std::string_view group,
              std::string_view id, hid_t h5_dtype,
              std::span<const hsize_t> shape,
              const void* value, std::size_t value_bytes);

    observe_registry& reg_;
};

} // namespace observe

#endif // COAST_OBSERVE_ACCESSOR_H
