//
// Created for the COAST observe system
//

#ifndef COAST_OBSERVE_DTYPE_H
#define COAST_OBSERVE_DTYPE_H

#include <hdf5.h>

#include "hdf5/native_type.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace observe {

/// Compile-time mapping from a C++ value type to its HDF5 native dtype, shape,
/// and per-value byte count. Used by observe_accessor::observe() to forward
/// dtype/shape information to the registry on first site discovery.
///
/// Specialise for additional types as needed; the primary template is left
/// undefined so misuse fails to compile rather than silently mis-mapping.
template<class T>
struct dtype_traits;

// Scalars: every type hdf5::native_type maps (the project's one table).
template<class T>
    requires hdf5::has_native_type<T>
struct dtype_traits<T> {
    static hid_t h5_dtype() { return hdf5::native_type<T>(); }
    static std::span<const hsize_t> shape() { return {}; }
    static constexpr std::size_t value_bytes = sizeof(T);
};

template<class T, std::size_t N>
struct dtype_traits<std::array<T, N>> {
    static hid_t h5_dtype() { return dtype_traits<T>::h5_dtype(); }
    static std::span<const hsize_t> shape() {
        static const hsize_t s[1] = { static_cast<hsize_t>(N) };
        return { s, 1 };
    }
    static constexpr std::size_t value_bytes = sizeof(T) * N;
};


} // namespace observe

#endif // COAST_OBSERVE_DTYPE_H
