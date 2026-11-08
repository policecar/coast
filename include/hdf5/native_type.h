//
// hdf5::native_type<T>() — the one C++ type -> libhdf5 native memory type
// table. Every writer and reader that maps a C++ element type to an
// H5T_NATIVE_* id goes through it, so a mapping cannot be right in one place
// and wrong in another.
//
// Not constexpr, and executor-only like the rest of libhdf5: H5T_NATIVE_* are
// runtime globals whose first use may call H5open().
//

#ifndef COAST_HDF5_NATIVE_TYPE_H
#define COAST_HDF5_NATIVE_TYPE_H

#include <cstdint>
#include <type_traits>

#include <hdf5.h>

namespace hdf5 {

template <class T>
inline constexpr bool has_native_type =
    std::is_same_v<T, float>         || std::is_same_v<T, double>        ||
    std::is_same_v<T, std::int8_t>   || std::is_same_v<T, std::int16_t>  ||
    std::is_same_v<T, std::int32_t>  || std::is_same_v<T, std::int64_t>  ||
    std::is_same_v<T, std::uint8_t>  || std::is_same_v<T, std::uint16_t> ||
    std::is_same_v<T, std::uint32_t> || std::is_same_v<T, std::uint64_t>;

template <class T>
    requires has_native_type<T>
[[nodiscard]] inline hid_t native_type()
{
    if constexpr      (std::is_same_v<T, float>)         return H5T_NATIVE_FLOAT;
    else if constexpr (std::is_same_v<T, double>)        return H5T_NATIVE_DOUBLE;
    else if constexpr (std::is_same_v<T, std::int8_t>)   return H5T_NATIVE_INT8;
    else if constexpr (std::is_same_v<T, std::int16_t>)  return H5T_NATIVE_INT16;
    else if constexpr (std::is_same_v<T, std::int32_t>)  return H5T_NATIVE_INT32;
    else if constexpr (std::is_same_v<T, std::int64_t>)  return H5T_NATIVE_INT64;
    else if constexpr (std::is_same_v<T, std::uint8_t>)  return H5T_NATIVE_UINT8;
    else if constexpr (std::is_same_v<T, std::uint16_t>) return H5T_NATIVE_UINT16;
    else if constexpr (std::is_same_v<T, std::uint32_t>) return H5T_NATIVE_UINT32;
    else                                                 return H5T_NATIVE_UINT64;
}

} // namespace hdf5

#endif // COAST_HDF5_NATIVE_TYPE_H
