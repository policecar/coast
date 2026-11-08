#ifndef COAST_SCALAR_TYPE_H
#define COAST_SCALAR_TYPE_H

// h5browse scalar element types — the 10 dtypes the COAST writers emit
// (D10: data keeps its stored type end-to-end; f32 appears only in the
// pairwise matrix). Deliberately HDF5-free: kernels, cache, and filter
// depend on this header only, so they stay pure and unit-testable.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace h5b {

enum class scalar_type : std::uint8_t {
    f32, f64, i8, i16, i32, i64, u8, u16, u32, u64
};

inline constexpr std::size_t scalar_type_count = 10;

// Invokes f(std::type_identity<T>{}) with the C++ type behind t.
// All 10 branches return; a corrupted enum value is a program bug.
template <class F>
constexpr decltype(auto) dispatch(scalar_type t, F&& f)
{
    switch (t) {
        case scalar_type::f32: return f(std::type_identity<float>{});
        case scalar_type::f64: return f(std::type_identity<double>{});
        case scalar_type::i8:  return f(std::type_identity<std::int8_t>{});
        case scalar_type::i16: return f(std::type_identity<std::int16_t>{});
        case scalar_type::i32: return f(std::type_identity<std::int32_t>{});
        case scalar_type::i64: return f(std::type_identity<std::int64_t>{});
        case scalar_type::u8:  return f(std::type_identity<std::uint8_t>{});
        case scalar_type::u16: return f(std::type_identity<std::uint16_t>{});
        case scalar_type::u32: return f(std::type_identity<std::uint32_t>{});
        case scalar_type::u64: return f(std::type_identity<std::uint64_t>{});
    }
    std::unreachable();
}

constexpr std::size_t scalar_size(scalar_type t)
{
    return dispatch(t, [](auto id) {
        return sizeof(typename decltype(id)::type);
    });
}

constexpr const char* scalar_name(scalar_type t)
{
    switch (t) {
        case scalar_type::f32: return "f32";
        case scalar_type::f64: return "f64";
        case scalar_type::i8:  return "i8";
        case scalar_type::i16: return "i16";
        case scalar_type::i32: return "i32";
        case scalar_type::i64: return "i64";
        case scalar_type::u8:  return "u8";
        case scalar_type::u16: return "u16";
        case scalar_type::u32: return "u32";
        case scalar_type::u64: return "u64";
    }
    std::unreachable();
}

} // namespace h5b

#endif // COAST_SCALAR_TYPE_H
