// h5browse scalar_type — dispatch round trips, sizes, names.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "h5browse/scalar_type.h"

#include <cstdint>
#include <string_view>
#include <type_traits>

using h5b::dispatch;
using h5b::scalar_name;
using h5b::scalar_size;
using h5b::scalar_type;

TEST_CASE("scalar_type: sizes match the dispatched C++ types")
{
    CHECK(scalar_size(scalar_type::f32) == 4);
    CHECK(scalar_size(scalar_type::f64) == 8);
    CHECK(scalar_size(scalar_type::i8) == 1);
    CHECK(scalar_size(scalar_type::i16) == 2);
    CHECK(scalar_size(scalar_type::i32) == 4);
    CHECK(scalar_size(scalar_type::i64) == 8);
    CHECK(scalar_size(scalar_type::u8) == 1);
    CHECK(scalar_size(scalar_type::u16) == 2);
    CHECK(scalar_size(scalar_type::u32) == 4);
    CHECK(scalar_size(scalar_type::u64) == 8);
}

TEST_CASE("scalar_type: dispatch yields the exact type")
{
    auto is = [](scalar_type t, auto tag) {
        using Want = typename decltype(tag)::type;
        return dispatch(t, [](auto id) {
            return std::is_same_v<typename decltype(id)::type, Want>;
        });
    };
    CHECK(is(scalar_type::f32, std::type_identity<float>{}));
    CHECK(is(scalar_type::f64, std::type_identity<double>{}));
    CHECK(is(scalar_type::i8, std::type_identity<std::int8_t>{}));
    CHECK(is(scalar_type::i16, std::type_identity<std::int16_t>{}));
    CHECK(is(scalar_type::i32, std::type_identity<std::int32_t>{}));
    CHECK(is(scalar_type::i64, std::type_identity<std::int64_t>{}));
    CHECK(is(scalar_type::u8, std::type_identity<std::uint8_t>{}));
    CHECK(is(scalar_type::u16, std::type_identity<std::uint16_t>{}));
    CHECK(is(scalar_type::u32, std::type_identity<std::uint32_t>{}));
    CHECK(is(scalar_type::u64, std::type_identity<std::uint64_t>{}));

    CHECK_FALSE(is(scalar_type::f32, std::type_identity<double>{}));
    CHECK_FALSE(is(scalar_type::i8, std::type_identity<std::uint8_t>{}));
}

TEST_CASE("scalar_type: names")
{
    CHECK(std::string_view{scalar_name(scalar_type::f32)} == "f32");
    CHECK(std::string_view{scalar_name(scalar_type::u64)} == "u64");
    CHECK(std::string_view{scalar_name(scalar_type::i16)} == "i16");
}

TEST_CASE("scalar_type: dispatch works in constexpr context")
{
    constexpr std::size_t sz = scalar_size(scalar_type::i32);
    static_assert(sz == 4);
    CHECK(sz == 4);
}
