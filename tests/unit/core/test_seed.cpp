#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cstdint>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "core/seed.h"

// The seed rule of core/seed.h: 32 bits, composed modulo 2^32; input is an
// integer in [0, 2^32 - 1], for config seeds, config offsets and --seed-offset.

namespace {

struct cfg_t {
    seed_t        random_seed{};
    seed_offset_t random_seed_offset{};
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(cfg_t, random_seed, random_seed_offset)

template <class Json>
std::uint32_t seed_of(const char* text)
{
    return std::to_underlying(Json::parse(text).template get<cfg_t>().random_seed);
}

} // namespace

TEST_CASE_TEMPLATE("seed: JSON input takes [0, 2^32 - 1]", Json, nlohmann::json, nlohmann::ordered_json)
{
    CHECK(seed_of<Json>(R"({"random_seed": 7})")           == 7u);
    CHECK(seed_of<Json>(R"({"random_seed": 0})")           == 0u);
    CHECK(seed_of<Json>(R"({"random_seed": 4294967295})")  == 4294967295u);
    CHECK(seed_of<Json>(R"({})")                           == 0u);   // the default

    CHECK_THROWS_AS(seed_of<Json>(R"({"random_seed": 4294967296})"),           std::invalid_argument);
    // a negative seed used to be taken as its two's complement (#167); refused since #199
    CHECK_THROWS_AS(seed_of<Json>(R"({"random_seed": -1})"),                   std::invalid_argument);
    CHECK_THROWS_AS(seed_of<Json>(R"({"random_seed": -2147483648})"),          std::invalid_argument);
    CHECK_THROWS_AS(seed_of<Json>(R"({"random_seed": 18446744073709551615})"), std::invalid_argument);
    // nlohmann's own conversion would truncate these silently (1.7 -> 1)
    CHECK_THROWS_AS(seed_of<Json>(R"({"random_seed": 1.0})"),                  std::invalid_argument);
    CHECK_THROWS_AS(seed_of<Json>(R"({"random_seed": -1.5})"),                 std::invalid_argument);
    CHECK_THROWS_AS(seed_of<Json>(R"({"random_seed": "3"})"),                  std::invalid_argument);
}

TEST_CASE_TEMPLATE("seed: offsets follow the same input rule", Json, nlohmann::json, nlohmann::ordered_json)
{
    const auto c = Json::parse(R"({"random_seed_offset": 4294967293})").template get<cfg_t>();
    CHECK(c.random_seed_offset == seed_offset_t{ 4294967293u });
    CHECK_THROWS_AS(Json::parse(R"({"random_seed_offset": -3})").template get<cfg_t>(), std::invalid_argument);
    CHECK_THROWS_AS(Json::parse(R"({"random_seed_offset": 0.5})").template get<cfg_t>(), std::invalid_argument);
}

TEST_CASE_TEMPLATE("seed: written back as its uint32 value", Json, nlohmann::json, nlohmann::ordered_json)
{
    const Json j = cfg_t{ seed_t{ 4294967295u }, seed_offset_t{ 3 } };
    CHECK(j.at("random_seed").is_number_unsigned());
    CHECK(j.at("random_seed").template get<std::uint64_t>() == 4294967295u);
    CHECK(j.at("random_seed_offset").template get<std::uint64_t>() == 3u);
}

TEST_CASE("seed: composition wraps modulo 2^32")
{
    static_assert(seed_t{ 4294967295u } + seed_offset_t{ 1 } == seed_t{ 0 });
    static_assert(seed_t{ 0 } + seed_offset_t{ 4294967295u } == seed_t{ 4294967295u });   // 0 + (-1)
    static_assert(seed_offset_t{ 4294967295u } + seed_offset_t{ 2 } == seed_offset_t{ 1 });
    CHECK(std::to_underlying(seed_t{ 10 } + seed_offset_t{ 5 } + seed_offset_t{ 7 }) == 22u);
}

TEST_CASE("seed: seed_bits_of is the non-throwing JSON check json_loader uses")
{
    using J = nlohmann::ordered_json;
    CHECK(core::seed_bits_of(J(0))          == 0u);
    CHECK(core::seed_bits_of(J(4294967295)) == 4294967295u);
    CHECK(core::seed_bits_of(J(std::int64_t{ 5 })) == 5u);   // a signed-typed non-negative value
    CHECK_FALSE(core::seed_bits_of(J(4294967296)).has_value());
    CHECK_FALSE(core::seed_bits_of(J(-1)).has_value());
    CHECK_FALSE(core::seed_bits_of(J(1.0)).has_value());
    CHECK_FALSE(core::seed_bits_of(J("3")).has_value());
    CHECK_FALSE(core::seed_bits_of(J(true)).has_value());
    CHECK_FALSE(core::seed_bits_of(J(nullptr)).has_value());
}

TEST_CASE("seed: seed_bits is the shared input range check")
{
    CHECK(core::seed_bits(0)           == 0u);
    CHECK(core::seed_bits(4294967295)  == 4294967295u);
    CHECK_FALSE(core::seed_bits(-1).has_value());
    CHECK_FALSE(core::seed_bits(4294967296).has_value());
}
