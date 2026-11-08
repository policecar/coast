#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <expected>
#include <string>

#include "core/require.h"
#include "core/step_error.h"

// Note: this TU includes BOTH doctest's REQUIRE and COAST's COAST_REQUIRE. That
// it compiles is the standing proof that the COAST_ prefix avoids the collision
// that a bare REQUIRE would cause.

namespace {

std::expected<int, core::step_error> checked(int x) {
    COAST_REQUIRE(x >= 0,  core::error_kind::contract_violation);
    COAST_REQUIRE(x < 100, core::error_kind::non_finite);
    return x * 2;
}

}  // namespace

TEST_CASE("COAST_REQUIRE passes through when conditions hold") {
    auto r = checked(5);
    REQUIRE(r.has_value());
    CHECK(*r == 10);
}

TEST_CASE("COAST_REQUIRE returns unexpected on the first violated condition") {
    auto r = checked(-1);
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().kind == core::error_kind::contract_violation);
    CHECK(r.error().entity_id == core::step_error::no_entity);
    // The site literal is captured at the use site inside checked().
    CHECK(std::string(r.error().site).find("test_step_error.cpp") != std::string::npos);
}

TEST_CASE("COAST_REQUIRE evaluates conditions in order") {
    auto r = checked(200);  // satisfies the first check (>=0), trips the second (<100)
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().kind == core::error_kind::non_finite);
}

TEST_CASE("error_kind to_string") {
    CHECK(std::string(core::to_string(core::error_kind::contract_violation)) ==
          "contract_violation");
    CHECK(std::string(core::to_string(core::error_kind::device_fault)) == "device_fault");
    CHECK(std::string(core::to_string(core::error_kind::exception)) == "exception");
}
