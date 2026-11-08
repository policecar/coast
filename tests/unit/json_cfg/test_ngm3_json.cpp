// Unit test 3.2 — ngm3 config (JSON) deserialization and its error behavior.
//
// ngm3_json.h binds the config with NLOHMANN_DEFINE_TYPE_..._WITH_DEFAULT plus
// COAST_JSON_STRICT_ENUM (json_cfg/strict_enum.h) for bank_t. That combination
// has three behaviors worth pinning, because each is easy to get wrong silently
// in a config file:
//
//   1. A *missing* field takes the struct default (WITH_DEFAULT).
//   2. A present field of the *wrong type* throws (only missing fields default).
//   3. An unmapped bank string throws std::invalid_argument. (Under the former
//      NLOHMANN_JSON_SERIALIZE_ENUM binding it silently fell back to the first
//      listed enumerator, apical, so a typo'd bank rerouted with no error.)
//      An *omitted* bank still defaults to proximal per (1).
//
// Pure deserialization — no sim, no files.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

#include "json_cfg/ngm3_json.h"

using ngm3::bank_t;
using ngm3::input_route_t;
using ngm3::neuron_group_params_t;

TEST_CASE("ngm3 config: a fully specified group round-trips") {
    auto j = nlohmann::json::parse(R"({
        "neuron_cnt": 8,
        "proximal_cnt": 4,
        "apical_cnt": 2,
        "max_branch_level": 3,
        "branch_interval": 250,
        "inputs": [
            {"channel_name": "in_apical",   "bank": "apical"},
            {"channel_name": "in_proximal", "bank": "proximal"}
        ],
        "random_seed": 7
    })");
    auto p = j.get<neuron_group_params_t>();

    CHECK(p.neuron_cnt == 8);
    CHECK(p.proximal_cnt == 4);
    CHECK(p.apical_cnt == 2);
    CHECK(p.max_branch_level == 3);
    CHECK(p.branch_interval == 250);
    CHECK(p.random_seed == seed_t{ 7 });
    REQUIRE(p.inputs.size() == 2);
    CHECK(p.inputs[0].channel_name == "in_apical");
    CHECK(p.inputs[0].bank == bank_t::apical);
    CHECK(p.inputs[1].channel_name == "in_proximal");
    CHECK(p.inputs[1].bank == bank_t::proximal);
}

TEST_CASE("ngm3 config: omitted fields fall back to struct defaults") {
    auto p = nlohmann::json::parse(R"({"neuron_cnt": 32})").get<neuron_group_params_t>();
    CHECK(p.neuron_cnt == 32);           // set
    CHECK(p.proximal_cnt == 4);          // default
    CHECK(p.max_branch_level == 3); // default
    CHECK(p.branch_interval == 5000);    // default
    CHECK(p.inputs.empty());             // default
    CHECK(p.permanence_threshold == doctest::Approx(0.3f));
}

TEST_CASE("ngm3 config: bank enum maps the two valid spellings") {
    auto a = nlohmann::json::parse(R"({"channel_name":"c","bank":"apical"})")
                 .get<input_route_t>();
    CHECK(a.bank == bank_t::apical);
    auto p = nlohmann::json::parse(R"({"channel_name":"c","bank":"proximal"})")
                 .get<input_route_t>();
    CHECK(p.bank == bank_t::proximal);
}

TEST_CASE("ngm3 config: an OMITTED bank takes the struct default (proximal)") {
    auto r = nlohmann::json::parse(R"({"channel_name":"c"})").get<input_route_t>();
    CHECK(r.bank == bank_t::proximal);
}

TEST_CASE("ngm3 config: an UNKNOWN bank string is rejected") {
    // Config enums are strict (json_cfg/strict_enum.h). This used to fall back
    // silently to the first enumerator (bank_t::apical), so a typo'd bank changed
    // routing with no error.
    auto bogus = nlohmann::json::parse(R"({"channel_name":"c","bank":"proxximal"})");
    CHECK_THROWS_AS(bogus.get<input_route_t>(), std::invalid_argument);
}

TEST_CASE("ngm3 config: a present field of the wrong type is rejected") {
    // Only *missing* fields default; a present neuron_cnt that is not a number
    // must throw rather than silently fall back.
    auto j = nlohmann::json::parse(R"({"neuron_cnt": "not-a-number"})");
    CHECK_THROWS(j.get<neuron_group_params_t>());
}
