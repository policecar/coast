// Unit test 1.1 — observe config (JSON) deserialization.
//
// Covers the `observe` section of the config: groups, step ranges, and the
// combine mode, plus the error paths that a typo'd config would hit. This is
// the gap that would otherwise let a malformed capture window silently
// misconfigure (or, for the range/enum errors, surface as an opaque load
// failure with no test pinning the contract).
//
// Pure deserialization — no registry, no sim, no HDF5.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "observe/observe_cfg.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

using observe::combine_mode_t;
using observe::observe_cfg_t;

TEST_CASE("observe config: valid groups parse with ranges and combine") {
    auto j = nlohmann::json::parse(R"({
        "output_dir": "./obs_out",
        "groups": {
            "layer1": { "ranges": [[100, 200], [500, 600]], "combine": "and" },
            "layer2": { "ranges": [[0, 50]] }
        }
    })");

    auto cfg = j.get<observe_cfg_t>();

    CHECK(cfg.output_dir == "./obs_out");
    REQUIRE(cfg.groups.size() == 2);

    REQUIRE(cfg.groups.count("layer1") == 1);
    const auto& l1 = cfg.groups.at("layer1");
    REQUIRE(l1.ranges.size() == 2);
    CHECK(l1.ranges[0].start_step == 100);
    CHECK(l1.ranges[0].stop_step == 200);
    CHECK(l1.ranges[1].start_step == 500);
    CHECK(l1.ranges[1].stop_step == 600);
    CHECK(l1.combine == combine_mode_t::AND_);

    // combine defaults to OR_ when omitted (NLOHMANN ..._WITH_DEFAULT)
    const auto& l2 = cfg.groups.at("layer2");
    REQUIRE(l2.ranges.size() == 1);
    CHECK(l2.combine == combine_mode_t::OR_);
}

TEST_CASE("observe config: combine string is case-insensitive") {
    auto cfg = nlohmann::json::parse(R"({"groups": {"g": {"combine": "AND"}}})")
                   .get<observe_cfg_t>();
    CHECK(cfg.groups.at("g").combine == combine_mode_t::AND_);

    auto cfg2 = nlohmann::json::parse(R"({"groups": {"g": {"combine": "Or"}}})")
                    .get<observe_cfg_t>();
    CHECK(cfg2.groups.at("g").combine == combine_mode_t::OR_);
}

TEST_CASE("observe config: a step range that is not [start, stop] is rejected") {
    // 3-element range — step_range_t requires exactly two endpoints.
    auto j = nlohmann::json::parse(R"({"groups": {"g": {"ranges": [[1, 2, 3]]}}})");
    CHECK_THROWS_AS((void)j.get<observe_cfg_t>(), std::runtime_error);
}

TEST_CASE("observe config: an unknown combine mode is rejected") {
    auto j = nlohmann::json::parse(R"({"groups": {"g": {"combine": "xor"}}})");
    CHECK_THROWS_AS((void)j.get<observe_cfg_t>(), std::runtime_error);
}

TEST_CASE("observe config: empty config yields no groups and an empty output-dir leaf") {
    auto cfg = nlohmann::json::parse("{}").get<observe_cfg_t>();
    CHECK(cfg.groups.empty());
    // Empty leaf = "defer to the app default": the app_config's paths.observe_dir
    // applies, resolved under config_base by the app layer. (Was "./observe_out"
    // before the app-level location config; the hardcoded relative default moved
    // into app_config as the "observe" leaf default.)
    CHECK(cfg.output_dir.empty());
}
