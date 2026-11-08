#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "inhibition/inhibition.h"
#include "inhibition/inhibition_cfg.h"

using namespace inhibition;

namespace {

const std::unordered_map<std::string, inhibition_params_t> k_dict = {
    { "ridx", inhibition_params_t{ .type = inhibition_type_t::random_index, .ri_reach = 0.25 } },
};

// The message of the runtime_error `f` throws; fails the test if it throws
// anything else or nothing.
template <class F>
std::string runtime_error_of(F&& f)
{
    try {
        f();
    } catch (const std::runtime_error& e) {
        return e.what();
    } catch (...) {
        FAIL("threw something other than std::runtime_error");
    }
    FAIL("did not throw");
    return {};
}

} // namespace

TEST_CASE("resolve_params: an empty name synthesizes max_softmax with the fallback as beta")
{
    const auto p = resolve_params(k_dict, "", 3.5);
    CHECK(p.type == inhibition_type_t::max_softmax);
    CHECK(p.strength == 3.5);
}

TEST_CASE("resolve_params: a known name returns its entry")
{
    const auto p = resolve_params(k_dict, "ridx", 3.5);
    CHECK(p.type == inhibition_type_t::random_index);
    CHECK(p.ri_reach == 0.25);
}

// A typo in inhibition_cfg must say what is wrong, whoever calls first — not
// surface as a bare std::out_of_range from map::at.
TEST_CASE("resolve_params: an unknown name throws a runtime_error naming it")
{
    const auto msg = runtime_error_of([] { (void)resolve_params(k_dict, "ridz", 3.5); });
    CHECK(msg.find("\"ridz\"") != std::string::npos);
    CHECK(msg.find("not found in inhibitions dict") != std::string::npos);
}

TEST_CASE("make_inhibitor: an unknown name is rejected through resolve_params")
{
    const auto msg = runtime_error_of([] { (void)make_inhibitor(k_dict, "ridz", 3.5, seed_offset_t{}); });
    CHECK(msg.find("\"ridz\"") != std::string::npos);
}

// Issue #190: ri_seed composes like every other config seed (S.3.24) — config
// seed + the group's offset (global --seed-offset + node offset), modulo 2^32.
TEST_CASE("make_inhibitor: ri_seed composes with the group's seed offset, modulo 2^32")
{
    const std::unordered_map<std::string, inhibition_params_t> dict = {
        { "ridx", inhibition_params_t{ .type = inhibition_type_t::random_index, .ri_seed = seed_t{ 10 } } },
    };
    CHECK(make_inhibitor(dict, "ridx", 3.5, seed_offset_t{})->params().ri_seed == seed_t{ 10 });
    CHECK(make_inhibitor(dict, "ridx", 3.5, seed_offset_t{ 5 })->params().ri_seed == seed_t{ 15 });
    // -11 as 32-bit two's complement: 10 - 11 wraps to 2^32 - 1.
    CHECK(make_inhibitor(dict, "ridx", 3.5, seed_offset_t{ 4294967285u })->params().ri_seed
          == seed_t{ 4294967295u });
}

TEST_CASE("random_state_seed: ri_seed for random_index, nothing for max_softmax")
{
    CHECK(random_state_seed(inhibition_params_t{ .type = inhibition_type_t::random_index,
                                                 .ri_seed = seed_t{ 3 } }) == seed_t{ 3 });
    CHECK_FALSE(random_state_seed(inhibition_params_t{ .type = inhibition_type_t::max_softmax,
                                                       .ri_seed = seed_t{ 3 } }).has_value());
}

// Every neuron group writes its inhibitor's entry into the configuration record
// on save, so a kind append_config does not handle makes every checkpoint of a
// group using it throw. A new kind must be added there too.
TEST_CASE("append_config: every inhibition kind writes its type and settings")
{
    std::vector<std::string> types;
    for (int t = 0; t < static_cast<int>(inhibition_type_t::TYPE_COUNT); ++t) {
        CAPTURE(t);
        inhibition_params_t p{ .type = static_cast<inhibition_type_t>(t), .strength = 7.5 };
        logging::config_record rec;
        REQUIRE_NOTHROW(append_config(rec, p));
        const auto type = std::find_if(rec.begin(), rec.end(),
                                       [](const auto& e) { return e.name == "inhibition.type"; });
        REQUIRE(type != rec.end());
        REQUIRE(std::holds_alternative<std::string>(type->value));
        types.push_back(std::get<std::string>(type->value));
        CHECK(rec.size() > 1);   // a type alone would not catch a changed setting
    }
    std::sort(types.begin(), types.end());
    CHECK(std::adjacent_find(types.begin(), types.end()) == types.end());   // one name per kind
}

TEST_CASE("append_config: max_softmax records its beta (strength)")
{
    logging::config_record rec;
    append_config(rec, inhibition_params_t{ .type = inhibition_type_t::max_softmax, .strength = 7.5 });
    const auto s = std::find_if(rec.begin(), rec.end(),
                                [](const auto& e) { return e.name == "inhibition.strength"; });
    REQUIRE(s != rec.end());
    CHECK(std::get<double>(s->value) == 7.5);
}
