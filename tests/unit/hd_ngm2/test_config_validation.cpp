// tests/unit/hd_ngm2/test_config_validation.cpp
//
// ngm2's construction guards. ngm2 configures permanence_threshold as a double but
// runs it as a float, so the range check has to test the float: checked in double,
// a threshold just below 1 passed and then ran as 1.0f, where nothing can connect.
// ngm3 and ngm25 hold the threshold as a float and check that.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "hd_ngm2/hd_ngm2_neuron_group.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

void build(double permanence_threshold)
{
    ngm2::cfg_store cs;
    cs.dendrites["default"].permanence_threshold = permanence_threshold;
    cs.neurons["default"].dendrites              = { { "d", "default", 2 } };
    cs.neuron_groups["default"].neurons          = { { "n", "default", 4 } };
    const ngm2::neuron_group_t g(cs, "default", /*output_id=*/0);
}

// True iff build throws a runtime_error whose message contains `needle`.
bool throws_with(double permanence_threshold, std::string_view needle)
{
    try {
        build(permanence_threshold);
    } catch (const std::runtime_error& e) {
        return std::string_view(e.what()).find(needle) != std::string_view::npos;
    }
    return false;
}

} // namespace

TEST_CASE("permanence_threshold just below 1 that runs as 1.0f is rejected")
{
    const double tau = std::nextafter(1.0, 0.0);
    REQUIRE(tau < 1.0);
    REQUIRE(static_cast<float>(tau) == 1.0f);
    CHECK(throws_with(tau, "permanence_threshold"));
}

TEST_CASE("permanence_threshold just below 0.01 that runs as 0.01f is accepted, as in ngm3 and ngm25")
{
    const double tau = std::nextafter(static_cast<double>(0.01f), 0.0);
    REQUIRE(tau < 0.01);
    REQUIRE(static_cast<float>(tau) == 0.01f);
    CHECK_NOTHROW(build(tau));
}

TEST_CASE("a group channel mapping with an empty from_channel is rejected")
{
    // After this PR's fix, the group-level from_channel is what the channel
    // record (SPEC.md M.7.5) carries all the way down to the dendrite that
    // ends up using it; an empty one would otherwise only fail later, on
    // save (config_record.h's write_channel_record throws on an empty name),
    // same as ngm3's and ngm25's route guard. Refused here instead, at
    // configuration time.
    ngm2::cfg_store cs;
    cs.neurons["default"].dendrites              = { { "d", "default", 2 } };
    cs.neuron_groups["default"].neurons          = { { "n", "default", 4 } };
    cs.neuron_groups["default"].channel_mappings = { { "", "n", "na" } };
    ngm2::neuron_group_t g(cs, "default", /*output_id=*/0);
    std::string err;
    try { g.init_io({ { "", sim::output_channel_t{ 0, 8 } } }); }
    catch (const std::runtime_error& e) { err = e.what(); }
    CHECK_MESSAGE(err.find("empty") != std::string::npos, "unexpected message: " << err);
}

// Issue #199: both values feed std::uniform_real_distribution(a, b), whose
// a <= b precondition made a reversed noise range or a winning threshold above 1
// UB (libstdc++ happened to sample the reversed interval).
TEST_CASE("a reversed or non-finite neuron noise range is rejected")
{
    const auto noise_error = [](std::pair<double, double> range) {
        ngm2::cfg_store cs;
        cs.neurons["default"].dendrites                  = { { "d", "default", 2 } };
        cs.neurons["default"].default_neuron_noise_range = range;
        cs.neuron_groups["default"].neurons              = { { "n", "default", 4 } };
        try { const ngm2::neuron_group_t g(cs, "default", /*output_id=*/0); }
        catch (const std::runtime_error& e) { return std::string(e.what()); }
        return std::string();
    };
    CHECK(noise_error({ 0.05, 0.01 }).find("default_neuron_noise_range") != std::string::npos);
    CHECK(noise_error({ 0.01, INFINITY }).find("default_neuron_noise_range") != std::string::npos);
    // finite as a double, inf as the float the distribution gets (review #3)
    CHECK(noise_error({ 0.0, 1e300 }).find("default_neuron_noise_range") != std::string::npos);
    CHECK(noise_error({ 0.01, 0.05 }).empty());
    CHECK(noise_error({ 0.0, 0.0 }).empty());   // the documented "no noise" setting
}

TEST_CASE("a stochastic_win_thres outside [0, 1] is rejected")
{
    const auto thres_error = [](double thres) {
        ngm2::cfg_store cs;
        cs.neurons["default"].dendrites                       = { { "d", "default", 2 } };
        cs.neuron_groups["default"].neurons                   = { { "n", "default", 4 } };
        cs.neuron_groups["default"].default_stochastic_win_thres = thres;
        try { const ngm2::neuron_group_t g(cs, "default", /*output_id=*/0); }
        catch (const std::runtime_error& e) { return std::string(e.what()); }
        return std::string();
    };
    CHECK(thres_error(1.2).find("default_stochastic_win_thres") != std::string::npos);
    CHECK(thres_error(-0.1).find("default_stochastic_win_thres") != std::string::npos);
    CHECK(thres_error(0.0).empty());
    CHECK(thres_error(1.0).empty());
}
