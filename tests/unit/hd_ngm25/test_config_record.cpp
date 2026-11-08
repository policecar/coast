// tests/unit/hd_ngm25/test_config_record.cpp
//
// ngm3 and ngm25 build their configuration records (SPEC.md M.7.8) from one
// shared field list (ngm/common_config_record.h). Pins that list, so a setting
// cannot drop out of the record unnoticed, and that ngm25 differs only by
// penalty_policy.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "hd_ngm25/hd_ngm25_cfg.h"
#include "hd_ngm3/hd_ngm3_cfg.h"
#include "logging/config_record.h"

#include <string>
#include <variant>
#include <vector>

namespace {

const std::vector<std::string> k_ngm3_names = {
    "permanence_threshold", "local_inhibition_strength", "stochastic_win_thres",
    "weight_filter.steepness", "weight_filter.transition_point", "common_learning_rate",
    "primary_learning_rate", "secondary_learning_rate", "mismatch_act_thres",
    "mismatch_smoothing", "branch_interval", "min_mismatch_deviation",
    "accumulated_theta_thres", "min_mismatch_percentage", "neuron_noise_range[0]",
    "neuron_noise_range[1]", "effective_seed", "inhibition.type",
    "inhibition.strength",
};

std::vector<std::string> names(const logging::config_record& rec)
{
    std::vector<std::string> n;
    for (const auto& e : rec) n.push_back(e.name);
    return n;
}

const inhibition::inhibition_params_t k_inh{ inhibition::inhibition_type_t::max_softmax, 10.0 };

} // namespace

TEST_CASE("ngm3 records every setting of the shared list")
{
    CHECK(names(ngm3::config_record(ngm3::neuron_group_params_t{}, seed_t{ 1 }, k_inh)) == k_ngm3_names);
}

TEST_CASE("ngm25 records the same settings plus penalty_policy")
{
    ngm25::neuron_group_params_t p;
    p.penalty_policy = ngm25::penalty_policy_t::expected_value;
    const auto rec = ngm25::config_record(p, seed_t{ 1 }, k_inh);

    auto expected = k_ngm3_names;
    expected.push_back("penalty_policy");
    CHECK(names(rec) == expected);
    CHECK(std::get<std::string>(rec.back().value) == "expected_value");
}
