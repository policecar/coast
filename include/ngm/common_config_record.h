//
// The configuration record (SPEC.md M.7.8) shared by ngm3 and ngm25.
//
// ngm25 is the ngm3 variant and its neuron_group_params_t carries the same
// settings under the same names (plus penalty_policy). One field list for both
// keeps a setting from being recorded by one model and silently left out by the
// other; each model's config_record() starts from this and adds its own.
//

#ifndef COAST_NGM_COMMON_CONFIG_RECORD_H
#define COAST_NGM_COMMON_CONFIG_RECORD_H

#include <cstdint>
#include <string>

#include "hd_ngm2/hd_ngm2_tools.h"
#include "inhibition/inhibition_cfg.h"
#include "logging/config_record.h"

namespace ngm {

// Every setting of `p` the two models share, the effective seed (random state,
// not applied on load) and the inhibitor's settings. P is ngm3's or ngm25's
// neuron_group_params_t.
template <class P>
logging::config_record common_config_record(const P&                               p,
                                            seed_t                                 effective_seed,
                                            const inhibition::inhibition_params_t& inh)
{
    auto sigmoid = [](logging::config_record& r, const std::string& name, const ngm2::sigmoid_shape_t& s) {
        r.push_back({ name + ".steepness",        s.steepness });
        r.push_back({ name + ".transition_point", s.transition_point });
    };
    logging::config_record r;
    r.push_back({ "permanence_threshold",      p.permanence_threshold });
    r.push_back({ "local_inhibition_strength", p.local_inhibition_strength });
    r.push_back({ "stochastic_win_thres",      p.stochastic_win_thres });
    sigmoid(r, "weight_filter", p.weight_filter);
    r.push_back({ "common_learning_rate",      p.common_learning_rate });
    r.push_back({ "primary_learning_rate",     p.primary_learning_rate });
    r.push_back({ "secondary_learning_rate",   p.secondary_learning_rate });
    r.push_back({ "mismatch_act_thres",        p.mismatch_act_thres });
    r.push_back({ "mismatch_smoothing",        p.mismatch_smoothing });
    r.push_back({ "branch_interval",           static_cast<std::uint64_t>(p.branch_interval) });
    r.push_back({ "min_mismatch_deviation",    p.min_mismatch_deviation });
    r.push_back({ "accumulated_theta_thres",   p.accumulated_theta_thres });
    r.push_back({ "min_mismatch_percentage",   p.min_mismatch_percentage });
    r.push_back({ "neuron_noise_range[0]",     p.neuron_noise_range.first });
    r.push_back({ "neuron_noise_range[1]",     p.neuron_noise_range.second });
    r.push_back({ "effective_seed",            std::uint64_t{ std::to_underlying(effective_seed) }, /*random_state=*/true });
    inhibition::append_config(r, inh);
    return r;
}

} // namespace ngm

#endif // COAST_NGM_COMMON_CONFIG_RECORD_H
