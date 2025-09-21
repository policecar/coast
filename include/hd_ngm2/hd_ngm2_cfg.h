//
// Created by jk on 04.09.25.
//

#ifndef HD_NGM2_CFG_H
#define HD_NGM2_CFG_H

#include "hd_ngm2_neuron_group.h"

namespace ngm2 {
inline neuron_group_t::params_t basic_cng(
    partial_id_t id,
    std::size_t  neuron_cnt,
    std::size_t  input_size,
    const std::set<partial_id_t>& input_ids,
    int          rnd_seed = 0
){
    constexpr float learning_multiplier = 1.0f;

    neuron_group_t::params_t ngm_params;
    ngm_params.id = id;
    ngm_params.random_seed = rnd_seed++;
    ngm_params.default_weight_filter = { 0.5, 0.33 };
    ngm_params.default_common_learning_rate = 0.0001f * learning_multiplier;
    ngm_params.default_local_inhibition_strength = 5.0f;
    ngm_params.default_stochastic_win_thres = 0.8f;
    ngm_params.neuron_params.resize(neuron_cnt);
    for (auto &np : ngm_params.neuron_params) {
        np.default_activity_learning_window =
            std::make_pair<sigmoid_shape_t,sigmoid_shape_t>(
                { 0.6, 0.33 },
                {0.6, 0.66}
            );
        np.default_branch_interval = 5000;
        np.dendrite_params.resize(3); // 5
        for (auto &dp : np.dendrite_params) {
            dp.permanence_threshold            = 0.3;
            dp.input_size                      = input_size;
            dp.input_ids                       = input_ids;
            dp.default_accumulated_theta_thres = 2.0f;
            dp.default_min_mismatch_deviation  = 1.0f;
            dp.default_min_mismatch_percentage = 0.002f;
            dp.default_mismatch_smoothing      = 0.001f;
            dp.default_primary_learning_rate   = 0.01f * learning_multiplier;
            dp.default_secondary_learning_rate = 0.0001f * learning_multiplier;
            dp.max_branch_level                = 2;//3;
            dp.rnd_seed                        = rnd_seed++;
            dp.type = ngm2::dendrite_t::type_t::proximal;
        }
    }
    return ngm_params;
}

}


#endif //HD_NGM2_CFG_H
