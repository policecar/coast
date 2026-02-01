//
// Created by jk on 04.09.25.
//

#ifndef HD_NGM2_CFG_H
#define HD_NGM2_CFG_H

#include <string>
#include <unordered_map>
#include <set>

#include "hd_ngm2_tools.h"
/*
 * utility function to generate a default parameterization of a cortical neuron group without apical dendrites
 */

namespace ngm2 {

enum class dendrite_type_t {
    apical,          // this type of dendrite modulates the neurons activity (if present)
    proximal,        // main type of dendrite that determines the neuron activity
    TYPE_COUNT
};

// usable default values given here (input_size and input_ids have no "usable" default)
struct dendrite_params_t {
    dendrite_type_t type                        = dendrite_type_t::proximal;
    std::vector<std::string> input_channels     = {};
    double      permanence_threshold            = 0.3;
    uint8_t     max_branch_level                = 3;
    int         rnd_seed                        = 0;
    double      default_primary_learning_rate   = 0.01;
    double      default_secondary_learning_rate = 0.0001;
    double      default_mismatch_act_thres      = 0.8;
    double      default_mismatch_smoothing      = 0.001;
    double      default_accumulated_theta_thres = 2.0;
    double      default_min_mismatch_deviation  = 1.0;
    double      default_min_mismatch_percentage = 0.002;
};

using learning_window_t = std::pair<sigmoid_shape_t,sigmoid_shape_t>;

struct dendrite_set_t
{
    std::string name;
    std::string dendrite_cfg;
    std::size_t dendrite_count;
};

struct channel_mapping_t
{
    std::string from_channel;
    std::string to_name;
    std::string to_channel;
};

// usable default values, apart from dendrite params
struct neuron_params_t {
    std::vector<std::string>       input_channels                   = {};
    std::vector<dendrite_set_t>    dendrites                        = {};
    std::vector<channel_mapping_t> channel_mappings                 = {};
    std::size_t                    default_branch_interval          = 5000;
    learning_window_t              default_activity_learning_window = {{ 0.66, 0.33 },{ 0.66, 0.66 }};
    int                            random_seed                      = 0;
};

struct neuron_set_t
{
    std::string name;
    std::string neuron_cfg;
    std::size_t neuron_count;
};

struct neuron_group_params_t {
    std::vector<std::string>       input_channels                    = {};
    std::vector<neuron_set_t>      neurons                           = {};
    std::vector<channel_mapping_t> channel_mappings                  = {};
    double                         default_local_inhibition_strength = 5.0;
    double                         default_common_learning_rate      = 0.0001;
    sigmoid_shape_t                default_weight_filter             = { 0.5, 0.33 };
    double                         default_stochastic_win_thres      = 0.8;
    int                            random_seed                       = 0;
};

struct cfg_store {

    std::unordered_map<std::string,dendrite_params_t>     dendrites;
    std::unordered_map<std::string,neuron_params_t>       neurons;
    std::unordered_map<std::string,neuron_group_params_t> neuron_groups;

    cfg_store()
    {
        dendrites["default"];
        neurons["default"];
        neuron_groups["default"];
    }
};

}


#endif //HD_NGM2_CFG_H
