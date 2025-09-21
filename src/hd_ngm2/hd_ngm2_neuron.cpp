//
// Created by jk on 26.08.25.
//
#include <algorithm>
#include "hd_ngm2_neuron.h"

namespace ngm2 {

neuron_t::neuron_t(params_t _params) :
    params( std::move(_params) ),
    neuron_activity(0.0f),
    dendrite_type_activity(),
    input_count(0),
    branch_interval(params.default_branch_interval),
    activity_learning_window(params.default_activity_learning_window),
    energy(1.0f),
    rgen(params.dendrite_params[0].rnd_seed)
{
    // create dendrites
    dendrites.reserve(params.dendrite_params.size());
    for (const auto &dp : params.dendrite_params)
        dendrites.emplace_back(dp);
}

/*
void neuron_t::set_input(const partial_id_t id, const std::span<const float> data)
{
    for (auto &dendrite : dendrites)
        dendrite.set_input(id,data);
}
*/

void neuron_t::set_inp_func(partial_id_t id, const std::function<sim::io_buffer::inp_buf_t()> &inp_func)
{
    for (auto &dendrite : dendrites)
        dendrite.set_inp_func(id,inp_func);
}

float neuron_t::get_response()
{
    constexpr int ai = static_cast<int>(dendrite_t::type_t::apical);
    constexpr int pi = static_cast<int>(dendrite_t::type_t::proximal);
    neuron_activity   = 0.0f;
    dendrite_type_activity[ai] = -1.0f;
    dendrite_type_activity[pi] =  0.0f;
    for (auto &dendrite : dendrites) {
        const int type_idx = static_cast<int>(dendrite.get_params().type);
        dendrite_type_activity[type_idx] = std::max(dendrite_type_activity[type_idx], dendrite.get_response());
    }
    if (dendrite_type_activity[ai] < 0.0f)
        dendrite_type_activity[ai] = 1.0f;

    dendrite_type_activity[ai] = std::clamp(dendrite_type_activity[ai], 0.0f, 1.0f);
    dendrite_type_activity[pi] = std::clamp(dendrite_type_activity[pi], 0.0f, 1.0f);

    std::uniform_real_distribution<float> dis(0.01,0.05);
    neuron_activity = std::clamp(
        dendrite_type_activity[ai] * dendrite_type_activity[pi] + dis(rgen),
        0.0f,
        1.0f//sigmoid(energy,{0.66,0.33})
    );
    /*
    const float drain = 0.1f;
    energy = energy * (1.0f - drain) + (1.0f - neuron_activity) * drain;
    */
    return neuron_activity;
}

void neuron_t::adapt(float weight)
{

    const float synapse_weight = weight * std::min(       sigmoid(neuron_activity,activity_learning_window.first),
                                                   1.0f - sigmoid(neuron_activity,activity_learning_window.second));

    //const float synapse_weight = weight;
    for (auto &dendrite : dendrites) {
        dendrite.adapt_synapses( dendrite_type_activity[static_cast<int>(dendrite.get_params().type)], synapse_weight );
    }

    if (++input_count % branch_interval)
        return;

    for (auto &dendrite : dendrites) {
        dendrite.adapt_branches();
    }
}

dendrite_t::seg_id_t neuron_t::get_representation_count() const
{
    dendrite_t::seg_id_t result = 0;
    for (const auto &dendrite : dendrites)
        result += dendrite.get_representation_count();
    return result;
}

const dendrite_t & neuron_t::get_dendrite(std::size_t idx) const
{
    return dendrites[idx];
}

std::size_t neuron_t::get_dendrite_count() const
{
    return dendrites.size();
}

std::size_t neuron_t::get_synapse_count() const
{
    std::size_t result = 0;
    for (const auto &dendrite : dendrites)
        result += dendrite.get_synapse_count();
    return result;
}

}

