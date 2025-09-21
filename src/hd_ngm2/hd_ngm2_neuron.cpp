//
// Created by jk on 26.08.25.
//
#include <algorithm>
#include "hd_ngm2_neuron.h"

namespace ngm2 {

/*
 * main initialization of a neuron
 */
neuron_t::neuron_t(params_t _params) :
    params( std::move(_params) ),
    neuron_activity(0.0f),
    dendrite_type_activity(),
    input_count(0),
    branch_interval(params.default_branch_interval),
    activity_learning_window(params.default_activity_learning_window),
    energy(1.0f),
    rgen(params.random_seed),
    id(-1)
{
    // create dendrites
    dendrites.reserve(params.dendrite_params.size());
    for (const auto &dp : params.dendrite_params)
        dendrites.emplace_back(dp);
}

/*
 * interface function that allows the neuron group to hand over functions that provide access to the respective
 * input buffers. We do not store these functions here in the neuron, but hand it over to the dendrites...
 */
void neuron_t::set_inp_func(partial_id_t id, const std::function<sim::io_buffer::inp_buf_t()> &inp_func)
{
    for (auto &dendrite : dendrites)
        dendrite.set_inp_func(id,inp_func);
}


/*
 * main function that models the response of a neuron to a current input
 *
 */
float neuron_t::get_response()
{
    // generate indices from the possible types of dendrites. Currently, only apical and proximal dendrites are
    // modelled.
    constexpr int ai = static_cast<int>(dendrite_t::type_t::apical);
    constexpr int pi = static_cast<int>(dendrite_t::type_t::proximal);

    // reset the neuron activity and initialized the type-specific activities. The initialization value of the
    // apical activity is used as a sentinel in case no apical dendrites are present. Note that neuron_activity is
    // a class variable that holds the "most recent neuron activity", which is used in the adaptation function
    // further below.
    neuron_activity            = 0.0f;
    dendrite_type_activity[ai] = -1.0f;
    dendrite_type_activity[pi] =  0.0f;

    // we get the response of every dendrite (branch) and store the maximum response for every dendrite type
    for (auto &dendrite : dendrites) {
        const int type_idx = static_cast<int>(dendrite.get_params().type);
        dendrite_type_activity[type_idx] = std::max(dendrite_type_activity[type_idx], dendrite.get_response());
    }

    // we check for our sentinel and set the apical activity to 1 if no apical dendrite is present
    if (dendrite_type_activity[ai] < 0.0f)
        dendrite_type_activity[ai] = 1.0f;

    // we ensure that all activites are in a suitable range
    dendrite_type_activity[ai] = std::clamp(dendrite_type_activity[ai], 0.0f, 1.0f);
    dendrite_type_activity[pi] = std::clamp(dendrite_type_activity[pi], 0.0f, 1.0f);

    // we modulate the proximal activity by the apical activity and add 1% to 5% of noise
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

/*
 * modelling the adaptation of a neuron
 */
void neuron_t::adapt(float weight)
{
    // we only want to learn if our neuron activity was somewhere in the middle. If the neurons response was very low
    // or very high, we reduce the weight towards 0
    const float synapse_weight = weight * std::min(       sigmoid(neuron_activity,activity_learning_window.first),
                                                   1.0f - sigmoid(neuron_activity,activity_learning_window.second));

    // we adapt all dendrites and provide each dendrite with the information about the maximum activity among their
    // dendrite type. With that information the dendrite can determine, if it was "the winning" dendrite among all
    // the dendrites.
    for (auto &dendrite : dendrites) {
        dendrite.adapt_synapses( dendrite_type_activity[static_cast<int>(dendrite.get_params().type)], synapse_weight );
    }

    // in order to see if we should check for further branching of our dendrites we count the inputs and see if we are
    // at a branch interval
    if (++input_count % branch_interval)
        return;

    // if that is the case we check if the dendrites should branch
    for (auto &dendrite : dendrites) {
        dendrite.adapt_branches();
    }
}

/*
 *  introspection functions used by, e.g., visualization components
 */
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

