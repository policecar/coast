//
// Created by jk on 26.08.25.
//
#include <hd_ngm2_neuron_group.h>

#include <utility>
#include <algorithm>
#include <cassert>
#include <execution>

namespace ngm2 {

/*
 * Main initialization of the neuron group
 */
neuron_group_t::neuron_group_t(params_t  _params) :
    params                    ( std::move( _params )                     ),
    local_inhibition_strength ( params.default_local_inhibition_strength ),
    common_learning_rate      ( params.default_common_learning_rate      ),
    weight_filter             ( params.default_weight_filter             ),
    stochastic_win_thres      ( params.default_stochastic_win_thres      ),
    rgen                      ( params.random_seed                       )
{
    // temporary set to gather all input IDs from the dendrites of all
    // neurons in this group
    std::set<partial_id_t> tmp;

    // create neurons and collect the input IDs in the temporary set
    neurons.reserve(params.neuron_params.size());
    for (const auto &np : params.neuron_params) {
        auto &new_neuron = neurons.emplace_back(np);
        new_neuron.id    = neurons.size()-1;
        for (const auto &dp : np.dendrite_params) {
            tmp.insert_range(dp.input_ids);
        }
    }

    // copy the set of input IDs over to the inp_ids vector
    inp_ids.resize(tmp.size());
    std::ranges::copy(tmp,inp_ids.begin());
}

/*
 * implementing the interface function that allows the simulation environment to hand over
 * a function that will provide us with the current output buffer. This function is
 * stored in "output_mem".
 */
void neuron_group_t::set_outp_func(std::function<std::span<float>()> outp_func)
{
    output_mem = std::move(outp_func);
}

/*
 * implementing the interface function that allows the simulation environment to hand over
 * functions that provide access to the respective input buffers. We do not store these
 * functions here in the group, but hand it over to the neurons (which will in turn hand it
 * over to their dendrites...)
 */
void neuron_group_t::set_inp_func(partial_id_t id, const std::function<sim::io_buffer::inp_buf_t()> &inp_func)
{
    for (auto &neuron : neurons)
        neuron.set_inp_func(id,inp_func);
}

/*
 * main function that models the neuron group's behavior for one processing step
 */
void neuron_group_t::process()
{
    assert(output_mem != nullptr); // only checked in debug mode...

    // acquire our current output array that will hold all the activities of the neurons in this group
    std::span<float> out = output_mem();

    assert(out.size() == neurons.size());

    // get the current activity of all neurons in parallel
    // (parallel processing might in the future move up to the level of the simulation)
    std::for_each(
        std::execution::par,
        neurons.begin(),neurons.end(),
        [&](auto &neuron) {
            out[neuron.id] = neuron.get_response();
        }
    );

    // simulate local inhibition within the neuron group
    // (defined in hd_ngm2_tools.h)
    local_inhibition(out,local_inhibition_strength);

    /*
     * Simulate the adaption of the neurons in the neuron group to the current input signal.
     * 1) We determine the maximum activity in the neuron group.
     * 2) We determine a stochastic "winning" threshold.
     * 3) All neurons that reach the winning threshold will primarily adapt to the input.
     *    The strength of the adaptation depends on the activity of the neuron and is limited
     *    by a weight filter that reduces adaptation if the neuron is already strongly active
     *    in response to an input. This results in adaption happening mostly for inputs that
     *    are not yet well known.
     * 4) All neurons irrespective of their activity will adapt somewhat to an input.
     *    The strength of the adaption depends on the neurons activity in relation to the overall
     *    activity of the neuron group and a filter that reduces adaption of already strongly activated
     *    neurons. The strength is also scaled down by the "common learning rate" parameter.
     */

    // 1)
    const float mx_act = std::reduce(out.begin(),out.end(),0.0f,[](const float a, const float b){ return std::max(a,b); });

    // 2)
    std::uniform_real_distribution<float> dis ( mx_act * stochastic_win_thres, mx_act );
    float win_act = dis(rgen);

    // 3)
    for (std::size_t idx = 0; idx < out.size(); ++idx)
        if (out[idx] + std::numeric_limits<float>::epsilon() >= win_act) {
            neurons[idx].adapt( sigmoid(1.0f - out[idx], weight_filter) );
            break;
        }

    // 4)
    const float act_sum = std::reduce(out.begin(),out.end());
    std::for_each(
        std::execution::par, // parallelization might move up to the simulation layer at some point
        neurons.begin(), neurons.end(),
        [&](auto &neuron) {
            const float sec_weight = sigmoid(1.0f - (out[neuron.id] / act_sum), weight_filter);
            neuron.adapt(sec_weight * common_learning_rate);
        }
    );


}

// interface function that allows the simulation environment to query the output ID of this io entity
std::size_t neuron_group_t::get_outp_id() const
{
    return params.id;
}

// interface function that allows the simulation environment to query the output size of this io entity
std::size_t neuron_group_t::get_outp_size() const
{
    return neurons.size();
}

// interface function that allows the simulation environment to query the input IDs required by this io entity
std::span<const std::size_t> neuron_group_t::get_inp_ids() const
{
    return { inp_ids.begin(), inp_ids.end() };
}

// interface function that can be used to report a status string to, e.g., the gui or a log
std::string neuron_group_t::status_str() const
{
    std::string status { "Neuron Group" };
    status += " | id: " + std::to_string(get_outp_id());
    status += "\n | neurons: " + std::to_string(get_neuron_count());
    status += " | representations: " + std::to_string(get_representation_count());
    status += " | synapses: " + std::to_string(get_synapse_count());
    status += " | max mm: " + std::to_string(get_max_mismatch());
    status += " | avg mm: " + std::to_string(get_avg_mismatch());
    status += " | max at: " + std::to_string(get_max_acc_theta());
    status += " | avg at: " + std::to_string(get_avg_acc_theta());
    return status;
}

/*
 *  introspection functions used by, e.g., visualization components
 */
const neuron_t & neuron_group_t::get_neuron(std::size_t idx) const
{
    return neurons[idx];
}

std::size_t neuron_group_t::get_neuron_count() const
{
    return neurons.size();
}

dendrite_t::seg_id_t neuron_group_t::get_max_representation_count() const
{
    dendrite_t::seg_id_t result = 0;
    for (const auto &neuron : neurons)
        result = std::max(result, neuron.get_representation_count());
    return result;
}

std::size_t neuron_group_t::get_representation_count() const
{
    std::size_t result = 0;
    for (const auto &neuron : neurons)
        result += neuron.get_representation_count();
    return result;
}

std::size_t neuron_group_t::get_synapse_count() const
{
    std::size_t result = 0;
    for (const auto &neuron : neurons)
        result += neuron.get_synapse_count();
    return result;
}

float neuron_group_t::get_max_mismatch() const
{
    float result = 0.0f;
    for (const auto &neuron : neurons) {
        std::size_t dc = neuron.get_dendrite_count();
        for (std::size_t di = 0; di < dc; ++di) {
            result = std::max(result, *std::ranges::max_element(neuron.get_dendrite(di).get_synapses().mismatch));
        }
    }
    return result;
}

float neuron_group_t::get_avg_mismatch() const
{
    float result = 0.0f;
    float cnt = 0.0f;
    for (const auto &neuron : neurons) {
        std::size_t dc = neuron.get_dendrite_count();
        for (std::size_t di = 0; di < dc; ++di) {
            auto &vec = neuron.get_dendrite(di).get_synapses().mismatch;
            result += std::reduce(vec.begin(), vec.end(), 0.0f);
            cnt += static_cast<float>(vec.size());
        }
    }
    return result / cnt;
}

float neuron_group_t::get_max_acc_theta() const
{
    float result = 0.0f;
    for (const auto &neuron : neurons) {
        std::size_t dc = neuron.get_dendrite_count();
        for (std::size_t di = 0; di < dc; ++di) {
            result = std::max(result, *std::ranges::max_element(neuron.get_dendrite(di).get_synapses().adapt_history));
        }
    }
    return result;
}

float neuron_group_t::get_avg_acc_theta() const
{
    float result = 0.0f;
    float cnt = 1.0f;
    for (const auto &neuron : neurons) {
        std::size_t dc = neuron.get_dendrite_count();
        for (std::size_t di = 0; di < dc; ++di) {
            auto &vec = neuron.get_dendrite(di).get_synapses().adapt_history;
            const float summed_theta = std::reduce(vec.begin(), vec.end(), 0.0f);
            assert(std::isnormal(summed_theta) + std::numeric_limits<float>::epsilon());
            result += summed_theta;
            cnt += static_cast<float>(vec.size());
        }
    }
    assert(std::isnormal(result) + std::numeric_limits<float>::epsilon());
    assert(std::isnormal(cnt));
    return result / cnt;
}
}
