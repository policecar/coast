//
// Created by jk on 26.08.25.
//
#include <hd_ngm2_neuron_group.h>

#include <utility>
#include <algorithm>
#include <cassert>
#include <execution>

namespace ngm2 {

neuron_group_t::neuron_group_t(params_t  _params) :
    params                   ( std::move( _params ) ),
    local_inhibition_strength( params.default_local_inhibition_strength ),
    common_learning_rate     ( params.default_common_learning_rate      ),
    weight_filter            ( params.default_weight_filter             ),
    rgen                     (params.neuron_params[0].dendrite_params[0].rnd_seed)
{
    std::set<partial_id_t> tmp;
    // create neurons
    neurons.reserve(params.neuron_params.size());
    for (const auto &np : params.neuron_params) {
        neurons.emplace_back(np);
        for (const auto &dp : np.dendrite_params) {
            tmp.insert_range(dp.input_ids);
        }
    }
    inp_ids.resize(tmp.size());
    std::ranges::copy(tmp,inp_ids.begin());
}

void neuron_group_t::set_outp_func(std::function<std::span<float>()> outp_func)
{
    output_mem = std::move(outp_func);
}

void neuron_group_t::set_inp_func(partial_id_t id, const std::function<sim::io_buffer::inp_buf_t()> &inp_func)
{
    for (auto &neuron : neurons)
        neuron.set_inp_func(id,inp_func);
}

void neuron_group_t::process()
{
    assert(output_mem != nullptr);
    std::span<float> out = output_mem();
    assert(out.size() == neurons.size());
    std::for_each(
        std::execution::par,
        neurons.begin(),neurons.end(),
        [&](auto &neuron) {
            std::size_t idx = &neuron - neurons.data();
            out[idx] = neuron.get_response();
        }
    );

    //softmax(out,local_inhibition_strength);
    local_inhibition4(out,local_inhibition_strength,0.00f);

    const float mx_act = std::reduce(out.begin(),out.end(),0.0f,[](const float a, const float b){ return std::max(a,b); });

    std::uniform_real_distribution<float> dis ( mx_act * 0.8f, mx_act );
    float win_act = dis(rgen);
    for (std::size_t idx = 0; idx < out.size(); ++idx)
        if (out[idx] + std::numeric_limits<float>::epsilon() >= win_act) {
            neurons[idx].adapt( sigmoid(1.0f - out[idx], weight_filter) );
            break;
        }

    const float act_sum = std::reduce(out.begin(),out.end());

    std::for_each(
        std::execution::par,
        neurons.begin(), neurons.end(),
        [&](auto &neuron) {
            std::size_t idx = &neuron - neurons.data();

            const float sec_weight = sigmoid(1.0f - (out[idx] / act_sum), weight_filter);

            neuron.adapt(sec_weight * common_learning_rate);
        }
    );


    /*

    auto mx_iter = std::ranges::max_element( out );
    if (mx_iter == out.end())
        return;
    const auto mx_val  = *mx_iter;
    const auto mx_idx  = std::distance(out.begin(), mx_iter);

    const float avg = std::reduce(out.begin(),out.end()) / static_cast<float>(out.size());

    // debug
    last_max = mx_val;
    avg_max = avg_max * (1.0f - 0.001f) + last_max * 0.001f;

    const float high_thres = avg / 2.0f;
    const float weight = sigmoid( 1.0 - ((mx_val - high_thres) / (1.0f - high_thres)), weight_filter);

    neurons[mx_idx].adapt(weight);

    std::for_each(
        std::execution::par,
        neurons.begin(), neurons.end(),
        [&](auto &neuron) {
            std::size_t idx = &neuron - neurons.data();

            const float sec_weight = out[idx] > high_thres ?
                    sigmoid(1.0 - ((out[idx] - high_thres) / (1.0f - high_thres)), weight_filter) :
                    sigmoid(1.0 - ((high_thres - out[idx]) / high_thres), weight_filter);

            neuron.adapt(sec_weight * common_learning_rate);
        }
    );
    */

}

std::size_t neuron_group_t::get_outp_id() const
{
    return params.id;
}

std::size_t neuron_group_t::get_outp_size() const
{
    return neurons.size();
}

std::span<const std::size_t> neuron_group_t::get_inp_ids() const
{
    return { inp_ids.begin(), inp_ids.end() };
}

std::string neuron_group_t::status_str() const
{
    std::string status { "Neuron Group" };
    status += " | id: " + std::to_string(get_outp_id());
    status += "\n | neurons: " + std::to_string(get_neuron_count());
    status += " | representations: " + std::to_string(get_representation_count());
    status += " | synapses: " + std::to_string(get_synapse_count());
    status += " | last_max: " + std::to_string(last_max);
    status += " | avg_max: " + std::to_string(avg_max);
    status += " | max mm: " + std::to_string(get_max_mismatch());
    status += " | avg mm: " + std::to_string(get_avg_mismatch());
    status += " | max at: " + std::to_string(get_max_acc_theta());
    status += " | avg at: " + std::to_string(get_avg_acc_theta());
    return status;
}

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
