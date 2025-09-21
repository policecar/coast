//
// Created by jk on 26.08.25.
//

#include "hd_ngm2_dendrite.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <ranges>
#include <utility>

#include "hd_ngm2_tools.h"

namespace ngm2 {
void dendrite_t::synapses_t::reserve(std::size_t size)
{
    permanence.reserve(size);
    mismatch.reserve(size);
    adapt_history.reserve(size);
    segment_idx.reserve(size);
    input_inc.reserve(size);
}

void dendrite_t::synapses_t::resize(std::size_t size)
{
    permanence.resize(size);
    mismatch.resize(size);
    adapt_history.resize(size);
    segment_idx.resize(size);
    input_inc.resize(size);
}

std::size_t dendrite_t::synapses_t::size() const
{
    return permanence.size();
}

dendrite_t::syn_tuple_ref_t dendrite_t::synapses_t::operator[](std::size_t idx)
{
    return {
        permanence[idx], mismatch[idx], adapt_history[idx], segment_idx[idx], input_inc[idx]
    };
}

dendrite_t::syn_tuple_t dendrite_t::synapses_t::operator[](std::size_t idx) const
{
    return {
        permanence[idx], mismatch[idx], adapt_history[idx], segment_idx[idx], input_inc[idx]
    };
}


constexpr dendrite_t::seg_id_t dendrite_t::calc_max_segment_idx(const seg_id_t max_branch_level)
{
    return (1 << (max_branch_level+1)) - 1;
}

dendrite_t::dendrite_t(params_t _params) :
    params                  ( std::move(_params) ),
    max_segment_idx         ( calc_max_segment_idx(params.max_branch_level) ),
    primary_learning_rate   ( params.default_primary_learning_rate   ),
    secondary_learning_rate ( params.default_secondary_learning_rate ),
    mismatch_smoothing      ( params.default_mismatch_smoothing      ),
    accumulated_theta_thres ( params.default_accumulated_theta_thres ),
    min_mismatch_deviation  ( params.default_min_mismatch_deviation  ),
    min_mismatch_percentage ( params.default_min_mismatch_percentage ),
    last_max_inp            ( 0.0f ),
    rgen                    ( params.rnd_seed                        )
{
    // initializing random synapses
    synapses.reserve(params.input_size * 2);
    synapses.resize(params.input_size);
    std::poisson_distribution<> rdis(static_cast<int>(100.0f * params.permanence_threshold * 1.f));
    //std::uniform_real_distribution<float> rdis(0.2f, params.permanence_threshold + 0.02f);
    for (std::size_t i = 0; i < params.input_size; ++i) {
        synapses.permanence[i]    = std::clamp(static_cast<float>(rdis(rgen)) / 100.0f, 0.0f, 1.0f);
//        synapses.permanence[i]    = rdis(rgen);
        synapses.mismatch[i]      = 0.0f;
        synapses.adapt_history[i] = 0.0f;
        synapses.segment_idx[i]   = 1;
        synapses.input_inc[i]     = 1;
    }
    // init segment structures
    segment_activity.resize(max_segment_idx + 1, 0.0f);
    segment_weights.resize(max_segment_idx + 1,  0.0f);
}

void dendrite_t::set_inp_func(partial_id_t id, std::function<sim::io_buffer::inp_buf_t()> inp_func)
{
    if (params.input_ids.contains(id))
        input_mem[id] = std::move(inp_func);
}

float dendrite_t::get_response()
{

/* Calculate the responses of all possible paths through the dendritic
 * branch and return the maximum response
 *
 * desired properties:
 * - permanences express how established a synaptic connection is
 * - input signals are included in the calculation if the permanence is
 *   above a given threshold (maybe add a probabilistic element to the decision)
 * - if the permanence is above threshold and the input is "low", the
 *   response should be negatively affected by it, i.e., having a synapse
 *   has to have a cost compared to not having a synapse. The difficulty
 *   is determining, what constitutes "low" in a given signal. Ideas:
 *   + use (1 - x)^(something) as probability for a penalty
 *   + use the permanence value or a derivative as penaltiy, as a
 *     strongly connected synapse is "more wrong" than a weakly connected
 *     one...
 * - biologically a dendrite reacts to "broad, intense input" with an
 *   increase in current leakage across its membrane, i.e., the time
 *   window for integration is reduced in order to detect simultaneous
 *   signals within a potentially noisy signal. How can/should this be
 *   emulated? Ideas:
 *   + use some form of enhancement for the strong input components
 *     (something like softmax, preferably simpler)
 *   + use (normalized )entropy as a measure of "noise level"
 *     (tho normalized entropy has a non-intuitive distribution of values ...
 *      effectively the only relevant range is 0.8 to 1.0 as soon as each
 *      channel has 1% noise. Only absolutely clean signals (lots of 0s)
 *      have values significantly lower than 0.8. If the signal has only
 *      "weak" components (no value > 0.3) than lowest entropy is around
 *      0.87, mnist characters range from 0.6 to 0.78, adding 1% noise
 *      increases minimum to 0.68, 5% noise raises minimum to 0.79)
 *      hypothesis: if a minimum noise floor of 5% can be guaranteed,
 *      signals with strong content range from 0.8 to 0.9, weak signals
 *      up to noise make up 0.9 to 1.0
 *   +
 */


    // clear current segment activities and perm counts
    std::ranges::fill(segment_activity, 0.0f);

    float inp_sum = 0.0f;
    float mx_inp  = 0.0f;
    for (const auto &partial_input_func : input_mem | std::views::values) {
        const auto [partial_input, pi_stats] = partial_input_func();
        inp_sum += pi_stats.sum;
        mx_inp   = std::max(mx_inp, pi_stats.max_val);
    }
    last_max_inp = mx_inp;
    if (!std::isnormal(inp_sum)) {
        //std::printf("ERROR: input sum in dendrite_t::get_response() is not normal.\n");
        return 0.0f;
    }


    // calculate segment activities
    const std::size_t syn_cnt = synapses.size();
    auto pi = input_mem.begin();
    auto [cur_input,cur_stats] = pi->second();
    auto cur_inp_it    = cur_input.begin();
    auto cur_inp_end   = cur_input.end();
    float nse = cur_stats.nse;
    float avg_nse = nse;
    std::uniform_real_distribution<float> dis1 { 0.0f, cur_stats.max_val / 2.0f };
    for (std::size_t i = 0; i < syn_cnt; ++i) {
        if (cur_inp_it == cur_inp_end) {
            ++pi;
            auto [new_input,new_stats] = pi->second();
            cur_input = new_input;
            cur_stats = new_stats;
            cur_inp_it  = cur_input.begin();
            cur_inp_end = cur_input.end();
            nse = std::min(cur_stats.nse,nse);
            avg_nse += nse;
            dis1 = std::uniform_real_distribution<float> {0.0f, cur_stats.max_val / 2.0f};
        }
        if (synapses.permanence[i] > params.permanence_threshold) {
            segment_activity[synapses.segment_idx[i]] += *cur_inp_it;
            // we want to punish this connection if the input is "low"
            // we want to punish higher permanence values more than lower
            const float inp_contrib   = *cur_inp_it / cur_stats.sum;
            const float perm_strength = (synapses.permanence[i] - params.permanence_threshold) / (1.0f - params.permanence_threshold);

            if (dis1(rgen) > *cur_inp_it) {
                segment_activity[synapses.segment_idx[i]] -= perm_strength * (1.0f - inp_contrib);
                if (segment_activity[synapses.segment_idx[i]] < 0.0f)
                    segment_activity[synapses.segment_idx[i]] = 0.0f;
            }

            // the thing below might be a bit too aggressive...
            /*
            if (dis1(rgen) > *cur_inp_it) {
                segment_activity[synapses.segment_idx[i]] -= synapses.permanence[i]/2.0;
            }
            */

        }
        cur_inp_it += synapses.input_inc[i];
    }
    avg_nse /= static_cast<float>(input_mem.size());

    // push activities and perm_cnts to the leafs
    const std::size_t leaf_begin = (max_segment_idx + 1) / 2;
    for (std::size_t si = 1; si < leaf_begin; ++si) {
        segment_activity[si * 2 + 0] += segment_activity[si];
        segment_activity[si * 2 + 1] += segment_activity[si];
    }

    // normalize and find maximum activity among the leaves
    // attenuate based on normalized shannon entropy
    float attenuation = 1.0f - sigmoid((nse - 0.8f) / 0.2f, {0.25f,0.5f});
    float max_activity = 0.0f;
    for (std::size_t si = leaf_begin; si <= max_segment_idx; ++si) {
        segment_activity[si] = std::clamp( segment_activity[si] * attenuation / inp_sum , 0.0f, 1.0f );
        max_activity = std::max(max_activity, segment_activity[si]);
    }

    return max_activity;
}

void dendrite_t::adapt_synapses(const float max_activity, const float weight)
{
    if (!std::isnormal(max_activity)) {
        return;
    }

    // calculate segment weights
    std::ranges::fill(segment_weights, 0.0f);
    bool max_response_seen       = false;
    constexpr float eps          = std::numeric_limits<float>::epsilon();
    const std::size_t leaf_begin = (max_segment_idx + 1) / 2;
    for (std::size_t si = leaf_begin; si <= max_segment_idx; ++si) {
        if ((!max_response_seen) && (segment_activity[si] + eps >= max_activity)) {
            max_response_seen = true;
            segment_weights[si] = weight * primary_learning_rate;
            continue;
        }
        segment_weights[si] = segment_activity[si] * weight * secondary_learning_rate / max_activity;
    }

    // push weights and activity from the leaves to the root of the dendritic tree
    // using always the maximum of the two childs of a node
    for (std::size_t level_start = leaf_begin; level_start > 1; level_start /= 2)
        for (std::size_t si = level_start; si < level_start * 2; si += 2) {
            segment_weights[si / 2]  = std::max( segment_weights[si],  segment_weights[si+1]  );
            segment_activity[si / 2] = std::max( segment_activity[si], segment_activity[si+1] );
        }

    // perform adaptation of synapses and update mismatch
    const std::size_t syn_cnt = synapses.size();
    auto pi = input_mem.begin();
    auto [cur_input,cur_stats] = pi->second();
    auto cur_inp_it    = cur_input.begin();
    auto cur_inp_end   = cur_input.end();
    float nse = cur_stats.nse;
    float attenuation = 1.0f - sigmoid((nse - 0.8f) / 0.2f);
    for (std::size_t i = 0; i < syn_cnt; ++i) {
        if (cur_inp_it == cur_inp_end) {
            ++pi;
            assert(pi != input_mem.end());
            auto [new_input,new_stats] = pi->second();
            cur_input = new_input;
            cur_stats = new_stats;
            cur_inp_it    = cur_input.begin();
            cur_inp_end   = cur_input.end();
            //nse = std::min(cur_stats.nse,nse);
            nse = cur_stats.nse;
            attenuation = 1.0f - sigmoid((nse - 0.8f) / 0.2f);
        }
        // we want to learn strongly when the signal is strong and
        // if the signal is clear in total
        const float high_thres = ((cur_stats.avg) / 2.0f) + std::numeric_limits<float>::epsilon();
        const float theta = std::clamp( segment_weights[ synapses.segment_idx[i] ] * ( *cur_inp_it > high_thres ?
                                (*cur_inp_it - high_thres) / (1.0f - high_thres) :
                                (high_thres - *cur_inp_it) / high_thres
                            ) * attenuation, 0.0f, 1.0f);

        synapses.permanence[i] = std::clamp(synapses.permanence[i] * (1.0f - theta) + (*cur_inp_it > high_thres ? theta : 0.0f), 0.0f, 1.0f);
        synapses.adapt_history[i] += theta;
        // calculate mismatch
        float mismatch = 0.0f;
        const float act_ratio = segment_activity[ synapses.segment_idx[i] ] / max_activity;
        if (act_ratio > 0.8f) {
            const float inp_ratio = *cur_inp_it / last_max_inp;
            mismatch = synapses.permanence[i] > params.permanence_threshold ? 1.0f - inp_ratio : inp_ratio;
            mismatch *= act_ratio;
            synapses.mismatch[i] = synapses.mismatch[i] * (1.0f - mismatch_smoothing) + mismatch * mismatch_smoothing;
        }
        cur_inp_it += synapses.input_inc[i];
    }
}

void dendrite_t::adapt_branches()
{
    const float syn_cnt_f = static_cast<float>(synapses.size());
    const float mm_avg = std::reduce(synapses.mismatch.begin(), synapses.mismatch.end(), 0.0f) / syn_cnt_f;
    const float mm_std = std::transform_reduce(synapses.mismatch.begin(), synapses.mismatch.end(),
                            0.0f, std::plus<>(),
                            [mm_avg](const float mm_val) {
                                return std::pow(mm_avg - mm_val, 2.0f);
                            }) / syn_cnt_f;
    const float mm_thres = mm_avg + mm_std * min_mismatch_deviation + 1.0f / static_cast<float>(params.input_size);
    std::size_t mm_cnt = 0;
    const std::size_t syn_cnt = synapses.size();
    for (std::size_t i = 0; i < syn_cnt; ++i)
        if ((synapses.adapt_history[i]       >= accumulated_theta_thres) &&
            (synapses.mismatch[i]            >= mm_thres)                &&
            (synapses.segment_idx[i] * 2 + 1 <= max_segment_idx )          ) ++mm_cnt;

    if (static_cast<float>(mm_cnt) < static_cast<float>(params.input_size) * min_mismatch_percentage)
        return;

    // expand and update synapse memory
    std::size_t last_synapse_idx = syn_cnt - 1;
    synapses.resize(syn_cnt + mm_cnt);

    //std::poisson_distribution<> rdis(static_cast<int>(10.0f / params.permanence_threshold));
    std::uniform_real_distribution<float> rdis(-0.1f,0.1f);

    // update from back to front
    std::size_t cur_idx = synapses.size() - 1;
    while (cur_idx > last_synapse_idx) {
        // copy synapse
        synapses[cur_idx--] = synapses[last_synapse_idx];
        // check if the synapse does not need to be cloned
        if ((synapses.adapt_history[last_synapse_idx]       < accumulated_theta_thres) ||
            (synapses.mismatch[last_synapse_idx]            < mm_thres)                ||
            (synapses.segment_idx[last_synapse_idx] * 2 + 1 > max_segment_idx)           )
        {
            --last_synapse_idx;
            continue;
        }
        // we need to clone this synapse
        synapses[cur_idx] =  synapses[last_synapse_idx];
        // update the segment index of the cloned synapses
        const seg_id_t old_segment_idx = synapses.segment_idx[cur_idx];
        synapses.segment_idx[cur_idx + 0] = old_segment_idx * 2 + 0;
        synapses.segment_idx[cur_idx + 1] = old_segment_idx * 2 + 1;

        // clean learning history
        synapses.adapt_history[cur_idx + 0] = 0.0f;
        synapses.adapt_history[cur_idx + 1] = 0.0f;

        // "wiggle" permanences
        const float old_mm   = synapses.mismatch[cur_idx];
        const float old_perm = synapses.permanence[cur_idx];
        /*
        synapses.permanence[cur_idx + 0] = std::clamp(
                                               old_perm - static_cast<float>(rdis(rgen)) / 100.0f * old_mm,
                                               0.0f, 1.0f
                                           );
                                           */
        synapses.permanence[cur_idx + 0] = std::clamp(old_perm + rdis(rgen),
                                               0.0f, 1.0f
                                           );
        synapses.permanence[cur_idx + 1] = std::clamp(
                                               old_perm + rdis(rgen),
                                               0.0f, 1.0f
                                           );

        // clear mismatch values
        synapses.mismatch[cur_idx + 0] = 0.0f;
        synapses.mismatch[cur_idx + 1] = 0.0f;

        // disable input advancement of "lower" synapse
        synapses.input_inc[cur_idx + 0] = 0;

        // advance to next synapse
        --cur_idx;
        --last_synapse_idx;
    }
}

std::vector<uint8_t> dendrite_t::get_leaf_mask() const
{
    std::vector<uint8_t> leaf_detection(max_segment_idx+1,0);

    // mark existing segments
    for (const auto seg_idx : synapses.segment_idx) {
        leaf_detection[seg_idx] = 1;
    }

    // unmark inner segments
    seg_id_t level = (max_segment_idx+1) / 2;
    while (level > 1) {
        for (seg_id_t si = level; si < level * 2; ++si)
            if (leaf_detection[si] == 1) {
                seg_id_t lower = si;
                while (lower > 0) {
                    lower /= 2;
                    leaf_detection[lower] = 0;
                }
            }
        level /= 2;
    }

    return leaf_detection;
}

dendrite_t::seg_id_t dendrite_t::get_representation_count() const
{
    const auto mask = get_leaf_mask();
    return static_cast<seg_id_t>( std::count(mask.begin(),mask.end(),1) );
}

std::vector<float> dendrite_t::get_representation(seg_id_t idx) const
{
    std::vector<float> result;
    result.reserve(params.input_size);

    auto leaf_mask = get_leaf_mask();

    // find branch
    std::size_t i = 0;
    idx += 1;
    for (;i<leaf_mask.size(); ++i) {
        idx -= leaf_mask[i];
        if (idx == 0)
            break;
    }

    // clear mask and mark entire branch
    std::ranges::fill(leaf_mask, 0);
    leaf_mask[i] = 1;
    while (i > 1) {
        i /= 2;
        leaf_mask[i] = 1;
    }

    // conditionally copy permanences into result
    std::size_t syn_cnt = synapses.size();
    for (std::size_t si = 0; si < syn_cnt; ++si) {
        if (leaf_mask[ synapses.segment_idx[si] ] == 1)
            result.push_back(synapses.permanence[si]);
    }

    return result;

}

std::size_t dendrite_t::get_representation_size() const
{
    return params.input_size;
}

std::size_t dendrite_t::get_synapse_count() const
{
    return synapses.size();
}

const dendrite_t::synapses_t & dendrite_t::get_synapses() const
{
    return synapses;
}

dendrite_t::seg_id_t dendrite_t::get_max_segment_idx() const
{
    return max_segment_idx;
}

std::size_t dendrite_t::get_input_size() const
{
    return params.input_size;
}
}
