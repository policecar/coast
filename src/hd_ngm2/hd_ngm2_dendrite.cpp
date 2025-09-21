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

/*
 * Helper functions to manage SOA data layout
 */
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



/*
 * helper function to determine the maximum index of the dendritic segments that could occur in our dendritic branch
 */
constexpr dendrite_t::seg_id_t dendrite_t::calc_max_segment_idx(const seg_id_t max_branch_level)
{
    return (1 << (max_branch_level+1)) - 1;
}

/*
 * main initialization of our dendritic branch model
 */
dendrite_t::dendrite_t(params_t _params) :
    params                  ( std::move(_params)                            ),
    max_segment_idx         ( calc_max_segment_idx(params.max_branch_level) ),
    primary_learning_rate   ( params.default_primary_learning_rate          ),
    secondary_learning_rate ( params.default_secondary_learning_rate        ),
    mismatch_smoothing      ( params.default_mismatch_smoothing             ),
    accumulated_theta_thres ( params.default_accumulated_theta_thres        ),
    min_mismatch_deviation  ( params.default_min_mismatch_deviation         ),
    min_mismatch_percentage ( params.default_min_mismatch_percentage        ),
    mismatch_act_thres      ( params.default_mismatch_act_thres             ),
    last_max_inp            ( 0.0f                                          ),
    rgen                    ( params.rnd_seed                               )
{
    // initializing random synapses
    synapses.reserve(params.input_size * 2);
    synapses.resize(params.input_size);

    // We use a poisson distribution around the permanence threshold.
    // As the available poisson distribution operates on integers rather than floats we need to
    // transform the permanence threshold ([0..1]) to a suitable integer range ([0..100]) and
    // transform the value back when using the distribution...
    std::poisson_distribution<> poisson_dis(static_cast<int>(100.0f * params.permanence_threshold));
    for (std::size_t i = 0; i < params.input_size; ++i) {
        synapses.permanence[i]    = std::clamp(static_cast<float>(poisson_dis(rgen)) / 100.0f, 0.0f, 1.0f);
        synapses.mismatch[i]      = 0.0f;
        synapses.adapt_history[i] = 0.0f;
        synapses.segment_idx[i]   = 1;
        synapses.input_inc[i]     = 1;
    }

    // Internally we represent the binary tree structure of the dendritic branch as linear arrays of appropriate size
    segment_activity.resize(max_segment_idx + 1, 0.0f);
    segment_weights.resize(max_segment_idx + 1,  0.0f);
}

/*
 * Interface function to receive for each input id the respective input function. Since we might get offered input
 * functions that this dendritic branch might not need, we check if we are interested in the ID and only then store
 * the input function in the hashmap input_mem
 */
void dendrite_t::set_inp_func(partial_id_t id, std::function<sim::io_buffer::inp_buf_t()> inp_func)
{
    if (params.input_ids.contains(id))
        input_mem[id] = std::move(inp_func);
}


/*
 * main function that models the response of a dendritic branch to the current input of its input space(s)
 */
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

    // clear current segment activities
    std::ranges::fill(segment_activity, 0.0f);

    // gather sum and max of all partial inputs
    // sum will be used to normalize the response at the end, but we calculate it
    // here to allow for an early exit if the input is entirely zero or malformed
    // max will only be used later during adaptation...
    float inp_sum = 0.0f;
    last_max_inp  = 0.0f;
    for (const auto &partial_input_func : input_mem | std::views::values) {
        const auto [partial_input, pi_stats] = partial_input_func();
        inp_sum += pi_stats.sum;
        last_max_inp = std::max(last_max_inp, pi_stats.max_val);
    }

    // early exit if the input is zero or malformed
    if (!std::isnormal(inp_sum)) {
        //std::printf("ERROR: input sum in dendrite_t::get_response() is not normal.\n");
        return 0.0f;
    }

    /*
    * The algorithm to calculate the responses of all possible paths through the dendritic
    * branch is as follows:
    * 1) calculate the activity of each individual dendritic segment in the binary dendritic tree
    * 2) push the activities from the root segment to all leaf segments and then check the leaf
    *    segments for the highest activity.
    */

    /* 1)
     * The synapses of the dendritic branch are not actually stored in a "tree-shaped" data structure
     * but instead live in multiple contiguous arrays contained in the single synapses struct.
     * The attribution of a single synapse to a specific dendritic segment is given by its entry in the
     * segment_idx array. The synapse's association with a given input is determined indirectly by its
     * position in the SOA. Input synapses that process a common input are grouped together in the SOA
     * and control over the input reference is facilitated by the input_inc array. Together, determining
     * the activity of all dendritic segments is achieved by a single sweep through the synapses SOA
     */
    const std::size_t syn_cnt = synapses.size();

    // getting first partial input from hashmap
    // and getting cur_input array and statistics for the first partial input
    auto pi = input_mem.begin();
    auto [cur_input,cur_stats] = pi->second();

    // set up start and end of current input
    auto cur_inp_it    = cur_input.begin();
    auto cur_inp_end   = cur_input.end();

    // getting the normalized shannon entropy statistic for first partial input
    float nse = cur_stats.nse;

    // setting up a uniform random distribution that will be used to stochastically determine if an input is "low"
    std::uniform_real_distribution<float> dis1 { 0.0f, cur_stats.max_val / 2.0f };

    // linear sweep through all synapses
    for (std::size_t i = 0; i < syn_cnt; ++i) {

        // if the current partial input has ended, we continue with the next one and update our variables accordingly
        if (cur_inp_it == cur_inp_end) {
            ++pi;
            auto [new_input,new_stats] = pi->second();
            cur_input = new_input;
            cur_stats = new_stats;
            cur_inp_it  = cur_input.begin();
            cur_inp_end = cur_input.end();
            nse = std::min(cur_stats.nse,nse);
            dis1 = std::uniform_real_distribution<float> {0.0f, cur_stats.max_val / 2.0f};
        }

        /* we only process an input if the "permanence" [0..1] of the corresponding synapse is above a given
        *  permanence threshold (e.g., 0.3). The concept of "permanence" stems from Hawkins et al. (Numenta) and
        *  represents if and how well an axon has made contact with a synapse. It does NOT represent a connection
        *  weight as it would be used, e.g., in a perceptron. Instead, it is binary. If a connection is made (i.e.,
        *  the permanence is above threshold) the input is taken in "as is" (see 1.1).
        *  However, we also need to encode the information that a synaptic connection might be present / strong while
        *  there is no input. In this case, we need to "punish" this connection. From a biological perspective this
        *  idea resembles that of a "leaky synapse" that will reduce the cell membrane potential if no corresponding
        *  strong input is present. Another perspective would be: there has to be a metabolical cost to having a synapse
        *  that is not used properly. As it is diffcult to state when an input is actually "low", we follow a stochastic
        *  approach and decide if the input was low via a uniform distribution between 0 and max_input_value / 2. (see 1.2)
        */
        if (synapses.permanence[i] > params.permanence_threshold) {
            // 1.1
            segment_activity[synapses.segment_idx[i]] += *cur_inp_it;

            // 1.2
            if (dis1(rgen) > *cur_inp_it) {
                const float inp_contrib   = *cur_inp_it / cur_stats.sum;
                const float perm_strength = (synapses.permanence[i] - params.permanence_threshold) / (1.0f - params.permanence_threshold);
                segment_activity[synapses.segment_idx[i]] -= perm_strength * (1.0f - inp_contrib);
                if (segment_activity[synapses.segment_idx[i]] < 0.0f)
                    segment_activity[synapses.segment_idx[i]] = 0.0f;
            }

        }
        // we advance the current input iterator only if the respective input signal is not needed by further
        // synapses, i.e., the values in input_inc are either 0 or 1. For a group of synapses that all receive input
        // from a given input dimension, all input_inc values are 0 except from the last synapse of the group.
        cur_inp_it += synapses.input_inc[i];
    }

    // 2) push activities to the leafs
    const std::size_t leaf_begin = (max_segment_idx + 1) / 2;
    for (std::size_t si = 1; si < leaf_begin; ++si) {
        segment_activity[si * 2 + 0] += segment_activity[si];
        segment_activity[si * 2 + 1] += segment_activity[si];
    }

    // determine the maximum activity among the leafs of the dendritic branch
    // and attenuate the activity if the normalized shannon entropy (NSE) indicates that the input is basically noise.
    // For inputs that carry information the NSE ranges mostly between 0.8 and 0.9. From there on (0.9 to 1.0) inputs
    // are likely to be predominantly noise.
    // Please note that the segment_activity is used below in the adapt_synapses function. Hence the attenuation needs
    // to be applied to every element and not just the max_activity
    float attenuation = 1.0f - sigmoid((nse - 0.8f) / 0.2f, {0.25f,0.5f});
    float max_activity = 0.0f;
    for (std::size_t si = leaf_begin; si <= max_segment_idx; ++si) {
        segment_activity[si] = std::clamp( segment_activity[si] * attenuation / inp_sum , 0.0f, 1.0f );
        max_activity = std::max(max_activity, segment_activity[si]);
    }

    return max_activity;
}

/*
 * the main function that models the adaptation of a dendritic branch
 */
void dendrite_t::adapt_synapses(const float max_activity, const float weight)
{
    // early return if the max_activity is somehow "broken" or zero
    if (!std::isnormal(max_activity)) {
        return;
    }

    /*
     * For all dendritic paths through the dendritic branch we need to determine
     * for each segment the maximum "leaf" activity it participates in, and determine
     * a weight that regulates how strongly the synapses on that segment adapt to
     * the current input.
     * To this end we first (see 1) determine the weights along the leaves, where
     * the leaf that corresponds to the maximum activity (if the maximum activity is
     * actually present in this dendritic branch - it may not) will get a weight proportional
     * to a primary learning rate, while all other leafs receive a weight that is proportional
     * to a secondary learning rate and their activity relative to the maximum activity.
     * Subsequently, the weights and the leaf activities are pushed down towards the root
     * segment of the dendritic branch always using the maximum values from the two possible
     * child segments (see 2).
     */

    // 1 calculate segment weights
    std::ranges::fill(segment_weights, 0.0f);
    bool max_response_seen       = false;
    constexpr float eps          = std::numeric_limits<float>::epsilon();
    const std::size_t leaf_begin = (max_segment_idx + 1) / 2;
    for (std::size_t si = leaf_begin; si <= max_segment_idx; ++si) {
        if ((!max_response_seen) && (segment_activity[si] + eps >= max_activity)) {
            max_response_seen = true;
            segment_weights[si] = weight * primary_learning_rate;
        } else {
            segment_weights[si] = segment_activity[si] * weight * secondary_learning_rate / max_activity;
        }
    }

    // 2 push weights and activity from the leaves to the root of the dendritic tree
    // using always the maximum of the two childs of a node
    for (std::size_t level_start = leaf_begin; level_start > 1; level_start /= 2)
        for (std::size_t si = level_start; si < level_start * 2; si += 2) {
            segment_weights[si / 2]  = std::max( segment_weights[si],  segment_weights[si+1]  );
            segment_activity[si / 2] = std::max( segment_activity[si], segment_activity[si+1] );
        }

    /*
     * Similar to the calculation of the dendritic branch activity we adapt the synapses of the dendritic
     * branch within one sweep through all synapses for all partial inputs.
     * At its core we want to increase the permanence value of a synapse if the corresponding input value is
     * close to 1 and decrease the permanence value if it is close to zero. In addition, we only want to adapt
     * the permanence if the partial input signal is "clear" as opposed to "noisy" (see 3).
     * Lastly, we collect some statistical information to (later) decide if a synapse should be "cloned" and moved
     * from a lower dendritic segment towards a higher one (see 4.1 & 4.2)
     */
    const std::size_t syn_cnt = synapses.size();

    // getting first partial input from hashmap
    // and getting cur_input array and statistics for the first partial input
    auto pi = input_mem.begin();
    auto [cur_input,cur_stats] = pi->second();

    // set up start and end of current input
    auto cur_inp_it    = cur_input.begin();
    auto cur_inp_end   = cur_input.end();

    // calculate an attenuation factor depending on the normalized shannon entropy of this partial input
    // (see also description of the attenuation in the get_response method)
    float attenuation = 1.0f - sigmoid((cur_stats.nse - 0.8f) / 0.2f);

    // linear sweep through all synapses
    for (std::size_t i = 0; i < syn_cnt; ++i) {

        // if the current partial input has ended, we continue with the next one and update our variables accordingly
        if (cur_inp_it == cur_inp_end) {
            ++pi;
            auto [new_input,new_stats] = pi->second();
            cur_input = new_input;
            cur_stats = new_stats;
            cur_inp_it    = cur_input.begin();
            cur_inp_end   = cur_input.end();
            attenuation = 1.0f - sigmoid((cur_stats.nse - 0.8f) / 0.2f);
        }

        // 3 we want to learn strongly when the particular input is either near 1 or near 0 and
        // if the partial input is not noise
        const float high_thres = ((cur_stats.avg) / 2.0f) + std::numeric_limits<float>::epsilon();
        const float theta = std::clamp( segment_weights[ synapses.segment_idx[i] ] * ( *cur_inp_it > high_thres ?
                                (*cur_inp_it - high_thres) / (1.0f - high_thres) :
                                (high_thres - *cur_inp_it) / high_thres
                            ) * attenuation, 0.0f, 1.0f);

        synapses.permanence[i] = std::clamp(synapses.permanence[i] * (1.0f - theta) + (*cur_inp_it > high_thres ? theta : 0.0f), 0.0f, 1.0f);

        // 4.1 we collect some statistical information on the strength of our permanence adaptation. We need this information
        // below in the adapt branches function to decide whether or not to move the synapse to a higher dendritic segment
        synapses.adapt_history[i] += theta;

        // 4.2 the adaptation history is not enough to decide if a synapse should "branch". Hence, we also calculate a
        // mismatch heuristic that describes how well the permanence value of the synapse agrees with the activity of
        // the input. However, we only collect this information if the corresponding segment activity is reasonably high.
        // Furthermore, the mismatch value is implemented as IIR-Filter that emphasizes more recent mismatches.
        const float act_ratio = segment_activity[ synapses.segment_idx[i] ] / max_activity;
        if (act_ratio >= mismatch_act_thres) {
            const float inp_ratio = *cur_inp_it / last_max_inp;
            float mismatch = synapses.permanence[i] > params.permanence_threshold ? 1.0f - inp_ratio : inp_ratio;
            mismatch *= act_ratio;
            synapses.mismatch[i] = synapses.mismatch[i] * (1.0f - mismatch_smoothing) + mismatch * mismatch_smoothing;
        }

        // we advance the current input iterator only if the respective input signal is not needed by further
        // synapses, i.e., the values in input_inc are either 0 or 1. For a group of synapses that all receive input
        // from a given input dimension, all input_inc values are 0 except from the last synapse of the group.
        cur_inp_it += synapses.input_inc[i];
    }
}

/*
 * main function modeling the growth of the dendritic branch. All synapses of a dendritic branch start out at the
 * base dendritic segment. If a synapse turns out to be "ambiguous", the synapse is cloned and moved to the two child
 * dendritic segments with respect to the synapse's current dendritic segment. "Ambiguity" of a synapse is determined
 * by the synapse's mismatch value and its adaptation history.
 */
void dendrite_t::adapt_branches()
{
    // We first count the number of synapses that are ambiguous.
    // To this end we check the mismatch value against a threshold that is based on the mean and standard deviation
    // of all mismatch values in the dendritic branch
    const float syn_cnt_f = static_cast<float>(synapses.size());
    const float mm_avg = std::reduce(synapses.mismatch.begin(), synapses.mismatch.end(), 0.0f) / syn_cnt_f;
    const float mm_std = std::transform_reduce(synapses.mismatch.begin(), synapses.mismatch.end(),
                            0.0f, std::plus<>(),
                            [mm_avg](const float mm_val) {
                                return std::pow(mm_avg - mm_val, 2.0f);
                            }) / syn_cnt_f;
    const float mm_thres = mm_avg + mm_std * min_mismatch_deviation + 1.0f / static_cast<float>(params.input_size);

    // synapses count as ambiguous if they have accumulated enough "adaptation effort" (see 1.1), if their mismatch
    // value is significantly higher than the mean mismatch value plus a minimum absolute (1/N) to avoid weird edge case
    // behavior (see 1.2), and if the synapse is not yet on the highest dendritic segment allowed for this dendritic
    // branch (see 1.3)
    std::size_t mm_cnt = 0;
    const std::size_t syn_cnt = synapses.size();
    for (std::size_t i = 0; i < syn_cnt; ++i)
        if ((synapses.adapt_history[i]       >= accumulated_theta_thres) &&  // 1.1
            (synapses.mismatch[i]            >= mm_thres)                &&  // 1.2
            (synapses.segment_idx[i] * 2 + 1 <= max_segment_idx )          ) // 1.3
        {
            ++mm_cnt;
        }

    // early exit if not enough synapses are ambiguous
    if (static_cast<float>(mm_cnt) < static_cast<float>(params.input_size) * min_mismatch_percentage)
        return;

    /*
     * Now that we know the number of synapses that we want to clone and move we can expand the synapse memory. Please
     * note that synapses is a SOA and hence the operation is rather costly. This is the primary reason we counted
     * all ambigous synapses above before (see 2.1).
     * We efficiently update and copy the synapse array by copying synapses from the old end of the data structure to
     * the new end (see 2.2) while checking that the respective synapse needs NOT to be cloned (see 2.3). If the
     * synapse needs to be cloned, we proceed (see 2.4) and update both the moved and the cloned synapse:
     * - update the segment idx
     * - clean the learning history and mismatch values
     * - wiggle the permanence values
     * - disable input advancement of "lower" synapse (lower == lower index)
     */

    // 2.1 expand and update synapse memory
    std::size_t last_synapse_idx = syn_cnt - 1;
    synapses.resize(syn_cnt + mm_cnt);

    // uniform random distribution required to "wiggle" the permanence values of the moved synapses
    std::uniform_real_distribution<float> rdis(-0.1f,0.1f);

    // 2.2 update from back to front
    std::size_t cur_idx = synapses.size() - 1;
    while (cur_idx > last_synapse_idx) {
        // copy synapse
        synapses[cur_idx--] = synapses[last_synapse_idx];
        // 2.3 check if the synapse does not need to be cloned, otherwise go ahead and clone
        if (((synapses.adapt_history[last_synapse_idx]       >= accumulated_theta_thres) &&
             (synapses.mismatch[last_synapse_idx]            >= mm_thres)                &&
             (synapses.segment_idx[last_synapse_idx] * 2 + 1 <= max_segment_idx )          ) == false)
        {
            --last_synapse_idx;
            continue;
        }
        // 2.4 we need to clone this synapse
        synapses[cur_idx] =  synapses[last_synapse_idx];
        // update the segment index of the cloned synapses
        const seg_id_t old_segment_idx = synapses.segment_idx[cur_idx];
        synapses.segment_idx[cur_idx + 0] = old_segment_idx * 2 + 0;
        synapses.segment_idx[cur_idx + 1] = old_segment_idx * 2 + 1;

        // clean learning history
        synapses.adapt_history[cur_idx + 0] = 0.0f;
        synapses.adapt_history[cur_idx + 1] = 0.0f;

        // clear mismatch values
        synapses.mismatch[cur_idx + 0] = 0.0f;
        synapses.mismatch[cur_idx + 1] = 0.0f;

        // "wiggle" permanences
        const float old_perm = synapses.permanence[cur_idx];
        synapses.permanence[cur_idx + 0] = std::clamp(old_perm + rdis(rgen),
                                               0.0f, 1.0f
                                           );
        synapses.permanence[cur_idx + 1] = std::clamp(
                                               old_perm + rdis(rgen),
                                               0.0f, 1.0f
                                           );

        // disable input advancement of "lower" synapse
        synapses.input_inc[cur_idx + 0] = 0;

        // advance to next synapse
        --cur_idx;
        --last_synapse_idx;
    }
}

/*
 *  introspection functions used by, e.g., visualization components
 */
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
