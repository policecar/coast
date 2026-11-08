//
// Created by jk on 26.08.25.
//

#include "hd_ngm2/hd_ngm2_dendrite.h"

#include <algorithm>
#include <cassert>
#include <format>
#include <limits>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <utility>

#include "hd_ngm2/hd_ngm2_tools.h"
#include "ngm/cold_init.h"
#include "logging/state_checks.h"
#include "logging/state_io.h"
#include "logging/rng_state_io.h"
#include "observe/observe_accessor.h"

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
dendrite_t::dendrite_t(const cfg_store &cs, const std::string &type, int stream_index, seed_t parent_seed) :
    params                  ( [&cs,&type]() -> dendrite_params_t
                              {
                                  dendrite_params_t p {};
                                  if (const auto it = cs.dendrites.find(type); it != cs.dendrites.cend())
                                      p = it->second;
                                  // Validated here, before max_segment_idx shifts by
                                  // max_branch_level below (SPEC.md M.0.6 c: reject,
                                  // never compute something else silently).
                                  // L <= 4, as in ngm3 and the ngm25 GPU path: deeper
                                  // trees are not a model configuration in use, and
                                  // every step walks all 2^(L+1)-1 segment slots
                                  // (depth 15, the u16 limit, cost ~25x depth 3, #237).
                                  if (p.max_branch_level > 4)
                                      throw std::runtime_error("ngm2: max_branch_level must be <= 4; got "
                                                               + std::to_string(p.max_branch_level));
                                  // Same range as ngm3/ngm25: below 0.01 the initial
                                  // Poisson mean int(100*tau) is 0 (a precondition
                                  // violation); at 1 or above nothing can connect.
                                  // Checked as the float it runs as (perm_thres_): in
                                  // double, a tau just below 1 passed and ran as 1.0f.
                                  if (const float tau = static_cast<float>(p.permanence_threshold);
                                      !(tau >= 0.01f && tau < 1.0f))
                                      throw std::runtime_error("ngm2: permanence_threshold must be in [0.01, 1.0); got "
                                                               + std::to_string(p.permanence_threshold));
                                  return p;
                              }()                                           ),
    input_size              (0                                              ),
    max_segment_idx         ( calc_max_segment_idx(params.max_branch_level) ),
    perm_thres_             ( static_cast<float>(params.permanence_threshold) ),
    inv_perm_range_         ( 1.0f / (1.0f - perm_thres_)                   ),
    primary_learning_rate   ( params.default_primary_learning_rate          ),
    secondary_learning_rate ( params.default_secondary_learning_rate        ),
    mismatch_smoothing      ( params.default_mismatch_smoothing             ),
    accumulated_theta_thres ( params.default_accumulated_theta_thres        ),
    min_mismatch_deviation  ( params.default_min_mismatch_deviation         ),
    min_mismatch_percentage ( params.default_min_mismatch_percentage        ),
    mismatch_act_thres      ( params.default_mismatch_act_thres             ),
    last_max_inp            ( 0.0f                                          ),
    rgen                    ( std::to_underlying(stream_seed(params.rnd_seed, stream_index, parent_seed)) )
{
    // initializing random synapses moved to init_input function

    // Internally we represent the binary tree structure of the dendritic branch as linear arrays of appropriate size
    segment_activity.resize(max_segment_idx + 1, 0.0f);
    segment_weights.resize(max_segment_idx + 1,  0.0f);

}

template <class Channel>
std::vector<Channel> declared_channels(
    const std::vector<std::string> &input_channels,
    const std::unordered_map<std::string, Channel> &input_map,
    std::string_view who, std::string_view feeder)
{
    auto channel_error = [&](const std::string &what) {
        return std::runtime_error(std::format("{}: input_channels {}", who, what));
    };
    const std::vector<std::string> declared =
        input_channels.empty() && input_map.size() == 1
            ? std::vector<std::string>{ input_map.begin()->first }
            : input_channels;
    std::vector<Channel> channels;
    channels.reserve(declared.size());
    for (const auto &name : declared) {
        const auto it = input_map.find(name);
        if (it == input_map.cend())
            throw channel_error(std::format("declares '{}', but no {} feeds it", name, feeder));
        if (std::ranges::count(declared, name) > 1)
            throw channel_error(std::format("declares '{}' more than once", name));
        channels.push_back(it->second);
    }
    for (const auto &name : input_map | std::views::keys)
        if (std::ranges::find(declared, name) == declared.cend())
            throw channel_error(std::format("does not declare '{}', although a {} feeds it", name, feeder));
    return channels;
}

// The group walks its own inputs, the dendrite the channels its mappings feed.
template std::vector<sim::output_channel_t> declared_channels(
    const std::vector<std::string> &, const std::unordered_map<std::string, sim::output_channel_t> &,
    std::string_view, std::string_view);
template std::vector<channel_source_t> declared_channels(
    const std::vector<std::string> &, const channel_source_map_t &, std::string_view, std::string_view);

void dendrite_t::init_input(const channel_source_map_t &input_map, std::string_view who)
{
    // The inputs are laid out in the declared channel order (params.input_channels).
    // The synapses cover the channels one after another, so this order decides
    // which input each permanence meets and, through the per-synapse clamp, the
    // response (SPEC.md M.7.5).
    input_size = 0;
    input_buf.clear();
    channels.clear();
    for (const auto &[channel, source] : declared_channels(params.input_channels, input_map, who, "channel mapping")) {
        // The record (SPEC.md M.7.5) names the group-level channel this one
        // resolves to, not the dendrite's own key: a reorder made purely through
        // channel_mappings keeps the keys and changes only the sources.
        if (source.empty())
            throw std::runtime_error(std::format(
                "{}: a channel resolves to a group input with no name "
                "(an empty channel name cannot be checkpointed)", who));
        input_buf.emplace_back(channel.first, nullptr);
        channels.push_back({ source, channel.second });
        input_size += channel.second;
    }
    // A dendrite needs at least one channel: get_response() and
    // adapt_synapses() read input_buf's first entry unconditionally.
    if (input_buf.empty())
        throw std::runtime_error(std::format(
            "{}: has no input channels: none are declared and none are mapped to it", who));

    // initializing random synapses (was in constructor)

    synapses.reserve(input_size * 2);
    synapses.resize(input_size);

    // Cold permanences, Poisson around the permanence threshold (ngm/cold_init.h),
    // drawn from this dendrite's stream in input order.
    ngm::cold_permanence_t cold_perm(perm_thres_);
    for (std::size_t i = 0; i < input_size; ++i) {
        synapses.permanence[i]    = cold_perm(rgen);
        synapses.mismatch[i]      = 0.0f;
        synapses.adapt_history[i] = 0.0f;
        synapses.segment_idx[i]   = 1;
        synapses.input_inc[i]     = 1;
    }

}

/*
 * Interface function to receive for each input id the respective input function. Since we might get offered input
 * functions that this dendritic branch might not need, we check if we are interested in the ID and only then store
 * the buffer in its declared slot of input_buf (every slot fed by that ID, should one source feed several)
 */
void dendrite_t::set_input_buffer(sim::output_id_t id, sim::io_buffer& buf)
{
    for (auto &[cid, cbuf] : input_buf)
        if (cid == id)
            cbuf = &buf;
}


/*
 * main function that models the response of a dendritic branch to the current input of its input space(s)
 */
float dendrite_t::get_response(const input_stats_map_t& stats)
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
 *   + the gate now uses 1 - Hoyer sparsity instead (see noise_attenuation
 *     in hd_ngm2_tools.h); it does not depend on the input size
 */

    // clear current segment activities
    std::ranges::fill(segment_activity, 0.0f);

    // gather sum and max of all partial inputs
    // sum will be used to normalize the response at the end, but we calculate it
    // here to allow for an early exit if the input is entirely zero or malformed
    // max will only be used later during adaptation...
    // The noise gate's min over the partial inputs is gathered here as well.
    float inp_sum = 0.0f;
    last_max_inp  = 0.0f;
    noise_gate_min_t gate_min;
    for (const auto &[pid, buf] : input_buf) {
        const input_stats_t &pi_stats = stats.at(pid);
        inp_sum += pi_stats.sum;
        last_max_inp = std::max(last_max_inp, pi_stats.max_val);
        gate_min.add(pi_stats);
    }

    // early exit if the input is zero or malformed. segment_activity and
    // attenuation are not computed on this step, so mark their observed snapshots
    // NaN rather than leaving the previous step's stale values in place. The
    // observation pass still emits a row for every active step (keeping the
    // per-dendrite series gap-free), and the NaN flags the step as invalid to
    // offline analysis. synapse_count / representation_count are structural and
    // remain valid, so they are left as-is.
    if (!std::isnormal(inp_sum)) {
        //std::printf("ERROR: input sum in dendrite_t::get_response() is not normal.\n");
        observed_segment_activity_ = std::numeric_limits<float>::quiet_NaN();
        observed_attenuation_      = std::numeric_limits<float>::quiet_NaN();
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

    // getting the first partial input (in declared order)
    // and getting cur_input array and statistics for the first partial input
    auto pi = input_buf.begin();
    auto           cur_input = pi->second->cur_read_buffer();
    input_stats_t  cur_stats = stats.at(pi->first);

    // set up start and end of current input
    auto cur_inp_it  = cur_input.begin();
    auto cur_inp_end = cur_input.end();

    // precompute reciprocal for sum (used in penalty calculation)
    float inv_sum = 1.0f / cur_stats.sum;

    // setting up a uniform random distribution that will be used to stochastically determine if an input is "low"
    std::uniform_real_distribution<float> dis1 { 0.0f, cur_stats.max_val / 2.0f };

    for (std::size_t i = 0; i < syn_cnt; ++i) {

        // if the current partial input has ended, advance to the next one
        if (cur_inp_it == cur_inp_end) {
            ++pi;
            cur_input    = pi->second->cur_read_buffer();
            cur_stats    = stats.at(pi->first);
            cur_inp_it   = cur_input.begin();
            cur_inp_end  = cur_input.end();
            inv_sum      = 1.0f / cur_stats.sum;
            dis1         = std::uniform_real_distribution<float>{0.0f, cur_stats.max_val / 2.0f};
        }

        const float inp         = *cur_inp_it;
        const float inp_contrib = inp * inv_sum;

        if (synapses.permanence[i] > perm_thres_) {
            segment_activity[synapses.segment_idx[i]] += inp;
            if (dis1(rgen) > inp) {
                const float perm_strength =
                    (synapses.permanence[i] - perm_thres_) * inv_perm_range_;
                segment_activity[synapses.segment_idx[i]] -= perm_strength * (1.0f - inp_contrib);
                if (segment_activity[synapses.segment_idx[i]] < 0.0f)
                    segment_activity[synapses.segment_idx[i]] = 0.0f;
            }
        }

        cur_inp_it += synapses.input_inc[i];
    }


    // 2) push activities to the leafs
    const std::size_t leaf_begin = (max_segment_idx + 1) / 2;
    for (std::size_t si = 1; si < leaf_begin; ++si) {
        segment_activity[si * 2 + 0] += segment_activity[si];
        segment_activity[si * 2 + 1] += segment_activity[si];
    }

    // determine the maximum activity among the leafs of the dendritic branch
    // and attenuate the activity if the noise gate (see noise_attenuation in hd_ngm2_tools.h) indicates that the
    // input is basically noise. Over several partial inputs the gate takes the least noisy and the darkest of those
    // that carry shape evidence (noise_gate_min_t).
    // Please note that the segment_activity is used below in the adapt_synapses function. Hence the attenuation needs
    // to be applied to every element and not just the max_activity
    const float attenuation = gate_min.attenuation(response_noise_window);
    const float inv_inp_sum = 1.0f / inp_sum;
    float max_activity = 0.0f;
    for (std::size_t si = leaf_begin; si <= max_segment_idx; ++si) {
        segment_activity[si] = std::clamp( segment_activity[si] * attenuation * inv_inp_sum , 0.0f, 1.0f );
        max_activity = std::max(max_activity, segment_activity[si]);
    }

    // Probe dendrite state
    PROBE("segment_activity", max_activity);
    PROBE("synapse_count", static_cast<double>(synapses.size()));
    PROBE("representation_count", static_cast<double>(get_representation_count()));

    PROBE("attenuation",attenuation);
//    PROBE("attenuation_2",attenuation_2);

    // Snapshot the transient observed values; the post-process observation pass
    // reads them back via perform_observations(). max_activity and attenuation
    // are locals here, so they must be stored to survive to end-of-step.
    observed_segment_activity_ = max_activity;
    observed_attenuation_      = attenuation;

    return max_activity;
}

void dendrite_t::perform_observations(observe::observe_accessor& acc)
{
    if (!acc.group_active("dendrite")) return;
    acc.observe(*this, "dendrite", "segment_activity",     observed_segment_activity_);
    acc.observe(*this, "dendrite", "synapse_count",        static_cast<std::uint32_t>(synapses.size()));
    acc.observe(*this, "dendrite", "attenuation",          observed_attenuation_);
    acc.observe(*this, "dendrite", "representation_count", static_cast<std::uint32_t>(get_representation_count()));
}

/*
 * the main function that models the adaptation of a dendritic branch
 */
void dendrite_t::adapt_synapses(const float max_activity, const float weight, const input_stats_map_t& stats)
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
    const float inv_max_activity = 1.0f / max_activity;
    for (std::size_t si = leaf_begin; si <= max_segment_idx; ++si) {
        if ((!max_response_seen) && (segment_activity[si] + eps >= max_activity)) {
            max_response_seen = true;
            segment_weights[si] = weight * primary_learning_rate;
        } else {
            segment_weights[si] = segment_activity[si] * weight * secondary_learning_rate * inv_max_activity;
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

    // getting the first partial input (in declared order)
    // and getting cur_input array and statistics for the first partial input
    auto pi = input_buf.begin();
    auto           cur_input = pi->second->cur_read_buffer();
    input_stats_t  cur_stats = stats.at(pi->first);

    // set up start and end of current input
    auto cur_inp_it    = cur_input.begin();
    auto cur_inp_end   = cur_input.end();

    // calculate an attenuation factor from the noise gate of this partial input alone
    // (see noise_attenuation in hd_ngm2_tools.h and the attenuation in the get_response method)
    float attenuation = noise_attenuation(cur_stats.noise_hoyer, cur_stats.avg, learning_noise_window);

    // precompute reciprocals for theta calculation (per partial input)
    float high_thres     = (cur_stats.avg / 2.0f) + std::numeric_limits<float>::epsilon();
    float inv_high_range = 1.0f / (1.0f - high_thres);
    float inv_high_thres = 1.0f / high_thres;

    // precompute reciprocal for mismatch calculation
    const float inv_last_max_inp = 1.0f / last_max_inp;

    // linear sweep through all synapses
    for (std::size_t i = 0; i < syn_cnt; ++i) {

        // if the current partial input has ended, we continue with the next one and update our variables accordingly
        if (cur_inp_it == cur_inp_end) {
            ++pi;
            cur_input = pi->second->cur_read_buffer();
            cur_stats = stats.at(pi->first);
            cur_inp_it    = cur_input.begin();
            cur_inp_end   = cur_input.end();
            attenuation    = noise_attenuation(cur_stats.noise_hoyer, cur_stats.avg, learning_noise_window);
            high_thres     = (cur_stats.avg / 2.0f) + std::numeric_limits<float>::epsilon();
            inv_high_range = 1.0f / (1.0f - high_thres);
            inv_high_thres = 1.0f / high_thres;
        }

        // 3 we want to learn strongly when the particular input is either near 1 or near 0 and
        // if the partial input is not noise
        const float theta = std::clamp( segment_weights[ synapses.segment_idx[i] ] * ( *cur_inp_it > high_thres ?
                                (*cur_inp_it - high_thres) * inv_high_range :
                                (high_thres - *cur_inp_it) * inv_high_thres
                            ) * attenuation, 0.0f, 1.0f);

        synapses.permanence[i] = std::clamp(synapses.permanence[i] * (1.0f - theta) + (*cur_inp_it > high_thres ? theta : 0.0f), 0.0f, 1.0f);

        // 4.1 we collect some statistical information on the strength of our permanence adaptation. We need this information
        // below in the adapt branches function to decide whether or not to move the synapse to a higher dendritic segment
        synapses.adapt_history[i] += theta;

        // 4.2 the adaptation history is not enough to decide if a synapse should "branch". Hence, we also calculate a
        // mismatch heuristic that describes how well the permanence value of the synapse agrees with the activity of
        // the input. However, we only collect this information if the corresponding segment activity is reasonably high.
        // Furthermore, the mismatch value is implemented as IIR-Filter that emphasizes more recent mismatches.
        const float act_ratio = segment_activity[ synapses.segment_idx[i] ] * inv_max_activity;
        if (act_ratio >= mismatch_act_thres) {
            const float inp_ratio = *cur_inp_it * inv_last_max_inp;
            float mismatch = synapses.permanence[i] > perm_thres_ ? 1.0f - inp_ratio : inp_ratio;
            mismatch *= act_ratio;
            synapses.mismatch[i] = synapses.mismatch[i] * (1.0f - mismatch_smoothing) + mismatch * mismatch_smoothing;
        }

        // advance the iterator only at the trailing synapse of each pack (low bit set).
        // The high bits of input_inc at the pack start hold the pack_size for get_response();
        // masking them out here keeps adapt_synapses() oblivious to that encoding.
        cur_inp_it += synapses.input_inc[i] & 1u;
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
    const float mm_thres = mm_avg + mm_std * min_mismatch_deviation + 1.0f / static_cast<float>(input_size);

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
    if (static_cast<float>(mm_cnt) < static_cast<float>(input_size) * min_mismatch_percentage)
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

        // disable input advancement of "lower" synapse (lower == lower index)
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

    // unmark inner segments. The indices are std::size_t, not seg_id_t: a u16
    // counter cannot reach the bound level * 2 of a tree at the top of the u16
    // range, and wrapped forever there before depth was capped (#237).
    std::size_t level = (std::size_t{max_segment_idx} + 1) / 2;
    while (level > 1) {
        for (std::size_t si = level; si < level * 2; ++si)
            if (leaf_detection[si] == 1) {
                std::size_t lower = si;
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
    result.reserve(input_size);

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
    return input_size;
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
    return input_size;
}

void dendrite_t::save_state(logging::state_writer& writer) const
{
    // The learning parameters are settings, not state: they are in the group's
    // configuration record (append_config), and the configuration wins on load.

    // Structure (M.7.18): the type and the tree depth, which neither the synapse
    // counts nor the segment ids pin down. Format version 5 on; check_shape.
    writer.write("apical", std::uint64_t{ params.type == dendrite_type_t::apical });
    writer.write("max_branch_level", static_cast<std::uint64_t>(params.max_branch_level));

    // Save synapse data
    writer.write("permanence", synapses.permanence);
    writer.write("mismatch", synapses.mismatch);
    writer.write("adapt_history", synapses.adapt_history);
    writer.write("segment_idx", synapses.segment_idx);
    writer.write("input_inc", synapses.input_inc);

    logging::write_channel_record(writer, "input_channels", channels);

    // PRNG state — required for a bit-identical stream continuation on resume
    logging::write_rng(writer, "rng", rgen);
}

void dendrite_t::append_config(logging::config_record& rec, const std::string& prefix) const
{
    rec.push_back({ prefix + "permanence_threshold",    params.permanence_threshold });
    rec.push_back({ prefix + "max_branch_level",        static_cast<std::uint64_t>(params.max_branch_level) });
    rec.push_back({ prefix + "primary_learning_rate",   primary_learning_rate });
    rec.push_back({ prefix + "secondary_learning_rate", secondary_learning_rate });
    rec.push_back({ prefix + "mismatch_act_thres",      mismatch_act_thres });
    rec.push_back({ prefix + "mismatch_smoothing",      mismatch_smoothing });
    rec.push_back({ prefix + "accumulated_theta_thres", accumulated_theta_thres });
    rec.push_back({ prefix + "min_mismatch_deviation",  min_mismatch_deviation });
    rec.push_back({ prefix + "min_mismatch_percentage", min_mismatch_percentage });
    rec.push_back({ prefix + "rnd_seed",                std::uint64_t{ std::to_underlying(params.rnd_seed) }, /*random_state=*/true });
}

dendrite_t::synapses_t dendrite_t::check_shape(logging::state_reader& reader, std::string_view who) const
{
    // The synapse arrays take the checkpoint's size, since dendrites grow via
    // branching during training; all five must agree on it. They are read and
    // checked before any is assigned.
    auto corrupt = [&](std::string_view what) {
        return std::runtime_error(std::format("{}: corrupt checkpoint — {}", who, what));
    };

    // The type and the tree depth first (M.7.18), recorded from format version 5
    // on. Without them a checkpoint of a proximal dendrite loaded into an apical
    // one of the same width, and one of a shallower tree into a deeper one, since
    // its segment ids all fit (#236). Older checkpoints: the group checks the depth
    // against their configuration record (neuron_group_t::load_state).
    if (reader.format_version() >= 5)
        logging::check_shape(reader, who, {
            { "apical",           std::uint64_t{ params.type == dendrite_type_t::apical } },
            { "max_branch_level", params.max_branch_level },
        });

    if (!reader.has_field("permanence"))
        throw corrupt("permanence missing");
    auto perm    = reader.read_float_vec("permanence");
    auto mism    = reader.read_float_vec("mismatch");
    auto adapt   = reader.read_float_vec("adapt_history");
    auto seg_idx = reader.read_uint16_vec("segment_idx");
    auto inp_inc = reader.read_uint8_vec("input_inc");
    if (mism.size() != perm.size() || adapt.size() != perm.size() ||
        seg_idx.size() != perm.size() || inp_inc.size() != perm.size())
        throw corrupt(std::format(
            "mismatch/adapt_history/segment_idx/input_inc hold {}/{}/{}/{} values, permanence {}",
            mism.size(), adapt.size(), seg_idx.size(), inp_inc.size(), perm.size()));

    // The synapses must also fit this dendrite (M.7.18). Each input's run of
    // synapses advances the input by one, at its last synapse, so the walks in
    // get_response() and adapt_synapses() cover exactly input_size inputs; a
    // checkpoint from another input width would read past the input page or leave
    // part of it unread. And every segment id must exist in this dendrite's tree,
    // or the response writes past segment_activity.
    auto misfit = [&](std::string_view what) {
        return std::runtime_error(std::format("{}: checkpoint does not fit this dendrite — {}", who, what));
    };
    std::size_t width = 0;
    for (const auto inc : inp_inc) {
        if (inc > 1)
            throw corrupt(std::format("input_inc holds {}, expected 0 or 1", inc));
        width += inc;
    }
    if (!inp_inc.empty() && inp_inc.back() != 1)
        throw corrupt("the last synapse does not advance the input");
    if (width != input_size)
        throw misfit(std::format("input width {} (checkpoint) vs {} (configured)", width, input_size));
    for (const auto s : seg_idx) {
        if (s == 0)
            throw corrupt("segment id 0");
        if (s > max_segment_idx)
            throw misfit(std::format("segment id {} (checkpoint) vs at most {} (configured)", s, max_segment_idx));
    }

    return { std::move(perm), std::move(mism), std::move(adapt), std::move(seg_idx), std::move(inp_inc) };
}

void dendrite_t::load_state(logging::state_reader& reader, std::string_view who)
{
    // The learning parameters keep their configured values (M.7.8): the group
    // compares and reports them, nothing here reads them back.
    synapses = check_shape(reader, who);
    logging::read_rng(reader, "rng", rgen, who);
}
}
