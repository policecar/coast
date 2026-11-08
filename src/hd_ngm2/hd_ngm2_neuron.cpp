//
// Created by jk on 26.08.25.
//
#include <algorithm>
#include <cmath>
#include <format>
#include <ranges>
#include <stdexcept>
#include "hd_ngm2/hd_ngm2_neuron.h"

#include "hd_ngm2/hd_ngm2_tools.h"
#include "logging/state_checks.h"
#include "logging/state_io.h"
#include "logging/rng_state_io.h"
#include "observe/observe_accessor.h"

namespace ngm2 {

/*
 * main initialization of a neuron
 */
neuron_t::neuron_t(const cfg_store &cs, const std::string &type, int stream_index, seed_t parent_seed) :
    params( [&cs,&type]() -> neuron_params_t
            {
                  if (const auto it = cs.neurons.find(type); it != cs.neurons.cend())
                      return it->second;
                  return neuron_params_t {};
            }() ),
    neuron_activity(0.0f),
    max_activity(0.0f),
    dendrite_type_activity(),
    input_count(0),
    branch_interval(params.default_branch_interval),
    energy(1.0f),
    rgen(std::to_underlying(stream_seed(params.random_seed, stream_index, parent_seed))),
    id(-1)
{
    // A zero interval made the branch gate (++input_count % branch_interval) a
    // modulo by zero. ngm3 and ngm25 reject it too.
    if (branch_interval == 0)
        throw std::runtime_error("ngm2: default_branch_interval must be >= 1");
    // std::uniform_real_distribution<float>(a, b) requires a <= b and a finite
    // b - a (UB otherwise), checked on the floats the distribution gets.
    if (const float lo = static_cast<float>(params.default_neuron_noise_range.first),
                    hi = static_cast<float>(params.default_neuron_noise_range.second);
        !(lo <= hi && std::isfinite(hi - lo)))
        throw std::runtime_error("ngm2: default_neuron_noise_range must have first <= second and a width second - first that is finite in float; got ["
                                 + std::to_string(lo) + ", " + std::to_string(hi) + "]");

    // create dendrites
    std::size_t dendrite_cnt = 0;
    for (const auto &d : params.dendrites)
        dendrite_cnt += d.dendrite_count;
    dendrites.reserve(dendrite_cnt);
    // Each dendrite's stream hangs off this neuron's stream seed.
    const auto own_seed = stream_seed(params.random_seed, stream_index, parent_seed);
    int dendrite_index = 0;
    for (const auto &d : params.dendrites)
        for (std::size_t n = 0; n < d.dendrite_count; ++n)
            dendrites.emplace_back(cs,d.dendrite_cfg,dendrite_index++,own_seed);
}

void neuron_t::init_input(const channel_source_map_t &input_map, std::string_view who)
{
    std::size_t cur_dendrite_idx = 0;
    for (const auto &d : params.dendrites) {
        // create mapping. it->second (channel_source_t) is forwarded as-is: its
        // `source` is the group-level channel name set at neuron_group_t::init_io
        // and must survive this hop's own renaming unchanged (SPEC.md M.7.5).
        channel_source_map_t dendrite_input_map;
        for (const auto &cmap : params.channel_mappings
                              | std::views::filter( [&d] (const auto &elem) { return elem.to_name == d.name; }))
        {
            auto it = input_map.find(cmap.from_channel);
            if (it == input_map.cend())
                throw std::runtime_error(std::format(
                    "{}: dendrite set '{}': no matching source (from_channel) for channel '{}'",
                    who, d.name, cmap.from_channel));
            // Two mappings onto the same to_channel are refused: the second
            // would replace the first and drop its channel.
            if (!dendrite_input_map.try_emplace(cmap.to_channel, it->second).second)
                throw std::runtime_error(std::format(
                    "{}: dendrite set '{}': two channel mappings both target '{}'; "
                    "the second would silently replace the first",
                    who, d.name, cmap.to_channel));
        }
        for (std::size_t k = 0; k < d.dendrite_count; ++k) {
            auto &cur_dendrite = dendrites[cur_dendrite_idx];
            cur_dendrite.init_input(dendrite_input_map,
                                     std::format("{}: dendrite {} ({})", who, cur_dendrite_idx, d.dendrite_cfg));
            ++cur_dendrite_idx;
        }
    }
}

/*
 * interface function that allows the neuron group to hand over functions that provide access to the respective
 * input buffers. We do not store these functions here in the neuron, but hand it over to the dendrites...
 */
void neuron_t::set_input_buffer(sim::output_id_t id, sim::io_buffer& buf)
{
    for (auto &dendrite : dendrites)
        dendrite.set_input_buffer(id, buf);
}


/*
 * main function that models the response of a neuron to a current input
 *
 */
float neuron_t::get_response(const input_stats_map_t& stats)
{
    // generate indices from the possible types of dendrites. Currently, only apical and proximal dendrites are
    // modelled.
    constexpr int ai = static_cast<int>(dendrite_type_t::apical);
    constexpr int pi = static_cast<int>(dendrite_type_t::proximal);

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
        dendrite_type_activity[type_idx] = std::max(dendrite_type_activity[type_idx], dendrite.get_response(stats));
    }

    // we check for our sentinel and set the apical activity to 1 if no apical dendrite is present
    if (dendrite_type_activity[ai] < 0.0f)
        dendrite_type_activity[ai] = 1.0f;

    // we ensure that all activites are in a suitable range
    dendrite_type_activity[ai] = std::clamp(dendrite_type_activity[ai], 0.0f, 1.0f);
    dendrite_type_activity[pi] = std::clamp(dendrite_type_activity[pi], 0.0f, 1.0f);

    // we modulate the proximal activity by the apical activity and add configurable noise
    // (default 1%-5%; set neuron cfg "default_neuron_noise_range":[0,0] to freeze for eval)
    std::uniform_real_distribution<float> dis(
        static_cast<float>(params.default_neuron_noise_range.first),
        static_cast<float>(params.default_neuron_noise_range.second));
    neuron_activity = std::clamp(
        dendrite_type_activity[ai] * dendrite_type_activity[pi] + dis(rgen),
        0.0f,
        1.0f//sigmoid(energy,{0.66,0.33})
    );

    // Probe neuron state
    PROBE("activity", neuron_activity);
    PROBE("dendrite_count", static_cast<double>(dendrites.size()));
    PROBE("representation_count", static_cast<double>(get_representation_count()));


    /*
    const float drain = 0.1f;
    energy = energy * (1.0f - drain) + (1.0f - neuron_activity) * drain;
    */
    return neuron_activity;
}

void neuron_t::perform_observations(observe::observe_accessor& acc)
{
    if (acc.group_active("neuron")) {
        // neuron_activity is a member set once per step in get_response();
        // stable to end-of-step (adapt() reads it but does not modify it).
        acc.observe(*this, "neuron", "activity",       neuron_activity);
        acc.observe(*this, "neuron", "dendrite_count", static_cast<std::uint32_t>(dendrites.size()));
    }
    entity_element::perform_observations(acc);   // recurse into dendrites
}

/*
 * modelling the adaptation of a neuron
 */
void neuron_t::update_max_activity()
{
    // Read only by adapt()'s weight filter. Updated from the group's learning
    // phase rather than in get_response(), so a step that does not learn (frozen,
    // or a group output summing to 0) leaves it untouched.
    max_activity *= 0.99999f;
    max_activity  = std::max(max_activity,neuron_activity);
}

void neuron_t::adapt(float weight, const input_stats_map_t& stats, const learning_window_t& window)
{
    // we only want to learn if our neuron activity was somewhere in the middle. If the neurons response was very low
    // or very high, we reduce the weight towards 0
    // remark: this poses a problem in the beginning, where no neuron has learned anything and activations
    // might be low overall. Since the filtering of low activity is to prevent unlearning, this filtering should
    // only occur if the neuron has already learned something.
    const float synapse_weight = weight * std::min(
        std::max(1.0f - sigmoid(max_activity),sigmoid(neuron_activity,window.first)),
        1.0f - sigmoid(neuron_activity,window.second)
    );

    // we adapt all dendrites and provide each dendrite with the information about the maximum activity among their
    // dendrite type. With that information the dendrite can determine, if it was "the winning" dendrite among all
    // the dendrites.
    for (auto &dendrite : dendrites) {
        dendrite.adapt_synapses( dendrite_type_activity[static_cast<int>(dendrite.get_params().type)], synapse_weight, stats );
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
// std::size_t, not seg_id_t: a sum of per-dendrite counts is not bounded by
// the u16 segment-id range (#237).
std::size_t neuron_t::get_representation_count() const
{
    std::size_t result = 0;
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

void neuron_t::save_state(logging::state_writer& writer) const
{
    writer.write("dendrite_count", static_cast<std::uint64_t>(dendrites.size()));

    // Carried per-step state: branch-interval counter and the decaying running
    // activity max (read in adapt()); both must persist for an exact resume.
    writer.write("input_count", static_cast<std::uint64_t>(input_count));
    writer.write("max_activity", max_activity);

    logging::write_rng(writer, "rng", rgen);

    writer.begin_group("dendrites");
    for (std::size_t di = 0; di < dendrites.size(); ++di) {
        writer.begin_group(std::to_string(di));
        dendrites[di].save_state(writer);
        writer.end_group();
    }
    writer.end_group();
}

void neuron_t::append_config(logging::config_record& rec, const std::string& prefix) const
{
    rec.push_back({ prefix + "branch_interval", static_cast<std::uint64_t>(branch_interval) });
    rec.push_back({ prefix + "neuron_noise_range[0]", params.default_neuron_noise_range.first });
    rec.push_back({ prefix + "neuron_noise_range[1]", params.default_neuron_noise_range.second });
    rec.push_back({ prefix + "random_seed", std::uint64_t{ std::to_underlying(params.random_seed) }, /*random_state=*/true });
}

void neuron_t::load_state(logging::state_reader& reader, std::string_view who)
{
    logging::check_shape(reader, who, { { "dendrite_count", dendrites.size() } });

    auto corrupt = [&](std::string_view what) {
        return std::runtime_error(std::format("{}: corrupt checkpoint — {}", who, what));
    };
    if (!reader.has_field("dendrites"))
        throw corrupt("dendrites missing");

    input_count  = static_cast<std::size_t>(reader.read_uint64("input_count", input_count));
    max_activity = reader.read_float("max_activity", max_activity);

    logging::read_rng(reader, "rng", rgen, who);

    reader.enter_group("dendrites");
    for (std::size_t di = 0; di < dendrites.size(); ++di) {
        if (!reader.enter_group(std::to_string(di)))
            throw corrupt(std::format("dendrite {} missing", di));
        dendrites[di].load_state(reader, std::format("{}: dendrite {}", who, di));
        reader.leave_group();
    }
    reader.leave_group();
}

}
