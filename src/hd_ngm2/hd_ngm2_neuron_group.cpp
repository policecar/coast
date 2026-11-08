//
// Created by jk on 26.08.25.
//
#include "hd_ngm2/hd_ngm2_neuron_group.h"
#include "logging/config_record.h"
#include "logging/state_checks.h"
#include "logging/state_io.h"
#include "logging/rng_state_io.h"
#include "core/require.h"
#include "observe/observe_accessor.h"

#include <utility>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <map>
#include <optional>
#include <ranges>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/partitioner.h>

namespace ngm2 {

namespace {

// Walks every neuron and dendrite read-only, before anything is assigned
// (M.7.18), for every format version: each neuron's dendrite_count, then each
// dendrite's channel record (format 4 on) and synapse arrays. The record comes
// first so it names the channels of a width mismatch. Neurons or dendrites
// missing from the checkpoint are left to the load proper, as corrupt.
void check_structure(logging::state_reader &reader, const std::vector<neuron_t> &neurons,
                     std::string_view who)
{
    if (!reader.enter_group("neurons")) return;
    for (std::size_t ni = 0; ni < neurons.size(); ++ni) {
        if (!reader.enter_group(std::to_string(ni))) continue;
        const auto &dends = neurons[ni].get_dendrites();
        const auto neuron_who = std::format("{}: neuron {}", who, ni);
        logging::check_shape(reader, neuron_who, { { "dendrite_count", dends.size() } });
        if (reader.enter_group("dendrites")) {
            for (std::size_t di = 0; di < dends.size(); ++di) {
                if (!reader.enter_group(std::to_string(di))) continue;
                const auto dendrite_who = std::format("{}: dendrite {}", neuron_who, di);
                if (reader.format_version() >= 4)
                    logging::check_channel_record(reader, "input_channels", dends[di].get_channel_layout(),
                                                  dendrite_who);
                dends[di].check_shape(reader, dendrite_who);
                reader.leave_group();
            }
            reader.leave_group();
        }
        reader.leave_group();
    }
    reader.leave_group();
}

// Checkpoints before format version 5 record no dendrite type or tree depth as
// structure (#236). Their configuration record, where there is one, holds each
// dendrite cfg's max_branch_level as a setting, under the key current_config()
// writes; every instance of a cfg has its depth (there is no setter), so the
// cfg-level key is the whole record. A differing value there refuses the load
// as a recorded depth would (M.7.18). The lookup assumes the checkpoint gave the
// same cfg names to the same dendrites, which is all the record can tell.
// Returns the cfgs whose depth the checkpoint does not record.
std::vector<std::string> check_recorded_depths(logging::state_reader &reader,
                                               const std::vector<neuron_t> &neurons, std::string_view who)
{
    std::map<std::string, std::uint64_t> built;   // by dendrite cfg
    for (const auto &n : neurons) {
        std::size_t di = 0;
        for (const auto &ds : n.get_params().dendrites)
            for (std::size_t m = 0; m < ds.dendrite_count; ++m, ++di)
                built.emplace(ds.dendrite_cfg, n.get_dendrite(di).get_params().max_branch_level);
    }

    std::map<std::string, std::uint64_t> recorded;
    if (reader.enter_group("config")) {
        try {
            for (const auto &cfg : built | std::views::keys)
                if (const auto key = std::format("dendrite[{}].max_branch_level", cfg); reader.has_field(key))
                    recorded.emplace(cfg, reader.read_uint64(key));
        } catch (...) {
            reader.leave_group();
            throw;
        }
        reader.leave_group();
    }

    std::vector<std::string> unrecorded;
    std::string diffs;
    for (const auto &[cfg, depth] : built) {
        const auto it = recorded.find(cfg);
        if (it == recorded.end()) unrecorded.push_back(cfg);
        else if (it->second != depth)
            diffs += std::format("{}dendrite[{}].max_branch_level {} (checkpoint's configuration record) vs {} "
                                 "(configured)", diffs.empty() ? "" : ", ", cfg, it->second, depth);
    }
    if (!diffs.empty())
        throw std::runtime_error(std::format("{}: checkpoint does not fit this group — {}", who, diffs));
    return unrecorded;
}

// The seed the checkpoint's random state came from: the group's effective_seed
// field (format version 5 on), else its configuration record's entry (older
// checkpoints that have one); nullopt for a checkpoint with neither. A record
// value wider than 32 bits (a checkpoint from before seeds were 32-bit) seeded
// the rng with its low 32 bits, which are what is kept.
std::optional<seed_t> checkpoint_seed(logging::state_reader &reader)
{
    if (reader.has_field("effective_seed"))
        return seed_t{ reader.read_uint32("effective_seed") };
    if (!reader.enter_group("config")) return std::nullopt;
    std::optional<seed_t> seed;
    try {
        if (reader.has_field("effective_seed"))
            seed = seed_t{ static_cast<std::uint32_t>(reader.read_uint64("effective_seed")) };
    } catch (...) {
        reader.leave_group();
        throw;
    }
    reader.leave_group();
    return seed;
}

} // namespace

/*
 * Main initialization of the neuron group
 */
neuron_group_t::neuron_group_t(const cfg_store &cs, const std::string &type, sim::output_id_t output_id, seed_offset_t seed_offset) :
    params(cs.group_or_default(type)),
    id                        ( output_id                                ),
    local_inhibition_strength ( params.default_local_inhibition_strength ),
    common_learning_rate      ( params.default_common_learning_rate      ),
    weight_filter             ( params.default_weight_filter             ),
    stochastic_win_thres      ( params.default_stochastic_win_thres      ),
    rgen                      ( std::to_underlying(params.random_seed + seed_offset) )
{
    // Record the effective seed so the run manifest can surface it without
    // re-deriving coast's seed-composition rule (config seed + seed_offset).
    effective_seed_ = params.random_seed + seed_offset;

    // temporary set to gather all input IDs from the dendrites of all
    // neurons in this group

    //std::set<partial_id_t> tmp;

    // create neurons and collect the input IDs in the temporary set
    std::size_t neuron_cnt = 0;
    for (const auto &n : params.neurons)
        neuron_cnt += n.neuron_count;
    // The total over all neuron sets counts: one set of 1 next to others is fine.
    if (neuron_cnt < 2)
        throw std::runtime_error("ngm2: neuron_count must be >= 2 in total over the group's neuron sets "
                                 "(a neuron group needs at least two neurons to compete); got "
                                 + std::to_string(neuron_cnt));
    // The winning threshold is drawn from uniform(mx_act * thres, mx_act), which
    // requires thres <= 1 (UB otherwise).
    if (!(params.default_stochastic_win_thres >= 0.0 && params.default_stochastic_win_thres <= 1.0))
        throw std::runtime_error("ngm2: default_stochastic_win_thres must be in [0, 1]; got "
                                 + std::to_string(params.default_stochastic_win_thres));
    neurons.reserve(neuron_cnt);
    int seed_delta = 0;
    for (const auto &n : params.neurons)
        for (std::size_t k = 0; k < n.neuron_count; ++k) {
            // parent seed = the group's effective seed, so random_seed and the
            // seed offset both reach every neuron's stream (as in ngm3/ngm25).
            auto &new_neuron = neurons.emplace_back(cs,n.neuron_cfg,seed_delta++,effective_seed_);
            new_neuron.id    = neurons.size()-1;
            /*
            for (const auto &d : new_neuron.get_dendrites()) {
                tmp.insert_range(d.get_params().input_ids);
            }
            */
        }

    // copy the set of input IDs over to the inp_ids vector
    /*
    inp_ids.resize(tmp.size());
    std::ranges::copy(tmp,inp_ids.begin());
    */

    // Construct this group's inhibition. An empty inhibition_cfg synthesizes a
    // max_softmax with beta = local_inhibition_strength.
    inhibitor_ = inhibition::make_inhibitor(cs.inhibitions, params.inhibition_cfg,
                                            local_inhibition_strength, seed_offset);

    // NOTE: init_structure() is called in init_io(), not here.
    // This is because the entity vector may reallocate when more entities are added,
    // which would invalidate the root_entity_ pointers set by init_structure().
}

void neuron_group_t::init_io(const std::unordered_map<std::string,output_channel_t> &input_map){
    // Initialize entity element structure for probe system.
    // This must be called after all entities are created to avoid pointer invalidation
    // from vector reallocation.
    init_structure();

    // input channels in declared order (input_map is unordered)
    const auto who = std::format("ngm2 neuron group {}", id);
    inp_ids.clear();
    input_bufs.clear();
    for (const auto &channel : declared_channels(params.input_channels, input_map, who, "group input")) {
        inp_ids.push_back(channel.first);
        input_bufs.emplace_back(channel.first, nullptr);
    }

    // init neuron inputs
    std::size_t cur_neuron_idx = 0;
    for (const auto &n : params.neurons) {
        // create mapping; each entry carries its group-level name (from_channel)
        // down to the dendrite's channel record (M.7.5).
        channel_source_map_t neuron_input_map;
        for (const auto &cmap : params.channel_mappings
                              | std::views::filter( [&n] (const auto &elem) { return elem.to_name == n.name; }))
        {
            if (cmap.from_channel.empty())
                throw std::runtime_error(std::format(
                    "{}: neuron set '{}': a channel mapping has an empty from_channel "
                    "(an empty channel name cannot be checkpointed)", who, n.name));
            auto it = input_map.find(cmap.from_channel);
            if (it == input_map.cend())
                throw std::runtime_error(std::format(
                    "{}: neuron set '{}': no matching source (from_channel) for channel '{}'",
                    who, n.name, cmap.from_channel));
            // Two mappings onto the same to_channel are refused: the second
            // would replace the first and drop its channel.
            if (!neuron_input_map.try_emplace(cmap.to_channel,
                                              channel_source_t{ it->second, cmap.from_channel }).second)
                throw std::runtime_error(std::format(
                    "{}: neuron set '{}': two channel mappings both target '{}'; "
                    "the second would silently replace the first",
                    who, n.name, cmap.to_channel));
        }
        for (std::size_t k = 0; k < n.neuron_count; ++k) {
            auto &cur_neuron = neurons[cur_neuron_idx];
            cur_neuron.init_input(neuron_input_map, std::format("{}: neuron {}", who, cur_neuron_idx));
            ++cur_neuron_idx;
        }
    }
}

/*
 * implementing the interface function that allows the simulation environment to hand over
 * a function that will provide us with the current output buffer. This function is
 * stored in "output_mem".
 */
void neuron_group_t::set_output_buffer(sim::io_buffer& buf)
{
    output_buf = &buf;
}

/*
 * implementing the interface function that allows the simulation environment to hand over
 * functions that provide access to the respective input buffers. We do not store these
 * functions here in the group, but hand it over to the neurons (which will in turn hand it
 * over to their dendrites...)
 */
void neuron_group_t::set_input_buffer(output_id_t id, sim::io_buffer& buf)
{
    // remember the buffer in every slot fed by this id, and hand it down to
    // the neurons (and in turn their dendrites).
    for (auto &[cid, cbuf] : input_bufs)
        if (cid == id)
            cbuf = &buf;
    for (auto &neuron : neurons)
        neuron.set_input_buffer(id, buf);
}

/*
 * main function that models the neuron group's behavior for one processing step
 */
sim::step_result neuron_group_t::process()
{
    COAST_REQUIRE(output_buf != nullptr, core::error_kind::contract_violation);

    // acquire our current output array that will hold all the activities of the neurons in this group
    std::span<float> out = output_buf->cur_write_buffer();

    COAST_REQUIRE(out.size() == neurons.size(), core::error_kind::contract_violation);

    // Compute the per-buffer input statistics once for this step. Each input
    // buffer's statistics are computed exactly once here and shared (by const
    // reference) with every neuron / dendrite that reads from it — this is the
    // single computation point that previously lived inside io_buffer.
    input_stats_map_t input_stats;
    input_stats.reserve(input_bufs.size());
    for (const auto &[iid, buf] : input_bufs) {
        COAST_REQUIRE(buf != nullptr, core::error_kind::contract_violation);
        if (!input_stats.contains(iid))
            input_stats.emplace(iid, compute_input_stats(buf->cur_read_buffer()));
    }

    // copy input to observe buffer (read back in perform_observations)
    inp_observe.clear();
    for (const auto &[iid, buf] : input_bufs)
        inp_observe.append_range(buf->cur_read_buffer());

    // get the current activity of all neurons in parallel
    // Static partitioner pins each index range to the same thread across calls,
    // improving L3 cache reuse for synapse data compared to TBB's default
    // work-stealing partitioner.
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, neurons.size()),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t i = r.begin(); i != r.end(); ++i)
                out[i] = neurons[i].get_response(input_stats);
        },
        tbb::static_partitioner{}
    );

    // simulate local inhibition within the neuron group (swappable; default
    // max_softmax). out is inhibited in place (raw and out alias).
    inhibitor_->inhibit(out, out);

    /*
     * Simulate the adaption of the neurons in the neuron group to the current input signal.
     * 1) We determine the maximum activity in the neuron group.
     * 2) We determine a stochastic "winning" threshold.
     * 3) All neurons that reach the winning threshold will primarily adapt to the input.
     *    The strength of the adaptation depends on the activity of the neuron and is limited
     *    by a weight filter that reduces adaptation if the neuron is already strongly active
     *    in response to an input. This results in adaption happening mostly for inputs that
     *    are not yet well known.
     * 4) All other neurons irrespective of their activity will adapt somewhat to an input.
     *    The strength of the adaption depends on the neurons activity in relation to the overall
     *    activity of the neuron group and a filter that reduces adaption of already strongly activated
     *    neurons. The strength is also scaled down by the "common learning rate" parameter.
     *    The winner is not adapted again here: every neuron adapts exactly once per step, the
     *    winner with its factor from 3) instead of the common one (SPEC.md M.4.14).
     */

    // ---- frozen (inference-only) mode ----
    // Everything below mutates learning state (winner draw, adapt_synapses,
    // adapt_branches). A frozen group is a pure forward pass: stop here.
    if (params.frozen) {
        return {};
    }

    // ---- zero group output (SPEC.md M.4.15) ----
    // When every output is 0 (they lie in [0,1], so the sum is 0 exactly then)
    // no neuron produced output to attribute learning to: the step is not a
    // learning step and, like a frozen one, changes no learning state.
    const float act_sum = std::reduce(out.begin(),out.end());
    if (act_sum == 0.0f) {
        return {};
    }

    for (auto &neuron : neurons)
        neuron.update_max_activity();

    // 1)
    const float mx_act = std::reduce(out.begin(),out.end(),0.0f,[](const float a, const float b){ return std::max(a,b); });
    PROBE("max activity",mx_act);

    // 2)
    std::uniform_real_distribution<float> dis ( mx_act * stochastic_win_thres, mx_act );
    float win_act = dis(rgen);
    PROBE("winning thres",win_act);
    // Which of the neurons clearing win_act wins: uniform among them, drawn on
    // every learning step (SPEC.md M.4.4).
    const float win_pick = std::uniform_real_distribution<float>(0.0f, 1.0f)(rgen);

    // Precompute sigmoid params once — eliminates pow() from every per-neuron sigmoid call below.
    const sigmoid_params_t wf_params(weight_filter);

    // Activity window tracking: this step adapts with the windows the edges
    // reached at the end of the previous learning step.
    const learning_window_t window_winner = ngm::window_tracking::winner_window(window_edges);
    const learning_window_t window_other  = ngm::window_tracking::other_window(window_edges);

    // 3)
    double winner_weight = 0.0;
    const std::size_t winner_idx = pick_winner(out, win_act, win_pick);   // out.size() = no winner
    if (winner_idx < out.size()) {
        winner_weight = sigmoid(1.0f - out[winner_idx], wf_params);
        neurons[winner_idx].adapt( (float)winner_weight, input_stats, window_winner );
    }
    PROBE("winner weight",winner_weight);

    // 4)
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, neurons.size()),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                if (i == winner_idx) continue;   // adapted once, in 3)
                const float sec_weight = sigmoid(1.0f - (out[i] / act_sum), wf_params);
                neurons[i].adapt(sec_weight * common_learning_rate, input_stats, window_other);
            }
        },
        tbb::static_partitioner{}
    );

    // calculate average secondary weight for probe
    const float avg_sec = std::reduce(out.begin(),out.end(),0.0f,[&](const float a, const float b) {
        return a + sigmoid(1.0f - (b / act_sum), wf_params);
    }) / (float)out.size();
    PROBE("average sec. weight",avg_sec);

    // advance the window edges with this step's raw activities
    activity_scratch.resize(neurons.size());
    for (std::size_t i = 0; i < neurons.size(); ++i)
        activity_scratch[i] = neurons[i].get_activity();
    ngm::window_tracking::update(window_edges, activity_scratch, winner_idx);
    PROBE("window winner lower",window_edges.winner_lower);
    PROBE("window winner upper",window_edges.winner_upper);
    PROBE("window other lower",window_edges.other_lower);
    PROBE("window other upper",window_edges.other_upper);


    return {};
}

std::pair<learning_window_t,learning_window_t> neuron_group_t::get_tracked_windows() const
{
    return { ngm::window_tracking::winner_window(window_edges), ngm::window_tracking::other_window(window_edges) };
}

void neuron_group_t::perform_observations(observe::observe_accessor& acc)
{
    if (acc.group_active("population")) {
        // inp_observe holds this step's concatenated input (rebuilt each
        // process(), before the observation pass runs).
        acc.observe(*this, "population", "in", std::span<const float>(inp_observe));
        if (output_buf != nullptr) {
            // The write page still holds this step's post-inhibition output: the
            // observation pass runs in the post-process hook, before swap_io().
            std::span<float> out = output_buf->cur_write_buffer();
            acc.observe(*this, "population", "out", std::span<const float>(out));
        }
    }
    entity_element::perform_observations(acc);   // recurse into neurons -> dendrites
}

// interface function that allows the simulation environment to query the output ID of this io entity

sim::output_id_t neuron_group_t::get_outp_id() const
{
    return id;
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

std::size_t neuron_group_t::get_max_representation_count() const
{
    std::size_t result = 0;
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
            assert(std::isfinite(summed_theta));  // allow 0; reject NaN/inf
            result += summed_theta;
            cnt += static_cast<float>(vec.size());
        }
    }
    assert(std::isfinite(result));  // allow 0; reject NaN/inf
    assert(std::isnormal(cnt));
    return result / cnt;
}

// State persistence implementation

logging::config_record neuron_group_t::current_config() const
{
    logging::config_record r;
    r.push_back({ "local_inhibition_strength",      local_inhibition_strength });
    r.push_back({ "common_learning_rate",           common_learning_rate });
    r.push_back({ "weight_filter.steepness",        weight_filter.steepness });
    r.push_back({ "weight_filter.transition_point", weight_filter.transition_point });
    r.push_back({ "stochastic_win_thres",           stochastic_win_thres });
    r.push_back({ "effective_seed", std::uint64_t{ std::to_underlying(effective_seed_) }, /*random_state=*/true });
    inhibition::append_config(r, inhibitor_->params());

    // Neurons are built set by set, dendrites dendrite set by dendrite set (see the
    // constructors), so a running index finds each instance's cfg. The first
    // instance of a cfg records its settings as the cfg's ("neuron[<cfg>].*"); a
    // setter can change one instance alone, so every later instance records the
    // settings in which it differs from that first one, under its index:
    // "neuron[<cfg>]#<n>.*", "dendrite[<cfg>]#<n>:<d>.*" (the checkpoint's
    // neurons/<n>/dendrites/<d>).
    std::map<std::string, logging::config_record> first_neuron, first_dendrite;   // by cfg
    auto record = [&r](auto& first, const std::string& cfg, const std::string& kind,
                       const std::string& index, const auto& instance) {
        logging::config_record own;
        auto it = first.find(cfg);
        if (it == first.end()) {
            instance.append_config(own, kind + "[" + cfg + "].");
            r.insert(r.end(), own.begin(), own.end());
            first.emplace(cfg, std::move(own));
            return;
        }
        // Same function, same order: entry k is the same setting in both records.
        instance.append_config(own, kind + "[" + cfg + "]#" + index + ".");
        for (std::size_t k = 0; k < own.size(); ++k)
            if (own[k].value != it->second[k].value) r.push_back(std::move(own[k]));
    };
    std::size_t ni = 0;
    for (const auto& ns : params.neurons)
        for (std::size_t k = 0; k < ns.neuron_count; ++k, ++ni) {
            const auto& n = neurons[ni];
            record(first_neuron, ns.neuron_cfg, "neuron", std::to_string(ni), n);
            std::size_t di = 0;
            for (const auto& ds : n.get_params().dendrites)
                for (std::size_t m = 0; m < ds.dendrite_count; ++m, ++di)
                    record(first_dendrite, ds.dendrite_cfg, "dendrite", std::format("{}:{}", ni, di),
                           n.get_dendrite(di));
        }
    return r;
}

void neuron_group_t::save_state(logging::state_writer& writer) const
{
    // Settings are not state: the configuration wins on load (M.7.8), and the
    // record written last is what a load compares against.

    // Save neuron count for validation on load
    writer.write("neuron_count", static_cast<std::uint64_t>(neurons.size()));

    // The seed the random state below descends from (M.2.4): a load restores both.
    writer.write("effective_seed", std::to_underlying(effective_seed_));
    logging::write_rng(writer, "rng", rgen);

    // Save each neuron's state - delegate to neuron_t::save_state for consistency with load path
    writer.begin_group("neurons");
    for (std::size_t ni = 0; ni < neurons.size(); ++ni) {
        writer.begin_group(std::to_string(ni));
        neurons[ni].save_state(writer);
        writer.end_group();
    }
    writer.end_group();

    // Inhibition state (scoped). max_softmax is stateless so this is then an
    // empty group, but stateful kinds round-trip here — string-keyed, like rng.
    writer.begin_group("inhibition");
    inhibitor_->save_state(writer);
    writer.end_group();

    ngm::window_tracking::save(writer, window_edges);

    logging::write_config_record(writer, current_config());
}

void neuron_group_t::load_state(logging::state_reader& reader)
{
    // A checkpoint from a differently shaped group cannot be loaded into this one.
    const auto who = std::format("ngm2 neuron group {}", id);
    logging::check_shape(reader, who, { { "neuron_count", neurons.size() } });

    // Checkpoints older than format version 3 were written in wiring order and
    // do not record a dendrite's channel order (SPEC.md M.7.5), so a dendrite
    // with several channels cannot match its learned synapses to them: refuse
    // before anything is assigned. A single channel has no order, so it loads.
    if (reader.format_version() < 3)
        for (std::size_t ni = 0; ni < neurons.size(); ++ni) {
            const auto &dends = neurons[ni].get_dendrites();
            for (std::size_t di = 0; di < dends.size(); ++di)
                if (dends[di].get_channel_count() > 1)
                    throw std::runtime_error(std::format(
                        "{}: neuron {}: dendrite {}: checkpoint format version {} does not record the "
                        "channel order, and this dendrite has {} input channels, so its learned synapses "
                        "cannot be matched to them; the checkpoint cannot be loaded "
                        "(single-channel dendrites are unaffected)",
                        who, ni, di, reader.format_version(), dends[di].get_channel_count()));
        }

    // Every structural size -- each neuron's dendrite count and each
    // dendrite's type, tree depth and synapse shape (input width, segment ids)
    // -- is checked here, read-only, before anything below is assigned (M.7.18).
    // The channel record itself (M.7.5) is only checked from format version 4
    // on, a dendrite's type and depth from version 5 on; an older checkpoint's
    // depths are checked against its configuration record instead, and its
    // types cannot be checked at all (#236).
    std::vector<std::string> unrecorded_depths;
    if (reader.format_version() < 5)
        unrecorded_depths = check_recorded_depths(reader, neurons, who);
    check_structure(reader, neurons, who);
    if (reader.format_version() < 5) {
        std::string note(who);
        for (std::size_t i = 0; i < unrecorded_depths.size(); ++i)
            note += std::format("{}{}", i == 0 ? " (nor the tree depth of dendrite cfg " : ", ",
                                unrecorded_depths[i]);
        reader.note_unchecked_dendrite_types(unrecorded_depths.empty() ? note : note + ")");
    }
    if (reader.format_version() < 4)
        reader.note_unrecorded_channels(who);

    // The configuration wins (M.7.8): settings keep the values this group was
    // built with, whatever the checkpoint recorded or `frozen` says; only the
    // differences are reported. Learned state and random state follow.
    logging::check_config_record(reader, current_config(), who);

    // Each neuron checks its part before assigning it, but a neuron or dendrite
    // that throws leaves the ones before it loaded: unlike ngm3, a throw here does
    // not leave the group unchanged.
    auto corrupt = [&](std::string_view what) {
        return std::runtime_error(std::format("{}: corrupt checkpoint — {}", who, what));
    };
    // The streams are the checkpoint's, so is the seed they came from (M.2.4):
    // effective_seed() -- the run manifest's -- and the next save name it, not
    // the configured one, which check_config_record above has reported as not
    // applied. Read before the rng, so a malformed seed throws before either is
    // assigned. A checkpoint without an rng (predating it) or without any
    // record of its seed keeps the constructed one (pre-field rule, state_checks.h).
    const auto saved_seed = checkpoint_seed(reader);
    if (logging::read_rng(reader, "rng", rgen, who) && saved_seed)
        effective_seed_ = *saved_seed;

    if (!reader.enter_group("neurons"))
        throw corrupt("neurons missing");

    for (std::size_t ni = 0; ni < neurons.size(); ++ni) {
        if (!reader.enter_group(std::to_string(ni)))
            throw corrupt(std::format("neuron {} missing", ni));
        neurons[ni].load_state(reader, std::format("{}: neuron {}", who, ni));
        reader.leave_group();
    }

    reader.leave_group();

    if (reader.enter_group("inhibition")) {
        inhibitor_->load_state(reader, neurons.size());
        reader.leave_group();
    }

    // Window edges (a format-1 checkpoint has none: start over, see read()).
    window_edges = ngm::window_tracking::read(reader, who);
}

}
