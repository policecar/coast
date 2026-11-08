//
// Created for the COAST swappable-inhibition feature.
//
// Polymorphic inhibition interface owned by a neuron group. The set of kinds is
// open (active experimentation), so this is a real hierarchy rather than the
// single-class-plus-enum pattern used for the fixed dendrite types. Derives
// core::entity_element so an inhibitor can be a probe-able child of its group
// (uniform with how dendrite_t/neuron_t are entity_elements for probing).
//

#ifndef COAST_INHIBITION_H
#define COAST_INHIBITION_H

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>

#include "core/entity_element.h"
#include "inhibition/inhibition_cfg.h"

namespace logging {
class state_writer;
class state_reader;
}

namespace inhibition {

class inhibitor_t : public core::entity_element {
public:
    ~inhibitor_t() override = default;

    // The parameters this inhibitor runs with: the resolved entry it was built
    // from, so a seed drawn at construction (ri_seed_from_time) is the drawn seed.
    // The owning group's configuration record (M.7.8) is written from this,
    // except a drawn seed, which is random state and saved with the state.
    [[nodiscard]] const inhibition_params_t& params() const { return params_; }

    // Transform raw per-neuron activities into the inhibited output. raw and out
    // may alias (CPU groups that inhibit in place pass out for both); when they
    // differ, the implementation copies raw -> out first. Precondition:
    // raw.size() == out.size() == neuron count.
    virtual void inhibit(std::span<const float> raw, std::span<float> out) = 0;

    // State persistence. Default no-op for stateless kinds (e.g. max_softmax). The
    // owning group scopes these under begin_group("inhibition") so stateful
    // kinds round-trip across save/load (string-keyed, like write_rng).
    // load_state is all or nothing: it reads and checks the whole checkpoint
    // first and throws, leaving the inhibitor as it was, if it does not fit a
    // group of `neuron_count` neurons. The owning group calls it before it
    // commits its own state (M.7.18).
    virtual void save_state(logging::state_writer&) const {}
    virtual void load_state(logging::state_reader&, [[maybe_unused]] std::size_t neuron_count) {}

protected:
    explicit inhibitor_t(const inhibition_params_t& p) : params_(p) {}

    inhibition_params_t params_;
};

// Construct the inhibitor selected by params.type. Central, model-agnostic
// factory; adding a kind appends one case (no registry needed for first-party
// research kinds — promote to a self-registering map only if this ever bites).
std::unique_ptr<inhibitor_t> make_inhibitor(const inhibition_params_t& params);

// Resolve a named entry from `dict` and construct it (resolve_params). An empty
// `name` synthesizes a max_softmax with beta = `fallback_strength` (the group's
// local_inhibition_strength); an unknown one throws. Model-agnostic: the group passes
// its own cfg_store dict.
//
// `seed_offset` is the owning group's offset (global --seed-offset + the node's
// random_seed_offset). It is added to the entry's ri_seed like to every other
// config seed (S.3.24), so a seed sweep varies the random_index matrix too. A
// seed drawn with ri_seed_from_time replaces the composed one.
std::unique_ptr<inhibitor_t> make_inhibitor(
    const std::unordered_map<std::string, inhibition_params_t>& dict,
    const std::string& name,
    double fallback_strength,
    seed_offset_t seed_offset);

// The seed of a stateful inhibitor's random state (random_index: ri_seed, as
// composed or drawn at construction, or as restored from a checkpoint), for the
// run manifest; nullopt for kinds without one.
[[nodiscard]] inline std::optional<seed_t> random_state_seed(const inhibition_params_t& p)
{
    if (p.type == inhibition_type_t::random_index)
        return p.ri_seed;
    return std::nullopt;
}

}

#endif // COAST_INHIBITION_H
