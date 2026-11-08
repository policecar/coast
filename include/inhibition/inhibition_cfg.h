//
// Created for the COAST swappable-inhibition feature.
//
// Typed configuration for neuron-group inhibition. Mirrors the established
// dendrite_params_t pattern (a `type` enum discriminator + a superset of
// per-kind fields), so it (de)serializes via the same NLOHMANN macros and is
// editable by the config editor with no bespoke code.
//

#ifndef COAST_INHIBITION_CFG_H
#define COAST_INHIBITION_CFG_H

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "core/seed.h"
#include "logging/config_record.h"

namespace inhibition {

// Kind selector. Open set — new kinds append a value here and a matching case
// in make_inhibitor(). (For a fixed set this would be over-engineering, but
// inhibition is an active experimentation surface; see fast-plan.md.)
enum class inhibition_type_t : std::uint8_t {
    random_index,  // random recurrent lateral inhibition (CPU-only; see random_index.h)
    max_softmax,   // max-rescaled softmax, out = max*exp(-strength*(max-v)) (CPU + GPU; see max_softmax.h)
    TYPE_COUNT
};

// Variant selector for the random_index kind. Currently a single variant; the
// enum leaves room for more (the threshold rule is the only modelled one so far).
enum class random_index_variant_t : std::uint8_t {
    threshold_linear,  // T = (M < reach) ? strength : 0;  inh = T·raw
    TYPE_COUNT
};

// One typed config entry, referenced by name from a neuron group's
// `inhibition_cfg`. Fields are a superset across kinds (like dendrite_params_t).
struct inhibition_params_t {
    inhibition_type_t type     = inhibition_type_t::max_softmax;
    double            strength = 10.0;  // max_softmax: beta

    // random_index fields (ignored by other kinds). The matrix itself is not a
    // config field — it is generated from ri_seed on first use and persisted in
    // the simulation state, not the config.
    random_index_variant_t ri_variant        = random_index_variant_t::threshold_linear;
    double                 ri_reach          = 0.1;   // P[connection]; fraction of M thresholded into T
    double                 ri_strength       = 0.5;   // weight of each inhibitory connection
    seed_t                 ri_seed           = {};     // seed for the permanence matrix; composed with the group's offsets (make_inhibitor)
    bool                   ri_seed_from_time = false; // resolve a fresh seed at construction
};

// The parameters a group's inhibitor is built from: the entry `name` of `dict`, or
// for an empty name a max_softmax with beta = `fallback_strength` (the default
// for configs that name no entry). A non-empty unknown
// name is a config typo: rejected rather than silently degraded to the default.
inline inhibition_params_t resolve_params(
    const std::unordered_map<std::string, inhibition_params_t>& dict,
    const std::string& name,
    double fallback_strength)
{
    if (name.empty()) return inhibition_params_t{ inhibition_type_t::max_softmax, fallback_strength };
    const auto it = dict.find(name);
    if (it == dict.end())
        throw std::runtime_error("inhibition_cfg \"" + name + "\" not found in inhibitions dict");
    return it->second;
}

// Appends the settings `p` uses to a group's configuration record (M.7.8), under
// "inhibition.*": the kind and the fields that kind reads. random_index's seed is
// random state (its matrix M comes from the checkpoint); reach and strength are
// settings, applied to the saved M.
void append_config(logging::config_record& rec, const inhibition_params_t& p);

}

#endif // COAST_INHIBITION_CFG_H
