//
// Created for the COAST swappable-inhibition feature.
//

#include "inhibition/inhibition.h"
#include "inhibition/random_index.h"
#include "inhibition/max_softmax.h"

#include <stdexcept>

namespace inhibition {

std::unique_ptr<inhibitor_t> make_inhibitor(const inhibition_params_t& params)
{
    switch (params.type) {
        case inhibition_type_t::random_index:
            return std::make_unique<random_index_t>(params);
        case inhibition_type_t::max_softmax:
            return std::make_unique<max_softmax_t>(params);
        case inhibition_type_t::TYPE_COUNT:
            break;
    }
    throw std::runtime_error("make_inhibitor: unhandled inhibition_type_t");
}

void append_config(logging::config_record& rec, const inhibition_params_t& p)
{
    switch (p.type) {
        case inhibition_type_t::max_softmax:
            rec.push_back({ "inhibition.type", std::string("max_softmax") });
            rec.push_back({ "inhibition.strength", p.strength });
            return;
        case inhibition_type_t::random_index:
            rec.push_back({ "inhibition.type", std::string("random_index") });
            rec.push_back({ "inhibition.ri_variant", static_cast<std::uint64_t>(p.ri_variant) });
            rec.push_back({ "inhibition.ri_reach", p.ri_reach });
            rec.push_back({ "inhibition.ri_strength", p.ri_strength });
            // ri_seed here is the composed seed (config + offsets, make_inhibitor),
            // like the groups' effective_seed. A seed drawn at construction is
            // random state, not a setting: every
            // construction draws a new one, so recording it would report a change
            // on every resume. The state saves it with the matrix.
            if (!p.ri_seed_from_time)
                rec.push_back({ "inhibition.ri_seed", std::uint64_t{ std::to_underlying(p.ri_seed) }, /*random_state=*/true });
            rec.push_back({ "inhibition.ri_seed_from_time", p.ri_seed_from_time, /*random_state=*/true });
            return;
        case inhibition_type_t::TYPE_COUNT:
            break;
    }
    throw std::runtime_error("append_config: unhandled inhibition_type_t");
}

std::unique_ptr<inhibitor_t> make_inhibitor(
    const std::unordered_map<std::string, inhibition_params_t>& dict,
    const std::string& name,
    double fallback_strength,
    seed_offset_t seed_offset)
{
    // resolve_params rejects an unknown name and synthesizes the max_softmax
    // default for an empty one.
    auto p = resolve_params(dict, name, fallback_strength);
    p.ri_seed = p.ri_seed + seed_offset;
    return make_inhibitor(p);
}

}
