//
// Created for the COAST swappable-inhibition feature.
//

#include "inhibition/gpu_inhibition.h"
#include "inhibition/gpu_max_softmax.h"

#include <stdexcept>

namespace inhibition {

std::unique_ptr<gpu_inhibitor_t> make_gpu_inhibitor(const inhibition_params_t& params)
{
    switch (params.type) {
        case inhibition_type_t::random_index:
            // CPU-only this cycle: no GPU port yet. Reject with a clear message
            // rather than crash or silently fall back to a different kind.
            throw std::runtime_error(
                "make_gpu_inhibitor: inhibition kind \"random_index\" is CPU-only; "
                "run this configuration on the CPU backend");
        case inhibition_type_t::max_softmax:
            return std::make_unique<gpu_max_softmax_t>(params);
        case inhibition_type_t::TYPE_COUNT:
            break;
    }
    throw std::runtime_error("make_gpu_inhibitor: unhandled inhibition_type_t");
}

std::unique_ptr<gpu_inhibitor_t> make_gpu_inhibitor(
    const std::unordered_map<std::string, inhibition_params_t>& dict,
    const std::string& name,
    double fallback_strength)
{
    // resolve_params rejects an unknown name and synthesizes the max_softmax
    // default for an empty one.
    return make_gpu_inhibitor(resolve_params(dict, name, fallback_strength));
}

}
