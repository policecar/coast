//
// Created for the COAST swappable-inhibition feature.
//

#include "inhibition/max_softmax.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace inhibition {

max_softmax_t::max_softmax_t(const inhibition_params_t& p)
    : inhibitor_t(p), beta_(static_cast<float>(p.strength))
{
    if (!std::isfinite(beta_) || beta_ < 0.0f)
        throw std::invalid_argument("max_softmax: beta (strength) must be finite and >= 0, got " +
                                    std::to_string(beta_));
}

void max_softmax_apply(std::span<const float> raw, std::span<float> out, float beta)
{
    assert(std::isfinite(beta) && beta >= 0.0f);
    assert(raw.size() == out.size());
    if (raw.data() != out.data())
        std::copy(raw.begin(), raw.end(), out.begin());
    if (out.empty())
        return;

    // Closed form of the max-rescaled softmax (see max_softmax.h). Computed via
    // the gap to the max, which is also the numerically stable softmax form.
    const float max_val = std::reduce(out.begin(), out.end(), 0.0f,
                                      [](float a, float b) { return std::max(a, b); });
    if (!std::isnormal(max_val))
        return;
    for (auto &val : out)
        val = max_val * std::exp(-beta * (max_val - val));
}

}
