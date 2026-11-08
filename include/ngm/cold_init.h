//
// The cold (initial) permanence draw shared by every ngm-family model (ngm2,
// ngm2.5, ngm3 CPU and GPU host init).
//
// A fresh synapse's permanence is Poisson-distributed around the permanence
// threshold tau: std::poisson_distribution operates on integers, so tau is
// scaled to [0, 100], the draw scaled back and clamped to [0, 1]:
//
//     perm = clamp(Poisson(int(100.0f * tau)) / 100, 0, 1)
//
// The scale is computed in float on purpose: in double, int(100 * tau) differs
// for some tau (0.29 gives 28 in double, 29 in float), and every model has
// always used the float value.
//
// The ENGINE is the caller's: the models draw from different engines (ngm2 a
// dendrite's stream, ngm3 the group master, ngm2.5 its D5 cold engine), and
// their draw ORDER is part of each model's reproducibility contract, so this
// helper owns neither. It owns the distribution: keep one cold_permanence_t
// across a whole fill (the scope at which each model has always kept its
// std::poisson_distribution), since a distribution object may carry state
// between draws.
//

#ifndef COAST_NGM_COLD_INIT_H
#define COAST_NGM_COLD_INIT_H

#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>

namespace ngm {

class cold_permanence_t {
public:
    // tau must give a Poisson mean >= 1 (std::poisson_distribution requires a
    // positive mean); every model enforces tau in [0.01, 1), which does.
    explicit cold_permanence_t(float permanence_threshold)
      : dist_(mean_of(permanence_threshold)) {}

    template <class URBG>
    float operator()(URBG &engine)
    {
        return std::clamp(static_cast<float>(dist_(engine)) / 100.0f, 0.0f, 1.0f);
    }

private:
    static int mean_of(float permanence_threshold)
    {
        // Range first: the int conversion of a NaN or huge value is UB.
        const bool in_range = permanence_threshold > 0.0f && permanence_threshold <= 1.0f;
        const int  mean     = in_range ? static_cast<int>(100.0f * permanence_threshold) : 0;
        if (mean < 1)
            throw std::invalid_argument(
                "ngm::cold_permanence_t: permanence_threshold "
                + std::to_string(permanence_threshold)
                + " gives a Poisson mean < 1 (need tau >= 0.01)");
        return mean;
    }

    std::poisson_distribution<int> dist_;
};

} // namespace ngm

#endif // COAST_NGM_COLD_INIT_H
