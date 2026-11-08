//
// Created for the COAST swappable-inhibition feature.
//
// max_softmax: softmax over the group output, rescaled so the winner keeps its
// pre-inhibition activity. The softmax normaliser cancels in that rescaling:
//
//   out_i = max(v) * softmax(beta*v)_i / max_j softmax(beta*v)_j
//         = max(v) * exp(-beta * (max(v) - v_i))
//
// so each neuron is set to the leader's activity times exp(-beta * gap), where
// gap is how far it trails the leader in absolute activity. The leader is never
// weakened and nothing exceeds it, so a weak response stays at or below its own
// leader (a plain softmax would scale it up to sum 1). Not damping-only:
// a neuron is raised when exp(-beta*gap) > v_i/max (small beta, or a weak leader
// with small gaps); beta = 0 sets every neuron to the max. beta is the inhibition
// strength in units of 1/activity: a neuron 0.1 behind the leader ends at
// exp(-0.1*beta) times the leader's activity.
//

#ifndef COAST_INHIBITION_MAX_SOFTMAX_H
#define COAST_INHIBITION_MAX_SOFTMAX_H

#include <span>

#include "core/entity_element_macros.h"
#include "inhibition/inhibition.h"

namespace inhibition {

// Free function holding the math, exposed for isolated unit testing.
// Copies raw -> out when they differ (no-op when aliased), then applies the
// in-place transform to out. Precondition: beta is finite and >= 0.
void max_softmax_apply(std::span<const float> raw, std::span<float> out, float beta);

class max_softmax_t : public inhibitor_t {
    float beta_;
public:
    // Throws std::invalid_argument unless beta is finite and >= 0.
    explicit max_softmax_t(const inhibition_params_t& p);  // beta = p.strength

    ELEMENT_TYPE(max_softmax_t, "max_softmax")

    void inhibit(std::span<const float> raw, std::span<float> out) override {
        max_softmax_apply(raw, out, beta_);
    }

    // Stateless: save/load_state use the no-op base defaults.

    [[nodiscard]] float beta() const { return beta_; }
};

}

#endif // COAST_INHIBITION_MAX_SOFTMAX_H
