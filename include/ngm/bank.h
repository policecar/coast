//
// Dendrite bank identity and input routing, shared by ngm3 and ngm2.5 (both
// re-export them: ngm3::bank_t, ngm25::bank_t, ... are these types). One
// definition, so the two models' configs, records and routing cannot drift.
//

#ifndef COAST_NGM_BANK_H
#define COAST_NGM_BANK_H

#include <cstdint>
#include <string>

namespace ngm {

// Which bank of a neuron a dendrite (or a routed input channel) belongs to.
// The values are part of the checkpoint format (ngm25 dend_type) — do not
// renumber.
enum class bank_t : std::uint8_t { apical = 0, proximal = 1 };

// One configured input channel and the bank it feeds.
struct input_route_t {
    std::string channel_name;
    bank_t      bank = bank_t::proximal;
};

} // namespace ngm

#endif // COAST_NGM_BANK_H
