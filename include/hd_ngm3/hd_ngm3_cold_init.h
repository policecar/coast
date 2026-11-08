//
// ngm3 cold initialisation — the ONE place that owns the group's initial RNG
// contract, shared by the CPU group (neuron_group_t::init_storage) and the GPU
// group (neuron_group_gpu_t::build_step_plan).
//
// The two groups must start from bit-identical permanences (the CPU<->GPU
// checkpoint and equivalence tests compare them). Before this lived here, the
// GPU replayed the CPU constructor's draws by hand — "fresh seed, then
// neuron_cnt pre-draws" — so any change to the CPU's per-neuron seeding broke
// parity silently. Now both call cold_init(), and the order below is the
// contract, in one place:
//
//   1. master = fast_rng_t(seed)                     seed = random_seed + seed_offset
//   2. neuron_cnt draws from master seed the per-neuron streams, neuron 0 first
//   3. rep 0 of every apical dendrite is drawn from master, dendrite-major
//      ([neuron][dendrite] order), input positions ascending within a dendrite,
//      with one ngm::cold_permanence_t for the whole bank; reps 1.. are copies
//   4. the same for the proximal bank
//
// The CPU group continues its group-level draws (winner selection, split
// wiggle) from the master as cold_init() leaves it; the GPU group uses only
// the permanences (its device draws are hash-keyed).
//

#ifndef HD_NGM3_COLD_INIT_H
#define HD_NGM3_COLD_INIT_H

#include <cstddef>
#include <vector>

#include "core/rng/fast_rng.h"
#include "core/seed.h"

namespace ngm3 {

// A bank's extent. The bank is absent (no storage, no draws) when either is 0.
struct bank_extent_t {
    std::size_t d_cnt      = 0;   // dendrites per neuron
    std::size_t input_size = 0;   // routed input width
};

struct cold_state_t {
    fast_rng_t              master;         // after every cold draw
    std::vector<fast_rng_t> neuron_rgens;   // [neuron_cnt]
    // [neuron][dendrite][rep][input]; every rep a copy of rep 0. Empty for an
    // absent bank.
    std::vector<float>      apical_perm;
    std::vector<float>      proximal_perm;
};

// Throws std::invalid_argument on rep_cnt == 0 or a permanence_threshold that
// gives no valid Poisson mean (see ngm/cold_init.h).
[[nodiscard]] cold_state_t cold_init(seed_t        seed,
                                     std::size_t   neuron_cnt,
                                     std::size_t   rep_cnt,
                                     float         permanence_threshold,
                                     bank_extent_t apical,
                                     bank_extent_t proximal);

} // namespace ngm3

#endif // HD_NGM3_COLD_INIT_H
