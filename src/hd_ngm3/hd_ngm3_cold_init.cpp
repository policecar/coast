#include "hd_ngm3/hd_ngm3_cold_init.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>

#include "ngm/cold_init.h"

namespace ngm3 {

namespace {

// Step 3/4 of the contract (header): one bank, drawn from `master`.
std::vector<float> cold_bank(fast_rng_t &master, std::size_t neuron_cnt, std::size_t rep_cnt,
                             float permanence_threshold, bank_extent_t bank)
{
    if (bank.d_cnt == 0 || bank.input_size == 0) return {};

    const std::size_t dendrite_block = rep_cnt * bank.input_size;
    std::vector<float> perm(neuron_cnt * bank.d_cnt * dendrite_block, 0.0f);

    ngm::cold_permanence_t cold_perm(permanence_threshold);
    for (std::size_t d_block = 0; d_block < neuron_cnt * bank.d_cnt; ++d_block) {
        const std::span<float> dendrite{ perm.data() + d_block * dendrite_block, dendrite_block };
        for (std::size_t i = 0; i < bank.input_size; ++i)
            dendrite[i] = cold_perm(master);
        for (std::size_t r = 1; r < rep_cnt; ++r)
            std::copy_n(dendrite.begin(), bank.input_size, dendrite.begin() + r * bank.input_size);
    }
    return perm;
}

} // namespace

cold_state_t cold_init(seed_t seed, std::size_t neuron_cnt, std::size_t rep_cnt,
                       float permanence_threshold, bank_extent_t apical, bank_extent_t proximal)
{
    if (rep_cnt == 0)
        throw std::invalid_argument("ngm3::cold_init: rep_cnt must be >= 1");

    cold_state_t s{ fast_rng_t(std::to_underlying(seed)), {}, {}, {} };

    s.neuron_rgens.reserve(neuron_cnt);
    for (std::size_t n = 0; n < neuron_cnt; ++n)
        s.neuron_rgens.emplace_back(static_cast<std::uint32_t>(s.master()));

    s.apical_perm   = cold_bank(s.master, neuron_cnt, rep_cnt, permanence_threshold, apical);
    s.proximal_perm = cold_bank(s.master, neuron_cnt, rep_cnt, permanence_threshold, proximal);
    return s;
}

} // namespace ngm3
