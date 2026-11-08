// tests/unit/hd_ngm3/test_cold_init.cpp
//
// ngm3::cold_init owns the cold RNG contract the CPU and GPU group share
// (hd_ngm3_cold_init.h). These pin that contract against a spelled-out
// reference, and check that the CPU group's initial state IS cold_init's — the
// GPU group's use of it is covered by the CPU<->GPU checkpoint/equivalence
// tests (they need an adapter).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "hd_ngm3/hd_ngm3_cfg.h"
#include "hd_ngm3/hd_ngm3_cold_init.h"
#include "hd_ngm3/hd_ngm3_neuron_group.h"
#include "logging/mem_state_io.h"
#include "logging/rng_state_io.h"
#include "ngm/cold_init.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t k_neurons = 3;
constexpr std::size_t k_reps    = 4;       // branch depth 2
constexpr float       k_tau     = 0.29f;   // int(100 * tau) differs float vs double
constexpr seed_t      k_seed{ 11 };

// The contract, spelled out independently of cold_init: one bank's rep-0
// draws, dendrite-major, inputs ascending, one distribution per bank.
std::vector<float> reference_bank(fast_rng_t& master, ngm3::bank_extent_t b)
{
    if (b.d_cnt == 0 || b.input_size == 0) return {};
    std::vector<float> perm;
    ngm::cold_permanence_t cold(k_tau);
    for (std::size_t nd = 0; nd < k_neurons * b.d_cnt; ++nd) {
        std::vector<float> rep0;
        for (std::size_t i = 0; i < b.input_size; ++i) rep0.push_back(cold(master));
        for (std::size_t r = 0; r < k_reps; ++r) perm.insert(perm.end(), rep0.begin(), rep0.end());
    }
    return perm;
}

void check_contract(ngm3::bank_extent_t apical, ngm3::bank_extent_t proximal)
{
    auto s = ngm3::cold_init(k_seed, k_neurons, k_reps, k_tau, apical, proximal);

    fast_rng_t master(std::to_underlying(k_seed));
    std::vector<std::uint32_t> neuron_seeds;
    for (std::size_t n = 0; n < k_neurons; ++n)
        neuron_seeds.push_back(static_cast<std::uint32_t>(master()));
    const auto ref_apical   = reference_bank(master, apical);
    const auto ref_proximal = reference_bank(master, proximal);

    CHECK(s.apical_perm   == ref_apical);
    CHECK(s.proximal_perm == ref_proximal);
    REQUIRE(s.neuron_rgens.size() == k_neurons);
    for (std::size_t n = 0; n < k_neurons; ++n) {
        fast_rng_t ref(neuron_seeds[n]);
        CHECK(s.neuron_rgens[n]() == ref());
    }
    // The master is left after the last cold draw: the CPU group's next
    // group-level draw is the reference's next draw.
    CHECK(s.master() == master());
}

} // namespace

TEST_CASE("cold_init: draws in the documented order (seeds, apical, proximal)") {
    check_contract({ 2, 5 }, { 1, 7 });
}

TEST_CASE("cold_init: an absent bank is empty and draws nothing") {
    check_contract({ 0, 5 }, { 2, 7 });   // no apical dendrites
    check_contract({ 2, 0 }, { 2, 7 });   // no apical input routed
}

TEST_CASE("cold_init: draws are Poisson(int(100 tau))/100 clamped to [0,1]") {
    auto s = ngm3::cold_init(k_seed, k_neurons, k_reps, k_tau, { 0, 0 }, { 2, 64 });
    REQUIRE(!s.proximal_perm.empty());
    double sum = 0.0;
    for (const float p : s.proximal_perm) {
        CHECK(p >= 0.0f);
        CHECK(p <= 1.0f);
        CHECK(std::round(p * 100.0f) == doctest::Approx(p * 100.0f));   // a multiple of 1/100
        sum += p;
    }
    // Mean 29/100 (float scale; double would give 28). 384 samples of
    // Poisson(29)/100 have a standard error of ~0.0028.
    CHECK(sum / static_cast<double>(s.proximal_perm.size()) == doctest::Approx(0.29).epsilon(0.05));
}

TEST_CASE("cold_permanence_t: refuses a threshold without a valid Poisson mean") {
    CHECK_THROWS_AS(ngm::cold_permanence_t(0.0f),   std::invalid_argument);
    CHECK_THROWS_AS(ngm::cold_permanence_t(0.005f), std::invalid_argument);
    CHECK_THROWS_AS(ngm::cold_permanence_t(-1.0f),  std::invalid_argument);
    CHECK_THROWS_AS(ngm::cold_permanence_t{std::numeric_limits<float>::quiet_NaN()},
                    std::invalid_argument);
    CHECK_NOTHROW(ngm::cold_permanence_t(0.01f));   // the models' lower bound
}

TEST_CASE("cold_init: the CPU group's initial state is cold_init's") {
    struct test_group_t : ngm3::neuron_group_t {
        using ngm3::neuron_group_t::neuron_group_t;
        using ngm3::neuron_group_t::init_storage;
    };
    ngm3::cfg_store cs;
    auto& p = cs.neuron_groups["g"];
    p.neuron_cnt            = k_neurons;
    p.apical_cnt            = 2;
    p.proximal_cnt          = 1;
    p.max_branch_level = 2;
    p.permanence_threshold  = k_tau;
    p.random_seed           = seed_t{ 4 };
    test_group_t g(cs, "g", /*output_id=*/1, seed_offset_t{ 7 });
    g.init_storage(/*apical_in_size=*/5, /*proximal_in_size=*/7);

    // seed = random_seed + seed_offset (11 = k_seed).
    const auto s = ngm3::cold_init(k_seed, k_neurons, k_reps, k_tau, { 2, 5 }, { 1, 7 });

    logging::mem_state_writer w;
    g.save_state(w);
    const auto tree = w.take_root();
    CHECK(tree.groups.at("apical").fields.at("permanences")
          == logging::mem_value{ s.apical_perm });
    CHECK(tree.groups.at("proximal").fields.at("permanences")
          == logging::mem_value{ s.proximal_perm });

    logging::mem_state_writer ref;
    logging::write_rng(ref, "rng", s.master);
    ref.begin_group("neuron_rgens");
    ref.write("count", static_cast<std::uint64_t>(s.neuron_rgens.size()));
    for (std::size_t i = 0; i < s.neuron_rgens.size(); ++i)
        logging::write_rng(ref, std::to_string(i), s.neuron_rgens[i]);
    ref.end_group();
    const auto ref_tree = ref.take_root();
    CHECK(tree.groups.at("rng")          == ref_tree.groups.at("rng"));
    CHECK(tree.groups.at("neuron_rgens") == ref_tree.groups.at("neuron_rgens"));
}
