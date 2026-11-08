// tests/integration_ngm3/branch_gate_after_load_gpu_test.cpp
//
// SPEC.md M.7.21 / M.0.6: the ngm3 GPU group must branch on exactly the steps
// the CPU group does. Both gate on the count of learning steps (input_cnt):
// branching fires when (++input_cnt) % branch_interval == 0. The GPU group used
// to gate on its own step counter instead, which keys its stateless noise draws
// and is NOT the learning-step count: it is absent from a CPU checkpoint (so it
// restarts at 0 after loading one) and it advances on frozen steps. After such
// a load the GPU branched branch_interval steps after the load instead of at
// the next multiple of branch_interval.
//
// The test moves a CPU group's state, taken mid-interval, into a GPU group of
// the same shape and checks the step of the next branch event on both. A short
// interval and a low adapt-history gate make every gate firing split synapses;
// the CPU control asserts that, so the GPU check cannot pass vacuously.
//
// Needs a real GPU adapter (registered with the gpu label, _gpu suffix).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "frozen_inert_helpers.h"
#include "hd_ngm3/hd_ngm3_gpu_neuron_group.h"
#include "hd_ngm3/hd_ngm3_neuron_group.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using frozen_inert::njson;
using frozen_inert::state_of;
using frozen_inert::write_patched;

constexpr std::uint64_t k_interval = 50;
constexpr std::uint64_t k_cpu_steps = 70;   // mid-interval: last branch at 50, next due at 100

void short_interval(njson& j)
{
    auto& g = j["ngm3"]["neuron_groups"]["L4-mnist-ngm3"];
    g["branch_interval"] = k_interval;
    // Every gate firing must find synapses to split. The 1 - Hoyer noise gate
    // damps learning on mnist (attenuation ~0.6), so 30 steps after the load
    // leave the GPU's largest adapt history near 0.03; 0.02 clears it with
    // thousands of synapses on both CPU and GPU.
    g["accumulated_theta_thres"] = 0.02;
}

std::vector<std::uint8_t> segment_ids(const logging::mem_group& state)
{
    const auto& bank = state.groups.at("proximal");
    return std::get<std::vector<std::uint8_t>>(bank.fields.at("segment_ids"));
}

template <typename Group>
Group& the_group(coast::sim_instance& sim)
{
    auto vec = sim.get_sim_env().template get_entities<Group>();
    REQUIRE(vec.has_value());
    REQUIRE((*vec)->size() == 1);
    return (**vec)[0];
}

} // namespace

TEST_CASE("ngm3 GPU branches on the CPU's schedule after loading a mid-interval CPU checkpoint")
{
    const auto cpu_cfg = write_patched("configs/mnist_ngm3.json", "branch_gate_cpu.json", short_interval);
    const auto gpu_cfg = write_patched("configs/mnist_ngm3_gpu_full.json", "branch_gate_gpu.json", short_interval);

    hdf5::executor h5;

    // CPU: run to input_cnt = 70 and take the group's state.
    coast::cli::options cpu_opts;
    cpu_opts.config_path = cpu_cfg.string();
    auto cpu = coast::app::build_env(cpu_opts, h5);
    REQUIRE(cpu);
    auto& cpu_group = the_group<ngm3::neuron_group_t>(*cpu);
    coast::test::run_for(*cpu, k_cpu_steps);
    const auto checkpoint = state_of(cpu_group);

    // CPU control: no branch until input_cnt reaches 100, then a branch.
    const auto cpu_at_70 = segment_ids(checkpoint);
    coast::test::run_for(*cpu, 2 * k_interval - k_cpu_steps - 1);   // -> input_cnt 99
    REQUIRE_MESSAGE(segment_ids(state_of(cpu_group)) == cpu_at_70,
                    "CPU control: segment ids changed before the gate step");
    coast::test::run_for(*cpu, 1);                                   // -> input_cnt 100
    REQUIRE_MESSAGE(segment_ids(state_of(cpu_group)) != cpu_at_70,
                    "CPU control: no branch at the gate step, so this test would be vacuous");

    // GPU: load the CPU state (it carries input_cnt = 70 and no GPU step key).
    coast::cli::options gpu_opts;
    gpu_opts.config_path = gpu_cfg.string();
    auto gpu = coast::app::build_env(gpu_opts, h5);
    REQUIRE(gpu);
    auto& gpu_group = the_group<ngm3::neuron_group_gpu_t>(*gpu);
    {
        logging::mem_state_reader r(checkpoint, logging::k_format_version);
        gpu_group.load_state(r);
    }
    const auto gpu_at_70 = segment_ids(state_of(gpu_group));
    REQUIRE(gpu_at_70 == cpu_at_70);

    coast::test::run_for(*gpu, 2 * k_interval - k_cpu_steps - 1);   // -> input_cnt 99
    CHECK_MESSAGE(segment_ids(state_of(gpu_group)) == gpu_at_70,
                  "GPU branched before the CPU's gate step");
    coast::test::run_for(*gpu, 1);                                   // -> input_cnt 100
    CHECK_MESSAGE(segment_ids(state_of(gpu_group)) != gpu_at_70,
                  "GPU did not branch at the CPU's gate step (input_cnt 100)");
}

namespace {

// Equal apical and proximal banks, both fed the proximal mnist channel, with
// every learning rate 0 (so adaptation cannot move a permanence) and the
// adapt-history gate at 0 (so mismatch alone decides splits). The change in a
// split cell's permanence across the branch step is then exactly its jitter.
void two_equal_banks_no_learning(njson& j)
{
    auto& g = j["ngm3"]["neuron_groups"]["L4-mnist-ngm3"];
    g["apical_cnt"] = g["proximal_cnt"];
    g["inputs"].push_back({ { "channel_name", "in_proximal" }, { "bank", "apical" } });
    g["primary_learning_rate"] = 0.0;
    g["secondary_learning_rate"] = 0.0;
    g["common_learning_rate"] = 0.0;
    g["accumulated_theta_thres"] = 0.0;
    g["branch_interval"] = k_interval;
}

struct bank_view {
    std::vector<float>        perm;
    std::vector<std::uint8_t> seg;
};

bank_view bank(const logging::mem_group& state, const char* name)
{
    const auto& b = state.groups.at(name);
    return { std::get<std::vector<float>>(b.fields.at("permanences")),
             std::get<std::vector<std::uint8_t>>(b.fields.at("segment_ids")) };
}

} // namespace

TEST_CASE("ngm3 GPU branch jitter is drawn independently for the apical and the proximal bank")
{
    // SPEC.md M.0.6 (b): a split clone's permanence jitter is an independent
    // draw per synapse. The GPU keyed it on the cell's flat index within its
    // bank only, so apical and proximal cells at the same index that split on
    // the same step received identical jitter. (The CPU draws them in sequence
    // from one generator, so they are independent there.)
    const auto cfg = write_patched("configs/mnist_ngm3_gpu_full.json", "branch_jitter_gpu.json",
                                   two_equal_banks_no_learning);
    hdf5::executor h5;
    coast::cli::options opts;
    opts.config_path = cfg.string();
    auto sim = coast::app::build_env(opts, h5);
    REQUIRE(sim);
    auto& g = the_group<ngm3::neuron_group_gpu_t>(*sim);

    coast::test::run_for(*sim, k_interval - 1);   // input_cnt 49: no branch yet
    const auto s0 = state_of(g);
    coast::test::run_for(*sim, 1);                // input_cnt 50: first branch step
    const auto s1 = state_of(g);

    const auto a0 = bank(s0, "apical"),   a1 = bank(s1, "apical");
    const auto p0 = bank(s0, "proximal"), p1 = bank(s1, "proximal");
    REQUIRE(a0.perm.size() == p0.perm.size());

    std::size_t shared = 0, identical = 0;
    for (std::size_t x = 0; x < a0.perm.size(); ++x) {
        const bool a_split = a1.seg[x] != a0.seg[x];
        const bool p_split = p1.seg[x] != p0.seg[x];
        // Same cell, same parent and same child in both banks -> same draw slot.
        if (!(a_split && p_split && a0.seg[x] == p0.seg[x] && a1.seg[x] == p1.seg[x])) continue;
        // Skip clamped results: they hide the jitter.
        if (a1.perm[x] <= 0.0f || a1.perm[x] >= 1.0f || p1.perm[x] <= 0.0f || p1.perm[x] >= 1.0f) continue;
        ++shared;
        const float da = a1.perm[x] - a0.perm[x];
        const float dp = p1.perm[x] - p0.perm[x];
        // Same draw added to different parents differs only by rounding (~1e-8);
        // independent draws from U(-0.1, 0.1) differ by ~0.1.
        if (std::fabs(da - dp) < 1e-6f) ++identical;
    }
    MESSAGE("cells split identically in both banks: " << shared << ", with identical jitter: " << identical);
    REQUIRE_MESSAGE(shared >= 20, "precondition: too few cells split identically in both banks");
    CHECK_MESSAGE(identical * 100 < shared,
                  identical << " of " << shared << " cells got the same jitter in both banks");
}
