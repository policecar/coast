// tests/unit/hd_ngm25/test_gpu_readback_faults.cpp
//
// #234, ngm25 side: the GPU group pulls perm / adapt_hist / mismatch / the
// running maxima from the device into its host arena before a save, before a
// growth (branch) decision and for the vis mirror. A failed readback must not pass
// as zeros:
//   - save_state throws when any of its readbacks fails;
//   - a failed growth readback throws without touching live state: the arena is
//     not repacked, nothing is uploaded, the branch counters are not advanced, so
//     the next sync retries the same branch event;
//   - a failed vis refresh keeps the last good mirror.
//
// The device is an in-memory gpu_device (tests/support/fault_gpu_device.h): no
// adapter, runs in the sandbox and in a null-backend build. Compute is a no-op,
// so the test writes the device-updated arrays itself where it needs learned
// state (a crossing count, a mismatch pattern that makes dendrites split).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "fault_gpu_device.h"
#include "hd_ngm25/hd_ngm25_gpu_neuron_group.h"
#include "logging/mem_state_io.h"
#include "sim_core/io_buffer.h"
#include "sim_core/io_buffer_gpu.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace g = coast::gpu;

namespace {

constexpr std::size_t k_width   = 8;
constexpr std::size_t k_neurons = 3;

ngm25::cfg_store make_cfg()
{
    ngm25::cfg_store cs;
    ngm25::neuron_group_params_t p;
    p.neuron_cnt       = k_neurons;
    p.apical_cnt       = 0;
    p.proximal_cnt     = 2;
    p.max_branch_level = 3;
    p.branch_interval  = 2;          // a crossing is due on the second sync
    p.inputs           = { { "in", ngm25::bank_t::proximal } };
    cs.neuron_groups["grp"] = p;
    return cs;
}

// One ngm25 GPU group bound to a fault-injection device, plan built, not stepped.
struct rig {
    coast::test::fault_gpu_device dev;
    ngm25::cfg_store              cs = make_cfg();
    ngm25::neuron_group_gpu_t     grp{ cs, "grp", 1 };
    sim::io_buffer_cpu            in{ k_width };
    sim::io_buffer_gpu            out{ k_neurons };
    std::size_t                   base = 0;   // first buffer the group registered

    rig()
    {
        grp.init_io({ { "in", { 0, k_width } } });
        out.gpu_register(dev, dev.register_buffer(g::topology::gpu_to_gpu, k_neurons * sizeof(float), true),
                         g::topology::gpu_to_gpu);
        grp.set_input_buffer(0, in);
        grp.set_output_buffer(out);
        base = dev.buffers.size();
        grp.build_step_plan(dev);
    }

    // The group's buffers in build_step_plan's registration order (checked by size
    // in the tests, so a reordering there fails loudly instead of silently).
    [[nodiscard]] g::buffer_id id(std::size_t k) const { return static_cast<g::buffer_id>(base + k); }
    [[nodiscard]] g::buffer_id perm()       const { return id(0); }
    [[nodiscard]] g::buffer_id adapt_hist() const { return id(3); }
    [[nodiscard]] g::buffer_id mismatch()   const { return id(4); }
    [[nodiscard]] g::buffer_id learn_cnt()  const { return id(17); }

    void fail_next(std::size_t k) { dev.reads = 0; dev.fail_read = k; }
    void disarm() { dev.fail_read = 0; dev.fail_if = nullptr; }
};

} // namespace

TEST_CASE("every readback save_state makes is required: failing any one of them throws")
{
    rig r;
    {
        logging::mem_state_writer w;
        r.fail_next(0);
        r.grp.save_state(w);
    }
    const std::size_t n_reads = r.dev.reads;
    REQUIRE(n_reads >= 6);   // perm, adapt_hist, mismatch, running max, counter, window edges

    for (std::size_t k = 1; k <= n_reads; ++k) {
        CAPTURE(k);
        r.fail_next(k);
        logging::mem_state_writer w;
        CHECK_THROWS_AS(r.grp.save_state(w), std::runtime_error);
    }

    r.disarm();
    logging::mem_state_writer w;
    CHECK_NOTHROW(r.grp.save_state(w));
}

TEST_CASE("a failed growth readback throws and leaves live state for the retry")
{
    rig r;
    const std::size_t n = r.grp.get_synapse_count();
    REQUIRE(r.dev.at(r.perm()).size()       == n * sizeof(float));
    REQUIRE(r.dev.at(r.adapt_hist()).size() == n * sizeof(float));
    REQUIRE(r.dev.at(r.mismatch()).size()   == n * sizeof(float));
    REQUIRE(r.dev.at(r.learn_cnt()).size()  == sizeof(std::uint32_t));

    // Device-side "learning": every neuron crossed the branch interval, and every
    // dendrite has qualifying synapses (half at mismatch 1, half at 0: the high
    // ones clear avg + var + 1/width; adapt_hist clears accumulated_theta_thres).
    r.dev.as<std::uint32_t>(r.learn_cnt())[0] = 2;
    auto mm = r.dev.as<float>(r.mismatch());
    auto ah = r.dev.as<float>(r.adapt_hist());
    for (std::size_t s = 0; s < n; ++s) { mm[s] = (s % 2 == 0) ? 1.0f : 0.0f; ah[s] = 1.0e6f; }
    const std::vector<float> perm_before(r.dev.as<float>(r.perm()).begin(), r.dev.as<float>(r.perm()).end());

    // The perm readback of the growth sync fails.
    const g::buffer_id perm_id = r.perm();
    r.dev.fail_if = [perm_id](g::buffer_id id, std::size_t) { return id == perm_id; };
    const std::size_t uploads_before = r.dev.uploads;
    bool threw = false;
    for (int i = 0; i < 4 && !threw; ++i) {
        try { r.grp.read_probes(); }
        catch (const std::runtime_error&) { threw = true; }
    }
    CHECK(threw);
    CHECK(r.grp.get_synapse_count() == n);       // no repack
    CHECK(r.grp.get_branch_epoch()  == 0);
    CHECK(r.dev.uploads == uploads_before);      // nothing pushed to the device
    const auto perm_now = r.dev.as<float>(r.perm());
    CHECK(std::vector<float>(perm_now.begin(), perm_now.end()) == perm_before);

    // The counters were not advanced: once the device reads again, the same sync
    // finds the crossing and branches.
    r.disarm();
    CHECK_NOTHROW(r.grp.read_probes());
    CHECK(r.grp.get_branch_epoch()  == 1);
    CHECK(r.grp.get_synapse_count() > n);
}

TEST_CASE("a failed vis-mirror refresh keeps the last good mirror instead of zeros")
{
    rig r;
    r.grp.refresh_vis_mirror();
    const auto good_span = r.grp.get_proximal_representation(0, 0, 0);
    const std::vector<float> good(good_span.begin(), good_span.end());
    REQUIRE(std::any_of(good.begin(), good.end(), [](float x) { return x != 0.0f; }));

    r.dev.fail_if = [](g::buffer_id, std::size_t) { return true; };
    CHECK_NOTHROW(r.grp.refresh_vis_mirror());   // a GUI-thread refresh must not throw
    const auto now = r.grp.get_proximal_representation(0, 0, 0);
    CHECK(std::vector<float>(now.begin(), now.end()) == good);
}
