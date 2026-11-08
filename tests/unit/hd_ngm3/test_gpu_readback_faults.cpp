// tests/unit/hd_ngm3/test_gpu_readback_faults.cpp
//
// #234: a failed GPU readback of learned state must not pass as zeros. Every
// readback save_state makes is of required state (the counter, each bank's
// permanences / segment_ids / adapt_history / mismatch, the running maxima, the
// window edges), so a failure of any one of them makes save_state throw — and a
// checkpoint whose extraction throws is never published: the previous good file
// stays as it was. The vis mirror is diagnostic, but a failed refresh keeps the
// last good mirror rather than showing zeros.
//
// The device is an in-memory gpu_device (tests/support/fault_gpu_device.h), so
// this needs no adapter and runs in the sandbox and in a null-backend build.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "fault_gpu_device.h"
#include "hd_ngm3/hd_ngm3_gpu_neuron_group.h"
#include "hdf5/executor.h"
#include "logging/mem_state_io.h"
#include "logging/state_logger.h"
#include "sim_core/io_buffer_gpu.h"
#include "sim_core/sim_env.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
namespace g  = coast::gpu;

namespace {

constexpr std::size_t k_prox = 4;   // proximal input width
constexpr std::size_t k_apic = 3;   // apical input width
constexpr std::size_t k_neurons = 2;

ngm3::cfg_store make_cfg()
{
    ngm3::cfg_store cs;
    auto& p        = cs.neuron_groups["g"];
    p.neuron_cnt   = k_neurons;
    p.proximal_cnt = 1;
    p.apical_cnt   = 1;              // both banks, so both bank saves are covered
    p.inputs       = { { "in", ngm3::bank_t::proximal }, { "ctx", ngm3::bank_t::apical } };
    return cs;
}

// One ngm3 GPU group bound to a fault-injection device, plan built, not stepped.
struct rig {
    coast::test::fault_gpu_device dev;
    ngm3::cfg_store               cs = make_cfg();
    ngm3::neuron_group_gpu_t      grp{ cs, "g", 1, seed_offset_t{} };
    sim::io_buffer_cpu_to_gpu     prox{ k_prox };
    sim::io_buffer_cpu_to_gpu     apic{ k_apic };
    sim::io_buffer_gpu            out{ k_neurons };

    rig()
    {
        grp.init_io({ { "in", { 0, k_prox } }, { "ctx", { 2, k_apic } } });
        auto reg = [&](sim::io_buffer& b, std::size_t n, g::topology t) {
            b.gpu_register(dev, dev.register_buffer(t, n * sizeof(float), true), t);
        };
        reg(prox, k_prox,    g::topology::cpu_to_gpu);
        reg(apic, k_apic,    g::topology::cpu_to_gpu);
        reg(out,  k_neurons, g::topology::gpu_to_gpu);
        grp.set_input_buffer(0, prox);
        grp.set_input_buffer(2, apic);
        grp.set_output_buffer(out);
        grp.build_step_plan(dev);
    }

    // Arm the device to fail readback number `k` from now on (1-based).
    void fail_next(std::size_t k) { dev.reads = 0; dev.fail_read = k; }
    void disarm() { dev.fail_read = 0; dev.fail_if = nullptr; }
};

std::size_t count_nonzero(const std::vector<float>& v)
{
    return static_cast<std::size_t>(std::count_if(v.begin(), v.end(), [](float x) { return x != 0.0f; }));
}

// An io_entity that saves the rig's GPU group, so state_logger checkpoints it
// through the real extract/serialize path.
struct holder_entity : sim::io_entity {
    rig* r;
    explicit holder_entity(rig* rr) : r(rr) {}
    sim::step_result process() override { return {}; }
    void save_state(logging::state_writer& w) const override { r->grp.save_state(w); }
    void load_state(logging::state_reader&) override {}
    [[nodiscard]] sim::output_id_t get_outp_id() const override { return 1; }
    [[nodiscard]] std::size_t get_outp_size() const override { return 0; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }
};

std::string slurp(const fs::path& p)
{
    std::ifstream f(p, std::ios::binary);
    return { std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>() };
}

} // namespace

TEST_CASE("a good save reads learned, nonzero state")
{
    rig r;
    logging::mem_state_writer w;
    r.grp.save_state(w);
    const auto tree = w.take_root();
    const auto& perm = std::get<std::vector<float>>(tree.groups.at("proximal").fields.at("permanences"));
    // The fixture is only meaningful if the state a failure would zero is not zero.
    CHECK(count_nonzero(perm) > 0);
    CHECK(r.dev.reads >= 11);   // counter + 2 banks x 4 + running max + window edges
}

TEST_CASE("every readback save_state makes is required: failing any one of them throws")
{
    rig r;
    {
        logging::mem_state_writer w;
        r.fail_next(0);
        r.grp.save_state(w);
    }
    const std::size_t n_reads = r.dev.reads;
    REQUIRE(n_reads >= 11);

    for (std::size_t k = 1; k <= n_reads; ++k) {
        CAPTURE(k);
        r.fail_next(k);
        logging::mem_state_writer w;
        CHECK_THROWS_AS(r.grp.save_state(w), std::runtime_error);
    }

    // The failures left the group able to save once the device reads again.
    r.disarm();
    logging::mem_state_writer w;
    CHECK_NOTHROW(r.grp.save_state(w));
}

TEST_CASE("a checkpoint whose GPU readback fails does not replace the previous good file")
{
    rig r;
    sim::env env;
    env.emplace_back<holder_entity>(&r);
    hdf5::executor h5;

    const fs::path dir = fs::temp_directory_path() / ("coast_234_ngm3_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    const fs::path path = dir / "state.h5";
    logging::state_save_config cfg;
    cfg.output_path = path;

    REQUIRE(logging::state_logger{}.save_state(env, 7, cfg, h5).success);
    const std::string good = slurp(path);
    REQUIRE_FALSE(good.empty());

    // Fail each bank read in turn (read 1 is the counter, which already threw
    // before #234; reads 2.. are the bank arrays and the running state).
    for (std::size_t k = 2; k <= 11; ++k) {
        CAPTURE(k);
        r.fail_next(k);
        bool reported = false;
        try {
            reported = !logging::state_logger{}.save_state(env, 8, cfg, h5).success;
        } catch (const std::runtime_error&) {
            reported = true;
        }
        CHECK(reported);
        CHECK(slurp(path) == good);
    }
    // Nothing but the good checkpoint is left in the directory (no stray .tmp).
    CHECK(std::distance(fs::directory_iterator(dir), fs::directory_iterator{}) == 1);
    fs::remove_all(dir);
}

TEST_CASE("a failed vis-mirror refresh keeps the last good mirror instead of zeros")
{
    rig r;
    r.grp.refresh_vis_mirror();
    const auto good_span = r.grp.get_proximal_representation(0, 0, 0);
    const std::vector<float> good(good_span.begin(), good_span.end());
    REQUIRE(count_nonzero(good) > 0);

    r.dev.fail_if = [](g::buffer_id, std::size_t) { return true; };
    CHECK_NOTHROW(r.grp.refresh_vis_mirror());   // a GUI-thread refresh must not throw
    const auto now = r.grp.get_proximal_representation(0, 0, 0);
    CHECK(std::vector<float>(now.begin(), now.end()) == good);
}
