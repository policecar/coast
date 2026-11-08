// tests/integration_ngm3/gpu_layout_check_test.cpp
//
// The ngm3 GPU group's load_state runs the same layout check as the CPU group
// (hd_ngm3_layout.h): the split kernel rewrites a whole run from its first row,
// so a misaligned run is a cross-workgroup write, and replica rows that differ
// make the bank something no learning step produces. A checkpoint that breaks
// the layout must throw before anything is uploaded, leaving the group as it
// was; a checkpoint the group wrote must load.
//
// The checkpoint comes from a real GPU run with a short branch interval, so the
// edits land in split columns and the alignment and full-run laws are exercised
// through the GPU load path, not only coherence. (The exhaustive per-column
// coverage of the laws is in tests/unit/hd_ngm3/test_layout_check.cpp.)
//
// Needs a real GPU adapter (registered with the gpu label).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "frozen_inert_helpers.h"
#include "hd_ngm3/hd_ngm3_gpu_neuron_group.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using frozen_inert::njson;
using frozen_inert::state_of;
using frozen_inert::write_patched;

constexpr std::uint64_t k_steps = 200;   // four branch events at interval 50

void branch_often(njson& j)
{
    auto& g = j["ngm3"]["neuron_groups"]["L4-mnist-ngm3"];
    g["branch_interval"] = 50;
    g["accumulated_theta_thres"] = 0.05;
}

using group_t = ngm3::neuron_group_gpu_t;

group_t& the_group(coast::sim_instance& sim)
{
    auto vec = sim.get_sim_env().get_entities<group_t>();
    REQUIRE(vec.has_value());
    REQUIRE((*vec)->size() == 1);
    return (**vec)[0];
}

struct shape_t { std::size_t neurons, dendrites, rows, inputs; };

shape_t shape_of(const logging::mem_group& state)
{
    auto u = [&](const char* f) { return static_cast<std::size_t>(
        std::get<std::uint64_t>(state.fields.at(f))); };
    return { u("neuron_cnt"), u("proximal_cnt"),
             std::size_t{1} << u("max_branch_level"), u("proximal_input_size") };
}

template <typename T>
std::vector<T>& field(logging::mem_group& state, const char* name)
{
    return std::get<std::vector<T>>(state.groups.at("proximal").fields.at(name));
}

// load_state must throw, name the law, and leave the group as it was.
void check_refused(group_t& g, const logging::mem_group& tree, const char* law)
{
    const auto before = state_of(g);
    logging::mem_state_reader r(tree, logging::k_format_version);
    std::string what;
    try { g.load_state(r); } catch (const std::runtime_error& e) { what = e.what(); }
    INFO("message: ", what);
    CHECK(what.find("corrupt checkpoint") != std::string::npos);
    CHECK(what.find(law) != std::string::npos);
    CHECK(state_of(g) == before);
}

} // namespace

TEST_CASE("ngm3 gpu load_state checks the replica layout")
{
    const auto cfg = write_patched("configs/mnist_ngm3_gpu_full.json",
                                   "ngm3_gpu_layout_check.json", branch_often);
    coast::cli::options opts;
    opts.config_path = cfg.string();
    hdf5::executor h5;
    auto sim = coast::app::build_env(opts, h5);
    REQUIRE(sim);
    REQUIRE(sim->get_sim_env().has_gpu());

    coast::test::run_for(*sim, k_steps);
    auto& g = the_group(*sim);
    const auto tree = state_of(g);
    const auto sh   = shape_of(tree);
    // Step on, so the group no longer holds the checkpoint's values: a refused
    // load that applied any of them (counters, seed, banks) shows as a change.
    coast::test::run_for(*sim, 5);

    // A column split once at the root, both children unsplit: rows [0, R/2)
    // hold 2 and rows [R/2, R) hold 3. The edits below need it.
    auto copy = tree;
    const auto& seg = field<std::uint8_t>(copy, "segment_ids");
    auto at = [&](std::size_t col, std::size_t r) {
        const std::size_t nd = col / sh.inputs, k = col % sh.inputs;
        return (nd * sh.rows + r) * sh.inputs + k;
    };
    std::size_t col = 0;
    bool found = false;
    for (std::size_t c = 0; c < sh.neurons * sh.dendrites * sh.inputs && !found; ++c)
        if (seg[at(c, 0)] == 2 && seg[at(c, sh.rows - 1)] == 3) { col = c; found = true; }
    REQUIRE_MESSAGE(found, "the run must branch, or the edits miss split columns");

    SUBCASE("a checkpoint the group wrote loads") {
        logging::mem_state_reader r(tree, logging::k_format_version);
        CHECK_NOTHROW(g.load_state(r));
        CHECK(state_of(g) == tree);
    }
    SUBCASE("a replica row that differs from its run is refused") {
        auto edited = tree;
        auto& v = field<float>(edited, "permanences")[at(col, 1)];   // row 1 of segment 2's run
        v = std::nextafter(v, 2.0f);
        check_refused(g, edited, "(L5)");
    }
    SUBCASE("a misaligned run is refused") {
        // Swap the two children: every id in range, every run full, both
        // misplaced (the shape that wrote into the next dendrite).
        auto edited = tree;
        auto& s = field<std::uint8_t>(edited, "segment_ids");
        for (std::size_t r = 0; r < sh.rows; ++r) s[at(col, r)] ^= 1u;   // 2 <-> 3
        check_refused(g, edited, "(L3)");
    }
    SUBCASE("a run that does not fill its rows is refused") {
        auto edited = tree;
        field<std::uint8_t>(edited, "segment_ids")[at(col, 0)] = 1;   // root over one row of segment 2
        check_refused(g, edited, "(L4)");
    }
}
