// tests/integration/seed_composition/test_seed_composition.cpp
//
// Issue #167 through the full sim path (config -> json_loader -> sim_builder ->
// entity): a node's effective seed is config seed + --seed-offset + the node's
// random_seed_offset, modulo 2^32 (core/seed.h), and the value an entity reports
// via effective_seed() — what the run manifest's node_seeds records — is the
// value its RNG was seeded with.
//
// Runs four configs, covering five of the entity kinds whose sim_builder create
// function composes a seed: mnist_input, ngm2, ngm3 and ngm25 groups (CPU),
// retina. Not run: the other CPU datasets and the GPU entities. Every node
// gets its own random_seed_offset (node i: -(i + 1)), so a create function that
// drops the per-node offset, or applies another node's, reports a wrong seed.
//
// The second half is checked by behaviour, not by reading a field: a run whose
// seeds are composed from offsets must produce exactly the io_buffer state of a
// run configured directly with the reported seeds. A run without offsets must differ
// in every node's buffer (negative control: the comparison is not vacuous).
//
// Issue #190: a group's random_index inhibitor seeds its matrix from ri_seed,
// which composes with the same offsets. The ngm cases run a second time with
// every group switched to a random_index inhibition of its own; there the
// direct run must set ri_seed to the reported inhibition_seed() as well, and a
// direct run that sets only the group seeds must differ (control: the matrix
// seed reaches the output within k_steps, so the check is not vacuous).
//
// Self-comparing, no goldens.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "integration_helpers.h"

#include "app/build_env.h"
#include "cli/options.h"
#include "core/seed.h"
#include "hdf5/executor.h"
#include "sim_core/sim_env.h"
#include "sim_core/sim_instance.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using njson = nlohmann::json;

constexpr std::uint64_t k_steps = 50;
constexpr seed_offset_t k_global_offset{ 4294967294u };   // --seed-offset -2

njson read_config(const char* config)
{
    njson j;
    std::ifstream in(config);
    REQUIRE(in.good());
    in >> j;
    return j;
}

// Where the seed of a node's config entry lives, by entity type.
njson::json_pointer seed_pointer(const njson& node)
{
    const auto type = node.at("entity_type").get<std::string>();
    const auto cfg  = node.at("entity_cfg").get<std::string>();
    if (type == "mnist_input")        return njson::json_pointer("/datasets/mnist/" + cfg + "/random_seed");
    if (type == "ngm2_neuron_group")  return njson::json_pointer("/ngm2/neuron_groups/" + cfg + "/random_seed");
    if (type == "ngm3_neuron_group")  return njson::json_pointer("/ngm3/neuron_groups/" + cfg + "/random_seed");
    if (type == "ngm25_neuron_group") return njson::json_pointer("/ngm25/neuron_groups/" + cfg + "/random_seed");
    if (type == "retina")             return njson::json_pointer("/retina/" + cfg + "/random_seed");
    FAIL("no seed pointer for entity type " << type);
    return njson::json_pointer("");
}

struct node_seed {
    std::uint64_t       id;        // output id
    njson::json_pointer ptr;       // its config seed
    seed_offset_t       offset;    // its random_seed_offset in the by_offsets run
    seed_t              expected;  // config seed + global offset + offset
    // Its random_index inhibition's ri_seed and the composed value, if it has one.
    std::optional<njson::json_pointer> ri_ptr;
    std::optional<seed_t>              ri_expected;
};

// The model section ("ngm2", ...) of an ngm group node; empty for other kinds.
std::string ngm_section(const njson& node)
{
    const auto type = node.at("entity_type").get<std::string>();
    for (const char* m : { "ngm2", "ngm3", "ngm25" })
        if (type == std::string(m) + "_neuron_group") return m;
    return {};
}

// Where the ri_seed of a node's random_index inhibition lives, if it has one.
std::optional<njson::json_pointer> ri_seed_pointer(const njson& j, const njson& node)
{
    const auto m = ngm_section(node);
    if (m.empty()) return std::nullopt;
    const njson::json_pointer cfg_ptr("/" + m + "/neuron_groups/" + node.at("entity_cfg").get<std::string>() + "/inhibition_cfg");
    if (!j.contains(cfg_ptr)) return std::nullopt;
    const auto name = j.at(cfg_ptr).get<std::string>();
    if (name.empty()) return std::nullopt;
    const njson::json_pointer entry("/" + m + "/inhibitions/" + name);
    if (j.at(entry).value("type", std::string{}) != "random_index") return std::nullopt;
    return entry / "ri_seed";
}

// `base` with every ngm group given a random_index inhibition of its own (node
// i: ri_seed i), so each has a distinct ri_seed to configure directly. Small
// seeds, so the composed ones wrap modulo 2^32 like the group seeds.
// reach/strength as in configs/mnist_ridx_inhib.json: a stronger inhibition
// (e.g. strength 0.2 at reach 0.3) clamps every output of a 50-neuron group
// to 0 regardless of the matrix, which would make the control below vacuous.
njson with_random_index(njson j)
{
    auto& nodes = j.at("sim").at("nodes");
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto m = ngm_section(nodes[i]);
        if (m.empty()) continue;
        const auto name = "ridx_node" + std::to_string(i);
        j[m]["inhibitions"][name] = { { "type", "random_index" }, { "ri_reach", 0.301 },
                                      { "ri_strength", 0.035 }, { "ri_seed", i } };
        j[m]["neuron_groups"][nodes[i].at("entity_cfg").get<std::string>()]["inhibition_cfg"] = name;
    }
    return j;
}

std::vector<node_seed> plan(const njson& j)
{
    std::vector<node_seed> r;
    const auto& nodes = j.at("sim").at("nodes");
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto ptr    = seed_pointer(nodes[i]);
        const seed_t cfg  = j.contains(ptr) ? j.at(ptr).get<seed_t>() : seed_t{};
        const seed_offset_t off{ static_cast<std::uint32_t>(-static_cast<std::int64_t>(i + 1)) };
        node_seed ns{ nodes[i].at("id").get<std::uint64_t>(), ptr, off, cfg + k_global_offset + off, {}, {} };
        if (const auto ri = ri_seed_pointer(j, nodes[i])) {
            ns.ri_ptr      = *ri;
            ns.ri_expected = j.at(*ri).get<seed_t>() + k_global_offset + off;
        }
        r.push_back(std::move(ns));
    }
    // Two nodes sharing a config entry could not be given distinct direct seeds.
    for (std::size_t a = 0; a < r.size(); ++a)
        for (std::size_t b = a + 1; b < r.size(); ++b) {
            REQUIRE(r[a].ptr != r[b].ptr);
            REQUIRE((!r[a].ri_ptr || r[a].ri_ptr != r[b].ri_ptr));
        }
    return r;
}

fs::path write_patched(const njson& base, const std::string& name, const std::function<void(njson&)>& patch)
{
    njson j = base;
    for (auto& n : j["sim"]["nodes"]) n.erase("random_seed_offset");
    patch(j);

    const fs::path dir = fs::temp_directory_path() / "coast_seed_composition";
    fs::create_directories(dir);
    const fs::path path = dir / name;
    std::ofstream out(path);
    REQUIRE(out.good());
    out << j.dump(2);
    return path;
}

struct run_result {
    std::map<std::uint64_t, seed_t>             seeds;     // output id -> effective_seed()
    std::map<std::uint64_t, seed_t>             ri_seeds;  // output id -> inhibition_seed()
    std::map<std::size_t, std::vector<float>>   buffers;   // io_buffer id -> read page
};

run_result run(const fs::path& cfg, seed_offset_t seed_offset)
{
    coast::cli::options opts;
    opts.config_path = cfg.string();
    opts.seed_offset = seed_offset;

    hdf5::executor h5;
    auto sim = coast::app::build_env(opts, h5);
    REQUIRE(sim);

    run_result r;
    for (auto& e : sim->get_sim_env().iterate_entities()) {
        const auto id = static_cast<std::uint64_t>(e.get_outp_id());
        if (const auto s = e.effective_seed())  r.seeds[id]    = *s;
        if (const auto s = e.inhibition_seed()) r.ri_seeds[id] = *s;
    }

    coast::test::run_for(*sim, k_steps);
    for (const auto& [id, buf] : sim->get_sim_env().get_io_buffers()) {
        const auto v = buf->cur_read_buffer();
        r.buffers[id].assign(v.begin(), v.end());
    }
    return r;
}

// Per buffer, so a failure names the buffer and its first differing element.
void check_same_buffers(const run_result& a, const run_result& b)
{
    REQUIRE(a.buffers.size() == b.buffers.size());
    for (const auto& [id, va] : a.buffers) {
        CAPTURE(id);
        REQUIRE(b.buffers.contains(id));
        const auto& vb = b.buffers.at(id);
        REQUIRE(va.size() == vb.size());
        const auto [ia, ib] = std::mismatch(va.begin(), va.end(), vb.begin());
        if (ia != va.end())
            FAIL_CHECK("first difference at element " << (ia - va.begin()) << ": " << *ia << " vs " << *ib);
    }
}

void check_composition(const njson& base, const char* tag)
{
    const auto nodes = plan(base);
    const std::string t = tag;

    // Seeds from offsets: --seed-offset -2 for every node, -(i + 1) more on node i.
    const auto by_offsets = run(write_patched(base, t + "_by_offsets.json", [&](njson& j) {
        auto& jn = j["sim"]["nodes"];
        for (std::size_t i = 0; i < nodes.size(); ++i)
            jn[i]["random_seed_offset"] = std::to_underlying(nodes[i].offset);
    }), k_global_offset);
    REQUIRE(by_offsets.seeds.size() == nodes.size());
    std::size_t ri_count = 0;
    for (const auto& n : nodes) {
        CAPTURE(n.id);
        CHECK(by_offsets.seeds.at(n.id) == n.expected);
        if (n.ri_expected) {
            ++ri_count;
            REQUIRE(by_offsets.ri_seeds.contains(n.id));
            CHECK(by_offsets.ri_seeds.at(n.id) == *n.ri_expected);
        }
    }
    CHECK(by_offsets.ri_seeds.size() == ri_count);

    // The reported seeds, configured directly.
    const auto direct = run(write_patched(base, t + "_direct.json", [&](njson& j) {
        for (const auto& n : nodes) {
            j[n.ptr] = std::to_underlying(n.expected);
            if (n.ri_ptr) j[*n.ri_ptr] = std::to_underlying(*n.ri_expected);
        }
    }), seed_offset_t{ 0 });
    CHECK(direct.seeds == by_offsets.seeds);
    CHECK(direct.ri_seeds == by_offsets.ri_seeds);
    check_same_buffers(direct, by_offsets);

    // Control for the inhibition seeds: the group seeds alone, ri_seed left at
    // its config value, must change every random_index group's output.
    if (ri_count > 0) {
        const auto groups_only = run(write_patched(base, t + "_groups_only.json", [&](njson& j) {
            for (const auto& n : nodes) j[n.ptr] = std::to_underlying(n.expected);
        }), seed_offset_t{ 0 });
        for (const auto& n : nodes) {
            if (!n.ri_ptr) continue;
            CAPTURE(n.id);
            CHECK(groups_only.ri_seeds.at(n.id) != by_offsets.ri_seeds.at(n.id));
            const auto& out = by_offsets.buffers.at(n.id);
            REQUIRE(std::any_of(out.begin(), out.end(), [](float v) { return v != 0.0f; }));
            CHECK(groups_only.buffers.at(n.id) != out);
        }
    }

    // Negative control: no offsets, so every node runs different streams.
    const auto plain = run(write_patched(base, t + "_plain.json", [](njson&) {}), seed_offset_t{ 0 });
    for (const auto& n : nodes) {
        CAPTURE(n.id);
        CHECK(plain.seeds.at(n.id) != by_offsets.seeds.at(n.id));
        CHECK(plain.buffers.at(n.id) != by_offsets.buffers.at(n.id));
    }
}

} // namespace

TEST_CASE("seed composition: mnist_input + ngm2 group (configs/mnist.json)")
{
    check_composition(read_config("configs/mnist.json"), "ngm2");
}

TEST_CASE("seed composition: mnist_input + ngm3 group (configs/mnist_ngm3.json)")
{
    check_composition(read_config("configs/mnist_ngm3.json"), "ngm3");
}

TEST_CASE("seed composition: mnist_input + ngm25 group (configs/mnist_ngm25.json)")
{
    check_composition(read_config("configs/mnist_ngm25.json"), "ngm25");
}

TEST_CASE("seed composition: two retinas + ngm2 group (configs/mnist_retina.json)")
{
    check_composition(read_config("configs/mnist_retina.json"), "retina");
}

TEST_CASE("seed composition: ngm2 group with random_index inhibition")
{
    check_composition(with_random_index(read_config("configs/mnist.json")), "ngm2_ridx");
}

TEST_CASE("seed composition: ngm3 group with random_index inhibition")
{
    check_composition(with_random_index(read_config("configs/mnist_ngm3.json")), "ngm3_ridx");
}

TEST_CASE("seed composition: ngm25 group with random_index inhibition")
{
    check_composition(with_random_index(read_config("configs/mnist_ngm25.json")), "ngm25_ridx");
}
