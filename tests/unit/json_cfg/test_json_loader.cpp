#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <utility>
#include <vector>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

// Every section's JSON binding, ahead of json_loader.h (as json_loader.cpp
// orders them) so global_cfg_t can be serialised here.
#include "json_cfg/ngm2_json.h"
#include "json_cfg/ngm3_json.h"
#include "json_cfg/ngm25_json.h"
#include "json_cfg/sim_json.h"
#include "json_cfg/dataset_json.h"
#include "json_cfg/retina_json.h"
#include "json_cfg/vis_hints_json.h"
#include "json_cfg/json_loader.h"
#include "inhibition/inhibition_cfg.h"
#include "json_cfg/inhibition_json.h"
#include "diag/diag.h"

namespace {
constexpr const char* fixtures_dir() { return COAST_JSON_FIXTURES_DIR; }
std::string fixture(const char* name) {
    return std::string(fixtures_dir()) + "/" + name;
}
} // namespace

TEST_CASE("json_loader: empty JSON {} loads, all fields take their defaults") {
    loader::json_loader L(fixture("empty.json"));
    const auto& cfg = L.get_config();
    // Empty leaf = "defer to the app default": app_config's paths.snapshot_dir
    // ("snapshots") applies, resolved under config_base by the app layer. (Was
    // "snapshots" here before the app-level location config moved that default
    // into app_config.)
    CHECK(cfg.snapshot_dir.empty());
    // datasets / ngm2 / sim / vis_hints / observe are all default-constructed;
    // we don't peek into them here (they're sub-config types with their own
    // covered surfaces).
}

TEST_CASE("json_loader: snapshot_dir overrides the default") {
    loader::json_loader L(fixture("snapshot_dir_only.json"));
    CHECK(L.get_config().snapshot_dir == "custom/path/snaps");
}

TEST_CASE("json_loader: missing file throws std::runtime_error") {
    CHECK_THROWS_AS(
        loader::json_loader(fixture("does_not_exist.json")),
        std::runtime_error);
}

TEST_CASE("json_loader: malformed JSON throws (nlohmann parse_error → std::exception)") {
    // nlohmann's json::parse_error derives from std::exception. We only care
    // here that loading fails loudly; the concrete type is library-internal.
    CHECK_THROWS_AS(
        loader::json_loader(fixture("malformed.json")),
        std::exception);
}

TEST_CASE("json_loader: a rejected config value names the file and keeps its exception type") {
    const std::string path = fixture("unknown_entity_type.json");
    try {
        loader::json_loader L(path);
        FAIL("expected std::invalid_argument");
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        CHECK(msg.find("config '" + path + "': ") == 0);
        CHECK(msg.find("\"ngm2_neuron_grop\"") != std::string::npos);
    }
}

TEST_CASE("json_loader: a nlohmann error names the file too") {
    const std::string path = fixture("malformed.json");
    try {
        loader::json_loader L(path);
        FAIL("expected std::runtime_error");
    } catch (const std::runtime_error& e) {
        CHECK(std::string(e.what()).find("config '" + path + "': ") == 0);
    }
}

TEST_CASE("json_loader: real config (configs/mnist.json) loads cleanly") {
    // This pins that the test suite stays in sync with the canonical config.
    // If someone adds a required field without a default, this trips.
    // Path is relative to repo root (tests are launched from CMAKE_SOURCE_DIR).
    CHECK_NOTHROW(loader::json_loader("configs/mnist.json"));
}

TEST_CASE("inhibition_json: random_index params survive a JSON round-trip") {
    // Mirrors a `random_index` inhibition block as written by save_config (the
    // max_softmax `strength` plus the random_index `ri_*` superset are all present).
    // The exact ri_reach / ri_strength values are the float32-rounded literals a
    // config editor save produces, so an Approx match is the right bar.
    const json src = json::parse(R"({
        "type": "random_index",
        "strength": 1.0,
        "ri_variant": "threshold_linear",
        "ri_reach": 0.30099999904632568,
        "ri_strength": 0.035000000149011612,
        "ri_seed": 7,
        "ri_seed_from_time": false
    })");

    auto p = src.get<inhibition::inhibition_params_t>();
    CHECK(p.type        == inhibition::inhibition_type_t::random_index);
    CHECK(p.ri_variant  == inhibition::random_index_variant_t::threshold_linear);
    CHECK(p.ri_reach    == doctest::Approx(0.301));
    CHECK(p.ri_strength == doctest::Approx(0.035));
    CHECK(p.ri_seed     == seed_t{ 7 });
    CHECK(p.ri_seed_from_time == false);

    // save_config is `json j = config; j.dump(4)`. Re-serialising then reloading
    // then re-serialising must reach a fixed point: load → save → load → save is
    // stable. This is exactly the "loads and re-saves identically" criterion.
    const json j1 = p;
    const auto p2  = j1.get<inhibition::inhibition_params_t>();
    const json j2 = p2;
    CHECK(j1 == j2);

    // Enums must serialise as their names, never as raw integers — a config a
    // human reads and edits.
    CHECK(j1.at("type")       == "random_index");
    CHECK(j1.at("ri_variant") == "threshold_linear");
}

// ---- strict config enums (json_cfg/strict_enum.h) ----

TEST_CASE("strict enums: an unknown inhibition type throws instead of becoming the default") {
    try {
        (void)json::parse(R"({"type": "max_sotfmax"})").get<inhibition::inhibition_params_t>();
        FAIL("expected std::invalid_argument");
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        CHECK(msg.find("inhibition_type_t") != std::string::npos);
        CHECK(msg.find("\"max_sotfmax\"") != std::string::npos);
        CHECK(msg.find("\"max_softmax\"") != std::string::npos);   // lists the accepted values
    }
    // The removed soft_wta kind is an unknown value now, not a silent alias.
    CHECK_THROWS_AS(json::parse(R"({"type": "soft_wta"})").get<inhibition::inhibition_params_t>(),
                    std::invalid_argument);
}

TEST_CASE("strict enums: unknown entity type, dendrite type and ngm25 bank throw") {
    CHECK_THROWS_AS(json::parse(R"("ngm2_neuron_grop")").get<sim::io_entity_type_t>(), std::invalid_argument);
    CHECK_THROWS_AS(json::parse(R"("proxymal")").get<ngm2::dendrite_type_t>(), std::invalid_argument);
    CHECK_THROWS_AS(json::parse(R"("proxymal")").get<ngm25::bank_t>(), std::invalid_argument);
    CHECK_THROWS_AS(json::parse(R"(3)").get<inhibition::inhibition_type_t>(), std::invalid_argument);   // wrong JSON type
}

TEST_CASE("strict enums: known values round-trip") {
    CHECK(json::parse(R"("random_index")").get<inhibition::inhibition_type_t>() == inhibition::inhibition_type_t::random_index);
    CHECK(json(sim::io_entity_type_t::mnist_input) == "mnist_input");
}

TEST_CASE("strict enums: JSON null is rejected, not mapped to the TYPE_COUNT sentinel") {
    // A null-mapped sentinel parsed into a non-setting: dendrite TYPE_COUNT indexed
    // past the 2-element per-type activity array, ri_variant TYPE_COUNT left every
    // random-index weight 0.
    CHECK_THROWS_AS(json::parse("null").get<inhibition::inhibition_type_t>(), std::invalid_argument);
    CHECK_THROWS_AS(json::parse("null").get<inhibition::random_index_variant_t>(), std::invalid_argument);
    CHECK_THROWS_AS(json::parse("null").get<ngm2::dendrite_type_t>(), std::invalid_argument);
    CHECK_THROWS_AS(json::parse("null").get<sim::io_entity_type_t>(), std::invalid_argument);
    CHECK_THROWS_AS(json::parse(R"({"type": "random_index", "ri_variant": null})").get<inhibition::inhibition_params_t>(),
                    std::invalid_argument);

    // ...and the sentinels have no JSON name, so serialising one is a loud error.
    CHECK_THROWS_AS((void)json(inhibition::inhibition_type_t::TYPE_COUNT), std::invalid_argument);
    CHECK_THROWS_AS((void)json(inhibition::random_index_variant_t::TYPE_COUNT), std::invalid_argument);
    CHECK_THROWS_AS((void)json(ngm2::dendrite_type_t::TYPE_COUNT), std::invalid_argument);
    CHECK_THROWS_AS((void)json(sim::io_entity_type_t::ENTITY_COUNT), std::invalid_argument);
}

TEST_CASE("strict enums: the accepted-values list no longer offers null") {
    try {
        (void)json::parse(R"("bogus")").get<ngm2::dendrite_type_t>();
        FAIL("expected std::invalid_argument");
    } catch (const std::invalid_argument& e) {
        CHECK(std::string(e.what()).find("null") == std::string::npos);
    }
}

TEST_CASE("strict enums: an enumerator missing from the table throws on to_json") {
    // bank_t's table has no entry for a value outside {apical, proximal}
    CHECK_THROWS_AS((void)json(static_cast<ngm25::bank_t>(42)), std::invalid_argument);
}

TEST_CASE("strict enums: every shipped config in configs/ loads and re-serialises") {
    // Re-serialising mirrors config_editor::save_config (ordered_json j = config),
    // where the strict to_json would throw on any sentinel that slipped through.
    namespace fs = std::filesystem;
    int n = 0;
    for (const auto& entry : fs::directory_iterator(COAST_CONFIGS_DIR)) {
        if (entry.path().extension() != ".json") continue;
        INFO(entry.path().string());
        loader::json_loader L(entry.path().string());
        CHECK_NOTHROW((void)nlohmann::ordered_json(L.get_config()));
        ++n;
    }
    CHECK(n > 10);
}

TEST_CASE("inhibition_params_t: max_softmax round-trips through JSON") {
    const json src = json::parse(R"({"type": "max_softmax", "strength": 10.0})");
    auto p = src.get<inhibition::inhibition_params_t>();
    CHECK(p.type     == inhibition::inhibition_type_t::max_softmax);
    CHECK(p.strength == 10.0);
    const json back = p;
    CHECK(back.at("type") == "max_softmax");
}

// ---------------------------------------------------------------------------
// Seeds (core/seed.h) through the real loader: every config section that holds a
// seed or a seed offset is checked, and a refused value is named by its JSON path
// (json_loader's seed-key walk; the from_json hooks cannot see their key).
// ---------------------------------------------------------------------------

namespace {

// Every place a seed or seed offset can sit in a config, with the field it binds
// to. Kept by hand: a seed field added later is covered once it is added here.
using seed_field = std::function<std::uint32_t(const loader::global_cfg_t&)>;
const std::vector<std::pair<std::string, seed_field>> k_seed_fields = {
    { "/datasets/mnist/x/random_seed",       [](const auto& c) { return std::to_underlying(c.datasets.mnist.at("x").random_seed); } },
    { "/datasets/dsprites/x/random_seed",    [](const auto& c) { return std::to_underlying(c.datasets.dsprites.at("x").random_seed); } },
    { "/datasets/emnist/x/random_seed",      [](const auto& c) { return std::to_underlying(c.datasets.emnist.at("x").random_seed); } },
    { "/datasets/kmnist/x/random_seed",      [](const auto& c) { return std::to_underlying(c.datasets.kmnist.at("x").random_seed); } },
    { "/datasets/cifar10/x/random_seed",     [](const auto& c) { return std::to_underlying(c.datasets.cifar10.at("x").random_seed); } },
    { "/datasets/cifar100/x/random_seed",    [](const auto& c) { return std::to_underlying(c.datasets.cifar100.at("x").random_seed); } },
    { "/datasets/chairs3d/x/random_seed",    [](const auto& c) { return std::to_underlying(c.datasets.chairs3d.at("x").random_seed); } },
    { "/datasets/color_mnist/x/random_seed", [](const auto& c) { return std::to_underlying(c.datasets.color_mnist.at("x").random_seed); } },
    { "/retina/x/random_seed",               [](const auto& c) { return std::to_underlying(c.retina.at("x").random_seed); } },
    { "/ngm2/dendrites/x/rnd_seed",          [](const auto& c) { return std::to_underlying(c.ngm2.dendrites.at("x").rnd_seed); } },
    { "/ngm2/neurons/x/random_seed",         [](const auto& c) { return std::to_underlying(c.ngm2.neurons.at("x").random_seed); } },
    { "/ngm2/neuron_groups/x/random_seed",   [](const auto& c) { return std::to_underlying(c.ngm2.neuron_groups.at("x").random_seed); } },
    { "/ngm2/inhibitions/x/ri_seed",         [](const auto& c) { return std::to_underlying(c.ngm2.inhibitions.at("x").ri_seed); } },
    { "/ngm3/neuron_groups/x/random_seed",   [](const auto& c) { return std::to_underlying(c.ngm3.neuron_groups.at("x").random_seed); } },
    { "/ngm3/inhibitions/x/ri_seed",         [](const auto& c) { return std::to_underlying(c.ngm3.inhibitions.at("x").ri_seed); } },
    { "/ngm25/neuron_groups/x/random_seed",  [](const auto& c) { return std::to_underlying(c.ngm25.neuron_groups.at("x").random_seed); } },
    { "/ngm25/inhibitions/x/ri_seed",        [](const auto& c) { return std::to_underlying(c.ngm25.inhibitions.at("x").ri_seed); } },
    { "/sim/nodes/0/random_seed_offset",     [](const auto& c) { return std::to_underlying(c.sim.nodes.at(0).random_seed_offset); } },
};

// Writes {path: value} as a config file and loads it. "sim" needs its
// node_sets key, so it is present.
loader::json_loader load_with(const std::string& path, const nlohmann::json& value)
{
    nlohmann::json j = { { "sim", { { "nodes", nlohmann::json::array() }, { "node_sets", nlohmann::json::array() } } } };
    j[nlohmann::json::json_pointer(path)] = value;
    const auto file = std::filesystem::temp_directory_path() / "coast_seed_loader.json";
    std::ofstream(file) << j.dump();
    return loader::json_loader(file.string());
}

} // namespace

TEST_CASE("json_loader: a refused seed is named by its JSON path, in every config section") {
    for (const auto& [path, field] : k_seed_fields)
        for (const nlohmann::json& bad : { nlohmann::json(1.5), nlohmann::json(4294967296), nlohmann::json("3"),
                                           nlohmann::json(-1) }) {
            CAPTURE(path);
            CAPTURE(bad.dump());
            try {
                load_with(path, bad);
                FAIL("expected the seed to be refused");
            } catch (const std::invalid_argument& e) {
                CHECK(std::string(e.what()).find(path + ": a random seed") != std::string::npos);
            }
        }
}

// The loaded value, not just "it loads": a field that is renamed or no longer
// bound keeps its default 0 and fails here.
TEST_CASE("json_loader: 4294967295 loads as seed 4294967295, in every config section") {
    for (const auto& [path, field] : k_seed_fields) {
        CAPTURE(path);
        const auto l = load_with(path, 4294967295u);
        CHECK(field(l.get_config()) == 4294967295u);
    }
}

TEST_CASE("json_loader: the seed-key walk checks scalars only, and escapes its paths") {
    // A group named like a seed key is an object, not a seed: it loads.
    const auto named = load_with("/ngm2/neuron_groups/random_seed/random_seed", 3);
    CHECK(named.get_config().ngm2.neuron_groups.at("random_seed").random_seed == seed_t{ 3 });

    // A '/' in a name is escaped (RFC 6901): the path names one level, a~1b.
    try {
        load_with("/ngm2/neuron_groups/a~1b/random_seed", 1.5);
        FAIL("expected the seed to be refused");
    } catch (const std::invalid_argument& e) {
        CHECK(std::string(e.what()).find("/ngm2/neuron_groups/a~1b/random_seed: a random seed") != std::string::npos);
    }

    // A '~' is escaped too (as ~0), so a name containing "~1" stays distinct from
    // one containing '/'.
    try {
        load_with("/ngm2/neuron_groups/a~01b/random_seed", 1.5);
        FAIL("expected the seed to be refused");
    } catch (const std::invalid_argument& e) {
        CHECK(std::string(e.what()).find("/ngm2/neuron_groups/a~01b/random_seed: a random seed") != std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// Issue #199: sigmoid() takes inv_step = 1/(1 - steepness^0.1). steepness == 1
// made every weight_filter output NaN and learning stopped without a word (1.0
// was the config editor slider's own maximum). The shared sigmoid_shape_t
// adapter now refuses a shape outside [0, 1), for all three models.
// ---------------------------------------------------------------------------

namespace {
const char* const k_weight_filter_paths[] = {
    "/ngm2/neuron_groups/x/default_weight_filter",
    "/ngm3/neuron_groups/x/weight_filter",
    "/ngm25/neuron_groups/x/weight_filter",
};
} // namespace

TEST_CASE("json_loader: a weight_filter steepness outside [0, 1) is refused, in every model") {
    for (const char* path : k_weight_filter_paths)
        // 0.9999999 < 1, but pow(float(s), 0.1f) rounds to 1.0f: inv_step = inf (review #1)
        for (const double bad : { 1.0, 1.5, -0.1, 1e300, 0.9999999 }) {
            CAPTURE(path);
            CAPTURE(bad);
            try {
                load_with(std::string(path) + "/steepness", bad);
                FAIL("expected the steepness to be refused");
            } catch (const std::invalid_argument& e) {
                CHECK(std::string(e.what()).find("sigmoid steepness must be in [0, 1)") != std::string::npos);
            }
        }
}

TEST_CASE("json_loader: a weight_filter steepness in [0, 1) loads as given, in every model") {
    for (const double ok : { 0.0, 0.5, 0.999, 0.9999997 }) {   // 0.9999997: largest float with a finite inv_step
        CAPTURE(ok);
        const auto l = load_with(std::string(k_weight_filter_paths[0]) + "/steepness", ok);
        CHECK(l.get_config().ngm2.neuron_groups.at("x").default_weight_filter.steepness == ok);
        CHECK(load_with(std::string(k_weight_filter_paths[1]) + "/steepness", ok)
                  .get_config().ngm3.neuron_groups.at("x").weight_filter.steepness == ok);
        CHECK(load_with(std::string(k_weight_filter_paths[2]) + "/steepness", ok)
                  .get_config().ngm25.neuron_groups.at("x").weight_filter.steepness == ok);
    }
}

TEST_CASE("json_loader: a weight_filter without steepness keeps the default and round-trips") {
    const auto l = load_with(std::string(k_weight_filter_paths[0]) + "/transition_point", 0.25);
    const auto& wf = l.get_config().ngm2.neuron_groups.at("x").default_weight_filter;
    CHECK(wf.steepness == ngm2::sigmoid_shape_t{}.steepness);
    CHECK(wf.transition_point == 0.25);
    const json back = wf;
    CHECK(back == json::parse(R"({"steepness": 0.5, "transition_point": 0.25})"));
}

// ---------------------------------------------------------------------------
// Issue #199: the config as written against the config as bound and serialised
// again. A key that does not come back is unknown (bound to nothing); a number
// that comes back different did not fit its C++ type.
// ---------------------------------------------------------------------------

namespace {

// Collects the warnings of the "config" category while alive.
struct config_warnings {
    std::vector<std::string> msgs;
    config_warnings() {
        coast::diag::set_sink([this](coast::diag::level, std::string_view cat, std::string_view msg) {
            if (cat == "config") msgs.emplace_back(msg);
        });
    }
    ~config_warnings() { coast::diag::set_sink({}); }
    [[nodiscard]] bool any_contains(std::string_view needle) const {
        return std::ranges::any_of(msgs, [&](const std::string& m) { return m.find(needle) != std::string::npos; });
    }
};

// The message of the std::invalid_argument load_with throws, or "" if it loads.
std::string load_error(const std::string& path, const nlohmann::json& value)
{
    try { load_with(path, value); }
    catch (const std::invalid_argument& e) { return e.what(); }
    return "";
}

} // namespace

TEST_CASE("json_loader: a number that does not fit its type is refused, named by its path") {
    const std::string g = "/ngm3/neuron_groups/x";
    // uint8_t: 259 used to load as 3 and pass the [1,4] depth check
    CHECK(load_error(g + "/max_branch_level", 259).find(g + "/max_branch_level: 259 -> 3") != std::string::npos);
    // size_t: wrap, and a fraction truncated
    CHECK(load_error(g + "/neuron_cnt", -1).find(g + "/neuron_cnt: -1 -> 18446744073709551615") != std::string::npos);
    CHECK(load_error(g + "/neuron_cnt", 2.7).find(g + "/neuron_cnt: 2.7 -> 2") != std::string::npos);
    // float: out of range becomes inf
    CHECK(load_error(g + "/stochastic_win_thres", 1e300).find(g + "/stochastic_win_thres: 1e+300") != std::string::npos);
}

TEST_CASE("json_loader: a number that fits its type loads as written") {
    const std::string g = "/ngm3/neuron_groups/x";
    CHECK(load_with(g + "/neuron_cnt", 4.0).get_config().ngm3.neuron_groups.at("x").neuron_cnt == 4);   // whole float
    CHECK(load_with(g + "/max_branch_level", 255).get_config().ngm3.neuron_groups.at("x").max_branch_level == 255);
    // a float field keeps a double input to float precision, which is not a misread
    CHECK(load_with(g + "/stochastic_win_thres", 0.1).get_config().ngm3.neuron_groups.at("x").stochastic_win_thres == 0.1f);
}

TEST_CASE("json_loader: a float beyond a 64-bit integer field is refused, its largest neighbour loads") {
    const std::string n = "/ngm3/neuron_groups/x/neuron_cnt";   // size_t
    // 2^64: ARM saturates it to UINT64_MAX, which converts back to 2^64 exactly
    CHECK(load_error(n, 18446744073709551616.0).find(n + ": 1.8446744073709552e+19 -> ") != std::string::npos);
    CHECK(load_error(n, 1e20).find(n + ": 1e+20 -> ") != std::string::npos);
    CHECK(load_error(n, -1.0).find(n + ": -1.0 -> ") != std::string::npos);
    // 2^63 is representable in size_t
    CHECK(load_with(n, 9223372036854775808.0).get_config().ngm3.neuron_groups.at("x").neuron_cnt
          == 9223372036854775808ULL);
}

namespace {
// A node set whose node_deltas are `deltas`, written in that order.
nlohmann::json node_set_with(nlohmann::json deltas)
{
    return { { "base_id", 10 }, { "base_label", "set" }, { "nodes", { 1, 2, 3 } },
             { "node_deltas", std::move(deltas) } };
}
} // namespace

TEST_CASE("json_loader: node_deltas are compared by key, not by the map's iteration order") {
    // an unordered_map writes these back in its own order ([[3,…],[2,…],[1,…]] on libstdc++)
    const nlohmann::json deltas = { { 1, { { "add", 5, "a" } } }, { 2, { { "remove", 6, "b" } } },
                                    { 3, nlohmann::json::array() } };
    const auto l = load_with("/sim/node_sets/0", node_set_with(deltas));
    const auto& got = l.get_config().sim.node_sets.at(0).node_deltas;
    REQUIRE(got.size() == 3);
    CHECK(std::get<1>(got.at(1).at(0)) == 5);
    CHECK(std::get<2>(got.at(2).at(0)) == "b");
    CHECK(got.at(3).empty());
}

TEST_CASE("json_loader: node_deltas still have their keys and values checked") {
    const std::string d = "/sim/node_sets/0/node_deltas";
    // a value inside a delta that does not fit: found under the entry's own index
    CHECK(load_error("/sim/node_sets/0", node_set_with({ { 1, nlohmann::json::array() },
                                                         { 2, { { "add", -1, "a" } } } }))
              .find(d + "/1/1/0/1: -1 -> 18446744073709551615") != std::string::npos);
    // a key that does not fit
    CHECK(load_error("/sim/node_sets/0", node_set_with({ { 1, nlohmann::json::array() },
                                                         { -1, nlohmann::json::array() } }))
              .find(d + "/1/0: key -1 does not fit its type") != std::string::npos);
    // a repeated key: the map keeps the first, so the second is refused, not dropped
    CHECK(load_error("/sim/node_sets/0", node_set_with({ { 1, nlohmann::json::array() },
                                                         { 1, { { "add", 5, "a" } } } }))
              .find(d + "/1/0: key 1 does not fit its type, or repeats an earlier key") != std::string::npos);
}

TEST_CASE("json_loader: an unknown key is warned about by its path, and still loads") {
    const config_warnings w;
    const auto l = load_with("/ngm3/neuron_groups/x/stochastic_win_thre", 0.5);   // misspelt
    CHECK(l.get_config().ngm3.neuron_groups.at("x").stochastic_win_thres == ngm3::neuron_group_params_t{}.stochastic_win_thres);
    CHECK(w.any_contains("unknown key(s) ignored: /ngm3/neuron_groups/x/stochastic_win_thre"));
}

TEST_CASE("json_loader: an unknown object is reported once, not key by key") {
    const config_warnings w;
    (void)load_with("/ngm4", nlohmann::json{ { "a", 1 }, { "b", { { "c", 2 } } } });
    CHECK(w.any_contains("unknown key(s) ignored: /ngm4"));
    CHECK_FALSE(w.any_contains("/ngm4/"));
}

TEST_CASE("json_loader: a removed key keeps its own message, and is not reported as unknown too") {
    const config_warnings w;
    (void)load_with("/ngm3/neuron_groups/x/activity_learning_window", nlohmann::json::array({ 0.1, 0.9 }));
    CHECK(w.any_contains("/ngm3/neuron_groups/x/activity_learning_window ignored: the activity learning window"));
    CHECK_FALSE(w.any_contains("unknown key"));
}

TEST_CASE("json_loader: a renamed key is refused with its new name, not defaulted") {
    // An unknown key would load with the default depth: another model.
    CHECK(load_error("/ngm3/neuron_groups/x/dendrite_branch_depth", 4)
              .find("/ngm3/neuron_groups/x/dendrite_branch_depth is now max_branch_level") != std::string::npos);
    CHECK(load_error("/ngm25/neuron_groups/x/max_branch_depth", 4)
              .find("/ngm25/neuron_groups/x/max_branch_depth is now max_branch_level") != std::string::npos);
}

TEST_CASE("json_loader: freezing and continuation are the run's, so the old keys are refused") {
    // An unknown key would load and be ignored: a frozen trial that learns (SPEC M.7.6),
    // a sequence that continues where it should not (S.5.14).
    for (const char* m : { "ngm2", "ngm3", "ngm25" })
        CHECK(load_error(std::string("/") + m + "/neuron_groups/x/frozen", true)
                  .find(std::string("/") + m + "/neuron_groups/x/frozen is now the top-level frozen")
              != std::string::npos);
    CHECK(load_error("/datasets/mnist/m/restore_sequence_from_checkpoint", false)
              .find("/datasets/mnist/m/restore_sequence_from_checkpoint is now the top-level run_id")
          != std::string::npos);
    CHECK(load_with("/frozen", true).get_config().frozen);
    const auto defaults = load_with("/snapshot_dir", "s");
    CHECK_FALSE(defaults.get_config().frozen);
    CHECK(defaults.get_config().run_id == "auto");
    CHECK(load_with("/run_id", "trial-7").get_config().run_id == "trial-7");
    CHECK(load_error("/run_id", "").find("run_id is empty") != std::string::npos);
}

TEST_CASE("json_loader: a known config warns about nothing") {
    const config_warnings w;
    (void)loader::json_loader("configs/mnist.json");
    CHECK(w.msgs.empty());
}

TEST_CASE("json_loader: a null section means its defaults, as S.3.19 requires") {
    // _WITH_DEFAULT binds null to the defaults; the round trip must not call that a misread
    const auto ds = load_with("/datasets/mnist/m", nullptr);
    CHECK(ds.get_config().datasets.mnist.at("m").change_interval == mnist::mnist_param_t{}.change_interval);
    // the hand-written sigmoid adapter keeps that behaviour
    const auto wf = load_with(std::string(k_weight_filter_paths[1]), nullptr);
    CHECK(wf.get_config().ngm3.neuron_groups.at("x").weight_filter.steepness == ngm2::sigmoid_shape_t{}.steepness);
}
