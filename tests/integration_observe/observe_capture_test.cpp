// tests/integration_observe/observe_capture_test.cpp
//
// End-to-end observe capture (plan item 3.1). The observe unit tests cover the
// pieces in isolation (config parse, dtype lock, trigger flags, writer -> HDF5).
// This test proves the WHOLE pipeline through a real run: entity code firing
// OBSERVE -> per-step gate/commit decision -> the per-group writer -> a flushed
// HDF5 file with the right rows.
//
// In-process rather than B-tier subprocess: sim_instance attaches the observe
// registry in its constructor and detaches (flushing every writer) in its
// destructor, so scoping the sim drives the exact attach -> capture -> flush
// lifecycle. We then read the file back with the HDF5 C API (coast_lib already
// links HDF5). The CLI/run_headless wrapper is covered separately in B-tier.
//
// Config: derived at runtime from the canonical ngm2 mnist config (which fires
// OBSERVE("population","out") from the neuron group) so we don't hand-roll a
// net. Only an observe section is injected; output_dir is made absolute so the
// run is hermetic. Dataset paths stay relative — the test runs from the repo
// root (ctest WORKING_DIRECTORY), where ./data/mnist resolves.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <hdf5.h>
#include <nlohmann/json.hpp>

#include "app/build_env.h"
#include "cli/options.h"
#include "h5_root_links.h"
#include "hdf5/executor.h"
#include "integration_helpers.h"
#include "sim_core/sim_instance.h"

namespace fs = std::filesystem;

namespace {

using coast::test::root_link_names;

hsize_t dataset_rows(hid_t file, const std::string& name) {
    const hid_t d  = H5Dopen2(file, ("/" + name).c_str(), H5P_DEFAULT);
    REQUIRE(d >= 0);
    const hid_t sp = H5Dget_space(d);
    hsize_t dims[8] = {0};
    H5Sget_simple_extent_dims(sp, dims, nullptr);
    H5Sclose(sp);
    H5Dclose(d);
    return dims[0];
}

// The unique value dataset for an OBSERVE site whose local id is `local_id`.
// The value dataset is the flattened full_id, which is either exactly the local
// id or ends in "__<local_id>"; the "__step_index" siblings are skipped. Matching
// the *full* "__<local_id>" suffix keeps "in" from colliding with "in_apical".
// Uniqueness is the real contract (the registry's dtype/shape lock rejects a
// second OBSERVE against the same id). Returns "" when the site never fired.
std::string site_value_ds(const std::vector<std::string>& links,
                          std::string_view local_id) {
    const std::string exact(local_id);
    const std::string suffix = "__" + exact;
    const auto ends_with = [](const std::string& s, std::string_view suf) {
        return s.size() >= suf.size() &&
               s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
    };
    std::string found;
    for (const auto& nm : links) {
        if (ends_with(nm, "__step_index")) continue;   // skip step siblings
        if (nm == exact || ends_with(nm, suffix)) {
            REQUIRE(found.empty());                     // exactly one such site
            found = nm;
        }
    }
    return found;
}

} // namespace

TEST_CASE("observe capture e2e: a configured range writes an HDF5 file with rows inside the window") {
    const fs::path scratch = fs::temp_directory_path() / "coast_observe_e2e";
    std::error_code ec;
    fs::remove_all(scratch, ec);
    fs::create_directories(scratch);
    const fs::path out_dir  = scratch / "out";
    const fs::path cfg_path = scratch / "cfg.json";

    constexpr std::uint64_t k_lo = 50, k_hi = 100;

    // --- derive the observe config from the ngm2 mnist base -------------------
    {
        std::ifstream in(COAST_BASE_CONFIG);
        REQUIRE(in.good());
        nlohmann::json j;
        in >> j;
        j["observe"] = {
            {"output_dir", out_dir.string()},
            {"groups", {{"population", {
                {"ranges",  nlohmann::json::array({nlohmann::json::array({k_lo, k_hi})})},
                {"combine", "or"},
            }}}},
        };
        std::ofstream o(cfg_path);
        o << j.dump();
    }

    // --- run past the capture window, then flush via sim teardown ------------
    hdf5::executor h5;
    {
        coast::cli::options opts;
        opts.config_path = cfg_path.string();
        opts.seed_offset = seed_offset_t{ 0 };
        auto sim = coast::app::build_env(opts, h5);
        REQUIRE(sim);
        coast::test::run_for(*sim, k_hi + 20);
    } // ~sim_instance() -> observe detach() -> writer stop_and_join() flushes

    // --- read the population group's file back -------------------------------
    // output_dir is ABSOLUTE here, so it wins outright and bypasses config_base:
    // the file lands at the given absolute dir, NOT under the default
    // <config_dir>/<config>/observe (scratch/cfg/observe).
    const fs::path h5_path = out_dir / "population.observe.h5";
    REQUIRE(fs::exists(h5_path));
    CHECK(!fs::exists(scratch / "cfg" / "observe"));   // config_base not used

    const hid_t file = H5Fopen(h5_path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);

    // This test fires OBSERVE("population","out"). The value dataset is the
    // namespaced+flattened full_id (which we do not hardcode), and it ends in
    // the site's local id ("out"); the step sibling is <value>__step_index.
    // A group may legitimately write more than one site into this single file,
    // so select the site under test *by name* rather than assuming it is the
    // only one present — and assert that pair is present and unique, which is
    // the real contract (the registry's dtype/shape lock rejects a second
    // OBSERVE against the same id).
    const std::string step_suffix = "__step_index";
    const auto ends_with = [](const std::string& s, std::string_view suf) {
        return s.size() >= suf.size() &&
               s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
    };

    const std::vector<std::string> links = root_link_names(file);
    std::string value_ds;
    for (const auto& nm : links) {
        if (ends_with(nm, step_suffix)) continue;     // skip step datasets
        if (nm == "out" || ends_with(nm, "__out")) {  // the site under test
            REQUIRE(value_ds.empty());                // exactly one "out" value dataset
            value_ds = nm;
        }
    }
    REQUIRE_FALSE(value_ds.empty());                  // the site fired and wrote

    const std::string step_ds = value_ds + step_suffix;
    REQUIRE(std::find(links.begin(), links.end(), step_ds) != links.end());

    // Capture happened, and bounded by the window: 1..(k_hi-k_lo+1) rows.
    const hsize_t rows = dataset_rows(file, value_ds);
    CHECK(rows >= 1);
    CHECK(rows <= (k_hi - k_lo + 1));
    CHECK(dataset_rows(file, step_ds) == rows);

    // Every captured step index lies strictly within the configured window —
    // proves the gate opened only inside the range, not before/after.
    {
        const hid_t d = H5Dopen2(file, ("/" + step_ds).c_str(), H5P_DEFAULT);
        REQUIRE(d >= 0);
        std::vector<std::uint64_t> steps(static_cast<std::size_t>(rows), 0);
        REQUIRE(H5Dread(d, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, steps.data()) >= 0);
        H5Dclose(d);
        for (std::uint64_t s : steps) {
            CHECK(s >= k_lo);
            CHECK(s <= k_hi);
        }
    }

    H5Fclose(file);
    fs::remove_all(scratch, ec);
}

// M3 (app-level location config): with NO output_dir in the config, the observe
// output directory defaults to the app_config leaf ("observe") under config_base
// ("<config_dir>/<config>/") — i.e. next to the config file, isolated per config.
// This is the config-relative DEFAULT path (the prior behavior was CWD-verbatim
// "./observe_out"); sim_instance resolves it from its parsed config under
// config_base. Teeth: if resolution were wrong (empty dir, or not routed under
// config_base) the file would not be where we REQUIRE it.
TEST_CASE("observe capture e2e default_base: an omitted output_dir lands under <config_dir>/<config>/observe") {
    const fs::path scratch = fs::temp_directory_path() / "coast_observe_default_base";
    std::error_code ec;
    fs::remove_all(scratch, ec);
    fs::create_directories(scratch);
    const fs::path cfg_path = scratch / "cfg.json";   // stem "cfg"

    constexpr std::uint64_t k_lo = 50, k_hi = 100;

    // Derive from the ngm2 mnist base; arm the population group but OMIT
    // output_dir so the app default applies (empty leaf -> "observe" under base).
    {
        std::ifstream in(COAST_BASE_CONFIG);
        REQUIRE(in.good());
        nlohmann::json j;
        in >> j;
        j["observe"] = {
            {"groups", {{"population", {
                {"ranges",  nlohmann::json::array({nlohmann::json::array({k_lo, k_hi})})},
                {"combine", "or"},
            }}}},
        };
        std::ofstream o(cfg_path);
        o << j.dump();
    }

    hdf5::executor h5;
    {
        coast::cli::options opts;
        opts.config_path = cfg_path.string();
        opts.seed_offset = seed_offset_t{ 0 };
        auto sim = coast::app::build_env(opts, h5);   // default app_config
        REQUIRE(sim);
        coast::test::run_for(*sim, k_hi + 20);
    } // ~sim_instance() flushes the writer

    // Default config_base "<config_dir>/<config>/" + observe leaf "observe":
    // scratch/cfg/observe/population.observe.h5 (config-relative, NOT the CWD).
    const fs::path h5_path = scratch / "cfg" / "observe" / "population.observe.h5";
    REQUIRE(fs::exists(h5_path));

    const hid_t file = H5Fopen(h5_path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    const std::vector<std::string> links = root_link_names(file);
    CHECK_FALSE(site_value_ds(links, "out").empty());   // the site fired and wrote rows
    H5Fclose(file);

    fs::remove_all(scratch, ec);
}

// NOTE: this case builds a *second* sim in the process. The observe registry is a
// process-wide singleton and the OBSERVE macro's per-call-site caches (keyed by
// owner pointer) are not invalidated across a registry detach, so running this
// case in the same process as the one above can dereference a stale site if the
// second sim's entities reuse freed addresses. CMake therefore runs each case as
// its own ctest process (--test-case filter); keep that isolation if adding cases.
//
// hd_ngm3 (CPU) parity: the group emits OBSERVE("population","in") — the proximal
// feedforward input, mirroring ngm2's "in" — alongside the pre-existing
// OBSERVE("population","out"). Independently, the dataset loader emits the
// model-agnostic OBSERVE("input","index") (already present, model-independent).
// Enabling both groups on an ngm3 run must produce input/index, population/in,
// and population/out — the trio an eval harness needs to align outputs with the
// presented sample. (mnist_ngm3 has apical_cnt=0, so population/in_apical is
// dormant here; it fires only when an apical bank is configured.)
TEST_CASE("observe capture e2e (ngm3): population/in + population/out + input/index are written") {
    const fs::path scratch = fs::temp_directory_path() / "coast_observe_e2e_ngm3";
    std::error_code ec;
    fs::remove_all(scratch, ec);
    fs::create_directories(scratch);
    const fs::path out_dir  = scratch / "out";
    const fs::path cfg_path = scratch / "cfg.json";

    constexpr std::uint64_t k_lo = 50, k_hi = 100;

    // Derive from the CPU ngm3 mnist config; capture the same window for both the
    // "population" group (fires "in" and "out" from the neuron group) and the
    // "input" group (the loader's "index" channel).
    {
        std::ifstream in(COAST_NGM3_CONFIG);
        REQUIRE(in.good());
        nlohmann::json j;
        in >> j;
        const auto range = nlohmann::json{
            {"ranges",  nlohmann::json::array({nlohmann::json::array({k_lo, k_hi})})},
            {"combine", "or"},
        };
        j["observe"] = {
            {"output_dir", out_dir.string()},
            {"groups", {{"population", range}, {"input", range}}},
        };
        std::ofstream o(cfg_path);
        o << j.dump();
    }

    hdf5::executor h5;
    {
        coast::cli::options opts;
        opts.config_path = cfg_path.string();
        opts.seed_offset = seed_offset_t{ 0 };
        auto sim = coast::app::build_env(opts, h5);
        REQUIRE(sim);
        coast::test::run_for(*sim, k_hi + 20);
    } // ~sim_instance() flushes every writer

    const auto check_pair_bounded = [&](hid_t file, const std::vector<std::string>& links,
                                        const std::string& value_ds) {
        REQUIRE_FALSE(value_ds.empty());               // the site fired and wrote
        const std::string step_ds = value_ds + "__step_index";
        REQUIRE(std::find(links.begin(), links.end(), step_ds) != links.end());
        const hsize_t rows = dataset_rows(file, value_ds);
        CHECK(rows >= 1);
        CHECK(rows <= (k_hi - k_lo + 1));
        CHECK(dataset_rows(file, step_ds) == rows);
    };

    // population group: the new "in" (proximal sample) plus the existing "out".
    {
        const fs::path h5_path = out_dir / "population.observe.h5";
        REQUIRE(fs::exists(h5_path));
        const hid_t file = H5Fopen(h5_path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        REQUIRE(file >= 0);
        const std::vector<std::string> links = root_link_names(file);
        check_pair_bounded(file, links, site_value_ds(links, "in"));
        check_pair_bounded(file, links, site_value_ds(links, "out"));
        H5Fclose(file);
    }

    // input group: the model-agnostic "index" channel from the dataset loader.
    {
        const fs::path h5_path = out_dir / "input.observe.h5";
        REQUIRE(fs::exists(h5_path));
        const hid_t file = H5Fopen(h5_path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        REQUIRE(file >= 0);
        const std::vector<std::string> links = root_link_names(file);
        check_pair_bounded(file, links, site_value_ds(links, "index"));
        H5Fclose(file);
    }

    fs::remove_all(scratch, ec);
}
