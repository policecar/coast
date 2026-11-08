// tests/integration_observe/observe_contract_test.cpp
//
// Producer side of observe_contract.json (beside this file): the CPU observe
// contract that offline consumers (roast) read. Three pure checkers compare the
// contract with what a run emitted; the synthetic cases plant one mismatch per
// failure branch to prove each checker can fail. The real-run cases (ngm3, ngm25)
// then check a small CPU run's population and population_structure emission — site
// names, dtype, width, out == inhibit(out_pre_inhibition) per step, and
// representation_count against a recount. Other sites get only the static
// pattern/kind check.
//
// The real runs need build_env: the observe registry routes emission only after
// attach() to a live sim::env, so perform_observations() cannot be driven bare.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <hdf5.h>
#include <nlohmann/json.hpp>

#include "app/build_env.h"
#include "cli/options.h"
#include "h5_root_links.h"
#include "hd_ngm25/hd_ngm25_neuron_group.h"
#include "hd_ngm3/hd_ngm3_neuron_group.h"
#include "hdf5/executor.h"
#include "inhibition/inhibition.h"     // the real inhibit() math, not a reimplementation
#include "integration_helpers.h"
#include "io/mnist_db.h"               // the real on-disk input width
#include "json_cfg/inhibition_json.h"  // production JSON adaptors for inhibition_params_t
#include "sim_core/sim_instance.h"

namespace fs = std::filesystem;

namespace coast::test {

// Whole dataset as flat row-major values; empty on any failure (callers REQUIRE).
template <typename T>
std::vector<T> read_dataset(hid_t file, const std::string& path, hid_t mem_type) {
    std::vector<T> out;
    const hid_t d = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    if (d < 0) return out;
    const hid_t sp = H5Dget_space(d);
    const hssize_t n = H5Sget_simple_extent_npoints(sp);
    if (n > 0) {
        out.resize(static_cast<std::size_t>(n));
        if (H5Dread(d, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) < 0)
            out.clear();
    }
    H5Sclose(sp);
    H5Dclose(d);
    return out;
}

} // namespace coast::test

namespace {

// local_id = final "__"-separated component of a dataset name; mirrors roast's
// RoastIO._local_id so both sides agree on what a "site" is.
std::string local_id_of(const std::string& full_name) {
    const auto pos = full_name.rfind("__");
    return pos == std::string::npos ? full_name : full_name.substr(pos + 2);
}

// Top-level objects that are not observe values; mirrors RoastIO._is_reserved_observe_name.
bool is_reserved_observe_name(const std::string& name) {
    return name.ends_with("__step_index") || name == "metadata";
}

// Append new fields at the END: the synthetic cases brace-init this positionally.
struct contract_entry_t {
    std::string              name;
    std::string              group;
    std::string              site;
    std::vector<std::string> backends;
    std::string              dtype;     // "float32" / "uint64"
    std::string              shape;     // "group_width" / "input_width" / "scalar"
    std::string              kind;      // "group_level" / "per_neuron" / ...
    std::string              pattern;   // local_id regex roast builds its reader from
    std::vector<std::string> models;    // which cpu model(s) emit this site
};

std::vector<contract_entry_t> load_observe_contract(const fs::path& path) {
    std::ifstream in(path);
    REQUIRE(in.good());
    nlohmann::json arr;
    in >> arr;
    REQUIRE(arr.is_array());

    std::vector<contract_entry_t> out;
    out.reserve(arr.size());
    for (const auto& item : arr) {
        using strings = std::vector<std::string>;
        out.push_back({item.at("name").get<std::string>(), item.at("group").get<std::string>(),
                       item.at("site").get<std::string>(), item.value("backends", strings{}),
                       item.value("dtype", ""), item.value("shape", ""),
                       item.value("kind", ""), item.value("pattern", ""),
                       item.value("models", strings{})});
    }
    return out;
}

bool has_cpu_backend(const contract_entry_t& e) {
    return std::find(e.backends.begin(), e.backends.end(), "cpu") != e.backends.end();
}

// Name check: the cpu sites the contract lists for `group` and `model` must equal
// the set emitted to that group's file. population/in_apical is exempt while
// apical_cnt == 0 (dormant by design). An emitted site with no cpu entry in this
// group (spec drift, including a site emitted under the wrong group) is reported
// apart from one whose `models` merely omits this model.
std::vector<std::string> check_group_cpu_contract(
    const std::vector<contract_entry_t>& contract,
    const std::string&                   group,
    const std::set<std::string>&         emitted_sites,
    const std::string&                   model,
    bool                                 apical_configured)
{
    std::set<std::string> required_for_model;
    std::set<std::string> any_cpu_sites;
    for (const auto& e : contract) {
        if (e.group != group || !has_cpu_backend(e)) continue;
        any_cpu_sites.insert(e.site);
        if (std::find(e.models.begin(), e.models.end(), model) != e.models.end()) {
            required_for_model.insert(e.site);
        }
    }

    std::vector<std::string> mismatches;

    for (const auto& site : required_for_model) {
        if (site == "in_apical" && !apical_configured) continue;
        if (emitted_sites.find(site) == emitted_sites.end()) {
            mismatches.push_back(
                "observe_contract.json declares " + group + "/" + site +
                " for model '" + model + "' (backends=[\"cpu\"]) but the " + model +
                " CPU run did not emit it "
                "(producer regression, or the contract needs updating)");
        }
    }

    for (const auto& site : emitted_sites) {
        if (any_cpu_sites.find(site) == any_cpu_sites.end()) {
            mismatches.push_back(
                "the " + model + " CPU run emitted " + group + "/" + site +
                " but observe_contract.json has no backends=[\"cpu\"] entry for it "
                "(spec drift — update observe_contract.json)");
        } else if (required_for_model.find(site) == required_for_model.end()) {
            mismatches.push_back(
                "the " + model + " CPU run emitted " + group + "/" + site +
                " but observe_contract.json's \"models\" for that site does not list '" +
                model + "' (contract drift — update observe_contract.json's \"models\")");
        }
    }

    return mismatches;
}

std::string join_mismatches(const std::vector<std::string>& mismatches) {
    std::string joined;
    for (const auto& m : mismatches) { joined += m; joined += '\n'; }
    return joined;
}

// A planted mismatch must yield exactly one report, naming every fragment.
void require_single_mismatch(const std::vector<std::string>& m,
                             std::initializer_list<std::string_view> fragments) {
    REQUIRE_MESSAGE(m.size() == 1, join_mismatches(m));
    for (const auto f : fragments) CHECK_MESSAGE(m[0].find(f) != std::string::npos, f);
}

// Static check of every entry, emitted or not: `pattern` must match `site` (roast
// builds its reader regex from it), and every population-level entry's kind must
// be group_level (population_structure_representation_count shares a pattern with
// the per-dendrite entry but must not share its kind).
std::vector<std::string> check_contract_pattern_and_kind(
    const std::vector<contract_entry_t>& contract)
{
    std::vector<std::string> mismatches;
    for (const auto& e : contract) {
        bool pattern_matches = false;
        try {
            pattern_matches = std::regex_match(e.site, std::regex(e.pattern));
        } catch (const std::regex_error&) {
            pattern_matches = false;   // an invalid regex also fails the check below
        }
        if (!pattern_matches) {
            mismatches.push_back(e.group + "/" + e.site + ": pattern '" + e.pattern +
                "' does not match site name '" + e.site + "' (contract drift — "
                "roast builds its local_id reader regex from this field)");
        }
        if ((e.group == "population" || e.group == "population_structure") &&
            e.kind != "group_level") {
            mismatches.push_back(e.group + "/" + e.site + ": kind '" + e.kind +
                "' != expected 'group_level' for a population-level entry");
        }
    }
    return mismatches;
}

struct emitted_site_t {
    std::string dtype;   // mapped from the HDF5 dataset type
    hsize_t     width;   // per-step element count (dims[1], or 1 for a 1-D scalar)
    std::string link = {};   // full dataset name under the file root
};

std::string h5_dtype_string(hid_t t) {
    const H5T_class_t cls = H5Tget_class(t);
    const size_t      sz  = H5Tget_size(t);
    if (cls == H5T_FLOAT)   return sz == 4 ? "float32" : sz == 8 ? "float64" : "float?";
    if (cls == H5T_INTEGER) {
        const std::string p = (H5Tget_sign(t) == H5T_SGN_NONE) ? "uint" : "int";
        return p + std::to_string(sz * 8);
    }
    return "?";
}

// Every observe value in `file`, keyed by site.
std::map<std::string, emitted_site_t> read_emitted_sites(hid_t file) {
    std::map<std::string, emitted_site_t> sites;
    for (const auto& nm : coast::test::root_link_names(file)) {
        if (is_reserved_observe_name(nm)) continue;
        const hid_t d  = H5Dopen2(file, ("/" + nm).c_str(), H5P_DEFAULT);
        REQUIRE(d >= 0);
        const hid_t t  = H5Dget_type(d);
        const hid_t sp = H5Dget_space(d);
        const int   nd = H5Sget_simple_extent_ndims(sp);
        hsize_t dims[8] = {0};
        H5Sget_simple_extent_dims(sp, dims, nullptr);
        sites[local_id_of(nm)] = { h5_dtype_string(t), nd >= 2 ? dims[1] : hsize_t{1}, nm };
        H5Tclose(t); H5Sclose(sp); H5Dclose(d);
    }
    return sites;
}

// "group_width" = neuron_cnt of the model section (the top-level key carrying
// "neuron_groups", as in roast's _template_model).
hsize_t config_neuron_cnt(const nlohmann::json& cfg) {
    for (const auto& v : cfg) {   // top-level section values
        if (v.is_object() && v.contains("neuron_groups")) {
            for (const auto& gv : v.at("neuron_groups")) {
                if (gv.contains("neuron_cnt")) return gv.at("neuron_cnt").get<hsize_t>();
            }
        }
    }
    return 0;
}

// "input_width" is not in the model config (proximal_cnt/apical_cnt are dendrite
// counts), so read it from the dataset the mnist_input node loads, with the
// loader's own mnist_db. Returns 0 if any lookup step fails.
hsize_t config_input_width(const nlohmann::json& cfg) {
    std::string entity_cfg;
    if (cfg.contains("sim") && cfg.at("sim").contains("nodes")) {
        for (const auto& node : cfg.at("sim").at("nodes")) {
            if (node.value("entity_type", std::string{}) == "mnist_input") {
                entity_cfg = node.value("entity_cfg", std::string{});
                break;
            }
        }
    }
    if (entity_cfg.empty()) return 0;
    if (!cfg.contains("datasets") || !cfg.at("datasets").contains("mnist")) return 0;
    const auto& mnist_entries = cfg.at("datasets").at("mnist");
    if (!mnist_entries.contains(entity_cfg)) return 0;
    const auto& entry = mnist_entries.at(entity_cfg);
    const std::string image_file = entry.value("image_file", std::string{});
    const std::string label_file = entry.value("label_file", std::string{});
    if (image_file.empty() || label_file.empty()) return 0;
    const mnist::mnist_db db(image_file, label_file);
    return static_cast<hsize_t>(db.get_image_size());
}

// Rebuilds the inhibitor_t the run's neuron group built, exactly as hd_ngm3 and
// hd_ngm25 do (make_inhibitor(cs.inhibitions, inhibition_cfg,
// local_inhibition_strength)), so the out == inhibit(out_pre_inhibition) check
// uses the real math rather than a copy that could drift.
std::unique_ptr<inhibition::inhibitor_t> build_group_inhibitor(const nlohmann::json& cfg) {
    std::unordered_map<std::string, inhibition::inhibition_params_t> dict;
    std::string inhibition_cfg_name;
    float local_inhibition_strength = 10.0f;  // ngm3/ngm25 neuron_group_params_t default
    for (const auto& v : cfg) {
        if (!(v.is_object() && v.contains("neuron_groups"))) continue;
        if (v.contains("inhibitions")) {
            for (const auto& [k, val] : v.at("inhibitions").items()) {
                dict.emplace(k, val.get<inhibition::inhibition_params_t>());
            }
        }
        for (const auto& gv : v.at("neuron_groups")) {
            if (gv.contains("local_inhibition_strength"))
                local_inhibition_strength = gv.at("local_inhibition_strength").get<float>();
            if (gv.contains("inhibition_cfg"))
                inhibition_cfg_name = gv.at("inhibition_cfg").get<std::string>();
        }
    }
    return inhibition::make_inhibitor(dict, inhibition_cfg_name, local_inhibition_strength, seed_offset_t{0});
}

// Data check: a right-named site can still carry the wrong buffer. For each cpu
// site emitted to `group`, dtype must match exactly and width must match the
// resolved shape (group_width / input_width / scalar = 1; any other shape skips).
std::vector<std::string> check_group_cpu_data_contract(
    const std::vector<contract_entry_t>&         contract,
    const std::string&                           group,
    const std::map<std::string, emitted_site_t>& emitted,
    hsize_t                                      group_width,
    hsize_t                                      input_width,
    bool                                         apical_configured)
{
    std::vector<std::string> mismatches;
    for (const auto& e : contract) {
        if (e.group != group || !has_cpu_backend(e)) continue;
        if (e.site == "in_apical" && !apical_configured) continue;
        const auto it = emitted.find(e.site);
        if (it == emitted.end()) continue;   // presence is the name check's job
        if (it->second.dtype != e.dtype) {
            mismatches.push_back(group + "/" + e.site + ": emitted dtype '" +
                it->second.dtype + "' != contract '" + e.dtype + "'");
        }
        const hsize_t want = (e.shape == "group_width") ? group_width
                           : (e.shape == "input_width") ? input_width
                           : (e.shape == "scalar")       ? hsize_t{1}
                           :                               hsize_t{0};   // unspecified → skip
        if (want != 0 && it->second.width != want) {
            mismatches.push_back(group + "/" + e.site + ": emitted width " +
                std::to_string(it->second.width) + " != contract shape '" + e.shape +
                "' (expected " + std::to_string(want) + ")");
        }
    }
    return mismatches;
}

} // namespace

// ---------------------------------------------------------------------------
// Synthetic (no sim run): each checker passes on a matching fixture and flags
// one planted mismatch per failure branch.
// ---------------------------------------------------------------------------

TEST_CASE("check_group_cpu_contract: emitted sites per group and model") {
    // in_apical is ngm3-only, out_pre_inhibition ngm25-only. neuron/activity and
    // population_structure/representation_count list ngm3 but sit in other groups,
    // so a population check must never require them.
    const std::vector<contract_entry_t> contract = {
        {"population_out",                "population", "out",                {"cpu"}, "", "", "", "", {"ngm3", "ngm25"}},
        {"population_in_apical",          "population", "in_apical",          {"cpu"}, "", "", "", "", {"ngm3"}},
        {"population_out_pre_inhibition", "population", "out_pre_inhibition", {"cpu"}, "", "", "", "", {"ngm25"}},
        {"neuron_activity",               "neuron",     "activity",           {"cpu"}, "", "", "", "", {"ngm3"}},
        {"population_structure_representation_count", "population_structure", "representation_count", {"cpu"}, "", "", "", "", {"ngm3"}},
    };
    const auto check = [&](const std::set<std::string>& emitted, const std::string& model, bool apical) {
        return check_group_cpu_contract(contract, "population", emitted, model, apical);
    };

    SUBCASE("matching sets produce zero mismatches") {
        const auto m3  = check({"out", "in_apical"}, "ngm3", /*apical=*/true);
        const auto m25 = check({"out", "out_pre_inhibition"}, "ngm25", /*apical=*/false);
        const auto ms  = check_group_cpu_contract(contract, "population_structure", {"representation_count"}, "ngm3", false);
        CHECK_MESSAGE(m3.empty(), join_mismatches(m3));
        CHECK_MESSAGE(m25.empty(), join_mismatches(m25));
        CHECK_MESSAGE(ms.empty(), join_mismatches(ms));
    }
    SUBCASE("in_apical is not required while apical is unconfigured") {
        const auto m = check({"out"}, "ngm3", /*apical=*/false);
        CHECK_MESSAGE(m.empty(), join_mismatches(m));
    }
    SUBCASE("planted: declared site not emitted (in_apical once apical is configured) is flagged") {
        const auto m = check({"out"}, "ngm3", /*apical=*/true);
        require_single_mismatch(m, {"in_apical", "did not emit"});
    }
    SUBCASE("planted: emitted site with no cpu contract entry is flagged") {
        const auto m = check({"out", "in_apical", "mystery_site"}, "ngm3", true);
        require_single_mismatch(m, {"mystery_site", "spec drift"});
    }
    SUBCASE("planted: a site emitted under the wrong group is flagged") {
        const auto m = check({"out", "in_apical", "representation_count"}, "ngm3", true);
        require_single_mismatch(m, {"population/representation_count", "spec drift"});
    }
    SUBCASE("planted: emitted site whose models omit this model is flagged") {
        const auto m = check({"out", "in_apical", "out_pre_inhibition"}, "ngm3", true);
        require_single_mismatch(m, {"out_pre_inhibition", "\"models\""});
    }
}

TEST_CASE("check_contract_pattern_and_kind: pattern matches site, population-level kind is group_level") {
    std::vector<contract_entry_t> contract = {
        {"population_out",  "population", "out",      {"cpu"}, "float32", "group_width", "group_level", "^out$"},
        {"neuron_activity", "neuron",     "activity", {"cpu"}, "float32", "scalar",      "per_neuron",  "^activity$"},
        {"population_structure_representation_count", "population_structure", "representation_count", {"cpu"}, "uint64", "scalar", "group_level", "^representation_count$"},
    };

    SUBCASE("matching entries produce zero mismatches") {
        const auto m = check_contract_pattern_and_kind(contract);
        CHECK_MESSAGE(m.empty(), join_mismatches(m));
    }
    SUBCASE("planted: pattern no longer matching its site is flagged") {
        contract[0].pattern = "^outx$";
        const auto m = check_contract_pattern_and_kind(contract);
        require_single_mismatch(m, {"pattern"});
    }
    SUBCASE("planted: population kind edited away from group_level is flagged") {
        contract[0].kind = "per_dendrite";
        const auto m = check_contract_pattern_and_kind(contract);
        require_single_mismatch(m, {"kind"});
    }
    SUBCASE("planted: population_structure kind edited away from group_level is flagged") {
        contract[2].kind = "per_dendrite";
        const auto m = check_contract_pattern_and_kind(contract);
        require_single_mismatch(m, {"kind"});
    }
    SUBCASE("observe_contract.json itself is self-consistent") {
        const auto real = load_observe_contract(fs::path(COAST_OBSERVE_CONTRACT));
        REQUIRE_FALSE(real.empty());
        const auto m = check_contract_pattern_and_kind(real);
        CHECK_MESSAGE(m.empty(), join_mismatches(m));
    }
}

TEST_CASE("check_group_cpu_data_contract: dtype and resolved width per shape") {
    const std::vector<contract_entry_t> contract = {
        {"population_out", "population", "out", {"cpu"}, "float32", "group_width"},
        {"population_in",  "population", "in",  {"cpu"}, "float32", "input_width"},
        {"population_tally", "population", "tally", {"cpu"}, "uint64", "scalar"},
    };
    std::map<std::string, emitted_site_t> emitted = {
        {"out", {"float32", 50}}, {"in", {"float32", 784}}, {"tally", {"uint64", 1}},
    };
    const auto check = [&] {
        return check_group_cpu_data_contract(contract, "population", emitted, /*group_width=*/50, /*input_width=*/784, false);
    };

    SUBCASE("matching dtype and widths produce zero mismatches") {
        const auto m = check();
        CHECK_MESSAGE(m.empty(), join_mismatches(m));
    }
    SUBCASE("planted: wrong dtype (uint64 aliasing a float32 site) is flagged") {
        emitted["out"].dtype = "uint64";
        const auto m = check();
        require_single_mismatch(m, {"dtype"});
    }
    SUBCASE("planted: wrong group_width is flagged") {
        emitted["out"].width = 49;
        const auto m = check();
        require_single_mismatch(m, {"width"});
    }
    SUBCASE("planted: wrong input_width (truncated population/in) is flagged") {
        emitted["in"].width = 783;
        const auto m = check();
        require_single_mismatch(m, {"width"});
    }
    SUBCASE("planted: scalar site emitted as a per-neuron vector is flagged") {
        emitted["tally"].width = 50;
        const auto m = check();
        require_single_mismatch(m, {"width"});
    }
}

// ---------------------------------------------------------------------------
// The real assertion: a small CPU run's actual perform_observations() emission,
// checked against observe_contract.json — run for BOTH ngm3 and ngm25, which add the
// identical sites (population/out_pre_inhibition, population_structure/representation_count).
//
// Modeled on the ngm3 TEST_CASE in observe_capture_test.cpp (same config, same
// build_env + run_for + HDF5-readback shape). The observe registry is a process-wide
// singleton (one sim per process), so each real-run case is registered as its own
// --test-case-filtered ctest entry (see CMakeLists.txt) — the observe_capture
// convention.
// ---------------------------------------------------------------------------

namespace {

// The group's distinct seg ids per dendrite, counted here from the public row
// accessors rather than by the model's own scan, so a wrong scan cannot agree
// with itself. Proximal only: both configs set apical_cnt=0.
template <typename Group>
std::uint64_t recount_representations(const Group& g) {
    REQUIRE(g.get_apical_count() == 0);
    g.refresh_vis_mirror();
    std::uint64_t total = 0;
    for (std::size_t n = 0; n < g.get_neuron_count(); ++n) {
        for (std::size_t d = 0; d < g.get_proximal_count(); ++d) {
            std::set<std::uint8_t> ids;
            for (std::size_t r = 0; r < g.get_rep_count(); ++r) {
                const auto row = g.get_proximal_segment_ids(n, d, r);
                ids.insert(row.begin(), row.end());
            }
            total += ids.size();
        }
    }
    return total;
}

template <typename Group>
void run_population_contract_case(const char* config_path, const std::string& label) {
    coast::test::scratch_dir_t scratch_guard("coast_observe_contract_" + label);
    const fs::path& scratch  = scratch_guard.path();
    const fs::path  out_dir  = scratch / "out";
    const fs::path  cfg_path = scratch / "cfg.json";

    // The windows outlast the run, so the last observed row is the final state.
    // population_structure's starts earlier: its extra rows prove that its gate does
    // not depend on `population`, which would make every population run pay for the scan.
    constexpr std::uint64_t k_structure_lo = 50, k_population_lo = 200;
    constexpr std::uint64_t k_steps = 300, k_hi = k_steps + 20;

    // Both configs set apical_cnt=0, so population/in_apical stays dormant this run.
    nlohmann::json j;   // kept alive past this block: build_group_inhibitor() needs it later
    hsize_t group_width = 0;
    hsize_t input_width = 0;
    {
        std::ifstream in(config_path);
        REQUIRE(in.good());
        in >> j;
        group_width = config_neuron_cnt(j);
        input_width = config_input_width(j);
        REQUIRE(group_width > 0);
        REQUIRE(input_width > 0);
        // Branch early and often, so representation_count moves inside the window.
        for (auto& g : j[label]["neuron_groups"]) {
            g["branch_interval"] = 50;
            g["accumulated_theta_thres"] = 0.05;
        }
        const auto window = [&](std::uint64_t lo) {
            return nlohmann::json{
                {"ranges",  nlohmann::json::array({nlohmann::json::array({lo, k_hi})})},
                {"combine", "or"},
            };
        };
        j["observe"] = {
            {"output_dir", out_dir.string()},
            {"groups", {{"population", window(k_population_lo)},
                        {"population_structure", window(k_structure_lo)}}},
        };
        std::ofstream o(cfg_path);
        o << j.dump();
    }

    std::uint64_t recount = 0;
    hdf5::executor h5;
    {
        coast::cli::options opts;
        opts.config_path = cfg_path.string();
        opts.seed_offset = seed_offset_t{0};
        auto sim = coast::app::build_env(opts, h5);
        REQUIRE(sim);
        coast::test::run_for(*sim, k_steps);
        auto groups = sim->get_sim_env().template get_entities<Group>();
        REQUIRE(groups.has_value());
        REQUIRE((*groups)->size() == 1);
        recount = recount_representations((**groups)[0]);
    } // ~sim_instance() -> observe detach() -> writer stop_and_join() flushes

    const auto contract = load_observe_contract(fs::path(COAST_OBSERVE_CONTRACT));
    CAPTURE(label);
    CAPTURE(group_width);
    CAPTURE(input_width);

    const auto open_group = [&](const std::string& group) {
        const fs::path h5_path = out_dir / (group + ".observe.h5");
        REQUIRE(fs::exists(h5_path));
        const hid_t file = H5Fopen(h5_path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        REQUIRE(file >= 0);
        return file;
    };
    // Names, dtypes and widths of one group's file against the contract.
    const auto check_group = [&](const std::string& group, hid_t file) {
        auto sites = read_emitted_sites(file);
        REQUIRE_FALSE(sites.empty());
        std::set<std::string> names;
        for (const auto& [site, data] : sites) names.insert(site);
        const auto name_mismatches = check_group_cpu_contract(
            contract, group, names, label, /*apical_configured=*/false);
        const auto data_mismatches = check_group_cpu_data_contract(
            contract, group, sites, group_width, input_width, /*apical_configured=*/false);
        CHECK_MESSAGE(name_mismatches.empty(), join_mismatches(name_mismatches));
        CHECK_MESSAGE(data_mismatches.empty(), join_mismatches(data_mismatches));
        return sites;
    };

    std::size_t population_rows = 0;
    {
        const hid_t file  = open_group("population");
        const auto  sites = check_group("population", file);
        REQUIRE(sites.contains("out"));
        REQUIRE(sites.contains("out_pre_inhibition"));
        const auto out_vals     = coast::test::read_dataset<float>(file, "/" + sites.at("out").link, H5T_NATIVE_FLOAT);
        const auto out_pre_vals = coast::test::read_dataset<float>(file, "/" + sites.at("out_pre_inhibition").link, H5T_NATIVE_FLOAT);
        H5Fclose(file);
        REQUIRE_FALSE(out_vals.empty());
        CAPTURE(out_vals.size());
        CAPTURE(out_pre_vals.size());
        // out MUST equal inhibit(out_pre_inhibition), per step, using the SAME
        // inhibitor_t the real run constructed (build_group_inhibitor rebuilds it
        // from the run config), not a reimplementation of inhibit() that could
        // itself drift from the real math. Both payloads are row-major [rows x width].
        REQUIRE(out_vals.size() % group_width == 0);
        REQUIRE(out_vals.size() == out_pre_vals.size());
        const auto inhibitor = build_group_inhibitor(j);
        REQUIRE(inhibitor);
        population_rows = out_vals.size() / group_width;
        std::vector<float> computed(group_width);
        for (std::size_t r = 0; r < population_rows; ++r) {
            const std::size_t off = r * static_cast<std::size_t>(group_width);
            const std::span<const float> pre_row(out_pre_vals.data() + off, group_width);
            inhibitor->inhibit(pre_row, std::span<float>(computed));
            for (hsize_t k = 0; k < group_width; ++k) {
                CAPTURE(r);
                CAPTURE(k);
                CHECK(computed[k] == doctest::Approx(out_vals[off + k]));
            }
        }
    }

    {
        const hid_t file  = open_group("population_structure");
        const auto  sites = check_group("population_structure", file);
        REQUIRE(sites.contains("representation_count"));
        const auto counts = coast::test::read_dataset<std::uint64_t>(
            file, "/" + sites.at("representation_count").link, H5T_NATIVE_UINT64);
        H5Fclose(file);
        REQUIRE_FALSE(counts.empty());
        CAPTURE(counts.front());
        CAPTURE(counts.back());
        CHECK(counts.size() > population_rows);  // gated on its own group
        CHECK(counts.back() > counts.front());   // a constant stub cannot pass
        CHECK(counts.back() == recount);
    }
}   // scratch_guard's destructor removes the scratch dir
} // namespace

TEST_CASE("observe_contract.json: ngm3 CPU population set matches perform_observations (real run)") {
    run_population_contract_case<ngm3::neuron_group_t>(COAST_NGM3_CONFIG, "ngm3");
}

TEST_CASE("observe_contract.json: ngm25 CPU population set matches perform_observations (real run)") {
    run_population_contract_case<ngm25::neuron_group_t>(COAST_NGM25_CONFIG, "ngm25");
}
