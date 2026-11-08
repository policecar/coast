// Unit test 3.3 — run manifest: format contract + error path.
//
// coast_runtime.json is "the stable interface a downstream analysis harness
// reads". This test pins the parts of that contract that are pure and
// deterministic, plus the documented failure behavior:
//   - iso8601_utc: deterministic UTC formatting (downstream parses these).
//   - to_string(exit_reason_t): the exact reason labels.
//   - write_run_manifest happy path: returns true; the file is valid JSON with
//     the schema_version + recorded fields.
//   - the (n_steps_requested, unlimited) pair in both of its states, since a 0
//     in the former means "unbounded" only when the latter is true.
//   - write_run_manifest unwritable path: returns false (does NOT throw — it is
//     noexcept) and produces no file.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <system_error>

#include <nlohmann/json.hpp>

#include "sim_core/run_manifest.h"
#include "json_cfg/json_loader.h"

namespace fs = std::filesystem;
using sim::exit_reason_t;
using sim::run_manifest_inputs_t;

TEST_CASE("run_manifest: iso8601_utc formats UTC deterministically") {
    using namespace std::chrono;
    // system_clock's epoch is the Unix epoch (guaranteed since C++20).
    const auto epoch = system_clock::time_point{};
    CHECK(sim::iso8601_utc(epoch) == "1970-01-01T00:00:00Z");
    CHECK(sim::iso8601_utc(epoch + hours(1) + minutes(2) + seconds(3))
          == "1970-01-01T01:02:03Z");
}

TEST_CASE("run_manifest: every exit_reason has a distinct, stable label") {
    CHECK(sim::to_string(exit_reason_t::completed_n_steps)    == "completed_n_steps");
    CHECK(sim::to_string(exit_reason_t::fatal_error)          == "fatal_error");
    CHECK(sim::to_string(exit_reason_t::user_signal)          == "user_signal");
    CHECK(sim::to_string(exit_reason_t::step_budget_exceeded) == "step_budget_exceeded");
}

TEST_CASE("run_manifest: happy path writes valid JSON with the recorded fields") {
    const fs::path out = fs::temp_directory_path() / "coast_run_manifest_ok.json";
    std::error_code ec;
    fs::remove(out, ec);

    run_manifest_inputs_t in;
    in.output_path       = out;
    in.seed_offset       = seed_offset_t{ 42 };
    in.n_steps_requested = 1000;
    in.n_steps_completed = 1000;
    in.exit_reason       = exit_reason_t::completed_n_steps;
    in.started_at_utc    = "1970-01-01T00:00:00Z";
    in.completed_at_utc  = "1970-01-01T00:00:10Z";

    in.node_seeds = { {0, "mnist", seed_t{ 0 }, std::nullopt},
                      {1, "group", seed_t{ 5 }, seed_t{ 4294967295u }} };

    loader::global_cfg_t cfg;  // defaults are fine; observe block is empty
    cfg.sim.nodes.push_back(sim::sim_node_t{
        .id = 0,
        .label = "mnist",
        .entity_type = sim::io_entity_type_t::mnist_input,
        .entity_cfg = "default",
        .inputs = {},
        .random_seed_offset = seed_offset_t{ 0 }
    });

    REQUIRE(sim::write_run_manifest(in, cfg));
    REQUIRE(fs::exists(out));

    std::ifstream is(out);
    nlohmann::json j;
    is >> j;

    CHECK(j.at("schema_version").get<std::string>() == "1");
    CHECK(j.at("seed_offset").get<std::uint32_t>() == 42);
    CHECK(j.at("n_steps_requested").get<std::uint64_t>() == 1000);
    // `unlimited` disambiguates n_steps_requested: a 0 there means "unbounded
    // run" only when this is true, never "unset". A finite run must report false.
    CHECK(j.at("unlimited").get<bool>() == false);
    CHECK(j.at("n_steps_completed").get<std::uint64_t>() == 1000);
    CHECK(j.at("exit_reason").get<std::string>() == "completed_n_steps");
    CHECK(j.at("savestate_outputs").is_array());
    CHECK(j.at("savestate_outputs").empty());
    CHECK(j.contains("observe"));
    CHECK(j.contains("datasets"));
    CHECK(j.at("datasets").contains("mnist"));
    CHECK(j.at("gpu_adapter").is_null());  // empty adapter -> null (CPU-only run)
    CHECK(j.contains("node_seeds"));
    REQUIRE(j.at("node_seeds").is_array());
    REQUIRE(j.at("node_seeds").size() == 2);
    CHECK(j.at("node_seeds")[0].at("id") == 0);
    CHECK(j.at("node_seeds")[0].at("label") == "mnist");
    CHECK(j.at("node_seeds")[0].at("effective_seed") == 0);
    // inhibition_seed (#190) is written only for a node that has one.
    CHECK_FALSE(j.at("node_seeds")[0].contains("inhibition_seed"));
    CHECK(j.at("node_seeds")[1].at("effective_seed") == 5);
    CHECK(j.at("node_seeds")[1].at("inhibition_seed").get<std::uint32_t>() == 4294967295u);
    CHECK(j.at("retina").empty());   // no retina node

    fs::remove(out, ec);
}

TEST_CASE("run_manifest: the resolved config of each active retina node is recorded") {
    const fs::path out = fs::temp_directory_path() / "coast_run_manifest_retina.json";
    std::error_code ec;
    fs::remove(out, ec);

    run_manifest_inputs_t in;
    in.output_path = out;
    in.exit_reason = exit_reason_t::completed_n_steps;

    loader::global_cfg_t cfg;
    cfg.retina["on"].surround_weight = 0.7;
    cfg.retina["unused"].gain        = 5.0;
    cfg.sim.nodes.push_back(sim::sim_node_t{
        .id = 1, .label = "retina_on", .entity_type = sim::io_entity_type_t::retina,
        .entity_cfg = "on", .inputs = {}, .random_seed_offset = seed_offset_t{ 0 } });

    REQUIRE(sim::write_run_manifest(in, cfg));
    std::ifstream is(out);
    nlohmann::json j;
    is >> j;

    REQUIRE(j.at("retina").size() == 1);   // the unused config is not listed
    CHECK(j.at("retina").at("on").at("surround_weight") == 0.7);
    CHECK(j.at("retina").at("on").at("polarity") == "on");
    CHECK_FALSE(j.at("datasets").contains("retina"));   // not a dataset

    fs::remove(out, ec);
}

TEST_CASE("run_manifest: an unlimited run serializes unlimited=true alongside a 0 request") {
    // The other half of the disambiguation contract checked above: a downstream
    // harness reads `n_steps_requested` and `unlimited` together, so the 0/true
    // pairing an unbounded run produces must survive serialization. This mirrors
    // exactly what run_headless.cpp builds for --unlimited:
    //     manifest.unlimited         = (steps_mode == run_length::unlimited);
    //     manifest.n_steps_requested = manifest.unlimited ? 0 : opts.max_steps;
    // No CLI path reaches this today (--unlimited rejects --emit-run-manifest at
    // parse time), so the writer is the only thing under test here.
    const fs::path out = fs::temp_directory_path() / "coast_run_manifest_unlimited.json";
    std::error_code ec;
    fs::remove(out, ec);

    run_manifest_inputs_t in;
    in.output_path       = out;
    in.n_steps_requested = 0;      // "unbounded", NOT "unset" — `unlimited` says which
    in.unlimited         = true;
    in.n_steps_completed = 7331;
    in.exit_reason       = exit_reason_t::user_signal;
    in.started_at_utc    = "1970-01-01T00:00:00Z";
    in.completed_at_utc  = "1970-01-01T00:00:10Z";

    const loader::global_cfg_t cfg;  // no nodes: the pairing is what's under test

    REQUIRE(sim::write_run_manifest(in, cfg));
    REQUIRE(fs::exists(out));

    std::ifstream is(out);
    nlohmann::json j;
    is >> j;

    CHECK(j.at("unlimited").get<bool>() == true);
    CHECK(j.at("n_steps_requested").get<std::uint64_t>() == 0);
    CHECK(j.at("n_steps_completed").get<std::uint64_t>() == 7331);
    CHECK(j.at("exit_reason").get<std::string>() == "user_signal");

    fs::remove(out, ec);
}

TEST_CASE("run_manifest: path fields use the bundle frame (./ under the manifest dir, absolute outside)") {
    const fs::path root = fs::temp_directory_path() / "coast_manifest_frame";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);  // the manifest must be writable here
    const fs::path out = root / "coast_runtime.json";

    run_manifest_inputs_t in;
    in.output_path      = out;
    in.started_at_utc   = "1970-01-01T00:00:00Z";
    in.completed_at_utc = "1970-01-01T00:00:00Z";

    // observe dir is BELOW the manifest dir -> recorded relative, marked with "./".
    in.observe_output_dir = root / "observe";

    // record_savestate stores raw ABSOLUTE paths; the writer picks the frame. One
    // save under the manifest dir (-> "./…"), one in an unrelated tree (-> absolute,
    // never climbed to with "../..").
    const fs::path outside = fs::temp_directory_path() / "coast_manifest_elsewhere" / "state.h5";
    in.savestate_outputs = {
        { (root / "snapshots" / "state.h5").generic_string(), 50,  0 },
        { outside.lexically_normal().generic_string(),        100, 0 },
    };

    loader::global_cfg_t cfg;  // empty observe/datasets is fine
    REQUIRE(sim::write_run_manifest(in, cfg));

    std::ifstream is(out);
    nlohmann::json j;
    is >> j;

    CHECK(j.at("observe").at("output_dir").get<std::string>() == "./observe");

    const auto& ss = j.at("savestate_outputs");
    REQUIRE(ss.size() == 2);
    CHECK(ss[0].at("path").get<std::string>() == "./snapshots/state.h5");  // below -> relative
    const auto p_out = ss[1].at("path").get<std::string>();
    CHECK(fs::path{p_out}.is_absolute());                                   // outside -> absolute
    CHECK(p_out == outside.lexically_normal().generic_string());
    CHECK(p_out.find("..") == std::string::npos);                          // never climbed out

    fs::remove_all(root, ec);
}

TEST_CASE("run_manifest: a missing parent directory is created (like --checkpoint/--save-state)") {
    // A nested, not-yet-existing output dir must be created rather than failing —
    // matching the auto-mkdir the other CLI output paths now get.
    const fs::path root = fs::temp_directory_path() / "coast_manifest_mkdir";
    std::error_code ec;
    fs::remove_all(root, ec);
    const fs::path out = root / "deep" / "nested" / "coast_runtime.json";
    REQUIRE_FALSE(fs::exists(out.parent_path()));

    run_manifest_inputs_t in;
    in.output_path = out;
    in.exit_reason = exit_reason_t::completed_n_steps;
    const loader::global_cfg_t cfg;

    CHECK(sim::write_run_manifest(in, cfg));
    CHECK(fs::exists(out));

    fs::remove_all(root, ec);
}

TEST_CASE("run_manifest: an unwritable output path returns false without throwing") {
    // A genuinely unwritable location: a regular FILE used as the parent directory.
    // create_directories can't turn a file into a dir, so the tmpfile open fails.
    // (A merely-missing dir is now auto-created, so it no longer forces this path.)
    // The writer is noexcept and must return false, not crash, and leave no file.
    const fs::path blocker = fs::temp_directory_path() / "coast_manifest_blocker_file";
    std::error_code ec;
    fs::remove_all(blocker, ec);
    { std::ofstream(blocker) << "not a directory\n"; }
    REQUIRE(fs::is_regular_file(blocker));

    run_manifest_inputs_t in;
    in.output_path = blocker / "manifest.json";   // parent is a file -> uncreatable
    in.exit_reason = exit_reason_t::fatal_error;

    const loader::global_cfg_t cfg;

    CHECK_FALSE(sim::write_run_manifest(in, cfg));
    CHECK_FALSE(fs::exists(in.output_path));

    fs::remove(blocker, ec);
}
