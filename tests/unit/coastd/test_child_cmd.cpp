#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <nlohmann/json.hpp>

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "daemon/child_cmd.h"

using namespace coastd;

namespace fs = std::filesystem;

namespace {

// A unique temp run dir with a snapshots/ subdir; caller drops checkpoints in.
struct temp_run {
    fs::path dir;
    explicit temp_run(const std::string& tag) {
        dir = fs::temp_directory_path() / ("coastd_childcmd_" + tag + "_" +
              std::to_string(::getpid()));
        fs::remove_all(dir);
        fs::create_directories(dir / "snapshots");
    }
    ~temp_run() { std::error_code ec; fs::remove_all(dir, ec); }

    void write_checkpoint(std::uint64_t step) {
        std::ofstream(dir / "snapshots" / ("state_" + std::to_string(step) + ".h5")) << "x";
    }
};

job job_at(const fs::path& run_dir, std::uint64_t target) {
    job j;
    j.id           = "j1";
    j.run_dir      = run_dir.string();
    j.steps_target = target;
    j.ref_sha      = "abc1234";
    return j;
}

// The value of the "-n" argument in a launch plan.
std::string n_arg(const launch_plan& p) {
    for (std::size_t i = 0; i + 1 < p.args.size(); ++i)
        if (p.args[i] == "-n") return p.args[i + 1];
    return "<none>";
}

bool has_arg(const launch_plan& p, const std::string& flag) {
    for (const auto& a : p.args) if (a == flag) return true;
    return false;
}

} // namespace

// coast requires an explicit run length: `-n <count>` with count >= 1, or
// --unlimited. `-n 0` is a PARSE ERROR (it used to mean "run forever"), so a
// bounded job whose newest checkpoint already sits at or past its target must
// never reach a zero budget — that command line would not run at all.
TEST_CASE("plan_launch: bounded job already at target never emits -n 0") {
    temp_run t("at_target");
    t.write_checkpoint(1000);
    daemon_config cfg;
    const launch_plan p = plan_launch(job_at(t.dir, /*target=*/1000), cfg, "/bin/coast");
    CHECK(n_arg(p) != "0");
    CHECK(!has_arg(p, "--unlimited"));
}

// A genuinely unbounded job (target 0) says so with --unlimited and emits no -n at
// all, even with a checkpoint present. --unlimited normally refuses --save-state /
// --emit-run-manifest; --save-on-term is what lifts that refusal, so the plan must
// carry it or the launch is a parse error rather than a run.
TEST_CASE("plan_launch: unbounded job gets --unlimited, never -n 0") {
    temp_run t("unbounded");
    t.write_checkpoint(1000);
    daemon_config cfg;
    const launch_plan p = plan_launch(job_at(t.dir, /*target=*/0), cfg, "/bin/coast");
    CHECK(has_arg(p, "--unlimited"));
    CHECK(n_arg(p) == "<none>");
    CHECK(has_arg(p, "--save-on-term"));
    CHECK(has_arg(p, "--save-state"));
    CHECK(has_arg(p, "--emit-run-manifest"));
}

// A normal resume computes the remaining budget relative to the newest checkpoint.
TEST_CASE("plan_launch: bounded resume uses remaining = target - checkpoint") {
    temp_run t("resume");
    t.write_checkpoint(1000);
    daemon_config cfg;
    const launch_plan p = plan_launch(job_at(t.dir, /*target=*/2000), cfg, "/bin/coast");
    CHECK(n_arg(p) == "1000");
    // A resume also loads the newest checkpoint.
    bool has_load = false;
    for (const auto& a : p.args) if (a == "--load-state") has_load = true;
    CHECK(has_load);
}

// A fresh bounded run (no checkpoint) uses the whole target and does NOT resume.
TEST_CASE("plan_launch: fresh bounded run uses -n target, no --load-state") {
    temp_run t("fresh");
    daemon_config cfg;
    const launch_plan p = plan_launch(job_at(t.dir, /*target=*/500), cfg, "/bin/coast");
    CHECK(n_arg(p) == "500");
    for (const auto& a : p.args) CHECK(a != "--load-state");
}

// prepare_run_config exposes a configured dataset_base as a `data` symlink in the
// run dir, so a job config's relative "data/..." path resolves from the child CWD.
TEST_CASE("prepare_run_config: links dataset_base as a `data` symlink in the run dir") {
    temp_run t("ds_link");
    // A real directory to point at, so the link is not dangling.
    const fs::path base = t.dir.parent_path() /
                          ("coastd_childcmd_dsbase_" + std::to_string(::getpid()));
    fs::remove_all(base);
    fs::create_directories(base);

    const job j = job_at(t.dir, /*target=*/100);
    REQUIRE(prepare_run_config("{}", j, base));

    const fs::path link = t.dir / "data";
    std::error_code ec;
    CHECK(fs::is_symlink(link, ec));
    CHECK(fs::equivalent(link, base, ec));   // resolves to the configured base

    // Idempotent: a second prepare (e.g. a resume path) must not error or duplicate.
    REQUIRE(prepare_run_config("{}", j, base));
    CHECK(fs::is_symlink(link, ec));

    fs::remove_all(base);
}

// SPEC.md S.5.14: a job's run_id is fixed when coastd writes its run config, so every
// launch of the job (a resume reads this file) is the same run and continues its
// datasets' sequences. "auto", and no run_id (its default), become a fresh id; an
// explicit one is kept; the archived submission keeps what was submitted.
TEST_CASE("prepare_run_config: replaces run_id \"auto\" with a fresh id, once per job") {
    const auto written = [](const job& j) {
        std::ifstream in(run_config_path(j));
        return nlohmann::json::parse(in);
    };
    temp_run a("run_id_auto"), b("run_id_none"), c("run_id_named");
    const job ja = job_at(a.dir, 100), jb = job_at(b.dir, 100), jc = job_at(c.dir, 100);
    REQUIRE(prepare_run_config(R"({"run_id": "auto"})", ja, {}));
    REQUIRE(prepare_run_config("{}", jb, {}));
    REQUIRE(prepare_run_config(R"({"run_id": "trial-1"})", jc, {}));
    const std::string ida = written(ja).at("run_id"), idb = written(jb).at("run_id");
    CHECK(ida.size() == 32);
    CHECK(idb.size() == 32);
    CHECK(ida != idb);                                    // each job its own run
    CHECK(written(jc).at("run_id") == "trial-1");
    std::ifstream sub(run_submitted_config_path(ja));
    CHECK(nlohmann::json::parse(sub).at("run_id") == "auto");   // verbatim
}

// A run_id that is not a string is kept as submitted, not thrown on: coast refuses
// the config when the job launches (a type error, exit 1).
TEST_CASE("prepare_run_config: a non-string run_id is kept, not thrown on") {
    temp_run t("run_id_number");
    const job j = job_at(t.dir, 100);
    bool ok = false;
    CHECK_NOTHROW(ok = prepare_run_config(R"({"run_id": 7})", j, {}));
    REQUIRE(ok);
    std::ifstream in(run_config_path(j));
    CHECK(nlohmann::json::parse(in).at("run_id") == 7);
}

// An empty dataset_base leaves the mechanism off: no `data` entry is created.
TEST_CASE("prepare_run_config: empty dataset_base creates no `data` link") {
    temp_run t("ds_none");
    const job j = job_at(t.dir, /*target=*/100);
    REQUIRE(prepare_run_config("{}", j, {}));   // empty base
    std::error_code ec;
    CHECK_FALSE(fs::exists(t.dir / "data", ec));
}

// The daemon-owned app_config is handed to the child via --app-config so ambient
// $COAST_APP_CONFIG / ./coast.json cannot silently redirect a job's outputs. It is
// emitted only when a path is configured (empty = mechanism disabled).
TEST_CASE("plan_launch: passes --app-config when set, omits it when empty") {
    temp_run t("appcfg");

    daemon_config unset;   // app_config_path defaults empty
    const launch_plan p0 = plan_launch(job_at(t.dir, /*target=*/500), unset, "/bin/coast");
    for (const auto& a : p0.args) CHECK(a != "--app-config");

    daemon_config set;
    set.app_config_path = "/etc/coastd/coast_app_config.json";
    const launch_plan p1 = plan_launch(job_at(t.dir, /*target=*/500), set, "/bin/coast");
    bool found = false;
    for (std::size_t i = 0; i + 1 < p1.args.size(); ++i)
        if (p1.args[i] == "--app-config") {
            CHECK(p1.args[i + 1] == "/etc/coastd/coast_app_config.json");  // flag carries the path
            found = true;
        }
    CHECK(found);
}

// ---- load_state: the initial state of a job's FIRST launch ------------------

namespace {
// The value following `flag`, or "<none>".
std::string arg_after(const launch_plan& p, const std::string& flag) {
    for (std::size_t i = 0; i + 1 < p.args.size(); ++i)
        if (p.args[i] == flag) return p.args[i + 1];
    return "<none>";
}
} // namespace

// coast continues a loaded state's step count, so the first launch of a job
// submitted at load_state_step 50000 for 3000 steps (steps_target stored as 53000)
// must say -n 3000. A plan that ignored load_state_step would say -n 53000.
TEST_CASE("plan_launch: a first launch loads load_state and runs target - its step") {
    temp_run t("ls_first");
    job j = job_at(t.dir, /*target=*/53000);
    j.load_state      = "/runs/train/final_state.h5";
    j.load_state_step = 50000;
    const launch_plan p = plan_launch(j, daemon_config{}, "/bin/coast");
    CHECK(arg_after(p, "--load-state") == "/runs/train/final_state.h5");
    CHECK(n_arg(p) == "3000");
}

// A resume from the job's OWN checkpoint wins: the initial state must not be
// loaded again (that would rewind the run), and -n counts from the checkpoint.
TEST_CASE("plan_launch: the job's own checkpoint wins over load_state") {
    temp_run t("ls_resume");
    t.write_checkpoint(51000);
    job j = job_at(t.dir, /*target=*/53000);
    j.load_state      = "/runs/train/final_state.h5";
    j.load_state_step = 50000;
    const launch_plan p = plan_launch(j, daemon_config{}, "/bin/coast");
    CHECK(arg_after(p, "--load-state") == (t.dir / "snapshots" / "state_51000.h5").string());
    for (const auto& a : p.args) CHECK(a != "/runs/train/final_state.h5");
    CHECK(n_arg(p) == "2000");
}

// load_state_from names a JOB: only a done/ job with a final_state.h5 qualifies,
// and its steps_done is the step the new job continues from.
TEST_CASE("initial_state_from: a done job's final state is accepted with its steps_done") {
    temp_run t("isf_ok");
    std::ofstream(t.dir / "final_state.h5") << "x";
    job src = job_at(t.dir, /*target=*/50000);
    src.steps_done = 50000;
    const auto v = initial_state_from("train", &src, bucket::done);
    REQUIRE_MESSAGE(v.has_value(), (v ? "" : v.error()));
    CHECK(v->path == fs::canonical(t.dir / "final_state.h5"));
    CHECK(v->step == 50000);
}

TEST_CASE("initial_state_from: empty, unknown, not done, or without a final state is refused") {
    temp_run t("isf_bad");
    job src = job_at(t.dir, /*target=*/50000);
    src.steps_done = 50000;

    const auto empty = initial_state_from("", &src, bucket::done);
    REQUIRE_FALSE(empty.has_value());
    CHECK(empty.error().find("empty") != std::string::npos);

    const auto unknown = initial_state_from("nope", nullptr, bucket::active);
    REQUIRE_FALSE(unknown.has_value());
    CHECK(unknown.error().find("no known job") != std::string::npos);

    std::ofstream(t.dir / "final_state.h5") << "x";   // present, so only the bucket refuses
    for (const bucket b : {bucket::active, bucket::failed, bucket::cancelled, bucket::quarantine}) {
        const auto v = initial_state_from("train", &src, b);
        REQUIRE_FALSE(v.has_value());
        CHECK(v.error().find("not done/") != std::string::npos);
    }

    fs::remove(t.dir / "final_state.h5");
    const auto missing = initial_state_from("train", &src, bucket::done);
    REQUIRE_FALSE(missing.has_value());
    CHECK(missing.error().find("no final_state.h5") != std::string::npos);

    fs::create_directory(t.dir / "final_state.h5");   // a directory is not a state file
    const auto dir = initial_state_from("train", &src, bucket::done);
    REQUIRE_FALSE(dir.has_value());
    CHECK(dir.error().find("no final_state.h5") != std::string::npos);
}
