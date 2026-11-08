#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <optional>
#include <sstream>
#include <fstream>
#include <thread>

#include <nlohmann/json.hpp>

#include "daemon/child_cmd.h"   // plan_launch
#include "daemon/core.h"
#include "daemon/proc_util.h"

#include "await_progress.h"

#include <cstdlib>

// RAII owner for the core loop thread. The cases below assert with REQUIRE while
// the loop runs, and REQUIRE THROWS on failure — with a bare std::thread that
// means ~std::thread on a joinable thread, i.e. std::terminate: one failed
// assertion would abort the whole binary (skipping every later case) while the
// core thread kept running against a scratch dir the fixture destructor is busy
// deleting. Stopping and joining from a destructor makes a failure a plain test
// failure. (test_fault_matrix.cpp's `rig` already owns its daemon this way.)
struct core_runner {
    coastd::core&  c;
    std::thread    t;
    explicit core_runner(coastd::core& core_ref) : c(core_ref), t([this] { c.run(); }) {}
    core_runner(const core_runner&)            = delete;
    core_runner& operator=(const core_runner&) = delete;
    ~core_runner() { stop_join(); }
    void stop_join() {
        c.stop();
        if (t.joinable()) t.join();
    }
};

using namespace coastd;
namespace fs = std::filesystem;

namespace {
// Launches so far: coastd writes one <run_dir>/logs/run_* per launch.
int launches(const job_view& v) {
    if (v.j.run_dir.empty()) return 0;
    std::error_code ec;
    int n = 0;
    for (const auto& de : fs::directory_iterator(fs::path(v.j.run_dir) / "logs", ec))
        if (de.path().filename().string().starts_with("run_")) ++n;
    return n;
}

// The heartbeat of ONE job: everything observable about it. It stands still when
// the job is stuck, however busy the core loop is (core::passes would keep
// ticking on its scan_period timer and hide a stuck job).
std::string fingerprint(const job_view& v) {
    return std::to_string(static_cast<int>(v.b)) + '|' +
           std::to_string(static_cast<int>(v.j.status)) + '|' + std::to_string(v.j.steps_done) +
           '|' + std::to_string(v.live_step) + '|' + std::to_string(v.j.pid) + '|' +
           std::to_string(v.j.launch_failures) + '|' + v.j.ref_sha + '|' + v.j.failure_reason +
           '|' + std::to_string(launches(v));
}

// The child's own work: CPU time (utime+stime, /proc/<pid>/stat) and bytes read
// (rchar, /proc/<pid>/io). A real coast reports progress only every 10000 steps,
// so during a short run — or its dataset load from a slow NAS — this is the only
// thing that moves while it works. Only for a child that does real work: a shell
// stand-in looping `sleep` forks and reads /bin/sleep every iteration, so its
// counters would tick even while the job is stuck. Empty where /proc is absent.
std::string child_work(std::int64_t pid) {
    if (pid <= 0) return {};
    std::string out;
    std::ifstream st("/proc/" + std::to_string(pid) + "/stat");
    if (std::string line; std::getline(st, line)) {
        if (const auto rp = line.rfind(')'); rp != std::string::npos && rp + 2 <= line.size()) {
            std::istringstream fields(line.substr(rp + 2));   // field 3 (state) onwards
            std::string f;
            for (int i = 3; i <= 15 && (fields >> f); ++i)
                if (i == 14 || i == 15) out += f + ',';       // utime, stime
        }
    }
    std::ifstream io("/proc/" + std::to_string(pid) + "/io");
    for (std::string line; std::getline(io, line);)
        if (line.starts_with("rchar:")) out += line.substr(6);
    return out;
}

// Query `id` until pred(view) holds — or the job settles, or it is launched more
// than `max_launches` times (so neither an early terminal bucket nor a relaunch
// loop is waited on forever). The job's own fingerprint is the heartbeat: a stuck
// job ends the wait after the stall limit (CLAUDE.md "Testing"). With
// `child_works`, the child's own CPU time and reads join the heartbeat (a real
// coast child; see child_work). Returns the last view seen; callers assert on it.
template <class Pred>
std::optional<job_view> await_job(core& c, const job_id& id, Pred pred, int max_launches = 1,
                                  bool child_works = false) {
    std::optional<job_view> last;
    (void)coast_test::await_progress(
        [&] {
            last = c.query(id).get();
            return last && (pred(*last) || is_terminal(last->b) || launches(*last) > max_launches);
        },
        [&] {
            const auto v = c.query(id).get();
            if (!v) return std::string("?");
            return child_works ? fingerprint(*v) + '|' + child_work(v->j.pid) : fingerprint(*v);
        });
    return last;
}
bool terminal_view(const job_view& v) { return is_terminal(v.b); }

} // namespace
using namespace std::chrono_literals;

namespace {
std::string repo_root() {
#ifdef COAST_TEST_REPO
    return COAST_TEST_REPO;
#else
    return ".";
#endif
}

struct scratch {
    fs::path root;
    scratch() {
        static std::atomic<int> ctr{0};
        root = fs::temp_directory_path() /
               ("coastd_e2e_" + std::to_string(::getpid()) + "_" + std::to_string(ctr.fetch_add(1)));
        fs::create_directories(root);
    }
    ~scratch() { std::error_code ec; fs::remove_all(root, ec); }
};

// Mark a pre-placed worktree as build-complete: reconcile keys spawn-readiness on
// the .coastd-ready sentinel (not the directory), so a pre-placed worktree needs
// it to be treated as runnable. Mirrors what builder::build_task writes last.
void mark_worktree_ready(const fs::path& worktree_dir) {
    std::ofstream(worktree_dir / build_ready_marker) << "ready";
}

// A stand-in `coast`: parses the argv coastd builds, emits progress lines, writes
// the --save-state file and a clean run manifest, exits 0. Lets us drive the full
// lifecycle without a 2-minute real build (the real build is the separate gate).
// Also drops the readiness sentinel so reconcile spawns instead of rebuilding.
void write_fake_coast(const fs::path& bin) {
    fs::create_directories(bin.parent_path());
    mark_worktree_ready(bin.parent_path().parent_path());  // <wt>/build/coast -> <wt>
    std::ofstream f(bin);
    f << R"SH(#!/usr/bin/env bash
n=0; save=""; manifest=""
while [ $# -gt 0 ]; do
  case "$1" in
    -n) n="$2"; shift 2;;
    --save-state) save="$2"; shift 2;;
    --emit-run-manifest) manifest="$2"; shift 2;;
    *) shift;;
  esac
done
echo "@coast/progress v1 phase=start step=0 start=0 target=$n sps=0.000 logged=0"
echo "@coast/progress v1 phase=done step=$n start=0 target=$n sps=1000.000 logged=0"
[ -n "$save" ] && echo fake-state > "$save"
[ -n "$manifest" ] && printf '{"n_steps_completed": %s, "exit_reason": "completed_n_steps"}\n' "$n" > "$manifest"
exit 0
)SH";
    f.close();
    fs::permissions(bin, fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec);
}

// A stand-in `coast` that emits ONE mid-run progress line (step=1234, well below
// target) and then BLOCKS without ever writing a checkpoint or exiting. Lets us
// observe a job that is running-but-not-yet-checkpointed: live_step should be
// visible via the log tail while steps_done stays 0. Dies on the supervisor's
// SIGTERM at shutdown.
constexpr std::uint64_t kLiveStep = 1234;
void write_fake_coast_live(const fs::path& bin) {
    fs::create_directories(bin.parent_path());
    mark_worktree_ready(bin.parent_path().parent_path());
    std::ofstream f(bin);
    f << R"SH(#!/usr/bin/env bash
n=0
while [ $# -gt 0 ]; do
  case "$1" in
    -n) n="$2"; shift 2;;
    *) shift;;
  esac
done
echo "@coast/progress v1 phase=step step=1234 start=0 target=$n sps=500.000 logged=0"
# Stay alive so the job remains 'running' and never checkpoints.
while true; do sleep 0.2; done
)SH";
    f.close();
    fs::permissions(bin, fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec);
}

// A stand-in `coast` that gets through the load and prints its start line, which a
// loaded run reports at the loaded step, then dies before completing a step. It
// leaves no checkpoint and no manifest.
void write_fake_coast_dies_at_start(const fs::path& bin, std::uint64_t step) {
    fs::create_directories(bin.parent_path());
    mark_worktree_ready(bin.parent_path().parent_path());
    std::ofstream f(bin);
    f << "#!/usr/bin/env bash\n"
      << "echo \"@coast/progress v1 phase=start step=" << step << " start=" << step
      << " target=" << step + 10 << " sps=0.000 logged=0\"\n"
      << "exit 1\n";
    f.close();
    fs::permissions(bin, fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec);
}

daemon_config cfg_at(const scratch& s) {
    daemon_config c;
    c.state_root      = s.root / "state";
    c.runs_root       = s.root / "runs";
    c.worktrees_root  = s.root / "worktrees";
    c.repo_root       = repo_root();
    c.app_config_path = s.root / "coast_app_config.json";  // boot writes it; children get --app-config
    c.scan_period     = 50ms;   // snappy for the test
    c.fetch_enabled   = false;  // never touch the network (or the dev's clone) in a unit test
    return c;
}

std::string head_sha(const std::string& repo) {
    auto r = run_capture("git", {"-C", repo, "rev-parse", "HEAD"}, {});
    REQUIRE(r.ok());
    std::string s = r.output;
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}
} // namespace

// boot() writes the daemon-owned app_config that every child is launched with
// (--app-config). It pins config_base to "<config_dir>/" so a job's app-config-
// governed outputs (observe, logs) land under its own run dir, and — being an
// explicit --app-config — shadows any ambient $COAST_APP_CONFIG.
TEST_CASE("core: boot writes the daemon app_config with config_base pinned to <config_dir>/") {
    scratch s;
    const auto cfg = cfg_at(s);
    core c(cfg);
    c.boot();

    REQUIRE(fs::exists(cfg.app_config_path));
    std::ifstream in(cfg.app_config_path);
    nlohmann::json j;
    in >> j;
    CHECK(j.at("paths").at("config_base").get<std::string>() == "<config_dir>/");
}

TEST_CASE("core: a submitted job resolves, (pre-)builds, runs, and completes to done") {
    scratch s;
    const auto cfg = cfg_at(s);
    const std::string sha = head_sha(cfg.repo_root.string());

    // Pre-place a fake coast at the worktree the resolved SHA maps to, so reconcile
    // sees the worktree on disk and goes straight to spawn (skipping the real build).
    write_fake_coast(cfg.worktrees_root / sha / "build" / "coast");

    // A minimal source config (coastd rewrites snapshot_dir to an absolute path).
    const auto src_cfg = s.root / "src_config.json";
    { std::ofstream f(src_cfg); f << R"({"note":"e2e"})"; }

    core c(cfg);
    c.boot();
    core_runner loop(c);

    job_spec spec;
    spec.ref_symbolic = "HEAD";
    spec.config_path  = src_cfg;
    spec.label        = "e2e";
    spec.steps_target = 4000;
    const job_id id = c.submit(std::move(spec)).get();
    CHECK_FALSE(id.empty());

    // Wait until the job reaches a terminal bucket.
    const auto fv = await_job(c, id, terminal_view);
    REQUIRE(fv.has_value());
    const bucket final_bucket = fv->b;
    const job    final_job    = fv->j;

    loop.stop_join();

    CHECK(final_bucket == bucket::done);          // ran to completion
    CHECK(final_job.ref_sha == sha);              // ref was resolved and frozen
    CHECK(final_job.steps_done == 4000);          // manifest completed steps recorded
    CHECK(final_job.failure_reason.empty());
    // The run dir and its final state exist under runs_root.
    CHECK(fs::exists(fs::path(final_job.run_dir) / "final_state.h5"));
    CHECK(fs::exists(fs::path(final_job.run_dir) / "config.json"));
}

// Requirement: coastd surfaces the LIVE step it already parses from each running
// child's log (diag/progress.h) via list/query, so a running job advances before
// its first checkpoint. This is the acceptance test whose absence let the wire go
// unconnected (progress_of() had no caller): it fails if apply_live_step() is not
// invoked, since live_step would stay 0 forever.
TEST_CASE("core: list/query report the live tailed step of a running, un-checkpointed job") {
    scratch s;
    const auto cfg = cfg_at(s);
    const std::string sha = head_sha(cfg.repo_root.string());

    // Fake coast emits step=1234 then blocks WITHOUT checkpointing.
    write_fake_coast_live(cfg.worktrees_root / sha / "build" / "coast");

    const auto src_cfg = s.root / "src_config.json";
    { std::ofstream f(src_cfg); f << R"({"note":"live"})"; }

    core c(cfg);
    c.boot();
    core_runner loop(c);

    job_spec spec;
    spec.ref_symbolic = "HEAD";
    spec.config_path  = src_cfg;
    spec.steps_target = 4000;               // 1234 < 4000 -> the job is NOT done
    const job_id id = c.submit(std::move(spec)).get();
    CHECK_FALSE(id.empty());

    // Wait until the live step reaches query() (running child, progress tailed).
    std::optional<job_view> v = await_job(c, id, [](const job_view& q) { return q.live_step > 0; });
    if (v && v->live_step == 0) v.reset();

    loop.stop_join();

    REQUIRE(v.has_value());                 // the parser's output actually reached query
    CHECK(v->live_step == kLiveStep);       // the exact tailed step
    CHECK(v->j.steps_done == 0);            // durable step untouched — no checkpoint written
    CHECK(v->b == bucket::active);          // still running, not terminal
}

// ---- the real-coast gates ---------------------------------------------------
// Both cases below carry doctest::skip(): a plain `ctest` run of this binary does
// NOT execute them, and — the point — does not REPORT them as passing either.
// They run only under tests/unit/coastd/run_real_gate.sh (ctest name
// gate.coastd.real_coast), which passes --no-skip and supplies the environment.
// If that environment is missing here, the case FAILS rather than returning
// quietly: a gate that skips itself into a green tick is worse than no gate,
// because the suite's pass count then reads as coverage it does not have.
namespace {
struct gate_env { std::string coast, config; };

gate_env require_gate_env() {
    const char* real = std::getenv("COASTD_REAL_COAST");
    const char* cfgp = std::getenv("COASTD_REAL_CONFIG");
    REQUIRE_MESSAGE(real != nullptr,
                    "COASTD_REAL_COAST unset — run this case via run_real_gate.sh, "
                    "not directly");
    REQUIRE_MESSAGE(cfgp != nullptr, "COASTD_REAL_CONFIG unset — see run_real_gate.sh");
    return {real, cfgp};
}

} // namespace

// The M2 gate: a job run THROUGH coastd (real coast, pre-placed to skip the build)
// produces sim state bit-identical to a direct coast run of the same config/seed/
// steps.
TEST_CASE("core: coastd run is bit-identical to a direct coast run (real coast; opt-in)"
          * doctest::skip()) {
    const auto env  = require_gate_env();
    const char* real = env.coast.c_str();
    const char* cfgp = env.config.c_str();
    const std::uint64_t steps = 4000;

    scratch s;
    auto cfg = cfg_at(s);
    const std::string sha = head_sha(cfg.repo_root.string());

    // Pre-place the real coast where the resolved SHA maps (skip the 2-min build;
    // the build sequence itself is validated by spike 02 + the builder unit test).
    const auto bin = cfg.worktrees_root / sha / "build" / "coast";
    fs::create_directories(bin.parent_path());
    fs::create_symlink(real, bin);
    mark_worktree_ready(cfg.worktrees_root / sha);

    core c(cfg);
    c.boot();
    core_runner loop(c);

    job_spec spec;
    spec.ref_symbolic = "HEAD";
    spec.config_path  = cfgp;
    spec.steps_target = steps;
    const job_id id = c.submit(std::move(spec)).get();

    // A real coast: its own CPU time and reads are the heartbeat while it runs.
    const auto fv = await_job(c, id, terminal_view, /*max_launches=*/1, /*child_works=*/true);
    REQUIRE(fv.has_value());
    const bucket b         = fv->b;
    const job    final_job = fv->j;
    loop.stop_join();

    REQUIRE_MESSAGE(b == bucket::done, "job ended in bucket " << bucket_dir(b)
                    << " (failure_reason: '"
                    << final_job.failure_reason << "')");
    const fs::path coastd_state = fs::path(final_job.run_dir) / "final_state.h5";
    REQUIRE(fs::exists(coastd_state));

    // Direct reference run of the same coast/config/steps.
    const fs::path ref = s.root / "ref_state.h5";
    auto rr = run_capture(real, {"--headless", "-c", cfgp, "-n", std::to_string(steps),
                                 "--save-state", ref.string()}, {});
    REQUIRE(rr.ok());
    REQUIRE(fs::exists(ref));

    // Compare sim state, excluding the /metadata group (wall-clock timestamp +
    // config path legitimately differ between the two runs).
    auto diff = run_capture("h5diff", {"--exclude-path", "/metadata",
                                       coastd_state.string(), ref.string()}, {});
    if (!diff.spawned) { MESSAGE("skipped: h5diff not available"); return; }
    INFO("h5diff output:\n" << diff.output);
    CHECK(diff.exit_code == 0);  // 0 = identical
}

TEST_CASE("core: cancelling a ready (unslotted) job moves it to cancelled") {
    scratch s;
    auto cfg = cfg_at(s);
    cfg.slots = 0;   // the job resolves + finds its worktree but never gets a slot -> stays ready
    const std::string sha = head_sha(cfg.repo_root.string());
    write_fake_coast(cfg.worktrees_root / sha / "build" / "coast");

    core c(cfg);
    c.boot();
    core_runner loop(c);

    job_spec spec;
    spec.ref_symbolic = "HEAD";
    spec.config_path  = (s.root / "cfg.json");
    { std::ofstream f(spec.config_path); f << R"({"note":"cancel"})"; }
    spec.steps_target = 100;
    const job_id id = c.submit(std::move(spec)).get();

    // Wait until the ref is resolved (so we know it is genuinely sitting ready,
    // not merely un-processed), then cancel.
    const auto ready = await_job(c, id, [](const job_view& v) { return !v.j.ref_sha.empty(); });
    REQUIRE(ready.has_value());
    REQUIRE_FALSE(ready->j.ref_sha.empty());   // ready for certain, not merely un-processed
    CHECK(c.cancel(id).get());

    const auto fv = await_job(c, id, terminal_view);
    REQUIRE(fv.has_value());
    const bucket b = fv->b;
    loop.stop_join();
    CHECK(b == bucket::cancelled);
}

// A job submitted with load_state_from whose final state is gone by its first launch must
// fail visibly. Starting it fresh instead would run a different experiment under
// the same job; letting coast fail on the missing file would burn every launch
// strike and file it under a misleading crash-loop reason.
TEST_CASE("core: a load_state_from final state that vanished before the first launch fails the job, never starts fresh") {
    scratch s;
    const auto cfg = cfg_at(s);
    const std::string sha = head_sha(cfg.repo_root.string());
    write_fake_coast(cfg.worktrees_root / sha / "build" / "coast");

    core c(cfg);
    c.boot();
    core_runner loop(c);

    job_spec spec;
    spec.ref_symbolic    = "HEAD";
    spec.config_path     = s.root / "cfg.json";
    { std::ofstream f(spec.config_path); f << R"({"note":"gone"})"; }
    spec.steps_target    = 10;
    // As the comms edge would hand it over, but the file has since disappeared.
    spec.load_state_from = "train";
    spec.load_state      = (cfg.runs_root / "train" / "final_state.h5").string();
    spec.load_state_step = 500;
    const job_id id = c.submit(std::move(spec)).get();

    const auto fv = await_job(c, id, terminal_view);
    REQUIRE(fv.has_value());
    const bucket b = fv->b;
    const job    j = fv->j;
    loop.stop_join();

    CHECK(b == bucket::failed);
    CHECK(j.failure_reason.find("load_state_from train") != std::string::npos);
    CHECK(j.steps_target == 510);   // stored in coast's step space: 500 + 10
    // Never launched: the fake coast would have left a log and a final state.
    CHECK(fs::is_empty(fs::path(j.run_dir) / "logs"));
    CHECK_FALSE(fs::exists(fs::path(j.run_dir) / "final_state.h5"));
}

// An unbounded job (steps_target 0) stays unbounded when it starts from another
// job's state. do_submit must keep 0 rather than add load_state_step to it: the sum
// would make it a bounded job already at its target, i.e. -n 1 after the clamp.
TEST_CASE("core: an unbounded load_state_from job stays --unlimited") {
    scratch s;
    auto cfg = cfg_at(s);
    cfg.slots = 0;   // never launched; only the recorded job and its plan matter
    core c(cfg);
    c.boot();
    core_runner loop(c);

    job_spec spec;
    spec.ref_symbolic    = "HEAD";
    spec.config_path     = s.root / "cfg.json";
    { std::ofstream f(spec.config_path); f << R"({"note":"unbounded"})"; }
    spec.steps_target    = 0;
    spec.load_state_from = "train";
    spec.load_state      = (s.root / "final_state.h5").string();
    spec.load_state_step = 50000;
    const job_id id = c.submit(std::move(spec)).get();
    const auto v = c.query(id).get();
    loop.stop_join();

    REQUIRE(v.has_value());
    CHECK(v->j.steps_target == 0);
    const launch_plan p = plan_launch(v->j, cfg, "/bin/coast");
    CHECK(std::find(p.args.begin(), p.args.end(), "--unlimited") != p.args.end());
    CHECK(std::find(p.args.begin(), p.args.end(), "-n") == p.args.end());
}

// coast reports a loaded run's start at the loaded step, so a job's steps_done must
// start there too. From 0, that start line reads as steps run: a child that died
// without completing a step would be filed as "ran real steps but checkpointed
// nothing" instead of a crash-loop, and list would show 0/510.
TEST_CASE("core: a load_state_from job that dies at its start line is a crash-loop strike") {
    scratch s;
    auto cfg = cfg_at(s);
    cfg.max_launch_failures = 1;   // the first strike files the job, naming its counter
    const std::string sha = head_sha(cfg.repo_root.string());
    write_fake_coast_dies_at_start(cfg.worktrees_root / sha / "build" / "coast", 500);

    core c(cfg);
    c.boot();
    core_runner loop(c);

    job_spec spec;
    spec.ref_symbolic    = "HEAD";
    spec.config_path     = s.root / "cfg.json";
    { std::ofstream f(spec.config_path); f << R"({"note":"dies"})"; }
    spec.steps_target    = 10;
    spec.load_state_from = "train";
    spec.load_state      = (s.root / "final_state.h5").string();
    { std::ofstream f(spec.load_state); f << "fake-state"; }
    spec.load_state_step = 500;
    const job_id id = c.submit(std::move(spec)).get();

    const auto fv = await_job(c, id, terminal_view);
    REQUIRE(fv.has_value());
    const bucket b = fv->b;
    const job    j = fv->j;
    loop.stop_join();

    CHECK(b == bucket::failed);
    CHECK(j.steps_done == 500);
    CHECK(j.launch_failures == 1);
    CHECK(j.no_checkpoint_launches == 0);
    CHECK(j.failure_reason.find("crash-loop") != std::string::npos);
}
