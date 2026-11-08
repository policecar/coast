#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <thread>

#include "daemon/child_proc.h"   // read_start_token / is_same_process
#include "daemon/core.h"
#include "daemon/job_store.h"
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
using namespace std::chrono_literals;

extern char** environ;

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
               ("coastd_rec_" + std::to_string(::getpid()) + "_" + std::to_string(ctr.fetch_add(1)));
        fs::create_directories(root);
    }
    ~scratch() { std::error_code ec; fs::remove_all(root, ec); }
};

daemon_config cfg_at(const scratch& s) {
    daemon_config c;
    c.state_root     = s.root / "state";
    c.runs_root      = s.root / "runs";
    c.worktrees_root = s.root / "worktrees";
    c.repo_root      = repo_root();
    c.scan_period    = 50ms;
    c.fetch_enabled  = false;   // never touch the network (or the dev's clone) in a unit test
    return c;
}

std::string head_sha(const std::string& repo) {
    auto r = run_capture("git", {"-C", repo, "rev-parse", "HEAD"}, {});
    REQUIRE(r.ok());
    std::string s = r.output;
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

// A fake `coast` that ALWAYS crashes: consumes no args, writes no --save-state,
// no checkpoint, no run manifest, and exits non-zero. Every launch therefore makes
// zero forward progress -> the crash-loop cap must eventually fail the job.
void write_crashing_coast(const fs::path& bin) {
    fs::create_directories(bin.parent_path());
    { std::ofstream f(bin); f << "#!/usr/bin/env bash\nexit 1\n"; }
    fs::permissions(bin, fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec);
    // Reconcile keys spawn-readiness on the .coastd-ready sentinel, not the
    // worktree directory — drop it so the pre-placed tree is treated as built.
    std::ofstream(bin.parent_path().parent_path() / build_ready_marker) << "ready";
}

// Spawn a real, long-lived process to stand in for an orphaned child that survived
// a previous daemon (the off-systemd case R1's pid-guard must handle). Returns its pid.
pid_t spawn_orphan() {
    const char* argv[] = {"sleep", "30", nullptr};
    pid_t pid = 0;
    const int rc = posix_spawnp(&pid, "sleep", nullptr, nullptr,
                                const_cast<char* const*>(argv), environ);
    REQUIRE(rc == 0);
    return pid;
}

job mk_running(const std::string& id, const scratch& s, pid_t pid, const std::string& token) {
    job j;
    j.id           = id;
    j.submitted_at = "2026-07-20T10:00:00Z";
    j.ref_symbolic = "HEAD";
    j.ref_sha      = "sha_orphan";
    j.run_dir      = (s.root / "runs" / id).string();
    j.steps_target = 100000;
    j.status       = job_status::running;
    j.pid          = pid;
    j.start_token  = token;
    return j;
}

} // namespace

// ---------------------------------------------------------------------------
// pid duplicate-guard: an orphan that outlived the previous daemon must be killed
// at boot and its stale identity cleared, so reconcile never puts a second child
// on the same run dir (the R1 duplicate-run nightmare).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// PLATFORM GATE. The cases below need coastd's Linux-only pid-guard: either
// read_start_token() directly, or a running daemon — and the daemon REFUSES to
// start where start_token_supported() is false (main.cpp), because without a
// recycle-safe start token the duplicate-run guard is inoperative and silent
// corruption is on the table (ADR-16). Off Linux they cannot pass, so they skip
// rather than hard-fail: macOS is a first-class dev platform here and a Mac dev
// running the documented `ctest` must not be handed red that means "wrong OS".
// Gated on the seam, not on __linux__, so porting read_start_token() re-enables
// them automatically — the same predicate the daemon's own startup gate asks.
// ---------------------------------------------------------------------------
constexpr bool skip_off_linux = !coastd::start_token_supported();

TEST_CASE("recovery: boot kills a surviving orphan child and clears its identity" * doctest::skip(skip_off_linux)) {
    scratch s;
    const auto cfg = cfg_at(s);

    const pid_t orphan = spawn_orphan();
    const auto tok = read_start_token(orphan);
    if (!tok) {  // no /proc (non-Linux): the token path is unavailable, skip the kill assertion.
        ::kill(orphan, SIGKILL);
        int st = 0; ::waitpid(orphan, &st, 0);
        // doctest has no first-class skip-at-runtime, and MESSAGE alone leaves the
        // case reporting PASSED with zero assertions — indistinguishable from a
        // real pass in the summary. Fail explicitly instead: this suite's own
        // CMake target is Linux-gated, so reaching here means the gate is wrong,
        // which is exactly what should be loud.
        FAIL("read_start_token unavailable on this platform — the pid-guard test "
             "cannot run here, and reporting a pass would misstate coverage");
        return;
    }
    REQUIRE(is_same_process(orphan, *tok));  // sanity: our stand-in is genuinely live

    const std::string id = "20260720T100000Z_orphan_abc123";
    {
        job_store store(cfg);
        store.init_store();
        store.put(mk_running(id, s, orphan, *tok));  // persist "job X is running as pid <orphan>"
    }

    // Boot a fresh daemon over that state. recover_at_boot() must kill the orphan.
    core c(cfg);
    c.boot();

    // We parented the orphan, so reap the zombie the SIGKILL produced and confirm
    // the signal was actually delivered.
    int st = 0;
    const pid_t w = ::waitpid(orphan, &st, 0);
    CHECK(w == orphan);
    CHECK(WIFSIGNALED(st));
    CHECK(WTERMSIG(st) == SIGKILL);

    // The stale pid/start_token were cleared on disk, so the next reconcile treats
    // the job as not-running and relaunches it cleanly from its checkpoint.
    job_store reloaded(cfg);
    reloaded.load_all();
    auto jj = reloaded.get(id);
    REQUIRE(jj.has_value());
    CHECK(jj->pid == 0);
    CHECK(jj->start_token.empty());
}

// ---------------------------------------------------------------------------
// crash-loop cap: a child that keeps dying with no forward progress must be failed
// after cfg.max_launch_failures relaunches, not relaunched forever.
// ---------------------------------------------------------------------------
TEST_CASE("recovery: a job that crashes with no progress is failed after the cap" * doctest::skip(skip_off_linux)) {
    scratch s;
    auto cfg = cfg_at(s);
    cfg.slots               = 1;
    cfg.max_launch_failures = 3;
    // Shrink the relaunch backoff. The subject here is the CAP -- that a
    // no-progress child is failed after max_launch_failures rather than
    // relaunched forever -- not the wall-clock spacing between attempts. With the
    // 2 s default the three attempts spend 2 s + 4 s = 6 s of pure backoff inside
    // a 10 s poll budget: a ~40% margin that turns into a spurious failure under
    // a sanitizer build or a loaded machine. 1 s (the field's resolution) plus a
    // doubled poll budget keeps the ordering identical -- backoff still doubles and
    // still gates each relaunch -- while taking the deterministic cost to 1 + 2 = 3 s
    // against 20 s.
    cfg.relaunch_backoff_base = std::chrono::seconds(1);
    const std::string sha = head_sha(cfg.repo_root.string());

    // Pre-place a crashing coast at the resolved-SHA worktree so reconcile goes
    // straight to spawn (skipping the real build) and the child dies every launch.
    write_crashing_coast(cfg.worktrees_root / sha / "build" / "coast");

    const auto src_cfg = s.root / "src_config.json";
    { std::ofstream f(src_cfg); f << R"({"note":"crashloop"})"; }

    core c(cfg);
    c.boot();
    core_runner loop(c);

    job_spec spec;
    spec.ref_symbolic = "HEAD";
    spec.config_path  = src_cfg;
    spec.label        = "crashloop";
    spec.steps_target = 4000;
    const job_id id = c.submit(std::move(spec)).get();

    // Wait on the job's own progress, not a deadline: the 3 s of backoff inside
    // are the core's schedule, and a loaded box only stretches it.
    // A missing cap would relaunch forever with the heartbeat still ticking, so
    // exceeding the cap ends the wait as well (and fails below).
    std::optional<job_view> last;
    REQUIRE(coast_test::await_progress(
        [&] {
            last = c.query(id).get();
            return last && (is_terminal(last->b) ||
                            last->j.launch_failures > cfg.max_launch_failures);
        },
        // The job's own state is the heartbeat (each relaunch changes it; the cap
        // bounds the loop above) — core::passes would tick on its timer even for a
        // job that is stuck.
        [&] {
            const auto v = c.query(id).get();
            return v ? std::to_string(static_cast<int>(v->b)) + '|' +
                           std::to_string(static_cast<int>(v->j.status)) + '|' +
                           std::to_string(v->j.launch_failures) + '|' +
                           std::to_string(v->j.last_launch_ms)
                     : std::string("?");
        }));
    const bucket b         = last->b;
    const job    final_job = last->j;
    loop.stop_join();

    CHECK(b == bucket::failed);                       // capped, not relaunched forever
    CHECK(final_job.launch_failures == 3);            // exactly the configured cap
    CHECK(final_job.failure_reason.find("crash-loop") != std::string::npos);
}
