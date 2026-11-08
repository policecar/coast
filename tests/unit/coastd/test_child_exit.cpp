// -----------------------------------------------------------------------------
// coastd on_child_exited event-ordering folds — the two guards that only fire on
// a specific interleaving of the single-writer event queue, unreachable through
// the public RPC surface without racing a real child against the loop.
//
//   M2: a stale child_exited applied AFTER the job was already moved to a terminal
//       bucket (a cancel/failure that landed ahead of it in the queue) must fold
//       only the monotonic pid/steps/checkpoint bookkeeping — never re-stamp an
//       active-phase status or a crash-loop strike into a terminal bucket.
//   L6: a child that exited because the daemon SIGTERM'd it for a CANCEL (still in
//       the active bucket, no durable progress) must NOT be charged a crash-loop
//       strike — a strike would set a "crash-loop" failure_reason and (reconcile
//       checks failure before cancel) misfile the job to failed/ instead of
//       cancelled/.
//
// The friend seam core_test_access applies a synthetic event SYNCHRONOUSLY on this
// thread with the core loop NOT running, so the store has exactly one writer and
// the interleaving is deterministic — no sleeps, no real subprocess.
// -----------------------------------------------------------------------------
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <unistd.h>

#include <atomic>
#include <string>

#include "daemon/core.h"

using namespace coastd;
namespace fs = std::filesystem;

// The test-only definition of the friend declared in core.h.
namespace coastd {
struct core_test_access {
    static void       apply(core& c, event e) { c.apply(e); }
    static job_store& store(core& c)          { return c.store_; }
};
} // namespace coastd

namespace {

struct scratch {
    fs::path root;
    scratch() {
        static std::atomic<int> ctr{0};
        root = fs::temp_directory_path() /
               ("coastd_cexit_" + std::to_string(::getpid()) + "_" + std::to_string(ctr.fetch_add(1)));
        fs::create_directories(root);
    }
    ~scratch() { std::error_code ec; fs::remove_all(root, ec); }
};

daemon_config cfg_at(const scratch& s) {
    daemon_config c;
    c.state_root     = s.root / "state";
    c.runs_root      = s.root / "runs";
    c.worktrees_root = s.root / "worktrees";
    c.repo_root      = ".";
    return c;
}

// A job seeded directly into the store with the fields these guards depend on.
job seed_job(const daemon_config& cfg, const std::string& id) {
    job j;
    j.id           = id;
    j.ref_sha      = "abc1234";
    j.run_dir      = (cfg.runs_root / id).string();
    j.steps_target = 100;   // bounded, so a no-progress exit is "not done"
    j.pid          = 4242;
    j.start_token  = "tok";
    return j;
}

} // namespace

// M2 — a stale child_exited on a job already in cancelled/ folds only bookkeeping.
TEST_CASE("on_child_exited: terminal-bucket job is not re-stamped (M2)") {
    scratch s;
    const auto cfg = cfg_at(s);
    core c(cfg);
    c.boot();
    auto& store = core_test_access::store(c);

    // A job that was cancelled and already moved to cancelled/: status cancelled,
    // and — critically — one strike short of the crash-loop cap, so an erroneous
    // strike here WOULD trip a terminal "crash-loop" failure_reason (making the bug
    // observable). It has some durable progress on record.
    job j = seed_job(cfg, "m2job");
    j.status          = job_status::cancelled;
    j.launch_failures = cfg.max_launch_failures - 1;
    j.steps_done      = 10;
    j.checkpoint      = "snapshots/state_10.h5";
    store.put(j);                              // lands in active/
    store.move_bucket(j.id, bucket::cancelled);// now terminal, like a real cancel move

    // A stale child_exited that carries NEWER durable progress but NO strike-worthy
    // signal beyond "no forward progress since prev" (steps_done here == recorded).
    child_exited ce;
    ce.id          = "m2job";
    ce.steps_done  = 20;                        // > recorded 10: must fold monotonically
    ce.checkpoint  = "snapshots/state_20.h5";   // newer: must fold
    ce.tailed_step = 0;
    ce.fatal       = false;
    core_test_access::apply(c, ce);

    const auto b = store.bucket_of("m2job");
    const auto after = store.get("m2job");
    REQUIRE(b.has_value());
    REQUIRE(after.has_value());

    // The guard held: still terminal, status untouched, no strike, no failure_reason.
    CHECK(*b == bucket::cancelled);
    CHECK(after->status == job_status::cancelled);        // NOT re-stamped to ready
    CHECK(after->launch_failures == cfg.max_launch_failures - 1);  // NO strike added
    CHECK(after->failure_reason.empty());                  // NOT relabelled crash-loop
    // ...but the monotonic bookkeeping WAS folded (proves we didn't just skip all).
    CHECK(after->steps_done == 20);
    CHECK(after->checkpoint == "snapshots/state_20.h5");
    CHECK(after->pid == 0);                                // stale identity cleared
    CHECK(after->start_token.empty());
}

// L6 — a cancel-induced exit (active bucket) charges no crash-loop strike.
TEST_CASE("on_child_exited: cancel-induced exit is not a crash-loop strike (L6)") {
    scratch s;
    const auto cfg = cfg_at(s);
    core c(cfg);
    c.boot();
    auto& store = core_test_access::store(c);

    // A running job that has a cancel pending, one strike short of the cap, and made
    // no durable progress this launch. Without the cancel guard the no-progress
    // branch would strike -> hit the cap -> set a "crash-loop" failure_reason.
    job j = seed_job(cfg, "l6job");
    j.status           = job_status::running;
    j.cancel_requested = true;
    j.launch_failures  = cfg.max_launch_failures - 1;
    j.steps_done       = 0;
    store.put(j);                              // stays in active/

    child_exited ce;
    ce.id          = "l6job";
    ce.steps_done  = 0;      // no durable progress
    ce.tailed_step = 0;      // and no real work seen on the tail
    ce.fatal       = false;
    core_test_access::apply(c, ce);

    const auto b = store.bucket_of("l6job");
    const auto after = store.get("l6job");
    REQUIRE(b.has_value());
    REQUIRE(after.has_value());

    // Still active (reconcile hasn't run to move it), NO strike, NO failure_reason:
    // the pending cancel will file it to cancelled/, not failed/.
    CHECK(*b == bucket::active);
    CHECK(after->launch_failures == cfg.max_launch_failures - 1);  // strike NOT charged
    CHECK(after->failure_reason.empty());                          // NOT crash-loop-failed
    CHECK(after->cancel_requested);                                // cancel still pending
    CHECK(after->pid == 0);                                        // child identity cleared
}
