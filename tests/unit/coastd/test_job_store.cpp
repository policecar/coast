#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include "daemon/job_store.h"
#include "daemon/proc_util.h"   // run_capture — to observe the async committer's git output

#include "await_progress.h"

using namespace coastd;
namespace fs = std::filesystem;

namespace {

// A unique scratch directory that cleans itself up. Uses pid + a counter (no
// randomness needed) so parallel ctest runs of this binary do not collide.
struct scratch {
    fs::path root;
    scratch() {
        static std::atomic<int> ctr{0};
        root = fs::temp_directory_path() /
               ("coastd_store_" + std::to_string(::getpid()) + "_" +
                std::to_string(ctr.fetch_add(1)));
        fs::create_directories(root);
    }
    ~scratch() {
        std::error_code ec;
        fs::remove_all(root, ec);
    }
};

daemon_config cfg_at(const fs::path& state_root) {
    daemon_config c;
    c.state_root = state_root;
    return c;
}

job mk(std::string id) {
    job j;
    j.id           = std::move(id);
    j.submitted_at = "2026-07-20T10:00:00Z";
    j.ref_symbolic = "feat/x";
    j.ref_sha      = "sha_a";
    j.steps_target = 1000;
    return j;
}

} // namespace

TEST_CASE("job_store: commit is async and the committer eventually lands it in git") {
    scratch s;
    auto cfg = cfg_at(s.root);
    job_store store(cfg);
    store.init_store();   // git init + bucket dirs

    // Give the scratch repo a git identity so the commit can succeed regardless of
    // the host's global git config (all local, no network).
    const std::string sr = cfg.state_root.string();
    REQUIRE(run_capture("git", {"-C", sr, "config", "user.email", "t@example.com"}).ok());
    REQUIRE(run_capture("git", {"-C", sr, "config", "user.name",  "coastd-test"}).ok());

    store.put(mk("j1"));
    store.commit("submit j1");   // returns immediately; the committer thread does the git work

    // The commit lands asynchronously, by external git (not a process we control:
    // a generous bound is legitimate, CLAUDE.md "Testing"; never a perf budget).
    CHECK(coast_test::await_condition(
        [&] {
            auto r = run_capture("git", {"-C", sr, "log", "--oneline"}, {});
            return r.ok() && r.output.find("submit j1") != std::string::npos;
        },
        std::chrono::minutes(5)));
}

TEST_CASE("job_store: a burst of commits coalesces and drains without hanging") {
    scratch s;
    auto cfg = cfg_at(s.root);
    job_store store(cfg);
    store.init_store();
    const std::string sr = cfg.state_root.string();
    REQUIRE(run_capture("git", {"-C", sr, "config", "user.email", "t@example.com"}).ok());
    REQUIRE(run_capture("git", {"-C", sr, "config", "user.name",  "coastd-test"}).ok());

    // Fire many commits back-to-back; the committer coalesces (one `git add -A`
    // captures the whole tree) so they collapse into far fewer real commits.
    for (int i = 0; i < 20; ++i) {
        store.put(mk("job" + std::to_string(i)));
        store.commit("change " + std::to_string(i));
    }
    // At least one commit must land; the exact count is nondeterministic by design
    // (coalescing). The store destructor must also join the committer cleanly (no
    // hang) at end of scope — the real assertion is that this test returns.
    CHECK(coast_test::await_condition(
        [&] {
            auto r = run_capture("git", {"-C", sr, "log", "--oneline"}, {});
            return r.ok() && !r.output.empty();
        },
        std::chrono::minutes(5)));
}

TEST_CASE("job_store: try_put updates a known job but never creates a phantom") {
    scratch s;
    job_store store(cfg_at(s.root));
    store.init_store();

    // Unknown id: no durable record exists, so try_put must NOT create one (that
    // would be a phantom job the daemon acts on with nothing on disk).
    CHECK_FALSE(store.try_put(mk("ghost")));
    CHECK_FALSE(store.get("ghost").has_value());
    CHECK(store.size() == 0);

    // Known id: durable write succeeds, cache reflects the update, returns true.
    store.put(mk("j1"));
    job updated = mk("j1");
    updated.pid = 0;                         // the child-exit clear this exists for
    updated.status = job_status::ready;
    CHECK(store.try_put(updated));
    auto got = store.get("j1");
    REQUIRE(got.has_value());
    CHECK(got->status == job_status::ready);
    CHECK(got->pid == 0);
}

TEST_CASE("job_store: all_jobs returns a defined order (active first, scheduler order within)") {
    scratch s;
    job_store store(cfg_at(s.root));
    store.init_store();

    auto with = [](std::string id, int prio, std::string at) {
        job j = mk(std::move(id));
        j.priority = prio;
        j.submitted_at = std::move(at);
        return j;
    };
    // Insert deliberately out of order.
    store.put(with("lo",   0, "2026-07-20T10:00:00Z"));
    store.put(with("hi",   5, "2026-07-20T10:00:05Z"));
    store.put(with("mid1", 2, "2026-07-20T10:00:02Z"));  // same prio, earlier -> first
    store.put(with("mid2", 2, "2026-07-20T10:00:09Z"));  // same prio, later  -> second

    auto all = store.all_jobs();
    REQUIRE(all.size() == 4);
    // priority desc, then submitted_at asc (FIFO) — the scheduler's own order.
    CHECK(all[0].first.id == "hi");
    CHECK(all[1].first.id == "mid1");
    CHECK(all[2].first.id == "mid2");
    CHECK(all[3].first.id == "lo");
}

TEST_CASE("job_store: init creates bucket dirs; put/get/active round-trip") {
    scratch s;
    job_store store(cfg_at(s.root));
    store.init_store();

    CHECK(fs::is_directory(s.root / "jobs" / "active"));
    CHECK(fs::is_directory(s.root / "jobs" / "quarantine"));

    store.put(mk("j1"));
    store.put(mk("j2"));

    CHECK(store.size() == 2);
    CHECK(store.active_jobs().size() == 2);
    REQUIRE(store.bucket_of("j1").has_value());
    CHECK(store.bucket_of("j1").value() == bucket::active);
    REQUIRE(store.get("j1").has_value());
    CHECK(store.get("j1")->ref_sha == "sha_a");

    // The file is really on disk in the active bucket.
    CHECK(fs::exists(s.root / "jobs" / "active" / "j1.json"));
}

TEST_CASE("job_store: move_bucket relocates the file atomically, no content change") {
    scratch s;
    job_store store(cfg_at(s.root));
    store.init_store();
    store.put(mk("j1"));

    store.move_bucket("j1", bucket::done);

    CHECK_FALSE(fs::exists(s.root / "jobs" / "active" / "j1.json"));
    CHECK(fs::exists(s.root / "jobs" / "done" / "j1.json"));
    CHECK(store.bucket_of("j1").value() == bucket::done);
    CHECK(store.active_jobs().empty());  // no longer active
}

TEST_CASE("job_store: state survives a reload from disk") {
    scratch s;
    {
        job_store store(cfg_at(s.root));
        store.init_store();
        store.put(mk("alpha"));
        store.put(mk("beta"));
        store.move_bucket("beta", bucket::failed);
    }
    // Fresh store, cold load.
    job_store reloaded(cfg_at(s.root));
    reloaded.load_all();
    CHECK(reloaded.size() == 2);
    CHECK(reloaded.bucket_of("alpha").value() == bucket::active);
    CHECK(reloaded.bucket_of("beta").value()  == bucket::failed);
    CHECK(reloaded.active_jobs().size() == 1);
}

TEST_CASE("job_store: an unparseable file is quarantined, never crashing load") {
    scratch s;
    job_store store(cfg_at(s.root));
    store.init_store();
    store.put(mk("good"));

    // Drop a corrupt file directly into the active bucket.
    {
        std::ofstream bad(s.root / "jobs" / "active" / "corrupt.json");
        bad << "{ not valid json ";
    }

    store.load_all();  // must not throw

    // The good job loads; the corrupt file is moved out of active into quarantine.
    CHECK(store.bucket_of("good").value() == bucket::active);
    CHECK_FALSE(fs::exists(s.root / "jobs" / "active" / "corrupt.json"));
    CHECK(fs::exists(s.root / "jobs" / "quarantine" / "corrupt.json"));
}

TEST_CASE("job_store: a newer-schema file is quarantined untouched") {
    scratch s;
    job_store store(cfg_at(s.root));
    store.init_store();
    {
        std::ofstream f(s.root / "jobs" / "active" / "future.json");
        f << R"({"schema_version": 9999, "id": "future"})";
    }
    store.load_all();
    CHECK(fs::exists(s.root / "jobs" / "quarantine" / "future.json"));
    // Content preserved (not mangled): still declares the future version.
    std::ifstream in(s.root / "jobs" / "quarantine" / "future.json");
    std::string body((std::istreambuf_iterator<char>(in)), {});
    CHECK(body.find("9999") != std::string::npos);
}

TEST_CASE("job_store: sweep_stale_tmp removes leftover *.tmp, leaves *.json untouched") {
    scratch s;
    job_store store(cfg_at(s.root));
    store.init_store();
    store.put(mk("keep"));  // a real job file (keep/.json)

    // Simulate a crash mid-atomic_write: a stale tmp beside the job file, and one
    // in another bucket for good measure.
    { std::ofstream f(s.root / "jobs" / "active" / "keep.json.tmp"); f << "partial"; }
    { std::ofstream f(s.root / "jobs" / "done"   / "orphan.json.tmp"); f << "partial"; }

    const std::size_t removed = store.sweep_stale_tmp();

    CHECK(removed == 2);
    CHECK_FALSE(fs::exists(s.root / "jobs" / "active" / "keep.json.tmp"));
    CHECK_FALSE(fs::exists(s.root / "jobs" / "done"   / "orphan.json.tmp"));
    CHECK(fs::exists(s.root / "jobs" / "active" / "keep.json"));  // real file survives
    CHECK(store.get("keep").has_value());

    CHECK(store.sweep_stale_tmp() == 0);  // idempotent
}

TEST_CASE("job_store: commit is best-effort and never throws") {
    scratch s;
    job_store store(cfg_at(s.root));
    store.init_store();      // git init (best-effort)
    store.put(mk("j1"));
    CHECK_NOTHROW(store.commit("test: add j1"));
}
