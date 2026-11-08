#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <atomic>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "daemon/builder.h"
#include "daemon/event_queue.h"
#include "daemon/proc_util.h"
#include "daemon/worker_pool.h"

using namespace coastd;
namespace fs = std::filesystem;

namespace {
// The builder resolves refs against a real repo. COAST_TEST_REPO is set by CTest
// to the source tree; fall back to CMAKE-provided define.
std::string repo_root() {
#ifdef COAST_TEST_REPO
    return COAST_TEST_REPO;
#else
    const char* r = std::getenv("COAST_TEST_REPO");
    return r ? r : ".";
#endif
}

daemon_config cfg() {
    daemon_config c;
    c.repo_root      = repo_root();
    c.worktrees_root = "/tmp/coastd_builder_wt_unused";  // not touched by resolve
    // Local-only resolution: these cases assert the verbatim rev-parse path, and a
    // unit test must never reach the network (slow, flaky, and it would mutate the
    // developer's own clone). The fetch path gets its own local file:// fixture below.
    c.fetch_enabled  = false;
    return c;
}

// Run a single builder task synchronously via a 1-thread pool and return the result.
worker_result run_task(worker_pool::task t,
                       worker_pool::task_key key = {task_kind::build, {}, {}}) {
    event_queue q;
    {
        worker_pool pool(1, q);
        pool.submit(key, std::move(t));
        event e;
        // The task runs external git/cmake (worktree add, submodule clones): not a
        // process we control, so a timeout is legitimate here (CLAUDE.md "Testing")
        // — a generous one, never a performance budget.
        REQUIRE(q.pop_until(std::chrono::steady_clock::now() + std::chrono::minutes(5), e));
        REQUIRE(std::holds_alternative<worker_result>(e));
        return std::get<worker_result>(e);
    }
}
} // namespace

TEST_CASE("builder: resolve HEAD yields a full commit SHA") {
    builder b(cfg());
    auto r = run_task(b.resolve_task("j1", "HEAD"));
    CHECK(r.kind == task_kind::resolve);
    CHECK(r.id == "j1");
    CHECK(r.ok);
    CHECK(r.sha.size() == 40);                       // full SHA
    CHECK(r.sha.find_first_not_of("0123456789abcdef") == std::string::npos);
}

TEST_CASE("builder: resolving a bogus ref fails cleanly (no throw, ok=false)") {
    builder b(cfg());
    auto r = run_task(b.resolve_task("j2", "definitely-not-a-ref-xyzzy"));
    CHECK(r.kind == task_kind::resolve);
    CHECK_FALSE(r.ok);
    CHECK(r.sha.empty());
    CHECK_FALSE(r.message.empty());
}

// -----------------------------------------------------------------------------
// Fetch + remote-first resolution. Fixture: an "upstream" repo and a clone of it,
// wired over file:// so the whole thing is local — no network, and the developer's
// own clone is never touched. The scenario is exactly the failure the fetch
// capability exists to prevent: upstream moves a branch forward AFTER the clone
// was taken, so a purely local resolve returns the OLD commit while claiming to
// have run the branch.
// -----------------------------------------------------------------------------
namespace {

struct git_fixture {
    // A COUNTER, not this->root's address: two fixtures alive at different times
    // can land on the same stack slot, so the address is not the unique token it
    // looks like. (pid still separates concurrent ctest runs.)
    static inline std::atomic<unsigned> seq{0};
    fs::path root{fs::temp_directory_path() /
                  ("coastd_fetch_" + std::to_string(::getpid()) + "_" +
                   std::to_string(seq.fetch_add(1)))};
    fs::path upstream{root / "upstream"};
    fs::path clone{root / "clone"};

    static void git(const fs::path& cwd, std::vector<std::string> args) {
        std::vector<std::string> full{"-C", cwd.string()};
        full.insert(full.end(), args.begin(), args.end());
        auto r = run_capture("git", full, {}, std::chrono::seconds(30));
        REQUIRE_MESSAGE(r.ok(), "git " << args.front() << " failed: " << r.output);
    }

    static std::string head_of(const fs::path& repo, const std::string& rev) {
        auto r = run_capture("git", {"-C", repo.string(), "rev-parse", rev}, {},
                             std::chrono::seconds(30));
        REQUIRE(r.ok());
        std::string s = r.output;
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        return s;
    }

    static void commit(const fs::path& repo, const std::string& text) {
        std::ofstream(repo / "file.txt", std::ios::app) << text << "\n";
        git(repo, {"add", "-A"});
        git(repo, {"-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", text});
    }

    git_fixture() {
        fs::create_directories(upstream);
        git(upstream, {"init", "-q", "-b", "main", "."});
        commit(upstream, "first");
        auto r = run_capture("git", {"clone", "-q", "file://" + upstream.string(),
                                     clone.string()}, {}, std::chrono::seconds(30));
        REQUIRE_MESSAGE(r.ok(), "git clone failed: " << r.output);
    }
    ~git_fixture() { std::error_code ec; fs::remove_all(root, ec); }
};

daemon_config fetch_cfg(const git_fixture& f) {
    daemon_config c;
    c.repo_root      = f.clone;
    c.worktrees_root = f.root / "wt";
    c.fetch_min_interval = std::chrono::seconds(0);   // no coalescing; every resolve fetches
    return c;   // fetch_enabled defaults to true — that is the behaviour under test
}

} // namespace

TEST_CASE("builder: fetch makes a moved upstream branch resolve to the NEW commit") {
    git_fixture f;
    const std::string before = git_fixture::head_of(f.clone, "main");

    // Upstream moves on. The clone's local `main` and its origin/main are now both
    // stale — this is the state a server clone sits in permanently, since coastd
    // never checks anything out in repo_root.
    git_fixture::commit(f.upstream, "second");
    const std::string after = git_fixture::head_of(f.upstream, "main");
    REQUIRE(before != after);

    builder b(fetch_cfg(f));
    auto r = run_task(b.resolve_task("j", "main"));
    REQUIRE(r.ok);
    CHECK(r.sha == after);                       // the fetched upstream tip
    CHECK(git_fixture::head_of(f.clone, "main") == before);  // local branch untouched
}

TEST_CASE("builder: with fetching disabled, resolution stays local (and stale)") {
    git_fixture f;
    const std::string before = git_fixture::head_of(f.clone, "main");
    git_fixture::commit(f.upstream, "second");

    daemon_config c = fetch_cfg(f);
    c.fetch_enabled = false;
    builder b(c);
    auto r = run_task(b.resolve_task("j", "main"));
    REQUIRE(r.ok);
    CHECK(r.sha == before);   // the documented opt-out: local-only, therefore stale
}

TEST_CASE("builder: a tag, a raw SHA and HEAD resolve verbatim, not via the remote") {
    git_fixture f;
    const std::string local_head = git_fixture::head_of(f.clone, "HEAD");
    git_fixture::git(f.clone, {"tag", "v1"});
    // Move upstream so a remote-first misresolution would be visible as a different SHA.
    git_fixture::commit(f.upstream, "second");

    builder b(fetch_cfg(f));

    // HEAD must mean THIS repo's HEAD, never origin/HEAD (the remote's default branch).
    CHECK(run_task(b.resolve_task("j", "HEAD")).sha == local_head);
    // A local-only tag has no origin/v1 — it must still resolve.
    CHECK(run_task(b.resolve_task("j", "v1")).sha == local_head);
    // A raw SHA is already exact.
    CHECK(run_task(b.resolve_task("j", local_head)).sha == local_head);
    // An explicitly qualified ref pins the local branch even though origin/main moved.
    CHECK(run_task(b.resolve_task("j", "refs/heads/main")).sha == local_head);
}

TEST_CASE("builder: a fetch failure is not fatal — resolution falls back to local") {
    git_fixture f;
    const std::string before = git_fixture::head_of(f.clone, "main");
    // Point the clone at a remote that cannot exist. The fetch must fail, be logged,
    // and leave resolution working against the objects already in the clone.
    git_fixture::git(f.clone, {"remote", "set-url", "origin",
                               "file:///nonexistent/coastd-test-remote"});

    builder b(fetch_cfg(f));
    auto r = run_task(b.resolve_task("j", "main"));
    CHECK(r.ok);
    CHECK(r.sha == before);
}

TEST_CASE("builder: the coalescing window suppresses a second fetch") {
    git_fixture f;
    daemon_config c = fetch_cfg(f);
    c.fetch_min_interval = std::chrono::seconds(3600);   // one fetch, then closed for the test
    builder b(c);

    auto first = run_task(b.resolve_task("j1", "main"));
    REQUIRE(first.ok);
    const std::string after_first = git_fixture::head_of(f.clone, "origin/main");

    // Upstream moves; the window is still open, so the second resolve must NOT fetch
    // and must therefore return the tip the first fetch brought in.
    git_fixture::commit(f.upstream, "second");
    auto second = run_task(b.resolve_task("j2", "main"));
    REQUIRE(second.ok);
    CHECK(second.sha == after_first);
    CHECK(git_fixture::head_of(f.clone, "origin/main") == after_first);
}

TEST_CASE("builder: resolves that arrive while a fetch runs wait for it") {
    // The eval submits a burst of jobs for a commit the daemon has never seen. The
    // first resolve fetches; the others land in the same window. They must resolve
    // against the fetched refs, not fail with "could not resolve ref" because they
    // looked before the fetch finished (seen on 2026-11-07: one job of a burst failed).
    git_fixture f;
    git_fixture::commit(f.upstream, "second");
    const std::string fresh = git_fixture::head_of(f.upstream, "main");
    // A remote that answers after 2 s and logs each fetch, so every resolve below
    // starts while the one fetch is still running. ext:: runs the command as the
    // transport; "% " is a space.
    const fs::path log = f.root / "fetches";
    git_fixture::git(f.clone, {"config", "protocol.ext.allow", "always"});
    git_fixture::git(f.clone, {"remote", "set-url", "origin",
                               "ext::sh -c echo% x>>" + log.string() +
                               ";sleep% 2;git% upload-pack% " + f.upstream.string()});

    daemon_config c = fetch_cfg(f);
    // SHORTER than the fetch: a resolve that waited out the fetch must still count as
    // inside its window (the window starts at the resolve's arrival), or the waiters
    // fetch again one after another — N resolves, N serial fetches.
    c.fetch_min_interval = std::chrono::seconds(1);
    builder b(c);
    // The tasks run directly on their own threads (doctest assertions are not
    // thread-safe, so nothing is checked until all have joined).
    std::vector<worker_result> results(6);
    std::vector<std::thread> burst;
    for (std::size_t i = 0; i < results.size(); ++i)
        burst.emplace_back([&b, &results, &fresh, i] {
            results[i] = b.resolve_task("j" + std::to_string(i), fresh)();
        });
    for (auto& t : burst) t.join();
    std::ifstream in(log);
    std::size_t fetches = 0;
    for (std::string line; std::getline(in, line);) ++fetches;
    CHECK(fetches == 1);   // one fetch for the whole burst
    for (const auto& r : results) {
        CAPTURE(r.message);
        CHECK(r.ok);
        CHECK(r.sha == fresh);
    }
}

TEST_CASE("builder: path layout is SHA-keyed") {
    builder b(cfg());
    const auto wt     = b.worktree_dir("abc123");
    const auto bin    = b.coast_binary("abc123");
    const auto marker = b.ready_marker("abc123");
    CHECK(wt.filename() == "abc123");
    CHECK(bin == wt / "build" / "coast");
    CHECK(marker == wt / build_ready_marker);   // readiness sentinel lives at the worktree root
}

// -----------------------------------------------------------------------------
// The partial-checkout self-heal. A crash/SIGKILL/timeout during `git worktree add`
// leaves the directory present but the checkout incomplete. The builder distinguishes
// the two with .coastd-checkout, written only after a clean add — so "directory
// exists" is NOT "tree is usable", and a tree without the marker must be wiped and
// re-added rather than configured on top of.
//
// This path had no test. It is also the branch you cannot afford to get wrong in
// either direction: skip the wipe and every retry re-fails on a broken tree until the
// job is terminally failed with a confusing cmake error; wipe when you should not and
// a good ~238 MB checkout is thrown away on every build.
//
// Both cases run against the throwaway git_fixture, never the developer's clone —
// the wipe path issues `git worktree remove/prune` against repo_root, which is a
// write to that repo's .git. The build is driven only far enough to observe the
// decision: a SHA-shaped ref that does not exist makes the following step fail
// immediately, so neither case reaches cmake.
// -----------------------------------------------------------------------------
TEST_CASE("builder: a worktree with no checkout marker is wiped and re-added") {
    git_fixture f;
    auto c = fetch_cfg(f);
    c.fetch_enabled = false;                       // nothing to fetch; keep it local
    builder b(c);

    const sha_t sha = "0123456789abcdef0123456789abcdef01234567";  // hex-shaped, absent
    const fs::path wt = c.worktrees_root / sha;
    fs::create_directories(wt);
    const fs::path witness = wt / "PARTIAL_CHECKOUT_LEFTOVER";
    { std::ofstream(witness) << "x\n"; }

    SUBCASE("marker ABSENT -> the tree is wiped") {
        REQUIRE_FALSE(fs::exists(wt / ".coastd-checkout"));
        const worker_result r = run_task(b.build_task(sha));
        CHECK_FALSE(r.ok);                         // the absent SHA fails the re-add
        // The wipe is the assertion: the leftover from the interrupted checkout is
        // gone, so the next attempt starts from a clean tree instead of building on
        // a partial one.
        CHECK_FALSE(fs::exists(witness));
    }

    SUBCASE("marker PRESENT -> the tree is REUSED, not wiped") {
        // The other direction, and the one a wipe-always implementation would fail:
        // a complete checkout must survive, or every build re-clones.
        { std::ofstream(wt / ".coastd-checkout") << sha << "\n"; }
        const worker_result r = run_task(b.build_task(sha));
        CHECK_FALSE(r.ok);                         // still fails later (not a real tree)
        CHECK(fs::exists(witness));                // but the tree was left alone
    }
}

// -----------------------------------------------------------------------------
// Build-worktree submodules: every submodule except the docs theme (1.1 GiB of
// history, not compiled), shallowly. The fixture mirrors the
// real .gitmodules paths, all pointing at one tiny two-commit repo — two commits so
// that a depth-1 clone is actually shallow (a one-commit history cannot be cut, so
// --is-shallow-repository would read false even with --depth 1). The URL is
// file://, not a bare path: git clones a bare-path URL by hardlinking and ignores
// --depth for it (measured: the json check below then reads false).
//
// The submodule URLs are local paths, and git refuses the `file` transport for
// submodule clones unless protocol.file.allow=always (CVE-2022-39253). The builder
// deliberately does not set it, and it ignores ~/.gitconfig (run_git pins
// GIT_CONFIG_GLOBAL=/dev/null) and the superproject's local config does not reach
// the submodule clone (measured), so the environment is the only channel — set
// here, for this case only, and restored on exit.
// -----------------------------------------------------------------------------
namespace {

struct scoped_env {
    std::vector<std::pair<std::string, std::optional<std::string>>> saved;
    explicit scoped_env(std::vector<std::pair<std::string, std::string>> vars) {
        for (auto& [k, v] : vars) {
            const char* old = std::getenv(k.c_str());
            saved.emplace_back(k, old ? std::optional<std::string>(old) : std::nullopt);
            ::setenv(k.c_str(), v.c_str(), 1);
        }
    }
    ~scoped_env() {
        for (auto& [k, v] : saved) {
            if (v) ::setenv(k.c_str(), v->c_str(), 1);
            else   ::unsetenv(k.c_str());
        }
    }
};

std::string git_out(const fs::path& repo, std::vector<std::string> args) {
    std::vector<std::string> full{"-C", repo.string()};
    full.insert(full.end(), args.begin(), args.end());
    auto r = run_capture("git", full, {}, std::chrono::seconds(30));
    REQUIRE_MESSAGE(r.ok(), "git " << args.front() << " failed: " << r.output);
    std::string s = r.output;
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

} // namespace

TEST_CASE("builder: the build worktree gets the 3rd_party submodules shallow, "
          "never the docs theme") {
    git_fixture f;
    const fs::path lib = f.root / "lib";
    fs::create_directories(lib);
    git_fixture::git(lib, {"init", "-q", "-b", "main", "."});
    git_fixture::commit(lib, "one");
    git_fixture::commit(lib, "two");

    const std::vector<std::string> paths{"3rd_party/imgui", "3rd_party/json",
                                         "3rd_party/implot", "3rd_party/ImNodeFlow",
                                         "docs/themes/hugo-theme-relearn"};
    for (const auto& p : paths)
        git_fixture::git(f.upstream, {"-c", "protocol.file.allow=always", "submodule",
                                      "add", "-q", "file://" + lib.string(), p});
    git_fixture::git(f.upstream, {"-c", "user.email=t@t", "-c", "user.name=t",
                                  "commit", "-q", "-m", "submodules"});
    git_fixture::git(f.clone, {"pull", "-q", "--ff-only"});

    auto c = fetch_cfg(f);
    c.fetch_enabled = false;
    builder b(c);
    const sha_t sha = git_fixture::head_of(f.clone, "HEAD");
    const fs::path wt = c.worktrees_root / sha;

    worker_result r;
    {
        scoped_env env({{"GIT_CONFIG_COUNT", "1"},
                        {"GIT_CONFIG_KEY_0", "protocol.file.allow"},
                        {"GIT_CONFIG_VALUE_0", "always"}});
        r = run_task(b.build_task(sha));
    }
    // The fixture has no CMakeLists.txt, so the build must stop at configure — i.e.
    // AFTER the submodule step. Assert that it got there; a failure at the
    // submodule step would leave the checks below vacuous for the theme.
    REQUIRE_FALSE(r.ok);
    REQUIRE_MESSAGE(r.message.starts_with("cmake configure"), r.message);

    for (const auto* p : {"3rd_party/imgui", "3rd_party/json", "3rd_party/implot",
                          "3rd_party/ImNodeFlow"})
        CHECK_MESSAGE(fs::exists(wt / p / ".git"), p << " was not initialised");
    CHECK(git_out(wt / "3rd_party/json", {"rev-parse", "--is-shallow-repository"}) == "true");

    const fs::path theme = wt / "docs/themes/hugo-theme-relearn";
    CHECK_FALSE(fs::exists(theme / ".git"));
    CHECK(fs::is_empty(theme));
}

// The shape of every other fixture that builds through the builder (the fault
// matrix's fake-coast repo): no .gitmodules at all. The submodule step must be a
// no-op there, not a failure. A step that named submodule paths would fail it
// with "pathspec ... did not match".
TEST_CASE("builder: a repo with no submodules gets past the submodule step") {
    git_fixture f;
    auto c = fetch_cfg(f);
    c.fetch_enabled = false;
    builder b(c);
    const worker_result r = run_task(b.build_task(git_fixture::head_of(f.clone, "HEAD")));
    // No CMakeLists.txt, so configure is where it must stop — the step AFTER
    // the submodule update.
    CHECK_FALSE(r.ok);
    CHECK_MESSAGE(r.message.starts_with("cmake configure"), r.message);
}
