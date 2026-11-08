#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include <unistd.h>

#include "daemon/proc_util.h"

using namespace coastd;
using namespace std::chrono_literals;

// These exercise run_capture's wall-clock deadline — the bound that keeps a hung
// git/cmake (an unreachable FetchContent remote, a stale lock) from parking a
// worker thread forever. Without the timeout, findings from review: one wedged
// build step silently blocks ALL future builds for the daemon's lifetime.

TEST_CASE("run_capture: a child that outlives the timeout is killed and flagged") {
    // `sleep 10` produces no output and would otherwise block for 10s; a 1s
    // deadline must reap it. No duration is asserted: a lost deadline lets the
    // sleep end normally, which `timed_out` reports deterministically.
    auto r = run_capture("sleep", {"10"}, {}, 1s);

    CHECK(r.spawned);
    CHECK(r.timed_out);
    CHECK_FALSE(r.ok());
}

TEST_CASE("run_capture: a fast child completes normally within its timeout") {
    auto r = run_capture("sh", {"-c", "printf ok"}, {}, 30s);
    CHECK(r.spawned);
    CHECK_FALSE(r.timed_out);
    CHECK(r.ok());
    CHECK(r.output == "ok");
}

TEST_CASE("run_capture: a zero timeout means unbounded (still completes)") {
    auto r = run_capture("sh", {"-c", "printf done"}, {}, 0s);
    CHECK(r.ok());
    CHECK_FALSE(r.timed_out);
    CHECK(r.output == "done");
}

// ---------------------------------------------------------------------------
// git hardening (run_git). These assert against REAL git behaviour rather than
// trusting that the flags mean what the docs say: each case builds a repo that
// would execute attacker-controlled code under plain `git`, then checks that the
// hardened seam does not. The negative control (run_capture("git", …) DOES fire
// the hook) is what makes the positive result evidence rather than a tautology —
// without it, a test that never triggers the hook would pass for the wrong reason.
// ---------------------------------------------------------------------------

namespace {

// A repo whose post-checkout hook drops a witness file. `git worktree add` on it
// fires the hook; the witness's existence is the measurement.
struct hook_repo {
    std::filesystem::path root;
    std::filesystem::path witness;

    explicit hook_repo(const std::string& tag) {
        root = std::filesystem::temp_directory_path() /
               ("coastd_hook_" + tag + "_" + std::to_string(::getpid()));
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root);
        witness = root / "HOOK_RAN";

        // Checked: a setup that silently failed (e.g. git timing out on a loaded
        // box) would leave no hook to fire, and every "the hook did not run" below
        // would pass for the wrong reason. External git — a generous bound.
        const auto sh = [&](const std::string& c) {
            REQUIRE(run_capture("sh", {"-c", "cd '" + root.string() + "' && " + c}, {},
                                std::chrono::minutes(5)).ok());
        };
        // -c on the command line so the harness does not depend on the operator's
        // global identity (which the env scrub pins to /dev/null anyway).
        sh("git init --quiet .");
        sh("git -c user.email=t@t -c user.name=t commit --quiet --allow-empty -m base");
        std::ofstream h(root / ".git" / "hooks" / "post-checkout");
        h << "#!/bin/sh\ntouch '" << witness.string() << "'\n";
        h.close();
        std::filesystem::permissions(root / ".git" / "hooks" / "post-checkout",
                                     std::filesystem::perms::owner_all);
    }
    ~hook_repo() { std::filesystem::remove_all(root); }

    [[nodiscard]] bool hook_ran() const { return std::filesystem::exists(witness); }
    [[nodiscard]] std::string wt(const std::string& n) const { return (root / n).string(); }
};

} // namespace

TEST_CASE("run_git: worktree add does NOT fire a post-checkout hook") {
    // The concrete cross-job persistence vector: a submitted job runs as the daemon
    // uid and can write repo_root/.git/hooks/post-checkout, which the daemon's own
    // `worktree add` would then execute for every future job on a new SHA.
    SUBCASE("negative control: an UNHARDENED git does fire it") {
        // Must strip the environment explicitly. run_capture itself now exports the
        // hardening via GIT_CONFIG_PARAMETERS (so cmake's own git children are
        // covered), which means run_capture("git", ...) is no longer an unhardened
        // control — it stopped firing the hook the moment that env change landed.
        // `env -u` gives a genuinely unprotected git, which is what makes the
        // hardened result below evidence rather than a test that never armed.
        hook_repo repo("ctl");
        auto r = run_capture("sh", {"-c",
            "env -u GIT_CONFIG_PARAMETERS -u GIT_CONFIG_GLOBAL -u GIT_CONFIG_SYSTEM "
            "git -C '" + repo.root.string() + "' worktree add --detach '" +
            repo.wt("wt_ctl") + "'"}, {}, 60s);
        REQUIRE(r.spawned);
        CHECK(repo.hook_ran());          // if this fails the harness is not arming the vector
    }
    SUBCASE("hardened: run_git does not") {
        hook_repo repo("safe");
        auto r = run_git({"-C", repo.root.string(), "worktree", "add",
                          "--detach", repo.wt("wt_safe")}, {}, 60s);
        REQUIRE(r.spawned);
        CHECK(r.ok());                   // the checkout itself must still succeed
        CHECK_FALSE(repo.hook_ran());
    }
}

TEST_CASE("run_git: a hooksPath poisoned in LOCAL config is overridden") {
    // `-c` outranks every config file, so an attacker relocating hooksPath cannot
    // dodge the /dev/null pin. Without this, the fix would be one `git config` away
    // from useless.
    hook_repo repo("relocated");
    const auto alt = repo.root / "evil_hooks";
    std::filesystem::create_directories(alt);
    std::filesystem::rename(repo.root / ".git" / "hooks" / "post-checkout",
                            alt / "post-checkout");
    (void)run_capture("git", {"-C", repo.root.string(), "config", "core.hooksPath",
                              alt.string()}, {}, 30s);

    auto r = run_git({"-C", repo.root.string(), "worktree", "add", "--detach",
                      repo.wt("wt_rel")}, {}, 60s);
    REQUIRE(r.spawned);
    CHECK(r.ok());
    CHECK_FALSE(repo.hook_ran());
}

TEST_CASE("run_capture: the child env pins global/system git config to /dev/null") {
    // The other half of the containment: the child cannot make the daemon honour a
    // ~/.gitconfig it wrote. Read the values back through git itself.
    auto g = run_capture("sh", {"-c", "printf %s \"$GIT_CONFIG_GLOBAL\""}, {}, 30s);
    CHECK(g.output == "/dev/null");
    auto s = run_capture("sh", {"-c", "printf %s \"$GIT_CONFIG_SYSTEM\""}, {}, 30s);
    CHECK(s.output == "/dev/null");
}

TEST_CASE("run_capture: the git hardening reaches children we do NOT exec directly") {
    // run_git's `-c` flags cover only the git process the daemon execs itself. The
    // build step execs `cmake`, and cmake/coast_build_info.cmake runs `git rev-parse`
    // / `git status` at configure time — measured: with the hardening on the argv
    // only, a poisoned core.fsmonitor in repo_root/.git/config executed on every
    // cmake configure, reopening the exact key the seam closes. GIT_CONFIG_PARAMETERS
    // on the ENV is what covers those grandchildren (and FetchContent's git).
    SUBCASE("the variable is exported in git's own format") {
        auto r = run_capture("sh", {"-c", "printf %s \"$GIT_CONFIG_PARAMETERS\""}, {}, 30s);
        CHECK(r.output.find("'core.hooksPath'='/dev/null'") != std::string::npos);
        CHECK(r.output.find("'core.fsmonitor'=''") != std::string::npos);
    }
    SUBCASE("safe.directory is present, and lands in PROTECTED scope") {
        // Required, not cosmetic. Pinning GIT_CONFIG_GLOBAL/SYSTEM to /dev/null
        // removes the only two scopes that may set safe.directory — git ignores it
        // in repo-local config on purpose — so without this entry a repo_root owned
        // by a foreign uid makes every `git rev-parse` exit 128 with "dubious
        // ownership", which fails ref resolution, which files EVERY submitted job to
        // failed/ permanently. Asserting the SCOPE (not just presence) is the point:
        // the key is inert unless git reads it as protected.
        auto r = run_capture("git", {"config", "--show-scope", "--get", "safe.directory"},
                             {}, 30s);
        CHECK(r.ok());
        CHECK(r.output.find("command") != std::string::npos);
        CHECK(r.output.find("*") != std::string::npos);
    }
    SUBCASE("a real git ACCEPTS it") {
        // The format is picky: a malformed value makes git abort with "bogus format
        // in GIT_CONFIG_PARAMETERS" on EVERY call — which would break the daemon
        // wholesale, so this is the case that must never regress.
        auto r = run_capture("git", {"--version"}, {}, 30s);
        CHECK(r.ok());
        CHECK(r.output.find("bogus") == std::string::npos);
    }
    SUBCASE("and it actually takes effect: hooks stay disabled without any -c") {
        // Exercised through plain run_capture (NOT run_git), which is what cmake and
        // its git children get.
        hook_repo repo("env");
        auto r = run_capture("git", {"-C", repo.root.string(), "worktree", "add",
                                     "--detach", repo.wt("wt_env")}, {}, 60s);
        REQUIRE(r.spawned);
        CHECK(r.ok());   // the checkout happened, so the hook WOULD have had its chance
        CHECK_FALSE(repo.hook_ran());
    }
}
