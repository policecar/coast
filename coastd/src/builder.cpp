#include "daemon/builder.h"

#include <algorithm>
#include <mutex>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <optional>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include "daemon/atomic_write.h"
#include "daemon/log_sweep.h"
#include "daemon/proc_util.h"
#include "diag/diag.h"

namespace coastd {

namespace {

// Trim leading/trailing whitespace (git rev-parse output has a trailing newline).
std::string trim(std::string s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Last ~40 lines of `s`, for a compact failure message.
std::string tail(const std::string& s, std::size_t lines = 40) {
    std::size_t pos = s.size(), seen = 0;
    while (pos > 0 && seen < lines) {
        const auto nl = s.rfind('\n', pos - 1);
        if (nl == std::string::npos) { pos = 0; break; }
        pos = nl;
        ++seen;
    }
    return s.substr(pos == 0 ? 0 : pos + 1);
}

// The build log is APPEND-ONLY across every attempt on this SHA and nothing else
// bounds it: a crash-looping build writes gigabytes within a day, well inside any
// sane compress-after window. append_capped (log_sweep.h) enforces a hard ceiling
// at write time; the age-based sweep handles the rest.

// Parallelism for `cmake --build -j N`. An operator override (cfg.build_jobs > 0)
// wins; 0 = auto (half the detected cores, so a build gets ~half the box while
// sims keep running).
int build_parallelism(int override) {
    if (override > 0) return override;
    const unsigned hc = std::thread::hardware_concurrency();
    return std::max(1, static_cast<int>(hc == 0 ? 2 : hc) / 2);
}

// (The SHA shape test every path-building step below gates on is types.h's
// is_sha_shaped — shared with rescan, which reads back what these steps write.)

// Would `ref` be looked for under refs/remotes/<remote>/ if we asked? Only a
// PLAIN branch-ish name qualifies for the remote-first preference. Everything
// else must resolve verbatim, because prefixing it would change its meaning:
//   "HEAD"        -> "origin/HEAD" is the REMOTE's default branch, not this repo's HEAD
//   "refs/heads/x"-> an explicit, fully-qualified ref the user deliberately pinned
//   a raw SHA     -> already an exact commit; nothing to freshen
//   "main~3", "x^{}", "@{u}" -> revision syntax; "origin/" + it is a different rev
bool looks_like_branch_name(const std::string& ref) {
    if (ref.empty()) return false;
    if (ref == "HEAD" || ref.starts_with("HEAD^") || ref.starts_with("HEAD~")) return false;
    if (ref.starts_with("refs/")) return false;
    if (ref.starts_with("-")) return false;
    if (is_sha_shaped(ref)) return false;
    return ref.find_first_of("^~:@?*[\\ \t") == std::string::npos;
}

// The remote to prefer refs from. An explicit cfg.git_remote wins; otherwise
// "origin" if it exists, otherwise the sole remote. With zero or several
// unnamed-origin remotes there is no defensible default, so we return "" and the
// remote-first candidate is simply skipped (resolution stays purely local).
std::string pick_remote(const std::filesystem::path& repo, const std::string& configured,
                        std::chrono::seconds timeout) {
    if (!configured.empty()) return configured;
    auto r = run_git({"-C", repo.string(), "remote"}, {}, timeout);
    if (!r.ok()) return {};
    std::vector<std::string> remotes;
    std::istringstream in(r.output);
    for (std::string line; std::getline(in, line);)
        if (auto t = trim(std::move(line)); !t.empty()) remotes.push_back(std::move(t));
    if (remotes.empty()) return {};
    for (const auto& e : remotes) if (e == "origin") return e;
    if (remotes.size() == 1) return remotes.front();
    return {};
}

std::int64_t now_epoch_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

// Refresh remote-tracking refs, at most once per fetch_min_interval across all
// concurrent resolve tasks. A task that finds a fetch in flight waits for it to
// finish: it must look up its ref in what that fetch brings in, or in a burst of
// jobs for a commit this clone has not seen yet, a job that looks before the fetch
// is done fails with "could not resolve ref" (seen 2026-11-07). The cost is that a resolve inside the
// window can wait for one fetch (at most fetch_timeout), as its first caller does.
//
// BEST-EFFORT BY DESIGN: a fetch failure (no network, no remote, a transient
// "cannot lock ref" race against a concurrent `git worktree add` in the same
// repo) must NOT fail the job. The local clone very often already holds the ref,
// and turning a network blip into a terminally-failed job would be a large
// reliability regression for a freshness feature. We log and resolve anyway —
// the caller then either succeeds against the local objects or reports the
// ordinary "could not resolve" error.
//
// `--prune` drops deleted remote branches so a resubmitted stale name fails
// loudly instead of resolving to a ref the remote no longer has; `--tags` keeps
// tag-named submissions resolvable.
void maybe_fetch(const daemon_config& cfg, fetch_gate& gate) {
    if (!cfg.fetch_enabled) return;

    // The task's ARRIVAL time, read before the lock: a task that waits out a fetch
    // longer than the window must still count as inside it, or the waiters fetch
    // again one after another (N tasks, N serial fetches).
    const std::int64_t now    = now_epoch_ms();
    const std::int64_t window = std::chrono::milliseconds(cfg.fetch_min_interval).count();
    // Held across the fetch: a task in the window blocks here until the fetch that
    // claimed it is done, then sees the stamp and resolves against the new refs.
    const std::lock_guard lock(gate.m);
    if (gate.last_ms != 0 && now - gate.last_ms < window) return;
    // Stamped BEFORE the fetch so a slow fetch throttles the next one from its
    // START, not its end; a failed fetch is stamped too and not retried in the window.
    gate.last_ms = now;

    auto r = run_git({"-C", cfg.repo_root.string(), "fetch", "--all", "--prune",
                                 "--tags"}, {}, cfg.fetch_timeout);
    if (!r.ok()) {
        COAST_WARN("coastd.builder",
                   "git fetch in {} failed ({}) — resolving against the local clone, which "
                   "may be stale", cfg.repo_root.string(),
                   r.timed_out ? "timed out" : "non-zero exit");
        return;
    }
    COAST_DEBUG("coastd.builder", "fetched {}", cfg.repo_root.string());
}

// rev-parse one candidate. nullopt = this candidate does not name a commit.
// `timed_out` is set (never cleared) when the git call hit its deadline, so the
// caller can distinguish "no such ref" from "git is wedged" across candidates.
std::optional<std::string> rev_parse_commit(const std::filesystem::path& repo,
                                            const std::string& rev,
                                            std::chrono::seconds timeout,
                                            bool& timed_out) {
    // `^{commit}` peels tags/branches to the commit. `--end-of-options` stops flag
    // parsing while keeping the argument a REV (a bare `--` would make rev-parse
    // treat it as a pathspec), so a ref beginning with '-' can't be read as an option.
    auto r = run_git({"-C", repo.string(), "rev-parse", "--verify", "-q",
                                 "--end-of-options", rev + "^{commit}"}, {}, timeout);
    if (r.timed_out) timed_out = true;
    if (!r.ok()) return std::nullopt;
    std::string sha = trim(r.output);
    if (!is_sha_shaped(sha)) return std::nullopt;
    return sha;
}

} // namespace

builder::builder(daemon_config cfg)
    : cfg_(std::move(cfg)),
      fetch_gate_(std::make_shared<fetch_gate>()) {}

std::filesystem::path builder::worktree_dir(const sha_t& sha) const {
    return cfg_.worktrees_root / sha;
}
std::filesystem::path builder::coast_binary(const sha_t& sha) const {
    return worktree_dir(sha) / "build" / "coast";
}
std::filesystem::path builder::ready_marker(const sha_t& sha) const {
    return worktree_dir(sha) / build_ready_marker;
}

worker_pool::task builder::resolve_task(job_id id, std::string ref_symbolic) const {
    const daemon_config cfg   = cfg_;
    auto                fetch = fetch_gate_;   // shared throttle; outlives the builder
    return [cfg, fetch, id = std::move(id), ref = std::move(ref_symbolic)]() -> worker_result {
        const std::filesystem::path& repo = cfg.repo_root;
        const std::chrono::seconds   to   = cfg.git_op_timeout;

        // Freshen remote-tracking refs FIRST, so "main" means what the remote calls
        // main right now rather than whatever this clone was left holding. Never
        // fatal — see maybe_fetch.
        maybe_fetch(cfg, *fetch);

        // Candidate order. The remote-tracking ref is preferred for plain branch
        // names (the remote is the source of truth on a server; a local branch here
        // is an accident of whoever last checked something out), and everything
        // else — tags, SHAs, HEAD, refs/…, revision syntax — falls through to the
        // verbatim resolution, which is also the fallback when the remote has no
        // such branch. Tied to fetch_enabled: with fetching off, resolution is
        // purely local, exactly as before this feature.
        std::string resolved_via;
        std::optional<std::string> sha;
        bool timed_out = false;
        if (cfg.fetch_enabled && looks_like_branch_name(ref)) {
            if (const std::string remote = pick_remote(repo, cfg.git_remote, to); !remote.empty()) {
                resolved_via = remote + "/" + ref;
                sha = rev_parse_commit(repo, resolved_via, to, timed_out);
            }
        }
        if (!sha) {
            resolved_via = ref;
            sha = rev_parse_commit(repo, ref, to, timed_out);
        }
        if (!sha) {
            return worker_result{task_kind::resolve, id, "", false,
                                 timed_out ? "timed out resolving ref '" + ref + "'"
                                           : "could not resolve ref '" + ref + "'"};
        }
        if (resolved_via != ref)
            COAST_DEBUG("coastd.builder", "resolved '{}' via '{}' -> {}", ref, resolved_via, *sha);
        return worker_result{task_kind::resolve, id, *sha, true, ""};
    };
}

worker_pool::task builder::build_task(sha_t sha) const {
    const daemon_config cfg = cfg_;
    const std::filesystem::path wt     = worktree_dir(sha);
    const std::filesystem::path bin    = coast_binary(sha);
    const std::filesystem::path marker = ready_marker(sha);
    // Written ONLY after a clean `git worktree add`. Its absence on an existing
    // worktree dir means the checkout was interrupted (SIGKILL/crash/timeout mid
    // add) and the tree is partial — so we wipe and re-add rather than build on it.
    const std::filesystem::path checkout_marker = wt / ".coastd-checkout";
    const std::filesystem::path build_dir = wt / "build";
    const std::filesystem::path log = wt / "coastd-build.log";
    const int jobs = build_parallelism(cfg_.build_jobs);

    const std::chrono::seconds git_to   = cfg_.git_op_timeout;
    const std::chrono::seconds build_to = cfg_.build_timeout;

    return [cfg, sha, wt, bin, marker, checkout_marker, build_dir, log, jobs, git_to, build_to]() -> worker_result {
        if (!is_sha_shaped(sha))
            return worker_result{task_kind::build, "", sha, false,
                                 "refusing to build: invalid (non-hex) SHA '" + sha + "'"};
        auto fail = [&](std::string_view step, const exec_result& r) {
            append_capped(log, step, r.output, cfg.build_log_max_bytes);
            const std::string why = r.timed_out ? " timed out:\n" : " failed:\n";
            return worker_result{task_kind::build, "", sha, false,
                                 std::string(step) + why + tail(r.output)};
        };

        // Retract any prior readiness up front: while this (incremental) build
        // runs, the SHA must NOT read as runnable. The marker is re-published only
        // once the fresh build fully succeeds below, so "marker present" always
        // means "a build completed since the last build started".
        std::error_code ec;
        std::filesystem::remove(marker, ec);

        // Validate any existing worktree BEFORE reusing it. A crash/SIGKILL/timeout
        // during a prior `git worktree add` leaves the directory present but a
        // partial, unbuildable checkout; the checkout marker (written only after a
        // clean add below) is absent in that case, so wipe and re-add rather than
        // configure/build on a broken tree and fail with no self-heal.
        if (std::filesystem::exists(wt, ec) && !std::filesystem::exists(checkout_marker, ec)) {
            COAST_WARN("coastd.builder", "worktree {} has no completed-checkout marker "
                       "(interrupted add?); wiping and re-adding", wt.string());
            (void)run_git({"-C", cfg.repo_root.string(), "worktree", "remove",
                                      "--force", wt.string()}, {}, git_to);
            std::filesystem::remove_all(wt, ec);
            if (ec) {
                // FAIL HERE, with the real reason. The old code warned that "the
                // re-add below will report the real failure" — but the re-add is
                // guarded by `if (!exists(wt))`, and a failed removal is precisely
                // the case where the tree still exists. So the re-add was SKIPPED and
                // control fell through to submodule-update/configure on the known-
                // partial checkout. That never converges: every retry re-enters this
                // same branch, and after max_build_failures every job on the SHA is
                // terminally failed with a confusing cmake error instead of the
                // permission/busy-mount problem that actually caused it.
                return worker_result{task_kind::build, "", sha, false,
                                     "could not remove the incomplete worktree " +
                                     wt.string() + ": " + ec.message()};
            }
            // PRUNE, as gc_task does. `worktree remove` fails outright when the
            // prior `add` was killed after git wrote .git/worktrees/<sha> but before
            // the tree's own .git file existed ("... is not a working tree"). The
            // registration then survives the remove_all above, and every subsequent
            // `worktree add --detach` is refused forever ("missing but already
            // registered") — with build_concurrency = 1 that occupies the single
            // build slot on every pass, i.e. the daemon-lifetime build stall the
            // timeouts exist to prevent, reached by another route. prune drops the
            // stale registration; it is a no-op when there is none.
            (void)run_git({"-C", cfg.repo_root.string(), "worktree", "prune"},
                              {}, git_to);
        }

        // Already built? cmake decides incremental staleness, but an existing
        // binary + configured tree means we can go straight to the build step.
        if (!std::filesystem::exists(wt, ec)) {
            auto r = run_git({"-C", cfg.repo_root.string(), "worktree", "add",
                                         "--detach", wt.string(), sha}, {}, git_to);
            if (!r.ok()) return fail("git worktree add", r);
            append_capped(log, "git worktree add", r.output, cfg.build_log_max_bytes);
            // Mark the checkout COMPLETE only now, atomically. A crash between here
            // and the build-ready marker still leaves a valid, reusable checkout.
            try {
                atomic_write(checkout_marker, sha);
            } catch (const std::exception& e) {
                return worker_result{task_kind::build, "", sha, false,
                                     std::string("failed to write checkout marker: ") + e.what()};
            }
        }

        // Worktrees do NOT inherit submodules — configure fails without this
        // (Demonstrate spike 02). Run inside the worktree. Network-bound (a cold
        // clone can be minutes), so it gets the generous build_timeout.
        //
        // Every submodule except the docs theme, at depth 1: each worktree clones
        // its own, so only the pinned commit is fetched. The theme
        // (docs/themes/hugo-theme-relearn, 1.1 GiB of history) is the one
        // submodule a coast build never compiles; `-c ...update=none` makes
        // `update` skip it, and is unused on a tree without it. It is excluded by
        // name, not the others listed by path, because a pathspec naming a
        // submodule the tree lacks fails the whole step, and the daemon builds any
        // tree: other SHAs, other layouts, test fixtures with no submodules.
        {
            const std::vector<std::string> args{
                "-C", wt.string(),
                "-c", "submodule.docs/themes/hugo-theme-relearn.update=none",
                "submodule", "update", "--init", "--recursive", "--depth", "1"};
            auto r = run_git(args, {}, build_to);
            if (!r.ok()) return fail("git submodule update", r);
            append_capped(log, "git submodule update", r.output, cfg.build_log_max_bytes);
        }

        // Configure. cmake FetchContent may hit the network here, hence build_timeout.
        {
            std::vector<std::string> args{"-S", wt.string(), "-B", build_dir.string(),
                                          "-DCMAKE_BUILD_TYPE=Release"};
            args.insert(args.end(), cfg.build_cmake_args.begin(), cfg.build_cmake_args.end());
            auto r = run_capture("cmake", args, {}, build_to);
            if (!r.ok()) return fail("cmake configure", r);
            append_capped(log, "cmake configure", r.output, cfg.build_log_max_bytes);
        }

        // Build just the coast target.
        {
            auto r = run_capture("cmake", {"--build", build_dir.string(), "-j",
                                           std::to_string(jobs), "--target", "coast"}, {}, build_to);
            if (!r.ok()) return fail("cmake build", r);
            append_capped(log, "cmake build", r.output, cfg.build_log_max_bytes);
        }

        if (!std::filesystem::exists(bin, ec))
            return worker_result{task_kind::build, "", sha, false,
                                 "build reported success but " + bin.string() + " is missing"};

        // Publish the readiness sentinel LAST, atomically. Only now — binary
        // present, every step succeeded — does the SHA become spawnable. Written
        // durably so a daemon restart trusts it; a failure here is a build failure
        // (the job simply rebuilds next pass) rather than a silent half-ready tree.
        try {
            atomic_write(marker, sha);
        } catch (const std::exception& e) {
            return worker_result{task_kind::build, "", sha, false,
                                 std::string("failed to publish build-ready marker: ") + e.what()};
        }

        COAST_INFO("coastd.builder", "built coast for {} at {}", sha, bin.string());
        return worker_result{task_kind::build, "", sha, true, ""};
    };
}

worker_pool::task builder::gc_task(sha_t sha) const {
    const daemon_config cfg = cfg_;
    const std::filesystem::path wt = worktree_dir(sha);
    const std::chrono::seconds  git_to = cfg_.git_op_timeout;
    return [cfg, sha, wt, git_to]() -> worker_result {
        if (!is_sha_shaped(sha)) {
            // Never let a tampered SHA reach remove_all with an escaped path.
            COAST_ERROR("coastd.builder", "refusing to gc worktree for invalid SHA '{}'", sha);
            return worker_result{task_kind::gc, "", sha, false,
                                 "invalid (non-hex) SHA '" + sha + "'"};
        }
        // Preferred: let git forget the worktree. Fall back to a raw sweep so a
        // half-registered tree can never wedge GC. Both git calls are best-effort
        // and bounded so a stuck git can't park the worker thread.
        (void)run_git({"-C", cfg.repo_root.string(), "worktree", "remove",
                                  "--force", wt.string()}, {}, git_to);
        std::error_code ec;
        std::filesystem::remove_all(wt, ec);
        (void)run_git({"-C", cfg.repo_root.string(), "worktree", "prune"}, {}, git_to);
        COAST_INFO("coastd.builder", "gc'd worktree {}", sha);
        return worker_result{task_kind::gc, "", sha, true, ""};
    };
}

} // namespace coastd
