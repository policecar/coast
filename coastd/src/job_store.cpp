#include "daemon/job_store.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <type_traits>   // is_nothrow_move_assignable_v (try_put's guarantee)
#include <thread>
#include <utility>

#include <nlohmann/json.hpp>

#include "daemon/atomic_write.h"
#include "daemon/job_json.h"
#include "daemon/proc_util.h"
#include "diag/diag.h"

namespace coastd {

namespace {

constexpr std::array<bucket, 5> all_buckets{
    bucket::active, bucket::done, bucket::failed, bucket::cancelled, bucket::quarantine};

// load_all's duplicate rule ("active always beats a terminal duplicate") is written
// to be scan-order INDEPENDENT, but with active scanned first its new_wins branch
// is in fact unreachable: the already-loaded entry is the active one, never the
// newly-found one. That is a fine state of affairs — the branch is a safety net,
// not dead weight — provided the dependency is explicit rather than accidental.
// Reordering this array makes the branch live, and this assert is the prompt to
// re-read the rule before doing so.
static_assert(all_buckets[0] == bucket::active,
              "load_all's duplicate resolution assumes active/ is scanned FIRST; if you "
              "reorder, re-check the new_wins branch — it stops being unreachable");

// "I could not READ this file" and "this file's CONTENT is not a job" are opposite
// diagnoses with opposite remedies, and this function used to conflate them: it
// ignored the stream state entirely, so an unopenable file yielded "" — which then
// failed to PARSE and was quarantined as corrupt. A job file whose bytes are
// perfectly intact would be moved aside for a reason that has nothing to do with
// its bytes. Because the failure is environmental it is rarely isolated: fd
// exhaustion, an NFS EIO/ESTALE, or an operator's `sudo` edit changing ownership
// hits every file in the walk, so ONE pass could quarantine the entire queue,
// abandoning children that are still running and leaving nothing behind to account
// for them.
//
// Reported as a distinct type so the caller can tell the two apart. It refuses to
// boot, exactly as an unreadable bucket DIRECTORY already does 60 lines below — the
// precedent this case should have followed from the start.
//
// Detection is exact for the case that matters and best-effort beyond it: an open
// failure is unambiguous (`!in`), while a mid-read error can only be spotted via
// badbit, since libstdc++'s filebuf reports a failed read() as end-of-input and is
// then indistinguishable from a legitimately short file. An EMPTY file is
// deliberately NOT an error here: zero bytes is a real (truncated-write) corruption
// and belongs in quarantine, which is where the parse failure will put it.
struct file_read_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

std::string read_file(const std::filesystem::path& path) {
    errno = 0;
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw file_read_error("cannot open " + path.string() + ": " + std::strerror(errno));
    std::ostringstream ss;
    ss << in.rdbuf();
    if (in.bad())
        throw file_read_error("I/O error while reading " + path.string());
    return ss.str();
}

// Run `git -C <cwd> <args...>`, output discarded. Returns the child's exit
// code, or -1 if the process could not be spawned OR was killed for exceeding
// `timeout`. Best-effort by design: git here is an AUDIT trail, not the
// correctness path — a stale index.lock / NFS stall / hung hook must not wedge
// the committer, so a timeout that logs and continues beats a hang. The bound is
// operator-tunable (cfg_.audit_git_timeout) and passed in; it is generous so a
// healthy-but-slow repo is never killed spuriously.
//
// A thin wrapper over coastd::run_git, which owns the whole subprocess discipline
// this call needs: own process group (the timeout kill reaches a pager /
// credential helper / hook child that could otherwise hold index.lock), bounded
// post-SIGKILL reap (a D-state child must not park the committer — or, via
// init_store, STARTUP), the git-prompt env scrub, and the hooks/config hardening.
// An earlier hand-rolled copy of that machinery lived here and had to re-fix each
// of those points. Named `audit_git` rather than `run_git` so the unqualified call
// below cannot silently resolve to this wrapper instead of the hardened seam.
int audit_git(const std::filesystem::path& cwd, const std::vector<std::string>& args,
            std::chrono::seconds timeout) {
    std::vector<std::string> full{"-C", cwd.string()};
    full.insert(full.end(), args.begin(), args.end());
    const exec_result r = run_git(full, {}, timeout);
    if (r.timed_out)
        COAST_WARN("coastd.store", "git timed out after {}s; killed (audit "
                   "commit skipped, run continues)", timeout.count());
    if (!r.spawned || r.timed_out) return -1;
    return r.exit_code;   // already -1 if the child was signaled
}

// Best-effort fsync of a directory so an entry renamed into/out of it is durable
// across an unclean reboot. Mirrors atomic_write's dir fsync: this is a weaker
// guarantee than the job file's own atomic write, and the bucket is only a cache
// reconcile re-derives from durable job content, so a failure here is logged, not
// fatal.
void fsync_dir(const std::filesystem::path& dir) {
    const int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dfd < 0) {
        COAST_WARN("coastd.store", "move_bucket: cannot open dir {} to fsync ({})",
                   dir.string(), std::strerror(errno));
        return;
    }
    if (::fsync(dfd) != 0)
        COAST_WARN("coastd.store", "move_bucket: dir fsync {} failed ({})",
                   dir.string(), std::strerror(errno));
    ::close(dfd);
}

} // namespace

job_store::job_store(daemon_config cfg) : cfg_(std::move(cfg)) {
    // Start the audit committer. It idles on the condition variable until commit()
    // enqueues a message, so a store nobody commits to spawns no git at all.
    committer_ = std::jthread([this](std::stop_token st) { committer_loop(st); });
}

std::filesystem::path job_store::jobs_root() const { return cfg_.state_root / "jobs"; }
std::filesystem::path job_store::bucket_path(bucket b) const {
    return jobs_root() / bucket_dir(b);
}
std::filesystem::path job_store::file_path(const job_id& id, bucket b) const {
    return bucket_path(b) / (id + ".json");
}

void job_store::init_store() {
    std::error_code ec;
    for (bucket b : all_buckets) std::filesystem::create_directories(bucket_path(b), ec);
    if (!std::filesystem::exists(cfg_.state_root / ".git")) {
        if (audit_git(cfg_.state_root, {"init", "--quiet"}, cfg_.audit_git_timeout) != 0)
            COAST_WARN("coastd.store", "git init failed in {} (git audit trail disabled)",
                       cfg_.state_root.string());
    }
    // Ensure a committer identity exists in THIS repo. On a box with no global
    // user.name/user.email, `git commit` fails permanently and the audit trail is
    // silently lost. Set a daemon-private local identity only when absent (so an
    // operator's own local config is respected). `git config <key>` exits non-zero
    // when the key is unset.
    if (audit_git(cfg_.state_root, {"config", "user.email"}, cfg_.audit_git_timeout) != 0)
        audit_git(cfg_.state_root, {"config", "user.email", "coastd@localhost"}, cfg_.audit_git_timeout);
    if (audit_git(cfg_.state_root, {"config", "user.name"}, cfg_.audit_git_timeout) != 0)
        audit_git(cfg_.state_root, {"config", "user.name", "coastd"}, cfg_.audit_git_timeout);
}

void job_store::load_all() {
    entries_.clear();
    // Losing duplicate files, quarantined AFTER the walk — renaming a file while
    // iterating its own bucket's directory_iterator is unspecified behaviour.
    std::vector<std::pair<job_id, bucket>> strays;
    // Un-interpretable files, quarantined after the walk for the SAME reason.
    std::vector<std::pair<std::filesystem::path, bucket>> unreadable;
    for (bucket b : all_buckets) {
        const auto dir = bucket_path(b);
        std::error_code ec;
        // A bucket dir that does not exist yet is normal (first boot, or a bucket
        // nothing has landed in). One that exists but cannot be STATTED is not.
        // MEASURED, not assumed: libstdc++'s is_directory(p, ec) sets ec to ENOENT
        // for a missing path rather than clearing it, so "ec is set" alone would
        // make every first boot fatal. Match on the condition instead.
        if (!std::filesystem::is_directory(dir, ec)) {
            if (ec && ec != std::errc::no_such_file_or_directory)
                throw std::runtime_error("coastd: cannot stat job bucket " + dir.string()
                                         + ": " + ec.message());
            continue;
        }

        for (const auto& de : std::filesystem::directory_iterator(dir, ec)) {
            if (!de.is_regular_file()) continue;
            const auto& path = de.path();
            if (path.extension() != ".json") continue;

            const std::string id = path.stem().string();  // the filename IS the id
            try {
                job j = parse_job(read_file(path));
                j.id  = id;                                // filename is authoritative

                // A job id must live in exactly ONE bucket (transitions are atomic
                // renames). Two files with the same id => a crash mid-move or a
                // manual edit left a stray. Rule: an ACTIVE file always wins over a
                // terminal duplicate — the terminal one is the stale pre-move copy;
                // between two terminals, first-loaded wins. Then QUARANTINE the
                // loser's file (after the walk) so the condition self-heals instead
                // of re-erroring on every boot.
                //
                // The rule is WRITTEN order-independently, but with active scanned
                // first (see the static_assert on all_buckets) new_wins is currently
                // unreachable: the already-loaded entry is the active one. Kept as a
                // safety net so the rule still holds if that order ever changes.
                if (auto ex = entries_.find(id); ex != entries_.end()) {
                    const bool new_wins = (b == bucket::active && ex->second.b != bucket::active);
                    const bucket loser = new_wins ? ex->second.b : b;
                    COAST_ERROR("coastd.store",
                                "duplicate job id {} in buckets '{}' and '{}'; keeping '{}', "
                                "quarantining the stray in '{}'",
                                id, bucket_dir(ex->second.b), bucket_dir(b),
                                bucket_dir(new_wins ? b : ex->second.b), bucket_dir(loser));
                    strays.emplace_back(id, loser);
                    if (!new_wins) continue;   // discard the new (losing) file from the view
                    // new (active) wins: it replaces the previously-loaded terminal.
                }
                entries_[id] = entry{std::move(j), b};
            } catch (const file_read_error& e) {
                // NOT a content problem — see file_read_error. Quarantining here
                // would move aside a job whose bytes may be perfectly good, and the
                // cause (fds, mount, permissions) almost certainly affects the rest
                // of the walk too. Refuse to boot, like an unreadable bucket dir.
                throw std::runtime_error(std::string("coastd: ") + e.what()
                                         + " — refusing to boot rather than quarantine a job "
                                           "that may be intact");
            } catch (const std::exception& e) {
                // Un-interpretable file: quarantine untouched (never mangle), skip.
                if (b == bucket::quarantine) {
                    COAST_WARN("coastd.store", "unreadable file in quarantine, skipping: {} ({})",
                               path.string(), e.what());
                    continue;
                }
                COAST_ERROR("coastd.store", "quarantining unreadable job {}: {}",
                            path.string(), e.what());
                // DEFERRED to after the walk, like the duplicate strays above and
                // for the identical reason: renaming an entry out of the directory
                // this loop is iterating leaves readdir's cursor unspecified, and
                // glibc's getdents buffering can then skip the FOLLOWING entry --
                // silently dropping a healthy job whose child is still running.
                unreadable.emplace_back(path, b);
            }
        }
        // An iterator that failed to construct compares equal to end(), so the loop
        // above just does not run: an unreadable active/ would report "loaded 0
        // job(s)" and start the daemon with an EMPTY queue while its children are
        // still running — the store would then never account for them, and a later
        // put() could resurrect ids it thinks are new. Refuse to boot instead; this
        // runs only at startup, where failing loudly is the recoverable outcome.
        if (ec)
            throw std::runtime_error("coastd: cannot read job bucket " + dir.string()
                                     + ": " + ec.message());
    }

    // Quarantine the un-interpretable files now that the walk is done. Keeps the
    // original destination semantics: quarantine/<id>.json, moved untouched (never
    // mangled) so it can be inspected as-is.
    for (const auto& [path, from] : unreadable) {
        std::error_code qec;
        std::filesystem::create_directories(bucket_path(bucket::quarantine), qec);
        // De-collide, exactly as the strays loop below already does. A plain
        // rename to quarantine/<id>.json OVERWRITES silently, and the thing being
        // overwritten is a previous quarantine of the same id — i.e. the evidence
        // from the earlier failure, destroyed by the later one. Quarantine exists
        // to be inspected afterwards; losing the first copy defeats it.
        auto dest = bucket_path(bucket::quarantine) / path.filename();
        for (int n = 1; std::filesystem::exists(dest, qec); ++n)
            dest = bucket_path(bucket::quarantine) /
                   (path.stem().string() + "-" + std::to_string(n) + path.extension().string());
        std::filesystem::rename(path, dest, qec);
        if (qec) {
            COAST_ERROR("coastd.store", "could not quarantine {}: {}",
                        path.string(), qec.message());
        } else {
            // Make the quarantine move durable in both directories, matching
            // move_bucket — otherwise an unclean reboot could resurrect the
            // un-interpretable file in its original bucket.
            fsync_dir(bucket_path(from));
            fsync_dir(bucket_path(bucket::quarantine));
        }
    }

    // Quarantine the losing duplicate files now that the walk is done. Rename each
    // to quarantine/ under a unique, non-.json-id name (so it can never re-collide
    // with a real job id nor another stray on a future boot). Best-effort: a failed
    // rename just leaves the stray to be re-flagged next boot, never loses a job.
    for (const auto& [id, from] : strays) {
        std::error_code ec;
        std::filesystem::create_directories(bucket_path(bucket::quarantine), ec);
        // ".quarantined" (not ".json") so load_all skips it forever after — it is
        // an inspection artifact, never a loadable job with a mangled id.
        const std::string base = id + ".dup-" + std::string(bucket_dir(from));
        std::filesystem::path dest = bucket_path(bucket::quarantine) / (base + ".quarantined");
        for (int n = 1; std::filesystem::exists(dest, ec); ++n)
            dest = bucket_path(bucket::quarantine) / (base + "-" + std::to_string(n) + ".quarantined");
        std::filesystem::rename(file_path(id, from), dest, ec);
        if (ec) {
            COAST_ERROR("coastd.store", "could not quarantine stray {} from '{}': {}",
                        id, bucket_dir(from), ec.message());
        } else {
            fsync_dir(bucket_path(from));
            fsync_dir(bucket_path(bucket::quarantine));
            COAST_INFO("coastd.store", "quarantined stray {} -> {}", id, dest.string());
        }
    }

    COAST_INFO("coastd.store", "loaded {} job(s) from {}", entries_.size(), jobs_root().string());
}

std::size_t job_store::sweep_stale_tmp() {
    std::size_t removed = 0;
    for (bucket b : all_buckets) {
        const auto dir = bucket_path(b);
        std::error_code ec;
        if (!std::filesystem::is_directory(dir, ec)) continue;
        for (const auto& de : std::filesystem::directory_iterator(dir, ec)) {
            if (!de.is_regular_file()) continue;
            if (de.path().extension() != ".tmp") continue;  // only atomic_write's leftovers
            std::error_code rec;
            if (std::filesystem::remove(de.path(), rec)) {
                ++removed;
                COAST_INFO("coastd.store", "swept stale tmp {}", de.path().string());
            }
        }
    }
    return removed;
}

std::vector<job> job_store::active_jobs() const {
    std::vector<job> out;
    for (const auto& [id, e] : entries_)
        if (e.b == bucket::active) out.push_back(e.j);
    return out;
}

std::vector<std::pair<job, bucket>> job_store::all_jobs() const {
    std::vector<std::pair<job, bucket>> out;
    out.reserve(entries_.size());
    for (const auto& [id, e] : entries_) out.emplace_back(e.j, e.b);
    // entries_ is an unordered_map, so iteration order is unspecified and rehash-
    // unstable. Return a DEFINED order so every client (list RPC, CLI, GUI) shows
    // the real queue: active bucket first, and within a bucket the scheduler's own
    // order (priority desc, then submitted_at asc / FIFO). The active-bucket order
    // then matches do_reorder's pending indexing exactly.
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second < b.second;  // bucket enum: active(0) first
        return job_queue_less(a.first, b.first);
    });
    return out;
}

std::optional<bucket> job_store::bucket_of(const job_id& id) const {
    auto it = entries_.find(id);
    if (it == entries_.end()) return std::nullopt;
    return it->second.b;
}

std::optional<job> job_store::get(const job_id& id) const {
    auto it = entries_.find(id);
    if (it == entries_.end()) return std::nullopt;
    return it->second.j;
}

void job_store::put(const job& j) {
    auto it = entries_.find(j.id);
    const bucket b = (it == entries_.end()) ? bucket::active : it->second.b;
    std::error_code ec;
    std::filesystem::create_directories(bucket_path(b), ec);
    atomic_write(file_path(j.id, b), dump_job(j));
    entries_[j.id] = entry{j, b};
}

bool job_store::try_put(const job& j) noexcept {
    auto it = entries_.find(j.id);
    if (it == entries_.end()) return false;   // never create a phantom (no durable record)
    const bucket b = it->second.b;
    bool durable = true;
    try {
        std::error_code ec;
        std::filesystem::create_directories(bucket_path(b), ec);
        atomic_write(file_path(j.id, b), dump_job(j));
    } catch (const std::exception& e) {
        durable = false;
        COAST_ERROR("coastd.store", "try_put durable write failed for {} ({}); "
                    "cache updated, disk left stale", j.id, e.what());
    }
    // Refreshing the cache is itself a throwing operation — `job` is a dozen strings,
    // so copy-assignment allocates — and it sat OUTSIDE the try, in a function marked
    // noexcept. A bad_alloc here did not degrade anything: it called std::terminate.
    // This is the call site chosen precisely BECAUSE the daemon must not fail here,
    // so it was unsafe under the one condition it exists to survive.
    //
    // Copy first, then move into place. The copy is the only step that can throw, and
    // the move is noexcept (asserted below), so the cache is either fully updated or
    // untouched — never half-assigned, which direct assignment would have left behind
    // after a throw partway through the member strings. A stale-but-coherent cached
    // job is recoverable; one with a new id and an old status is not.
    static_assert(std::is_nothrow_move_assignable_v<job>,
                  "try_put's strong guarantee depends on job's move-assign not throwing");
    try {
        job refreshed = j;
        it->second.j = std::move(refreshed);
    } catch (const std::exception& e) {
        COAST_ERROR("coastd.store", "try_put cache refresh failed for {} ({}); "
                    "cache left at its previous value", j.id, e.what());
        return false;   // neither disk nor cache reflects `j`
    }
    return durable;
}

void job_store::move_bucket(const job_id& id, bucket to) {
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        COAST_WARN("coastd.store", "move_bucket on unknown job {}", id);
        return;
    }
    const bucket from = it->second.b;
    if (from == to) return;

    std::error_code ec;
    std::filesystem::create_directories(bucket_path(to), ec);
    std::filesystem::rename(file_path(id, from), file_path(id, to), ec);
    if (ec) {
        COAST_ERROR("coastd.store", "move {} {} -> {} failed: {}", id,
                    bucket_dir(from), bucket_dir(to), ec.message());
        return;
    }
    // Make the rename durable in both directories so an unclean reboot cannot
    // leave the job's file visible in neither (or, via a replayed rename, both)
    // bucket dirs.
    fsync_dir(bucket_path(from));
    fsync_dir(bucket_path(to));
    it->second.b = to;
}

void job_store::commit(std::string_view message) {
    // Fire-and-forget: hand the message to the committer thread and return. The
    // core thread must never block on git (a slow commit — stale index.lock, cold
    // NFS, a hook — would freeze reconcile, kill-escalation, and every queued RPC).
    //
    // BOUNDED. The queue coalesces (one `git add -A` + one commit covers a whole
    // batch), so it drains fast in the normal case — but the core thread appends a
    // message per lifecycle transition regardless of whether the committer is
    // making progress, so a git that is slow or stalled makes this the one
    // unbounded buffer the core thread feeds. These messages are an AUDIT
    // annotation, not state (the filesystem is truth), so dropping the oldest is
    // the correct sacrifice; the drop itself is logged so the gap is never silent.
    constexpr std::size_t max_pending_commits = 10'000;
    std::size_t dropped = 0;
    {
        std::lock_guard lk(commit_mu_);
        while (commit_queue_.size() >= max_pending_commits) {
            commit_queue_.pop_front();
            ++dropped;
        }
        commit_queue_.emplace_back(message);
    }
    if (dropped > 0)
        COAST_WARN("coastd.store", "audit commit backlog exceeded {}; dropped {} oldest "
                   "message(s) (the job files themselves are unaffected)",
                   max_pending_commits, dropped);
    commit_cv_.notify_one();
}

void job_store::committer_loop(std::stop_token st) {
    for (;;) {
        std::vector<std::string> batch;
        {
            std::unique_lock lk(commit_mu_);
            // Wake on a queued message OR a stop request. On stop with an empty
            // queue we exit; a message enqueued right before stop is still drained
            // (we loop once more and the predicate returns immediately).
            commit_cv_.wait(lk, st, [&] { return !commit_queue_.empty(); });
            if (commit_queue_.empty()) return;
            batch.assign(std::make_move_iterator(commit_queue_.begin()),
                         std::make_move_iterator(commit_queue_.end()));
            commit_queue_.clear();
        }
        do_git_commit(batch);   // git runs OUTSIDE the lock, off the core thread
    }
}

void job_store::do_git_commit(const std::vector<std::string>& messages) {
    if (messages.empty()) return;
    if (audit_git(cfg_.state_root, {"add", "-A"}, cfg_.audit_git_timeout) != 0) {
        COAST_DEBUG("coastd.store", "git add failed (benign; FS is truth)");
        return;
    }
    // `git add -A` already staged the ENTIRE tree, so however many requests were
    // coalesced into this batch, one commit records them all. Keep the first
    // message and note how many more it stands in for. `commit` returns non-zero
    // when there is nothing to commit — benign.
    std::string msg = messages.front();
    if (messages.size() > 1)
        msg += " (+" + std::to_string(messages.size() - 1) + " more)";
    audit_git(cfg_.state_root, {"commit", "-m", msg, "--quiet"}, cfg_.audit_git_timeout);
}

} // namespace coastd
