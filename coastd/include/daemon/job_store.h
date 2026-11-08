#ifndef COAST_COASTD_JOB_STORE_H
#define COAST_COASTD_JOB_STORE_H

// -----------------------------------------------------------------------------
// coastd::job_store — the durable, authoritative persistence model.
//
// A git-versioned directory of one-file-per-job under <state_root>/jobs/<bucket>/.
// The FILESYSTEM is truth; git is an audit trail that MAY LAG (a crash between a
// rename and its commit leaves FS state correct and one commit missing — boot
// recovery re-commits catch-up). Git is never on the correctness path.
//
//   <state_root>/jobs/active/<id>.json     non-terminal
//   <state_root>/jobs/{done,failed,cancelled,quarantine}/<id>.json   terminal
//
// A lifecycle transition is an ATOMIC rename across bucket directories (content
// unchanged — `status` inside the file is advisory; the directory is truth). A
// content update (put) is an atomic tmp -> fsync -> rename -> fsync(dir) write,
// so a reader never sees a torn file and a crash never corrupts one.
//
// Robustness contract: a file that cannot be parsed, or declares a
// schema_version this daemon cannot interpret, is MOVED to quarantine/ untouched
// and skipped — never silently mangled, never able to crash-loop the queue.
// -----------------------------------------------------------------------------

#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "daemon/types.h"

namespace coastd {

class job_store {
public:
    explicit job_store(daemon_config cfg);

    // Create <state_root>/jobs/<bucket>/ directories, and `git init` the state
    // root if it is not already a repo. Best-effort on git; idempotent.
    void init_store();

    // Scan every bucket directory and load its jobs into memory. Unparseable or
    // unsupported-version files are moved to quarantine/ and skipped.
    void load_all();

    // Boot hygiene: remove leftover `*.tmp` files across the bucket dirs (a crash
    // between atomic_write's open and rename leaves one). `*.json` is never
    // touched. Returns the number removed. Idempotent.
    std::size_t sweep_stale_tmp();

    // The active (non-terminal) jobs — reconcile's input.
    [[nodiscard]] std::vector<job> active_jobs() const;

    // Every job with its bucket, for read-only queries (list).
    [[nodiscard]] std::vector<std::pair<job, bucket>> all_jobs() const;

    [[nodiscard]] std::size_t size() const { return entries_.size(); }
    [[nodiscard]] std::optional<bucket> bucket_of(const job_id& id) const;
    [[nodiscard]] std::optional<job>    get(const job_id& id) const;

    // Upsert: atomically write `j` into its current bucket (or active/ if new).
    void put(const job& j);

    // Crash-resilient UPDATE of an already-known job. Attempts the same durable
    // atomic write as put(), but never throws and ALWAYS refreshes the in-memory
    // cache; returns false if the durable write failed. Used where a lost update
    // would WEDGE the daemon rather than merely miss a persist — e.g. clearing a
    // child's pid on exit: if that write fails and the cache keeps a stale nonzero
    // pid, the spawn dispatcher's skew guard blocks the resume forever. Keeping
    // the cache correct lets the job proceed; the next successful put re-persists,
    // and a crash before that is healed by boot recovery re-deriving from disk.
    // NEVER creates a new entry (a create with no durable record would be a phantom
    // job), so an unknown id is a no-op returning false.
    bool try_put(const job& j) noexcept;

    // Atomically move a job's file to another bucket. No content change.
    void move_bucket(const job_id& id, bucket to);

    // Request a git audit commit. ASYNCHRONOUS: enqueues the message and returns
    // immediately, so the CALLER (the core thread) never blocks on git. A dedicated
    // committer thread serializes the actual `git add -A && git commit` — git is an
    // audit trail (FS is truth), so it must never stall the single-writer core, and
    // serialization avoids two concurrent gits racing on .git/index.lock. Failure
    // is benign (logged; boot catch-up re-commits). Queued requests coalesce: one
    // `git add -A` captures the whole tree, so a burst collapses to one commit.
    void commit(std::string_view message);

private:
    struct entry { job j; bucket b; };

    [[nodiscard]] std::filesystem::path jobs_root() const;
    [[nodiscard]] std::filesystem::path bucket_path(bucket b) const;
    [[nodiscard]] std::filesystem::path file_path(const job_id& id, bucket b) const;

    // The committer thread's loop + the actual git call it drives (see commit()).
    void committer_loop(std::stop_token st);
    void do_git_commit(const std::vector<std::string>& messages);

    daemon_config                     cfg_;
    std::unordered_map<job_id, entry> entries_;

    // Audit-commit pipeline. Producers (core thread) push messages under the mutex;
    // the single committer_ thread drains + coalesces them off the hot path.
    std::mutex                        commit_mu_;
    std::condition_variable_any       commit_cv_;
    std::deque<std::string>           commit_queue_;
    // Declared LAST so it is destroyed FIRST: the jthread's destructor requests
    // stop and joins while cfg_/queue/cv are still alive for its final drain.
    std::jthread                      committer_;
};

} // namespace coastd

#endif // COAST_COASTD_JOB_STORE_H
