#include "daemon/reality_probe.h"

#include <sys/statvfs.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <system_error>

#include "daemon/supervisor.h"
#include "diag/diag.h"

namespace coastd {

namespace {

// Free space available to a NON-privileged writer (f_bavail, not f_bfree — the
// reserved blocks a root-only writer could still use are not space this daemon
// or its children can count on). nullopt = could not be measured.
std::optional<std::uint64_t> free_bytes(const std::filesystem::path& p) {
    if (p.empty()) return std::nullopt;
    struct statvfs st{};
    if (::statvfs(p.c_str(), &st) != 0) return std::nullopt;   // absent root, or an
                                                               // fs that will not answer
    return static_cast<std::uint64_t>(st.f_bavail) * st.f_frsize;
}

} // namespace

reality rescan(const supervisor& sup, const daemon_config& cfg) {
    reality r;
    r.now = std::chrono::system_clock::now();

    // Worktrees present on disk, keyed by their directory name (the SHA), with
    // idle-age from the tree's mtime (used only by cool-off GC). A worktree is
    // reported READY (spawnable) only if its build-completion sentinel exists —
    // the directory alone appears seconds into a minutes-long build, so keying
    // readiness on directory presence would spawn against a not-yet-built binary.
    //
    // The scan's own error_code is read (see reality::worktrees_unreadable): a
    // failed directory_iterator construction — or a failed increment — yields
    // end(), so the loop silently reports "no worktrees, nothing built" for a root
    // that merely could not be read. That is the one misreport this scan must not
    // make. Per-ENTRY probes get their own error_code so a single entry vanishing
    // mid-walk (GC races this scan by design) is skipped rather than condemning the
    // whole pass.
    //
    // The root probe keeps its OWN error_code, because is_directory(p, ec) SETS ec
    // for a path that does not exist — measured on libstdc++, after assuming the
    // opposite cost a full fault-matrix run: sharing one error_code with the scan
    // made "worktrees root not created yet" (i.e. every first boot) look like an
    // unreadable root, which held every build forever and deadlocked the queue.
    // An absent root is ordinary and silent; anything else — EACCES on the root
    // itself, a stale mount — is genuinely unreadable and must be reported.
    std::error_code root_ec;
    const bool have_root = std::filesystem::is_directory(cfg.worktrees_root, root_ec);
    const bool root_unreadable = root_ec && root_ec != std::errc::no_such_file_or_directory;

    std::error_code ec;   // the SCAN's error, and nothing else
    if (have_root) {
        std::filesystem::directory_iterator it(cfg.worktrees_root, ec);
        const std::filesystem::directory_iterator end;
        for (; !ec && it != end; it.increment(ec)) {
            std::error_code eec;
            if (!it->is_directory(eec)) continue;   // false on error: skip this entry
            const sha_t sha = it->path().filename().string();
            // Only SHA-shaped names are worktrees. Anything else under this root —
            // an operator's scratch copy, a `.tmp` left by an interrupted move, an
            // editor backup — is not ours to manage, and reporting it as a worktree
            // makes GC try to `git worktree remove` a path git never registered.
            // That fails every cool-off pass, forever, logging an ERROR each time:
            // a permanent log loop over a directory the daemon should simply have
            // ignored. Same predicate the builder gates on at the way IN, by
            // construction (types.h is_sha_shaped) rather than by a copy.
            if (!is_sha_shaped(sha)) continue;
            r.worktrees_on_disk.insert(sha);
            if (std::filesystem::exists(it->path() / build_ready_marker, eec))
                r.builds_ready.insert(sha);
            const auto ft = std::filesystem::last_write_time(it->path(), eec);
            // file_clock::to_sys yields file_clock's native precision, which libc++
            // 22 widened to __int128 nanoseconds — no longer assignable to sys_time
            // (system_clock microseconds). Cast to sys_time's precision explicitly;
            // sub-microsecond mtime detail is irrelevant to worktree idle-timing.
            if (!eec)
                r.worktree_idle_since[sha] =
                    std::chrono::time_point_cast<sys_time::duration>(
                        std::chrono::file_clock::to_sys(ft));
        }
    }
    if (ec || root_unreadable) {
        COAST_ERROR("coastd.reality", "could not scan the worktrees root {}: {} — holding new "
                    "builds this pass rather than reporting an unreadable root as an empty one",
                    cfg.worktrees_root.string(), (ec ? ec : root_ec).message());
        r.worktrees_unreadable = true;
        // Partial results are worse than none here: a half-walked root would let
        // GC see an unreferenced subset and reclaim trees that are merely unlisted.
        r.worktrees_on_disk.clear();
        r.builds_ready.clear();
        r.worktree_idle_since.clear();
    }

    // Child liveness is the supervisor's registry — waitpid-authoritative.
    const auto live = sup.live_ids();
    for (const auto& id : live) r.child_alive[id] = true;
    r.running_count = static_cast<int>(live.size());

    // Disk headroom across the roots the daemon and its children write to: runs
    // (checkpoints + logs, by far the biggest consumer), worktrees (~238 MB per
    // build), state (small, but a full fs there costs the job store its ability to
    // record ANYTHING — the failure that loses work silently). Usually one
    // filesystem, so the three probes usually agree; taking the MINIMUM is correct
    // either way and costs ~250 ns per root.
    //
    // Unmeasurable (root not created yet at first boot, or statvfs refuses) is
    // reported as NO pressure: a daemon that blocks all work because it cannot see
    // the disk would turn an unreadable mount into a silent, permanent stall. The
    // real out-of-space failures still surface loudly at their write sites (the
    // atomic tmp+rename fence, the builder, coast itself).
    std::uint64_t min_free = std::numeric_limits<std::uint64_t>::max();
    for (const auto* root : {&cfg.runs_root, &cfg.worktrees_root, &cfg.state_root})
        if (const auto f = free_bytes(*root)) min_free = std::min(min_free, *f);

    if (min_free != std::numeric_limits<std::uint64_t>::max()) {
        r.disk_free_bytes = min_free;
        r.disk_pressure   = cfg.min_free_disk_bytes > 0 && min_free < cfg.min_free_disk_bytes;
    }

    return r;
}

} // namespace coastd
