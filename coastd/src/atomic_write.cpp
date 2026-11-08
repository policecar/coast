#include "daemon/atomic_write.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <functional>
#include <system_error>
#include <thread>

#include "diag/diag.h"

namespace coastd {

void atomic_write(const std::filesystem::path& path, std::string_view bytes) {
    // Per-writer tmp name. A FIXED "<path>.tmp" made two concurrent writers to one
    // target silently interleave into the same scratch file and rename each other's
    // partial bytes over the target — safe today only by external discipline (job
    // files are core-thread-only; the two worker call sites are deduped by
    // build_in_flight_/gc_in_flight_), which is a coupling invisible from a header
    // that presents a general-purpose durable write. pid+tid keeps the name unique
    // without needing a clock or RNG, and stays inside sweep_stale_tmp's ".tmp"
    // suffix match so leftovers are still swept.
    // (hash the thread id rather than stream it: its formatted form is
    // implementation-defined and need not be filename-safe.)
    const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    const std::filesystem::path tmp =
        path.string() + ".p" + std::to_string(::getpid()) + "t" + std::to_string(tid) + ".tmp";

    // O_NOFOLLOW: refuse if the tmp path's final component is a symlink. The name is
    // per-writer (above) but it is not a secret — anyone who can list the state root
    // can predict it from the pid. Without this, pre-creating that path as a symlink
    // turns this O_CREAT|O_TRUNC into an arbitrary-file truncate-and-overwrite AS THE
    // DAEMON. It presupposes write access to the daemon's own state root, so it is a
    // hardening measure rather than a live hole — but it costs one flag.
    //
    // Deliberately NOT O_EXCL, which would look stronger and is wrong here: a crash
    // can leave a tmp file behind, and after a pid+tid collision (pid reuse is
    // routine) O_EXCL would fail the write permanently rather than reclaim the stale
    // scratch file. O_TRUNC + O_NOFOLLOW keeps the reclaim and closes the symlink.
    int fd = ::open(tmp.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC | O_NOFOLLOW, 0644);
    if (fd < 0 && errno == ELOOP) {
        // A symlink sits at the scratch name. O_NOFOLLOW correctly refused it — but
        // refusing alone is a PERMANENT failure: sweep_stale_tmp only reclaims
        // regular files (is_regular_file() follows the link, so a DANGLING symlink is
        // invisible to it), so the link would stay and this target could never be
        // written again. Adding O_NOFOLLOW without this turned an overwrite bug into
        // a denial-of-service on the same precondition.
        //
        // Unlink and retry ONCE. unlink() removes the link itself, never its target,
        // so this cannot damage whatever the link pointed at. The retry keeps
        // O_NOFOLLOW, so an attacker re-planting a link in the gap is refused again
        // and we fail loudly instead of following it — the race is unwinnable for
        // them, only repeatable.
        ::unlink(tmp.c_str());
        fd = ::open(tmp.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC | O_NOFOLLOW, 0644);
    }
    if (fd < 0)
        throw std::system_error(errno, std::generic_category(),
                                "coastd: open tmp " + tmp.string());
    try {
        std::size_t off = 0;
        while (off < bytes.size()) {
            const ssize_t n = ::write(fd, bytes.data() + off, bytes.size() - off);
            if (n < 0) {
                if (errno == EINTR) continue;
                throw std::system_error(errno, std::generic_category(),
                                        "coastd: write " + tmp.string());
            }
            off += static_cast<std::size_t>(n);
        }
        if (::fsync(fd) != 0)
            throw std::system_error(errno, std::generic_category(),
                                    "coastd: fsync " + tmp.string());
    } catch (...) {
        ::close(fd);
        std::error_code ec;
        std::filesystem::remove(tmp, ec);  // do not leak the tmp on failure
        throw;
    }
    // close() can surface a deferred write error (e.g. EIO) even after a good
    // fsync on some filesystems — a failure here can mean lost data, so it is
    // checked, not ignored.
    if (::close(fd) != 0) {
        const int e = errno;
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        throw std::system_error(e, std::generic_category(), "coastd: close " + tmp.string());
    }

    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        // Preserve the rename error: cleaning up the tmp must not clobber the
        // error_code we are about to report (a successful remove would otherwise
        // turn the thrown message into "Success").
        std::error_code rm_ec;
        std::filesystem::remove(tmp, rm_ec);
        throw std::system_error(ec, "coastd: rename " + tmp.string() + " -> " + path.string());
    }

    // fsync the containing directory so the rename entry is durable. This is a
    // weaker guarantee than the data write above (the target inode already holds
    // the new bytes) and not every filesystem honours a directory fsync, so a
    // failure here is logged rather than thrown.
    const int dfd = ::open(path.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dfd < 0) {
        COAST_WARN("coastd.store", "atomic_write: cannot open dir {} to fsync rename ({})",
                   path.parent_path().string(), std::strerror(errno));
        return;
    }
    if (::fsync(dfd) != 0)
        COAST_WARN("coastd.store", "atomic_write: dir fsync {} failed ({})",
                   path.parent_path().string(), std::strerror(errno));
    ::close(dfd);
}

} // namespace coastd
