#ifndef COAST_COASTD_ATOMIC_WRITE_H
#define COAST_COASTD_ATOMIC_WRITE_H

// -----------------------------------------------------------------------------
// coastd::atomic_write — durable, atomic file replace.
//
// Write to a same-directory tmp, fsync it, close it (checked), rename over the
// target, then fsync the containing directory so the rename itself survives a
// crash. A reader never observes a torn file; a crash never corrupts the target;
// after a successful return the new contents are durable across power loss.
//
// Throws std::system_error on any failure that could lose data (open/write/fsync/
// close of the data file, or the rename). A failed durable write is SERIOUS
// (IDD §10) and is never swallowed. Directory-fsync failure — a weaker,
// rename-durability guarantee some filesystems cannot honour — is logged, not
// thrown: the data itself is already safe on the target inode.
//
// Shared by the job store (authoritative small state files) and the supervisor's
// per-run config.json, so both get identical crash-safe write semantics.
// -----------------------------------------------------------------------------

#include <filesystem>
#include <string_view>

namespace coastd {

void atomic_write(const std::filesystem::path& path, std::string_view bytes);

} // namespace coastd

#endif // COAST_COASTD_ATOMIC_WRITE_H
