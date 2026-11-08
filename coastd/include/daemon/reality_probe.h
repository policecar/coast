#ifndef COAST_COASTD_REALITY_PROBE_H
#define COAST_COASTD_REALITY_PROBE_H

// -----------------------------------------------------------------------------
// coastd::rescan — gather `reality` fresh from the world, every loop.
//
// reality is NEVER cached: worktrees come from a directory scan, child liveness
// from the supervisor's registry (waitpid-authoritative), worktree idle-age from
// filesystem mtime. A derived quantity cannot desync across a crash, which is why
// boot recovery is just "rescan, then reconcile" — the same path as steady state.
// -----------------------------------------------------------------------------

#include "daemon/types.h"

namespace coastd {

class supervisor;

[[nodiscard]] reality rescan(const supervisor& sup, const daemon_config& cfg);

} // namespace coastd

#endif // COAST_COASTD_REALITY_PROBE_H
