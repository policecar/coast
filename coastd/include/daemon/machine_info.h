#ifndef COAST_COASTD_MACHINE_INFO_H
#define COAST_COASTD_MACHINE_INFO_H

// -----------------------------------------------------------------------------
// make_machine_summary — the daemon's static self-description, built once from
// the immutable daemon_config. Shared by BOTH client-facing edges that answer
// "who are you": the TCP `machine_info` command and the UDP discovery responder.
// Keeping it here (not duplicated in each edge) means hostname/OS/version/roots
// are reported identically no matter which channel a client asks over.
// -----------------------------------------------------------------------------

#include <string_view>

#include "daemon/protocol.h"  // proto::machine_summary
#include "daemon/types.h"     // daemon_config

namespace coastd {

proto::machine_summary make_machine_summary(const daemon_config& cfg, std::string_view version);

} // namespace coastd

#endif // COAST_COASTD_MACHINE_INFO_H
