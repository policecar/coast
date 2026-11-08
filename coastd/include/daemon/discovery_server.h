#ifndef COAST_COASTD_DISCOVERY_SERVER_H
#define COAST_COASTD_DISCOVERY_SERVER_H

// -----------------------------------------------------------------------------
// coastd::discovery_server — UDP Model-P responder (soft convenience).
//
// The DETERMINISTIC way a client finds a daemon is the fixed host list in the
// client's own config; this responder is the optional layer on top. It is QUIET
// by default: it never announces itself, it only answers a probe. A client
// broadcasts a one-line probe to udp/51901; a daemon that recognises the probe
// token replies (unicast, to the sender) with a REDACTED machine_summary — the
// same struct the TCP `machine_info` command returns, minus the filesystem paths
// (see below). No core hop, no state — a pure static-info echo, so a discovery
// flood can never perturb the scheduler.
//
// SECURITY. The trust boundary is the same accepted one as the TCP command port
// (see the CONSCIOUS DESIGN DECISION note in comms_server.cpp), but this channel
// has one property the TCP side does not: a UDP source address is trivially
// forged, so the "attacker" and the party RECEIVING the reply need not be the same
// host. "They could have submitted anyway" therefore does not excuse it — that
// argument covers the prober, not the victim. Hence the following are implemented,
// not deferred (this closes design/coastd_cleanup_todo.md #19):
//
//   * Reply rate caps, global and per-source (daemon_config). The global cap is
//     the amplification bound; the per-source cap bounds what any single forged —
//     i.e. victim — address can be made to receive. Measured: ~23M probes/s in
//     yields ~4.4 KB/s out, aggregate gain far below 1.
//   * runs_root/repo_root are STRIPPED from this reply. The TCP command still
//     returns them; that path at least required a handshake. What survives is
//     hostname, OS, version, proto_version, slots, tcp_port, data_link_ok.
//   * discovery_bind restricts the responder to one interface. An unparseable
//     value DISABLES discovery rather than falling back to INADDR_ANY.
//   * discovery_enabled turns it off entirely. Default ON — on the intended
//     trusted-LAN target the remaining reply is a service banner, and the GUI's
//     discover button depends on it. Turn it OFF where untrusted peers can reach
//     the port. See INSTALL.md §6.3 and §9.
//
// Known and accepted: a flood can exhaust the global cap and starve legitimate
// probes. Discovery is a convenience layer degrading — the deterministic
// fixed-host path and the TCP command port are unaffected.
//
// Binding is best-effort: if the UDP port is unavailable the daemon logs a
// warning and runs on WITHOUT discovery (fixed-list connect is unaffected).
// -----------------------------------------------------------------------------

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

#include "daemon/types.h"  // daemon_config

namespace coastd {

class discovery_server {
public:
    discovery_server(daemon_config cfg, std::string version);
    ~discovery_server();

    discovery_server(const discovery_server&)            = delete;
    discovery_server& operator=(const discovery_server&) = delete;

    // Best-effort: binds udp/udp_port and spawns the responder. On bind failure it
    // logs and returns without a running responder (never throws — soft layer).
    void start();
    void stop();   // idempotent

    // The UDP port actually bound — cfg.udp_port, or the kernel's choice when that
    // was 0. 0 while no responder runs (disabled, or start() failed to bind).
    [[nodiscard]] std::uint16_t port() const noexcept { return port_; }

private:
    void listen_loop();

    const daemon_config cfg_;
    const std::string   version_;

    int               fd_   = -1;
    std::uint16_t     port_ = 0;   // written by start() before the listen thread exists
    std::atomic<bool> running_{false};
    std::thread       thread_;
};

} // namespace coastd

#endif // COAST_COASTD_DISCOVERY_SERVER_H
