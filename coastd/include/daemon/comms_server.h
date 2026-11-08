#ifndef COAST_COASTD_COMMS_SERVER_H
#define COAST_COASTD_COMMS_SERVER_H

// -----------------------------------------------------------------------------
// coastd::comms_server — the TCP client edge.
//
// Thread-per-connection, blocking sockets, JSONL (one minified JSON object per
// line). Connection threads are pure EDGES: a handler reaches the core only
// through its thread-safe RPC seam (submit/list/cancel/reorder), which marshals
// a closure onto the single-writer core loop and hands back a future the handler
// blocks on. So the comms layer adds NO new state and NO second writer — it just
// turns wire lines into core RPCs and RPC results back into wire lines.
//
// machine_info is served straight from the (immutable-after-boot) daemon_config,
// with NO core hop — it is a static capability description, not live state.
//
// Shutdown is ORDERED by design: stop() must run while the core is STILL alive,
// so any in-flight handler future resolves before the core loop exits. stop()
// closes the listen socket (breaks accept), shuts down every live connection fd
// (breaks a blocking recv on an idle keep-alive connection), then joins. A
// client error — malformed JSON, an unknown command, a mid-write disconnect —
// is answered with an {"ok":false,...} line or a dropped connection, NEVER a
// daemon crash (A10). SIGPIPE must be ignored process-wide (main does this).
// -----------------------------------------------------------------------------

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "daemon/types.h"  // daemon_config

namespace coastd {

class core;

class comms_server {
public:
    comms_server(core& c, daemon_config cfg, std::string version);
    ~comms_server();

    comms_server(const comms_server&)            = delete;
    comms_server& operator=(const comms_server&) = delete;

    // Bind + listen on cfg.tcp_port and spawn the accept thread. Throws
    // std::runtime_error on bind/listen failure (e.g. port already in use) — the
    // caller decides whether an occupied port is fatal. A tcp_port of 0 lets the
    // kernel choose; port() then says which it chose.
    void start();

    // The port actually listened on — cfg.tcp_port, or the kernel's choice when
    // that was 0. 0 before a successful start(). machine_info advertises THIS, so a
    // client never gets told "connect to port 0".
    [[nodiscard]] std::uint16_t port() const noexcept { return port_; }

    // Stop accepting, break every live connection, join all threads. Idempotent.
    // MUST be called before the core is stopped so in-flight futures can resolve.
    void stop();

private:
    void accept_loop();
    // Thread entry. A pure exception firewall (nothing may escape a thread entry);
    // the connection's actual work is serve_conn_impl.
    void serve_conn(int fd, std::uint64_t conn_id);
    void serve_conn_impl(int fd, std::uint64_t conn_id);
    void reap_finished();   // join+erase handler threads that have returned
    std::string handle_line(const std::string& line);  // one request line -> one reply line

    core&               core_;
    const daemon_config cfg_;      // immutable snapshot: machine_info + validation roots
    const std::string   version_;

    int               listen_fd_ = -1;
    std::uint16_t     port_      = 0;   // written by start() before the accept thread exists
    std::atomic<bool> running_{false};
    std::thread       accept_thread_;

    // Connection bookkeeping. Threads are keyed by a monotonic conn_id (NOT the
    // fd) so a closed-then-reused fd number can never collide with a still-tracked
    // entry. live_fds_ is the set of fds a running handler owns, used only to break
    // their blocking recv on stop(). finished_ lets a returning handler hand its
    // thread back for joining without joining itself.
    std::mutex                                 conns_mu_;
    std::uint64_t                              next_conn_id_ = 0;
    std::unordered_map<std::uint64_t, std::thread> conns_;
    std::unordered_set<int>                    live_fds_;
    std::vector<std::uint64_t>                 finished_;
};

} // namespace coastd

#endif // COAST_COASTD_COMMS_SERVER_H
