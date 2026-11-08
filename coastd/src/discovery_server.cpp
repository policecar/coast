#include "daemon/discovery_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>   // timeval, for SO_RCVTIMEO
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <nlohmann/json.hpp>

#include "daemon/machine_info.h"
#include "daemon/proc_util.h"  // cloexec_socket, local_port
#include "daemon/protocol.h"
#include "diag/diag.h"

namespace coastd {

namespace {
// How often the listen thread's recvfrom self-wakes to re-check running_. It is
// the sole shutdown wake (see start()), so it is also the upper bound on stop()'s
// latency — measured at 157 ms for a stop() issued 50 ms into a window, i.e. the
// remainder of one interval, as designed. Small enough that shutdown feels
// instant, large enough that the idle cost is 5 syscalls/s.
constexpr suseconds_t wake_interval_us = 200'000;   // 200 ms
} // namespace

discovery_server::discovery_server(daemon_config cfg, std::string version)
    : cfg_(std::move(cfg)), version_(std::move(version)) {}

discovery_server::~discovery_server() { stop(); }

void discovery_server::start() {
    // Opt-out switch. Checked here rather than at the call site so every caller
    // (daemon main, tests, any future embedder) obeys it identically.
    if (!cfg_.discovery_enabled) {
        COAST_INFO("coastd.discovery",
                   "responder disabled by config; the TCP machine_info command still "
                   "answers, so clients configured with an address are unaffected");
        return;
    }

    fd_ = cloexec_socket(AF_INET, SOCK_DGRAM, 0);   // CLOEXEC from creation
    if (fd_ < 0) {
        COAST_WARN("coastd.discovery", "socket() failed: {} — discovery disabled", std::strerror(errno));
        return;
    }
    int one = 1;
    ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    // An explicit bind address keeps the responder off interfaces it has no business
    // on (a public NIC) without needing a firewall rule. A malformed address is a
    // REFUSAL, not a silent fallback to INADDR_ANY: falling back would quietly do the
    // opposite of what the operator asked for, on the one channel they were trying to
    // restrict.
    if (!cfg_.discovery_bind.empty()) {
        if (::inet_pton(AF_INET, cfg_.discovery_bind.c_str(), &addr.sin_addr) != 1) {
            COAST_WARN("coastd.discovery",
                       "discovery_bind '{}' is not a valid IPv4 address — discovery "
                       "disabled rather than bound to every interface",
                       cfg_.discovery_bind);
            ::close(fd_);
            fd_ = -1;
            return;
        }
    }
    addr.sin_port        = htons(cfg_.udp_port);
    if (::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof addr) < 0) {
        COAST_WARN("coastd.discovery", "bind udp/{} failed: {} — discovery disabled",
                   cfg_.udp_port, std::strerror(errno));
        ::close(fd_);
        fd_ = -1;
        return;
    }

    // Portable shutdown wake. The listen thread blocks in recvfrom; stop() clears
    // running_ but cannot interrupt that blocked call uniformly — Linux wakes a
    // recvfrom on an unconnected UDP socket via ::shutdown, but macOS/BSD do NOT, so
    // a shutdown-based wake hung the thread there forever. A receive timeout instead
    // makes recvfrom return EAGAIN at a bounded interval, so the loop re-checks
    // running_ and exits on its own — no platform-specific wake, and stop() latency
    // is bounded by this interval. A real datagram still returns at once; the timeout
    // only fires while idle.
    //
    // This is the ONLY wake, so its failure is FATAL to shutdown, not cosmetic: a
    // socket without the timeout leaves recvfrom blocked forever and stop()'s join
    // never returns. Hence the return IS checked and a failure refuses to start —
    // better no responder than one that cannot be stopped. (Contrast comms_server,
    // where SO_RCVTIMEO is a defence-in-depth read deadline and best-effort is fine.)
    timeval rcv_to{};
    rcv_to.tv_usec = wake_interval_us;
    if (::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &rcv_to, sizeof rcv_to) < 0) {
        COAST_WARN("coastd.discovery",
                   "SO_RCVTIMEO failed: {} — discovery disabled rather than started "
                   "with an unstoppable listen thread",
                   std::strerror(errno));
        ::close(fd_);
        fd_ = -1;
        return;
    }

    port_ = local_port(fd_);
    running_.store(true);
    thread_ = std::thread([this] { listen_loop(); });
    COAST_INFO("coastd.discovery", "responder on {}:{} (<= {}/s overall, {}/s per source)",
               cfg_.discovery_bind.empty() ? "0.0.0.0" : cfg_.discovery_bind,
               port_, cfg_.discovery_max_replies_per_sec,
               cfg_.discovery_max_replies_per_source_per_sec);
}

void discovery_server::listen_loop() {
    using clock = std::chrono::steady_clock;

    // Fixed-window reply caps. A probe is ~15 bytes and a reply ~250, so an
    // unthrottled responder is a ~17x amplifier pointed at whatever source address
    // the probe claims — and UDP source addresses are trivially forged, so "the
    // source" is a VICTIM, not the attacker. Two caps, because they bound different
    // things: the global one bounds how much traffic this daemon can be made to
    // emit at all; the per-source one bounds what any single victim receives.
    //
    // Fixed windows rather than a token bucket on purpose: the caps sit orders of
    // magnitude above real use (a client probes once per manual refresh), so the
    // burst behaviour a bucket would smooth is irrelevant here, and the window
    // counter has no per-source state to age out except the map below.
    auto              window_start   = clock::now();
    int               replies_in_win = 0;
    std::unordered_map<std::uint32_t, int> per_source;   // key: network-order IPv4

    std::array<char, 1024> buf{};
    while (running_.load()) {
        sockaddr_in from{};
        socklen_t   from_len = sizeof from;
        const ssize_t n = ::recvfrom(fd_, buf.data(), buf.size(), 0,
                                     reinterpret_cast<sockaddr*>(&from), &from_len);
        if (n < 0) {
            // EAGAIN/EWOULDBLOCK is the SO_RCVTIMEO poll firing (set in start()): no
            // datagram this interval, so loop back and re-check running_ — this is
            // how stop() gets the thread to exit portably. EINTR is a signal. Neither
            // is an error; anything else is, so exit the loop.
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            break;   // genuine recvfrom error — stop the responder
        }
        const std::string_view msg(buf.data(), static_cast<std::size_t>(n));
        if (!msg.starts_with(proto::discovery_probe)) continue;   // ignore anything not our probe

        // Roll the window.
        const auto now = clock::now();
        if (now - window_start >= std::chrono::seconds(1)) {
            window_start   = now;
            replies_in_win = 0;
            per_source.clear();
        }
        // ORDER IS LOAD-BEARING, and it is what bounds per_source's memory — not the
        // window clear above, which only decides how often the bound resets. The
        // global check precedes the map lookup, and operator[] is what INSERTS, so
        // once the global budget for this window is spent no new key can be created:
        // the map can never exceed discovery_max_replies_per_sec entries. Swap these
        // two lines and a forged-address flood grows it without limit — one entry per
        // probe, at millions of probes per second. Do not reorder.
        if (replies_in_win >= cfg_.discovery_max_replies_per_sec) continue;  // silent: a reply is the thing being limited
        int& from_count = per_source[from.sin_addr.s_addr];
        if (from_count >= cfg_.discovery_max_replies_per_source_per_sec) continue;

        // Build the reply PER PROBE, not once at loop entry. make_machine_summary
        // documents that data_link_ok is deliberately checked per call because that
        // bit flips at runtime when the operator mounts the dataset share, and "a
        // stale 'no' would outlive the fix for the daemon's lifetime" — which is
        // precisely what a hoisted reply reintroduced, on the discovery channel
        // only, while the TCP machine_info command answered correctly. Cost is one
        // stat plus a serialize on a user-initiated, low-rate path.
        auto summary = make_machine_summary(cfg_, version_);
        // Strip the host's filesystem layout from this UNAUTHENTICATED reply. The
        // TCP machine_info command returns the same struct WITH these fields — that
        // path at least required the peer to complete a handshake with us, whereas
        // a discovery reply can be aimed at someone who never spoke to this daemon
        // at all. Nothing needs them to discover a daemon: hostname, port, version
        // and data_link_ok are what a client picks a machine on. Consumers already
        // tolerate absence (from_json defaults all three to "").
        summary.runs_root.clear();
        summary.repo_root.clear();
        summary.dataset_base.clear();
        const std::string reply = nlohmann::json(summary).dump();
        ::sendto(fd_, reply.data(), reply.size(), 0,
                 reinterpret_cast<sockaddr*>(&from), from_len);
        ++replies_in_win;
        ++from_count;
    }
}

void discovery_server::stop() {
    if (!running_.exchange(false)) {
        // Never ran (bind failed) but a socket may still be open — clean it up.
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
        return;
    }
    // Clear running_ (done above), then JOIN, then close — in that order. The listen
    // thread wakes ITSELF: its recvfrom carries a SO_RCVTIMEO (set in start()), so
    // within one poll interval it returns EAGAIN, sees running_ == false, and exits.
    // This replaces a ::shutdown(fd_) wake that only worked on Linux — on macOS/BSD
    // shutdown does NOT unblock a recvfrom on an unconnected UDP socket, so the thread
    // never woke and this join() hung forever. Joining before close() guarantees the
    // thread is done with fd_ before the number is freed and possibly recycled (the
    // supervisor opens log fds every poll).
    if (thread_.joinable()) thread_.join();
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    COAST_INFO("coastd.discovery", "discovery responder stopped");
}

} // namespace coastd
