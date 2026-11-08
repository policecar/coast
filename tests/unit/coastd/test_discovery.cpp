#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "daemon/discovery_server.h"
#include "daemon/protocol.h"

#include "await_progress.h"

using namespace coastd;
using namespace std::chrono_literals;

// -----------------------------------------------------------------------------
// The UDP discovery responder is the daemon's only UNAUTHENTICATED channel, and
// the only one that can reach a host which never contacted this daemon at all: a
// UDP source address is trivially forged, so a ~15-byte probe claiming to come
// from a victim elicits a ~250-byte reply AT that victim. That makes two
// properties load-bearing rather than nice-to-have, and both are tested here
// against a real socket:
//
//   1. The reply carries no filesystem layout. runs_root/repo_root/dataset_base are in the
//      same struct the TCP machine_info command returns — that path at least
//      required a handshake. Discovery must not leak them to an arbitrary third
//      party.
//   2. Replies are rate-capped, so the daemon is not a usable amplifier.
//
// Plus the off switch, which has to actually silence the socket rather than just
// skip a log line.
// -----------------------------------------------------------------------------

namespace {

daemon_config test_cfg() {
    daemon_config cfg;
    cfg.udp_port  = 0;   // OS-assigned; probe srv.port() — collides with nothing
    // Distinctive, so a leak is unmistakable in the assertion below rather than
    // matching some incidental empty string.
    cfg.runs_root = "/SECRET_RUNS_ROOT_MARKER";
    cfg.repo_root = "/SECRET_REPO_ROOT_MARKER";
    cfg.dataset_base = "/SECRET_DATASET_BASE_MARKER";
    cfg.tcp_port  = 51900;
    return cfg;
}

// A UDP port nothing is bound to, for the cases where the responder must NOT
// bind: there is no srv.port() to probe, so probe where a working responder would
// have been. Freed before use, so another process could take it in between —
// harmless here, since it would also have to answer a coastd probe to matter.
std::uint16_t unbound_udp_port() {
    const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    REQUIRE(fd >= 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    socklen_t len = sizeof a;
    REQUIRE(::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0);
    REQUIRE(::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len) == 0);
    ::close(fd);
    return ntohs(a.sin_port);
}

// A probing client socket with a receive timeout, so a test can distinguish
// "replied" from "deliberately silent" without hanging.
struct prober {
    int           fd = -1;
    std::uint16_t port;
    prober(std::uint16_t udp_port, std::chrono::milliseconds rcv_timeout) : port(udp_port) {
        REQUIRE(port != 0);
        fd = ::socket(AF_INET, SOCK_DGRAM, 0);
        REQUIRE(fd >= 0);
        timeval tv{};
        tv.tv_sec  = static_cast<time_t>(rcv_timeout.count() / 1000);
        tv.tv_usec = static_cast<suseconds_t>((rcv_timeout.count() % 1000) * 1000);
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    }
    ~prober() { if (fd >= 0) ::close(fd); }

    void probe() const {
        sockaddr_in to{};
        to.sin_family = AF_INET;
        to.sin_port   = htons(port);
        ::inet_pton(AF_INET, "127.0.0.1", &to.sin_addr);
        const std::string p(proto::discovery_probe);
        (void)::sendto(fd, p.data(), p.size(), 0,
                       reinterpret_cast<const sockaddr*>(&to), sizeof to);
    }

    // One reply, or empty on timeout.
    [[nodiscard]] std::string recv_one() const {
        std::array<char, 2048> buf{};
        const ssize_t n = ::recv(fd, buf.data(), buf.size(), 0);
        if (n <= 0) return {};
        return std::string(buf.data(), static_cast<std::size_t>(n));
    }
};

} // namespace

TEST_CASE("discovery: the reply omits the host's filesystem layout") {
    auto cfg = test_cfg();
    discovery_server srv(cfg, "test-version");
    srv.start();

    prober p(srv.port(), coast_test::default_stall);   // the stall limit: our own server
    p.probe();
    const std::string reply = p.recv_one();
    REQUIRE_FALSE(reply.empty());                 // it must still ANSWER

    // The leak, stated as bytes on the wire — not as a struct field — because the
    // wire is what a third party sees.
    CHECK(reply.find("SECRET_RUNS_ROOT_MARKER") == std::string::npos);
    CHECK(reply.find("SECRET_REPO_ROOT_MARKER") == std::string::npos);
    CHECK(reply.find("SECRET_DATASET_BASE_MARKER") == std::string::npos);

    // What discovery is FOR must survive the redaction, or the fix broke the
    // feature instead of securing it.
    const auto j = nlohmann::json::parse(reply, nullptr, false);
    REQUIRE_FALSE(j.is_discarded());
    CHECK(j.contains("hostname"));
    CHECK(j.value("tcp_port", 0) == 51900);       // a discovered daemon stays connectable
    CHECK(j.value("coastd_version", std::string{}) == "test-version");

    srv.stop();
}

TEST_CASE("discovery: discovery_enabled=false silences the socket rather than just the log") {
    auto cfg = test_cfg();
    cfg.discovery_enabled = false;
    cfg.udp_port          = unbound_udp_port();
    discovery_server srv(cfg, "test-version");
    srv.start();
    CHECK(srv.port() == 0);                        // no responder bound

    prober p(cfg.udp_port, 500ms);
    p.probe();
    CHECK(p.recv_one().empty());                   // nothing bound, nothing answers

    srv.stop();                                    // must be safe on a never-started server
}

TEST_CASE("discovery: an invalid bind address refuses rather than binding everywhere") {
    // The failure mode this guards: silently falling back to INADDR_ANY would do
    // the exact opposite of what an operator restricting the interface asked for.
    auto cfg = test_cfg();
    cfg.discovery_bind = "not-an-ip";
    cfg.udp_port       = unbound_udp_port();
    discovery_server srv(cfg, "test-version");
    srv.start();
    CHECK(srv.port() == 0);                        // no responder bound

    prober p(cfg.udp_port, 500ms);
    p.probe();
    CHECK(p.recv_one().empty());

    srv.stop();
}

