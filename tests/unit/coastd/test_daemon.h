#ifndef COAST_TESTS_COASTD_TEST_DAEMON_H
#define COAST_TESTS_COASTD_TEST_DAEMON_H

// -----------------------------------------------------------------------------
// Ports + lifetime for the tests that launch a REAL coastd.
//
// PORTS. Every such test starts the daemon with `--tcp-port 0 --udp-port 0`: the
// kernel picks free ports at bind time and the daemon writes what it got to
// <data-root>/coastd.ports once it accepts connections. Nothing is chosen up
// front, so nothing can collide — not two overlapping ctest runs, not a stale
// daemon, and not the host's own outbound connections.
//
// That last one is what fixed ports kept losing to. A client socket — established
// OR in TIME_WAIT, for 60 s after it closes — blocks bind() on its local port even
// with SO_REUSEADDR, and connect() hands out source ports from the ephemeral range
// (32768-60999 by default). A fixed test port inside that range (unit.console.repl
// used 53940) therefore failed whenever the CI runner's HTTPS long-polls or an
// earlier test's loopback clients had just used it. The previous scheme, a pid-
// derived block above 60999, avoided that only on hosts with the default range,
// and two pids 400 apart still shared a block.
//
// LIFETIME. A REQUIRE that fails after the spawn throws past any cleanup written
// at the end of the test. A daemon left running there keeps answering on its port
// for the rest of the machine's uptime. daemon_guard reaps it on every exit path.
//
// Header-only and free of coastd headers on purpose: unit.coastd.console and
// unit.console.repl exercise the console façade, whose point is NOT seeing the
// daemon's headers (see test_console.cpp).
// -----------------------------------------------------------------------------

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "await_progress.h"   // tests/support: load-independent waits

namespace coastd_test {

// What machine_info's `os` must report for a daemon running on this host — the
// value is the daemon's own os_name() (src/machine_info.cpp), spelled out here
// because this header may not include the daemon's.
#if defined(__APPLE__)
inline constexpr const char* expected_os = "macos";
#else
inline constexpr const char* expected_os = "linux";
#endif

struct daemon_ports {
    std::uint16_t tcp = 0;
    std::uint16_t udp = 0;   // 0 = the daemon runs without a discovery responder
};

// Has our child `pid` exited? Does NOT reap it (WNOWAIT), so its owner can still
// waitpid() it and read the status.
inline bool has_exited(pid_t pid) noexcept {
    siginfo_t si{};
    return ::waitid(P_PID, static_cast<id_t>(pid), &si, WEXITED | WNOHANG | WNOWAIT) == 0 &&
           si.si_pid == pid;
}

// The ports `pid` published under data_root, if it has. A file carrying another
// pid is a predecessor's leftover (a restarted daemon removes it at startup, but a
// reader can get there first) and is ignored, as is a file still being replaced.
inline std::optional<daemon_ports> read_ports(const std::filesystem::path& data_root, pid_t pid) {
    std::ifstream in(data_root / "coastd.ports");
    if (!in) return std::nullopt;
    const auto j = nlohmann::json::parse(in, nullptr, /*allow_exceptions=*/false);
    if (!j.is_object() || j.value("pid", -1LL) != static_cast<long long>(pid)) return std::nullopt;
    daemon_ports p;
    p.tcp = j.value("tcp_port", std::uint16_t{0});
    p.udp = j.value("udp_port", std::uint16_t{0});
    if (p.tcp == 0) return std::nullopt;
    return p;
}

// Wait until `pid` accepts connections, i.e. has published its ports. Returns
// nullopt the moment it exits instead (a bind failure, a bad flag), so a dead
// daemon fails the test at once. A daemon still booting has no heartbeat yet, so
// only the no-progress stall limit (await_condition) ends a wait that never
// completes — a loaded machine only stretches it.
inline std::optional<daemon_ports> wait_for_ports(const std::filesystem::path& data_root, pid_t pid) {
    std::optional<daemon_ports> p;
    bool                        exited = false;
    (void)coast_test::await_condition([&] {
        p = read_ports(data_root, pid);
        if (p) return true;
        exited = has_exited(pid);
        return exited;
    });
    return exited ? std::nullopt : p;
}

// ---- failure diagnostics -----------------------------------------------------
// A daemon test that fails in CI otherwise says only what the CLIENT saw ("no
// reply"), which cannot tell a dead daemon from a connection that reached some
// other program. daemon_report() gathers what can: meant for a lazy
// INFO(coastd_test::daemon_report(...)), which doctest evaluates only on a failed
// assertion or while an exception unwinds the test case. The latter runs it in a
// destructor, hence noexcept with everything caught inside.

// Try a TCP connect to addr:port (IPv4 or IPv6 literal): "accepts" or the error.
inline std::string probe_tcp(const char* addr, std::uint16_t port) {
    sockaddr_storage ss{};
    socklen_t        len = 0;
    int              family = AF_INET;
    if (auto* a4 = reinterpret_cast<sockaddr_in*>(&ss); ::inet_pton(AF_INET, addr, &a4->sin_addr) == 1) {
        a4->sin_family = AF_INET;
        a4->sin_port   = htons(port);
        len = sizeof *a4;
    } else if (auto* a6 = reinterpret_cast<sockaddr_in6*>(&ss);
               ::inet_pton(AF_INET6, addr, &a6->sin6_addr) == 1) {
        a6->sin6_family = AF_INET6;
        a6->sin6_port   = htons(port);
        len    = sizeof *a6;
        family = AF_INET6;
    } else {
        return "not an address literal";
    }
    const int fd = ::socket(family, SOCK_STREAM, 0);
    if (fd < 0) return std::string("socket: ") + std::strerror(errno);
    const bool ok = ::connect(fd, reinterpret_cast<sockaddr*>(&ss), len) == 0;
    const int  e  = errno;
    ::close(fd);
    return ok ? "accepts" : std::strerror(e);
}

// The address order getaddrinfo gives for "localhost" — the order the client
// library tries them in. It differs between hosts (::1 first on some).
inline std::string localhost_order() {
    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (::getaddrinfo("localhost", nullptr, &hints, &res) != 0) return "getaddrinfo failed";
    std::string out;
    for (addrinfo* ai = res; ai; ai = ai->ai_next) {
        char ip[INET6_ADDRSTRLEN] = {};
        const void* src = ai->ai_family == AF_INET6
            ? static_cast<const void*>(&reinterpret_cast<sockaddr_in6*>(ai->ai_addr)->sin6_addr)
            : static_cast<const void*>(&reinterpret_cast<sockaddr_in*>(ai->ai_addr)->sin_addr);
        ::inet_ntop(ai->ai_family, src, ip, sizeof ip);
        out += (out.empty() ? "" : ", ") + std::string(ip);
    }
    ::freeaddrinfo(res);
    return out;
}

inline std::string file_tail(const std::filesystem::path& p, std::size_t lines) {
    std::ifstream in(p);
    if (!in) return "(cannot open " + p.string() + ")";
    std::deque<std::string> last;
    for (std::string l; std::getline(in, l);) {
        last.push_back(std::move(l));
        if (last.size() > lines) last.pop_front();
    }
    std::string out;
    for (const auto& l : last) out += "    " + l + "\n";
    return out.empty() ? "    (empty)\n" : out;
}

// Liveness, what answers on the daemon's port over IPv4 and IPv6, the localhost
// resolution order, and the tail of the daemon's own log.
inline std::string daemon_report(pid_t pid, std::uint16_t tcp_port,
                                 const std::filesystem::path& log) noexcept {
    // doctest also prints active INFOs with every MESSAGE, so this can run after
    // the test reaped the daemon itself (daemon_guard::pid() is then 0). Probing
    // a port nobody serves any more would only mislead.
    if (pid <= 0) return "daemon already stopped and reaped by the test";
    try {
        std::ostringstream s;
        s << "daemon pid " << pid << ": " << (has_exited(pid) ? "EXITED" : "alive")
          << "; tcp port " << tcp_port << "\n"
          << "  127.0.0.1:" << tcp_port << " -> " << probe_tcp("127.0.0.1", tcp_port) << "\n"
          << "  ::1:" << tcp_port << " -> " << probe_tcp("::1", tcp_port) << "\n"
          << "  localhost resolves to: " << localhost_order() << "\n"
          << "  daemon log (" << log.string() << "):\n" << file_tail(log, 30);
        return s.str();
    } catch (...) {
        return "daemon_report itself failed";
    }
}

// Owns a spawned daemon: SIGTERM + bounded wait + SIGKILL fallback, on stop() or
// on scope exit. stop() returns whether the daemon exited on its own — a daemon
// that needs SIGKILL failed its shutdown contract, so tests CHECK it — and
// status() then holds its waitpid status. The grace is the stall limit, not a
// performance bound: an orderly shutdown takes ~0.2 s, so only a daemon that is
// stuck reaches it, however loaded the machine.
class daemon_guard {
public:
    explicit daemon_guard(pid_t pid) noexcept : pid_(pid) {}
    ~daemon_guard() { stop(); }
    daemon_guard(const daemon_guard&)            = delete;
    daemon_guard& operator=(const daemon_guard&) = delete;

    [[nodiscard]] pid_t pid() const noexcept { return pid_; }
    [[nodiscard]] int   status() const noexcept { return status_; }

    bool stop(std::chrono::milliseconds grace = coast_test::default_stall) noexcept {
        if (pid_ <= 0) return true;
        ::kill(pid_, SIGTERM);
        // Track the reap with a FLAG, not by inspecting a status: after a timed-out
        // poll the status is still its initial 0, and WIFEXITED(0) is true.
        bool reaped = false;
        const auto deadline = std::chrono::steady_clock::now() + grace;
        while (std::chrono::steady_clock::now() < deadline) {
            if (::waitpid(pid_, &status_, WNOHANG) == pid_) { reaped = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (!reaped) { ::kill(pid_, SIGKILL); ::waitpid(pid_, &status_, 0); }
        pid_ = 0;
        return reaped;
    }

private:
    pid_t pid_;
    int   status_ = 0;
};

} // namespace coastd_test

#endif // COAST_TESTS_COASTD_TEST_DAEMON_H
