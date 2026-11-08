#include "client/coastd_client.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <memory>
#include <sstream>
#include <thread>
#include <utility>

#include <nlohmann/json.hpp>

#include "daemon/sock_compat.h"

namespace coastd::client {

namespace {

// A single reply line may not exceed this. A daemon (or an impostor on the port)
// that streams bytes without a '\n' must not grow the client's buffer without
// bound. Replies are tiny (a job list of a few dozen entries), so 4 MiB is far
// beyond any legitimate reply.
constexpr std::size_t max_line_bytes = 4u << 20;

// Keep client sockets out of any child a client process spawns (e.g. the console
// spawning a headless coast): an inherited connection fd would hold the TCP
// connection half-open until that unrelated child exits. Portable set-after-
// create (SOCK_CLOEXEC is Linux-only); the window vs a concurrent spawn is
// negligible next to the always-leak it replaces.
void set_cloexec(int fd) noexcept {
    if (fd < 0) return;
    const int f = ::fcntl(fd, F_GETFD);
    if (f >= 0) ::fcntl(fd, F_SETFD, f | FD_CLOEXEC);
}

struct resolve_result {
    int                       gai = EAI_SYSTEM;
    bool                      timed_out = false;
    std::shared_ptr<addrinfo> res;   // freeaddrinfo runs on the last reference drop
};

// Detached resolvers currently outstanding. A timed-out resolve leaves its thread
// running (it cannot be cancelled), and the GUI panel retries on a ~2 s loop — so
// against a blackholed resolver, where every attempt times out and NONE ever
// finish, the thread count grows for as long as the panel stays open. Cap it and
// fail fast instead: past the cap the answer is already known ("this host is not
// resolving"), and spawning thread 200 to re-learn it is how a UI runs a process
// out of thread handles. The counter is decremented BY THE RESOLVER ITSELF, so it
// tracks genuinely-live threads rather than attempts.
std::atomic<int> g_inflight_resolves{0};
constexpr int    max_inflight_resolves = 8;

// getaddrinfo() can block for seconds to tens of seconds when DNS is unreachable
// (VPN dropped, blackholed resolver) — and connect_with_timeout runs on the GUI
// render thread, so a synchronous resolve would freeze the whole UI. Run it on a
// DETACHED thread and wait at most `timeout` for a result; if it doesn't finish,
// return timed_out and let the detached thread complete and free the result on its
// own (the shared_ptr keeps the addrinfo alive until the last reference drops).
// std::promise/std::future (not std::async) so the abandoned future never blocks
// on destruction. A zero timeout keeps the old unbounded, blocking semantics.
resolve_result resolve_bounded(const std::string& host, const std::string& port_s,
                               std::chrono::milliseconds timeout) {
    if (g_inflight_resolves.load(std::memory_order_relaxed) >= max_inflight_resolves) {
        resolve_result r;
        r.timed_out = true;   // reported to the caller as "DNS slow or unreachable"
        return r;
    }

    auto prom = std::make_shared<std::promise<resolve_result>>();
    auto fut  = prom->get_future();
    g_inflight_resolves.fetch_add(1, std::memory_order_relaxed);
    try {
        std::thread([host, port_s, prom]() {
            addrinfo hints{};
            // IPv4 ONLY, because that is all the daemon listens on (comms_server
            // binds AF_INET; change the two together). AF_UNSPEC let "localhost"
            // — the default host everywhere — resolve to ::1 first, and the connect
            // loop keeps the FIRST address that connects. If any IPv6-only service
            // held the daemon's port number on [::] (rpc-statd and the NFS lock
            // manager do, on kernel-assigned ports), the client connected to THAT
            // and got "no reply" from a perfectly healthy daemon.
            hints.ai_family   = AF_INET;
            hints.ai_socktype = SOCK_STREAM;
            addrinfo* res = nullptr;
            const int gai = ::getaddrinfo(host.c_str(), port_s.c_str(), &hints, &res);
            resolve_result r;
            r.gai = gai;
            if (gai == 0 && res) r.res = std::shared_ptr<addrinfo>(res, ::freeaddrinfo);
            prom->set_value(std::move(r));
            g_inflight_resolves.fetch_sub(1, std::memory_order_relaxed);
        }).detach();
    } catch (const std::system_error&) {
        // Thread creation failed (EAGAIN under handle exhaustion, a restrictive
        // rlimit). The header's contract is that errors are RETURNED — and
        // connection::open is called from the GUI's render thread, where a throw
        // would unwind through an in-progress ImGui frame. Report it as a failed
        // resolve instead.
        g_inflight_resolves.fetch_sub(1, std::memory_order_relaxed);
        resolve_result r;
        r.gai = EAI_SYSTEM;
        return r;
    }

    if (timeout.count() > 0 && fut.wait_for(timeout) != std::future_status::ready) {
        resolve_result r;
        r.timed_out = true;
        return r;
    }
    return fut.get();
}

// Blocking-connect with a timeout, via a temporarily non-blocking socket.
int connect_with_timeout(const endpoint& ep, std::chrono::milliseconds timeout, std::string& err,
                         std::chrono::milliseconds io_timeout) {
    const std::string port_s = std::to_string(ep.port);
    // ONE budget for resolve + connect together. Giving each the full `timeout`
    // made the real bound 2x what the caller asked for — and the caller here is the
    // GUI panel, on the ImGui render thread, where "500 ms" silently meant up to a
    // second of frozen UI. A zero timeout keeps the old unbounded semantics.
    const auto op_deadline = std::chrono::steady_clock::now() + timeout;
    const auto remaining = [&]() -> std::chrono::milliseconds {
        if (timeout.count() <= 0) return std::chrono::milliseconds{0};   // unbounded
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
            op_deadline - std::chrono::steady_clock::now());
        return left.count() > 0 ? left : std::chrono::milliseconds{1};   // never 0 == unbounded
    };

    const resolve_result rr = resolve_bounded(ep.host, port_s, remaining());
    if (rr.timed_out) {
        err = "resolve " + ep.host + ": timed out (DNS slow or unreachable)";
        return -1;
    }
    if (rr.gai != 0) {
        err = "resolve " + ep.host + ": " + ::gai_strerror(rr.gai);
        return -1;
    }

    // ONE deadline for the whole loop, not `timeout` per address. A dual-stack host
    // that resolves to several addresses, where one silently DROPS the SYN rather
    // than rejecting it, would otherwise burn the full timeout per candidate — 2x,
    // 3x the bound the caller relies on. The GUI panel connects from the ImGui
    // render thread, so that bound is what keeps a dead host from freezing the UI.
    // Shared with the resolve above: whatever it spent is already gone.
    const auto conn_deadline = op_deadline;
    int fd = -1;
    for (addrinfo* ai = rr.res.get(); ai; ai = ai->ai_next) {
        fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        set_cloexec(fd);

        const int flags = ::fcntl(fd, F_GETFL, 0);
        ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);

        int rc = ::connect(fd, ai->ai_addr, ai->ai_addrlen);
        if (rc == 0) { ::fcntl(fd, F_SETFL, flags); break; }   // immediate connect
        if (errno == EINPROGRESS) {
            // poll(), NOT select(). select's fd_set is a fixed-size bitmap (FD_SETSIZE,
            // 1024 on glibc) indexed by the RAW fd number, so FD_SET on fd >= 1024 is
            // an out-of-bounds write into this stack frame — undefined behaviour, not
            // a failed call. That is reachable here: this library is linked into the
            // GUI `coast` binary, which holds HDF5 files, GL resources and subprocess
            // pipes, and a long session can cross 1024 open descriptors. poll() takes
            // the fd as a plain int with no such ceiling. The daemon side already
            // uses poll(); this was the last select() left.
            struct pollfd pfd{};
            pfd.fd     = fd;
            pfd.events = POLLOUT;
            // Whatever is LEFT of the overall budget, never the full timeout again.
            const auto rem_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    conn_deadline - std::chrono::steady_clock::now()).count();
            rc = rem_ms > 0 ? ::poll(&pfd, 1, static_cast<int>(rem_ms)) : 0;
            if (rc > 0) {
                int soerr = 0;
                socklen_t len = sizeof soerr;
                ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &len);
                if (soerr == 0) { ::fcntl(fd, F_SETFL, flags); break; }   // connected
            }
        }
        ::close(fd);
        fd = -1;
    }
    if (fd < 0) { err = "connect " + ep.host + ":" + port_s + " failed within timeout"; return fd; }

    // Arm against SIGPIPE before the first send. A no-op where send() carries
    // MSG_NOSIGNAL; on macOS it is what stops a daemon that dies mid-exchange from
    // terminating the HOST application — this client is linked into `coast` itself,
    // so the process at stake is the GUI, not a throwaway CLI.
    set_nosigpipe(fd);

    // Bound every subsequent blocking recv/send. Without this, a daemon that
    // ACCEPTS the connection and then stalls without closing it (suspended process,
    // extreme load, a store-I/O hang) would block the caller forever — and the GUI
    // panel polls synchronously on the single ImGui render thread, so an unbounded
    // recv there freezes the entire UI, not just the panel. A finite timeout turns
    // that into a recoverable "no reply" instead of a hang.
    if (io_timeout.count() > 0) {
        timeval io_tv{};
        io_tv.tv_sec  = io_timeout.count() / 1000;
        io_tv.tv_usec = static_cast<long>((io_timeout.count() % 1000) * 1000);
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &io_tv, sizeof io_tv);
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &io_tv, sizeof io_tv);
    }
    return fd;
}

// write_all lives in sock_compat.h, shared with the daemon side.

// Read one '\n'-delimited line, buffering the overshoot in `buf` for next time.
//
// `budget` bounds the WHOLE reply, not each recv. SO_RCVTIMEO restarts on every
// arriving byte, so a peer that dribbles one byte per window satisfies it forever:
// the real bound was max_line_bytes / 4096 recvs x timeout — hours for a 4 MiB
// stream at 500 ms. The daemon side already grew a whole-request deadline for the
// identical slowloris; this is the client's half, and it matters more, because the
// caller is frequently the GUI's render thread. A zero budget means unbounded (the
// caller asked for no timeout at all).
bool read_line(int fd, std::string& buf, std::string& out, std::chrono::milliseconds budget) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    for (;;) {
        if (const auto nl = buf.find('\n'); nl != std::string::npos) {
            out.assign(buf, 0, nl);
            buf.erase(0, nl + 1);
            if (!out.empty() && out.back() == '\r') out.pop_back();
            return true;
        }
        if (budget.count() > 0 && std::chrono::steady_clock::now() >= deadline)
            return false;   // dribbling peer: give up on the whole reply
        char tmp[4096];
        const ssize_t n = ::recv(fd, tmp, sizeof tmp, 0);
        // EINTR retries; EAGAIN/EWOULDBLOCK is the SO_RCVTIMEO firing (daemon
        // accepted but never replied) -> treat as "no reply", never an infinite wait.
        if (n < 0) { if (errno == EINTR) continue; return false; }
        if (n == 0) return false;
        buf.append(tmp, static_cast<std::size_t>(n));
        if (buf.size() > max_line_bytes) return false;   // runaway/impostor: drop
    }
}

} // namespace

std::optional<connection> connection::open(const endpoint& ep,
                                           std::chrono::milliseconds connect_timeout,
                                           std::string& err,
                                           std::chrono::milliseconds io_timeout) {
    // Zero io_timeout means "same as connect" — the historical single-bound
    // behaviour, kept as the default so existing callers are unchanged.
    const auto io = io_timeout.count() > 0 ? io_timeout : connect_timeout;
    const int fd = connect_with_timeout(ep, connect_timeout, err, io);
    if (fd < 0) return std::nullopt;
    return connection(fd, io);
}

// Hand-written only because the source's fd must be nulled; every data member
// must still make the trip. timeout_ was once left behind here, which reset it
// to 0 (= unbounded) on the move out of open() — silently disabling the
// whole-reply deadline for every real caller.
connection::connection(connection&& o) noexcept
    : fd_(o.fd_), rbuf_(std::move(o.rbuf_)), timeout_(o.timeout_) {
    o.fd_ = -1;
}
connection& connection::operator=(connection&& o) noexcept {
    if (this != &o) {
        if (fd_ >= 0) ::close(fd_);
        fd_ = o.fd_;
        rbuf_ = std::move(o.rbuf_);
        timeout_ = o.timeout_;
        o.fd_ = -1;
    }
    return *this;
}
connection::~connection() { if (fd_ >= 0) ::close(fd_); }

void connection::poison() noexcept {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
    rbuf_.clear();
}

std::optional<nlohmann::json> connection::round_trip(std::string_view cmd, nlohmann::json args,
                                                     std::string& err) {
    // A prior transport error poisoned the socket; do not attempt a round-trip on a
    // dead (or, worse, desynced) connection.
    if (fd_ < 0) { err = "connection closed"; return std::nullopt; }

    args["cmd"] = std::string(cmd);
    // dump() THROWS (type_error.316) if any string in the request is not valid
    // UTF-8 — reachable via config_text if a caller filled it without going through
    // load_config_file(). This library reports failure through `err`, so an
    // exception escaping here would break that contract for every frontend, and in
    // the GUI's case would unwind through an ImGui frame. Convert it to an error.
    std::string line;
    try {
        line = args.dump();
    } catch (const std::exception& e) {
        err = std::string("request could not be serialized (invalid UTF-8 in a field?): ") + e.what();
        return std::nullopt;
    }
    line.push_back('\n');
    // ANY transport failure poisons the connection. A send failure or a timed-out
    // read means the request/reply framing is no longer trustworthy: the daemon's
    // reply may still arrive later and, with no request-id in the protocol, the
    // NEXT call would read it as its own (duplicate submits, cancel acting on a
    // stale reply). Close now so the caller reconnects instead of desyncing.
    if (!write_all(fd_, line)) {
        err = "send failed: " + std::string(std::strerror(errno));
        poison();
        return std::nullopt;
    }

    std::string reply;
    if (!read_line(fd_, rbuf_, reply, timeout_)) {
        err = "no reply (timed out or connection closed)";
        poison();
        return std::nullopt;
    }

    // Depth pre-scan BEFORE the DOM parser recurses. allow_exceptions=false does
    // not help here: over-deep nesting is a stack overflow (SIGSEGV), not a parse
    // error, so it is uncatchable. This reply arrives over the network from a
    // process we do not control; treat it exactly like the daemon treats a request.
    if (!proto::json_depth_ok(reply)) {
        err = "reply from daemon nested too deep";
        poison();
        return std::nullopt;
    }
    nlohmann::json j = nlohmann::json::parse(reply, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) {
        // A garbled reply is also a framing break — the stream position is unknown.
        err = "malformed reply from daemon";
        poison();
        return std::nullopt;
    }
    return j;
}

namespace {

// nlohmann's value() defaults ONLY on an absent key. A key that is PRESENT but
// wrong-typed throws type_error.302 — and that is neither a std::runtime_error nor
// caught by the guards further down, so it escapes the header's "errors are
// returned in `err`, never thrown" contract (coastd_client.h). A version-skewed or
// hostile daemon sending {"ok":"true"} or {"error":42} is enough to trigger it, and
// the GUI calls these from its render thread.
//
// The three helpers below are therefore the ONLY way this file touches a reply
// field: each uses find() + an explicit type test, so no accessor here can throw.
// Do not reintroduce j.value(...) on a reply.

// Extract "error" from an {"ok":false,...} reply (or a generic message).
std::string reply_error(const nlohmann::json& j) {
    const auto it = j.find("error");
    return (it != j.end() && it->is_string()) ? it->get<std::string>()
                                              : std::string("daemon reported failure");
}

// True iff the reply is a well-formed positive envelope. On a negative envelope
// `err` carries the daemon's message; on a malformed one it says so — a reply we
// cannot parse the verdict of is a failure, never a silent success.
bool envelope_ok(const nlohmann::json& j, std::string& err) {
    if (!j.is_object()) {
        err = "malformed reply from daemon: not a JSON object";
        return false;
    }
    const auto it = j.find("ok");
    if (it == j.end() || !it->is_boolean()) {
        err = "malformed reply from daemon: missing or non-boolean \"ok\"";
        return false;
    }
    if (it->get<bool>()) return true;
    err = reply_error(j);
    return false;
}

// A string field from a reply, or "" if absent or wrong-typed.
std::string reply_str(const nlohmann::json& j, const char* key) {
    const auto it = j.find(key);
    return (it != j.end() && it->is_string()) ? it->get<std::string>() : std::string{};
}

} // namespace

std::optional<std::vector<proto::job_summary>> connection::list(std::string& err) {
    auto r = round_trip(proto::cmd::list, nlohmann::json::object(), err);
    if (!r) return std::nullopt;
    if (!envelope_ok(*r, err)) return std::nullopt;
    std::vector<proto::job_summary> out;
    // from_json uses j.value(), which throws type_error.302 on a PRESENT but
    // wrong-typed field (a version-skewed daemon sending "steps_done":"12" is
    // enough). The header promises errors are returned, never thrown, and the
    // GUI calls this from its render thread — so convert here, as discover()
    // already does for the same hazard.
    try {
        if (const auto it = r->find("jobs"); it != r->end() && it->is_array())
            for (const auto& e : *it) out.push_back(e.get<proto::job_summary>());
    } catch (const std::exception& e) {
        err = "malformed job list from daemon: " + std::string(e.what());
        return std::nullopt;
    }
    return out;
}

bool load_config_file(submit_args& a, const std::filesystem::path& local_path, std::string& err) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(local_path, ec)) {
        err = "config not found or not a regular file: " + local_path.string();
        return false;
    }
    std::ifstream in(local_path, std::ios::binary);
    if (!in) { err = "config not readable: " + local_path.string(); return false; }

    std::ostringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();
    if (text.empty()) { err = "config is empty: " + local_path.string(); return false; }

    // Pre-flight only. A parse failure here also covers the invalid-UTF-8 case that
    // would otherwise throw out of dump() when the request is serialized: such a
    // file is not valid JSON either, so it cannot be a usable config regardless.
    const nlohmann::json probe = nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
    if (probe.is_discarded()) {
        err = "config is not valid JSON (or not valid UTF-8): " + local_path.string();
        return false;
    }
    if (!probe.is_object()) {
        err = "config top-level JSON must be an object: " + local_path.string();
        return false;
    }

    a.config_text = std::move(text);            // ORIGINAL bytes, never a re-dump
    a.config_name = local_path.filename().string();
    return true;
}

std::optional<std::string> connection::submit(const submit_args& a, std::string& err) {
    if (a.config_text.empty()) {
        err = "submit: config_text is empty — load it with load_config_file()";
        return std::nullopt;
    }
    nlohmann::json args{
        {"ref", a.ref}, {"config_text", a.config_text}, {"config_name", a.config_name},
        {"label", a.label},
        {"message", a.message}, {"steps_target", a.steps_target}, {"priority", a.priority}};
    auto r = round_trip(proto::cmd::submit, std::move(args), err);
    if (!r) return std::nullopt;
    if (!envelope_ok(*r, err)) return std::nullopt;
    // An OK envelope with no usable id is a malformed reply, not a success. Returning
    // an engaged optional holding "" would hand the caller an empty string AS the job
    // id — the console prints it back to the user and the GUI panel records a submit
    // that worked, for a job nobody can then cancel or query. Treated like any other
    // unparseable verdict: nullopt plus a reason.
    std::string id = reply_str(*r, "id");
    if (id.empty()) {
        err = "daemon accepted the submit but returned no job id";
        return std::nullopt;
    }
    return id;
}

bool connection::cancel(const std::string& id, std::string& err) {
    auto r = round_trip(proto::cmd::cancel, nlohmann::json{{"id", id}}, err);
    if (!r) return false;
    if (!envelope_ok(*r, err)) return false;
    return true;
}

bool connection::reorder_by(const std::string& id, int delta, std::string& err) {
    auto r = round_trip(proto::cmd::reorder, nlohmann::json{{"id", id}, {"delta", delta}}, err);
    if (!r) return false;
    if (!envelope_ok(*r, err)) return false;
    return true;
}

bool connection::reorder(const std::string& id, int pos, std::string& err) {
    auto r = round_trip(proto::cmd::reorder, nlohmann::json{{"id", id}, {"pos", pos}}, err);
    if (!r) return false;
    if (!envelope_ok(*r, err)) return false;
    return true;
}

std::optional<proto::machine_summary> connection::machine_info(std::string& err) {
    auto r = round_trip(proto::cmd::machine_info, nlohmann::json::object(), err);
    if (!r) return std::nullopt;
    if (!envelope_ok(*r, err)) return std::nullopt;
    // Same throwing-conversion hazard as list(); the header's contract is that
    // errors come back in `err`.
    try {
        return r->get<proto::machine_summary>();  // fields are top-level in the ok envelope
    } catch (const std::exception& e) {
        err = "malformed machine info from daemon: " + std::string(e.what());
        return std::nullopt;
    }
}

std::optional<std::map<std::string, std::string>>
connection::hash_inputs(const std::vector<std::string>& paths, std::string& err) {
    auto r = round_trip(proto::cmd::hash_inputs, nlohmann::json{{"paths", paths}}, err);
    if (!r) return std::nullopt;
    if (!envelope_ok(*r, err)) return std::nullopt;
    try {
        return r->at("sha256").get<std::map<std::string, std::string>>();
    } catch (const std::exception& e) {
        err = "malformed hash_inputs reply from daemon: " + std::string(e.what());
        return std::nullopt;
    }
}

std::vector<proto::machine_summary> discover(std::chrono::milliseconds timeout, std::uint16_t port) {
    std::vector<proto::machine_summary> found;
    const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return found;
    set_cloexec(fd);

    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_BROADCAST, &one, sizeof one);
    timeval tv{};
    tv.tv_sec  = timeout.count() / 1000;
    tv.tv_usec = static_cast<long>((timeout.count() % 1000) * 1000);
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);

    sockaddr_in bcast{};
    bcast.sin_family      = AF_INET;
    bcast.sin_addr.s_addr = htonl(INADDR_BROADCAST);
    bcast.sin_port        = htons(port);
    const std::string probe(proto::discovery_probe);
    ::sendto(fd, probe.data(), probe.size(), 0, reinterpret_cast<sockaddr*>(&bcast), sizeof bcast);

    // Collect every reply until the receive timeout elapses. recvfrom() captures
    // the SENDER's address — the actually-connectable IP — which the daemon cannot
    // know for itself; we stamp it onto the summary as the client-only `address`.
    // Sized to hold a whole UDP datagram (max payload 65507 B) so a machine_summary
    // with long paths is never silently truncated at the recvfrom boundary — a
    // truncated JSON reply would just fail to parse and drop the daemon from the
    // results with no clue why.
    //
    // Two bounds, because SO_RCVTIMEO alone is neither. It is an IDLE timeout: it
    // restarts on every datagram, so ANY host answering faster than one datagram per
    // window keeps this loop alive forever — and `found` grows ~200 bytes per
    // datagram the whole time. The probe is a broadcast, so every host on the LAN
    // knows the source port, and nothing here authenticates a reply; one spoofed or
    // simply buggy peer is enough. discover() is called SYNCHRONOUSLY from the ImGui
    // render thread (coastd_panel.cpp), so the failure is a permanently frozen GUI
    // plus unbounded memory, not a slow scan.
    //
    // So: an absolute deadline that a flood cannot push back, and a cap on results.
    // The cap is not a real-world limit — a LAN with 256 coastd daemons is not this
    // project's problem — it is the ceiling that keeps a hostile answer bounded.
    const auto disc_deadline = std::chrono::steady_clock::now() + timeout;
    constexpr std::size_t max_daemons = 256;

    char buf[65536];
    for (;;) {
        if (found.size() >= max_daemons) break;
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
            disc_deadline - std::chrono::steady_clock::now());
        if (left.count() <= 0) break;
        // Shrink the per-recv window to what is LEFT, so the last datagram cannot
        // extend collection past the deadline by a further full timeout.
        timeval rem{};
        rem.tv_sec  = left.count() / 1000;
        rem.tv_usec = static_cast<long>((left.count() % 1000) * 1000);
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rem, sizeof rem);

        sockaddr_in from{};
        socklen_t   from_len = sizeof from;
        const ssize_t n = ::recvfrom(fd, buf, sizeof buf, 0,
                                     reinterpret_cast<sockaddr*>(&from), &from_len);
        if (n <= 0) break;   // timeout (EAGAIN) or error ends collection
        const std::string_view datagram(buf, static_cast<std::size_t>(n));
        // Unauthenticated broadcast input: any LAN host can send this. Depth
        // pre-scan first — a 60k-'[' datagram fits well inside buf and would
        // stack-overflow the DOM parser, taking down the GUI/console that called
        // discover(). Drop it and keep scanning.
        if (!proto::json_depth_ok(datagram)) continue;
        nlohmann::json j = nlohmann::json::parse(datagram, nullptr, /*allow_exceptions=*/false);
        if (j.is_discarded() || !j.is_object()) continue;
        // Guard the DOM->struct conversion: from_json uses j.value(), which throws
        // type_error on a present-but-wrong-typed field. A single crafted, garbled,
        // or version-skewed datagram from ANY LAN peer would otherwise unwind out of
        // discover() and discard every legitimate reply already collected. Skip the
        // bad datagram and keep scanning.
        proto::machine_summary ms;
        try {
            ms = j.get<proto::machine_summary>();
        } catch (const std::exception&) {
            continue;
        }
        char ip[INET_ADDRSTRLEN] = {};
        if (::inet_ntop(AF_INET, &from.sin_addr, ip, sizeof ip)) ms.address = ip;
        found.push_back(std::move(ms));
    }
    ::close(fd);
    return found;
}

// ---- fixed-list host config (IDD §6.6, deterministic primary) ----------------

namespace {

// Is there a case-sensitive "localhost" entry already?
bool has_localhost(const std::vector<host_entry>& es) {
    for (const auto& e : es) if (e.name == "localhost") return true;
    return false;
}

} // namespace

std::filesystem::path host_config::default_path() {
    if (const char* h = std::getenv("COASTD_HOSTS"); h && *h)
        return std::filesystem::path(h);
    if (const char* xdg = std::getenv("XDG_CONFIG_HOME"); xdg && *xdg)
        return std::filesystem::path(xdg) / "coastd" / "hosts.json";
    if (const char* home = std::getenv("HOME"); home && *home)
        return std::filesystem::path(home) / ".config" / "coastd" / "hosts.json";
    return std::filesystem::path(".config") / "coastd" / "hosts.json";
}

host_config host_config::load(const std::filesystem::path& path, std::string& warn) {
    host_config cfg;

    std::error_code ec;
    if (!path.empty() && std::filesystem::is_regular_file(path, ec)) {
        std::ifstream in(path);
        std::ostringstream ss;
        ss << in.rdbuf();
        // Same depth guard the wire edge uses — a hand-written config should never
        // stack-overflow the client that reads it.
        if (!proto::json_depth_ok(ss.str())) {
            warn = "coastd hosts config nesting too deep: " + path.string();
        } else {
            nlohmann::json j = nlohmann::json::parse(ss.str(), nullptr, /*allow_exceptions=*/false);
            if (j.is_discarded() || !j.is_array()) {
                warn = "coastd hosts config is not a JSON array: " + path.string();
            } else {
                for (const auto& e : j) {
                    if (!e.is_object()) continue;
                    host_entry he;
                    // value() only substitutes the default when the key is ABSENT;
                    // a present-but-wrong-typed one ("port": "1234") throws. Letting
                    // that escape would break the documented contract that a
                    // malformed hosts file yields a warning plus a usable localhost
                    // default — the caller would get an exception instead. Skip the
                    // offending row, keep the rest of the file.
                    try {
                        he.name = e.value("name", std::string{});
                        he.host = e.value("host", std::string{});
                        he.port = e.value("port", proto::default_tcp_port);
                    } catch (const std::exception&) {
                        warn = "coastd hosts config has malformed entries (skipped): "
                             + path.string();
                        continue;
                    }
                    if (he.name.empty() || he.host.empty()) continue;  // skip malformed rows
                    cfg.entries_.push_back(std::move(he));
                }
            }
        }
    }

    // Guarantee a localhost entry so resolve("")/resolve("localhost") always works,
    // even with no config file at all.
    if (!has_localhost(cfg.entries_))
        cfg.entries_.insert(cfg.entries_.begin(),
                            host_entry{"localhost", "localhost", proto::default_tcp_port});
    return cfg;
}

host_config host_config::load_default(std::string& warn) {
    return load(default_path(), warn);
}

std::optional<endpoint> host_config::resolve(std::string_view name) const {
    const std::string_view want = name.empty() ? std::string_view("localhost") : name;
    for (const auto& e : entries_)
        if (e.name == want) return endpoint{e.host, e.port};
    return std::nullopt;
}

std::optional<connection> connect_to(std::string_view name, std::chrono::milliseconds timeout,
                                     std::string& err, std::chrono::milliseconds io_timeout) {
    std::string warn;
    const host_config cfg = host_config::load_default(warn);
    const auto ep = cfg.resolve(name);
    if (!ep) {
        err = "no daemon named '" + std::string(name) + "' in the host config ("
              + host_config::default_path().string() + ")";
        if (!warn.empty()) err += "; " + warn;
        return std::nullopt;
    }
    return connection::open(*ep, timeout, err, io_timeout);
}

} // namespace coastd::client
