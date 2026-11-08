#include "daemon/comms_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

#include <nlohmann/json.hpp>

#include "daemon/child_cmd.h"   // initial_state_from
#include "daemon/core.h"
#include "daemon/hash_inputs.h"   // hash_inputs command
#include "daemon/job_json.h"      // status_name
#include "daemon/machine_info.h"  // make_machine_summary
#include "daemon/proc_util.h"     // set_cloexec, local_port
#include "daemon/protocol.h"
#include "daemon/sock_compat.h"   // send_nosignal_flags, set_nosigpipe, write_all
#include "diag/diag.h"

namespace coastd {

namespace {
namespace fs = std::filesystem;

// How often the accept loop self-wakes to re-check running_ (see accept_loop). It
// bounds stop()'s latency on platforms where shutdown() does not wake accept();
// the same 200 ms as discovery_server's receive timeout.
constexpr int accept_wake_interval_ms = 200;

// The comms-edge bounds are operator-tunable and live in daemon_config:
//   cfg_.max_connections     — simultaneous client connections (flood cap). Beyond
//                              it the accept loop closes new fds until handlers drain.
//   cfg_.max_request_bytes   — a single request line; a runaway/binary stream past
//                              it is dropped rather than buffered without bound.
//   cfg_.max_config_bytes    — a submitted config file, rejected before it is read
//                              into memory (the one input boundary that was unbounded).
//   cfg_.first_request_timeout  — silence allowed before the first request line, so a
//                              connect-then-never-send socket cannot pin a slot forever.
//                              Applied only to the first read; cleared once a real
//                              request arrives, so an idle persistent session survives.
//   cfg_.line_assembly_timeout  — TOTAL time to assemble one request line once its
//                              first byte lands. SO_RCVTIMEO resets per byte, so a
//                              one-byte-at-a-time dribble (slowloris) would otherwise
//                              hold a slot forever; this bounds the in-progress line.
//   cfg_.send_timeout        — a single reply write must make progress within this,
//                              or a non-reading peer parks the handler thread in send()
//                              and, times max_connections, starves every real client.

// Turn on TCP keepalive so a peer that DIES while connected (crash, sleep, network
// partition) is detected and its slot freed, WITHOUT disconnecting one that is
// merely idle: keepalive probes are answered by a live peer's kernel even when its
// application sends nothing, so an idle session stays up while a dead one errors
// out of recv in ~idle + cnt*intvl seconds. Best-effort (setsockopt failures are
// non-fatal); the per-idle tuning is Linux-specific and skipped where unavailable.
void enable_keepalive(int fd) {
    int on = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &on, sizeof on);
#ifdef TCP_KEEPIDLE
    int idle = 60, intvl = 10, cnt = 3;   // dead peer detected in ~90s
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE,  &idle,  sizeof idle);
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof intvl);
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT,   &cnt,   sizeof cnt);
#endif
}

// Set SO_RCVTIMEO (a recv() longer than this fails with EAGAIN); a zero duration
// clears it, restoring an indefinite block. Best-effort.
void set_recv_timeout(int fd, std::chrono::milliseconds ms) {
    timeval tv{};
    tv.tv_sec  = static_cast<time_t>(ms.count() / 1000);
    tv.tv_usec = static_cast<suseconds_t>((ms.count() % 1000) * 1000);
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
}

// Set SO_SNDTIMEO so a send() that cannot make progress within `ms` fails with
// EAGAIN instead of blocking the handler thread forever. Best-effort.
void set_send_timeout(int fd, std::chrono::milliseconds ms) {
    timeval tv{};
    tv.tv_sec  = static_cast<time_t>(ms.count() / 1000);
    tv.tv_usec = static_cast<suseconds_t>((ms.count() % 1000) * 1000);
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
}

// Is `p` inside `root` (root a path-prefix of p)? Both weakly-canonicalized so
// "/a/b/../state" and symlink-free forms compare correctly without requiring the
// paths to exist.
// TRI-STATE on purpose. This is the guard between a submitted config's paths and
// the daemon-critical roots, and its two callers below read the answer in OPPOSITE
// senses — one uses "inside" to PERMIT (the job's own run dir), the other to
// REJECT. A bool therefore cannot fail closed for both: whichever value stood for
// "could not tell" would open one of the two. So report `unknown` and let each
// caller resolve it in its own safe direction.
//
// It could not tell before, either: only root's canonicalization was checked (and
// a single `ec` was reused across both calls, hiding the first result). With `pp`
// empty, std::mismatch stops at rp.begin(), which is != rp.end(), so the function
// answered "not under" — the permissive answer — for a path it had failed to
// canonicalize at all.
enum class under_result { yes, no, unknown };

under_result is_under(const fs::path& p, const fs::path& root) {
    std::error_code rec;
    std::error_code pec;
    const fs::path rp = fs::weakly_canonical(root, rec);
    const fs::path pp = fs::weakly_canonical(p, pec);
    if (rec || pec || rp.empty() || pp.empty()) return under_result::unknown;
    auto [rit, pit] = std::mismatch(rp.begin(), rp.end(), pp.begin(), pp.end());
    // every component of root matched a prefix of p
    return rit == rp.end() ? under_result::yes : under_result::no;
}

// Recursively scan a config JSON for any string that denotes a path which lands
// inside a daemon-critical directory. Returns the offending path if any.
// Key-agnostic on purpose: it does not trust field names, it inspects values.
//
// RELATIVE-PATH FRAME. A relative path in a config is resolved by the CHILD, and
// the child runs with its CWD set to the run dir (child_cmd.cpp: p.cwd =
// run_dir_of(j)). So the frame that matters is the run dir — NOT the directory
// the config file happened to live in. (Before configs travelled as content, this
// scan resolved relatives against the submitted file's parent; under that frame a
// relative path could never land under a forbidden root, so relative paths were
// effectively UNCHECKED. Resolving against the run dir closes that hole.)
//
// `own_run_dir` is a sentinel stand-in for the run dir of the job being submitted:
// the real id is not minted until the core creates the job, and this runs on the
// connection edge before that. Anything resolving INSIDE it is legitimate — that
// is the job's own output location. Anything that escapes it (`../other_job/…`)
// and lands under a forbidden root is rejected, which is exactly the cross-job
// clobber this guard exists to prevent.
std::optional<std::string> scan_paths(const nlohmann::json& node,
                                      const fs::path& own_run_dir,
                                      const std::vector<fs::path>& forbidden) {
    if (node.is_object() || node.is_array()) {
        for (const auto& child : node) {
            if (auto hit = scan_paths(child, own_run_dir, forbidden)) return hit;
        }
        return std::nullopt;
    }
    if (!node.is_string()) return std::nullopt;

    const std::string s = node.get<std::string>();
    fs::path candidate(s);
    const bool looks_like_path = candidate.is_absolute() || s.find('/') != std::string::npos;
    if (!looks_like_path) return std::nullopt;
    const bool was_relative = candidate.is_relative();
    if (was_relative) candidate = own_run_dir / candidate;

    // The job's own run dir is the ONE legitimate location inside runs_root.
    // Checked first so it wins over the blanket runs_root rejection below. Only a
    // DEFINITE yes exempts the path — an undecidable one falls through to the
    // rejection loop.
    //
    // RELATIVE paths only. own_run_dir is the `__pending__` STAND-IN, not a real
    // directory: a relative path is exempt because at runtime it rebases onto the
    // job's actual run dir (the child chdirs there), so the sentinel is just an
    // algebraic placeholder. An ABSOLUTE path that happens to name the sentinel —
    // "<runs_root>/__pending__/x", reachable by writing it out or via a "../"
    // that climbs back into it — gets no such rebasing: it stays that literal path
    // and lands in a REAL directory shared by every job, inside the runs_root this
    // function otherwise forbids wholesale. Restricting the exemption to relative
    // paths sends those to the rejection loop below, where runs_root catches them.
    if (was_relative && is_under(candidate, own_run_dir) == under_result::yes)
        return std::nullopt;

    // Anything but a definite no is rejected: the operator gets a submit error
    // naming the path, which is recoverable, where wrongly admitting it is not.
    for (const fs::path& root : forbidden)
        if (is_under(candidate, root) != under_result::no) return s;
    return std::nullopt;
}

// Submit-time config validation (IDD §6.7, minimal-structural variant). coastd
// already forces every OUTPUT location (snapshot_dir, --save-state, manifest,
// logs) under the run dir, so the residual risk this guards is a config that
// names a path inside a daemon-critical directory. Returns an error message on
// rejection, nullopt if the config is acceptable. NO store mutation and NO
// filesystem read — safe to run on the connection edge thread, before the job is
// ever created.
//
// The config arrives as BYTES over the wire (see protocol.h). Validating the
// received bytes — rather than re-reading a file — also closes the old
// submit-config TOCTOU by construction: there is no file to swap between the
// check and the use. The caller threads these same bytes into the job, so the run
// is always prepared from exactly what was validated.
std::optional<std::string> validate_submit_config(const std::string& text,
                                                  const daemon_config& cfg) {
    if (text.empty()) return "config is empty";
    if (text.size() > cfg.max_config_bytes)
        return "config too large (>" + std::to_string(cfg.max_config_bytes) + " bytes)";

    // Same stack-overflow guard the wire edge uses: reject pathologically-nested
    // JSON with an O(n) pre-scan before the recursive DOM parser ever sees it.
    if (!proto::json_depth_ok(text))
        return "config nesting too deep (>" + std::to_string(proto::max_json_depth) + ")";

    nlohmann::json cfg_json = nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
    if (cfg_json.is_discarded()) return "config is not valid JSON";
    // A coast config is a JSON object. Reject a top-level array/scalar here with a
    // clean message: prepare_run_config later does cfg["snapshot_dir"] = …, which
    // throws type_error on a non-object — and that throw would unwind past the
    // submit future as an opaque "internal error: … broken promise" with no job.
    if (!cfg_json.is_object()) return "config top-level JSON must be an object";

    // Stand-in for the not-yet-minted run dir (see scan_paths). The leading and
    // trailing underscores cannot occur in a real job id, whose shape is
    // <UTC>_<mnemonic>_<hex> with the mnemonic sanitized to [a-z0-9-], so this can
    // never alias an actual run dir.
    const fs::path own_run_dir = cfg.runs_root / "__pending__";
    // runs_root is included so a submitted config cannot direct a NON-overridden
    // path-bearing field (e.g. a buffer-log output) into another job's run dir and
    // clobber its artifacts — coastd only rewrites the output locations it owns
    // (snapshot_dir, --save-state, manifest, logs). repo_root is intentionally NOT
    // forbidden: configs legitimately reference datasets that may live under it.
    const std::vector<fs::path> forbidden{cfg.state_root, cfg.worktrees_root, cfg.runs_root};
    if (auto hit = scan_paths(cfg_json, own_run_dir, forbidden))
        return "config path '" + *hit + "' resolves inside a daemon-critical directory";
    return std::nullopt;
}

proto::job_summary job_summary_from(const job& j, bucket b, std::uint64_t live_step) {
    proto::job_summary s;
    s.id             = j.id;
    s.ref_symbolic   = j.ref_symbolic;
    s.submitted_at   = j.submitted_at;
    s.user_message   = j.user_message;
    s.status         = std::string(status_name(j.status));
    s.bucket         = std::string(bucket_dir(b));
    s.steps_done     = j.steps_done;
    s.live_step      = live_step;
    s.steps_target   = j.steps_target;
    s.checkpoint     = j.checkpoint;
    s.priority       = j.priority;
    s.failure_reason = j.failure_reason;
    // Pending-cancel feedback. Without this a cancelled running job reads plain
    // `running` for up to term_grace (10 s by default) and then jumps straight to
    // `cancelled` — so the operator's click looks like it did nothing, which is
    // exactly the "near-instant OR show it in progress" criterion failing.
    s.cancel_requested = j.cancel_requested;
    return s;
}

// Read '\n'-delimited lines off a blocking socket, buffering partial reads.
struct line_reader {
    int                  fd;
    std::string          buf;
    std::size_t          max_line_bytes;         // cfg_.max_request_bytes
    std::chrono::seconds line_assembly_timeout;  // cfg_.line_assembly_timeout
    // Set when next() returned false specifically because the line exceeded
    // max_line_bytes. Distinguishing this from EOF/timeout lets the caller answer
    // with a real error envelope instead of just hanging up: an over-long line is a
    // CLIENT MISTAKE with a fixable cause (a config past the cap), and a bare close
    // is indistinguishable from a daemon crash from the client's side — it surfaces
    // as "no reply (timed out or connection closed)", which sends the operator
    // hunting the wrong failure.
    bool                 oversize = false;

    // Fill `out` with the next line (CR trimmed), false on EOF/error/oversize/
    // slowloris. Idle time waiting for a NEW line's first byte is not bounded here
    // (SO_RCVTIMEO/keepalive govern that); but once a partial line is buffered, it
    // must complete within line_assembly_timeout — a client dribbling bytes forever
    // is dropped rather than pinning a connection slot.
    bool next(std::string& out) {
        bool armed = false;                       // is the assembly deadline running?
        std::chrono::steady_clock::time_point deadline;
        for (;;) {
            if (const auto nl = buf.find('\n'); nl != std::string::npos) {
                out.assign(buf, 0, nl);
                buf.erase(0, nl + 1);
                if (!out.empty() && out.back() == '\r') out.pop_back();
                return true;
            }
            // Mid-line (bytes buffered but no newline yet): enforce a total-assembly
            // deadline via poll so a byte-at-a-time trickle cannot stall forever.
            if (!buf.empty()) {
                if (!armed) { deadline = std::chrono::steady_clock::now() + line_assembly_timeout; armed = true; }
                const auto rem = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     deadline - std::chrono::steady_clock::now()).count();
                if (rem <= 0) return false;        // slowloris: line never completed in time
                pollfd pfd{fd, POLLIN, 0};
                const int pr = ::poll(&pfd, 1, static_cast<int>(rem));
                if (pr < 0) { if (errno == EINTR) continue; return false; }
                if (pr == 0) return false;         // assembly deadline hit
            }
            char tmp[4096];
            const ssize_t n = ::recv(fd, tmp, sizeof tmp, 0);
            if (n < 0) { if (errno == EINTR) continue; return false; }
            if (n == 0) return false;  // peer closed; a partial trailing line is dropped
            buf.append(tmp, static_cast<std::size_t>(n));
            if (buf.size() > max_line_bytes) { oversize = true; return false; }
        }
    }
};

} // namespace

comms_server::comms_server(core& c, daemon_config cfg, std::string version)
    : core_(c), cfg_(std::move(cfg)), version_(std::move(version)) {}

comms_server::~comms_server() { stop(); }

void comms_server::start() {
    // IPv4 only — and coastd_client resolves host names to IPv4 only to match
    // (resolve_bounded). Change the two together: a client that tried IPv6 first
    // could reach whatever IPv6-only service holds this port number on [::].
    listen_fd_ = cloexec_socket(AF_INET, SOCK_STREAM, 0);   // CLOEXEC from creation
    if (listen_fd_ < 0)
        throw std::runtime_error(std::string("coastd comms: socket() failed: ") + std::strerror(errno));

    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    // CONSCIOUS DESIGN DECISION (do not "harden" to loopback without discussion):
    // coastd binds INADDR_ANY so a single daemon serves remote clients AND localhost
    // over one uniform path (IDD §6.6 — remote headless machines are the whole point
    // of the daemon). The command protocol is deliberately UNAUTHENTICATED: the
    // intended deployment is a TRUSTED LAN / single-tenant lab box, and access
    // control is delegated to the network boundary (firewall / private subnet), not
    // re-implemented here. A `submit` builds+runs a git ref natively, so on an
    // untrusted network this port is remote code execution — that exposure is
    // accepted for the trusted-network target and MUST be reconsidered (auth token
    // or loopback bind) before any deployment reachable by untrusted peers.
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port        = htons(cfg_.tcp_port);

    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof addr) < 0) {
        const std::string msg = std::string("coastd comms: bind port ")
            + std::to_string(cfg_.tcp_port) + " failed: " + std::strerror(errno);
        ::close(listen_fd_);
        listen_fd_ = -1;
        throw std::runtime_error(msg);
    }
    if (::listen(listen_fd_, 16) < 0) {
        const std::string msg = std::string("coastd comms: listen failed: ") + std::strerror(errno);
        ::close(listen_fd_);
        listen_fd_ = -1;
        throw std::runtime_error(msg);
    }
    port_ = local_port(listen_fd_);
    if (port_ == 0) {
        const std::string msg = std::string("coastd comms: getsockname failed: ") + std::strerror(errno);
        ::close(listen_fd_);
        listen_fd_ = -1;
        throw std::runtime_error(msg);
    }

    running_.store(true);
    accept_thread_ = std::thread([this] { accept_loop(); });
    COAST_INFO("coastd.comms", "listening on tcp/{}", port_);
}

void comms_server::accept_loop() {
    while (running_.load()) {
        // Wait for a pending connection with a bounded poll, so the loop re-checks
        // running_ at least every accept_wake_interval_ms. stop()'s shutdown() wakes
        // this at once on Linux, but on macOS it does NOT wake a blocked accept():
        // the thread stayed in accept() and stop()'s join hung forever, so SIGTERM
        // never stopped the daemon. discovery_server had the same problem with
        // recvfrom and solves it the same way.
        pollfd pfd{listen_fd_, POLLIN, 0};
        const int pr = ::poll(&pfd, 1, accept_wake_interval_ms);
        if (pr == 0) continue;                           // idle: re-check running_
        if (pr < 0) {
            const int e = errno;
            if (!running_.load()) break;
            if (e == EINTR || e == EAGAIN || e == ENOMEM) continue;
            COAST_ERROR("coastd.comms", "poll() on the listener failed: {} — listener exiting",
                        std::strerror(e));
            break;
        }

        // CLOEXEC atomically: a child spawned between accept() and a separate
        // set_cloexec() would hold the client connection half-open for its whole run.
        const int fd = cloexec_accept(listen_fd_);
        if (fd < 0) {
            if (!running_.load()) break;                 // stop() shut the listen socket
            const int e = errno;
            if (e == EINTR || e == ECONNABORTED) continue;   // benign: retry immediately
            if (e == EMFILE || e == ENFILE || e == ENOBUFS || e == ENOMEM) {
                // Resource exhaustion is TRANSIENT — killing the listener for the
                // daemon's whole life would be the wrong response. Back off briefly
                // so we don't hot-spin while fds/memory are unavailable, then retry.
                COAST_WARN("coastd.comms", "accept() transient ({}); backing off", std::strerror(e));
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            // A genuinely unexpected accept() error. Log and exit the accept loop
            // rather than spin; the daemon keeps supervising its running jobs.
            COAST_ERROR("coastd.comms", "accept() failed: {} — listener exiting", std::strerror(e));
            break;
        }

        // Arm the socket against SIGPIPE before the first write can happen. A no-op
        // where send() carries MSG_NOSIGNAL; on macOS it is what stops a client that
        // disconnects mid-reply from killing the daemon outright.
        set_nosigpipe(fd);

        reap_finished();   // opportunistically join handlers that have already returned

        std::lock_guard lk(conns_mu_);
        if (!running_.load()) { ::close(fd); break; }
        if (live_fds_.size() >= cfg_.max_connections) {
            // Flood guard: refuse rather than spawn unbounded threads. The client
            // sees an immediate close; a legitimate client retries once load drains.
            COAST_WARN("coastd.comms", "connection cap {} reached; refusing connection",
                       cfg_.max_connections);
            ::close(fd);
            continue;
        }
        const std::uint64_t id = next_conn_id_++;
        live_fds_.insert(fd);
        // Allocate the map node FIRST, then start the thread into it. The obvious
        // spelling — emplace(id, std::thread(...)) — has a fatal ordering: the thread
        // is created as a temporary and the map node is allocated afterwards, so a
        // bad_alloc from the node destroys a JOINABLE thread temporary during
        // unwinding, and ~thread on a joinable thread calls std::terminate outright.
        // The catch below never runs. That is reachable under exactly the memory
        // pressure this file already quantifies (max_request_bytes x max_connections
        // ~ 1.28 GiB of in-flight buffers), and it kills the daemon and orphans every
        // supervised child.
        //
        // try_emplace default-constructs the thread — NOT joinable, so if it throws
        // there is nothing to destroy. The move-assignment that follows is the only
        // step that can leave a live thread, and by then the node exists.
        try {
            const auto [it, ok] = conns_.try_emplace(id);
            if (!ok) throw std::runtime_error("duplicate connection id");   // cannot happen
            try {
                it->second = std::thread([this, fd, id] { serve_conn(fd, id); });
            } catch (...) {
                conns_.erase(it);
                throw;
            }
        } catch (const std::exception& ex) {
            // Thread creation or node allocation failed under resource pressure. Left
            // unguarded the exception would escape accept_thread_'s entry function
            // -> std::terminate (daemon crash). Recover this one connection instead.
            live_fds_.erase(fd);
            ::close(fd);
            COAST_WARN("coastd.comms", "could not start connection thread: {}", ex.what());
        }
    }
}

void comms_server::serve_conn(int fd, std::uint64_t conn_id) {
    // Thread entry: NOTHING may escape. handle_line() has its own blanket catch, but
    // the surrounding machinery allocates outside it — line_reader::next's
    // buf.append, reply.push_back, the oversize .dump(). Those are driven by the
    // PEER: max_request_bytes (10 MiB default) x max_connections (128) is ~1.28 GiB
    // of in-flight buffers before std::string growth doubling, all reachable with
    // in-spec clients. One bad_alloc there would call std::terminate, killing the
    // daemon and orphaning every supervised child — exactly the outcome the header's
    // A10 invariant ("a client error is NEVER a daemon crash") promises cannot
    // happen. Contain it to the one connection, which is what A10 actually means.
    try {
        serve_conn_impl(fd, conn_id);
        return;
    } catch (const std::exception& e) {
        COAST_ERROR("coastd.comms", "connection {} handler failed: {} — dropping the "
                    "connection, daemon continues", conn_id, e.what());
    } catch (...) {
        COAST_ERROR("coastd.comms", "connection {} handler threw a non-exception — "
                    "dropping the connection, daemon continues", conn_id);
    }
    // The impl unregisters, self-registers for joining, and closes fd on its normal
    // path; on the throwing path none of that has happened, so repeat it here rather
    // than leaking an fd and a thread. Erase BEFORE close, for the same fd-reuse
    // reason as the normal path: once closed the number can be reissued to a new
    // connection, and a stale entry would let stop() shut down someone else's socket.
    {
        std::lock_guard lk(conns_mu_);
        live_fds_.erase(fd);
        // Best-effort: this push is itself an allocation, and we may be here BECAUSE
        // allocation is failing. Losing it costs early reaping only — stop() moves
        // and joins every thread in conns_ regardless.
        try {
            finished_.push_back(conn_id);
        } catch (...) {
            COAST_WARN("coastd.comms", "connection {} could not self-register for reaping; "
                       "it will be joined at shutdown instead", conn_id);
        }
    }
    ::close(fd);
}

void comms_server::serve_conn_impl(int fd, std::uint64_t conn_id) {
    enable_keepalive(fd);                             // reap a peer that dies mid-connection
    set_recv_timeout(fd, cfg_.first_request_timeout);   // reap a connect-then-silent client
    set_send_timeout(fd, std::chrono::duration_cast<std::chrono::milliseconds>(cfg_.send_timeout));
    // ^ never cleared: a reply must always make progress, idle session or not.

    line_reader rd{fd, {}, cfg_.max_request_bytes, cfg_.line_assembly_timeout};
    std::string line;
    bool first = true;
    while (running_.load() && rd.next(line)) {
        if (first) {
            // A real request arrived: this is not a wedged connection. Clear the
            // initial read timeout so a legitimately idle persistent session is not
            // severed between commands; keepalive still reaps it if the peer dies.
            set_recv_timeout(fd, std::chrono::milliseconds(0));
            first = false;
        }
        std::string reply = handle_line(line);
        reply.push_back('\n');
        if (!write_all(fd, reply)) break;   // client gone
    }
    // Tell an over-long request WHY it failed before hanging up. The line itself is
    // unusable (we stopped reading mid-stream, so framing is lost and the connection
    // cannot continue), but a diagnosable rejection beats a silent close.
    if (rd.oversize) {
        COAST_WARN("coastd.comms", "request line exceeded max_request_bytes ({}); rejecting",
                   cfg_.max_request_bytes);
        std::string reply = proto::err(
            "request too large (>" + std::to_string(cfg_.max_request_bytes) +
            " bytes for one line). If this was a submit, the config's content exceeds "
            "what this daemon accepts on the wire.").dump();
        reply.push_back('\n');
        (void)write_all(fd, reply);

        // Closing with unread bytes still queued makes the kernel send RST instead
        // of FIN, and an RST can discard data already sitting in the socket's send
        // buffer — i.e. destroy the very explanation just written. The client then
        // sees a bare connection reset for what is a diagnosable, deliberate
        // rejection. An over-long request is exactly the case with megabytes still
        // in flight, so this is the norm here rather than a corner.
        //
        // shutdown(SHUT_WR) sends FIN so the reply is committed, then drain what the
        // peer is still sending until it observes the close. Bounded twice — a byte
        // budget and the receive timeout already on the socket — because the point
        // is an orderly close, not to read an unbounded body from a peer that just
        // proved it sends too much.
        ::shutdown(fd, SHUT_WR);
        char sink[8192];
        std::size_t drained = 0;
        constexpr std::size_t max_drain = 1u << 20;   // 1 MiB is ample to reach EOF
        set_recv_timeout(fd, std::chrono::milliseconds(500));
        while (drained < max_drain) {
            const ssize_t n = ::recv(fd, sink, sizeof sink, 0);
            if (n <= 0) break;                        // EOF, timeout or error: done
            drained += static_cast<std::size_t>(n);
        }
    }
    {
        std::lock_guard lk(conns_mu_);
        live_fds_.erase(fd);        // erase BEFORE close so a reused fd can't be shut down here
        finished_.push_back(conn_id);  // hand our thread back to be joined (we cannot join self)
    }
    ::close(fd);
}

void comms_server::reap_finished() {
    // Move out the finished threads under the lock, then join OUTSIDE it: a handler
    // that is mid-return only needs the lock to record itself, never again, so this
    // cannot deadlock, and we avoid holding conns_mu_ across a blocking join.
    std::vector<std::thread> done;
    {
        std::lock_guard lk(conns_mu_);
        for (const std::uint64_t id : finished_) {
            if (auto it = conns_.find(id); it != conns_.end()) {
                done.push_back(std::move(it->second));
                conns_.erase(it);
            }
        }
        finished_.clear();
    }
    for (auto& t : done) if (t.joinable()) t.join();
}

std::string comms_server::handle_line(const std::string& line) {
    // dump() THROWS type_error.316 on a string that is not valid UTF-8, and job
    // strings carry bytes this daemon did not author: failure_reason quotes a
    // child's stderr tail, label/message come off the wire. One such job makes
    // EVERY `list` fail for as long as it exists — the queue becomes invisible
    // because one job's error text has a stray 0x80 in it. `replace` substitutes
    // U+FFFD instead: the reply stays valid JSON and the operator still sees the
    // job. Applied to every reply, since any of them may echo stored text.
    auto reply = [](const nlohmann::json& j) {
        return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
    };
    try {
        const auto req = proto::parse_line(line);
        if (!req) return reply(proto::err("malformed request line (expected a JSON object with \"cmd\")"));
        const nlohmann::json& a = req->args;

        // Typed field readers. json::value<T>() substitutes the default only when
        // the key is ABSENT — a present-but-wrong-typed one ("steps_target":"abc")
        // THROWS, which handle_line's blanket catch below turns into a bare
        // "internal error: ..." for what is really a malformed field. Every other
        // bad field in this handler gets a targeted message saying which one and
        // why; these give the same.
        const auto want_u64 = [&a](const char* key, std::uint64_t def,
                                   std::optional<std::string>& bad) -> std::uint64_t {
            if (!a.contains(key)) return def;
            const auto& v = a.at(key);
            if (!v.is_number_unsigned()) {
                bad = std::string(key) + " must be a non-negative integer";
                return def;
            }
            return v.get<std::uint64_t>();
        };
        // The string counterpart of want_u64/want_int, and needed for the same
        // reason: json::value<std::string>() substitutes the default only for an
        // ABSENT key, and THROWS type_error.302 for a present-but-wrong-typed one.
        // So {"cmd":"submit","ref":123} produced an opaque
        // "internal error: [json.exception.type_error.302] ..." from the blanket
        // catch, for what is simply a malformed field — while every hand-checked
        // field in this handler names itself and says why.
        const auto want_str = [&a](const char* key, std::optional<std::string>& bad) -> std::string {
            if (!a.contains(key)) return {};
            const auto& v = a.at(key);
            if (!v.is_string()) { bad = std::string(key) + " must be a string"; return {}; }
            return v.get<std::string>();
        };
        const auto want_int = [&a](const char* key, int def,
                                   std::optional<std::string>& bad) -> int {
            if (!a.contains(key)) return def;
            const auto& v = a.at(key);
            if (!v.is_number_integer()) { bad = std::string(key) + " must be an integer"; return def; }
            const auto n = v.get<std::int64_t>();
            if (n < std::numeric_limits<int>::min() || n > std::numeric_limits<int>::max()) {
                bad = std::string(key) + " is out of range";
                return def;
            }
            return static_cast<int>(n);
        };
        std::optional<std::string> bad_field;

        if (req->cmd == proto::cmd::submit) {
            job_spec spec;
            spec.ref_symbolic = want_str("ref", bad_field);
            spec.label        = want_str("label", bad_field);
            spec.user_message = want_str("message", bad_field);
            spec.steps_target = want_u64("steps_target", 0, bad_field);
            spec.priority     = want_int("priority", 0, bad_field);
            if (bad_field) return reply(proto::err("submit: " + *bad_field));
            if (spec.ref_symbolic.empty()) return reply(proto::err("submit: 'ref' is required"));

            // A path is NOT accepted (protocol.h: the client and the daemon are
            // different machines). Name the old field explicitly so a stale client
            // gets a migration message instead of a bare "'config_text' is required",
            // which would read as a malformed request rather than an outdated one.
            if (a.contains("config"))
                return reply(proto::err(
                    "submit: 'config' (a path on the daemon host) is no longer accepted — "
                    "send the config's CONTENT in 'config_text' (+ 'config_name'). "
                    "A path cannot cross a machine boundary, and one that exists on both "
                    "hosts with different content would silently run the wrong config."));
            if (!a.contains("config_text"))
                return reply(proto::err("submit: 'config_text' is required (the config's content, as a string)"));
            if (!a.at("config_text").is_string())
                return reply(proto::err("submit: 'config_text' must be a string (the config's raw bytes)"));

            std::string text = a.at("config_text").get<std::string>();
            if (auto e = validate_submit_config(text, cfg_)) return reply(proto::err(*e));
            spec.config_text = std::move(text);   // prepare from these exact bytes
            // Basename only, for the job-id mnemonic. Never opened, never resolved —
            // any directory part is dropped so a client cannot smuggle a path here.
            spec.config_name = fs::path(want_str("config_name", bad_field)).filename().string();
            if (bad_field) return reply(proto::err("submit: " + *bad_field));

            // Optional initial state for the first launch: another job's final state,
            // named by its JOB ID — never a path (ADR-10: the wire carries no paths).
            // Present means it must validate; it is never dropped.
            if (a.contains("load_state_from")) {
                const std::string src = want_str("load_state_from", bad_field);
                if (bad_field) return reply(proto::err("submit: " + *bad_field));
                const auto view = src.empty() ? std::nullopt : core_.query(src).get();
                const auto init = view ? initial_state_from(src, &view->j, view->b)
                                       : initial_state_from(src, nullptr, bucket::active);
                if (!init) return reply(proto::err("submit: " + init.error()));
                if (spec.steps_target > std::numeric_limits<std::uint64_t>::max() - init->step)
                    return reply(proto::err("submit: steps_target plus the source job's steps overflows"));
                spec.load_state_from = src;
                spec.load_state      = init->path.string();
                spec.load_state_step = init->step;
            }

            const job_id id = core_.submit(std::move(spec)).get();
            return reply(proto::ok(nlohmann::json{{"id", id}}));
        }
        if (req->cmd == proto::cmd::list) {
            const auto views = core_.list().get();
            nlohmann::json arr = nlohmann::json::array();
            for (const auto& v : views) arr.push_back(job_summary_from(v.j, v.b, v.live_step));
            return reply(proto::ok(nlohmann::json{{"jobs", std::move(arr)}}));
        }
        if (req->cmd == proto::cmd::cancel) {
            const std::string id = want_str("id", bad_field);
            if (bad_field)  return reply(proto::err("cancel: " + *bad_field));
            if (id.empty()) return reply(proto::err("cancel: 'id' is required"));
            const bool ok = core_.cancel(id).get();
            return reply(ok ? proto::ok()
                            : proto::err("cancel: unknown or already-terminal job '" + id + "'"));
        }
        if (req->cmd == proto::cmd::reorder) {
            const std::string id = want_str("id", bad_field);
            if (bad_field)         return reply(proto::err("reorder: " + *bad_field));
            if (id.empty())        return reply(proto::err("reorder: 'id' is required"));
            // Two forms, one command: 'delta' moves relative to the live queue
            // (what a polling client wants — its absolute positions are stale by
            // construction), 'pos' sets an absolute slot. 'delta' wins when both
            // are present; an additive field, so no proto_version bump.
            if (a.contains("delta")) {
                const int delta = want_int("delta", 0, bad_field);
                if (bad_field) return reply(proto::err("reorder: " + *bad_field));
                const bool ok = core_.reorder_relative(id, delta).get();
                return reply(ok ? proto::ok()
                                : proto::err("reorder: unknown or non-pending job '" + id + "'"));
            }
            if (!a.contains("pos")) return reply(proto::err("reorder: 'pos' or 'delta' is required"));
            const int pos = want_int("pos", 0, bad_field);
            if (bad_field) return reply(proto::err("reorder: " + *bad_field));
            const bool ok = core_.reorder(id, pos).get();
            return reply(ok ? proto::ok()
                            : proto::err("reorder: unknown or non-pending job '" + id + "'"));
        }
        if (req->cmd == proto::cmd::machine_info) {
            auto m = make_machine_summary(cfg_, version_);
            m.tcp_port = port_;   // the bound port, not a requested 0
            const nlohmann::json mi = m;
            return reply(proto::ok(mi));
        }
        if (req->cmd == proto::cmd::hash_inputs) {
            // Hashed on this handler thread: it blocks only this connection (~1 s
            // for the eval's 173 MB), and hash_limits bounds what one request costs.
            const auto it = a.find("paths");
            if (it == a.end() || !it->is_array())
                return reply(proto::err("hash_inputs: 'paths' must be an array of paths relative to dataset_base"));
            std::vector<std::string> rels;
            for (const auto& p : *it) {
                if (!p.is_string()) return reply(proto::err("hash_inputs: every path must be a string"));
                rels.push_back(p.get<std::string>());
            }
            auto hashes = coastd::hash_inputs(cfg_.dataset_base, rels);
            if (!hashes) return reply(proto::err("hash_inputs: " + hashes.error()));
            return reply(proto::ok(nlohmann::json{{"sha256", std::move(*hashes)}}));
        }
        return reply(proto::err("unknown command '" + req->cmd + "'"));
    } catch (const std::exception& e) {
        // A handler fault is a client-visible error, never a daemon crash (A10).
        return reply(proto::err(std::string("internal error: ") + e.what()));
    }
}

void comms_server::stop() {
    if (!running_.exchange(false)) return;   // idempotent; never started or already stopped

    // Break accept(), then JOIN, then close — in that order. shutdown() wakes the
    // blocked accept() on Linux (measured: it unblocks a listening socket, EINVAL).
    // On macOS it does NOT (sampled 2026-11-03: the accept thread stayed in accept()
    // and this join hung), so accept_loop also self-wakes from a bounded poll — the
    // same fix discovery_server uses for recvfrom. There the join waits at most one
    // accept_wake_interval_ms.
    // Joining guarantees the accept thread is provably done touching listen_fd_;
    // only then is the fd closed. Closing BEFORE the join would free the fd number
    // while the accept thread may still be in — or about to enter — accept() on it,
    // and that number can be recycled immediately (the supervisor opens log fds
    // every poll), so the syscall would silently target an unrelated fd. This
    // ordering removes that race. (close() alone does NOT reliably wake accept.)
    if (listen_fd_ >= 0) ::shutdown(listen_fd_, SHUT_RDWR);
    if (accept_thread_.joinable()) accept_thread_.join();
    if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }

    // Break every live connection's blocking recv, then join ALL handler threads
    // (both those that already finished and those still running). Take the mutex
    // only to snapshot + move the threads out — joining while holding it would
    // deadlock against a thread erasing its own fd.
    std::vector<std::thread> threads;
    {
        std::lock_guard lk(conns_mu_);
        for (const int fd : live_fds_) ::shutdown(fd, SHUT_RDWR);
        for (auto& [id, t] : conns_) threads.push_back(std::move(t));
        conns_.clear();
        finished_.clear();
    }
    for (auto& t : threads) if (t.joinable()) t.join();
    COAST_INFO("coastd.comms", "comms server stopped");
}

} // namespace coastd
