// -----------------------------------------------------------------------------
// coastd comms B-tier gate: launch the REAL coastd binary, drive the whole wire
// protocol through the coastd_client library, and assert every command's
// contract plus an ordered clean shutdown. This is the deterministic connect
// gate — it proves the server edge, the thin client, discovery, machine_info,
// submit-time path-safety, AND that a client blocked on a reply never wedges
// shutdown (comms stops while the core is still alive to drain in-flight).
//
// It runs on an ephemeral data-root and a THROWAWAY git repo, so resolving the
// submitted ref succeeds but no real coast build is ever launched. The daemon
// binds OS-assigned ports (test_daemon.h), so it collides with nothing on the
// host — and the machine_info / discovery checks below prove it advertises the
// port it actually got, not the 0 it was asked for.
// -----------------------------------------------------------------------------
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <arpa/inet.h>
#include <csignal>
#include <netinet/in.h>
#include <poll.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "client/coastd_client.h"
#include "daemon/sock_compat.h"   // dribbling-server test: outlive the client's hangup
#include "daemon/child_proc.h"    // start_token_supported — the platform gate below
#include "test_daemon.h"

using namespace std::chrono_literals;
namespace fs = std::filesystem;
namespace cc = coastd::client;

extern char** environ;

#ifndef COASTD_BIN
#define COASTD_BIN "coastd"
#endif

namespace {

// A unique scratch dir under the system temp root (TMPDIR may be unset).
fs::path make_temp_dir() {
    const char* base = std::getenv("TMPDIR");
    fs::path tmpl = fs::path(base && *base ? base : "/tmp") / "coastd_comms_XXXXXX";
    std::string s = tmpl.string();
    REQUIRE(::mkdtemp(s.data()) != nullptr);
    return fs::path(s);
}

void run(const std::string& cmd) { REQUIRE(std::system(cmd.c_str()) == 0); }

// Send a discovery probe by UNICAST to 127.0.0.1:port (always delivered to
// the INADDR_ANY-bound responder on the same host — deterministic, unlike a
// 255.255.255.255 broadcast which a container/sandbox may filter) and return the
// responder's reply text, or "" on timeout/error.
std::string unicast_probe(std::uint16_t port, std::chrono::milliseconds timeout) {
    const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return {};
    timeval tv{};
    tv.tv_sec  = timeout.count() / 1000;
    tv.tv_usec = static_cast<long>((timeout.count() % 1000) * 1000);
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);

    sockaddr_in to{};
    to.sin_family = AF_INET;
    to.sin_port   = htons(port);
    ::inet_pton(AF_INET, "127.0.0.1", &to.sin_addr);
    const std::string probe(coastd::proto::discovery_probe);
    ::sendto(fd, probe.data(), probe.size(), 0, reinterpret_cast<sockaddr*>(&to), sizeof to);

    char buf[2048];
    const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
    ::close(fd);
    return n > 0 ? std::string(buf, static_cast<std::size_t>(n)) : std::string{};
}

// Raw blocking TCP connect to 127.0.0.1:port (bypasses the client lib so a test
// can hold many sockets open without sending a request). -1 on failure.
int raw_connect(std::uint16_t port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    sockaddr_in to{};
    to.sin_family = AF_INET;
    to.sin_port   = htons(port);
    ::inet_pton(AF_INET, "127.0.0.1", &to.sin_addr);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&to), sizeof to) != 0) { ::close(fd); return -1; }
    return fd;
}

// Did the server orderly-close this connection (i.e. refuse it)? NON-blocking: a
// refused socket is readable and recv returns 0 (EOF); a still-held one (server
// blocked awaiting our request line) is not readable, so poll returns 0. This must
// stay non-blocking — probing 100+ held sockets with a per-fd timeout would serialize
// into seconds of dead wait. Callers poll it until the expected refusals show.
bool server_closed(int fd) {
    pollfd p{fd, POLLIN, 0};
    if (::poll(&p, 1, 0) <= 0) return false;   // nothing to read -> still held
    char b = 0;
    return ::recv(fd, &b, 1, MSG_DONTWAIT) == 0;   // readable + EOF -> refused/closed
}

// Launches of a job's child so far: coastd writes one <runs_root>/<id>/logs/run_*
// per launch (runs_root defaults to <data-root>/runs).
int launches(const fs::path& data_root, const std::string& id) {
    std::error_code ec;
    int n = 0;
    for (const auto& de : fs::directory_iterator(data_root / "runs" / id / "logs", ec))
        if (de.path().filename().string().starts_with("run_")) ++n;
    return n;
}

// Wait until pred(job) holds — or the job settles, or is launched more than
// `max_launches` times; either of the last two ends the wait too, so neither an
// early terminal bucket nor a relaunch loop is waited on forever. The heartbeat is
// the job's own state (everything observable about it): it stands still when the
// job is stuck, however busy the daemon is — the daemon's pass counter would keep
// ticking on its timer. Returns the last summary seen; callers assert on it.
template <class Pred>
std::optional<coastd::proto::job_summary> await_job(cc::connection& c, const fs::path& data_root,
                                                     const std::string& id, Pred pred,
                                                     int max_launches = 1) {
    const auto get = [&]() -> std::optional<coastd::proto::job_summary> {
        std::string e;
        auto l = c.list(e);
        if (!l) return std::nullopt;
        for (auto& j : *l) if (j.id == id) return j;
        return std::nullopt;
    };
    std::optional<coastd::proto::job_summary> last;
    (void)coast_test::await_progress(
        [&] {
            last = get();
            return last && (pred(*last) || last->bucket != "active" ||
                            launches(data_root, id) > max_launches);
        },
        [&] {
            const auto j = get();
            return j ? j->bucket + '|' + j->status + '|' + std::to_string(j->steps_done) + '|' +
                           std::to_string(j->live_step) + '|' + j->checkpoint + '|' +
                           (j->cancel_requested ? "c" : "-") + '|' + j->failure_reason + '|' +
                           std::to_string(launches(data_root, id))
                     : std::string("?");
        },
        coast_test::default_stall, std::chrono::milliseconds{20});
    return last;
}

// OS-assigned ports; the daemon publishes them in coastd.ports (test_daemon.h).
pid_t spawn_coastd(const fs::path& data_root, const fs::path& repo) {
    const char* argv[] = {COASTD_BIN,
                          "--data-root", data_root.c_str(),
                          "--repo",      repo.c_str(),
                          "--tcp-port",  "0",
                          "--udp-port",  "0",
                          "-q", nullptr};
    pid_t pid = 0;
    REQUIRE(::posix_spawn(&pid, COASTD_BIN, nullptr, nullptr,
                          const_cast<char* const*>(argv), environ) == 0);
    return pid;
}

} // namespace

// ---------------------------------------------------------------------------
// PLATFORM GATE. The cases below need coastd's Linux-only pid-guard: either
// read_start_token() directly, or a running daemon — and the daemon REFUSES to
// start where start_token_supported() is false (main.cpp), because without a
// recycle-safe start token the duplicate-run guard is inoperative and silent
// corruption is on the table (ADR-16). Off Linux they cannot pass, so they skip
// rather than hard-fail: macOS is a first-class dev platform here and a Mac dev
// running the documented `ctest` must not be handed red that means "wrong OS".
// Gated on the seam, not on __linux__, so porting read_start_token() re-enables
// them automatically — the same predicate the daemon's own startup gate asks.
// ---------------------------------------------------------------------------
constexpr bool skip_off_linux = !coastd::start_token_supported();

TEST_CASE("coastd comms end-to-end protocol + ordered shutdown" * doctest::skip(skip_off_linux)) {
    const fs::path root      = make_temp_dir();
    const fs::path data_root = root / "data";
    const fs::path repo      = root / "repo";
    fs::create_directories(data_root);
    fs::create_directories(repo);

    // Throwaway repo: resolve(HEAD) succeeds; no CMakeLists -> no real build.
    run("git -C '" + repo.string() + "' init -q");
    run("git -C '" + repo.string() + "' -c user.email=t@t -c user.name=t commit -q --allow-empty -m init");

    // A path-safe config, and a rejectable one that points into the daemon's state dir.
    { std::ofstream(data_root / "good.json") << "{\"seed\":1}\n"; }
    { std::ofstream(data_root / "bad.json")
          << "{\"snapshot_dir\":\"" << (data_root / "state" / "evil").string() << "\"}\n"; }

    // Daemon config (auto-discovered at <data-root>/coastd.json): shrink the two
    // comms timeouts to 1 s so the slowloris / silent-reap gates below trip fast.
    // Safe for the happy-path subcases — a real client sends a complete line within
    // milliseconds, so neither timeout ever fires for it.
    // A real dataset_base (an existing dir) so machine_info's data_link_ok can be
    // asserted TRUE below — the false case is the default and is checked in
    // test_machine_info.cpp, which does not need a live daemon.
    fs::create_directories(root / "datasets");
    { std::ofstream(data_root / "coastd.json")
          << "{\"comms_first_request_timeout_seconds\":1,"
             "\"comms_line_assembly_timeout_seconds\":1,"
             // An hour of grace: the pending-cancel window below is then held
             // open by construction, until the test releases its child.
             "\"term_grace_seconds\":3600,"
             "\"dataset_base\":\"" << (root / "datasets").string() << "\"}\n"; }

    coastd_test::daemon_guard daemon(spawn_coastd(data_root, repo));
    const auto ports = coastd_test::wait_for_ports(data_root, daemon.pid());
    REQUIRE_MESSAGE(ports.has_value(), "daemon exited or never published its ports");
    const std::uint16_t tcp_port = ports->tcp, udp_port = ports->udp;
    REQUIRE(udp_port != 0);   // the discovery subcases need a responder

    cc::endpoint ep;
    ep.host = "localhost";
    ep.port = tcp_port;

    // Wait for the listener. CONNECT stays short (this loop is a poll for "is it up
    // yet"), but the OPERATION bound is given explicitly and generously: with the
    // io_timeout argument omitted it defaults to the connect bound, and 200 ms is
    // not a whole-request budget here. Two subcases below push a 5 MiB and an 11 MiB
    // config down this connection, and the daemon must recv all of it before it can
    // answer — so a shared 200 ms turns the daemon's real, correct rejection into a
    // client-side timeout, and the test then fails on the wrong thing. Measured: it
    // fails deterministically under TSan and is a latent flake on a loaded box.
    std::string err;
    std::optional<cc::connection> conn;
    REQUIRE_MESSAGE(coast_test::await_condition([&] {
                        conn = cc::connection::open(ep, coast_test::default_stall, err,
                                                    /*io_timeout=*/coast_test::default_stall);
                        return conn.has_value();
                    }),
                    "connect: " << err);

    // Send one request now so the shared connection is "activated": the (shortened
    // to 1 s) first_request_timeout no longer applies to it once a real request has
    // arrived, so it survives the multi-second idles in the timeout subcases below
    // (TCP keepalive governs a genuinely idle session from here, not first-request).
    { std::string e0; conn->machine_info(e0); }

    SUBCASE("full protocol surface") {
        // machine_info — static, no core hop.
        auto mi = conn->machine_info(err);
        REQUIRE_MESSAGE(mi.has_value(), err);
        CHECK(!mi->hostname.empty());
        CHECK(mi->os == std::string(coastd_test::expected_os));
        CHECK(!mi->coastd_version.empty());
        CHECK(mi->slots >= 1);
        CHECK(mi->tcp_port == tcp_port);   // daemon advertises its command port
        // The client can tell, BEFORE submitting, whether a config's relative
        // "data/…" paths will resolve on this machine.
        CHECK(mi->data_link_ok);           // dataset_base is set and exists
        CHECK(mi->dataset_base == (root / "datasets").string());

        // hash_inputs — the wire round trip; the containment rules themselves are
        // covered in test_hash_inputs.cpp.
        { std::ofstream(root / "datasets" / "abc", std::ios::binary) << "abc"; }
        { auto h = conn->hash_inputs({"abc"}, err);
          REQUIRE_MESSAGE(h.has_value(), err);
          CHECK(h->at("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"); }
        { std::string e;
          CHECK_FALSE(conn->hash_inputs({"../coastd.json"}, e).has_value());
          CHECK(e.find("contains '..'") != std::string::npos); }

        // list — empty at start.
        auto l0 = conn->list(err);
        REQUIRE_MESSAGE(l0.has_value(), err);
        CHECK(l0->empty());

        // submit — path-unsafe config rejected, daemon stays up.
        { cc::submit_args a; a.ref = "HEAD";
          std::string e;
          REQUIRE(cc::load_config_file(a, data_root / "bad.json", e));
          auto id = conn->submit(a, e);
          CHECK_FALSE(id.has_value());
          CHECK(e.find("daemon-critical") != std::string::npos); }

        // submit — accepted.
        std::string job_id;
        { cc::submit_args a; a.ref = "HEAD";
          REQUIRE(cc::load_config_file(a, data_root / "good.json", err));
          a.label = "gate"; a.steps_target = 10;
          auto id = conn->submit(a, err);
          REQUIRE_MESSAGE(id.has_value(), err);
          CHECK_FALSE(id->empty());
          job_id = *id; }

        // list — one job, fields preserved.
        auto l1 = conn->list(err);
        REQUIRE_MESSAGE(l1.has_value(), err);
        REQUIRE(l1->size() == 1);
        CHECK((*l1)[0].id == job_id);
        CHECK((*l1)[0].bucket == "active");
        CHECK((*l1)[0].ref_symbolic == "HEAD");

        // reorder — known pending job OK, unknown rejected.
        CHECK(conn->reorder(job_id, 0, err));
        { std::string e; CHECK_FALSE(conn->reorder("no-such-job", 0, e)); }

        // reorder_by — the relative form the GUI's Up/Down buttons send. This is
        // the wire dispatch that once went missing while core/client/GUI all had
        // the feature: keep a real round-trip on it.
        CHECK(conn->reorder_by(job_id, -1, err));
        { std::string e; CHECK_FALSE(conn->reorder_by("no-such-job", -1, e)); }

        // cancel — known OK, unknown rejected.
        CHECK(conn->cancel(job_id, err));
        { std::string e; CHECK_FALSE(conn->cancel("no-such-job", e)); }

        // The connection is reusable across many round-trips (one blocking socket,
        // request/response framing) — confirm it is still alive after the batch.
        CHECK(conn->machine_info(err).has_value());
    }

    // The cancel-UX acceptance criterion (Discover §5): cancel is either
    // near-instant OR the client can see it is in progress. It is not instant —
    // the daemon SIGTERMs the child and waits out term_grace so it can write a
    // final checkpoint — so the flag must be visible on the wire during that
    // window. Driven end-to-end against the REAL daemon with a REAL child that
    // ignores SIGTERM, which is what holds the window open long enough to assert.
    SUBCASE("a pending cancel is visible on the wire while the child is stopping") {
        // Pre-place a "built" worktree for HEAD so the daemon spawns instead of
        // trying (and failing) to build this CMakeLists-less repo. Done here, after
        // boot, precisely because reconcile re-derives readiness every pass — the
        // same path a real build completing mid-run takes.
        run("git -C '" + repo.string() + "' rev-parse HEAD > '" +
            (root / "sha.txt").string() + "'");
        std::string sha;
        { std::ifstream in(root / "sha.txt"); std::getline(in, sha); }
        REQUIRE(sha.size() == 40);

        const fs::path wt = data_root / "worktrees" / sha;
        fs::create_directories(wt / "build");
        // Ignores the graceful stop and runs until the test releases it — so the
        // job stays "stopping" for as long as the test needs, not for a grace.
        const fs::path release = root / "release_cancelux";
        { std::ofstream f(wt / "build" / "coast");
          f << "#!/usr/bin/env bash\n"
               "trap '' TERM\n"                       // ignore the graceful stop
               "echo \"@coast/progress v1 phase=step step=1 start=0 target=0 sps=1.0 logged=0\"\n"
               "while [ ! -e '" << release.string() << "' ]; do sleep 0.2; done\n"; }
        fs::permissions(wt / "build" / "coast",
                        fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec);
        { std::ofstream(wt / ".coastd-ready") << "ready"; }

        std::string cid;
        { cc::submit_args a; a.ref = "HEAD";
          REQUIRE(cc::load_config_file(a, data_root / "good.json", err));
          a.label = "cancelux"; a.steps_target = 1'000'000;
          auto id = conn->submit(a, err);
          REQUIRE_MESSAGE(id.has_value(), err);
          cid = *id; }

        const auto find = [&](auto pred) {
            auto l = conn->list(err);
            REQUIRE_MESSAGE(l.has_value(), err);
            for (const auto& j : *l) if (j.id == cid && pred(j)) return true;
            return false;
        };

        // Wait until it is actually running (a child exists to be stopped).
        const auto r = await_job(*conn, data_root, cid,
                                 [](const auto& j) { return j.status == "running"; });
        REQUIRE_MESSAGE((r && r->status == "running"), "job never reached running — nothing to cancel");

        CHECK(conn->cancel(cid, err));

        // Now the assertion that matters: while the child is still being stopped,
        // the wire says so. Before this field existed the job read plain `running`
        // for the whole grace period and then jumped to `cancelled` — a click that
        // looked like it did nothing.
        // The core records the request inside the cancel RPC (core::do_cancel), and
        // the child cannot exit before the test releases it — so the very next
        // list must show the job still running with the flag set. No waiting.
        CHECK_MESSAGE(find([](const auto& j) {
                          return j.bucket == "active" && j.cancel_requested &&
                                 j.status == "running";   // real status, not a fake token
                      }),
                      "cancel_requested is not on the wire right after the cancel");

        // And it does settle once the child is gone. (The SIGKILL after the grace
        // is test_fault_matrix.cpp's "cancel: a child that outlives term_grace".)
        { std::ofstream(release) << "go"; }
        const auto settled = await_job(*conn, data_root, cid,
                                       [](const auto& j) { return j.bucket == "cancelled"; });
        CHECK_MESSAGE((settled && settled->bucket == "cancelled"), "job never settled into cancelled/");
    }

    SUBCASE("UDP discovery responder (deterministic, unicast loopback)") {
        // HARD assertion on the responder: a unicast probe on loopback must be
        // answered with a well-formed machine_summary. This is deterministic
        // (no broadcast), so it genuinely gates the discovery_server.
        const std::string reply = unicast_probe(udp_port, coast_test::default_stall);
        REQUIRE_MESSAGE(!reply.empty(), "discovery responder did not answer a unicast probe");
        auto j = nlohmann::json::parse(reply, nullptr, false);
        REQUIRE(j.is_object());
        const auto m = j.get<coastd::proto::machine_summary>();
        CHECK(!m.hostname.empty());
        CHECK(m.os == std::string(coastd_test::expected_os));
        CHECK(!m.coastd_version.empty());
        CHECK(m.tcp_port == tcp_port);   // reply advertises the connectable command port
    }

    SUBCASE("UDP discovery client helper (broadcast, best-effort)") {
        // The client discover() uses a 255.255.255.255 broadcast, which a
        // container/sandbox may filter — so a zero result is tolerated here. But
        // any daemon that DOES answer must be well-formed. The responder itself is
        // gated deterministically by the unicast subcase above.
        const auto found = cc::discover(500ms, udp_port);
        for (const auto& m : found) {
            CHECK(!m.hostname.empty());
            CHECK(m.os == std::string(coastd_test::expected_os));
            CHECK(!m.coastd_version.empty());
            CHECK(m.tcp_port == tcp_port);   // advertised port -> connectable
            CHECK(!m.address.empty());       // recvfrom() captured the source IP
        }
        MESSAGE("broadcast discover() found " << found.size() << " daemon(s)"
                << (found.empty() ? " (broadcast may be filtered here)" : ""));
    }

    SUBCASE("fixed-list connect_to (deterministic primary, IDD §6.6)") {
        // Point the client's fixed-list config at this daemon by NAME and connect
        // through connect_to — the deterministic path with no broadcast.
        const fs::path hosts = root / "hosts.json";
        { std::ofstream(hosts) << "[{\"name\":\"gate-host\",\"host\":\"127.0.0.1\",\"port\":"
                               << tcp_port << "}]\n"; }
        ::setenv("COASTD_HOSTS", hosts.c_str(), 1);

        std::string e;
        auto c2 = cc::connect_to("gate-host", coast_test::default_stall, e);
        REQUIRE_MESSAGE(c2.has_value(), "connect_to: " << e);
        CHECK(c2->machine_info(e).has_value());

        // An unknown name fails cleanly (helpful message, never a crash).
        auto c3 = cc::connect_to("no-such-host", 500ms, e);
        CHECK_FALSE(c3.has_value());
        CHECK(e.find("no-such-host") != std::string::npos);

        ::unsetenv("COASTD_HOSTS");
    }

    // ---- DoS-mitigation regression gates (coastd_cleanup_todo.md #29) ---------
    // These lock in two of the comms hardening bounds so a refactor can't silently
    // drop them. The slowloris (line_assembly / first_request) and send-timeout
    // bounds are NOT covered here: their thresholds are a hardcoded 30 s, so a fast
    // test cannot trip them — exercising those needs the timeouts made config-driven
    // first (noted in design/coastd_cleanup_todo.md #29).

    SUBCASE("oversized config is rejected by the daemon's own size bound") {
        // Bypass load_config_file deliberately: the client pre-flight would reject
        // this first, and the bound under test is the DAEMON's. A client is never
        // trusted, so the server-side cap must hold on its own. Valid JSON, > 4 MiB
        // (max_config_bytes), and well under max_request_bytes so it is the config
        // cap that trips and not the line cap.
        cc::submit_args a; a.ref = "HEAD"; a.config_name = "big.json";
        a.config_text = "{\"pad\":\"" + std::string(5u << 20, 'x') + "\"}";
        std::string e;
        auto id = conn->submit(a, e);
        CHECK_FALSE(id.has_value());
        CHECK(e.find("too large") != std::string::npos);

        // The daemon rejected it without dying: the connection is still serviceable.
        std::string e2;
        CHECK(conn->machine_info(e2).has_value());
    }

    SUBCASE("a config larger than one request line gets an ERROR, not a silent hangup") {
        // Past max_request_bytes the line reader stops mid-stream and the connection
        // cannot continue. It must still SAY so: a bare close is indistinguishable
        // from a daemon crash on the client side ("no reply (timed out or connection
        // closed)"), which points the operator at the wrong failure entirely.
        cc::submit_args a; a.ref = "HEAD"; a.config_name = "huge.json";
        a.config_text = "{\"pad\":\"" + std::string(11u << 20, 'x') + "\"}";  // > 10 MiB line cap
        std::string e;
        auto id = conn->submit(a, e);
        CHECK_FALSE(id.has_value());
        CHECK(e.find("too large") != std::string::npos);   // a real message, not "no reply"
        CHECK(e.find("no reply") == std::string::npos);
    }

    SUBCASE("a non-object (top-level array) config is cleanly rejected, not a broken promise") {
        // A top-level JSON ARRAY parses fine but is not a coast config. Before the
        // fix, validation accepted any valid JSON and prepare_run_config's
        // cfg["snapshot_dir"]=… threw a type_error that unwound past the submit
        // future — the client saw an opaque "internal error: … broken promise" and
        // no job existed. Now it is rejected at the edge with a clear message.
        // Sent raw, again to exercise the DAEMON's check rather than the client's.
        cc::submit_args a; a.ref = "HEAD"; a.config_name = "array.json";
        a.config_text = "[{\"seed\":1}]\n";
        std::string e;
        auto id = conn->submit(a, e);
        CHECK_FALSE(id.has_value());
        CHECK(e.find("object") != std::string::npos);          // helpful, specific message
        CHECK(e.find("broken promise") == std::string::npos);  // NOT the old opaque failure
        CHECK(e.find("internal error") == std::string::npos);
        // The daemon rejected it without dying: the connection is still serviceable.
        std::string e2;
        CHECK(conn->machine_info(e2).has_value());
    }

    SUBCASE("the config travels as CONTENT — a path is refused with a migration message") {
        // The defect this replaced: `config` named a file on the DAEMON's filesystem.
        // Across a machine boundary that is meaningless, and a path present on both
        // hosts with different content would have silently run the wrong config.
        // A stale client must learn that, not just see "'config_text' is required".
        // Sent over a raw socket: the client library cannot express the old shape
        // any more, which is itself the point — but the DAEMON must still answer a
        // legacy client intelligibly rather than with a generic complaint.
        const int fd = raw_connect(tcp_port);
        REQUIRE(fd >= 0);
        const nlohmann::json req{{"cmd", "submit"}, {"ref", "HEAD"},
                                 {"config", (data_root / "good.json").string()}};
        const std::string line = req.dump() + "\n";
        REQUIRE(::send(fd, line.data(), line.size(), 0) == static_cast<ssize_t>(line.size()));

        std::string buf(8192, '\0');
        const ssize_t n = ::recv(fd, buf.data(), buf.size(), 0);
        ::close(fd);
        REQUIRE(n > 0);
        buf.resize(static_cast<std::size_t>(n));

        const auto r = nlohmann::json::parse(buf, nullptr, false);
        REQUIRE_FALSE(r.is_discarded());
        CHECK_FALSE(r.value("ok", false));
        const std::string msg = r.value("error", std::string{});
        CHECK(msg.find("no longer accepted") != std::string::npos);
        CHECK(msg.find("config_text") != std::string::npos);   // says what to send instead
    }

    SUBCASE("load_state_from: a done job's final state starts the next job; bad ids are refused") {
        // Raw socket: the wire shape is the contract a non-C++ client (the eval
        // runner) speaks.
        const auto submit = [&](const nlohmann::json& from) {
            const nlohmann::json req{{"cmd", "submit"}, {"ref", "HEAD"},
                                     {"config_text", "{\"seed\":1}"}, {"steps_target", 3000},
                                     {"load_state_from", from}};
            const int fd = raw_connect(tcp_port);
            REQUIRE(fd >= 0);
            const std::string line = req.dump() + "\n";
            REQUIRE(::send(fd, line.data(), line.size(), 0) == static_cast<ssize_t>(line.size()));
            std::string buf(8192, '\0');
            const ssize_t n = ::recv(fd, buf.data(), buf.size(), 0);
            ::close(fd);
            REQUIRE(n > 0);
            buf.resize(static_cast<std::size_t>(n));
            const auto r = nlohmann::json::parse(buf, nullptr, false);
            REQUIRE_FALSE(r.is_discarded());
            return r;
        };
        // Wait for a job to settle; returns its bucket ("" if it got stuck first).
        const auto settled = [&](const std::string& id) {
            const auto j = await_job(*conn, data_root, id, [](const auto& v) { return v.bucket != "active"; });
            return (j && j->bucket != "active") ? j->bucket : std::string{};
        };

        // Pre-place a "built" HEAD worktree (see the cancel subcase) whose fake
        // coast records its argv, then writes --save-state and a clean manifest.
        // Like coast, it continues a loaded state's step count: its "state file"
        // holds the step, and n_steps_completed = loaded step + -n.
        run("git -C '" + repo.string() + "' rev-parse HEAD > '" + (root / "sha.txt").string() + "'");
        std::string sha;
        { std::ifstream in(root / "sha.txt"); std::getline(in, sha); }
        REQUIRE(sha.size() == 40);
        const fs::path wt = data_root / "worktrees" / sha;
        fs::create_directories(wt / "build");
        { std::ofstream f(wt / "build" / "coast");
          f << "#!/usr/bin/env bash\n"
               "echo \"$@\" > argv.txt\n"            // cwd = the run dir
               "n=0; start=0; save=''; manifest=''\n"
               "while [ $# -gt 0 ]; do case \"$1\" in\n"
               "  -n) n=\"$2\"; shift 2;; --save-state) save=\"$2\"; shift 2;;\n"
               "  --load-state) start=$(cat \"$2\"); shift 2;;\n"
               "  --emit-run-manifest) manifest=\"$2\"; shift 2;; *) shift;; esac; done\n"
               "echo $((start + n)) > \"$save\"\n"
               "printf '{\"n_steps_completed\": %s, \"exit_reason\": \"completed_n_steps\"}' "
               "$((start + n)) > \"$manifest\"\n"; }
        fs::permissions(wt / "build" / "coast",
                        fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec);
        { std::ofstream(wt / ".coastd-ready") << "ready"; }

        std::string train;
        { cc::submit_args a; a.ref = "HEAD";
          REQUIRE(cc::load_config_file(a, data_root / "good.json", err));
          a.label = "train"; a.steps_target = 40;
          auto id = conn->submit(a, err);
          REQUIRE_MESSAGE(id.has_value(), err);
          train = *id; }
        REQUIRE(settled(train) == "done");

        const auto ok = submit(train);
        REQUIRE_MESSAGE(ok.value("ok", false), ok.dump());
        const std::string id = ok.at("id").get<std::string>();
        REQUIRE(settled(id) == "done");

        // The job record (what a daemon restart relaunches from) carries the source,
        // its resolved final state and its step; steps_target is in coast's step space.
        const fs::path final_state = fs::canonical(data_root / "runs" / train / "final_state.h5");
        nlohmann::json rec;
        { std::ifstream in(data_root / "state" / "jobs" / "done" / (id + ".json")); in >> rec; }
        CHECK(rec.at("load_state_from") == train);
        CHECK(rec.at("load_state") == final_state.string());
        CHECK(rec.at("load_state_step") == 40);
        CHECK(rec.at("steps_target") == 3040);
        // And the first launch's command line loaded it and ran the requested steps.
        std::string argv;
        { std::ifstream in(data_root / "runs" / id / "argv.txt"); std::getline(in, argv); }
        CHECK(argv.find("--load-state " + final_state.string()) != std::string::npos);
        CHECK(argv.find("-n 3000 ") != std::string::npos);

        const auto unknown = submit("no-such-job");
        CHECK_FALSE(unknown.value("ok", true));
        CHECK(unknown.value("error", "").find("no known job") != std::string::npos);

        const auto empty = submit("");
        CHECK_FALSE(empty.value("ok", true));
        CHECK(empty.value("error", "").find("empty") != std::string::npos);

        const auto mistyped = submit(5);
        CHECK_FALSE(mistyped.value("ok", true));
        CHECK(mistyped.value("error", "").find("must be a string") != std::string::npos);
    }

    SUBCASE("the submitted bytes are archived verbatim, formatting intact") {
        // config.json is REWRITTEN by the daemon (snapshot_dir injected, re-dumped),
        // so it is not evidence of what the user submitted. config.submitted.json is.
        // This is the end-to-end byte-fidelity guarantee of the string wire form:
        // indentation, key order and newlines survive the JSONL round trip exactly.
        const std::string pretty =
            "{\n"
            "    \"seed\": 1,\n"
            "    \"aaa_last_alphabetically_but_first_in_file\": true,\n"
            "    \"nested\": {\n"
            "        \"deep\": [1, 2, 3]\n"
            "    }\n"
            "}\n";
        const fs::path src = data_root / "pretty.json";
        { std::ofstream o(src, std::ios::binary); o << pretty; }

        cc::submit_args a; a.ref = "HEAD";
        std::string e;
        REQUIRE(cc::load_config_file(a, src, e));
        CHECK(a.config_text == pretty);        // the client sends the ORIGINAL bytes
        CHECK(a.config_name == "pretty.json"); // basename only, for the mnemonic

        auto id = conn->submit(a, e);
        REQUIRE_MESSAGE(id.has_value(), e);

        const fs::path archived = data_root / "runs" / *id / "config.submitted.json";
        REQUIRE(fs::exists(archived));
        std::ifstream in(archived, std::ios::binary);
        std::ostringstream ss; ss << in.rdbuf();
        CHECK(ss.str() == pretty);             // byte-identical, indentation and all

        // And the file that actually RUNS is the rewritten one — different by design.
        const fs::path effective = data_root / "runs" / *id / "config.json";
        REQUIRE(fs::exists(effective));
        std::ifstream in2(effective, std::ios::binary);
        std::ostringstream ss2; ss2 << in2.rdbuf();
        CHECK(ss2.str() != pretty);
        CHECK(ss2.str().find("snapshot_dir") != std::string::npos);
    }

    SUBCASE("connection cap refuses a flood, and the daemon survives it") {
        // Open well past max_conns (128) sockets, each sending one complete request
        // right after its connect. An accepted socket answers it and then stays a
        // persistent session (the first-request timeout is cleared once a request
        // arrives), so it holds its slot until the test closes it. A refused one is
        // closed unanswered. Every socket thus ends in one of two final states, and
        // no timer decides which. (An idle socket would not do: the fixture's 1 s
        // first-request timeout closes it whether or not the cap exists.) `conn`
        // already holds one slot.
        constexpr int kAttempts = 160;   // > max_conns=128 by a comfortable margin
        const std::string ping = nlohmann::json{{"cmd", "machine_info"}}.dump() + "\n";
        std::vector<int> fds;
        fds.reserve(kAttempts);
        for (int i = 0; i < kAttempts; ++i)
            if (const int f = raw_connect(tcp_port); f >= 0) {
                fds.push_back(f);
                // A refused socket may already be closed: then this send fails, and
                // MSG_NOSIGNAL keeps that from killing the test with SIGPIPE.
                (void)::send(f, ping.data(), ping.size(), MSG_NOSIGNAL);
            }

        // 1 answered, 0 refused (EOF or reset without a reply), -1 not yet settled.
        // Non-blocking, like server_closed: a per-fd timeout over 160 sockets
        // would serialise into dead waits.
        std::vector<int> state(fds.size(), -1);
        const auto all_settled = [&] {
            bool all = true;
            for (std::size_t i = 0; i < fds.size(); ++i) {
                if (state[i] != -1) continue;
                pollfd p{fds[i], POLLIN, 0};
                if (::poll(&p, 1, 0) <= 0) { all = false; continue; }
                char b = 0;
                state[i] = ::recv(fds[i], &b, 1, MSG_DONTWAIT) > 0 ? 1 : 0;
            }
            return all;
        };
        CHECK(coast_test::await_condition(all_settled));
        const auto answered = std::count(state.begin(), state.end(), 1);
        const auto refused  = std::count(state.begin(), state.end(), 0);
        INFO("answered ", answered, ", refused ", refused, " of ", fds.size());
        CHECK(answered + refused == static_cast<std::ptrdiff_t>(fds.size()));
        CHECK(answered >= 1);
        CHECK(answered <= 128);   // never more sessions than max_connections
        CHECK(refused >= 1);      // the cap fired: the excess was closed unanswered

        for (const int f : fds) ::close(f);

        // Survived the flood and freed slots: a fresh round-trip still works.
        std::string e;
        CHECK(conn->machine_info(e).has_value());
    }

    SUBCASE("a connect-then-silent client is reaped (first_request_timeout)") {
        // Connect and send NOTHING. The daemon must drop it after the (config-set
        // 1 s) first-request timeout instead of pinning a connection slot forever.
        // Reaped once the timeout has passed — awaited, not slept for. (That it is
        // not reaped EARLIER cannot be checked without racing the 1 s timeout,
        // so it is not asserted.)
        const int fd = raw_connect(tcp_port);
        REQUIRE(fd >= 0);
        CHECK(coast_test::await_condition([&] { return server_closed(fd); }));   // reaped
        ::close(fd);
        std::string e; CHECK(conn->machine_info(e).has_value());   // daemon unaffected
    }

    SUBCASE("a slowloris partial line is dropped (line_assembly_timeout)") {
        // Send a partial request line (no newline) and then dribble nothing. Once a
        // partial line is buffered the daemon bounds TOTAL assembly time (config-set
        // 1 s), so this connection is dropped rather than held open forever.
        const int fd = raw_connect(tcp_port);
        REQUIRE(fd >= 0);
        const char* partial = "{\"cmd\":\"list\"";           // valid prefix, but NO '\n'
        REQUIRE(::send(fd, partial, std::strlen(partial), 0) > 0);
        CHECK(coast_test::await_condition([&] { return server_closed(fd); }));   // dropped
        ::close(fd);
        std::string e; CHECK(conn->machine_info(e).has_value());   // daemon unaffected
    }

    // Ordered shutdown: SIGTERM -> clean exit, no hang.
    conn.reset();
    CHECK(daemon.stop());
    // The wait macros need an lvalue on macOS: <sys/wait.h> takes the address.
    const int st = daemon.status();
    CHECK(WIFEXITED(st));
    CHECK(WEXITSTATUS(st) == 0);

    std::error_code ec;
    fs::remove_all(root, ec);
}

// The whole-reply deadline must survive the move out of connection::open().
// connection's move members are hand-written (the source fd must be nulled) and
// once dropped timeout_, silently resetting it to 0 = unbounded on the move out
// of open() — i.e. in every real caller. SO_RCVTIMEO alone cannot bound a
// dribbling peer (it restarts on every arriving byte), so the whole-reply
// deadline is the only thing standing between a byte-per-window server and a
// held GUI render thread. Regression: a server that dribbles bytes faster than
// SO_RCVTIMEO but never completes a reply line must be abandoned at roughly the
// io timeout, not held for as long as it keeps dribbling.
// The deterministic half of the regression below: the bound itself survives the
// move out of open(), move construction and move assignment — read back
// directly, without timing a reply (CLAUDE.md "Testing"). The listener never
// accepts: connect() completes against its backlog and nothing is ever sent.
TEST_CASE("client: io_timeout survives open(), move construction and move assignment") {
    const int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
    REQUIRE(lfd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = 0;                       // ephemeral: no fixture-port collision
    ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    REQUIRE(::bind(lfd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0);
    socklen_t alen = sizeof addr;
    REQUIRE(::getsockname(lfd, reinterpret_cast<sockaddr*>(&addr), &alen) == 0);
    REQUIRE(::listen(lfd, 4) == 0);
    const cc::endpoint ep{"127.0.0.1", ntohs(addr.sin_port)};

    std::string err;
    auto a = cc::connection::open(ep, 500ms, err, 400ms);
    REQUIRE_MESSAGE(a.has_value(), err);
    CHECK(a->io_timeout() == 400ms);           // out of open()

    cc::connection b(std::move(*a));
    CHECK(b.io_timeout() == 400ms);            // move construction

    auto c = cc::connection::open(ep, 500ms, err);   // io_timeout 0 = the connect bound
    REQUIRE_MESSAGE(c.has_value(), err);
    CHECK(c->io_timeout() == 500ms);
    *c = std::move(b);
    CHECK(c->io_timeout() == 400ms);           // move assignment

    ::close(lfd);
}

TEST_CASE("client: whole-reply deadline survives the move out of open()") {
    // In-test dribbling server on an ephemeral port (no daemon involved).
    const int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
    REQUIRE(lfd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = 0;                       // ephemeral: no fixture-port collision
    ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    REQUIRE(::bind(lfd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0);
    socklen_t alen = sizeof addr;
    REQUIRE(::getsockname(lfd, reinterpret_cast<sockaddr*>(&addr), &alen) == 0);
    REQUIRE(::listen(lfd, 1) == 0);

    std::atomic<bool> stop{false};
    std::thread server([&] {
        const int cfd = ::accept(lfd, nullptr, nullptr);
        if (cfd < 0) return;
        coastd::set_nosigpipe(cfd);                        // client WILL hang up on us
        char buf[512];
        (void)::recv(cfd, buf, sizeof buf, 0);             // swallow the request
        while (!stop.load()) {                             // dribbles until stopped
            if (::send(cfd, "x", 1, coastd::send_nosignal_flags) <= 0) break;
            std::this_thread::sleep_for(50ms);             // faster than SO_RCVTIMEO
        }
        ::close(cfd);
    });

    std::string err;
    auto conn = cc::connection::open({"127.0.0.1", ntohs(addr.sin_port)},
                                     500ms, err, 400ms);
    REQUIRE_MESSAGE(conn.has_value(), err);

    // The server dribbles until told to stop, so the call can only return through
    // the client's own whole-reply deadline: with timeout_ carried through the
    // move it does (~400 ms); with it dropped the 50 ms dribble defeats
    // SO_RCVTIMEO and the call never returns — named by the stall limit. (A server
    // thread starved past 400 ms trips SO_RCVTIMEO and hides the bug: a false
    // pass under load, never a false failure. The case above checks the move
    // itself deterministically; this one checks the deadline is ENFORCED.)
    std::atomic<bool> returned{false};
    bool              got_reply = true;
    std::thread caller([&] {
        std::string e;
        got_reply = conn->machine_info(e).has_value();
        returned.store(true);
    });
    CHECK(coast_test::await_condition([&] { return returned.load(); }));
    stop.store(true);   // a still-blocked caller now sees EOF and returns
    caller.join();
    CHECK_FALSE(got_reply);                                // reply never completes
    server.join();
    ::close(lfd);
}


// ---------------------------------------------------------------------------
// Reply-envelope robustness. nlohmann's value() defaults only on an ABSENT key;
// a PRESENT but wrong-typed one throws type_error.302, which is NOT a
// std::runtime_error. The client header promises errors come back in `err` and
// are never thrown, and the GUI panel calls these from its ImGui render thread —
// where an escaping exception unwinds through an in-progress frame. A
// version-skewed or hostile daemon is enough to trigger it, so the guard has to
// be structural (one checked accessor) rather than per-call-site.
//
// Data-driven rather than SUBCASE-driven on purpose: every server here MUST get a
// client, because its destructor joins a thread parked in accept(). A subcase
// layout that can enter the server branch without the calling branch deadlocks
// the suite instead of failing it.
// ---------------------------------------------------------------------------
namespace {

// Answers the first connection with one fixed line, then closes. The caller is
// required to connect — see above.
struct one_shot_server {
    int           lfd  = -1;
    std::uint16_t port = 0;
    std::thread   th;

    explicit one_shot_server(const std::string& line) {
        lfd = ::socket(AF_INET, SOCK_STREAM, 0);
        REQUIRE(lfd >= 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port   = 0;                   // ephemeral: no fixture-port collision
        ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        REQUIRE(::bind(lfd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0);
        socklen_t alen = sizeof addr;
        REQUIRE(::getsockname(lfd, reinterpret_cast<sockaddr*>(&addr), &alen) == 0);
        REQUIRE(::listen(lfd, 1) == 0);
        port = ntohs(addr.sin_port);

        th = std::thread([this, line] {
            const int cfd = ::accept(lfd, nullptr, nullptr);
            if (cfd < 0) return;
            coastd::set_nosigpipe(cfd);
            char buf[4096];
            (void)::recv(cfd, buf, sizeof buf, 0);      // swallow the request
            const std::string out = line + "\n";
            (void)::send(cfd, out.data(), out.size(), coastd::send_nosignal_flags);
            ::close(cfd);
        });
    }
    ~one_shot_server() {
        if (th.joinable()) th.join();
        if (lfd >= 0) ::close(lfd);
    }
};

} // namespace

TEST_CASE("client: a malformed reply envelope is returned rather than thrown") {
    // The shapes that used to escape: `ok` present but wrong-typed, `ok` absent,
    // `error` present but a number (reply_error's own value() call), a reply that
    // is not an object at all, and a wrong-typed `id` on the SUCCESS path.
    const char* const replies[] = {
        R"({"ok":"true"})",
        R"({"ok":1})",
        R"({"jobs":[]})",
        R"({"ok":false,"error":42})",
        R"(["ok"])",
        R"({"ok":true,"id":17})",
        R"({"ok":true,"jobs":[{"id":5}]})",   // wrong-typed field inside a job entry
    };

    for (const char* reply : replies) {
        CAPTURE(reply);
        const std::string r{reply};
        const bool positive = r.find("\"ok\":true") != std::string::npos;

        {   // list
            one_shot_server srv(r);
            std::string err;
            auto conn = cc::connection::open({"127.0.0.1", srv.port}, coast_test::default_stall, err,
                                             coast_test::default_stall);
            REQUIRE_MESSAGE(conn.has_value(), err);
            std::string e;
            std::optional<std::vector<coastd::proto::job_summary>> out;
            CHECK_NOTHROW(out = conn->list(e));
            if (!positive) {
                CHECK_FALSE(out.has_value());
                CHECK_FALSE(e.empty());        // a reason, never a silent failure
            }
        }
        {   // cancel — representative of the three bool-returning commands
            one_shot_server srv(r);
            std::string err;
            auto conn = cc::connection::open({"127.0.0.1", srv.port}, coast_test::default_stall, err,
                                             coast_test::default_stall);
            REQUIRE_MESSAGE(conn.has_value(), err);
            std::string e;
            bool ok = true;
            CHECK_NOTHROW(ok = conn->cancel("job-1", e));
            if (!positive) { CHECK_FALSE(ok); CHECK_FALSE(e.empty()); }
        }
        {   // machine_info
            one_shot_server srv(r);
            std::string err;
            auto conn = cc::connection::open({"127.0.0.1", srv.port}, coast_test::default_stall, err,
                                             coast_test::default_stall);
            REQUIRE_MESSAGE(conn.has_value(), err);
            std::string e;
            std::optional<coastd::proto::machine_summary> out;
            CHECK_NOTHROW(out = conn->machine_info(e));
            if (!positive) CHECK_FALSE(out.has_value());
        }
        {   // submit — also covers the wrong-typed `id` on an OK envelope
            one_shot_server srv(r);
            std::string err;
            auto conn = cc::connection::open({"127.0.0.1", srv.port}, coast_test::default_stall, err,
                                             coast_test::default_stall);
            REQUIRE_MESSAGE(conn.has_value(), err);
            cc::submit_args a;
            a.ref = "main";
            a.config_text = "{}";              // non-empty, else submit short-circuits
            a.config_name = "c.json";
            std::string e;
            std::optional<std::string> id;
            CHECK_NOTHROW(id = conn->submit(a, e));
            if (!positive) CHECK_FALSE(id.has_value());
            // {"ok":true,"id":17} must yield an EMPTY id, not a thrown type_error.
            if (r.find("\"id\":17") != std::string::npos)
                CHECK((!id.has_value() || id->empty()));
        }
    }
}
