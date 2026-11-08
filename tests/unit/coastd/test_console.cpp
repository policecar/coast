// -----------------------------------------------------------------------------
// coastd console façade test: launch the real coastd binary and drive it through
// coast::coastd::session — the REPL-facing façade that wraps coastd_client. This
// validates the façade's proto->plain-struct mapping and its throw-on-error
// contract. (The clang-repl JIT-invocation path itself needs an LLVM console
// build and is out of scope here — this test exercises the façade LOGIC.)
// -----------------------------------------------------------------------------
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <spawn.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include "coastd_console/coastd_console_api.h"
#include "test_daemon.h"

using namespace std::chrono_literals;
namespace fs = std::filesystem;
namespace cd = coast::coastd;

extern char** environ;

#ifndef COASTD_BIN
#define COASTD_BIN "coastd"
#endif

namespace {

fs::path make_temp_dir() {
    const char* base = std::getenv("TMPDIR");
    std::string s = (fs::path(base && *base ? base : "/tmp") / "coastd_console_XXXXXX").string();
    REQUIRE(::mkdtemp(s.data()) != nullptr);
    return fs::path(s);
}

void run(const std::string& c) { REQUIRE(std::system(c.c_str()) == 0); }

// OS-assigned ports; the daemon publishes them in coastd.ports (test_daemon.h).
// Its full log goes to `log`, so a failure can show what the daemon did.
pid_t spawn_coastd(const fs::path& data_root, const fs::path& repo, const fs::path& log,
                   std::uint16_t tcp_port = 0) {
    const std::string tcp = std::to_string(tcp_port);
    const char* argv[] = {COASTD_BIN, "--data-root", data_root.c_str(), "--repo", repo.c_str(),
                          "--tcp-port", tcp.c_str(), "--udp-port", "0", nullptr};
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, STDOUT_FILENO, log.c_str(),
                                     O_CREAT | O_WRONLY | O_APPEND, 0644);
    posix_spawn_file_actions_adddup2(&fa, STDOUT_FILENO, STDERR_FILENO);
    pid_t pid = 0;
    const int rc = ::posix_spawn(&pid, COASTD_BIN, &fa, nullptr,
                                 const_cast<char* const*>(argv), environ);
    posix_spawn_file_actions_destroy(&fa);
    REQUIRE(rc == 0);
    return pid;
}

} // namespace

// ---------------------------------------------------------------------------
// PLATFORM GATE. Every case in this file drives a REAL coastd process, and the
// daemon refuses to start where start_token_supported() is false (main.cpp):
// without a recycle-safe start token its duplicate-run guard is inoperative, and
// shipping that silent-corruption path was declined (ADR-16). So off Linux these
// cannot pass, and they skip rather than hard-fail — macOS is a first-class dev
// platform here and a Mac dev running the documented `ctest` must not be handed
// red that means "wrong OS". Gated on the seam, not __linux__, so porting
// read_start_token() re-enables them automatically.
// ---------------------------------------------------------------------------
// NOTE: spelled as __linux__ rather than coastd::start_token_supported(), unlike
// the sibling coastd tests. This TU deliberately cannot see coastd/include — it
// exercises the console façade, whose whole point is being a lean parse boundary
// that does not drag the daemon's headers in (see the PRIVATE link in
// CMakeLists.txt). Reaching for the seam here would breach the boundary the test
// exists to protect. Keep this list in sync with start_token_supported().
#if defined(__linux__) || defined(__APPLE__)
constexpr bool skip_off_linux = false;
#else
constexpr bool skip_off_linux = true;
#endif

TEST_CASE("coastd console façade drives the daemon" * doctest::skip(skip_off_linux)) {
    const fs::path root = make_temp_dir();
    const fs::path data_root = root / "data", repo = root / "repo";
    fs::create_directories(data_root);
    fs::create_directories(repo);
    run("git -C '" + repo.string() + "' init -q");
    run("git -C '" + repo.string() + "' -c user.email=t@t -c user.name=t commit -q --allow-empty -m init");
    { std::ofstream(data_root / "good.json") << "{\"seed\":1}\n"; }

    // A pre-placed "built" worktree for HEAD (no build runs), whose coast runs
    // until the test releases it, plus one slot: a first "busy" job occupies the
    // slot, so the job under test stays PENDING by construction — not for as long
    // as a failing build's retry backoff happens to last.
    run("git -C '" + repo.string() + "' rev-parse HEAD > '" + (root / "sha.txt").string() + "'");
    std::string sha;
    { std::ifstream in(root / "sha.txt"); std::getline(in, sha); }
    REQUIRE(sha.size() == 40);
    const fs::path release = root / "release_busy";
    {
        const fs::path wt = data_root / "worktrees" / sha;
        fs::create_directories(wt / "build");
        { std::ofstream f(wt / "build" / "coast");
          f << "#!/usr/bin/env bash\n"
               "while [ ! -e '" << release.string() << "' ]; do sleep 0.2; done\n"; }
        fs::permissions(wt / "build" / "coast",
                        fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec);
        { std::ofstream(wt / ".coastd-ready") << "ready"; }
    }
    { std::ofstream(data_root / "coastd.json") << "{\"slots\":1}\n"; }

    // connect() to a port nobody listens on must throw. Bound but never listened
    // on, so the port is refused for as long as `closed` holds it.
    {
        const int closed = ::socket(AF_INET, SOCK_STREAM, 0);
        REQUIRE(closed >= 0);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
        socklen_t len = sizeof a;
        REQUIRE(::bind(closed, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0);
        REQUIRE(::getsockname(closed, reinterpret_cast<sockaddr*>(&a), &len) == 0);
        CHECK_THROWS_AS((void)cd::session::connect("localhost", ntohs(a.sin_port), 200),
                        std::runtime_error);
        ::close(closed);
    }

    const fs::path log = root / "daemon.log";
    coastd_test::daemon_guard daemon(spawn_coastd(data_root, repo, log));
    const auto ports = coastd_test::wait_for_ports(data_root, daemon.pid());
    REQUIRE_MESSAGE(ports.has_value(), "daemon exited or never published its ports\n"
                                           << coastd_test::file_tail(log, 30));
    const std::uint16_t tcp_port = ports->tcp;
    // Lazy: built only if an assertion below fails or an exception unwinds the
    // test — and it is declared after `daemon`, so the daemon is still unreaped
    // (alive, or a zombie whose exit has_exited() can see) when it runs.
    INFO(coastd_test::daemon_report(daemon.pid(), tcp_port, log));

    // The ports file is written only once the listener accepts, so this connects
    // first time.
    std::optional<cd::session> s;
    // 60 s for connect and each reply: the stall limit for a daemon we run
    // ourselves, not a performance budget (CLAUDE.md "Testing").
    constexpr int k_io_ms = 60'000;
    s.emplace(cd::session::connect("localhost", tcp_port, k_io_ms, k_io_ms));

    // machine_info mapping — incl. the advertised command port.
    auto host = s->info();
    CHECK(!host.hostname.empty());
    CHECK(host.os == std::string(coastd_test::expected_os));
    CHECK(!host.coastd_version.empty());
    CHECK(host.slots >= 1);
    CHECK(host.port == tcp_port);

    // connect(host_info): the discover()->connect path. Build the endpoint the way
    // discover() would (address + advertised port) and connect through it.
    {
        cd::host_info d;
        d.address = "127.0.0.1";
        d.port    = tcp_port;
        auto s2 = cd::session::connect(d, k_io_ms, k_io_ms);   // throws on failure
        CHECK(s2.info().port == tcp_port);
    }

    // Empty queue.
    CHECK(s->jobs().empty());

    // The busy job takes the one slot and keeps it until released.
    const std::string busy = s->submit("HEAD", (data_root / "good.json").string(), "busy", "", 10, 0);
    REQUIRE(coast_test::await_condition([&] {
        for (const auto& j : s->jobs())
            if (j.id == busy && j.status == "running") return true;
        return false;
    }));

    // submit -> id; list mapping preserves fields.
    const std::string id = s->submit("HEAD", (data_root / "good.json").string(), "gate", "note", 10, 0);
    CHECK_FALSE(id.empty());
    auto jobs = s->jobs();
    REQUIRE(jobs.size() == 2);
    std::optional<cd::job_info> gate;
    for (const auto& j : jobs) if (j.id == id) gate = j;
    REQUIRE(gate.has_value());
    CHECK(gate->ref == "HEAD");
    CHECK(gate->bucket == "active");
    CHECK(gate->status != "running");   // pending behind the busy job
    CHECK(gate->user_message == "note");

    // reorder + cancel; errors throw.
    s->reorder(id, 0);
    s->cancel(id);
    CHECK_THROWS_AS(s->cancel("no-such-job"), std::runtime_error);
    CHECK_THROWS_AS(s->reorder("no-such-job", 0), std::runtime_error);

    // Path-unsafe submit throws with the daemon's reason.
    { std::ofstream(data_root / "bad.json")
          << "{\"snapshot_dir\":\"" << (data_root / "state" / "x").string() << "\"}\n"; }
    CHECK_THROWS_AS((void)s->submit("HEAD", (data_root / "bad.json").string()), std::runtime_error);

    // Object submit — the simc workflow: load a base config, EDIT it in memory,
    // then submit the run_config object (not a path). The daemon archives the
    // submitted bytes verbatim at submit time, so read them back and prove the
    // EDITED value arrived. Starting from a real config() object and asserting the
    // mutated value (7), not the base file's (1), is what makes this fail if submit
    // ever sent a path or the unedited base.
    {
        std::ofstream(data_root / "base.json") << "{\"seed\":1}\n";
    }
    auto cfg = cd::config((data_root / "base.json").string());
    cfg.set("/seed", 7).set("/note_field", "edited");
    const std::string oid = s->submit("HEAD", cfg, "objsub", "via-object", 10, 0);
    CHECK_FALSE(oid.empty());
    {
        // runs_root defaults to <data-root>/runs; config.submitted.json is written
        // synchronously inside submit, so it exists the moment submit() returns.
        const fs::path submitted = data_root / "runs" / oid / "config.submitted.json";
        REQUIRE(fs::exists(submitted));
        std::ifstream in(submitted);
        std::ostringstream ss;
        ss << in.rdbuf();
        const std::string body = ss.str();
        CHECK(body.find("\"seed\": 7") != std::string::npos);   // dump(2): "seed": 7
        CHECK(body.find("note_field") != std::string::npos);    // the added field arrived
        CHECK(body.find("\"seed\": 1") == std::string::npos);    // base value did NOT
    }
    // The edited job also shows up in the queue with its mnemonic/note mapped through.
    {
        bool found = false;
        for (const auto& j : s->jobs())
            if (j.id == oid) { CHECK(j.user_message == "via-object"); found = true; }
        CHECK(found);
    }
    s->cancel(oid);
    s->cancel(busy);                      // and let the busy child go
    { std::ofstream(release) << "go"; }

    s.reset();
    const auto t_kill = std::chrono::steady_clock::now();
    const bool reaped = daemon.stop();
    MESSAGE("daemon shutdown took " << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_kill).count() << " ms");
    CHECK(reaped);   // a daemon that needs SIGKILL failed its shutdown contract

    std::error_code ec;
    fs::remove_all(root, ec);
}

// Regression: "localhost" must reach the daemon even when an IPv6-only service
// holds the daemon's port number on [::]. The daemon listens on IPv4 only, so the
// two coexist; a client that resolved "localhost" to ::1 first connected to the
// OTHER service and reported "no reply" from a healthy daemon. That is what failed
// this suite in CI once: rpc-statd and the NFS lock manager sit on IPv6-only
// wildcard sockets with kernel-assigned ports, and a kernel-assigned daemon port can
// land on the same number. Here the trap is set on purpose, so the case is
// deterministic instead of a ~1e-4 draw.
TEST_CASE("coastd console façade: localhost is not captured by an IPv6-only listener"
          * doctest::skip(skip_off_linux)) {
    // The trap: listening, never accepting. A misrouted client connects (the kernel
    // completes the handshake) and then waits in vain for a reply.
    const int trap = ::socket(AF_INET6, SOCK_STREAM, 0);
    if (trap < 0) {
        MESSAGE("no IPv6 on this host — the trap cannot be built; nothing to test");
        return;
    }
    int one = 1;
    REQUIRE(::setsockopt(trap, IPPROTO_IPV6, IPV6_V6ONLY, &one, sizeof one) == 0);
    sockaddr_in6 a6{};
    a6.sin6_family = AF_INET6;
    a6.sin6_addr   = in6addr_any;
    socklen_t len  = sizeof a6;
    REQUIRE(::bind(trap, reinterpret_cast<sockaddr*>(&a6), sizeof a6) == 0);
    REQUIRE(::listen(trap, 4) == 0);
    REQUIRE(::getsockname(trap, reinterpret_cast<sockaddr*>(&a6), &len) == 0);
    const std::uint16_t port = ntohs(a6.sin6_port);

    const fs::path root = make_temp_dir();
    const fs::path data_root = root / "data", repo = root / "repo", log = root / "daemon.log";
    fs::create_directories(data_root);
    fs::create_directories(repo);
    run("git -C '" + repo.string() + "' init -q");
    run("git -C '" + repo.string() + "' -c user.email=t@t -c user.name=t commit -q --allow-empty -m init");

    // Same number, IPv4: the bind must succeed next to the IPv6-only trap.
    coastd_test::daemon_guard daemon(spawn_coastd(data_root, repo, log, port));
    const auto ports = coastd_test::wait_for_ports(data_root, daemon.pid());
    REQUIRE_MESSAGE(ports.has_value(), "daemon exited or never published its ports\n"
                                           << coastd_test::file_tail(log, 30));
    REQUIRE(ports->tcp == port);
    INFO(coastd_test::daemon_report(daemon.pid(), port, log));

    // Reached the daemon, not the trap: a reply, and the right one. (Misrouted,
    // info() throws "no reply" once the I/O bound — the stall limit — expires.)
    auto s = cd::session::connect("localhost", port, 60'000, 60'000);
    const auto host = s.info();
    CHECK(host.port == port);
    CHECK(!host.coastd_version.empty());

    CHECK(daemon.stop());
    ::close(trap);
    std::error_code ec;
    fs::remove_all(root, ec);
}
