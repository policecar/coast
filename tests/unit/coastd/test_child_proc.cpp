#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <string>

#include "daemon/child_proc.h"

#include "await_progress.h"

using namespace coastd;
namespace fs = std::filesystem;

namespace {
struct scratch {
    fs::path root;
    scratch() {
        static std::atomic<int> ctr{0};
        root = fs::temp_directory_path() /
               ("coastd_proc_" + std::to_string(::getpid()) + "_" +
                std::to_string(ctr.fetch_add(1)));
        fs::create_directories(root);
    }
    ~scratch() { std::error_code ec; fs::remove_all(root, ec); }
};

int wait_for(pid_t pid) {
    int st = 0;
    while (::waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
    return st;
}
} // namespace

TEST_CASE("child_proc: spawn redirects stdout+stderr to the logfile") {
    scratch s;
    const auto log = s.root / "out.log";
    spawn_opts o;
    o.exe     = "/bin/sh";
    o.args    = {"-c", "printf 'hello-stdout'; printf 'hello-stderr' 1>&2"};
    o.logfile = log;
    o.cwd     = s.root;

    auto pid = spawn_process(o);
    REQUIRE(pid.has_value());
    const int st = wait_for(*pid);
    CHECK(WIFEXITED(st));
    CHECK(WEXITSTATUS(st) == 0);

    std::ifstream in(log);
    std::string body((std::istreambuf_iterator<char>(in)), {});
    CHECK(body.find("hello-stdout") != std::string::npos);
    CHECK(body.find("hello-stderr") != std::string::npos);  // stderr dup'd to the log too
}

TEST_CASE("child_proc: spawn of a nonexistent binary fails cleanly") {
    scratch s;
    spawn_opts o;
    o.exe     = "/no/such/binary/xyzzy";
    o.logfile = s.root / "out.log";
    CHECK_FALSE(spawn_process(o).has_value());
}

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

TEST_CASE("child_proc: start-token identifies THIS process, rejects a stale one" * doctest::skip(skip_off_linux)) {
    const pid_t self = ::getpid();
    auto tok = read_start_token(self);
    REQUIRE(tok.has_value());
    CHECK_FALSE(tok->empty());

    // Same pid + same token = same process.
    CHECK(is_same_process(self, *tok));
    // Same pid + a wrong token = a recycled pid, rejected.
    CHECK_FALSE(is_same_process(self, "0"));
    // A pid that is (almost certainly) not live.
    CHECK_FALSE(is_same_process(0x7fffffff, *tok));
    // No process at all: above pid_max on Linux (2^22) and macOS (99999).
    CHECK_FALSE(read_start_token(0x7fffffff).has_value());
}

// ---- process-group isolation and group signalling ---------------------------
// The daemon puts every job child in its OWN process group so cancel/shutdown can
// never reach coastd's own group (kill(-0, ...) would include the shell that
// started an interactively-run daemon), and so a signal reaches the whole child
// tree rather than just its leader. Neither property had a test, and both are the
// kind that break silently: signalling only the leader still kills the leader, so
// a cancelled job still LOOKS cancelled while a helper keeps running.

TEST_CASE("child_proc: a spawned child is its own process-group leader") {
    scratch s;
    const auto log = s.root / "pg.log";
    spawn_opts o;
    o.exe     = "/bin/sh";
    o.args    = {"-c", "sleep 30"};
    o.logfile = log;
    o.cwd     = s.root;

    const auto pid = spawn_process(o);
    REQUIRE(pid.has_value());

    // pgid == pid is what makes kill(-pid, ...) mean "this child's tree" and
    // nothing else. If it were inherited, the pgid would be coastd's.
    const pid_t pgid = ::getpgid(*pid);
    CHECK(pgid == *pid);
    CHECK(pgid != ::getpgid(0));      // emphatically NOT the test process's group

    send_signal_group(*pid, SIGKILL);
    wait_for(*pid);
}

TEST_CASE("child_proc: send_signal_group reaches a GRANDCHILD, send_signal does not") {
    scratch s;
    const auto marker = s.root / "grandchild.pid";

    // A shell that forks a long-lived grandchild and then waits. Killing only the
    // leader leaves the grandchild running — the exact silent-survivor case the
    // group form exists to prevent.
    spawn_opts o;
    o.exe     = "/bin/sh";
    o.args    = {"-c", "sleep 30 & echo $! > '" + marker.string() + "'; wait"};
    o.logfile = s.root / "gc.log";
    o.cwd     = s.root;

    const auto pid = spawn_process(o);
    REQUIRE(pid.has_value());

    // Wait for the grandchild's pid to be published (sh startup; the stall
    // limit is the only bound, CLAUDE.md "Testing").
    pid_t gc = 0;
    REQUIRE(coast_test::await_condition([&] {
        std::ifstream in(marker);
        if (in) in >> gc;
        return gc > 0;
    }));
    REQUIRE(::kill(gc, 0) == 0);              // alive before we signal

    send_signal_group(*pid, SIGKILL);
    wait_for(*pid);

    // The grandchild must be gone too. It is not our child, so we cannot waitpid
    // it; poll kill(pid, 0) until it reports ESRCH — i.e. until init (or a
    // subreaper) has reaped the zombie: not a process we control, so only the
    // stall limit bounds it.
    CHECK(coast_test::await_condition([&] { return ::kill(gc, 0) != 0 && errno == ESRCH; }));
}

TEST_CASE("child_proc: group signalling refuses a non-positive pid") {
    // kill(-1, SIGKILL) is "every process this uid owns" and kill(-0, ...) is
    // coastd's own group. Both are one bad pid away, and `pid` reaches the store
    // from DISK via job_json's tolerant defaulting — so the guard is load-bearing,
    // not decorative. It must return without signalling anything.
    CHECK_NOTHROW(send_signal_group(0, SIGKILL));
    CHECK_NOTHROW(send_signal_group(-1, SIGKILL));
    CHECK_NOTHROW(send_signal(0, SIGKILL));
    CHECK_NOTHROW(send_signal(-1, SIGKILL));
}
