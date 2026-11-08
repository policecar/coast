#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <functional>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "daemon/atomic_write.h"

// atomic_write is the durability primitive under EVERY job-store write, and it
// had no direct test — only its aftermath (sweep_stale_tmp) was covered. These
// cases pin the properties the rest of the daemon assumes: the target is replaced
// whole, a failure leaves no scratch behind, and concurrent writers to one path
// cannot interleave into a shared scratch file.

using namespace coastd;
namespace fs = std::filesystem;

namespace {

struct scratch {
    fs::path root;
    scratch() {
        static std::atomic<unsigned> seq{0};
        root = fs::temp_directory_path() /
               ("coastd_aw_" + std::to_string(::getpid()) + "_" + std::to_string(seq.fetch_add(1)));
        fs::create_directories(root);
    }
    ~scratch() { std::error_code ec; fs::remove_all(root, ec); }
};

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Any leftover scratch file in `dir` — atomic_write's tmp names end in ".tmp".
std::vector<fs::path> tmp_files(const fs::path& dir) {
    std::vector<fs::path> out;
    std::error_code ec;
    for (const auto& de : fs::directory_iterator(dir, ec))
        if (de.path().extension() == ".tmp") out.push_back(de.path());
    return out;
}

} // namespace

TEST_CASE("atomic_write: creates the target and leaves no tmp behind") {
    scratch s;
    const auto target = s.root / "a.json";

    atomic_write(target, "hello");

    CHECK(fs::exists(target));
    CHECK(read_file(target) == "hello");
    CHECK(tmp_files(s.root).empty());
}

TEST_CASE("atomic_write: replaces existing content wholesale, never appends") {
    scratch s;
    const auto target = s.root / "a.json";

    atomic_write(target, "a-much-longer-first-value");
    atomic_write(target, "short");

    // The rename replaces the inode; a shorter second write must not leave the
    // first value's tail behind (which an in-place rewrite would).
    CHECK(read_file(target) == "short");
    CHECK(fs::file_size(target) == 5);
    CHECK(tmp_files(s.root).empty());
}

TEST_CASE("atomic_write: an empty payload is a valid write") {
    scratch s;
    const auto target = s.root / "empty.json";
    atomic_write(target, "");
    CHECK(fs::exists(target));
    CHECK(fs::file_size(target) == 0);
}

TEST_CASE("atomic_write: throws and leaves nothing behind when the dir does not exist") {
    scratch s;
    const auto target = s.root / "missing_dir" / "a.json";

    CHECK_THROWS_AS(atomic_write(target, "x"), std::system_error);
    CHECK_FALSE(fs::exists(target));
    CHECK(tmp_files(s.root).empty());   // no scratch leaked into the parent either
}

TEST_CASE("atomic_write: concurrent writers to ONE path never interleave") {
    scratch s;
    const auto target = s.root / "contended.json";

    // Distinct, self-identifying payloads of different lengths. With a FIXED
    // "<path>.tmp" scratch name (as this used to use) the writers share one file
    // descriptor-visible path: one truncates while the other is mid-write, and the
    // rename publishes a spliced result. The reader below then sees a value that
    // is neither payload — which is exactly what atomic_write exists to prevent.
    const std::string a(4096, 'a');
    const std::string b(8192, 'b');

    std::atomic<bool> go{false};
    std::atomic<int>  failures{0};
    std::vector<std::thread> ts;
    for (int i = 0; i < 8; ++i) {
        ts.emplace_back([&, i] {
            while (!go.load()) std::this_thread::yield();
            for (int n = 0; n < 40; ++n) {
                try {
                    atomic_write(target, (i % 2 == 0) ? a : b);
                } catch (const std::exception&) {
                    ++failures;
                }
            }
        });
    }
    go.store(true);
    for (auto& t : ts) t.join();

    CHECK(failures.load() == 0);

    // Whatever landed last must be one of the two payloads, entire.
    const std::string got = read_file(target);
    CHECK((got == a || got == b));
    CHECK(tmp_files(s.root).empty());   // every writer cleaned up its own scratch
}

// The scratch path is per-writer (pid+tid) but not secret — anyone who can list the
// state root can predict it. Without O_NOFOLLOW, pre-creating it as a symlink turns
// this O_CREAT|O_TRUNC into an arbitrary-file truncate-and-overwrite as the daemon.
// (Measured: with the flag removed, the victim below really is overwritten.) It
// presupposes write access to the daemon's own state root, so this is hardening, not
// a live hole — but an unverified security flag is worth little, so the property is
// pinned here rather than assumed from the open() call.
//
// The SECOND case is the one that matters for availability: refusing a symlink is not
// enough on its own, because sweep_stale_tmp reclaims only regular files
// (is_regular_file() follows the link, so a dangling link is invisible to it). A bare
// refusal would leave the link in place and make that job file permanently
// unwritable — trading an overwrite bug for a denial of service on the same
// precondition. atomic_write unlinks and retries once, so both cases must SUCCEED.
TEST_CASE("atomic_write: reclaims a symlinked scratch path without following it") {
    scratch s;
    const auto target = s.root / "job.json";

    // Same pid+tid formula as atomic_write, so we plant the link where it will look.
    const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    const std::filesystem::path tmp =
        target.string() + ".p" + std::to_string(::getpid()) + "t" + std::to_string(tid) + ".tmp";

    SUBCASE("symlink pointing at a real file: the victim is NOT written") {
        const auto victim = s.root / "VICTIM";
        { std::ofstream(victim) << "precious\n"; }
        std::error_code ec;
        std::filesystem::create_symlink(victim, tmp, ec);
        REQUIRE_FALSE(ec);                    // the test is meaningless if this fails

        CHECK_NOTHROW(atomic_write(target, "payload"));

        CHECK(read_file(victim) == "precious\n");   // the actual security property
        CHECK(read_file(target) == "payload");      // and the write still completed
        CHECK(tmp_files(s.root).empty());           // scratch cleaned up
    }

    SUBCASE("DANGLING symlink: no permanent failure, and no file at the link target") {
        const auto nowhere = s.root / "NOWHERE";
        std::error_code ec;
        std::filesystem::create_symlink(nowhere, tmp, ec);
        REQUIRE_FALSE(ec);

        CHECK_NOTHROW(atomic_write(target, "payload"));

        CHECK(read_file(target) == "payload");
        // Following the dangling link would have CREATED this; O_NOFOLLOW means the
        // retry created the scratch file itself instead.
        CHECK_FALSE(std::filesystem::exists(nowhere));
        CHECK(tmp_files(s.root).empty());
    }
}
