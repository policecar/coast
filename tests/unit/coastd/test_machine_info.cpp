#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <string>

#include "daemon/child_cmd.h"   // dataset_link_target
#include "daemon/machine_info.h"

using namespace coastd;
namespace fs = std::filesystem;

// machine_info is the daemon's self-description. Everything in it is a straight
// copy of daemon_config except data_link_ok, which answers a question the client
// cannot answer for itself: will a config's relative "data/…" path resolve over
// there? Three states, and the middle one is the whole point — "configured" is
// not the same as "works".

namespace {
struct scratch {
    fs::path root;
    scratch() {
        static std::atomic<int> ctr{0};
        root = fs::temp_directory_path() /
               ("coastd_mi_" + std::to_string(::getpid()) + "_" +
                std::to_string(ctr.fetch_add(1)));
        fs::create_directories(root);
    }
    ~scratch() { std::error_code ec; fs::remove_all(root, ec); }
};
} // namespace

TEST_CASE("machine_info: data_link_ok reports whether relative data paths resolve") {
    scratch s;
    daemon_config cfg;
    cfg.runs_root = s.root / "runs";
    cfg.repo_root = s.root / "repo";

    SUBCASE("dataset_base unset -> false") {
        const auto m = make_machine_summary(cfg, "test");
        CHECK_FALSE(m.data_link_ok);
    }

    SUBCASE("dataset_base set but the target does not exist -> false") {
        // The daemon would create a DANGLING `data` symlink, so the job still dies
        // at dataset load. Answering yes on configuration alone would advertise a
        // promise the daemon cannot keep — the exact failure the flag exists to
        // warn about, reported as success.
        cfg.dataset_base = s.root / "not_mounted_yet";
        const auto m = make_machine_summary(cfg, "test");
        CHECK_FALSE(m.data_link_ok);
    }

    SUBCASE("dataset_base set and present -> true") {
        cfg.dataset_base = s.root / "datasets";
        fs::create_directories(cfg.dataset_base);
        const auto m = make_machine_summary(cfg, "test");
        CHECK(m.data_link_ok);
    }

    SUBCASE("it tracks the filesystem, not the boot-time answer") {
        // Deliberately not cached: the rest of the description is static, but this
        // bit flips when the operator mounts the share, and a stale "no" would
        // outlive the fix for the daemon's whole lifetime.
        cfg.dataset_base = s.root / "late_mount";
        CHECK_FALSE(make_machine_summary(cfg, "test").data_link_ok);
        fs::create_directories(cfg.dataset_base);
        CHECK(make_machine_summary(cfg, "test").data_link_ok);
    }
}

TEST_CASE("machine_info: the static fields are the config, verbatim") {
    scratch s;
    daemon_config cfg;
    cfg.runs_root = s.root / "runs";
    cfg.repo_root = s.root / "repo";
    cfg.slots     = 3;
    cfg.tcp_port  = 51987;
    cfg.dataset_base = s.root / "not_even_present";   // reported whether it exists or not

    const auto m = make_machine_summary(cfg, "9.9.9-test");
    CHECK(m.coastd_version == "9.9.9-test");
    CHECK(m.slots     == 3);
    CHECK(m.tcp_port  == 51987);          // a discovered daemon must be connectable
    CHECK(m.runs_root == cfg.runs_root.string());
    CHECK(m.repo_root == cfg.repo_root.string());
    CHECK(m.dataset_base == cfg.dataset_base.string());
    CHECK(!m.hostname.empty());
    CHECK(m.address.empty());             // client-populated only, never sent
}

TEST_CASE("machine_info: dataset_base is reported as the data link's target") {
    // The link site makes a relative dataset_base absolute (child_cmd.cpp); a client
    // compares a run dir's link with this field, so both must be the same string.
    daemon_config cfg;
    cfg.dataset_base = "rel/datasets";
    const auto m = make_machine_summary(cfg, "test");
    CHECK(m.dataset_base == (fs::current_path() / "rel/datasets").string());
    CHECK(m.dataset_base == dataset_link_target(cfg.dataset_base).string());
    cfg.dataset_base.clear();
    CHECK(make_machine_summary(cfg, "test").dataset_base.empty());
}
