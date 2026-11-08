#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "daemon/config_load.h"

using namespace coastd;
namespace fs = std::filesystem;

namespace {

// A unique temp config file, written from the given JSON text and removed on exit.
struct temp_cfg {
    fs::path path;
    explicit temp_cfg(const std::string& tag, const std::string& json) {
        path = fs::temp_directory_path() /
               ("coastd_cfgload_" + tag + "_" + std::to_string(::getpid()) + ".json");
        std::ofstream(path) << json;
    }
    ~temp_cfg() { std::error_code ec; fs::remove(path, ec); }
};

// Seed a config the way main() does before loading: the four roots derived from a
// data_root, repo_root defaulted to a stand-in for CWD.
daemon_config seeded(const fs::path& data_root) {
    daemon_config cfg;
    cfg.state_root     = data_root / "state";
    cfg.runs_root      = data_root / "runs";
    cfg.worktrees_root = data_root / "worktrees";
    cfg.repo_root      = "/default/cwd/repo";
    return cfg;
}

} // namespace

TEST_CASE("config_load: absent file keys leave the seeded defaults untouched") {
    const fs::path dr = "/srv/coastd";
    temp_cfg f("empty", "{}");
    daemon_config cfg = seeded(dr);
    std::string err;
    REQUIRE(load_config_file(f.path, dr, cfg, err));
    CHECK(cfg.state_root     == dr / "state");
    CHECK(cfg.runs_root      == dr / "runs");
    CHECK(cfg.worktrees_root == dr / "worktrees");
    CHECK(cfg.repo_root      == fs::path("/default/cwd/repo"));
}

TEST_CASE("config_load: absolute root keys override the derived defaults") {
    const fs::path dr = "/srv/coastd";
    temp_cfg f("roots", R"({
        "state_root":     "/reliable/coastd-state",
        "runs_root":      "/scratch/coastd-runs",
        "worktrees_root": "/scratch/coastd-wt",
        "repo_root":      "/opt/coast"
    })");
    daemon_config cfg = seeded(dr);
    std::string err;
    REQUIRE(load_config_file(f.path, dr, cfg, err));
    CHECK(cfg.state_root     == fs::path("/reliable/coastd-state"));
    CHECK(cfg.runs_root      == fs::path("/scratch/coastd-runs"));
    CHECK(cfg.worktrees_root == fs::path("/scratch/coastd-wt"));
    CHECK(cfg.repo_root      == fs::path("/opt/coast"));
}

TEST_CASE("config_load: the <data-root> token expands to the data root") {
    const fs::path dr = "/var/lib/coastd";
    temp_cfg f("token", R"({
        "runs_root":      "<data-root>/runs",
        "state_root":     "<data-root>/state",
        "worktrees_root": "<data-root>"
    })");
    daemon_config cfg = seeded(dr);
    std::string err;
    REQUIRE(load_config_file(f.path, dr, cfg, err));
    CHECK(cfg.runs_root      == dr / "runs");
    CHECK(cfg.state_root     == dr / "state");
    CHECK(cfg.worktrees_root == dr);              // token alone, no suffix
}

TEST_CASE("config_load: only the named roots move; the rest stay data-root-derived") {
    const fs::path dr = "/srv/coastd";
    temp_cfg f("partial", R"({ "runs_root": "/big/runs" })");
    daemon_config cfg = seeded(dr);
    std::string err;
    REQUIRE(load_config_file(f.path, dr, cfg, err));
    CHECK(cfg.runs_root      == fs::path("/big/runs"));   // moved
    CHECK(cfg.state_root     == dr / "state");            // untouched
    CHECK(cfg.worktrees_root == dr / "worktrees");        // untouched
}

TEST_CASE("config_load: dataset_base is a path key (default empty, token-aware)") {
    const fs::path dr = "/var/lib/coastd";
    SUBCASE("absent key leaves dataset_base empty (feature off)") {
        temp_cfg f("ds_absent", "{}");
        daemon_config cfg = seeded(dr);
        std::string err;
        REQUIRE(load_config_file(f.path, dr, cfg, err));
        CHECK(cfg.dataset_base.empty());
    }
    SUBCASE("absolute value is taken verbatim") {
        temp_cfg f("ds_abs", R"({ "dataset_base": "/srv/shared/data" })");
        daemon_config cfg = seeded(dr);
        std::string err;
        REQUIRE(load_config_file(f.path, dr, cfg, err));
        CHECK(cfg.dataset_base == fs::path("/srv/shared/data"));
    }
    SUBCASE("the <data-root> token expands") {
        temp_cfg f("ds_token", R"({ "dataset_base": "<data-root>/data" })");
        daemon_config cfg = seeded(dr);
        std::string err;
        REQUIRE(load_config_file(f.path, dr, cfg, err));
        CHECK(cfg.dataset_base == dr / "data");
    }
    SUBCASE("a wrong-typed value is ignored, leaving the default") {
        temp_cfg f("ds_badtype", R"({ "dataset_base": 42 })");
        daemon_config cfg = seeded(dr);
        std::string err;
        REQUIRE(load_config_file(f.path, dr, cfg, err));   // still loads
        CHECK(cfg.dataset_base.empty());
    }
}

TEST_CASE("config_load: relaunch backoff keys are parsed and range-checked") {
    const fs::path dr = "/srv/coastd";
    SUBCASE("valid values apply") {
        temp_cfg f("backoff_ok", R"({
            "relaunch_backoff_base_seconds": 5,
            "relaunch_backoff_max_seconds":  300
        })");
        daemon_config cfg = seeded(dr);
        std::string err;
        REQUIRE(load_config_file(f.path, dr, cfg, err));
        CHECK(cfg.relaunch_backoff_base == std::chrono::seconds(5));
        CHECK(cfg.relaunch_backoff_max  == std::chrono::seconds(300));
    }
    SUBCASE("out-of-range value is ignored, default kept") {
        const daemon_config def;   // compiled-in defaults
        temp_cfg f("backoff_bad", R"({ "relaunch_backoff_base_seconds": 0 })");  // below lo=1
        daemon_config cfg = seeded(dr);
        std::string err;
        REQUIRE(load_config_file(f.path, dr, cfg, err));
        CHECK(cfg.relaunch_backoff_base == def.relaunch_backoff_base);
    }
}

TEST_CASE("config_load: a wrong-typed root warns and keeps the default (never fatal)") {
    const fs::path dr = "/srv/coastd";
    temp_cfg f("badtype", R"({ "runs_root": 12345 })");   // number, not string
    daemon_config cfg = seeded(dr);
    std::string err;
    REQUIRE(load_config_file(f.path, dr, cfg, err));       // still loads
    CHECK(cfg.runs_root == dr / "runs");                   // default retained
}

TEST_CASE("config_load: an empty root string is skipped, not applied") {
    const fs::path dr = "/srv/coastd";
    temp_cfg f("emptyroot", R"({ "state_root": "" })");
    daemon_config cfg = seeded(dr);
    std::string err;
    REQUIRE(load_config_file(f.path, dr, cfg, err));
    CHECK(cfg.state_root == dr / "state");                 // not set to ""
}

TEST_CASE("clamp_edge_bounds: the compiled-in defaults already satisfy the floor") {
    // Regression guard. clamp_edge_bounds runs on every boot; if the DEFAULT config
    // trips it, the daemon emits a spurious warning with no config present (an
    // earlier 8 MiB max_request_bytes default was 64 KiB under 2*4 MiB + 64 KiB).
    // The clamp must only ever fire on an operator override.
    daemon_config def;                              // compiled-in defaults
    CHECK_FALSE(clamp_edge_bounds(def));            // no correction needed
    CHECK(def.max_request_bytes == 10u << 20);      // and the value is untouched

    // It still fires — and corrects — when an override reintroduces the trap.
    daemon_config bad;
    bad.max_config_bytes  = 8u << 20;               // request cap now below 2*config
    CHECK(clamp_edge_bounds(bad));
    CHECK(bad.max_request_bytes >= bad.max_config_bytes * 2 + (64u << 10));
}

TEST_CASE("config_load: non-object / unparseable file returns false") {
    const fs::path dr = "/srv/coastd";
    daemon_config cfg = seeded(dr);
    std::string err;
    SUBCASE("top-level array") {
        temp_cfg f("arr", R"([1,2,3])");
        CHECK_FALSE(load_config_file(f.path, dr, cfg, err));
        CHECK(!err.empty());
    }
    SUBCASE("malformed json") {
        temp_cfg f("malformed", R"({ "slots": )");
        CHECK_FALSE(load_config_file(f.path, dr, cfg, err));
        CHECK(!err.empty());
    }
    SUBCASE("missing file") {
        CHECK_FALSE(load_config_file(dr / "does-not-exist.json", dr, cfg, err));
    }
}
