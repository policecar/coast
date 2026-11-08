// Unit tests for app_config loading: built-in defaults when absent, key-wise
// overlay when present, and strict failure for an explicitly-named bad file.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "core/app_config.h"

namespace fs = std::filesystem;
using coast::paths::app_config;
using coast::paths::load_app_config;

namespace {
int g_counter = 0;
fs::path write_tmp(const std::string& content) {
    const auto p = fs::temp_directory_path() /
                   ("coast_appcfg_test_" + std::to_string(++g_counter) + ".json");
    std::ofstream(p) << content;
    return p;
}
} // namespace

TEST_CASE("app_config: no explicit path (and no discoverable file) => built-in defaults") {
    // Unit tests run from the build dir, which has no coast.json, and
    // $COAST_APP_CONFIG is not set in the test environment.
    const app_config c = load_app_config();
    CHECK(c.paths.config_base  == "<config_dir>/<config>/");
    CHECK(c.paths.snapshot_dir == "snapshots");
    CHECK(c.paths.observe_dir  == "observe");
    CHECK(c.paths.log_dir      == "logs");
    CHECK(c.assets.font.empty());
    CHECK(c.state.plot_themes  == "coast_plot_themes.json");
    CHECK(c.state.imgui_ini    == "imgui.ini");
    CHECK(c.output.plot_dir    == ".");
}

TEST_CASE("app_config: a full file overrides every field") {
    const auto p = write_tmp(R"({
      "paths":  {"config_base": "/runs/<config>/", "snapshot_dir": "snaps",
                 "observe_dir": "obs", "log_dir": "lg"},
      "assets": {"font": "/usr/share/fonts/x.ttf"},
      "state":  {"plot_themes": "themes.json", "imgui_ini": "ui.ini"},
      "output": {"plot_dir": "/exports"}
    })");
    const app_config c = load_app_config(p.string());
    CHECK(c.paths.config_base  == "/runs/<config>/");
    CHECK(c.paths.snapshot_dir == "snaps");
    CHECK(c.paths.observe_dir  == "obs");
    CHECK(c.paths.log_dir      == "lg");
    CHECK(c.assets.font        == "/usr/share/fonts/x.ttf");
    CHECK(c.state.plot_themes  == "themes.json");
    CHECK(c.state.imgui_ini    == "ui.ini");
    CHECK(c.output.plot_dir    == "/exports");
    fs::remove(p);
}

TEST_CASE("app_config: a partial file overrides only the named keys, rest stay default") {
    const auto p = write_tmp(R"({"paths": {"config_base": "./out/<config>/"}})");
    const app_config c = load_app_config(p.string());
    CHECK(c.paths.config_base  == "./out/<config>/");   // overridden
    CHECK(c.paths.snapshot_dir == "snapshots");          // default kept
    CHECK(c.paths.observe_dir  == "observe");            // default kept
    CHECK(c.state.imgui_ini    == "imgui.ini");          // default kept
    fs::remove(p);
}

TEST_CASE("app_config: an explicitly-named missing file is a hard error") {
    CHECK_THROWS_AS(load_app_config("/no/such/coast_appcfg.json"), std::runtime_error);
}

TEST_CASE("app_config: an explicitly-named malformed file is a hard error") {
    const auto p = write_tmp("{ this is not json ");
    CHECK_THROWS_AS(load_app_config(p.string()), std::runtime_error);
    fs::remove(p);
}

int main(int argc, char** argv)
{
    // load_app_config reads the real getenv("COAST_APP_CONFIG"); clear it so a
    // developer with that var exported doesn't see spurious failures (covers ctest
    // and a direct ./test_app_config run).
    ::unsetenv("COAST_APP_CONFIG");
    return doctest::Context(argc, argv).run();
}
