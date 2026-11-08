#ifndef COAST_CORE_APP_CONFIG_H
#define COAST_CORE_APP_CONFIG_H

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace coast::paths {

// Layering helper: the effective leaf name for a per-config output. The
// experiment config's value wins when non-empty; an empty value ("" = "defer
// to the app default") falls back to the app_config default leaf. Used with
// coast::paths::resolve_config_output to place snapshots / observe / logs.
inline std::string effective_dir(std::string_view experiment,
                                 std::string_view app_default) {
    return std::string(experiment.empty() ? app_default : experiment);
}

// Application-level configuration — deliberately DISTINCT from the per-experiment
// simulation config (loader::global_cfg_t / configs/*.json). Where the experiment
// config describes one simulation, app_config controls WHERE the application puts
// things (and finds its own assets), app-wide.
//
// Every field has a built-in default that reproduces the intended layout, so the
// config file is OPTIONAL: with no file present, these defaults apply. A file only
// overrides the fields it names.
struct app_config {
    struct paths_t {
        // Template for where a config's per-experiment outputs (snapshots,
        // observe, logs) go. Tokens: <config_dir> -> dir holding the .json;
        // <config> -> config stem. Default places each config's outputs next to
        // it, isolated per config (configs/mnist.json -> configs/mnist/...).
        std::string config_base = "<config_dir>/<config>/";

        // Default leaf directory names, used when the experiment config leaves
        // the corresponding field empty ("" = "defer to the app default"). The
        // experiment config's own value, when non-empty, wins.
        std::string snapshot_dir = "snapshots";
        std::string observe_dir  = "observe";
        std::string log_dir      = "logs";
    } paths;

    struct assets_t {
        // GUI font. Empty -> the compile-time COAST_FONT_PATH (the build-copied
        // font, an absolute path that works from any CWD).
        std::string font;
    } assets;

    struct state_t {
        // Persisted UI/preference files. Default to their historical CWD names.
        std::string plot_themes = "coast_plot_themes.json";
        std::string imgui_ini   = "imgui.ini";
    } state;

    struct output_t {
        // Default directory the GUI seeds into plot-export dialogs.
        std::string plot_dir = ".";
    } output;
};

// (De)serialization via nlohmann, matching the codebase convention (observe_cfg,
// global_cfg): _WITH_DEFAULT so an absent key keeps the struct's default — which
// is exactly the "the file is optional; it only overrides named keys" contract.
// Nested sub-structs are defined first so the aggregate's macro finds their
// serializers.
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(app_config::paths_t,
    config_base, snapshot_dir, observe_dir, log_dir)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(app_config::assets_t, font)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(app_config::state_t,
    plot_themes, imgui_ini)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(app_config::output_t, plot_dir)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(app_config,
    paths, assets, state, output)

// Load the app config using this discovery precedence:
//   explicit_path (from --app-config) > $COAST_APP_CONFIG > ./coast.json > built-in.
//
// An absent discovered file yields built-in defaults (the file is optional).
// A file named EXPLICITLY (via --app-config or $COAST_APP_CONFIG) that is missing or
// malformed is a hard error (throws std::runtime_error) — the user asked for it.
// An auto-discovered ./coast.json that is malformed logs a warning and falls back
// to built-in defaults (the app must still run).
app_config load_app_config(const std::string& explicit_path = {});

// Outcome of write_app_config_file. Distinguishing `exists` from `error` lets the
// caller print the right message while the writer keeps the never-clobber
// guarantee to itself (there is no path from the caller that can reintroduce it).
enum class write_status : std::uint8_t { written, exists, error };

// Write `cfg` (default: built-in defaults) to `path` as pretty JSON — the
// template for `coast --init`. EXCLUSIVE create (O_EXCL via std::ios::noreplace):
// the file is created only if it does not already exist, so the never-clobber
// guarantee is enforced atomically in the open itself — not by a separate
// exists() check the caller races against. Returns `exists` if `path` was already
// there (nothing written), `error` on any other open/write failure, `written` on
// success.
write_status write_app_config_file(const std::filesystem::path& path,
                                   const app_config&            cfg = {});

// Resolve the SNAPSHOT directory for a config. The config's `snapshot_dir` leaf
// (empty -> the app default `paths.snapshot_dir`) is placed under
// `paths.config_base` (an absolute leaf wins). The ONE resolver every snapshot
// site routes through (headless resume + --checkpoint, GUI auto-save + resume +
// auto-checkpoint) so write and read always agree on the location (design §2.3).
// Unlike the observe leaf, the snapshot leaf is taken from the already-parsed
// config (passed in), so there is no file peek.
std::filesystem::path resolve_snapshot_output_dir(const std::string& config_path,
                                                  std::string_view   config_snapshot_leaf,
                                                  const app_config&  app_cfg);

} // namespace coast::paths

#endif // COAST_CORE_APP_CONFIG_H
