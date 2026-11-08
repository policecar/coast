#include "core/app_config.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>

#include <nlohmann/json.hpp>

#include "core/path_resolver.h"
#include "diag/diag.h"

namespace coast::paths {

namespace {
// Surface a config_base misuse ONCE, at load time: <config_dir> only expands as a
// whole path component (it can be empty/multi-segment/absolute), so gluing it into
// a larger segment ("x<config_dir>y") silently leaves it literal. Warn here rather
// than inside expand_config_base, which is a pure hot-path function re-invoked per
// resolution (e.g. once per hovered GUI frame). <config> has no such restriction.
void warn_on_embedded_config_dir(std::string_view config_base)
{
    for (const std::filesystem::path& part : std::filesystem::path{std::string(config_base)}) {
        const std::string s = part.string();
        if (s != "<config_dir>" && s.find("<config_dir>") != std::string::npos) {
            COAST_WARN("app_config",
                       "config_base segment '{}' embeds <config_dir> in a larger path "
                       "component; <config_dir> only expands as a whole component — it "
                       "will be left literal", s);
        }
    }
}

// Hard-reject a path-leaf default that would escape config_base via "..". These
// leaves are joined under config_base by resolve_config_output; a relative ".."
// climbs above it and drops outputs outside the resolved tree. Thrown here, at
// load, so the pure per-frame resolver never has to. (An ABSOLUTE leaf is a
// sanctioned explicit escape and is not a "..": leaf_escapes_config_base exempts it.)
void reject_escaping_leaf(std::string_view field, std::string_view leaf)
{
    if (leaf_escapes_config_base(leaf)) {
        throw std::runtime_error(
            "app config paths." + std::string(field) + " = '" + std::string(leaf) +
            "' escapes config_base with '..'; use a relative name or an absolute path");
    }
}
} // namespace

app_config load_app_config(const std::string& explicit_path)
{
    app_config cfg; // built-in defaults

    // Resolve the discovery source and whether a failure is fatal. Only a path
    // named explicitly on the command line (--app-config) must be valid — that is
    // an in-the-moment request, so silently ignoring it would be worse than
    // failing. Both auto-discovered sources ($COAST_APP_CONFIG and ./coast.json) are
    // soft: if they name a missing/unusable file we warn and fall through.
    std::string       path   = explicit_path;
    const bool        strict = !explicit_path.empty();

    if (path.empty()) {
        // $COAST_APP_CONFIG is an env var — far stickier than a flag. A user exports
        // it once and may later move/delete the file; treating that as fatal would
        // brick EVERY coast invocation (incl. just opening the GUI) with a cause
        // nowhere near the command being run. So it is a soft source like
        // coast.json: use it only if the file actually exists, else warn + fall
        // through to ./coast.json / built-in defaults.
        if (const char* env = std::getenv("COAST_APP_CONFIG"); env && *env) {
            std::error_code ec;
            if (std::filesystem::exists(env, ec)) {
                path = env;
            } else {
                COAST_WARN("app_config",
                           "$COAST_APP_CONFIG points at '{}', which does not exist; "
                           "ignoring it", env);
            }
        }
    }
    if (path.empty()) {
        std::error_code ec;
        if (std::filesystem::exists("coast.json", ec)) {
            path = "coast.json"; // discovered, not strict
        }
    }
    if (path.empty()) {
        return cfg; // no file anywhere -> built-in defaults
    }

    std::ifstream in(path);
    if (!in) {
        if (strict) {
            throw std::runtime_error("cannot open app config: " + path);
        }
        COAST_WARN("app_config",
                   "cannot open app config '{}'; using built-in defaults", path);
        return cfg;
    }

    try {
        nlohmann::json j;
        in >> j;
        // _WITH_DEFAULT (app_config.h): absent keys keep the struct defaults, so
        // this is a key-wise overlay onto the built-in defaults, not a
        // whole-object replace.
        cfg = j.get<app_config>();
        warn_on_embedded_config_dir(cfg.paths.config_base);

        // Because every key is optional, a syntactically valid but UNRELATED JSON
        // (a typo'd $COAST_APP_CONFIG, an experiment config, a stale env var) parses
        // into an all-defaults app_config with nothing actually overridden — and
        // would otherwise do so silently. Warn if none of the recognized top-level
        // sections is present, so that mistake surfaces instead of vanishing. An
        // empty object "{}" is exempt: it is an unambiguous, intentional "use all
        // defaults" overlay (e.g. a committed placeholder), not a wrong file.
        static constexpr std::string_view known[] = {"paths", "assets", "state", "output"};
        const bool recognized =
            j.is_object() &&
            (j.empty() ||
             std::any_of(std::begin(known), std::end(known),
                         [&](std::string_view k) { return j.contains(k); }));
        if (!recognized) {
            COAST_WARN("app_config",
                       "app config '{}' has no recognized keys "
                       "(paths/assets/state/output); using built-in defaults — is "
                       "this the right file?", path);
        }
    } catch (const std::exception& e) {
        if (strict) {
            throw std::runtime_error("invalid JSON in app config '" + path +
                                     "': " + e.what());
        }
        COAST_WARN("app_config",
                   "invalid JSON in app config '{}': {}; using built-in defaults",
                   path, e.what());
        return cfg;
    }

    // Path-safety: a leaf that climbs above config_base via ".." is rejected
    // outright (same fatal weight as a malformed file — a content error the user
    // must fix, not transient environmental drift). Built-in defaults are safe,
    // so this only ever fires on a hand-edited file.
    reject_escaping_leaf("snapshot_dir", cfg.paths.snapshot_dir);
    reject_escaping_leaf("observe_dir",  cfg.paths.observe_dir);
    reject_escaping_leaf("log_dir",      cfg.paths.log_dir);

    COAST_INFO("app_config", "loaded app config from {}", path);
    return cfg;
}

write_status write_app_config_file(const std::filesystem::path& path, const app_config& cfg)
{
    // Exclusive create (C++23 std::ios::noreplace -> O_EXCL): the open fails if
    // `path` already exists, so the never-clobber guarantee is atomic — no
    // check-then-write window a concurrent --init could slip through. This is why
    // it is NOT the run_manifest tmp+rename pattern: rename() atomically REPLACES
    // its target, which is precisely the clobber this must not do.
    std::ofstream out(path, std::ios::out | std::ios::noreplace);
    if (!out) {
        // The dominant failure is "already exists"; classify it (for messaging
        // only — the guarantee is already enforced above) and fall back to a
        // generic error for anything else (unwritable dir, permissions, …).
        std::error_code ec;
        return std::filesystem::exists(path, ec) ? write_status::exists
                                                 : write_status::error;
    }
    // _WITH_DEFAULT's to_json emits every field, so the written file is a full,
    // self-documenting template the user can edit — not a sparse overlay.
    const nlohmann::json j = cfg;
    out << j.dump(2) << '\n';
    return out.good() ? write_status::written : write_status::error;
}

std::filesystem::path resolve_snapshot_output_dir(const std::string& config_path,
                                                  std::string_view   config_snapshot_leaf,
                                                  const app_config&  app_cfg)
{
    const std::string leaf =
        effective_dir(config_snapshot_leaf, app_cfg.paths.snapshot_dir);
    return coast::paths::resolve_config_output(config_path, leaf,
                                               app_cfg.paths.config_base);
}

} // namespace coast::paths
