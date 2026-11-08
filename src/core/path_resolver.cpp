#include "core/path_resolver.h"

#include <string>

namespace coast::paths {

namespace fs = std::filesystem;

namespace {
// Replace every occurrence of `token` in `s` with `value`. Advances past each
// replacement, so it terminates even if `value` contains `token`.
std::string replace_all(std::string_view s, std::string_view token, std::string_view value)
{
    std::string out;
    out.reserve(s.size());
    for (std::string_view::size_type i = 0;;) {
        const auto p = s.find(token, i);
        if (p == std::string_view::npos) { out.append(s.substr(i)); break; }
        out.append(s.substr(i, p - i));
        out.append(value);
        i = p + token.size();
    }
    return out;
}
} // namespace

fs::path expand_config_base(std::string_view config_base,
                            std::string_view config_path)
{
    const fs::path cfg{std::string(config_path)};

    // <config_dir>: the config's parent directory. A bare filename has an empty
    // parent — map that to "." so the template stays CWD-relative rather than
    // collapsing to an absolute "/<config>" path.
    fs::path config_dir = cfg.parent_path();
    if (config_dir.empty()) config_dir = ".";

    const std::string config_stem = cfg.stem().string();

    // Compose the template component by component. Iterating an fs::path yields
    // the root ("/") as its own element, so an absolute template keeps its root;
    // an absolute <config_dir> replaces the accumulator via operator/ semantics.
    // This function is PURE (no logging) so callers can re-invoke it freely — the
    // GUI resolves the snapshot dir once per hovered frame, for instance. The one
    // misuse worth surfacing (a <config_dir> token embedded in a larger segment)
    // is validated ONCE when the app config loads (see app_config.cpp), not here.
    fs::path result;
    for (const fs::path& part : fs::path{std::string(config_base)}) {
        const std::string s = part.string();

        // <config_dir> can expand to zero, one, or many path segments (and to an
        // absolute path), so it is only meaningful as a WHOLE component — splicing
        // a multi-segment/absolute directory into the middle of a segment is
        // ill-defined, so a <config_dir> glued into a larger segment is left literal.
        if (s == "<config_dir>") {
            result /= config_dir;
        } else if (s.find("<config>") != std::string::npos) {
            // <config> is always a single filename stem, so it substitutes cleanly
            // anywhere within a component: "<config>" -> "mnist", but also
            // "run_<config>" -> "run_mnist" and "<config>_out" -> "mnist_out".
            result /= fs::path{replace_all(s, "<config>", config_stem)};
        } else {
            result /= part;  // literal (incl. an embedded <config_dir>)
        }
    }
    return result;
}

fs::path resolve_config_output(std::string_view config_path,
                               std::string_view leaf,
                               std::string_view config_base)
{
    const fs::path leaf_path{std::string(leaf)};
    // An explicit absolute leaf wins outright. (operator/ would also replace the
    // base with an absolute rhs, but the guard makes the intent explicit.)
    if (leaf_path.is_absolute()) return leaf_path;
    return expand_config_base(config_base, config_path) / leaf_path;
}

bool leaf_escapes_config_base(std::string_view leaf)
{
    const fs::path p{std::string(leaf)};
    if (p.is_absolute()) return false;  // absolute leaf is a sanctioned escape
    for (const fs::path& part : p) {
        if (part == "..") return true;
    }
    return false;
}

} // namespace coast::paths
