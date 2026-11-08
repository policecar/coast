#ifndef COAST_CORE_PATH_RESOLVER_H
#define COAST_CORE_PATH_RESOLVER_H

#include <filesystem>
#include <string_view>

namespace coast::paths {

// Expand a `config_base` template against a config file path.
//
// Tokens:
//   <config_dir> -> the directory holding the config file ("." when the config
//                   was given as a bare filename with no directory component).
//                   Only expands as a WHOLE path component ("<config_dir>/…"),
//                   because it can be empty, multi-segment, or absolute — splicing
//                   such a value into the middle of a segment is ill-defined, so a
//                   glued token (e.g. "x<config_dir>y") is left literal. This
//                   function is pure/silent; that misuse is warned about once when
//                   the app config loads (load_app_config), not per resolution.
//   <config>     -> the config file stem (e.g. "mnist" for "configs/mnist.json").
//                   A single filename stem, so it substitutes anywhere within a
//                   component: "<config>", "run_<config>", "<config>_out" all work.
//
// Built COMPONENT-WISE via std::filesystem::path composition, never by string
// substitution with embedded '/' literals. This matters for correctness: a bare
// config filename has an empty parent_path(), and a naive "<config_dir>/<config>"
// substitution would collapse to "/<config>" — a leading slash → an ABSOLUTE path
// at the filesystem root, not a CWD-relative one. Composing per component (with an
// empty <config_dir> mapped to ".") keeps a relative template relative, while an
// absolute template ("/data/...") or an absolute <config_dir> stays absolute
// because the path iterator yields the root "/" as its own component.
std::filesystem::path expand_config_base(std::string_view config_base,
                                         std::string_view config_path);

// Resolve a per-config OUTPUT path. If `leaf` is absolute it wins verbatim
// (an explicit absolute output location overrides the base); otherwise it is
// placed under expand_config_base(config_base, config_path). The returned path
// still obeys standard filesystem semantics — a relative result resolves against
// the process CWD at use time, an absolute one is used as-is.
std::filesystem::path resolve_config_output(std::string_view config_path,
                                            std::string_view leaf,
                                            std::string_view config_base);

// True if `leaf` is a RELATIVE path with a ".." component — which, joined under
// config_base by resolve_config_output, would climb above the base and write
// outside the resolved output tree. Absolute leaves are exempt: they are a
// sanctioned explicit escape (resolve_config_output honours an absolute leaf
// verbatim), so a ".." inside one is just ordinary normalization, not an escape.
//
// This is a predicate, not an enforcement point: the block belongs at CONFIG
// LOAD time (load_app_config / json_loader), so a bad leaf is rejected once,
// up front. resolve_config_output stays a pure, non-throwing hot-path function
// (the GUI resolves the snapshot dir once per hovered frame) that trusts its
// inputs were validated at load.
bool leaf_escapes_config_base(std::string_view leaf);

} // namespace coast::paths

#endif // COAST_CORE_PATH_RESOLVER_H
