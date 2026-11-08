#ifndef COAST_COASTD_CONFIG_LOAD_H
#define COAST_COASTD_CONFIG_LOAD_H

#include <filesystem>
#include <string>

#include "daemon/types.h"

namespace coastd {

// Load a coastd.json into cfg, layering file values over cfg's current contents
// (which the caller has already seeded with the compiled-in / data_root-derived
// defaults). Precedence is therefore defaults < file; the caller applies any CLI
// overrides AFTER this returns, so CLI still wins over the file.
//
// data_root anchors the literal token "<data-root>" wherever it appears in a
// path-valued key (state_root/runs_root/worktrees_root/repo_root): the token is
// replaced with data_root's string form, so an operator can write
//   "runs_root": "<data-root>/runs"
// without repeating an absolute prefix. data_root itself is never a key here —
// the file is discovered at <data-root>/coastd.json, so it must be known before
// the file is read.
//
// Robustness contract: returns false ONLY if the file cannot be opened or does
// not parse as a JSON object. A key that is unknown, wrong-typed, out-of-range,
// or empty is skipped with a warning and the existing value is kept — a
// slightly-stale or partly-typo'd file still boots the daemon.
bool load_config_file(const std::filesystem::path& path,
                      const std::filesystem::path& data_root,
                      daemon_config& cfg,
                      std::string& err);

} // namespace coastd

#endif // COAST_COASTD_CONFIG_LOAD_H
