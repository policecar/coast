#ifndef COAST_COASTD_HASH_INPUTS_H
#define COAST_COASTD_HASH_INPUTS_H

// -----------------------------------------------------------------------------
// hash_inputs — the SHA-256 of files under the daemon's dataset_base, computed on
// the daemon's own filesystem. It answers the question a remote client cannot
// answer for itself: which bytes will a run's `./data/<rel>` read on THIS machine?
// (The run dir's `data` link points at dataset_base, child_cmd.cpp.) Pure: no
// daemon, no sockets; the `hash_inputs` command in comms_server is a thin wrapper.
//
// Every path is RELATIVE to dataset_base and must stay inside it. Refused, before
// any file is read: an absolute path, a `..` component, a path that resolves
// (after symlinks, std::filesystem::canonical) outside the canonical
// dataset_base, anything that is not a regular file, and a request beyond the
// caps. A symlink that stays inside dataset_base is followed.
//
// The canonical path is checked and then opened by name, so a symlink swapped in
// between could redirect the read. Not defended: whoever can write dataset_base
// can already change the data itself, and coastd runs any submitted ref's code.
// -----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace coastd {

struct hash_limits {
    std::size_t   max_paths       = 256;                // the eval asks for ~40
    std::uintmax_t max_total_bytes = std::uintmax_t{4} << 30;   // 4 GiB; the eval's set is 173 MB
};

// rel path -> lower-case hex SHA-256, or the reason the WHOLE request is refused
// (naming the offending path). No partial answers: a client that asked for a set
// of files needs all of them.
[[nodiscard]] std::expected<std::map<std::string, std::string>, std::string>
hash_inputs(const std::filesystem::path& dataset_base, const std::vector<std::string>& rels,
            const hash_limits& limits = {});

} // namespace coastd

#endif // COAST_COASTD_HASH_INPUTS_H
