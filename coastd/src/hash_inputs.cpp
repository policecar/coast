#include "daemon/hash_inputs.h"

#include <algorithm>
#include <fstream>
#include <system_error>
#include <utility>

#include "daemon/sha256.h"

namespace coastd {

namespace {
namespace fs = std::filesystem;

std::expected<std::string, std::string> hash_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return std::unexpected("cannot open " + p.string());
    sha256 h;
    std::vector<char> buf(std::size_t{1} << 20);
    while (in) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        h.update(buf.data(), static_cast<std::size_t>(in.gcount()));
    }
    if (in.bad()) return std::unexpected("read error on " + p.string());
    return to_hex(h.digest());
}

} // namespace

std::expected<std::map<std::string, std::string>, std::string>
hash_inputs(const fs::path& dataset_base, const std::vector<std::string>& rels,
            const hash_limits& limits) {
    if (dataset_base.empty()) return std::unexpected("dataset_base is not configured");
    if (rels.empty()) return std::unexpected("no paths given");
    if (rels.size() > limits.max_paths)
        return std::unexpected(std::to_string(rels.size()) + " paths exceed the cap of " +
                               std::to_string(limits.max_paths));
    std::error_code ec;
    const fs::path base = fs::canonical(dataset_base, ec);
    if (ec) return std::unexpected("dataset_base " + dataset_base.string() + ": " + ec.message());

    // Validate and size every path before reading any: a refusal costs no I/O, and
    // the byte cap is checked against the whole request, not discovered midway.
    std::vector<std::pair<std::string, fs::path>> todo;
    std::uintmax_t total = 0;
    for (const std::string& rel : rels) {
        const fs::path p(rel);
        if (rel.empty()) return std::unexpected("an empty path");
        if (p.has_root_path()) return std::unexpected("'" + rel + "' is absolute; give it relative to dataset_base");
        if (std::ranges::any_of(p, [](const fs::path& c) { return c == ".."; }))
            return std::unexpected("'" + rel + "' contains '..'");
        const fs::path real = fs::canonical(base / p, ec);
        if (ec) return std::unexpected("'" + rel + "': " + ec.message());
        // Component-wise prefix test (a string prefix would accept /data2 for /data).
        // real == base passes here and is refused just below as not a regular file.
        if (std::ranges::mismatch(base, real).in1 != base.end())
            return std::unexpected("'" + rel + "' resolves to " + real.string() +
                                   ", outside dataset_base " + base.string());
        if (!fs::is_regular_file(real, ec))
            return std::unexpected("'" + rel + "' is not a regular file");
        const std::uintmax_t size = fs::file_size(real, ec);
        if (ec) return std::unexpected("'" + rel + "': " + ec.message());
        total += size;
        if (total > limits.max_total_bytes)
            return std::unexpected("the files exceed the cap of " +
                                   std::to_string(limits.max_total_bytes) + " bytes");
        todo.emplace_back(rel, real);
    }

    std::map<std::string, std::string> out;
    for (const auto& [rel, real] : todo) {
        auto hex = hash_file(real);
        if (!hex) return std::unexpected("'" + rel + "': " + hex.error());
        out[rel] = std::move(*hex);
    }
    return out;
}

} // namespace coastd
