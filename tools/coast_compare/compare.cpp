#include "compare.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ios>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace coast::compare {

namespace fs   = std::filesystem;
using json     = nlohmann::json;

namespace {

// ----- tolerance arithmetic --------------------------------------------------

[[nodiscard]] bool within_tolerance(double a, double b, tolerance tol) {
    const double diff = std::fabs(a - b);
    const double thr  = std::max(tol.rel * std::max(std::fabs(a), std::fabs(b)),
                                 tol.abs_floor);
    return diff <= thr;
}

[[nodiscard]] double rel_delta(double a, double b) {
    const double denom = std::max(std::fabs(a), std::fabs(b));
    if (denom == 0.0) return 0.0;
    return std::fabs(a - b) / denom;
}

// Strip a trailing "[N]" array-index decoration from a path before looking it
// up in the override map. We want `group_2/spike_counts[42]` to inherit the
// override on `group_2/spike_counts`.
[[nodiscard]] std::string_view strip_index_suffix(std::string_view path) {
    if (!path.empty() && path.back() == ']') {
        const auto open = path.rfind('[');
        if (open != std::string_view::npos) {
            return path.substr(0, open);
        }
    }
    return path;
}

// ----- JSON loading ---------------------------------------------------------

[[nodiscard]] json read_json_file(const fs::path& p) {
    std::ifstream in(p);
    if (!in) {
        throw std::runtime_error("could not open '" + p.string() + "'");
    }
    json j;
    in >> j;
    if (!in && !in.eof()) {
        throw std::runtime_error("read failure on '" + p.string() + "'");
    }
    return j;
}

// Returns the platform subdir name for the current OS.  update_golden.sh
// writes non-Linux goldens into goldens/<platform>/ so they can be gitignored
// without affecting the flat Linux layout.  The comparator mirrors this by
// also scanning that subdir when looking for a matching golden.
#if defined(__APPLE__)
inline constexpr const char* current_platform_subdir() { return "macos"; }
#elif defined(__linux__)
inline constexpr const char* current_platform_subdir() { return "linux"; }
#else
inline constexpr const char* current_platform_subdir() { return ""; }
#endif

[[nodiscard]] std::vector<fs::path> list_goldens_chronological(const fs::path& dir) {
    std::vector<fs::path> out;
    if (!fs::exists(dir) || !fs::is_directory(dir)) return out;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.is_regular_file() && e.path().extension() == ".json") {
            out.push_back(e.path());
        }
    }
    // Also scan the platform-specific subdir (e.g. dir/macos/ on Darwin).
    // This is where update_golden.sh places non-Linux goldens so they can be
    // gitignored without touching the flat Linux layout.  If the subdir does
    // not exist the scan is a no-op.
    const char* platform_sub = current_platform_subdir();
    if (platform_sub && platform_sub[0] != '\0') {
        const fs::path subdir = dir / platform_sub;
        if (fs::exists(subdir) && fs::is_directory(subdir)) {
            for (const auto& e : fs::directory_iterator(subdir)) {
                if (e.is_regular_file() && e.path().extension() == ".json") {
                    out.push_back(e.path());
                }
            }
        }
    }
    std::sort(out.begin(), out.end());  // β-naming is lexicographically chrono
    return out;
}

// ----- binary sidecar reader ------------------------------------------------

struct binary_ref {
    std::string dtype;     // "f64" | "i64"
    std::size_t length = 0;
};

[[nodiscard]] bool is_binary_ref_node(const json& n) {
    return n.is_object() && n.contains("binary") && n.contains("dtype")
        && n.contains("shape");
}

// Parse a binary-ref node into (dtype, length, sidecar_path).
struct parsed_binary_ref { std::string dtype; std::size_t length; fs::path sidecar; };

[[nodiscard]] parsed_binary_ref parse_binary_ref(const json& n, const fs::path& parent) {
    if (!n["binary"].is_string()) {
        throw std::runtime_error("binary-ref: 'binary' is not a string");
    }
    if (!n["dtype"].is_string()) {
        throw std::runtime_error("binary-ref: 'dtype' is not a string");
    }
    if (!n["shape"].is_array() || n["shape"].size() != 1
        || !n["shape"][0].is_number_unsigned()) {
        throw std::runtime_error("binary-ref: 'shape' must be a 1-element unsigned array");
    }
    return {
        n["dtype"].get<std::string>(),
        n["shape"][0].get<std::size_t>(),
        parent / n["binary"].get<std::string>()
    };
}

template <typename T>
[[nodiscard]] std::vector<T> read_sidecar(const fs::path& p, std::size_t length) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        throw std::runtime_error("missing sidecar '" + p.string() + "'");
    }
    std::vector<T> v(length);
    in.read(reinterpret_cast<char*>(v.data()),
            static_cast<std::streamsize>(length * sizeof(T)));
    if (in.gcount() != static_cast<std::streamsize>(length * sizeof(T))) {
        throw std::runtime_error("short read on sidecar '" + p.string() + "'");
    }
    return v;
}

// ----- result emission ------------------------------------------------------

void emit_structural(std::vector<field_result>& out, std::string path, std::string note) {
    field_result r;
    r.status = field_status::fail_structural;
    r.path   = std::move(path);
    r.note   = std::move(note);
    out.push_back(std::move(r));
}

void emit_scalar_result(std::vector<field_result>& out, std::string path,
                        double a_golden, double b_new, tolerance tol) {
    // NaN-on-either-side → structural error (IDD-confirmed by user)
    if (std::isnan(a_golden) || std::isnan(b_new)) {
        emit_structural(out, std::move(path), "NaN encountered");
        return;
    }
    field_result r;
    r.status       = within_tolerance(a_golden, b_new, tol)
                       ? field_status::pass : field_status::fail_tol;
    r.path         = std::move(path);
    r.golden_value = a_golden;
    r.new_value    = b_new;
    r.rel_delta    = rel_delta(a_golden, b_new);
    r.applied_tol  = tol;
    out.push_back(std::move(r));
}

// ----- recursive walker -----------------------------------------------------

void walk(const json& golden,
          const json& cand,
          const fs::path& parent_golden,
          const fs::path& parent_cand,
          const std::string& path,
          const tolerance_set& tols,
          compare_mode mode,
          std::vector<field_result>& out);

// ----- aggregate reduction ---------------------------------------------------

// Population statistics over an array. max is seeded from the first element (NOT
// 0) so all-negative buffers reduce correctly; empty → all-zero.
struct reduced { double mean = 0.0; double max = 0.0; double stddev = 0.0; };

[[nodiscard]] reduced reduce_stats(const std::vector<double>& v) {
    reduced r;
    if (v.empty()) return r;
    double sum = 0.0;
    double mx  = v[0];
    for (double x : v) { sum += x; if (x > mx) mx = x; }
    r.mean = sum / static_cast<double>(v.size());
    double sq = 0.0;
    for (double x : v) { const double d = x - r.mean; sq += d * d; }
    r.stddev = std::sqrt(sq / static_cast<double>(v.size()));
    r.max = mx;
    return r;
}

// Reduce golden+candidate to (mean, max, std) and emit one scalar result per
// statistic, each with its own (path, stat) aggregate tolerance. emit_scalar_result
// promotes a NaN on either side to a structural error, so NaNs still surface.
void emit_aggregate(std::vector<field_result>& out, const std::string& path,
                    const std::vector<double>& gv, const std::vector<double>& cv,
                    const tolerance_set& tols) {
    const reduced g = reduce_stats(gv);
    const reduced c = reduce_stats(cv);
    emit_scalar_result(out, path + ":mean", g.mean,   c.mean,   tols.get_aggregate(path, agg_stat::mean));
    emit_scalar_result(out, path + ":max",  g.max,    c.max,    tols.get_aggregate(path, agg_stat::max));
    emit_scalar_result(out, path + ":std",  g.stddev, c.stddev, tols.get_aggregate(path, agg_stat::stddev));
}

void walk_array_aggregate(const json& golden,
                          const json& cand,
                          const std::string& path,
                          const tolerance_set& tols,
                          std::vector<field_result>& out) {
    if (golden.size() != cand.size()) {
        emit_structural(out, path,
            "array length differs: golden=" + std::to_string(golden.size())
            + " new=" + std::to_string(cand.size()));
        return;
    }
    std::vector<double> gv, cv;
    gv.reserve(golden.size());
    cv.reserve(cand.size());
    for (std::size_t i = 0; i < golden.size(); ++i) {
        if (!golden[i].is_number() || !cand[i].is_number()) {
            emit_structural(out, path + "[" + std::to_string(i) + "]",
                "array element is not numeric (golden type=" +
                std::string(golden[i].type_name()) +
                ", new type=" + std::string(cand[i].type_name()) + ")");
            return;  // a malformed array can't be reduced
        }
        gv.push_back(golden[i].get<double>());
        cv.push_back(cand[i].get<double>());
    }
    emit_aggregate(out, path, gv, cv, tols);
}

void walk_binary_ref_aggregate(const json& golden,
                               const json& cand,
                               const fs::path& parent_golden,
                               const fs::path& parent_cand,
                               const std::string& path,
                               const tolerance_set& tols,
                               std::vector<field_result>& out) {
    parsed_binary_ref g, c;
    try { g = parse_binary_ref(golden, parent_golden); }
    catch (const std::exception& e) {
        emit_structural(out, path, std::string("golden binary-ref: ") + e.what());
        return;
    }
    try { c = parse_binary_ref(cand, parent_cand); }
    catch (const std::exception& e) {
        emit_structural(out, path, std::string("new binary-ref: ") + e.what());
        return;
    }
    if (g.dtype != c.dtype) {
        emit_structural(out, path,
            "binary dtype differs: golden=" + g.dtype + " new=" + c.dtype);
        return;
    }
    if (g.length != c.length) {
        emit_structural(out, path,
            "binary length differs: golden=" + std::to_string(g.length)
            + " new=" + std::to_string(c.length));
        return;
    }
    std::vector<double> gv, cv;
    try {
        if (g.dtype == "f64") {
            gv = read_sidecar<double>(g.sidecar, g.length);
            cv = read_sidecar<double>(c.sidecar, c.length);
        } else if (g.dtype == "i64") {
            const auto gi = read_sidecar<std::int64_t>(g.sidecar, g.length);
            const auto ci = read_sidecar<std::int64_t>(c.sidecar, c.length);
            gv.assign(gi.begin(), gi.end());
            cv.assign(ci.begin(), ci.end());
        } else {
            emit_structural(out, path,
                "unknown binary dtype '" + g.dtype + "' (expected f64 or i64)");
            return;
        }
    } catch (const std::exception& e) {
        emit_structural(out, path, e.what());
        return;
    }
    emit_aggregate(out, path, gv, cv, tols);
}

void walk_array(const json& golden,
                const json& cand,
                const std::string& path,
                tolerance tol,
                std::vector<field_result>& out) {
    if (golden.size() != cand.size()) {
        emit_structural(out, path,
            "array length differs: golden=" + std::to_string(golden.size())
            + " new=" + std::to_string(cand.size()));
        return;
    }
    for (std::size_t i = 0; i < golden.size(); ++i) {
        const auto& gv = golden[i];
        const auto& cv = cand[i];
        std::string elem_path = path + "[" + std::to_string(i) + "]";
        if (!gv.is_number() || !cv.is_number()) {
            emit_structural(out, std::move(elem_path),
                "array element is not numeric (golden type=" +
                std::string(gv.type_name()) +
                ", new type=" + std::string(cv.type_name()) + ")");
            continue;
        }
        emit_scalar_result(out, std::move(elem_path),
                           gv.get<double>(), cv.get<double>(), tol);
    }
}

void walk_binary_ref(const json& golden,
                     const json& cand,
                     const fs::path& parent_golden,
                     const fs::path& parent_cand,
                     const std::string& path,
                     tolerance tol,
                     std::vector<field_result>& out) {
    parsed_binary_ref g, c;
    try { g = parse_binary_ref(golden, parent_golden); }
    catch (const std::exception& e) {
        emit_structural(out, path, std::string("golden binary-ref: ") + e.what());
        return;
    }
    try { c = parse_binary_ref(cand, parent_cand); }
    catch (const std::exception& e) {
        emit_structural(out, path, std::string("new binary-ref: ") + e.what());
        return;
    }
    if (g.dtype != c.dtype) {
        emit_structural(out, path,
            "binary dtype differs: golden=" + g.dtype + " new=" + c.dtype);
        return;
    }
    if (g.length != c.length) {
        emit_structural(out, path,
            "binary length differs: golden=" + std::to_string(g.length)
            + " new=" + std::to_string(c.length));
        return;
    }
    if (g.dtype == "f64") {
        std::vector<double> gv, cv;
        try {
            gv = read_sidecar<double>(g.sidecar, g.length);
            cv = read_sidecar<double>(c.sidecar, c.length);
        } catch (const std::exception& e) {
            emit_structural(out, path, e.what());
            return;
        }
        for (std::size_t i = 0; i < g.length; ++i) {
            emit_scalar_result(out, path + "[" + std::to_string(i) + "]",
                               gv[i], cv[i], tol);
        }
    } else if (g.dtype == "i64") {
        std::vector<std::int64_t> gv, cv;
        try {
            gv = read_sidecar<std::int64_t>(g.sidecar, g.length);
            cv = read_sidecar<std::int64_t>(c.sidecar, c.length);
        } catch (const std::exception& e) {
            emit_structural(out, path, e.what());
            return;
        }
        for (std::size_t i = 0; i < g.length; ++i) {
            emit_scalar_result(out, path + "[" + std::to_string(i) + "]",
                               static_cast<double>(gv[i]),
                               static_cast<double>(cv[i]), tol);
        }
    } else {
        emit_structural(out, path,
            "unknown binary dtype '" + g.dtype + "' (expected f64 or i64)");
    }
}

void walk(const json& golden,
          const json& cand,
          const fs::path& parent_golden,
          const fs::path& parent_cand,
          const std::string& path,
          const tolerance_set& tols,
          compare_mode mode,
          std::vector<field_result>& out) {
    // Scalars/strings/bools use the path tolerance: strict → the strict lookup;
    // aggregate → the aggregate base for the path (per-stat tolerances apply only
    // to reduced array statistics, handled in walk_*_aggregate).
    const std::string_view tol_key = strip_index_suffix(path);
    const tolerance tol = (mode == compare_mode::aggregate)
                            ? tols.get_aggregate_base(tol_key)
                            : tols.get(tol_key);

    if (is_binary_ref_node(golden) || is_binary_ref_node(cand)) {
        if (!(is_binary_ref_node(golden) && is_binary_ref_node(cand))) {
            emit_structural(out, path,
                "type mismatch: one side is a binary-ref node, the other is not");
            return;
        }
        if (mode == compare_mode::aggregate) {
            walk_binary_ref_aggregate(golden, cand, parent_golden, parent_cand, path, tols, out);
        } else {
            walk_binary_ref(golden, cand, parent_golden, parent_cand, path, tol, out);
        }
        return;
    }

    if (golden.is_object() && cand.is_object()) {
        // Missing or extra keys are structural.
        for (auto it = golden.begin(); it != golden.end(); ++it) {
            const auto& key = it.key();
            const std::string sub = path.empty() ? key : (path + "/" + key);
            if (!cand.contains(key)) {
                emit_structural(out, sub, "field missing in new blob");
                continue;
            }
            walk(it.value(), cand[key], parent_golden, parent_cand, sub, tols, mode, out);
        }
        for (auto it = cand.begin(); it != cand.end(); ++it) {
            if (!golden.contains(it.key())) {
                const std::string sub = path.empty() ? it.key() : (path + "/" + it.key());
                emit_structural(out, sub, "field present in new blob, absent in golden");
            }
        }
        return;
    }

    if (golden.is_array() && cand.is_array()) {
        if (mode == compare_mode::aggregate) {
            walk_array_aggregate(golden, cand, path, tols, out);
        } else {
            walk_array(golden, cand, path, tol, out);
        }
        return;
    }

    if (golden.is_number() && cand.is_number()) {
        emit_scalar_result(out, path, golden.get<double>(), cand.get<double>(), tol);
        return;
    }

    if (golden.is_string() && cand.is_string()) {
        if (golden.get<std::string>() != cand.get<std::string>()) {
            emit_structural(out, path,
                "string mismatch: golden='" + golden.get<std::string>()
                + "' new='" + cand.get<std::string>() + "'");
        } else {
            field_result r;
            r.status      = field_status::pass;
            r.path        = path;
            r.applied_tol = tol;
            out.push_back(std::move(r));
        }
        return;
    }

    if (golden.is_boolean() && cand.is_boolean()) {
        if (golden.get<bool>() != cand.get<bool>()) {
            emit_structural(out, path, "boolean mismatch");
        } else {
            field_result r;
            r.status      = field_status::pass;
            r.path        = path;
            r.applied_tol = tol;
            out.push_back(std::move(r));
        }
        return;
    }

    emit_structural(out, path,
        std::string("type mismatch: golden=") + golden.type_name()
        + " new=" + cand.type_name());
}

// ----- aggregate tolerance parsing ------------------------------------------

[[nodiscard]] tolerance parse_tol_obj(const json& o, tolerance fallback) {
    if (!o.is_object()) {
        throw std::runtime_error("tolerance spec is not an object");
    }
    tolerance t = fallback;
    if (o.contains("rel"))       t.rel       = o["rel"].get<double>();
    if (o.contains("abs_floor")) t.abs_floor = o["abs_floor"].get<double>();
    return t;
}

// Parse one aggregate entry: a base (rel/abs_floor, falling back to base_fallback)
// plus optional per-statistic refinements under "mean"/"max"/"std", each of which
// falls back to this entry's base when a field is omitted.
[[nodiscard]] agg_tolerance parse_agg_entry(const json& o, tolerance base_fallback) {
    if (!o.is_object()) {
        throw std::runtime_error("aggregate tolerance entry is not an object");
    }
    agg_tolerance e;
    e.base = parse_tol_obj(o, base_fallback);
    if (o.contains("mean")) e.mean   = parse_tol_obj(o["mean"], e.base);
    if (o.contains("max"))  e.max    = parse_tol_obj(o["max"],  e.base);
    if (o.contains("std"))  e.stddev = parse_tol_obj(o["std"],  e.base);
    return e;
}

} // namespace

// ===== tolerance_set =========================================================

void tolerance_set::set_override(std::string path, tolerance t) {
    overrides_.insert_or_assign(std::move(path), t);
}

tolerance tolerance_set::get(std::string_view path) const {
    auto it = overrides_.find(std::string(path));
    if (it != overrides_.end()) return it->second;
    return default_;
}

namespace {
[[nodiscard]] const std::optional<tolerance>& pick_stat(const agg_tolerance& e, agg_stat s) {
    switch (s) {
        case agg_stat::mean:   return e.mean;
        case agg_stat::max:    return e.max;
        case agg_stat::stddev: return e.stddev;
    }
    return e.mean;  // unreachable
}
} // namespace

tolerance tolerance_set::get_aggregate(std::string_view path, agg_stat stat) const {
    auto it = agg_overrides_.find(std::string(path));
    if (it != agg_overrides_.end()) {
        const agg_tolerance& e = it->second;
        if (const auto& ps = pick_stat(e, stat)) return *ps;
        return e.base;
    }
    if (const auto& ps = pick_stat(agg_default_, stat)) return *ps;
    return agg_default_.base;
}

tolerance tolerance_set::get_aggregate_base(std::string_view path) const {
    auto it = agg_overrides_.find(std::string(path));
    if (it != agg_overrides_.end()) return it->second.base;
    return agg_default_.base;
}

void tolerance_set::set_aggregate_default(agg_tolerance t) {
    has_aggregate_ = true;
    agg_default_   = t;
}

void tolerance_set::set_aggregate_override(std::string path, agg_tolerance t) {
    has_aggregate_ = true;
    agg_overrides_.insert_or_assign(std::move(path), t);
}

tolerance_set tolerance_set::load(const fs::path& json_path) {
    tolerance_set ts;
    auto j = read_json_file(json_path);
    if (!j.is_object()) {
        throw std::runtime_error("tolerances file is not a JSON object");
    }
    if (j.contains("default")) {
        const auto& d = j["default"];
        if (!d.is_object()) {
            throw std::runtime_error("tolerances 'default' is not an object");
        }
        if (d.contains("rel"))       ts.default_.rel       = d["rel"].get<double>();
        if (d.contains("abs_floor")) ts.default_.abs_floor = d["abs_floor"].get<double>();
    }
    if (j.contains("overrides")) {
        const auto& ov = j["overrides"];
        if (!ov.is_object()) {
            throw std::runtime_error("tolerances 'overrides' is not an object");
        }
        for (auto it = ov.begin(); it != ov.end(); ++it) {
            tolerance t = ts.default_;
            const auto& spec = it.value();
            if (!spec.is_object()) {
                throw std::runtime_error("override '" + it.key() + "' is not an object");
            }
            if (spec.contains("rel"))       t.rel       = spec["rel"].get<double>();
            if (spec.contains("abs_floor")) t.abs_floor = spec["abs_floor"].get<double>();
            ts.overrides_.insert_or_assign(it.key(), t);
        }
    }
    // Aggregate base falls back to the strict default unless the aggregate block
    // overrides it — so aggregate mode still runs sensibly without an explicit
    // 'aggregate' block.
    ts.agg_default_.base = ts.default_;
    if (j.contains("aggregate")) {
        const auto& agg = j["aggregate"];
        if (!agg.is_object()) {
            throw std::runtime_error("tolerances 'aggregate' is not an object");
        }
        ts.has_aggregate_ = true;
        if (agg.contains("default")) {
            ts.agg_default_ = parse_agg_entry(agg["default"], ts.default_);
        }
        if (agg.contains("overrides")) {
            const auto& ov = agg["overrides"];
            if (!ov.is_object()) {
                throw std::runtime_error("tolerances 'aggregate.overrides' is not an object");
            }
            for (auto it = ov.begin(); it != ov.end(); ++it) {
                ts.agg_overrides_.insert_or_assign(
                    it.key(), parse_agg_entry(it.value(), ts.agg_default_.base));
            }
        }
    }
    return ts;
}

// ===== golden_match queries =================================================

bool golden_match::all_pass() const noexcept {
    for (const auto& f : fields) if (f.status != field_status::pass) return false;
    return true;
}
bool golden_match::any_structural() const noexcept {
    for (const auto& f : fields) if (f.status == field_status::fail_structural) return true;
    return false;
}
std::size_t golden_match::failing_count() const noexcept {
    std::size_t n = 0;
    for (const auto& f : fields) if (f.status != field_status::pass) ++n;
    return n;
}
double golden_match::max_rel_delta() const noexcept {
    double m = 0.0;
    for (const auto& f : fields) if (f.rel_delta > m) m = f.rel_delta;
    return m;
}

// ===== driver ===============================================================

compare_outcome compare_against_goldens(
    const fs::path& candidate_blob,
    const fs::path& goldens_dir,
    const tolerance_set& tolerances,
    compare_mode mode)
{
    compare_outcome out;
    out.goldens_considered = list_goldens_chronological(goldens_dir);

    if (out.goldens_considered.empty()) {
        out.kind = compare_outcome::result::structural_error;
        out.note = fs::exists(goldens_dir)
                     ? "goldens directory has no *.json files"
                     : "goldens directory does not exist";
        return out;
    }

    json candidate;
    try {
        candidate = read_json_file(candidate_blob);
    } catch (const std::exception& e) {
        out.kind = compare_outcome::result::structural_error;
        out.note = std::string("candidate read failed: ") + e.what();
        return out;
    }
    const fs::path parent_cand = candidate_blob.parent_path();

    // Walk every golden; keep first all-pass; otherwise pick closest.
    std::optional<golden_match> closest_clean;     // no structural fails
    std::optional<golden_match> closest_any;       // any (used only if no clean exists)

    for (const auto& g_path : out.goldens_considered) {
        json golden;
        try {
            golden = read_json_file(g_path);
        } catch (const std::exception& e) {
            golden_match gm{ g_path, {} };
            field_result r;
            r.status = field_status::fail_structural;
            r.note   = std::string("golden read failed: ") + e.what();
            gm.fields.push_back(std::move(r));
            if (!closest_any || gm.failing_count() < closest_any->failing_count()) {
                closest_any = gm;
            }
            continue;
        }
        const fs::path parent_golden = g_path.parent_path();

        golden_match gm{ g_path, {} };
        walk(golden, candidate, parent_golden, parent_cand, "", tolerances, mode, gm.fields);

        if (gm.all_pass()) {
            out.kind           = compare_outcome::result::matched;
            out.matched_golden = std::move(gm);
            return out;
        }

        if (!gm.any_structural()) {
            if (!closest_clean ||
                gm.failing_count() < closest_clean->failing_count() ||
                (gm.failing_count() == closest_clean->failing_count() &&
                 gm.max_rel_delta() < closest_clean->max_rel_delta())) {
                closest_clean = gm;
            }
        }
        if (!closest_any || gm.failing_count() < closest_any->failing_count()) {
            closest_any = gm;
        }
    }

    if (closest_clean) {
        out.kind           = compare_outcome::result::no_match;
        out.closest_golden = std::move(closest_clean);
    } else {
        out.kind           = compare_outcome::result::structural_error;
        out.closest_golden = std::move(closest_any);
    }
    return out;
}

// ===== report renderer ======================================================

namespace {

std::string fmt_double(double v) {
    std::ostringstream o;
    o << std::setprecision(8) << v;
    return o.str();
}
std::string fmt_pct(double rel) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(2) << (rel * 100.0) << "%";
    return o.str();
}
std::string fmt_tol(tolerance t) {
    std::ostringstream o;
    o << "rel=" << t.rel << " abs=" << t.abs_floor;
    return o.str();
}

void render_fields(std::ostream& out, std::vector<field_result> fields, bool verbose) {
    // Severity sort: structural first (rel_delta is meaningless for them, but
    // they signal harder failures), then fail_tol by rel_delta desc.
    std::stable_sort(fields.begin(), fields.end(),
        [](const field_result& a, const field_result& b) {
            auto rank = [](field_status s) {
                switch (s) {
                    case field_status::fail_structural: return 0;
                    case field_status::fail_tol:        return 1;
                    case field_status::pass:            return 2;
                }
                return 3;
            };
            const int ra = rank(a.status), rb = rank(b.status);
            if (ra != rb) return ra < rb;
            return a.rel_delta > b.rel_delta;
        });

    bool any_failure = false;
    for (const auto& f : fields) {
        if (f.status == field_status::pass) continue;
        any_failure = true;
        if (f.status == field_status::fail_structural) {
            out << "STRUCT  " << f.path << "  " << f.note << "\n";
        } else {
            out << "FAIL    " << f.path
                << "  golden=" << fmt_double(f.golden_value)
                << "  new="    << fmt_double(f.new_value)
                << "  Δ="      << fmt_pct(f.rel_delta)
                << "  tol="    << fmt_tol(f.applied_tol)
                << "\n";
        }
    }
    if (verbose) {
        for (const auto& f : fields) {
            if (f.status == field_status::pass) {
                out << "PASS    " << f.path << "  tol=" << fmt_tol(f.applied_tol) << "\n";
            }
        }
    }
    if (!any_failure && !verbose) {
        out << "(no failing fields)\n";
    }
}

} // namespace

void render_report(std::ostream& out, const compare_outcome& outcome, bool verbose) {
    using R = compare_outcome::result;
    switch (outcome.kind) {
        case R::matched: {
            out << "Matched golden: "
                << outcome.matched_golden->golden_path.filename().string() << "\n\n";
            render_fields(out, outcome.matched_golden->fields, verbose);
            break;
        }
        case R::no_match: {
            out << "No retained golden matched; closest was "
                << outcome.closest_golden->golden_path.filename().string() << "\n";
            out << "(considered " << outcome.goldens_considered.size()
                << " retained golden(s))\n\n";
            render_fields(out, outcome.closest_golden->fields, verbose);
            break;
        }
        case R::structural_error: {
            out << "Structural error";
            if (!outcome.note.empty()) out << ": " << outcome.note;
            out << "\n";
            if (outcome.closest_golden) {
                out << "Closest considered: "
                    << outcome.closest_golden->golden_path.filename().string() << "\n\n";
                render_fields(out, outcome.closest_golden->fields, verbose);
            } else {
                out << "(no goldens considered)\n";
            }
            break;
        }
    }
}

int exit_code_of(const compare_outcome& outcome) noexcept {
    switch (outcome.kind) {
        case compare_outcome::result::matched:          return 0;
        case compare_outcome::result::no_match:         return 1;
        case compare_outcome::result::structural_error: return 2;
    }
    return 2;
}

} // namespace coast::compare
