#include "campaign/campaign_api.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace coast::campaign {

namespace {

using cfg_json = nlohmann::ordered_json; // ordered: axis order follows config key order

constexpr const char* k_config_name   = "config.json";
constexpr const char* k_manifest_name = "coast_runtime.json";
constexpr const char* k_observe_dir   = "observe";
constexpr const char* k_observe_ext   = ".observe.h5";
constexpr const char* k_final_state   = "final_state.h5";
constexpr const char* k_snapshot_dir  = "snapshots";

// Maximum JSON nesting depth accepted in a config. See flatten().
constexpr int k_max_json_depth = 64;

// coast's own output locations. They differ per run by construction (coastd writes
// each run's own directory into its config), so they would otherwise be discovered as
// an axis with one value per run. See scan_options::ignore_pointers for why this is a
// deny-list rather than a "distinct == run count" heuristic.
constexpr std::string_view k_builtin_denied[] = {
    "/snapshot_dir",
    "/observe/output_dir",
};

// ------------------------------------------------------------------ value order
//
// Total order over values, used to sort an axis's discovered set. Kinds are ordered
// absent < boolean < number < text so a partially-present leaf sorts predictably.
int kind_rank(value::kind k)
{
    switch (k) {
    case value::kind::absent:  return 0;
    case value::kind::boolean: return 1;
    case value::kind::number:  return 2;
    case value::kind::text:    return 3;
    }
    return 4;
}

bool value_less(const value& a, const value& b)
{
    if (a.k != b.k) return kind_rank(a.k) < kind_rank(b.k);
    switch (a.k) {
    case value::kind::absent:  return false;
    case value::kind::boolean: return !a.flag && b.flag;
    case value::kind::number:  return a.num < b.num;
    case value::kind::text:    return a.str < b.str;
    }
    return false;
}

// Exact equality, deliberately. Distinctness is about "did the writer emit two
// different values", so grouping with a tolerance here could merge two genuinely
// distinct sweep points. Tolerance belongs at QUERY time (see near_enough), where the
// user states the value they mean.
bool value_eq(const value& a, const value& b)
{
    if (a.k != b.k) return false;
    switch (a.k) {
    case value::kind::absent:  return true;
    case value::kind::boolean: return a.flag == b.flag;
    case value::kind::number:  return a.num == b.num;
    case value::kind::text:    return a.str == b.str;
    }
    return false;
}

// Relative comparison scaled by the larger magnitude, so it behaves at both 1e-4 and
// 1e2. Exact zero is handled separately: a relative tolerance around 0 has no reach,
// so 0 matches only 0.
bool near_enough(double a, double b, double rel_tol)
{
    if (a == b) return true;
    // A non-finite query value would otherwise make `scale` infinite and the test
    // `|a-b| <= rel_tol*scale` degenerate to `inf <= inf` — TRUE for every finite
    // value, i.e. where(axis, 1.0/0.0) would silently select the whole campaign
    // instead of reporting that no such value exists. NaN already falls out false.
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    const double scale = std::max(std::abs(a), std::abs(b));
    if (scale == 0.0) return true; // both zero, caught by a == b, but keep it total
    return std::abs(a - b) <= rel_tol * scale;
}

// --------------------------------------------------------------- config flatten
//
// Depth-first walk to leaf JSON Pointers. Empty objects/arrays are not leaves and are
// skipped; a null is recorded as kind::absent so a leaf that is null in one run and
// set in another still registers as a difference.
// `depth` is a HARD guard, not tuning. This walk is recursive, so nesting depth is
// stack depth: measured, ~30 000 levels segfaults on an 8 MB stack, and 30 000 levels
// is a 60 KB file. A SIGSEGV cannot be caught, so without this cap one malformed
// config anywhere under the root kills the process and defeats the whole
// "a bad file is reported, never fatal" contract. The reference campaign needs 4.
bool flatten(const cfg_json& j, std::string prefix, int depth,
             std::vector<std::pair<std::string, value>>& out)
{
    if (depth > k_max_json_depth) return false;

    if (j.is_object()) {
        for (const auto& [k, v] : j.items()) {
            // RFC 6901 escaping, so a key containing '/' or '~' round-trips.
            std::string esc;
            esc.reserve(k.size());
            for (const char c : k) {
                if (c == '~')      esc += "~0";
                else if (c == '/') esc += "~1";
                else               esc += c;
            }
            std::string child = prefix;
            child += '/';
            child += esc;
            if (!flatten(v, std::move(child), depth + 1, out)) return false;
        }
        return true;
    }
    if (j.is_array()) {
        for (std::size_t i = 0; i < j.size(); ++i) {
            std::string child = prefix;
            child += '/';
            child += std::to_string(i);
            if (!flatten(j[i], std::move(child), depth + 1, out)) return false;
        }
        return true;
    }

    value val;
    if (j.is_number()) {
        val.k   = value::kind::number;
        val.num = j.get<double>();
    } else if (j.is_boolean()) {
        val.k    = value::kind::boolean;
        val.flag = j.get<bool>();
    } else if (j.is_string()) {
        val.k   = value::kind::text;
        val.str = j.get<std::string>();
    } else {
        val.k = value::kind::absent; // null
    }
    out.emplace_back(std::move(prefix), std::move(val));
    return true;
}

cfg_json::json_pointer to_pointer(std::string_view ptr)
{
    try {
        return cfg_json::json_pointer(std::string(ptr));
    } catch (const std::exception& e) {
        throw std::runtime_error("invalid JSON pointer '" + std::string(ptr) + "': " +
                                 e.what());
    }
}

std::vector<std::string> split_pointer(std::string_view ptr)
{
    std::vector<std::string> comps;
    std::size_t              i = 0;
    while (i < ptr.size()) {
        if (ptr[i] == '/') { ++i; continue; }
        const std::size_t start = i;
        while (i < ptr.size() && ptr[i] != '/') ++i;
        comps.emplace_back(ptr.substr(start, i - start));
    }
    return comps;
}

// Candidate short names for a pointer, from most to least abbreviated:
//   primary_learning_rate
//   ngm25/primary_learning_rate                       (root section + leaf)
//   ngm25/L4-mnist/primary_learning_rate              (root + growing tail)
//   ngm25/neuron_groups/L4-mnist/primary_learning_rate
// The root section comes first at every level because that is what distinguishes the
// per-model-family duplicates a coast config carries (ngm2 / ngm25 / ngm3).
std::vector<std::string> name_candidates(const std::vector<std::string>& comps)
{
    std::vector<std::string> cands;
    if (comps.empty()) return cands;

    cands.push_back(comps.back());
    if (comps.size() == 1) return cands;

    const std::string& root = comps.front();
    // tail = the last `t` components, growing until it reaches everything after root.
    // The bound is `t <= comps.size() - 1`, not `t + 1 <= comps.size() - 1`: the
    // latter dropped the root-plus-full-tail candidate entirely, and for a
    // TWO-component pointer the loop never ran at all — so `/ngm2/lr` vs `/ngm25/lr`
    // never produced the documented `ngm25/lr` and both axes fell back to their raw
    // pointers. The existing collision test used 4-component pointers and passed.
    for (std::size_t t = 1; t <= comps.size() - 1; ++t) {
        std::string s = root;
        for (std::size_t i = comps.size() - t; i < comps.size(); ++i) {
            s += "/";
            s += comps[i];
        }
        if (cands.empty() || cands.back() != s) cands.push_back(s);
    }
    return cands;
}

// A short rendering of what a JSON value actually is, for an error message that tells
// the reader why their file was rejected without making them go and look.
std::string kind_name(const cfg_json& v)
{
    if (v.is_null())             return "null";
    if (v.is_boolean())          return "a boolean";
    if (v.is_string())           return "a string";
    if (v.is_array())            return "an array";
    if (v.is_object())           return "an object";
    if (v.is_number_float())     return "a float";
    if (v.is_number_integer())   return "a negative integer";
    if (v.is_number_unsigned())  return "an unsigned integer";
    return "an unrecognised value";
}

// Render a value for an error message. Deliberately NOT a bare j.dump():
//
//   * dump() RECURSES (parse and ~json in the vendored nlohmann 3.12 are iterative,
//     dump is not), and the manifest never passes through flatten(), so
//     k_max_json_depth does not protect it. Measured: a 180 kB coast_runtime.json
//     holding {"hostname": [[[...90000 deep...]]]} segfaulted the process inside
//     read_manifest — an uncatchable death from one malformed file, exactly what
//     flatten's depth cap exists to prevent.
//   * dump() is unbounded in SIZE. Measured: a 7.7 MB field produced an ~8 MB error
//     string, held twice (manifest::error plus the scan_issue copy), 137 MB RSS for
//     one run — and the GUI runs CalcTextSize over the whole thing every frame.
//
// A container's contents are not what the reader needs anyway: the type name alone
// says what is wrong. So only PRIMITIVES are shown, and long strings are clipped.
std::string brief(const cfg_json& v)
{
    if (!v.is_primitive()) return {}; // "field X is an array; expected a string"
    std::string s = v.dump();
    constexpr std::size_t k_max_len = 80;
    if (s.size() > k_max_len) { s.resize(k_max_len); s += "..."; }
    return s;
}

// Read coast_runtime.json. NOTHING is coerced to make a bad file look good.
//
// A field that is ABSENT is fine and keeps its default — the manifest schema is
// additive (sim_core/run_manifest.h), so an older or newer writer legitimately omits
// keys, and treating that as an error would break compatibility in both directions.
// A field that is PRESENT AS null is likewise fine: null is the writer's own encoding
// of "not applicable", not a damaged value. sim_core/run_manifest.cpp:178 emits
// `gpu_adapter: null` for every CPU-only run precisely so a consumer can tell "CPU"
// from "unknown", and test_run_manifest.cpp:94 pins it. Rejecting it made EVERY
// CPU-only run report a malformed manifest — the first strict version of this reader
// did exactly that, because its fixture hand-built a manifest that omitted the field
// instead of generating one with the real writer.
// A field that is PRESENT WITH ANOTHER WRONG TYPE is an error: it is recorded in
// `error` (and hence in campaign::issues()) instead of being silently replaced.
//
// This is also what removes an undefined behaviour: the old code guarded the step
// counts with is_number(), which admits a float, and then handed it to nlohmann's
// get<uint64_t>() — a static_cast<uint64_t>(double) that is UB out of range (measured:
// -1.5 yielded 18446744073709551615, 1e300 yielded 0). The writer types both fields as
// uint64_t, so a float there never came from coast; it means the file is not ours.
manifest read_manifest(const std::filesystem::path& p)
{
    manifest m;

    std::ifstream in(p);
    if (!in) return m; // no manifest yet — a run in flight. Not an error.

    cfg_json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        m.error = std::string("parse error: ") + e.what();
        return m;
    }
    if (!j.is_object()) {
        m.error = "top-level value is " + kind_name(j) + "; expected an object";
        return m;
    }

    std::vector<std::string> bad; // collected, so one pass reports every wrong field
    const auto wrong = [&bad, &j](const char* key, const char* expected) {
        const std::string shown = brief(j.at(key));
        bad.push_back(shown.empty()
            ? std::format("field \"{}\" is {}; expected {}", key,
                          kind_name(j.at(key)), expected)
            : std::format("field \"{}\" is {} ({}); expected {}", key,
                          kind_name(j.at(key)), shown, expected));
    };

    // Absent or null: keep the default, say nothing. See the note above.
    const auto missing = [&j](const char* key, cfg_json::const_iterator& it) {
        it = j.find(key);
        return it == j.end() || it->is_null();
    };

    const auto str = [&](const char* key) -> std::string {
        cfg_json::const_iterator it;
        if (missing(key, it)) return {};
        if (!it->is_string()) { wrong(key, "a string"); return {}; }
        return it->get<std::string>();
    };
    const auto u64 = [&](const char* key) -> std::uint64_t {
        cfg_json::const_iterator it;
        if (missing(key, it)) return 0;
        if (!it->is_number_unsigned()) { wrong(key, "an unsigned integer"); return 0; }
        return it->get<std::uint64_t>();
    };
    const auto dbl = [&](const char* key) -> double {
        cfg_json::const_iterator it;
        if (missing(key, it)) return 0.0;
        if (!it->is_number()) { wrong(key, "a number"); return 0.0; }
        return it->get<double>();
    };

    m.coast_git_sha       = str("coast_git_sha");
    m.exit_reason         = str("exit_reason");
    m.hostname            = str("hostname");
    m.gpu_adapter         = str("gpu_adapter");
    m.started_at_utc      = str("started_at_utc");
    m.completed_at_utc    = str("completed_at_utc");
    m.steps_requested     = u64("n_steps_requested");
    m.steps_completed     = u64("n_steps_completed");
    m.wall_clock_seconds  = dbl("wall_clock_seconds");
    m.present             = true;

    for (std::size_t i = 0; i < bad.size(); ++i) {
        if (i) m.error += "; ";
        m.error += bad[i];
    }
    return m;
}

} // namespace

// -------------------------------------------------------------------- value
std::string value::label() const
{
    switch (k) {
    case value::kind::absent:  return "<absent>";
    case value::kind::boolean: return flag ? "true" : "false";
    case value::kind::number:  return std::format("{}", num);
    case value::kind::text:    return str;
    }
    return {};
}

bool operator==(const value& a, const value& b) { return value_eq(a, b); }

// ------------------------------------------------------------- campaign::impl
struct campaign::impl {
    std::filesystem::path root;

    // Per-run state, index-aligned with `runs`.
    std::vector<std::filesystem::path>                     dirs;
    std::vector<std::string>                               ids; // root-relative, unique
    std::vector<std::unordered_map<std::string, value>>    leaves; // pointer -> value
    std::vector<manifest>                                  manifests;
    std::vector<run>                                       runs;

    std::vector<axis>                                      axes;
    std::unordered_map<std::string, std::uint32_t>         by_name;    // short name
    std::unordered_map<std::string, std::uint32_t>         by_pointer; // full pointer

    std::vector<scan_issue>                                issues;

    // Resolve a short name or a full JSON Pointer to an axis index. `npos` if unknown.
    static constexpr std::uint32_t npos = 0xFFFF'FFFFu;

    [[nodiscard]] std::uint32_t axis_index(std::string_view name) const
    {
        const std::string key(name);
        if (const auto it = by_name.find(key); it != by_name.end()) return it->second;
        if (const auto it = by_pointer.find(key); it != by_pointer.end()) return it->second;
        // Tolerate a pointer written without its leading slash.
        if (!key.empty() && key.front() != '/') {
            if (const auto it = by_pointer.find("/" + key); it != by_pointer.end()) {
                return it->second;
            }
        }
        return npos;
    }

    [[nodiscard]] const axis& require_axis(std::string_view name) const
    {
        const std::uint32_t i = axis_index(name);
        if (i == npos) {
            std::string msg = "unknown axis '" + std::string(name) + "'; this campaign has ";
            if (axes.empty()) {
                msg += "no swept parameters (all configs are identical)";
            } else {
                msg += "{";
                for (std::size_t a = 0; a < axes.size(); ++a) {
                    if (a) msg += ", ";
                    msg += axes[a].name;
                }
                msg += "}";
            }
            throw std::runtime_error(msg);
        }
        return axes[i];
    }

    [[nodiscard]] value leaf_of(std::uint32_t run_idx, const std::string& ptr) const
    {
        const auto& m  = leaves[run_idx];
        const auto  it = m.find(ptr);
        return it == m.end() ? value{} : it->second;
    }
};

campaign::campaign() : p_(std::make_unique<impl>()) {}
campaign::~campaign() = default;

void campaign::bind_runs()
{
    // weak, not shared: impl owns the run handles, so a shared_ptr here would form a
    // cycle and the campaign would never be freed.
    const std::weak_ptr<const campaign> self = shared_from_this();
    p_->runs.resize(p_->dirs.size());
    for (std::uint32_t i = 0; i < p_->runs.size(); ++i) {
        p_->runs[i].owner_ = self;
        p_->runs[i].idx_   = i;
    }
}

const std::filesystem::path&  campaign::root() const { return p_->root; }
std::size_t                   campaign::size() const { return p_->runs.size(); }
const std::vector<axis>&      campaign::axes() const { return p_->axes; }
std::vector<run>              campaign::runs() const { return p_->runs; }

const std::vector<scan_issue>& campaign::issues() const { return p_->issues; }

const axis* campaign::find_axis(std::string_view name) const
{
    const std::uint32_t i = p_->axis_index(name);
    return i == impl::npos ? nullptr : &p_->axes[i];
}

selection campaign::all() const
{
    selection s;
    s.camp_ = shared_from_this();
    s.idx_.resize(p_->runs.size());
    for (std::uint32_t i = 0; i < p_->runs.size(); ++i) s.idx_[i] = i;
    return s;
}

selection campaign::select(const std::vector<std::string>& run_ids) const
{
    selection s;
    s.camp_ = shared_from_this();
    for (const std::string& want : run_ids) {
        for (std::uint32_t i = 0; i < p_->runs.size(); ++i) {
            // Compare against the same id run::id() hands out (root-relative), so a
            // list round-tripped through selection::ids() resolves exactly.
            if (p_->runs[i].id() != want) continue;
            // A repeated id selects the run once. Duplicating it would silently
            // double that run's weight in any aggregate computed over the selection.
            if (std::find(s.idx_.begin(), s.idx_.end(), i) == s.idx_.end()) s.idx_.push_back(i);
            break;
        }
    }
    return s;
}

selection campaign::where(std::string_view a, double v, double rel_tol) const
{
    return all().where(a, v, rel_tol);
}
selection campaign::where(std::string_view a, std::string_view v) const
{
    return all().where(a, v);
}
selection campaign::where(std::string_view a, bool v) const { return all().where(a, v); }
selection campaign::where(const std::function<bool(const run&)>& pred) const
{
    return all().where(pred);
}

std::string campaign::summary() const
{
    std::string s = std::format("campaign {} — {} run{}", p_->root.string(), p_->runs.size(),
                                p_->runs.size() == 1 ? "" : "s");
    if (p_->axes.empty()) {
        s += ", no swept parameters (all configs identical)";
    } else {
        s += std::format(", {} axis/axes:", p_->axes.size());
        for (const auto& a : p_->axes) {
            s += std::format("\n  {} ({} values: ", a.name, a.values.size());
            for (std::size_t i = 0; i < a.values.size(); ++i) {
                if (i) s += ", ";
                s += a.values[i].label();
            }
            s += ")  [" + a.pointer + "]";
        }
    }
    if (!p_->issues.empty()) {
        // Printed in full, not counted: a summary that says "3 problems" and makes the
        // reader go and find them is the same as hiding them.
        s += std::format("\n  {} problem{} found:", p_->issues.size(),
                         p_->issues.size() == 1 ? "" : "s");
        for (const auto& i : p_->issues) {
            // Root-relative, for the same reason run::id() is: two `run1` directories
            // under different parents both print as "run1", and the reader cannot tell
            // which one is broken. A walk issue carries no file, so no empty brackets.
            s += std::format("\n    {}{} {}{}", i.label,
                             i.file.empty() ? std::string{} : " [" + i.file + "]", i.message,
                             i.run_dropped ? "  (run excluded)" : "");
        }
    }
    return s;
}

// ---------------------------------------------------------------------- run
//
// Every accessor goes through owner(), so a handle that outlived its campaign reports
// itself instead of reading freed memory. The check is one weak_ptr::lock per call —
// irrelevant next to the filesystem and HDF5 work these handles exist to reach, and
// the GUI's hot table loop binds `const run&` rather than calling through repeatedly.
const campaign& run::owner() const
{
    const std::shared_ptr<const campaign> c = owner_.lock();
    if (!c) {
        throw std::runtime_error(
            "run outlived its campaign (a run is a handle, not an owner — keep the "
            "campaign alive, or iterate a selection, which holds it for you)");
    }
    return *c;
}

// The id is the path RELATIVE TO THE CAMPAIGN ROOT, not the directory name. A bare
// filename is not unique across a nested layout — `root/a/run1` and `root/b/run1` both
// yield "run1" — and select() matches on the id, so a GUI hand-off of both rows
// emitted select({"run1","run1"}) and silently analysed the first run twice while
// never touching the second. Relative paths are unique by construction, since two
// directories cannot share one path under the same root.
std::string run::id() const { return owner().p_->ids[idx_]; }
const std::filesystem::path& run::dir() const { return owner().p_->dirs[idx_]; }
const manifest&              run::info() const { return owner().p_->manifests[idx_]; }

// A MALFORMED manifest is not evidence of completion. The loader has already told the
// user those fields are not to be trusted; letting exit_reason through anyway would
// have a filter like where([](const run& r){ return r.complete(); }) quietly admit runs
// whose provenance the loader itself flagged as damaged.
bool run::complete() const
{
    const manifest& m = info();
    return m.present && !m.malformed() && m.exit_reason == "completed_n_steps";
}

value run::get(std::string_view a) const
{
    const axis& ax = owner().p_->require_axis(a);
    return owner().p_->leaf_of(idx_, ax.pointer);
}

bool run::has(std::string_view a) const { return get(a).k != value::kind::absent; }

double run::number(std::string_view a) const
{
    const value v = get(a);
    if (v.k != value::kind::number) {
        throw std::runtime_error("axis '" + std::string(a) + "' is " + v.label() +
                                 " for run " + id() + ", not a number");
    }
    return v.num;
}

std::string run::text(std::string_view a) const
{
    const value v = get(a);
    if (v.k != value::kind::text) {
        throw std::runtime_error("axis '" + std::string(a) + "' is " + v.label() +
                                 " for run " + id() + ", not a string");
    }
    return v.str;
}

bool run::has_config(std::string_view ptr) const
{
    (void)to_pointer(ptr); // validate syntax, so a typo reports as such
    const value v = owner().p_->leaf_of(idx_, std::string(ptr));
    return v.k != value::kind::absent;
}

double run::config_number(std::string_view ptr) const
{
    (void)to_pointer(ptr);
    const value v = owner().p_->leaf_of(idx_, std::string(ptr));
    if (v.k != value::kind::number) {
        throw std::runtime_error("config leaf '" + std::string(ptr) + "' is " + v.label() +
                                 " for run " + id() + ", not a number");
    }
    return v.num;
}

std::string run::config_string(std::string_view ptr) const
{
    (void)to_pointer(ptr);
    const value v = owner().p_->leaf_of(idx_, std::string(ptr));
    if (v.k != value::kind::text) {
        throw std::runtime_error("config leaf '" + std::string(ptr) + "' is " + v.label() +
                                 " for run " + id() + ", not a string");
    }
    return v.str;
}

namespace {

// Open `d` for listing, telling "there is nothing here" apart from "I could not look".
// Returning an empty list for both made "the observe directory is unreadable" render as
// "run X has no observe files", and made a loop over snapshots() silently analyse zero
// checkpoints on an EIO/ESTALE directory. An ABSENT directory is genuinely empty and
// returns false without an error; anything else throws with the errno text.
bool list_dir(const std::filesystem::path& d, const std::string& what,
              const std::string& run_id, std::filesystem::directory_iterator& it)
{
    {
        std::error_code ec;
        const bool      there = std::filesystem::exists(d, ec);
        if (ec) {
            throw std::runtime_error("run " + run_id + ": " + what + " directory " +
                                     d.string() + " cannot be queried (" + ec.message() + ")");
        }
        if (!there) return false;
    }
    std::error_code ec;
    it = std::filesystem::directory_iterator(d, ec);
    if (ec) {
        throw std::runtime_error("run " + run_id + ": " + what + " directory " + d.string() +
                                 " cannot be listed (" + ec.message() + ")");
    }
    return true;
}

} // namespace

std::vector<std::string> run::observe_groups() const
{
    std::vector<std::string> names;
    const auto               d = dir() / k_observe_dir;

    std::filesystem::directory_iterator it;
    if (!list_dir(d, "observe", id(), it)) return names;

    std::error_code                           ec;
    const std::filesystem::directory_iterator last;
    for (; it != last; it.increment(ec)) {
        if (ec) { // the throwing operator++ is deliberately avoided here
            throw std::runtime_error("run " + id() + ": listing " + d.string() +
                                     " stopped part-way (" + ec.message() + ")");
        }
        const std::string fn = it->path().filename().string();
        const std::string ext(k_observe_ext);
        if (fn.size() > ext.size() && fn.compare(fn.size() - ext.size(), ext.size(), ext) == 0) {
            names.push_back(fn.substr(0, fn.size() - ext.size()));
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

std::filesystem::path run::observe_path(std::string_view group) const
{
    std::string name(group);

    // `group` becomes a path component. An absolute path REPLACES everything to its
    // left in operator/, and "../.." walks out of the run, so a group name is required
    // to be a plain name — this is a GUI text field and a REPL argument.
    if (name.find('/') != std::string::npos || name.find('\\') != std::string::npos ||
        name == ".." || name == ".") {
        throw std::runtime_error("observe group '" + name +
                                 "' is not a plain name (it must not contain a path)");
    }

    if (name.empty()) {
        const auto groups = observe_groups();
        if (groups.empty()) {
            throw std::runtime_error("run " + id() + " has no observe files under " +
                                     (dir() / k_observe_dir).string());
        }
        if (groups.size() > 1) {
            std::string msg = "run " + id() + " has " + std::to_string(groups.size()) +
                              " observe groups — name one of {";
            for (std::size_t i = 0; i < groups.size(); ++i) {
                if (i) msg += ", ";
                msg += groups[i];
            }
            throw std::runtime_error(msg + "}");
        }
        name = groups.front();
    }

    const auto path = dir() / k_observe_dir / (name + k_observe_ext);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        throw std::runtime_error("run " + id() + " has no observe group '" + name + "' (" +
                                 path.string() + ")");
    }
    return path;
}

std::shared_ptr<h5b::console::file_console> run::observe(std::string_view group) const
{
    return h5b::console::open(observe_path(group));
}

std::shared_ptr<h5b::console::file_console> run::final_state() const
{
    const auto      path = dir() / k_final_state;
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        throw std::runtime_error("run " + id() + " has no " + k_final_state);
    }
    return h5b::console::open(path);
}

std::vector<std::filesystem::path> run::snapshots() const
{
    std::vector<std::filesystem::path> out;
    const auto                         d = dir() / k_snapshot_dir;

    std::filesystem::directory_iterator it;
    if (!list_dir(d, "snapshot", id(), it)) return out;

    std::error_code                           ec;
    const std::filesystem::directory_iterator last;
    for (; it != last; it.increment(ec)) {
        if (ec) {
            throw std::runtime_error("run " + id() + ": listing " + d.string() +
                                     " stopped part-way (" + ec.message() + ")");
        }
        if (it->path().extension() == ".h5") out.push_back(it->path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

// ---------------------------------------------------------------- selection
run selection::iterator::operator*() const { return (*sel_)[i_]; }

selection::iterator selection::begin() const
{
    iterator it;
    it.sel_ = this;
    it.i_   = 0;
    return it;
}

selection::iterator selection::end() const
{
    iterator it;
    it.sel_ = this;
    it.i_   = idx_.size();
    return it;
}

// A default-constructed selection has no campaign. Every accessor that would
// dereference one checks first: `camp::selection s;` is a natural thing for a REPL
// user to write, and a null dereference inside a JIT interpreter is a process crash
// with no diagnostic.
const campaign& selection::bound() const
{
    if (!camp_) {
        throw std::runtime_error(
            "selection is not bound to a campaign (default-constructed; obtain one "
            "from campaign::all(), where() or select())");
    }
    return *camp_;
}

run selection::operator[](std::size_t i) const
{
    return bound().p_->runs[idx_[i]];
}

run selection::at(std::size_t i) const
{
    if (i >= idx_.size()) {
        throw std::runtime_error(std::format("selection index {} out of range ({} run{})",
                                             i, idx_.size(), idx_.size() == 1 ? "" : "s"));
    }
    return (*this)[i];
}

namespace {

std::string value_set_of(const axis& a)
{
    std::string s = "{";
    for (std::size_t i = 0; i < a.values.size(); ++i) {
        if (i) s += ", ";
        s += a.values[i].label();
    }
    return s + "}";
}

} // namespace

selection selection::where(std::string_view a, double v, double rel_tol) const
{
    const axis& ax = bound().p_->require_axis(a);

    // Precheck against the DISCOVERED value set, so a value that matches nothing is a
    // message rather than an empty result. Matching every value within tolerance (not
    // just the nearest) also keeps a campaign that stored one logical value in two
    // representations — e.g. 0.01 alongside its float32 round-trip — from splitting.
    const bool any = std::any_of(ax.values.begin(), ax.values.end(), [&](const value& c) {
        return c.k == value::kind::number && near_enough(c.num, v, rel_tol);
    });
    if (!any) {
        throw std::runtime_error(std::format(
            "no value of axis '{}' within {} (relative) of {}; this campaign has {}",
            ax.name, rel_tol, v, value_set_of(ax)));
    }

    selection out;
    out.camp_ = camp_;
    for (const std::uint32_t i : idx_) {
        const value c = bound().p_->leaf_of(i, ax.pointer);
        if (c.k == value::kind::number && near_enough(c.num, v, rel_tol)) out.idx_.push_back(i);
    }
    return out;
}

selection selection::where(std::string_view a, std::string_view v) const
{
    const axis& ax = bound().p_->require_axis(a);

    const bool any = std::any_of(ax.values.begin(), ax.values.end(), [&](const value& c) {
        return c.k == value::kind::text && c.str == v;
    });
    if (!any) {
        throw std::runtime_error(std::format("axis '{}' has no value '{}'; this campaign has {}",
                                             ax.name, v, value_set_of(ax)));
    }

    selection out;
    out.camp_ = camp_;
    for (const std::uint32_t i : idx_) {
        const value c = bound().p_->leaf_of(i, ax.pointer);
        if (c.k == value::kind::text && c.str == v) out.idx_.push_back(i);
    }
    return out;
}

selection selection::where(std::string_view a, bool v) const
{
    const axis& ax = bound().p_->require_axis(a);

    // Prechecked like the numeric and string overloads. Without this the one filter
    // that could not report a miss returned an empty selection instead — the exact
    // silent-empty-result failure the snapping design exists to prevent, and the
    // easiest one to hit, since asking for `true` on a {"on","off"} axis is natural.
    const bool any = std::any_of(ax.values.begin(), ax.values.end(), [&](const value& c) {
        return c.k == value::kind::boolean && c.flag == v;
    });
    if (!any) {
        throw std::runtime_error(std::format(
            "axis '{}' has no boolean value {}; this campaign has {}", ax.name,
            v ? "true" : "false", value_set_of(ax)));
    }

    selection out;
    out.camp_ = camp_;
    for (const std::uint32_t i : idx_) {
        const value c = bound().p_->leaf_of(i, ax.pointer);
        if (c.k == value::kind::boolean && c.flag == v) out.idx_.push_back(i);
    }
    return out;
}

selection selection::where(const std::function<bool(const run&)>& pred) const
{
    selection out;
    out.camp_ = camp_;
    for (const std::uint32_t i : idx_) {
        if (pred(bound().p_->runs[i])) out.idx_.push_back(i);
    }
    return out;
}

selection selection::sort_by(std::string_view a) const
{
    const axis& ax = bound().p_->require_axis(a);

    selection out = *this;
    std::stable_sort(out.idx_.begin(), out.idx_.end(),
                     [&](std::uint32_t l, std::uint32_t r) {
                         return value_less(bound().p_->leaf_of(l, ax.pointer),
                                           bound().p_->leaf_of(r, ax.pointer));
                     });
    return out;
}

std::vector<selection> selection::group_by(std::string_view a) const
{
    const axis& ax = bound().p_->require_axis(a);

    std::vector<selection> groups;
    groups.reserve(ax.values.size());
    for (const value& want : ax.values) { // axis order, so groups come out ordered
        selection g;
        g.camp_ = camp_;
        for (const std::uint32_t i : idx_) {
            if (value_eq(bound().p_->leaf_of(i, ax.pointer), want)) g.idx_.push_back(i);
        }
        if (!g.idx_.empty()) groups.push_back(std::move(g));
    }
    return groups;
}

std::vector<double> selection::values(std::string_view a) const
{
    const axis& ax = bound().p_->require_axis(a);

    std::vector<double> out;
    out.reserve(idx_.size());
    for (const std::uint32_t i : idx_) {
        const value c = bound().p_->leaf_of(i, ax.pointer);
        if (c.k != value::kind::number) {
            throw std::runtime_error("axis '" + ax.name + "' is " + c.label() + " for run " +
                                     bound().p_->runs[i].id() +
                                     ", so this selection has no numeric value vector");
        }
        out.push_back(c.num);
    }
    return out;
}

std::vector<std::string> selection::labels(std::string_view a) const
{
    const axis& ax = bound().p_->require_axis(a);

    std::vector<std::string> out;
    out.reserve(idx_.size());
    for (const std::uint32_t i : idx_) out.push_back(bound().p_->leaf_of(i, ax.pointer).label());
    return out;
}

std::vector<std::string> selection::ids() const
{
    std::vector<std::string> out;
    out.reserve(idx_.size());
    for (const std::uint32_t i : idx_) out.push_back(bound().p_->runs[i].id());
    return out;
}

std::vector<std::filesystem::path> selection::dirs() const
{
    std::vector<std::filesystem::path> out;
    out.reserve(idx_.size());
    for (const std::uint32_t i : idx_) out.push_back(bound().p_->dirs[i]);
    return out;
}

// --------------------------------------------------------------------- open()
namespace {

// The two shapes of scan_issue, built through named helpers so every field is written
// in one place. `label` stays empty here and is filled once the root is known.
scan_issue walk_issue(const std::filesystem::path& dir, std::string message)
{
    return scan_issue{dir, {}, {}, std::move(message), false};
}
scan_issue run_issue(const std::filesystem::path& dir, std::string_view file,
                     std::string message, bool dropped)
{
    return scan_issue{dir, {}, std::string(file), std::move(message), dropped};
}

// Recursive run discovery. A directory holding config.json IS a run and is not
// descended into. Every filesystem query takes a FRESH error_code: sharing one with a
// following call turns "this path does not exist" into "this tree is unreadable"
// (fs::is_directory(p, ec) sets ec for absent paths), and the range-for form of
// directory_iterator uses the THROWING operator++.
// `seen` holds the CANONICAL path of every directory already walked. Without it the
// walk follows symlinks (directory_entry::is_directory resolves through status()) with
// max_depth as the only brake, which is both a correctness and a cost problem —
// measured: a directory holding five `ln -s .` entries produced 488 281 visits in
// 0.75 s on tmpfs, and on NFS at ~1 ms per round trip that is days of stat traffic
// against a shared server. A cycle also discovers the SAME run directory several
// times over, so it would appear as several independent runs.
//
// EVERY way of not walking a subtree is REPORTED. The walk used to have four silent
// early returns, and a silently skipped subtree does not merely lose runs — it changes
// the ANSWER, because the axes are discovered by diffing whatever was found. Measured
// on three runs a=1,2,3 with the third under a chmod-000 parent: the campaign came back
// "1 run, no swept parameters (all configs identical)", a confidently false statement
// about a campaign that has a 3-valued axis. On NFS a transient EACCES/ESTALE does the
// same. `run_dropped` stays false for these: the count of excluded runs is unknown
// (that is the point), so it must not be added to a number the GUI presents as exact.
//
// A symlink whose target lies OUTSIDE the root is not followed. The canonical/`seen`
// pair stops cycles, not escapes: measured, a link inside the root pointing at a
// sibling tree pulled a foreign run in and merged its config keys into the axis
// namespace, and gave that run an id of "../../outside/secret_run".
void discover(const std::filesystem::path& dir, int depth, int max_depth,
              const std::filesystem::path& root_canon,
              std::set<std::filesystem::path>& seen,
              std::vector<std::filesystem::path>& out,
              std::vector<scan_issue>& issues)
{
    {
        std::error_code ec;
        std::filesystem::path canon = std::filesystem::canonical(dir, ec);
        if (ec) {
            issues.push_back(walk_issue(dir, "directory cannot be resolved (" + ec.message() +
                                        "); it and anything under it are not in this campaign"));
            return;
        }
        if (!seen.insert(canon).second) return; // already walked, via a link or a cycle

        // depth 0 is the root itself, which defines the frame rather than escaping it.
        if (depth > 0 && !root_canon.empty()) {
            const auto rel = canon.lexically_relative(root_canon);
            if (rel.empty() || *rel.begin() == "..") {
                issues.push_back(walk_issue(dir, "resolves to " + canon.string() +
                                            ", outside the campaign root; not followed"));
                return;
            }
        }
    }
    {
        std::error_code ec;
        if (std::filesystem::exists(dir / k_config_name, ec) && !ec) {
            out.push_back(dir);
            return;
        }
    }
    if (depth >= max_depth) {
        issues.push_back(walk_issue(dir, std::format(
            "nested deeper than the scan limit of {} levels; not walked "
            "(raise scan_options::max_depth)", max_depth)));
        return;
    }

    std::error_code                     ec;
    std::filesystem::directory_iterator it(dir, ec);
    if (ec) {
        issues.push_back(walk_issue(dir, "directory cannot be listed (" + ec.message() +
                                    "); anything under it is not in this campaign"));
        return;
    }

    std::vector<std::filesystem::path> subdirs;
    const std::filesystem::directory_iterator last;
    for (; it != last; it.increment(ec)) {
        if (ec) {
            issues.push_back(walk_issue(dir, "listing stopped part-way (" + ec.message() +
                                        "); this directory is only partly scanned"));
            break;
        }
        std::error_code dec;
        if (it->is_directory(dec) && !dec) subdirs.push_back(it->path());
    }
    // Directory order is unspecified; sort so run order is reproducible.
    std::sort(subdirs.begin(), subdirs.end());
    for (const auto& s : subdirs)
        discover(s, depth + 1, max_depth, root_canon, seen, out, issues);
}

// A directory rendered relative to the campaign root — the identity used for run ids
// and the label shown for a scan issue. See run::id() in the header for why a bare
// directory name is not usable as an identity.
//
// relative() is specified as weakly_canonical(p).lexically_relative(base), so it
// resolves symlinks. Two shapes come back unusable as a name and fall back to the
// directory name, which is unique in both: the root being a run itself yields ".", and
// a directory reached through a link yields a "../"-laden host path. (discover() now
// bars the outside-the-root case; a link INSIDE the root can still produce one.)
std::string root_relative(const std::filesystem::path& d, const std::filesystem::path& root)
{
    std::error_code ec;
    const auto      rel = std::filesystem::relative(d, root, ec);
    if (ec || rel.empty() || rel == "." || *rel.begin() == "..") return d.filename().string();
    return rel.generic_string();
}

// Assign each axis the shortest candidate name that no other axis also wants at that
// level. A candidate is taken only if no OTHER STILL-UNNAMED axis wants it at this
// level and no axis has already been given it, so a later axis cannot steal an earlier
// one's name.
void assign_names(std::vector<axis>& axes, const std::vector<std::vector<std::string>>& cands)
{
    std::vector<bool> done(axes.size(), false);
    std::set<std::string> taken; // names already assigned

    std::size_t max_levels = 0;
    for (const auto& c : cands) max_levels = std::max(max_levels, c.size());

    // Repeat the whole sweep while anything was newly named. A single forward pass
    // never revisits a level, so once a competitor resolved deeper the name it
    // vacated at a shallower level was never re-offered: `/ngm25/lr` and
    // `/ngm25/groups/L4/lr` ended up as "/ngm25/lr" (pointer fallback) and
    // "ngm25/lr", two identifiers differing only by a leading slash, when "lr" was
    // free the moment the deep one took "ngm25/lr".
    bool progress = true;
    while (progress) {
    progress = false;
    for (std::size_t level = 0; level < max_levels; ++level) {
        // Count how many still-unnamed axes want each candidate at this level.
        std::map<std::string, int> want;
        for (std::size_t a = 0; a < axes.size(); ++a) {
            if (done[a] || level >= cands[a].size()) continue;
            ++want[cands[a][level]];
        }
        for (std::size_t a = 0; a < axes.size(); ++a) {
            if (done[a] || level >= cands[a].size()) continue;
            const std::string& n = cands[a][level];
            if (want[n] == 1 && !taken.contains(n)) {
                axes[a].name = n;
                taken.insert(n);
                done[a]      = true;
                progress     = true;
            }
        }
    }
    }
    // Anything still unnamed (two axes with identical pointers is impossible, so this
    // only fires for exotic shapes) falls back to the pointer, which is unique.
    for (std::size_t a = 0; a < axes.size(); ++a) {
        if (!done[a]) axes[a].name = axes[a].pointer;
    }
}

} // namespace

std::shared_ptr<campaign> open(const std::filesystem::path& root, const scan_options& opts)
{
    {
        // A failed query is NOT an absent path. exists(p, ec) reports ec for a dead
        // mount or an unreadable parent, and calling that "does not exist" sends the
        // user looking for a typo instead of at the mount.
        std::error_code ec;
        const bool      there = std::filesystem::exists(root, ec);
        if (ec) {
            throw std::runtime_error("campaign root cannot be queried (" + ec.message() +
                                     "): " + root.string());
        }
        if (!there) {
            throw std::runtime_error("campaign root does not exist: " + root.string());
        }
    }
    {
        std::error_code ec;
        if (!std::filesystem::is_directory(root, ec) || ec) {
            throw std::runtime_error("campaign root is not a directory: " + root.string());
        }
    }

    // The root's canonical form is the frame the escape check compares against. If it
    // cannot be resolved the check is skipped rather than the scan refused — an empty
    // frame disables it, and the walk is no worse than it was before.
    std::filesystem::path root_canon;
    {
        std::error_code ec;
        root_canon = std::filesystem::canonical(root, ec);
        if (ec) root_canon.clear();
    }

    std::vector<std::filesystem::path> dirs;
    std::set<std::filesystem::path>    visited;
    std::vector<scan_issue>            walk_issues;
    discover(root, 0, std::max(0, opts.max_depth), root_canon, visited, dirs, walk_issues);
    if (dirs.empty()) {
        std::string why = "no run directories under " + root.string() +
                          " (a run is a directory containing " + k_config_name + ")";
        // Without this the message blames the layout for what is a permissions or
        // mount fault: the one thing the caller needs to know is that the walk did
        // not get everywhere it meant to.
        for (const auto& i : walk_issues) why += "\n  " + i.dir.string() + ": " + i.message;
        throw std::runtime_error(why);
    }

    std::shared_ptr<campaign> c(new campaign());
    auto&                     P = *c->p_;
    P.root   = root;
    P.issues = std::move(walk_issues);

    // ---- read the configs (and only the configs) ----
    //
    // `order` keeps first-seen config key order so the axes come out in the order the
    // config file lists them rather than hash order.
    std::vector<std::string>                        order;
    std::unordered_map<std::string, std::uint32_t>  seen;

    for (const auto& d : dirs) {
        cfg_json      j;
        std::ifstream in(d / k_config_name);
        if (!in) {
            P.issues.push_back(run_issue(d, k_config_name, "cannot be opened for reading", true));
            continue;
        }
        try {
            in >> j;
        } catch (const std::exception& e) {
            P.issues.push_back(
                run_issue(d, k_config_name, std::string("parse error: ") + e.what(), true));
            continue;
        }
        if (!j.is_object()) {
            P.issues.push_back(run_issue(
                d, k_config_name, "top-level value is " + kind_name(j) + "; expected an object",
                true));
            continue;
        }

        std::vector<std::pair<std::string, value>> flat;
        if (!flatten(j, "", 0, flat)) {
            P.issues.push_back(run_issue(
                d, k_config_name,
                std::format("nesting deeper than {} levels; refusing to walk it",
                            k_max_json_depth),
                true));
            continue;
        }

        std::unordered_map<std::string, value> m;
        m.reserve(flat.size() * 2);
        for (auto& [ptr, v] : flat) {
            if (!seen.contains(ptr)) {
                seen.emplace(ptr, 0);
                order.push_back(ptr);
            }
            m.emplace(std::move(ptr), std::move(v));
        }

        P.dirs.push_back(d);
        P.leaves.push_back(std::move(m));
        manifest mf = read_manifest(d / k_manifest_name);
        if (mf.malformed()) {
            // The run is KEPT: its parameters are readable and only its provenance is
            // in doubt. Dropping it would discard good scientific data over a damaged
            // metadata file.
            P.issues.push_back(run_issue(d, k_manifest_name, mf.error, false));
        }
        P.manifests.push_back(std::move(mf));
    }

    if (P.dirs.empty()) {
        throw std::runtime_error("every run directory under " + root.string() +
                                 " has an unreadable " + k_config_name);
    }

    // Root-relative ids, derived once. See run::id() in the header for why a bare
    // directory name is not usable as an identity.
    P.ids.reserve(P.dirs.size());
    for (const auto& d : P.dirs) P.ids.push_back(root_relative(d, P.root));

    // Display labels for the issues, by the SAME rule as the ids — one derivation, so
    // the two cannot drift. Filled here rather than at each push_back because the walk
    // issues are produced before the campaign (and its root) exist.
    for (auto& i : P.issues) i.label = root_relative(i.dir, P.root);

    c->bind_runs();

    // ---- diff: a leaf with more than one distinct value across the runs is an axis ----
    const auto denied = [&](const std::string& ptr) {
        for (const std::string_view d : k_builtin_denied) {
            if (ptr == d) return true;
        }
        return std::find(opts.ignore_pointers.begin(), opts.ignore_pointers.end(), ptr) !=
               opts.ignore_pointers.end();
    };

    std::vector<std::vector<std::string>> cands;
    for (const std::string& ptr : order) {
        if (denied(ptr)) continue;

        std::vector<value> distinct;
        for (std::size_t r = 0; r < P.leaves.size(); ++r) {
            const auto it = P.leaves[r].find(ptr);
            const value v = (it == P.leaves[r].end()) ? value{} : it->second;
            const bool  known = std::any_of(distinct.begin(), distinct.end(),
                                            [&](const value& d) { return value_eq(d, v); });
            if (!known) distinct.push_back(v);
        }
        if (distinct.size() < 2) continue;

        std::sort(distinct.begin(), distinct.end(), value_less);

        axis a;
        a.pointer = ptr;
        a.values  = std::move(distinct);
        P.axes.push_back(std::move(a));
        cands.push_back(name_candidates(split_pointer(ptr)));
    }

    assign_names(P.axes, cands);
    for (std::uint32_t i = 0; i < P.axes.size(); ++i) {
        P.by_name.emplace(P.axes[i].name, i);
        P.by_pointer.emplace(P.axes[i].pointer, i);
    }

    return c;
}

} // namespace coast::campaign
