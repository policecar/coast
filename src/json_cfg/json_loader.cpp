//
// JSON config file loader implementation
//

#include <nlohmann/json.hpp>
#include <algorithm>
#include <iterator>
#include <span>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "json_cfg/ngm2_json.h"
#include "json_cfg/ngm3_json.h"
#include "json_cfg/ngm25_json.h"
#include "json_cfg/sim_json.h"
#include "json_cfg/dataset_json.h"
#include "json_cfg/retina_json.h"
#include "json_cfg/vis_hints_json.h"

#include "json_cfg/json_loader.h"
#include "core/path_resolver.h"
#include "core/seed.h"
#include "diag/diag.h"

using json = nlohmann::ordered_json;

namespace loader {

namespace {

// Settings a config may still carry but that no longer exist. Every unknown key is
// warned about; these get their own message, because they used to change the model.
constexpr std::string_view k_removed_keys[] = {
    "activity_learning_window",           // ngm25 / ngm3 groups
    "default_activity_learning_window",   // ngm2 neurons
};

// Settings that were renamed. The old name is an error, not an unknown key: an
// unknown key is warned about and then defaulted, which would run a different
// model than the config asks for.
constexpr std::pair<std::string_view, std::string_view> k_renamed_keys[] = {
    { "dendrite_branch_depth", "max_branch_level" },   // ngm3, 2026-11 (#206)
    { "max_branch_depth",      "max_branch_level" },   // ngm25, 2026-11 (#206)
    // A frozen run, and whether a load continues a sequence, are run-level, 2026-11:
    { "frozen",                           "the top-level frozen" },   // ngm2/25/3 groups
    { "restore_sequence_from_checkpoint", "the top-level run_id" },   // datasets (S.5.14)
};

std::string join(std::span<const std::string> items, std::string_view sep)
{
    std::string r;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i) r += sep;
        r += items[i];
    }
    return r;
}

// path + "/" + key as a JSON pointer: '~' and '/' inside a key are escaped
// (RFC 6901), so a group named "a/b" is not reported as two levels.
std::string pointer_step(const std::string& path, const std::string& key)
{
    std::string r = path + "/";
    for (const char c : key) {
        if      (c == '~') r += "~0";
        else if (c == '/') r += "~1";
        else               r += c;
    }
    return r;
}

// Keys that hold a seed or a seed offset (core/seed.h). Their from_json hooks
// refuse a bad value too, but cannot see their key; checking here first names it
// by its JSON path. A seed key missing from this list is still refused, only
// without the path. The list is kept by hand: unit.json_cfg.loader checks, for the
// seed fields known when it was written, that each is bound and named by its path;
// a seed field added later is covered only once it is added there too.
//
// Keys are matched by name anywhere in the config. An object or array under such a
// key is not a seed but a named entry (e.g. a neuron group called "random_seed"),
// so it is left to its own binding; a scalar is checked wherever it appears.
constexpr std::pair<std::string_view, const char*> k_seed_keys[] = {
    { "random_seed",        "a random seed"        },
    { "rnd_seed",           "a random seed"        },   // ngm2 dendrites
    { "ri_seed",            "a random seed"        },   // random_index inhibition
    { "random_seed_offset", "a random seed offset" },   // sim nodes
};

void find_bad_seeds(const json& j, const std::string& path, std::vector<std::string>& found)
{
    if (j.is_object()) {
        for (const auto& [key, value] : j.items()) {
            const std::string at = pointer_step(path, key);
            for (const auto& [name, what] : k_seed_keys)
                if (key == name && !value.is_structured() && !core::seed_bits_of(value))
                    found.push_back(at + ": " + what + " " + core::k_seed_rule + ", got " + value.dump());
            find_bad_seeds(value, at, found);
        }
    } else if (j.is_array()) {
        for (std::size_t i = 0; i < j.size(); ++i)
            find_bad_seeds(j[i], path + "/" + std::to_string(i), found);
    }
}

// The config as written (`in`) against the config as bound and serialised again
// (`back`): what the simulation will actually see. nlohmann binds by plain casts
// and ignores keys it does not know, so this is the one place a mismatch shows:
//   - a key of `in` missing from `back` is unknown: bound to nothing, ignored;
//   - a number that comes back different did not fit its C++ type: 2.7 -> 2 for
//     an integer, -1 -> 18446744073709551615 for size_t, 259 -> 3 for uint8_t,
//     1e300 -> inf for float. Floats compare to float precision (1e-6 relative).
// It relies on every binding writing back the keys it reads and the values it
// stored; a hook that deliberately rewrites a value would need an exemption here.

// True iff the bound value `back` does not hold the number the config wrote.
bool number_misread(const json& in, const json& back)
{
    if (!back.is_number()) return true;
    if (back.is_number_float()) {
        const double b = back.get<double>(), i = in.get<double>();
        return !std::isfinite(b) || std::abs(b - i) > 1e-6 * std::abs(i);
    }
    // back is an integer: the input must be the same whole number. nlohmann writes
    // an unsigned C++ type back as number_unsigned and a signed one as
    // number_integer, so back's json type tells the field's signedness.
    if (in.is_number_float()) {
        const double i = in.get<double>();
        // Range first: binding cast the float with static_cast, which is UB out
        // of range, and the result cannot be trusted to show it. ARM saturates
        // 2^64 to UINT64_MAX, and that converts back to exactly 2^64, so the
        // equality below would pass. This closes the 64-bit gap; a narrower field
        // (259.0 for a uint8_t) is still cast before this runs, and is caught
        // only because no supported compiler turns it back into the input.
        if (back.is_number_unsigned())
            return !(i > -1.0 && i < 0x1p64) || i != static_cast<double>(back.get<std::uint64_t>());
        return !(i >= -0x1p63 && i < 0x1p63) || i != static_cast<double>(back.get<std::int64_t>());
    }
    const bool in_neg   = in.is_number_integer() && !in.is_number_unsigned() && in.get<std::int64_t>() < 0;
    const bool back_neg = !back.is_number_unsigned() && back.get<std::int64_t>() < 0;
    if (in_neg != back_neg) return true;
    return in_neg ? in.get<std::int64_t>() != back.get<std::int64_t>()
                  : in.get<std::uint64_t>() != back.get<std::uint64_t>();
}

// Keys bound to a map with non-string keys. nlohmann writes such a map as an
// array of [key, value] pairs in the map's iteration order, which for an
// unordered_map is not the order written, so these are compared by key.
constexpr std::string_view k_keyed_arrays[] = {
    "node_deltas",   // sim node sets: unordered_map<sim_node_id_t, sim_node_delta_t>
};

void diff_round_trip(const json& in, const json& back, const std::string& path,
                     std::vector<std::string>& unknown, std::vector<std::string>& misread);

// A keyed array (k_keyed_arrays): each entry of `in` is matched to the entry of
// `back` with the same key, each `back` entry used once. An entry with no match
// had a key that did not fit its type, or repeated an earlier key (the map kept
// the first). Binding succeeded, so every entry is a [key, value] pair.
void diff_keyed_array(const json& in, const json& back, const std::string& path,
                      std::vector<std::string>& unknown, std::vector<std::string>& misread)
{
    std::vector<bool> used(back.size(), false);
    for (std::size_t i = 0; i < in.size(); ++i) {
        const std::string at = path + "/" + std::to_string(i);
        const json& key = in[i].at(0);
        std::size_t k = 0;
        while (k < back.size() && (used[k] || number_misread(key, back[k].at(0))))
            ++k;
        if (k == back.size()) {
            misread.push_back(at + "/0: key " + key.dump()
                              + " does not fit its type, or repeats an earlier key");
            continue;
        }
        used[k] = true;
        diff_round_trip(in[i], back[k], at, unknown, misread);
    }
}

void diff_round_trip(const json& in, const json& back, const std::string& path,
                     std::vector<std::string>& unknown, std::vector<std::string>& misread)
{
    if (in.is_null()) return;   // null means "use the defaults" (_WITH_DEFAULT; SPEC S.3.19)
    if (in.is_object() && back.is_object()) {
        for (const auto& [key, value] : in.items()) {
            const std::string at = pointer_step(path, key);
            if (const auto it = back.find(key); it == back.end()) unknown.push_back(at);
            else if (value.is_array() && it->is_array()
                     && std::ranges::find(k_keyed_arrays, key) != std::end(k_keyed_arrays))
                diff_keyed_array(value, *it, at, unknown, misread);
            else diff_round_trip(value, *it, at, unknown, misread);
        }
    } else if (in.is_array() && back.is_array()) {
        for (std::size_t i = 0; i < in.size() && i < back.size(); ++i)
            diff_round_trip(in[i], back[i], path + "/" + std::to_string(i), unknown, misread);
        if (in.size() != back.size())
            misread.push_back(path + " (array size " + std::to_string(in.size()) + " -> "
                              + std::to_string(back.size()) + ")");
    } else if (in.is_number()) {
        if (number_misread(in, back))
            misread.push_back(path + ": " + in.dump() + " -> " + back.dump());
    } else if (in.type() != back.type()) {
        misread.push_back(path + ": " + in.dump() + " -> " + back.dump());
    }
}

} // namespace

json_loader::json_loader(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + filename);
    }

    // Neither nlohmann's errors nor the strict-enum errors name the file, so
    // prefix it here. A strict-enum rejection keeps its std::invalid_argument
    // type; nlohmann's own errors (parse, type, range) become std::runtime_error
    // — every caller catches std::exception, none a nlohmann type.
    const std::string where = "config '" + filename + "': ";
    try {
        json j = json::parse(file);

        std::vector<std::string> bad_seeds;
        find_bad_seeds(j, "", bad_seeds);
        if (!bad_seeds.empty())
            throw std::invalid_argument(join(bad_seeds, "; "));

        // All sections (datasets, ngm2, sim, vis_hints, observe, snapshot_dir) are
        // bound via NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT; missing keys
        // fall back to the C++ default-initialised value, so this single
        // assignment handles both old and new configs.
        global_cfg = j;

        std::vector<std::string> unknown, misread;
        diff_round_trip(j, json(global_cfg), "", unknown, misread);
        if (!misread.empty())
            throw std::invalid_argument("value(s) that do not fit their type: " + join(misread, "; "));
        std::vector<std::string> renamed;
        for (const std::string& p : unknown) {
            const std::string_view key = std::string_view(p).substr(p.rfind('/') + 1);
            for (const auto& [old_name, new_name] : k_renamed_keys)
                if (key == old_name) renamed.push_back(p + " is now " + std::string(new_name));
        }
        if (!renamed.empty())
            throw std::invalid_argument("renamed or replaced setting(s): " + join(renamed, "; "));
        // The learning windows are tracked now (ngm/window_tracking.h); a
        // configured window is ignored, so it gets its own reason.
        const auto removed = std::ranges::stable_partition(unknown, [](const std::string& p) {
            const std::string_view key = std::string_view(p).substr(p.rfind('/') + 1);
            return std::ranges::find(k_removed_keys, key) == std::end(k_removed_keys);
        });
        if (removed.begin() != unknown.end())
            COAST_WARN("config", "{}{} ignored: the activity learning window is no longer "
                       "configurable, it follows the activity it gates (SPEC.md M.4.2a)",
                       where, join({ removed.begin(), removed.end() }, ", "));
        unknown.erase(removed.begin(), unknown.end());
        if (!unknown.empty())
            COAST_WARN("config", "{}unknown key(s) ignored: {}", where, join(unknown, ", "));
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(where + e.what());
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(where + e.what());
    }

    // Path-safety: a per-config output LEAF that climbs above config_base via
    // ".." would drop outputs outside the resolved tree. Reject it here, at load,
    // so coast::paths::resolve_config_output stays a pure per-frame resolver that
    // trusts its inputs. An absolute leaf is a sanctioned explicit escape (exempt).
    const auto reject_escape = [&](std::string_view field, std::string_view leaf) {
        if (coast::paths::leaf_escapes_config_base(leaf)) {
            throw std::runtime_error(
                "config '" + filename + "': " + std::string(field) + " = '" +
                std::string(leaf) +
                "' escapes config_base with '..'; use a relative name or an absolute path");
        }
    };
    if (global_cfg.run_id.empty())
        throw std::invalid_argument(where + "run_id is empty; name the run, or \"auto\" for a fresh id");
    reject_escape("snapshot_dir",        global_cfg.snapshot_dir);
    reject_escape("observe.output_dir",  global_cfg.observe.output_dir);
}



}
