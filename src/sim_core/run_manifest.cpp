//
// COAST run manifest writer implementation.
//

#include "sim_core/run_manifest.h"
#include "sim_core/coast_build_info.h"
#include "json_cfg/dataset_json.h"
#include "json_cfg/retina_json.h"
#include "json_cfg/json_loader.h"
#include "observe/observe_cfg.h"
#include "diag/diag.h"

#include <nlohmann/json.hpp>

#include <unistd.h>

#include <chrono>
#include <ctime>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>

namespace sim {

std::string to_string(exit_reason_t r) {
    switch (r) {
        case exit_reason_t::completed_n_steps:    return "completed_n_steps";
        case exit_reason_t::fatal_error:          return "fatal_error";
        case exit_reason_t::user_signal:          return "user_signal";
        case exit_reason_t::step_budget_exceeded: return "step_budget_exceeded";
    }
    return "unknown";
}

// ADL-discoverable JSON serializer for savestate_output_t.
// nlohmann::ordered_json preserves the key order specified here.
inline void to_json(nlohmann::ordered_json& j, const savestate_output_t& s) {
    j = nlohmann::ordered_json{
        {"path",  s.path},
        {"step",  s.step},
        {"bytes", s.bytes}
    };
}

std::string iso8601_utc(std::chrono::system_clock::time_point tp) {
    const std::time_t t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream os;
    os << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return os.str();
}

namespace {
std::string hostname_or_unknown() {
    char buf[256] = {0};
    if (gethostname(buf, sizeof(buf) - 1) == 0) {
        return std::string(buf);
    }
    return "unknown";
}

template <typename Map>
void add_active_dataset(nlohmann::ordered_json& out,
                        const char* family,
                        const std::string& name,
                        const Map& configs)
{
    const auto it = configs.find(name);
    if (it != configs.end()) {
        out[family][name] = it->second;
    }
}

nlohmann::ordered_json active_datasets_json(const loader::global_cfg_t& cfg)
{
    nlohmann::ordered_json out = nlohmann::ordered_json::object();
    for (const auto& node : cfg.sim.nodes) {
        const auto* info = find_entity_type(node.entity_type);
        if (info == nullptr || info->dataset_family == nullptr) continue;
        bool known = false;
        dataset::for_each_family(cfg.datasets, [&](std::string_view family, const auto& map) {
            if (family != info->dataset_family) return;
            known = true;
            add_active_dataset(out, info->dataset_family, node.entity_cfg, map);
        });
        if (!known)   // entity_types names a family cfg_store does not have
            throw std::logic_error(std::string("run manifest: no dataset family '")
                                   + info->dataset_family + "' for " + info->name);
    }
    return out;
}

// The configs in use by retina nodes, keyed by config name: the front end between
// a dataset and the groups shapes what the groups see. Each seed is the config
// seed before offsets, as its 32-bit value; each node's effective seed is in
// node_seeds.
nlohmann::ordered_json active_retina_json(const loader::global_cfg_t& cfg)
{
    nlohmann::ordered_json out = nlohmann::ordered_json::object();
    for (const auto& node : cfg.sim.nodes) {
        if (node.entity_type != io_entity_type_t::retina) continue;
        if (const auto it = cfg.retina.find(node.entity_cfg); it != cfg.retina.end())
            out[node.entity_cfg] = it->second;
    }
    return out;
}
// Render an output path for the manifest under a "self-describing bundle" rule,
// applied uniformly to EVERY path field so a consumer resolves them all the same
// way:
//   - if `p` is at or below the manifest's own directory, store it RELATIVE and
//     mark it with a leading "./" (resolve against the manifest location) — so the
//     manifest + its outputs form a relocatable bundle with no machine-specific
//     absolute paths;
//   - otherwise store it ABSOLUTE. We never climb out with "../..": a path that
//     isn't inside the bundle is recorded unambiguously instead of via a fragile
//     relative offset between two unrelated subtrees.
// Both paths are absolutized against the CWD (the run's, unchanged through exit)
// purely to decide containment; the check is lexical (no filesystem access), so it
// works for a directory that has not been created yet (e.g. a lazy observe dir).
std::string manifest_output_path(const std::filesystem::path& p,
                                 const std::filesystem::path& manifest_dir) {
    namespace fs = std::filesystem;
    std::error_code ec_p;
    std::error_code ec_dir;
    const fs::path abs_p = fs::absolute(p, ec_p).lexically_normal();
    const fs::path abs_dir =
        fs::absolute(manifest_dir.empty() ? fs::path(".") : manifest_dir, ec_dir)
            .lexically_normal();
    // If either path could not be absolutized (e.g. the run's CWD was removed
    // mid-run), fs::absolute returns empty and containment is undecidable —
    // recording that empty result would silently corrupt the manifest field a
    // downstream harness resolves. Fall back to the raw path as-is (the same
    // policy run_headless's savestate record uses). Separate error_codes so the
    // dir call cannot clear a failure the p call reported.
    if (ec_p || ec_dir) return p.generic_string();
    const fs::path rel = abs_p.lexically_relative(abs_dir);
    // "below or equal" iff lexically_relative yielded something that does not climb
    // out (its first component is not ".."). Empty means unrelated roots -> absolute.
    const bool below = !rel.empty() && rel.begin()->native() != "..";
    if (!below)      return abs_p.generic_string();
    if (rel == ".")  return std::string("./");       // p IS the manifest dir
    return "./" + rel.generic_string();
}

} // namespace

bool write_run_manifest(const run_manifest_inputs_t& inputs,
                        const loader::global_cfg_t&  cfg) noexcept {
    try {
        // The directory the manifest itself is written into — the anchor every
        // relative path field is resolved against (see manifest_output_path).
        const std::filesystem::path manifest_dir = inputs.output_path.parent_path();

        nlohmann::ordered_json j;
        j["schema_version"]     = std::string("1");
        j["coast_git_sha"]      = std::string(COAST_GIT_SHA);
        j["coast_git_dirty"]    = static_cast<bool>(COAST_GIT_DIRTY);
        j["coast_build_type"]   = std::string(COAST_BUILD_TYPE);
        j["coast_compiler"]     = std::string(COAST_COMPILER);
        j["coast_target_cpu"]   = std::string(COAST_TARGET_CPU);
        j["seed_offset"]        = inputs.seed_offset;
        j["n_steps_requested"]  = inputs.n_steps_requested;
        j["unlimited"]          = inputs.unlimited;
        j["n_steps_completed"]  = inputs.n_steps_completed;
        j["exit_reason"]        = to_string(inputs.exit_reason);
        j["wall_clock_seconds"] = inputs.wall_clock_seconds;
        j["hostname"]           = hostname_or_unknown();
        // GPU adapter identity when the run used a gpu_context; null for CPU-only
        // runs so a downstream harness can distinguish "CPU" from "unknown".
        if (inputs.gpu_adapter.empty()) j["gpu_adapter"] = nullptr;
        else                            j["gpu_adapter"] = inputs.gpu_adapter;
        j["started_at_utc"]     = inputs.started_at_utc;
        j["completed_at_utc"]   = inputs.completed_at_utc;

        // observe block: groups mirror cfg.observe verbatim; output_dir is the
        // RESOLVED directory (config leaf under config_base, or absolute override)
        // where files actually land — not the raw config leaf, which may be empty
        // when the config defers to the app default.
        nlohmann::ordered_json obs;
        obs["output_dir"] = manifest_output_path(inputs.observe_output_dir, manifest_dir);
        nlohmann::ordered_json groups_json;
        for (const auto& [name, gcfg] : cfg.observe.groups) {
            // step_range_t and combine_mode_t have to_json overloads in namespace observe.
            groups_json[name] = {
                {"ranges",  gcfg.ranges},
                {"combine", gcfg.combine}
            };
        }
        obs["groups"]    = groups_json;
        j["observe"]     = obs;

        // Mirror the active resolved dataset configs so a downstream harness can
        // audit what the run presented (index manifest, presentation params).
        j["datasets"] = active_datasets_json(cfg);
        // A frozen run: all learning off. run_id: the run whose checkpoints continue
        // its sequences (SPEC.md S.5.14).
        j["frozen"]   = cfg.frozen;
        j["run_id"]   = inputs.run_id;
        j["retina"]   = active_retina_json(cfg);

        // Effective RNG seed per active stochastic node, sourced from the built
        // entities in run_headless so a consumer needn't reconstruct coast's
        // seed-composition rule.
        {
            nlohmann::ordered_json seeds = nlohmann::ordered_json::array();
            for (const auto& ns : inputs.node_seeds) {
                nlohmann::ordered_json e;
                e["id"]             = ns.id;
                e["label"]          = ns.label;
                e["effective_seed"] = ns.effective_seed;
                if (ns.inhibition_seed)
                    e["inhibition_seed"] = *ns.inhibition_seed;
                seeds.push_back(std::move(e));
            }
            j["node_seeds"] = seeds;
        }

        // savestate_outputs: always a JSON array (empty if no state was saved).
        // ::array() yields an empty array rather than null for the empty case.
        {
            nlohmann::ordered_json arr = nlohmann::ordered_json::array();
            for (const auto& s : inputs.savestate_outputs) {
                nlohmann::ordered_json entry;
                to_json(entry, s);
                // Apply the same bundle rule as every other path field (record_savestate
                // stores the raw absolute path; the frame decision is made here).
                entry["path"] = manifest_output_path(s.path, manifest_dir);
                arr.push_back(entry);
            }
            j["savestate_outputs"] = arr;
        }

        // Create the manifest's parent directory if it doesn't exist yet, matching
        // the other CLI output paths (--checkpoint / --save-state auto-create via
        // hdf5::file::create). This does NOT rebase the path — it stays verbatim
        // (Tier-1); we only ensure its location exists. Best-effort: on failure the
        // tmpfile open below reports the real error.
        if (const auto parent = inputs.output_path.parent_path(); !parent.empty()) {
            std::error_code mk_ec;
            std::filesystem::create_directories(parent, mk_ec);
        }

        // Atomic write: serialize to string in memory, write tmp, rename.
        const std::string payload = j.dump(2);
        std::filesystem::path tmp_path = inputs.output_path;
        tmp_path += ".tmp";
        {
            std::ofstream os(tmp_path, std::ios::out | std::ios::binary | std::ios::trunc);
            if (!os) {
                COAST_ERROR("manifest", "failed to open tmpfile: {}", tmp_path.string());
                return false;
            }
            os.write(payload.data(), static_cast<std::streamsize>(payload.size()));
            if (!os) {
                COAST_ERROR("manifest", "failed to write tmpfile: {}", tmp_path.string());
                return false;
            }
        }
        std::error_code rn_ec;
        std::filesystem::rename(tmp_path, inputs.output_path, rn_ec);
        if (rn_ec) {
            COAST_ERROR("manifest", "rename failed: {} -> {}: {}",
                        tmp_path.string(), inputs.output_path.string(), rn_ec.message());
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        // Catch our own throws, do NOT propagate (the writer is noexcept). The
        // COAST_ERROR here is itself non-throwing by construction (emit_guarded
        // swallows the std::format OOM that would otherwise escape this noexcept
        // boundary → std::terminate), so the catch cannot re-raise.
        COAST_ERROR("manifest", "emit failed: {}", e.what());
        return false;
    } catch (...) {
        COAST_ERROR("manifest", "emit failed: unknown exception");
        return false;
    }
}

} // namespace sim
