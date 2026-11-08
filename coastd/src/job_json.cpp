#include "daemon/job_json.h"

#include <array>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <type_traits>
#include <utility>

#include <nlohmann/json.hpp>

namespace coastd {

namespace {

using nlohmann::json;

// Tolerant field read: return `dflt` when the key is ABSENT, explicitly `null`,
// or present-but-not-convertible. nlohmann's json::value() returns the default
// only for an ABSENT key — a present-but-null `checkpoint`/`failure_reason` or a
// wrong-typed `priority` THROWS, which (via parse_job -> load_all) would
// quarantine an otherwise-live job and orphan its child. Defaulting a single
// malformed advisory field is the additive-tolerance contract (§5.4); quarantine
// is reserved for unparseable JSON or a future schema_version, not one sloppy key.
// The tolerance above is the intent; the mechanism used to be wrong. It relied on
// get<T>() THROWING for a malformed value, and for numbers nlohmann does not throw
// — it silently coerces. MEASURED against the bundled version:
//
//     {"a": -1}  -> get<std::uint64_t>() == 18446744073709551615   (no throw)
//     {"b": 3.7} -> get<int>()           == 3                      (no throw)
//
// So a corrupt number did not fall back to the default, it became a PLAUSIBLE
// WRONG one — much worse, because every consumer downstream treats it as authored.
// A bit-flipped or hand-edited "steps_done": -1 becomes ~1.8e19, so reconcile's
// `steps_done >= steps_target` fires and a job that has run zero steps is filed
// `done/` and never runs again; symmetrically "steps_target": -1 turns a bounded
// run into an effectively unbounded one. Both are silent.
//
// Validate the JSON storage type FIRST, then range-check before narrowing, so
// "malformed" reaches the default the contract promises. Note is_number_unsigned()
// is what rejects -1 for an unsigned field: nlohmann stores a negative literal as
// number_integer, and only the explicit range check catches an unsigned value too
// large for a narrower signed target.
template <class T>
T get_or(const json& o, const char* key, T dflt) {
    const auto it = o.find(key);
    if (it == o.end() || it->is_null()) return dflt;

    // bool first: std::is_unsigned_v<bool> is true, so it would otherwise be
    // mis-routed into the integer branch below.
    if constexpr (std::is_same_v<T, bool>) {
        if (!it->is_boolean()) return dflt;
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (!it->is_string()) return dflt;
    } else if constexpr (std::is_unsigned_v<T>) {
        if (!it->is_number_unsigned()) return dflt;              // rejects -1 and 3.7
        const auto v = it->get<std::uint64_t>();
        if (v > static_cast<std::uint64_t>(std::numeric_limits<T>::max())) return dflt;
        return static_cast<T>(v);
    } else if constexpr (std::is_integral_v<T>) {
        if (!it->is_number_integer()) return dflt;               // rejects 3.7
        if (it->is_number_unsigned()) {
            const auto v = it->get<std::uint64_t>();
            if (v > static_cast<std::uint64_t>(std::numeric_limits<T>::max())) return dflt;
            return static_cast<T>(v);
        }
        const auto v = it->get<std::int64_t>();
        if (v < static_cast<std::int64_t>(std::numeric_limits<T>::min()) ||
            v > static_cast<std::int64_t>(std::numeric_limits<T>::max())) return dflt;
        return static_cast<T>(v);
    }
    try {
        return it->get<T>();
    } catch (const json::exception&) {
        return dflt;
    }
}

// Single source of truth for the status <-> string mapping.
constexpr std::array<std::pair<job_status, std::string_view>, 9> status_table{{
    {job_status::queued,     "queued"},
    {job_status::resolving,  "resolving"},
    {job_status::building,   "building"},
    {job_status::ready,      "ready"},
    {job_status::running,    "running"},
    {job_status::done,       "done"},
    {job_status::failed,     "failed"},
    {job_status::cancelled,  "cancelled"},
    {job_status::quarantine, "quarantine"},
}};

} // namespace

std::string_view status_name(job_status s) noexcept {
    for (const auto& [val, name] : status_table)
        if (val == s) return name;
    return "queued";
}

job_status status_from(std::string_view s) noexcept {
    for (const auto& [val, name] : status_table)
        if (name == s) return val;
    return job_status::queued;  // unknown token -> safe default (advisory field)
}

// The keys this build models. Anything else in a job file is preserved verbatim
// rather than dropped (parse_job/dump_job).
const std::set<std::string>& known_job_keys() {
    static const std::set<std::string> keys{
        "schema_version", "coastd_version", "id", "submitted_at", "user_message",
        "ref_symbolic", "ref_sha", "run_dir", "steps_target", "steps_done",
        "checkpoint", "priority", "cancel_requested", "status", "pid",
        "start_token", "launch_failures", "no_checkpoint_launches",
        "last_launch_ms", "failure_reason", "load_state_from", "load_state", "load_state_step"};
    return keys;
}

std::string dump_job(const job& j) {
    json o;
    o["schema_version"]   = j.schema_version;
    o["coastd_version"]   = j.coastd_version;
    o["id"]               = j.id;
    o["submitted_at"]     = j.submitted_at;
    o["user_message"]     = j.user_message;
    o["ref_symbolic"]     = j.ref_symbolic;
    o["ref_sha"]          = j.ref_sha;
    o["run_dir"]          = j.run_dir;
    o["steps_target"]     = j.steps_target;
    o["steps_done"]       = j.steps_done;
    o["checkpoint"]       = j.checkpoint;
    o["priority"]         = j.priority;
    o["cancel_requested"] = j.cancel_requested;
    o["status"]           = status_name(j.status);
    o["pid"]              = j.pid;
    o["start_token"]      = j.start_token;
    o["launch_failures"]  = j.launch_failures;
    o["no_checkpoint_launches"] = j.no_checkpoint_launches;
    o["last_launch_ms"]   = j.last_launch_ms;
    o["failure_reason"]   = j.failure_reason;
    o["load_state_from"]  = j.load_state_from;
    o["load_state"]       = j.load_state;
    o["load_state_step"]  = j.load_state_step;
    // Re-emit anything a newer daemon wrote that this build does not model (see
    // parse_job). Never allowed to overwrite a key we DO own: this build's value is
    // authoritative for its own fields.
    if (!j.extra_fields.empty()) {
        const json extras = json::parse(j.extra_fields, nullptr, /*allow_exceptions=*/false);
        if (extras.is_object())
            for (const auto& [key, value] : extras.items())
                if (!o.contains(key)) o[key] = value;
    }
    return o.dump(2);
}

job parse_job(std::string_view text) {
    const json o = json::parse(text);

    // Version gate FIRST: refuse to interpret a file from a newer daemon rather
    // than silently defaulting its fields. ABSENT defaults to 1 (a pre-versioning
    // file is v1-shaped anyway) — but PRESENT-and-malformed does not: get_or
    // substitutes the default on a type mismatch, so "2" or 2.0 read as v1, the
    // gate passed, every unrecognized field defaulted, and the next put() rewrote
    // the file AS v1. That is the silent mangling job_schema_error exists to
    // prevent, so a present-but-non-integer version is an unreadable file.
    if (const auto it = o.find("schema_version");
        it != o.end() && !it->is_null() && !it->is_number_integer())
        throw job_schema_error{-1};   // -1: "present but not a version number"
    const int version = get_or(o, "schema_version", job_schema_version);
    if (version > job_schema_version) throw job_schema_error{version};

    job j;
    j.schema_version   = version;
    j.coastd_version   = get_or(o, "coastd_version", std::string{});
    j.id               = get_or(o, "id", std::string{});
    j.submitted_at     = get_or(o, "submitted_at", std::string{});
    j.user_message     = get_or(o, "user_message", std::string{});
    j.ref_symbolic     = get_or(o, "ref_symbolic", std::string{});
    j.ref_sha          = get_or(o, "ref_sha", std::string{});
    j.run_dir          = get_or(o, "run_dir", std::string{});
    j.steps_target     = get_or(o, "steps_target", std::uint64_t{0});
    j.steps_done       = get_or(o, "steps_done", std::uint64_t{0});
    j.checkpoint       = get_or(o, "checkpoint", std::string{});
    j.priority         = get_or(o, "priority", 0);
    j.cancel_requested = get_or(o, "cancel_requested", false);
    j.status           = status_from(get_or(o, "status", std::string{"queued"}));
    j.pid              = get_or(o, "pid", std::int64_t{0});
    j.start_token      = get_or(o, "start_token", std::string{});
    j.launch_failures  = get_or(o, "launch_failures", 0);
    // Both guards below are WRITTEN by dump_job, so a file always carries them;
    // failing to read them back reset the guards on every daemon restart (H1).
    // last_launch_ms is epoch ms (wall clock), so it survives a restart honestly —
    // it is not a monotonic-clock value that would be meaningless in a new process.
    j.no_checkpoint_launches = get_or(o, "no_checkpoint_launches", 0);
    j.last_launch_ms   = get_or(o, "last_launch_ms", std::uint64_t{0});
    j.failure_reason   = get_or(o, "failure_reason", std::string{});
    j.load_state_from  = get_or(o, "load_state_from", std::string{});
    j.load_state       = get_or(o, "load_state", std::string{});
    j.load_state_step  = get_or(o, "load_state_step", std::uint64_t{0});

    // Preserve every key this build does not know about. The header promises an
    // ADDITIVE schema, but a parse that rebuilds the job from a fixed key list and
    // a dump that writes only those keys silently DELETES anything else on the
    // next status update — so a field added by a newer daemon at the same
    // schema_version is destroyed by an older one in a mixed fleet or after a
    // one-binary rollback. Carrying the extras through makes the additive promise
    // true rather than aspirational.
    json extras = json::object();
    for (const auto& [key, value] : o.items())
        if (!known_job_keys().contains(key)) extras[key] = value;
    if (!extras.empty()) j.extra_fields = extras.dump();
    return j;
}

} // namespace coastd
