#ifndef COAST_COASTD_PROTOCOL_H
#define COAST_COASTD_PROTOCOL_H

// -----------------------------------------------------------------------------
// coastd wire protocol — the single source of truth for the client<->daemon
// contract. HEADER-ONLY and deliberately dependency-light: only <nlohmann/json>
// and the standard library. It is shared verbatim by the server side (the
// comms_server inside coastd_core) and the client side (coastd_client, which
// frontends link WITHOUT coast_lib or coastd_core). A frontend that wants to
// talk to coastd pays for sockets + json and nothing else.
//
// Transport is JSONL: one minified JSON object per line, terminated by '\n'.
// Every request is  {"cmd":"<name>", ...args}  and every reply is an envelope
//   {"ok":true,  ...data}   |   {"ok":false, "error":"<human message>"}.
// The line orientation makes the whole surface `printf ... | nc` -speakable for
// ad-hoc use and scripted B-tier tests.
//
// The DTOs here (job_summary, machine_summary) are the CLIENT's view — a
// deliberately reduced projection of the internal `job`. They intentionally omit
// server-only fields (pid, start_token, run_dir, launch_failures): the wire is a
// status surface, not a state dump. The server builds a job_summary from a job;
// the client only ever deserializes one.
//
// CONFIG TRANSFER — THE CLIENT SENDS CONTENT, NEVER A PATH.
// `submit` carries the config's BYTES in `config_text` (plus `config_name`, the
// basename, purely for the job-id mnemonic). There is deliberately NO "path on
// the daemon host" field, because the client and the daemon are, in the general
// case, DIFFERENT MACHINES: a path is meaningless across that boundary, and — far
// worse than meaningless — a path that happens to exist on both hosts with
// different content submits a job that silently runs the WRONG config. Nothing on
// either side could detect that. A GUI has no path to send at all when the user
// has edited a config in-place. So the path form is not offered, not even as a
// fallback: an option to reintroduce a defect is the defect.
//
// The bytes travel as a JSON *string*, not as a nested JSON object. Two reasons:
//   1. Byte fidelity end-to-end. The daemon archives exactly what it received as
//      <run_dir>/config.submitted.json (the provenance record). Note this is NOT
//      the same file as <run_dir>/config.json, which coastd deliberately rewrites
//      to inject snapshot_dir and therefore re-serializes (keys reordered by
//      nlohmann's std::map, indentation normalized). Carrying a nested object here
//      would apply that same lossy reserialization to the wire as well, leaving no
//      copy of the user's actual file anywhere.
//   2. String escaping is lossless and JSONL-safe. Newlines become \n, so the
//      request stays on one line while indentation and layout survive exactly.
//      MEASURED on a real 8 KB config (243 newlines, 4312 indent spaces):
//      single-line yes, round-trip byte-exact, ~8.4% escape overhead — versus
//      ~33% for base64, which would buy nothing we do not already have.
// The one adjacent hazard: nlohmann's dump() THROWS on invalid UTF-8 inside a
// string. Such a file is not valid JSON either, so the outcome is a rejection
// regardless; clients pre-flight-parse the config locally so the user sees a
// clean "not valid JSON" instead of a raw type_error.316. Clients never send a
// re-dump of what they parsed — always the original bytes.
// -----------------------------------------------------------------------------

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "daemon/net_defaults.h"

namespace coastd::proto {

// Default ports — single-sourced from net_defaults.h (shared with daemon_config).
inline constexpr std::uint16_t default_tcp_port = coastd::default_tcp_port;
inline constexpr std::uint16_t default_udp_port = coastd::default_udp_port;

// The single-datagram UDP discovery probe payload (Model P: client asks, quiet
// daemon answers). Any datagram whose text starts with this token is a probe.
inline constexpr std::string_view discovery_probe = "coastd-probe-v1";

// ---- protocol version -------------------------------------------------------
//
// The wire is ADDITIVE by construction — every field is read with j.value() and a
// default, so an unknown field is ignored and a missing one takes its default. A
// client and a daemon that differ only by added fields interoperate in BOTH
// directions with no negotiation, which is why this went unversioned for so long.
//
// What that cannot express is a BREAKING change: a field whose meaning or units
// change, a command whose semantics shift, a reply that stops carrying something.
// Unversioned, those surface as a wrong answer or an opaque error string — the
// `config` -> `config_text` migration in comms_server is exactly this case, and it
// needed a hand-written per-field message to be diagnosable at all.
//
// So: every reply carries `proto_version`, and machine_info/discovery advertise it
// where a client can see it BEFORE committing work. The rule for changing it:
//
//   * ADD a field, add a command, add an enum token   -> version UNCHANGED
//   * change a field's type/meaning/units, remove a
//     field a peer relies on, change a command's
//     semantics                                       -> BUMP
//
// A client compares and decides; the daemon never refuses on version alone, since
// a mismatch in the additive direction is harmless and refusing would be the more
// disruptive answer. `min_compatible` is the oldest peer this build still expects
// to understand it — bump that only when dropping compatibility deliberately.
inline constexpr int proto_version   = 1;
inline constexpr int proto_min_compatible = 1;

// Human-readable summary of a version difference, for a client to log or show.
// Empty when the peer is compatible.
inline std::string version_complaint(int peer_version) {
    if (peer_version == 0)
        return "daemon predates protocol versioning (pre-v1); fields added since may be missing";
    if (peer_version > proto_version)
        return "daemon speaks protocol v" + std::to_string(peer_version) + ", this client is v" +
               std::to_string(proto_version) + " — update the client if something looks wrong";
    if (peer_version < proto_min_compatible)
        return "daemon speaks protocol v" + std::to_string(peer_version) +
               ", older than this client's minimum v" + std::to_string(proto_min_compatible);
    return {};
}

// ---- command names (stable wire tokens) ------------------------------------
namespace cmd {
inline constexpr std::string_view submit       = "submit";
inline constexpr std::string_view list         = "list";
inline constexpr std::string_view cancel       = "cancel";
inline constexpr std::string_view reorder      = "reorder";
inline constexpr std::string_view machine_info = "machine_info";
// {"paths":[rel…]} -> {"sha256":{rel: hex}}: the SHA-256 of files under
// dataset_base, computed on the daemon's filesystem (daemon/hash_inputs.h).
inline constexpr std::string_view hash_inputs  = "hash_inputs";
} // namespace cmd

// ---- DTOs ------------------------------------------------------------------

// A job as seen by a client. `status` and `bucket` are plain strings (the server
// maps its enums to the stable tokens) so this header stays free of the internal
// job vocabulary.
struct job_summary {
    std::string   id;
    std::string   ref_symbolic;
    std::string   submitted_at;
    std::string   user_message;
    std::string   status;          // job_status token: queued/resolving/…/quarantine
    std::string   bucket;          // bucket token: active/done/failed/cancelled/quarantine
    std::uint64_t steps_done   = 0;  // DURABLE progress: newest checkpoint / clean manifest
    std::uint64_t live_step    = 0;  // live step from the running child's log tail; 0 if not
                                     // running or none seen. Display-only; NOT resumable —
                                     // a client shows it so a running job advances before its
                                     // first checkpoint, but steps_done stays the durable point.
    std::uint64_t steps_target = 0;
    std::string   checkpoint;      // newest checkpoint basename, rel run_dir ("" = none)
    int           priority     = 0;
    std::string   failure_reason;  // "" unless a terminal failure was recorded
    // A cancel has been accepted but the job has not settled yet: the daemon has
    // SIGTERMed the child and is inside the term_grace, letting it write a final
    // checkpoint. Carried as a FLAG rather than as a `cancelling` status token on
    // purpose — `status` is the job's real, persisted status, and there is no
    // `cancelling` state in the store (reconcile derives the kill from this flag).
    // A wire-only token would invite clients to match on a value the store never
    // holds. Additive: an older client ignores the key and sees `running`, exactly
    // as before.
    bool          cancel_requested = false;
};

// Static machine description (served straight from daemon_config; no core hop).
struct machine_summary {
    // Protocol version this daemon speaks (see proto_version). Advertised here and
    // in the discovery datagram so a client can check BEFORE submitting work.
    // Defaults to 0 when absent, which identifies a pre-versioning daemon.
    int           proto_version = 0;
    std::string   hostname;
    std::string   os;
    std::string   coastd_version;
    int           slots    = 1;
    std::string   runs_root;   // where results/logs land (well-known results location)
    std::string   repo_root;   // the git repo coast is built from
    std::uint16_t tcp_port = 0; // the daemon's command port, so a discovered daemon is connectable
    // True iff a submitted config's conventional RELATIVE dataset paths
    // ("data/mnist/…") will resolve on this machine: the operator has set
    // dataset_base and its target exists, so the daemon drops a `data` symlink in
    // every run dir (children run with cwd = run dir). This is the "well-known
    // data location" a client actually needs — not the path itself, which the
    // client cannot use for anything (the config travels as content and runs on
    // the daemon's machine), but the yes/no that lets a client warn BEFORE submit
    // instead of letting the job fail minutes later at dataset load.
    bool          data_link_ok = false;
    // The target of every run dir's `data` link: dataset_base made absolute against
    // the daemon's working directory, exactly as the link site writes it ("" if
    // unset). Not something a client opens — the client may be another
    // machine — but the string it compares a run dir's `data` link against, to
    // prove the run read the directory whose files `hash_inputs` vouched for.
    std::string   dataset_base;
    // Client-populated ONLY: the source address a discovery reply arrived from —
    // the actually-connectable IP (the daemon cannot know how a client sees it, so
    // it is never sent on the wire; discover() fills it from recvfrom()).
    std::string   address;
};

inline void to_json(nlohmann::json& j, const job_summary& s) {
    j = nlohmann::json{
        {"id", s.id}, {"ref", s.ref_symbolic}, {"submitted_at", s.submitted_at},
        {"user_message", s.user_message}, {"status", s.status}, {"bucket", s.bucket},
        {"steps_done", s.steps_done}, {"live_step", s.live_step},
        {"steps_target", s.steps_target},
        {"checkpoint", s.checkpoint}, {"priority", s.priority},
        {"failure_reason", s.failure_reason},
        {"cancel_requested", s.cancel_requested}};
}

inline void from_json(const nlohmann::json& j, job_summary& s) {
    s.id            = j.value("id", std::string{});
    s.ref_symbolic  = j.value("ref", std::string{});
    s.submitted_at  = j.value("submitted_at", std::string{});
    s.user_message  = j.value("user_message", std::string{});
    s.status        = j.value("status", std::string{});
    s.bucket        = j.value("bucket", std::string{});
    s.steps_done    = j.value("steps_done", std::uint64_t{0});
    s.live_step     = j.value("live_step", std::uint64_t{0});
    s.steps_target  = j.value("steps_target", std::uint64_t{0});
    s.checkpoint    = j.value("checkpoint", std::string{});
    s.priority      = j.value("priority", 0);
    s.failure_reason= j.value("failure_reason", std::string{});
    s.cancel_requested = j.value("cancel_requested", false);
}

inline void to_json(nlohmann::json& j, const machine_summary& m) {
    // `address` is intentionally NOT serialized — it is client-side only.
    j = nlohmann::json{
        {"hostname", m.hostname}, {"os", m.os}, {"coastd_version", m.coastd_version},
        {"proto_version", m.proto_version},
        {"slots", m.slots}, {"runs_root", m.runs_root}, {"repo_root", m.repo_root},
        {"tcp_port", m.tcp_port}, {"data_link_ok", m.data_link_ok},
        {"dataset_base", m.dataset_base}};
}

inline void from_json(const nlohmann::json& j, machine_summary& m) {
    m.hostname       = j.value("hostname", std::string{});
    m.os             = j.value("os", std::string{});
    m.coastd_version = j.value("coastd_version", std::string{});
    m.proto_version  = j.value("proto_version", 0);   // 0 = pre-versioning daemon
    m.slots          = j.value("slots", 1);
    m.runs_root      = j.value("runs_root", std::string{});
    m.repo_root      = j.value("repo_root", std::string{});
    m.tcp_port       = j.value("tcp_port", std::uint16_t{0});
    m.data_link_ok   = j.value("data_link_ok", false);
    m.dataset_base   = j.value("dataset_base", std::string{});
}

// ---- envelope helpers ------------------------------------------------------

// Success envelope: {"ok":true, ...data}. `data` is merged in at the top level so
// e.g. list replies read {"ok":true,"jobs":[…]} and submit {"ok":true,"id":"…"}.
inline nlohmann::json ok(nlohmann::json data = nlohmann::json::object()) {
    nlohmann::json e = nlohmann::json::object();
    e["ok"] = true;
    // Stamped on EVERY reply, not only machine_info: a client that never calls
    // machine_info (the console's one-shot submit, a `printf | nc` script) still
    // gets to see what it is talking to. `data` is merged after, so a handler that
    // deliberately reports a different version can still override it.
    e["proto_version"] = proto_version;
    for (auto it = data.begin(); it != data.end(); ++it) e[it.key()] = it.value();
    return e;
}

// Failure envelope: {"ok":false, "error":msg}. Used for every rejection and every
// caught handler exception — a client error never becomes a daemon crash (A10).
inline nlohmann::json err(std::string_view msg) {
    // Versioned too — a rejection is exactly when a client most needs to know
    // whether it is simply talking to the wrong vintage of daemon.
    return nlohmann::json{{"ok", false}, {"error", std::string(msg)},
                          {"proto_version", proto_version}};
}

// A parsed request line: the command plus the whole object as its argument bag.
struct request {
    std::string    cmd;
    nlohmann::json args;   // the full request object (cmd included); read fields via .value()
};

// Maximum JSON nesting depth accepted at the transport edge. nlohmann's DOM
// parser recurses once per nesting level, so an adversarial deeply-nested line
// (`[[[[…]]]]`, still well within the byte cap) could overflow the stack — an
// uncatchable SIGSEGV that the handler's try/catch cannot defend against. We
// reject beyond this depth with an O(n) pre-scan before the parser ever recurses.
// 64 is far deeper than any legitimate coastd request (which nests ≤3).
inline constexpr int max_json_depth = 64;

// O(n) structural-depth pre-scan. String/escape aware so brackets INSIDE a JSON
// string are not miscounted. Returns false iff nesting exceeds max_json_depth.
inline bool json_depth_ok(std::string_view s) {
    int depth = 0, max_depth = 0;
    bool in_str = false, esc = false;
    for (const char c : s) {
        if (in_str) {
            if (esc)              esc = false;
            else if (c == '\\')   esc = true;
            else if (c == '"')    in_str = false;
            continue;
        }
        if (c == '"')                    in_str = true;
        else if (c == '{' || c == '[') { if (++depth > max_depth) max_depth = depth; }
        else if (c == '}' || c == ']') { if (depth > 0) --depth; }
    }
    return max_depth <= max_json_depth;
}

// Tolerant parse of ONE JSONL line. nullopt on over-deep nesting, malformed JSON,
// or a missing/blank "cmd" — the caller answers with err(), never throws (and now
// never stack-overflows) at the transport edge.
inline std::optional<request> parse_line(std::string_view line) {
    if (!json_depth_ok(line)) return std::nullopt;
    nlohmann::json j = nlohmann::json::parse(line, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) return std::nullopt;
    // j.value("cmd", ...) THROWS type_error.302 when "cmd" is PRESENT but not a
    // string ({"cmd":1}) — value() only substitutes the default for an ABSENT key.
    // This function's documented contract is that it never throws; today the
    // daemon's blanket handler catches it and answers "internal error", which is
    // both the wrong message and a trap for any future caller that trusts the
    // contract. Test the type first.
    const auto it = j.find("cmd");
    if (it == j.end() || !it->is_string()) return std::nullopt;
    const std::string c = it->get<std::string>();
    if (c.empty()) return std::nullopt;
    return request{c, std::move(j)};
}

} // namespace coastd::proto

#endif // COAST_COASTD_PROTOCOL_H
