// -----------------------------------------------------------------------------
// coastd — coastd.json loader.
//
// Extracted from main.cpp so the parsing (type checks, range clamps, the
// <data-root> token, the four directory-root keys) is unit-testable without
// linking the whole entry point. See config_load.h for the precedence and
// robustness contract.
// -----------------------------------------------------------------------------

#include "daemon/config_load.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>

#include <nlohmann/json.hpp>

#include "diag/diag.h"

namespace coastd {

bool load_config_file(const std::filesystem::path& path,
                      const std::filesystem::path& data_root,
                      daemon_config& cfg,
                      std::string& err) {
    std::ifstream in(path);
    if (!in) { err = "cannot open config " + path.string(); return false; }
    nlohmann::json j;
    try { in >> j; }
    catch (const std::exception& e) { err = "config " + path.string() + ": " + e.what(); return false; }
    if (!j.is_object()) { err = "config " + path.string() + ": top-level JSON must be an object"; return false; }

    auto warn_type = [&](const char* key) {
        COAST_WARN("coastd", "config key '{}' has an unexpected type; ignoring", key);
    };
    // Range-checked integer knob: a wrong-typed OR out-of-range value is skipped
    // with a warning (never fatal), keeping the compiled-in default. This mirrors
    // the CLI's parse_* bounds so the config file cannot set a value the flags
    // reject — e.g. scan_period_ms <= 0 (which makes the core loop's pop_until
    // deadline always past -> a 100% CPU busy-spin), a negative term_grace
    // (instant SIGKILL escalation), or negative slots (nothing ever schedules).
    auto get_int = [&](const char* key, long long lo, long long hi, auto&& apply) {
        if (auto it = j.find(key); it != j.end()) {
            if (!(it->is_number_integer() || it->is_number_unsigned())) { warn_type(key); return; }
            const long long v = it->template get<long long>();
            if (v < lo || v > hi) {
                COAST_WARN("coastd", "config key '{}' value {} out of range [{}..{}]; ignoring",
                           key, v, lo, hi);
                return;
            }
            apply(v);
        }
    };

    // Path-valued key. Substitutes the literal token "<data-root>" with the
    // resolved data root (see config_load.h) and skips an empty result rather
    // than silently setting a root to "". Taken verbatim otherwise — no implicit
    // absolutization, matching how --data-root and the derived defaults behave; a
    // relative root resolves against the daemon's working directory, so operators
    // should give an absolute path or use the <data-root> token.
    auto get_path = [&](const char* key, auto&& apply) {
        auto it = j.find(key);
        if (it == j.end()) return;
        if (!it->is_string()) { warn_type(key); return; }
        std::string s = it->template get<std::string>();
        const std::string tok = "<data-root>";
        for (std::size_t p = s.find(tok); p != std::string::npos; p = s.find(tok, p)) {
            s.replace(p, tok.size(), data_root.string());
            p += data_root.string().size();
        }
        if (s.empty()) {
            COAST_WARN("coastd", "config key '{}' is empty after substitution; ignoring", key);
            return;
        }
        apply(std::filesystem::path(std::move(s)));
    };

    // Directory roots. Each overrides the data_root-derived default the caller
    // seeded; --repo still overrides repo_root afterward (CLI beats file).
    get_path("state_root",     [&](std::filesystem::path p){ cfg.state_root     = std::move(p); });
    get_path("runs_root",      [&](std::filesystem::path p){ cfg.runs_root      = std::move(p); });
    get_path("worktrees_root", [&](std::filesystem::path p){ cfg.worktrees_root = std::move(p); });
    get_path("repo_root",      [&](std::filesystem::path p){ cfg.repo_root      = std::move(p); });

    // Shared dataset directory linked into each run dir as `data` (types.h). Like
    // the roots above it takes the <data-root> token and is stored verbatim; the
    // link site absolutizes it. Unset leaves the mechanism off.
    get_path("dataset_base",   [&](std::filesystem::path p){ cfg.dataset_base   = std::move(p); });

    get_int("slots",               1, 4096,          [&](long long v){ cfg.slots = static_cast<int>(v); });
    get_int("worker_pool_size",    1, 4096,          [&](long long v){ cfg.worker_pool_size = static_cast<int>(v); });
    get_int("build_concurrency",   1, 4096,          [&](long long v){ cfg.build_concurrency = static_cast<int>(v); });
    get_int("build_jobs",          0, 4096,          [&](long long v){ cfg.build_jobs = static_cast<int>(v); });  // 0 = auto
    get_int("max_launch_failures", 1, 1000,          [&](long long v){ cfg.max_launch_failures = static_cast<int>(v); });
    get_int("max_build_failures",  1, 1000,          [&](long long v){ cfg.max_build_failures = static_cast<int>(v); });
    get_int("checkpoint_interval", 1, 1'000'000'000'000LL, [&](long long v){ cfg.checkpoint_interval = static_cast<std::uint64_t>(v); });
    get_int("tcp_port",            1, 65535,         [&](long long v){ cfg.tcp_port = static_cast<std::uint16_t>(v); });
    get_int("udp_port",            1, 65535,         [&](long long v){ cfg.udp_port = static_cast<std::uint16_t>(v); });
    get_int("worktree_cooloff_seconds", 1, 86400,    [&](long long v){ cfg.worktree_cooloff = std::chrono::seconds(v); });
    get_int("scan_period_ms",      1, 3'600'000,     [&](long long v){ cfg.scan_period = std::chrono::milliseconds(v); });
    // Log housekeeping. 0 disables a stage (no compression / never delete), so
    // the ranges start at 0 unlike the timeout knobs above.
    get_int("log_compress_after_seconds", 0, 3'650LL*86400, [&](long long v){ cfg.log_compress_after = std::chrono::seconds(v); });
    get_int("log_delete_after_seconds",   0, 3'650LL*86400, [&](long long v){ cfg.log_delete_after = std::chrono::seconds(v); });
    get_int("log_sweep_period_seconds",   60, 86400,        [&](long long v){ cfg.log_sweep_period = std::chrono::seconds(v); });
    get_int("build_log_max_bytes", 1<<20, 1LL<<34,          [&](long long v){ cfg.build_log_max_bytes = static_cast<std::size_t>(v); });
    get_int("build_timeout_seconds",1, 86400,        [&](long long v){ cfg.build_timeout = std::chrono::seconds(v); });
    get_int("git_op_timeout_seconds",1, 86400,       [&](long long v){ cfg.git_op_timeout = std::chrono::seconds(v); });
    get_int("audit_git_timeout_seconds",1, 86400,    [&](long long v){ cfg.audit_git_timeout = std::chrono::seconds(v); });
    get_int("term_grace_seconds",  1, 86400,         [&](long long v){ cfg.term_grace = std::chrono::seconds(v); });
    get_int("fetch_timeout_seconds",1, 86400,        [&](long long v){ cfg.fetch_timeout = std::chrono::seconds(v); });
    // 0 is meaningful here (fetch on EVERY resolve, no coalescing), so the range
    // starts at 0 unlike the timeout knobs above.
    get_int("fetch_min_interval_seconds", 0, 86400,  [&](long long v){ cfg.fetch_min_interval = std::chrono::seconds(v); });

    // Relaunch backoff after a launch that made no durable progress (types.h):
    // base * 2^(failures-1), capped at max. base > max is not rejected — reconcile
    // clamps the product to max, so it just degrades to a constant backoff at max
    // — but the caller warns, since it almost always means a typo.
    get_int("relaunch_backoff_base_seconds", 1, 86400, [&](long long v){ cfg.relaunch_backoff_base = std::chrono::seconds(v); });
    get_int("relaunch_backoff_max_seconds",  1, 86400, [&](long long v){ cfg.relaunch_backoff_max  = std::chrono::seconds(v); });

    // Disk-pressure floor in bytes; 0 disables the check. Lower bound is 0, not 1,
    // precisely so an operator can turn it off (a tmpfs test rig, a machine where
    // the roots live on a quota-managed share statvfs reports uselessly for).
    get_int("min_free_disk_bytes", 0, 1LL << 50, [&](long long v){ cfg.min_free_disk_bytes = static_cast<std::uint64_t>(v); });

    if (auto it = j.find("fetch_enabled"); it != j.end()) {
        if (it->is_boolean()) cfg.fetch_enabled = it->get<bool>();
        else                  warn_type("fetch_enabled");
    }
    if (auto it = j.find("git_remote"); it != j.end()) {
        if (it->is_string()) cfg.git_remote = it->get<std::string>();
        else                 warn_type("git_remote");
    }

    // UDP discovery responder. See daemon_config for why this has knobs at all:
    // it is an unauthenticated reflector, and the only channel that can reach a
    // host which never contacted this daemon.
    if (auto it = j.find("discovery_enabled"); it != j.end()) {
        if (it->is_boolean()) cfg.discovery_enabled = it->get<bool>();
        else                  warn_type("discovery_enabled");
    }
    if (auto it = j.find("discovery_bind"); it != j.end()) {
        if (it->is_string()) cfg.discovery_bind = it->get<std::string>();
        else                 warn_type("discovery_bind");
    }
    get_int("discovery_max_replies_per_sec", 1, 1'000'000,
            [&](long long v){ cfg.discovery_max_replies_per_sec = static_cast<int>(v); });
    get_int("discovery_max_replies_per_source_per_sec", 1, 1'000'000,
            [&](long long v){ cfg.discovery_max_replies_per_source_per_sec = static_cast<int>(v); });

    // Client-edge (comms) bounds.
    get_int("comms_max_connections",   1, 1'048'576,       [&](long long v){ cfg.max_connections   = static_cast<std::size_t>(v); });
    get_int("comms_max_request_bytes", 1, 1LL << 30,       [&](long long v){ cfg.max_request_bytes = static_cast<std::size_t>(v); });
    get_int("comms_max_config_bytes",  1, 1LL << 30,       [&](long long v){ cfg.max_config_bytes  = static_cast<std::size_t>(v); });
    get_int("comms_first_request_timeout_seconds", 1, 86400, [&](long long v){ cfg.first_request_timeout = std::chrono::seconds(v); });
    get_int("comms_line_assembly_timeout_seconds", 1, 86400, [&](long long v){ cfg.line_assembly_timeout = std::chrono::seconds(v); });
    get_int("comms_send_timeout_seconds",          1, 86400, [&](long long v){ cfg.send_timeout          = std::chrono::seconds(v); });

    if (auto it = j.find("build_cmake_args"); it != j.end()) {
        if (it->is_array()) {
            cfg.build_cmake_args.clear();
            for (const auto& e : *it) if (e.is_string()) cfg.build_cmake_args.push_back(e.get<std::string>());
        } else {
            warn_type("build_cmake_args");
        }
    }
    return true;
}

} // namespace coastd
