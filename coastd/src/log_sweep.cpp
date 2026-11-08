#include "daemon/log_sweep.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "daemon/proc_util.h"
#include "diag/diag.h"

namespace coastd {

namespace fs = std::filesystem;

namespace {

// Seconds since a file was last written, or nullopt if it cannot be stated.
// file_clock -> system_clock so the comparison is against wall time, matching the
// operator-facing "older than 7 days" meaning of the config knobs.
std::optional<std::chrono::seconds> idle_for(const fs::path& p) {
    std::error_code ec;
    const auto ft = fs::last_write_time(p, ec);
    if (ec) return std::nullopt;
    const auto when = std::chrono::file_clock::to_sys(ft);
    const auto age  = std::chrono::system_clock::now() - when;
    // A file stamped in the FUTURE (clock step, a restored archive) has a negative
    // age. Treat it as brand new rather than as maximally old: the failure mode of
    // the other reading is deleting a log that was just written.
    if (age < std::chrono::system_clock::duration::zero()) return std::chrono::seconds{0};
    return std::chrono::duration_cast<std::chrono::seconds>(age);
}

// gzip one file in place. gzip removes the original on success, so afterwards
// exactly one of <p> / <p>.gz exists. Bounded, like every other subprocess here:
// a hung gzip must not park the worker thread that called us.
bool gzip_file(const fs::path& p) {
    // -n: no timestamp/name in the header, so the .gz is reproducible.
    // -q: no warnings on stderr for the ordinary cases.
    const auto r = run_capture("gzip", {"-9", "-n", "-q", p.string()}, {},
                               std::chrono::seconds(300));
    if (!r.spawned) {
        COAST_WARN("coastd.logsweep", "gzip is not available; leaving {} uncompressed",
                   p.string());
        return false;
    }
    if (r.exit_code != 0) {
        COAST_WARN("coastd.logsweep", "gzip {} failed (exit {}); leaving it uncompressed",
                   p.string(), r.exit_code);
        return false;
    }
    return true;
}

// Apply the policy to one file. `is_gz` selects the delete stage vs the compress
// stage; both are no-ops when their threshold is 0 (disabled).
void apply_policy(const fs::path& p, bool is_gz, const daemon_config& cfg, sweep_stats& st) {
    const auto idle = idle_for(p);
    if (!idle) { ++st.failed; return; }

    if (is_gz) {
        if (cfg.log_delete_after.count() <= 0) return;
        if (*idle < cfg.log_delete_after) return;
        std::error_code ec;
        fs::remove(p, ec);
        if (ec) { COAST_WARN("coastd.logsweep", "could not remove {}: {}", p.string(), ec.message());
                  ++st.failed; return; }
        ++st.deleted;
        return;
    }

    if (cfg.log_compress_after.count() <= 0) return;
    if (*idle < cfg.log_compress_after) return;   // still recent — possibly still being written
    if (gzip_file(p)) ++st.compressed; else ++st.failed;
}

// Every *.log / *.log.gz directly inside `dir`.
void sweep_dir(const fs::path& dir, const daemon_config& cfg, sweep_stats& st) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return;
    fs::directory_iterator it(dir, ec);
    const fs::directory_iterator end;
    for (; !ec && it != end; it.increment(ec)) {
        std::error_code eec;
        if (!it->is_regular_file(eec)) continue;
        const std::string name = it->path().filename().string();
        if (name.ends_with(".log"))         apply_policy(it->path(), false, cfg, st);
        else if (name.ends_with(".log.gz")) apply_policy(it->path(), true,  cfg, st);
    }
    if (ec)
        COAST_WARN("coastd.logsweep", "could not fully scan {}: {}", dir.string(), ec.message());
}

} // namespace

sweep_stats sweep_logs(const daemon_config& cfg) noexcept {
    sweep_stats st;
    try {
        // Run logs: <runs_root>/<job id>/logs/*.log
        std::error_code ec;
        if (fs::is_directory(cfg.runs_root, ec)) {
            fs::directory_iterator it(cfg.runs_root, ec);
            const fs::directory_iterator end;
            for (; !ec && it != end; it.increment(ec)) {
                std::error_code eec;
                if (!it->is_directory(eec)) continue;
                sweep_dir(it->path() / "logs", cfg, st);
            }
        }
        // Build logs: <worktrees_root>/<sha>/coastd-build.log
        ec.clear();
        if (fs::is_directory(cfg.worktrees_root, ec)) {
            fs::directory_iterator it(cfg.worktrees_root, ec);
            const fs::directory_iterator end;
            for (; !ec && it != end; it.increment(ec)) {
                std::error_code eec;
                if (!it->is_directory(eec)) continue;
                sweep_dir(it->path(), cfg, st);
            }
        }
    } catch (const std::exception& e) {
        // noexcept by contract: this runs on a worker thread whose only job is
        // housekeeping, and no log-tidying failure may propagate into the daemon.
        COAST_ERROR("coastd.logsweep", "log sweep aborted: {}", e.what());
    } catch (...) {
        COAST_ERROR("coastd.logsweep", "log sweep aborted (non-exception)");
    }
    return st;
}

void append_capped(const fs::path& log, std::string_view header,
                   const std::string& body, std::size_t max_bytes) noexcept {
    try {
        std::error_code ec;
        const auto sz = fs::file_size(log, ec);
        if (!ec && max_bytes > 0 && sz > max_bytes) {
            // Restart rather than grow. Truncating in place keeps the path stable
            // (the builder appends to it across steps) and needs no rename dance.
            std::ofstream reset(log, std::ios::trunc);
            reset << "===== log restarted: exceeded " << (max_bytes >> 20)
                  << " MiB; earlier output discarded =====\n";
            COAST_WARN("coastd.logsweep", "{} exceeded {} MiB and was restarted",
                       log.string(), max_bytes >> 20);
        }
        std::ofstream out(log, std::ios::app);
        out << "\n===== " << header << " =====\n" << body;
    } catch (...) {
        // A diagnostic that cannot be written must never fail a build.
    }
}

} // namespace coastd
