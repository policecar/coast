#ifndef COAST_DIAG_PROGRESS_H
#define COAST_DIAG_PROGRESS_H

// -----------------------------------------------------------------------------
// coast::diag::progress — structured run-progress contract
// -----------------------------------------------------------------------------
//
// Machine-readable progress for a headless COAST run. This is DELIBERATELY
// separate from the diagnostic logger (<diag/diag.h>): progress is not a log
// record stream, it is a small state feed that an external supervisor consumes.
//
// The problem this solves: cimc-coast-consim launches `coast` child processes
// and drives its progress table by sscanf-ing run_headless's human-readable
// prints ("from step A to B", "Steps: N", "Total steps: N"). Nothing tied the
// emitter and the parser together, so rewording a print silently broke the bar
// (see sim_console_api.cpp:parse_log_progress and its NOTE).
//
// The fix: ONE line grammar defined here, with both format_progress() (emitter,
// in coast) and parse_progress() (consumer, in consim) living in this single
// header. Both ends #include this file, so a field rename is a compile-time
// change on BOTH sides and can never drift into a silent runtime break.
//
// The structured line is emitted ALONGSIDE the existing human-readable output,
// not instead of it — humans still read headless stdout. It is one extra line
// per progress point, prefixed with a stable sentinel so it is trivially
// greppable and unambiguous.
//
// Line grammar (single line, LF-terminated, written to stdout):
//
//   @coast/progress v1 phase=<phase> step=<u64> start=<u64> target=<u64> sps=<f64> logged=<u64>
//
//   phase   : start | step | checkpoint | done | failed
//   step    : current completed step count
//   start   : step index at run start (checkpoint/progress base; 0 unless resumed)
//   target  : target step, or 0 for a run with no step limit (--unlimited).
//             This 0 is a WIRE sentinel retained for v1 compatibility with
//             existing parsers (consim, sim_console::parse_log_progress); the
//             CLI itself no longer has one — `-n 0` is rejected and an unbounded
//             run must say --unlimited. If this needs disambiguating on the wire,
//             append an `unlimited=<bool>` token rather than repurposing target:
//             the forward-compat rule below makes added tokens safe, changing an
//             existing field's meaning is not.
//   sps     : measured steps/second (0 if not yet known)
//   logged  : buffer-log frames written so far (0 if buffer logging is off)
//
// Forward-compat rule: a parser MUST ignore unknown trailing key=value tokens,
// so new fields can be appended without breaking older consumers.
//
// Portability: renders with std::format ({:.3f} float formatting is libc++ 17+ /
// GCC 13+, both under the supported macOS toolchains). Both integers and the one
// float field parse via std::from_chars — locale-independent, matching the
// emitter (strtod would honor LC_NUMERIC). Floating-point from_chars is
// GCC 11+ / libc++ 20+, both under the supported toolchains.
// -----------------------------------------------------------------------------

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <format>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace coast::diag {

inline constexpr std::string_view progress_sentinel = "@coast/progress";
inline constexpr int              progress_version  = 1;

enum class run_phase : std::uint8_t {
    start,       // run is starting; step/start/target populated
    step,        // periodic progress tick
    checkpoint,  // a checkpoint was written at `step`
    done,        // run finished cleanly
    failed,      // run exits 1 (fault, failed log/save, exception) at `step`
};

struct progress {
    run_phase     phase         = run_phase::step;
    std::uint64_t step          = 0;
    std::uint64_t start         = 0;
    std::uint64_t target        = 0;
    double        sps           = 0.0;
    std::uint64_t logged_frames = 0;
};

constexpr std::string_view phase_name(run_phase p) noexcept {
    switch (p) {
        case run_phase::start:      return "start";
        case run_phase::step:       return "step";
        case run_phase::checkpoint: return "checkpoint";
        case run_phase::done:       return "done";
        case run_phase::failed:     return "failed";
    }
    return "step";
}

constexpr std::optional<run_phase> phase_from(std::string_view s) noexcept {
    if (s == "start")      return run_phase::start;
    if (s == "step")       return run_phase::step;
    if (s == "checkpoint") return run_phase::checkpoint;
    if (s == "done")       return run_phase::done;
    if (s == "failed")     return run_phase::failed;
    return std::nullopt;
}

// ---- emitter (coast side) ---------------------------------------------------

inline std::string format_progress(const progress& p) {
    return std::format(
        "{} v{} phase={} step={} start={} target={} sps={:.3f} logged={}",
        progress_sentinel, progress_version, phase_name(p.phase),
        p.step, p.start, p.target, p.sps, p.logged_frames);
}

namespace detail {
inline std::mutex& progress_mtx() {
    static std::mutex m;
    return m;
}
}  // namespace detail

// Write one structured progress line to stdout, flushed and thread-safe.
// Flushing matters: headless stdout is block-buffered when redirected to a file,
// so without the flush a supervisor tailing the file sees nothing until exit.
// The dedicated mutex keeps this line intact even when the emitting checkpoint
// hook runs on the sim thread while the poll loop emits on the main thread.
inline void emit_progress(const progress& p) {
    std::string line = format_progress(p);
    line.push_back('\n');
    std::lock_guard lk(detail::progress_mtx());
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fflush(stdout);
}

// ---- consumer (consim side) -------------------------------------------------

// Parse one line. Returns nullopt unless the line begins (after optional leading
// whitespace) with the sentinel AND carries a recognized phase. Unknown keys are
// ignored; missing numeric fields default to 0. Non-throwing.
inline std::optional<progress> parse_progress(std::string_view line) {
    // trim leading whitespace
    while (!line.empty() && (line.front() == ' ' || line.front() == '\t'))
        line.remove_prefix(1);
    // trim trailing CR/LF/space (files may hand us '\r\n')
    while (!line.empty() &&
           (line.back() == '\n' || line.back() == '\r' || line.back() == ' '))
        line.remove_suffix(1);

    if (!line.starts_with(progress_sentinel)) return std::nullopt;

    progress out;
    bool saw_phase = false;

    std::size_t i = 0;
    while (i < line.size()) {
        // skip spaces
        while (i < line.size() && line[i] == ' ') ++i;
        if (i >= line.size()) break;
        const std::size_t tok_beg = i;
        while (i < line.size() && line[i] != ' ') ++i;
        const std::string_view tok = line.substr(tok_beg, i - tok_beg);

        const std::size_t eq = tok.find('=');
        // Tokens without '=' are the sentinel and the "v1" version tag. The
        // version is intentionally NOT validated: v1 is the only version, and
        // forward-compat is by append-only fields (unknown trailing key=values
        // are ignored below), so a current parser reads any v1-shaped line. The
        // tag is reserved so a future BREAKING grammar (v2) can be gated here.
        if (eq == std::string_view::npos) continue;
        const std::string_view key = tok.substr(0, eq);
        const std::string_view val = tok.substr(eq + 1);

        auto parse_u64 = [](std::string_view v, std::uint64_t& dst) {
            std::from_chars(v.data(), v.data() + v.size(), dst);
        };

        if (key == "phase") {
            if (auto ph = phase_from(val)) { out.phase = *ph; saw_phase = true; }
        } else if (key == "step") {
            parse_u64(val, out.step);
        } else if (key == "start") {
            parse_u64(val, out.start);
        } else if (key == "target") {
            parse_u64(val, out.target);
        } else if (key == "logged") {
            parse_u64(val, out.logged_frames);
        } else if (key == "sps") {
            // std::from_chars is locale-INDEPENDENT, matching the
            // locale-independent std::format emitter. strtod would honor
            // LC_NUMERIC and silently mis-parse "12345.678" as 12345.0 under a
            // comma-decimal locale — a real hazard for a cross-process wire
            // contract. Floating-point from_chars is GCC 11+ / libc++ 20+.
            std::from_chars(val.data(), val.data() + val.size(), out.sps);
        }
        // unknown keys: ignored (forward-compat)
    }

    if (!saw_phase) return std::nullopt;
    return out;
}

}  // namespace coast::diag

#endif  // COAST_DIAG_PROGRESS_H
