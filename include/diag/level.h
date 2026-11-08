#ifndef COAST_DIAG_LEVEL_H
#define COAST_DIAG_LEVEL_H

// The diag severity level, split out from <diag/diag.h> so headers that only
// need the enum (e.g. cli/options.h stores a resolved log_level) don't pull in
// the whole facility — <format>, <mutex>, <functional>, the inline sink state,
// and the COAST_* macros. Call sites that actually log include <diag/diag.h>.

#include <cstdint>
#include <optional>
#include <string_view>

namespace coast::diag {

// Ordered by ascending severity so `record_level >= min_level` is the filter.
enum class level : std::uint8_t {
    trace = 0,
    debug = 1,
    info  = 2,
    warn  = 3,
    error = 4,
    off   = 5,  // set as min_level to silence everything
};

constexpr std::string_view level_name(level l) noexcept {
    switch (l) {
        case level::trace: return "TRACE";
        case level::debug: return "DEBUG";
        case level::info:  return "INFO";
        case level::warn:  return "WARN";
        case level::error: return "ERROR";
        case level::off:   return "OFF";
    }
    return "?";
}

// Parse a CLI/env level name (case-insensitive). For --log-level.
constexpr std::optional<level> level_from(std::string_view s) noexcept {
    // Case-insensitive compare against a lowercase literal (ASCII only — level
    // names are all ASCII). Keeps `--log-level INFO` and `Info` working.
    auto ieq = [](std::string_view a, std::string_view b) noexcept {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            char c = a[i];
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            if (c != b[i]) return false;
        }
        return true;
    };
    if (ieq(s, "trace")) return level::trace;
    if (ieq(s, "debug")) return level::debug;
    if (ieq(s, "info"))  return level::info;
    if (ieq(s, "warn"))  return level::warn;
    if (ieq(s, "error")) return level::error;
    if (ieq(s, "off"))   return level::off;
    return std::nullopt;
}

}  // namespace coast::diag

#endif  // COAST_DIAG_LEVEL_H
