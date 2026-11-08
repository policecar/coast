//
// Local wall-clock time as text — the one formatter for file-name stamps and
// "created" attributes.
//
// std::localtime and std::ctime return pointers into one static buffer shared by
// every thread. COAST formats time on the caller's thread, the sim thread and the
// HDF5 executor concurrently, so those are data races; localtime_r / localtime_s
// fill a caller-owned std::tm instead.
//

#ifndef COAST_CORE_LOCAL_TIME_H
#define COAST_CORE_LOCAL_TIME_H

#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <string>

namespace core {

// std::ctime's layout ("Wed Jun 30 21:49:08 1993"), without its trailing newline.
inline constexpr const char* k_ctime_format = "%a %b %e %H:%M:%S %Y";

// The compact stamp used in snapshot file names ("20260630_214908").
inline constexpr const char* k_file_stamp_format = "%Y%m%d_%H%M%S";

// `tp` in local time, formatted by std::strftime with `format`. Returns "" if the
// time cannot be converted or the result does not fit 128 bytes.
[[nodiscard]] inline std::string format_local_time(std::chrono::system_clock::time_point tp,
                                                   const char*                           format)
{
    const std::time_t t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
#if defined(_WIN32)
    if (localtime_s(&tm, &t) != 0) return {};
#else
    if (localtime_r(&t, &tm) == nullptr) return {};
#endif
    char buf[128];
    const std::size_t n = std::strftime(buf, sizeof buf, format, &tm);
    return { buf, n };
}

// A snapshot of `config_name` at `step`, written now:
// <dir>/<config_name>_<YYYYMMDD_HHMMSS>_<step>.h5 (headless checkpoints and the
// GUI's auto-save; build_env's --resume and the GUI's resume dialog parse it).
[[nodiscard]] inline std::filesystem::path snapshot_path(const std::filesystem::path& dir,
                                                         const std::string&           config_name,
                                                         std::uint64_t                step)
{
    return dir / (config_name + "_" +
                  format_local_time(std::chrono::system_clock::now(), k_file_stamp_format) +
                  "_" + std::to_string(step) + ".h5");
}

} // namespace core

#endif // COAST_CORE_LOCAL_TIME_H
