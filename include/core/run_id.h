//
// A run's identity for checkpoint continuation (SPEC.md S.5.14).
//
// A config names its run with the top-level `run_id`; "auto" (the default) asks for
// a fresh one. A dataset continues its sequence from a checkpoint only when the
// checkpoint was written by the same run_id, so a user decides consciously: the
// checkpoint's id to continue, a new one (or "auto") to start afresh. coastd
// replaces "auto" in a job's run config when it writes it, so every relaunch of
// the job reads one concrete id; coast replaces a remaining "auto" per process.
//

#ifndef COAST_RUN_ID_H
#define COAST_RUN_ID_H

#include <string>
#include <string_view>

namespace coast {

inline constexpr std::string_view k_auto_run_id = "auto";

// 32 hex digits from std::random_device.
[[nodiscard]] std::string new_run_id();

// `configured`, or a new id for "auto".
[[nodiscard]] std::string resolve_run_id(std::string_view configured);

} // namespace coast

#endif // COAST_RUN_ID_H
