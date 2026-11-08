#ifndef COAST_COASTD_JOB_JSON_H
#define COAST_COASTD_JOB_JSON_H

// -----------------------------------------------------------------------------
// coastd job <-> JSON. One flat JSON object per job; the file IS the record.
//
// Evolution is TOLERANT + ADDITIVE: parse reads every field with a default, so a
// missing key is fine and an unknown key is ignored — additive schema changes
// need no migration. `schema_version` is the escape hatch for a NON-additive
// change: a file declaring a version this daemon cannot safely interpret throws
// `job_schema_error` and is NEVER mangled — the store quarantines it instead
// (a hard invariant, especially for an active/ job).
// -----------------------------------------------------------------------------

#include <stdexcept>
#include <string>
#include <string_view>

#include "daemon/types.h"

namespace coastd {

// The highest job-file schema version this build understands.
inline constexpr int job_schema_version = 1;

// Thrown by parse_job when a file declares a schema_version > job_schema_version.
// Carries the offending value so the store can log it precisely before
// quarantining the file untouched.
struct job_schema_error : std::runtime_error {
    int found;
    explicit job_schema_error(int v)
        : std::runtime_error("job schema_version " + std::to_string(v) +
                             " is newer than supported " +
                             std::to_string(job_schema_version)),
          found(v) {}
};

// Advisory status <-> wire string (stable tokens; unknown -> queued on parse).
[[nodiscard]] std::string_view status_name(job_status s) noexcept;
[[nodiscard]] job_status       status_from(std::string_view s) noexcept;

// Serialize `j` to a pretty JSON string (the on-disk form).
[[nodiscard]] std::string dump_job(const job& j);

// Parse a job from JSON text. Throws job_schema_error for an unsupported
// schema_version, or nlohmann::json::exception for malformed JSON. Missing
// fields take their struct defaults; unknown fields are ignored.
[[nodiscard]] job parse_job(std::string_view text);

} // namespace coastd

#endif // COAST_COASTD_JOB_JSON_H
