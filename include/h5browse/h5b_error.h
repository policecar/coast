#ifndef COAST_H5B_ERROR_H
#define COAST_H5B_ERROR_H

// h5browse error channel (IDD §10, D8): errors are values, never
// exceptions across the executor queue or the engine/GUI boundary.
// Consistent with the PR #83 step_result doctrine.

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <utility>

namespace h5b {

enum class error_category : std::uint8_t {
    open,      // file-level: cannot open / not readable at all
    schema,    // recognizable HDF5 but not a COAST log family
    structure, // family recognized, expected objects missing/broken
    read,      // per-dataset data read failure (containment boundary)
};

constexpr const char* error_category_name(error_category c)
{
    switch (c) {
        case error_category::open:      return "open";
        case error_category::schema:    return "schema";
        case error_category::structure: return "structure";
        case error_category::read:      return "read";
    }
    std::unreachable();
}

struct error {
    error_category category = error_category::open;
    std::string    message;     // triage-grade, user-facing
    std::string    object_path; // file path or HDF5 object path
};

template <class T>
using result = std::expected<T, error>;

// Harvests the deepest frame of the HDF5 default error stack (T3 verified:
// the deepest frame is the precise one in every observed failure class) and
// prepends recognized triage text ("file is currently being written…",
// "file is truncated…", "not an HDF5 file…"). Returns `fallback` when the
// stack is empty.
//
// MUST run on the executor worker thread, immediately after the failing
// libhdf5 call — the next call into the library resets the stack.
std::string harvest_h5_error(std::string_view fallback);

// Scoped suppression of libhdf5's auto-printed error stacks while the
// browser probes files that are corrupt by design. Construct only inside
// an executor job: the single worker thread guarantees no writer call
// interleaves with the silenced window, and the destructor restores the
// previous handler before the job returns — the process-wide setting is
// untouched outside the job.
class h5_error_silencer {
public:
    h5_error_silencer();
    ~h5_error_silencer();
    h5_error_silencer(const h5_error_silencer&)            = delete;
    h5_error_silencer& operator=(const h5_error_silencer&) = delete;

private:
    void* func_ = nullptr; // H5E_auto2_t, kept opaque to avoid hdf5.h here
    void* data_ = nullptr;
};

} // namespace h5b

#endif // COAST_H5B_ERROR_H
