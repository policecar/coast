#ifndef COAST_FILE_MODEL_H
#define COAST_FILE_MODEL_H

// Immutable per-open description of a COAST-written HDF5 log file
// (IDD §5/§6): family, semantic tree, and one dataset_desc per scannable
// dataset. Produced once by open_and_map on the hdf5::executor worker,
// then read-only.

#include "h5browse/h5b_error.h"
#include "h5browse/scalar_type.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace hdf5 {
class executor;
}

namespace h5b {

enum class file_family : std::uint8_t { observe, buffer_log, unknown };

constexpr const char* file_family_name(file_family f)
{
    switch (f) {
        case file_family::observe:    return "observe";
        case file_family::buffer_log: return "buffer log";
        case file_family::unknown:    return "unknown";
    }
    std::unreachable();
}

struct dataset_desc {
    std::string h5_path;      // absolute HDF5 path, e.g. "/steps/buffers/3/data"
    std::string display_name; // semantic name, e.g. "neuron[3]/activity"
    scalar_type dtype = scalar_type::f64;

    std::vector<std::uint64_t> shape; // logical per-row shape (dims[1..]); empty = scalar rows
    std::vector<std::uint64_t> chunk; // chunk dims; empty = contiguous layout
    std::uint64_t rows      = 0;      // dims[0] at open time
    std::uint64_t row_width = 1;      // flattened elements per row
    bool          compressed = false; // any filter on the dataset (gzip/shuffle/...)

    // Sibling dataset mapping row index -> simulation step, when the
    // family provides one (observe __step_index, probe steps, buffer_log
    // step_numbers).
    std::optional<std::string> step_index_path;
};

struct tree_node {
    enum class kind_t : std::uint8_t { root, group, dataset, meta };

    std::string name;
    kind_t      kind = kind_t::group;
    std::optional<std::uint32_t> dataset; // index into file_model::datasets
    std::string detail;                   // meta payload / diagnostics text
    std::vector<tree_node> children;
};

struct file_model {
    file_family           family = file_family::unknown;
    std::filesystem::path path;
    std::uint64_t         file_bytes = 0;
    std::string           schema_version;
    tree_node             root;
    std::vector<dataset_desc> datasets;

    // Family-level key/value facts for the GUI header (created, start_step,
    // counts, ...). Order = display order.
    std::vector<std::pair<std::string, std::string>> metadata;
};

// Opens `path` read-only and maps it into a file_model. All libhdf5 work
// runs in a single executor job; every HDF5 failure is caught inside the
// job and returned as an error value (never an exception across the
// queue). Per-dataset damage is contained: the broken dataset becomes a
// diagnostic tree node and mapping continues (D8); only file-level
// failures reject the whole open.
result<file_model> open_and_map(hdf5::executor& exec,
                                const std::filesystem::path& path);

} // namespace h5b

#endif // COAST_FILE_MODEL_H
