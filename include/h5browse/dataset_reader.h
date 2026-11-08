#ifndef COAST_DATASET_READER_H
#define COAST_DATASET_READER_H

// Executor-facing read layer (IDD §6). Encodes the T1 production rules:
//   rule 1 — read-job memspaces match the file-selection shape (rank+dims);
//            a flat 1-D memspace hits the H5Sselect_shape_same fallback and
//            is 15.9x slower.
//   rule 3 — the per-dataset chunk cache is disabled on scan opens of
//            uncompressed datasets (27x RSS, +40% speed on many-dataset
//            files) and left at defaults for compressed ones (-8% if
//            disabled there).
// read_pipeline implements the validated depth-2 read-ahead (+60% over
// serial, depth 4 adds nothing): while the scan thread accumulates batch
// N, the executor reads batch N+1.
//
// hid_t values cross this interface but every libhdf5 call runs inside an
// executor job; handles are opened and closed on the worker thread only.

#include "h5browse/chunk_cache.h"
#include "h5browse/file_model.h"
#include "h5browse/h5b_error.h"

#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace hdf5 {
class executor;
}

namespace h5b {

// Persistent read-only file handle for a browsing session (open_and_map
// closes its own handle; the engine keeps one of these for scans and
// fetches). Both run one executor job.
result<std::int64_t> open_file_readonly(hdf5::executor& exec,
                                        const std::filesystem::path& path);
void close_file_handle(hdf5::executor& exec, std::int64_t file);

// Rows per chunk-aligned block: chunk[0] when chunked, else sized to
// ~4 MB of native bytes. Always >= 1.
[[nodiscard]] std::uint64_t default_block_rows(const dataset_desc& d);

// One cache block via a single synchronous executor job (the fetch lane:
// ~ms per block, serviced between analysis batches).
result<std::shared_ptr<const row_block>>
read_block(hdf5::executor& exec, std::int64_t file, const dataset_desc& d,
           std::uint32_t dataset_index, std::uint64_t block_index,
           std::uint64_t rows_per_block);

// Streams [first_row, first_row + n_rows) in >= 4 MB whole-chunk batches
// with two read jobs in flight. The span returned by next() is valid
// until the following next() call or destruction.
class read_pipeline {
public:
    static constexpr std::uint64_t min_job_bytes = 4ull << 20; // T1-pinned

    read_pipeline(hdf5::executor& exec, std::int64_t file,
                  const dataset_desc& d, std::uint64_t first_row,
                  std::uint64_t n_rows);
    ~read_pipeline(); // drains in-flight jobs, closes the dataset handle

    read_pipeline(const read_pipeline&)            = delete;
    read_pipeline& operator=(const read_pipeline&) = delete;

    struct batch {
        std::uint64_t              first_row = 0;
        std::uint64_t              n_rows    = 0;
        std::span<const std::byte> data; // native dtype, row-major
    };

    // nullopt = all rows delivered. An error is sticky: further calls
    // keep returning it (per-dataset containment happens in the engine).
    result<std::optional<batch>> next();

    [[nodiscard]] std::uint64_t rows_per_batch() const { return rows_per_batch_; }

private:
    struct slot {
        std::vector<std::byte>            buffer;
        std::future<result<std::uint64_t>> pending; // n_rows read
        std::uint64_t                     first_row = 0;
        bool                              in_flight = false;
    };

    void submit_read(slot& s, std::uint64_t first, std::uint64_t n);

    hdf5::executor&     exec_;
    const std::int64_t  file_;
    const dataset_desc& desc_;
    std::int64_t        dset_ = -1; // opened on the worker in the ctor

    std::uint64_t rows_per_batch_ = 1;
    std::uint64_t end_row_        = 0;
    std::uint64_t next_submit_    = 0;

    slot                 slots_[2];
    int                  turn_ = 0;
    std::optional<error> open_error_; // ctor-time failure, reported by next()
};

} // namespace h5b

#endif // COAST_DATASET_READER_H
