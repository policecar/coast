#include "h5browse/dataset_reader.h"

#include "hdf5/executor.h"
#include "hdf5/native_type.h"

#include <hdf5.h>

#include <algorithm>
#include <cassert>
#include <exception>

namespace h5b {

namespace {

// Worker-thread only (H5T_NATIVE_* macros may call H5open()). The mapping is
// hdf5::native_type's, the one C++ type -> native type table.
hid_t native_hid(scalar_type t)
{
    return dispatch(t, []<class T>(std::type_identity<T>) { return hdf5::native_type<T>(); });
}

// Rule 3: disable the per-dataset chunk cache for uncompressed scans,
// keep defaults for filtered datasets. Returns a dapl (or H5P_DEFAULT).
hid_t make_scan_dapl(const dataset_desc& d)
{
    if (d.compressed || d.chunk.empty()) return H5P_DEFAULT;
    const hid_t dapl = H5Pcreate(H5P_DATASET_ACCESS);
    if (dapl < 0) return H5P_DEFAULT;
    H5Pset_chunk_cache(dapl, 0, 0, H5D_CHUNK_CACHE_W0_DEFAULT);
    return dapl;
}

// Rule 1: shape-matched read of rows [first, first+n). Worker-thread only.
// Returns rows read or an error message.
std::expected<std::uint64_t, std::string>
read_rows_worker(hid_t dset, const dataset_desc& d, std::uint64_t first,
                 std::uint64_t n, std::span<std::byte> out)
{
    const std::size_t rank = 1 + d.shape.size();
    std::vector<hsize_t> offset(rank, 0), count(rank);
    offset[0] = first;
    count[0]  = n;
    for (std::size_t i = 0; i < d.shape.size(); ++i) count[i + 1] = d.shape[i];

    const hid_t fspace = H5Dget_space(dset);
    if (fspace < 0) return std::unexpected(harvest_h5_error("cannot get dataspace"));
    if (H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset.data(), nullptr,
                            count.data(), nullptr) < 0) {
        H5Sclose(fspace);
        return std::unexpected(harvest_h5_error("hyperslab selection failed"));
    }
    // Memspace with the SAME rank and dims as the selection (rule 1).
    const hid_t mspace =
        H5Screate_simple(static_cast<int>(rank), count.data(), nullptr);
    if (mspace < 0) {
        H5Sclose(fspace);
        return std::unexpected(harvest_h5_error("memspace creation failed"));
    }

    assert(out.size() >= n * d.row_width * scalar_size(d.dtype));
    const herr_t rc = H5Dread(dset, native_hid(d.dtype), mspace, fspace,
                              H5P_DEFAULT, out.data());
    H5Sclose(mspace);
    H5Sclose(fspace);
    if (rc < 0) return std::unexpected(harvest_h5_error("H5Dread failed"));
    return n;
}

} // namespace

result<std::int64_t> open_file_readonly(hdf5::executor&              exec,
                                        const std::filesystem::path& path)
{
    return exec.submit_sync([&]() -> result<std::int64_t> {
        const h5_error_silencer silence;
        const hid_t fid = H5Fopen(path.string().c_str(), H5F_ACC_RDONLY,
                                  H5P_DEFAULT);
        if (fid < 0) {
            return std::unexpected(error{error_category::open,
                                         harvest_h5_error("cannot open file"),
                                         path.string()});
        }
        return static_cast<std::int64_t>(fid);
    });
}

void close_file_handle(hdf5::executor& exec, std::int64_t file)
{
    if (file < 0) return;
    exec.submit_sync([file] { H5Fclose(static_cast<hid_t>(file)); });
}

std::uint64_t default_block_rows(const dataset_desc& d)
{
    if (!d.chunk.empty() && d.chunk[0] > 0) return d.chunk[0];
    const std::uint64_t row_bytes =
        std::max<std::uint64_t>(1, d.row_width * scalar_size(d.dtype));
    return std::max<std::uint64_t>(1, (4ull << 20) / row_bytes);
}

result<std::shared_ptr<const row_block>>
read_block(hdf5::executor& exec, std::int64_t file, const dataset_desc& d,
           std::uint32_t dataset_index, std::uint64_t block_index,
           std::uint64_t rows_per_block)
{
    assert(rows_per_block > 0);
    const std::uint64_t first = block_index * rows_per_block;
    if (first >= d.rows) {
        return std::unexpected(error{error_category::read,
                                     "block index beyond dataset extent",
                                     d.h5_path});
    }
    const std::uint64_t n = std::min(rows_per_block, d.rows - first);

    return exec.submit_sync([&]() -> result<std::shared_ptr<const row_block>> {
        const h5_error_silencer silence;

        const hid_t dapl = make_scan_dapl(d);
        const hid_t dset = H5Dopen2(static_cast<hid_t>(file),
                                    d.h5_path.c_str(), dapl);
        if (dapl != H5P_DEFAULT) H5Pclose(dapl);
        if (dset < 0) {
            return std::unexpected(error{error_category::read,
                                         harvest_h5_error("cannot open dataset"),
                                         d.h5_path});
        }

        // submit_sync rethrows on the caller's thread, so a bad_alloc from
        // the block sizing (row_width comes from untrusted file dims) must
        // be caught here to honour the value-error contract. The dataset
        // handle is closed on both paths.
        try {
            auto blk       = std::make_shared<row_block>();
            blk->dataset   = dataset_index;
            blk->block     = block_index;
            blk->first_row = first;
            blk->n_rows    = n;
            blk->row_width = d.row_width;
            blk->dtype     = d.dtype;
            blk->bytes.resize(n * d.row_width * scalar_size(d.dtype));

            const auto rc = read_rows_worker(dset, d, first, n, blk->bytes);
            H5Dclose(dset);
            if (!rc) {
                return std::unexpected(
                    error{error_category::read, rc.error(), d.h5_path});
            }
            return std::shared_ptr<const row_block>(std::move(blk));
        } catch (const std::exception& e) {
            H5Dclose(dset);
            return std::unexpected(error{
                error_category::read,
                std::string{"block allocation/read failed: "} + e.what(),
                d.h5_path});
        }
    });
}

// ---------------------------------------------------------------- pipeline

read_pipeline::read_pipeline(hdf5::executor& exec, std::int64_t file,
                             const dataset_desc& d, std::uint64_t first_row,
                             std::uint64_t n_rows)
    : exec_(exec), file_(file), desc_(d)
{
    const std::uint64_t row_bytes =
        std::max<std::uint64_t>(1, d.row_width * scalar_size(d.dtype));

    // >= 4 MB per job, rounded up to whole chunks (T1: buffer layout flat
    // 0.25-16 MB, observe layout peaks at 4 MB).
    std::uint64_t rows = std::max<std::uint64_t>(1, min_job_bytes / row_bytes);
    if (!d.chunk.empty() && d.chunk[0] > 0) {
        const std::uint64_t c = d.chunk[0];
        rows                  = ((rows + c - 1) / c) * c;
    }
    rows_per_batch_ = rows;
    next_submit_    = first_row;
    end_row_        = std::min(first_row + n_rows, d.rows);

    // Open the dataset once, with the rule-3 access policy.
    const auto open_rc = exec_.submit_sync([&]() -> result<std::int64_t> {
        const h5_error_silencer silence;
        const hid_t dapl = make_scan_dapl(d);
        const hid_t ds =
            H5Dopen2(static_cast<hid_t>(file_), d.h5_path.c_str(), dapl);
        if (dapl != H5P_DEFAULT) H5Pclose(dapl);
        if (ds < 0) {
            return std::unexpected(error{error_category::read,
                                         harvest_h5_error("cannot open dataset"),
                                         d.h5_path});
        }
        return static_cast<std::int64_t>(ds);
    });
    if (!open_rc) {
        open_error_ = open_rc.error();
        return;
    }
    dset_ = *open_rc;

    // A batch buffer never needs to exceed the span we actually read, so cap
    // it there. This bounds a corrupt-huge chunk[0] (which inflated
    // rows_per_batch_) to the real dataset size for free — no allocation
    // beyond what a full read would touch anyway. A corrupt-huge row_width
    // can still overflow the product; that runs on the scan thread (not in a
    // job), where an uncaught bad_alloc would terminate the process, so trap
    // it and surface via next().
    const std::uint64_t span_rows =
        end_row_ > next_submit_ ? end_row_ - next_submit_ : 0;
    const std::uint64_t buf_rows    = std::min(rows_per_batch_, span_rows);
    const std::uint64_t batch_bytes = buf_rows * row_bytes;
    try {
        for (slot& s : slots_) s.buffer.resize(batch_bytes);
    } catch (const std::exception& e) {
        open_error_ = error{error_category::read,
                            std::string{"batch buffer allocation failed: "} +
                                e.what(),
                            d.h5_path};
        return;
    }

    // Prime both slots (depth 2).
    for (slot& s : slots_) {
        if (next_submit_ >= end_row_) break;
        const std::uint64_t n =
            std::min(rows_per_batch_, end_row_ - next_submit_);
        submit_read(s, next_submit_, n);
        next_submit_ += n;
    }
}

read_pipeline::~read_pipeline()
{
    // Jobs capture raw pointers into our slots — they MUST finish before
    // the buffers die.
    for (slot& s : slots_) {
        if (s.in_flight) s.pending.wait();
    }
    if (dset_ >= 0) {
        const std::int64_t ds = dset_;
        exec_.submit_sync([ds] { H5Dclose(static_cast<hid_t>(ds)); });
    }
}

void read_pipeline::submit_read(slot& s, std::uint64_t first, std::uint64_t n)
{
    auto promise = std::make_shared<std::promise<result<std::uint64_t>>>();
    s.pending    = promise->get_future();
    s.first_row  = first;
    s.in_flight  = true;

    // Async job: never throws out (the executor would swallow the
    // exception and the promise would never fire).
    std::byte*          out  = s.buffer.data();
    const std::size_t   cap  = s.buffer.size();
    const dataset_desc& d    = desc_;
    const std::int64_t  dset = dset_;
    exec_.submit([promise, out, cap, &d, dset, first, n] {
        const h5_error_silencer silence;
        try {
            auto rc = read_rows_worker(static_cast<hid_t>(dset), d, first, n,
                                       std::span<std::byte>(out, cap));
            if (!rc) {
                promise->set_value(std::unexpected(
                    error{error_category::read, rc.error(), d.h5_path}));
            } else {
                promise->set_value(*rc);
            }
        } catch (const std::exception& e) {
            promise->set_value(std::unexpected(error{
                error_category::read,
                std::string{"unexpected exception in read job: "} + e.what(),
                d.h5_path}));
        }
    });
}

result<std::optional<read_pipeline::batch>> read_pipeline::next()
{
    if (open_error_) return std::unexpected(*open_error_);

    slot& cur = slots_[turn_];
    if (!cur.in_flight) return std::optional<batch>{}; // drained

    auto rc = cur.pending.get();
    cur.in_flight = false;
    if (!rc) {
        open_error_ = rc.error(); // sticky
        return std::unexpected(*open_error_);
    }
    const std::uint64_t n     = *rc;
    const std::uint64_t first = cur.first_row;

    // Refill the OTHER slot (the caller is about to consume this one);
    // by the time we hand out slot A, slot B's read is already running.
    slot& other = slots_[turn_ ^ 1];
    if (!other.in_flight && next_submit_ < end_row_) {
        const std::uint64_t m =
            std::min(rows_per_batch_, end_row_ - next_submit_);
        submit_read(other, next_submit_, m);
        next_submit_ += m;
    }
    turn_ ^= 1;

    const std::uint64_t bytes = n * desc_.row_width * scalar_size(desc_.dtype);
    return std::optional<batch>{
        batch{first, n, std::span<const std::byte>(cur.buffer.data(), bytes)}};
}

} // namespace h5b
