//
// Created for the COAST observe system
//

#include "observe/observe_writer.h"
#include "core/local_time.h"

#include "observe/observe_chunk.h"
#include "hdf5/executor.h"
#include "logging/hdf5_utils.h"
#include "diag/diag.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <string>
#include <utility>
#include <vector>

namespace observe {

namespace {

// HDF5 prints its error stack on every failed call; a probe that may fail (an
// existing file that is unreadable after a crash) silences it for its duration.
class quiet_hdf5_errors {
    H5E_auto2_t func_ = nullptr;
    void*       data_ = nullptr;
public:
    quiet_hdf5_errors()  { H5Eget_auto2(H5E_DEFAULT, &func_, &data_); H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr); }
    ~quiet_hdf5_errors() { H5Eset_auto2(H5E_DEFAULT, func_, data_); }
    quiet_hdf5_errors(const quiet_hdf5_errors&)            = delete;
    quiet_hdf5_errors& operator=(const quiet_hdf5_errors&) = delete;
};

constexpr std::string_view k_step_suffix = "__step_index";

// The flat dataset name of a site: hierarchical full_ids are flattened with '__'
// so the HDF5 file holds no intermediate empty groups.
std::string flat_dataset_name(std::string_view full_id)
{
    std::string flat;
    flat.reserve(full_id.size() + 8);
    for (const char c : full_id) {
        if (c == '/') flat += "__";
        else          flat.push_back(c);
    }
    return flat;
}

// The chunk cache a site's value dataset is opened with: at least one full chunk,
// so single-row appends accumulate in cache (see the create path).
hid_t value_dataset_access(std::size_t value_bytes)
{
    const std::size_t chunk_bytes =
        static_cast<std::size_t>(observe::pick_chunk_rows(value_bytes)) * value_bytes;
    hid_t dapl = H5Pcreate(H5P_DATASET_ACCESS);
    H5Pset_chunk_cache(dapl, 521, std::max<std::size_t>(std::size_t{1} << 20, chunk_bytes), 1.0);
    return dapl;
}

// The run an existing file was written for (/metadata@run_id); "" if none.
std::string file_run_id(hid_t file)
{
    if (!hdf5::link_exists(file, "metadata")) return {};
    const hdf5::id_guard md(H5Gopen2(file, "metadata", H5P_DEFAULT), H5Gclose);
    if (md.id() < 0) return {};
    return hdf5::read_attr_string(md.id(), "run_id").value_or("");
}

// Every value dataset `<name>` with its `<name>__step_index` shrunk to the rows of
// steps below `from_step`. The step rows ascend (one append per step, in step
// order); a file where they do not is refused, as is any failed read or resize.
// An error message, or "" on success.
std::string drop_rows_from(hid_t file, std::uint64_t from_step)
{
    H5G_info_t info;
    if (H5Gget_info(file, &info) < 0) return "cannot list its datasets";
    std::vector<std::string> names;
    for (hsize_t i = 0; i < info.nlinks; ++i) {
        const ssize_t n = H5Lget_name_by_idx(file, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
        if (n < 0) return "cannot list its datasets";
        std::string name(static_cast<std::size_t>(n), '\0');
        H5Lget_name_by_idx(file, ".", H5_INDEX_NAME, H5_ITER_INC, i, name.data(), name.size() + 1, H5P_DEFAULT);
        names.push_back(std::move(name));
    }
    for (const auto& step_name : names) {
        if (step_name.size() <= k_step_suffix.size() || !step_name.ends_with(k_step_suffix)) continue;
        const std::string value_name = step_name.substr(0, step_name.size() - k_step_suffix.size());
        if (!hdf5::link_exists(file, value_name.c_str())) return "'" + step_name + "' has no value dataset";
        const hdf5::id_guard sds(H5Dopen2(file, step_name.c_str(), H5P_DEFAULT), H5Dclose);
        const hdf5::id_guard vds(H5Dopen2(file, value_name.c_str(), H5P_DEFAULT), H5Dclose);
        if (sds.id() < 0 || vds.id() < 0) return "cannot open '" + value_name + "'";
        const hdf5::id_guard sspace(H5Dget_space(sds.id()), H5Sclose);
        hsize_t rows = 0;
        H5Sget_simple_extent_dims(sspace.id(), &rows, nullptr);
        std::vector<std::uint64_t> steps(static_cast<std::size_t>(rows));
        if (rows > 0 && H5Dread(sds.id(), H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, steps.data()) < 0)
            return "cannot read '" + step_name + "'";
        if (!std::ranges::is_sorted(steps)) return "'" + step_name + "' is not in step order";
        const auto keep = static_cast<hsize_t>(std::ranges::lower_bound(steps, from_step) - steps.begin());
        const hdf5::id_guard vspace(H5Dget_space(vds.id()), H5Sclose);
        const int rank = H5Sget_simple_extent_ndims(vspace.id());
        if (rank < 1) return "cannot read the shape of '" + value_name + "'";
        std::vector<hsize_t> vdims(static_cast<std::size_t>(rank));
        H5Sget_simple_extent_dims(vspace.id(), vdims.data(), nullptr);
        if (vdims[0] != rows) return "'" + value_name + "' and its step index differ in length";
        vdims[0] = keep;
        if (H5Dset_extent(vds.id(), vdims.data()) < 0 || H5Dset_extent(sds.id(), &keep) < 0)
            return "cannot shrink '" + value_name + "'";
    }
    return {};
}

// Opens `path` to continue it for `run`, or returns -1 with the reason why not.
hid_t open_to_continue(const std::filesystem::path& path, const observe_run_t& run, std::string& why)
{
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) { why = "no earlier file"; return -1; }
    const quiet_hdf5_errors quiet;
    const hid_t id = H5Fopen(path.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (id < 0) { why = "the earlier file cannot be opened (a crash may have left it unreadable)"; return -1; }
    const std::string ran = file_run_id(id);
    if (ran != run.run_id) {
        why = "the earlier file is of " + (ran.empty() ? std::string("no recorded run") : "run " + ran);
        H5Fclose(id);
        return -1;
    }
    if (why = drop_rows_from(id, *run.from_step); !why.empty()) {
        why = "the earlier file " + why;
        H5Fclose(id);
        return -1;
    }
    return id;
}

} // namespace

observe_writer_t::observe_writer_t(std::filesystem::path h5_path,
                                   hdf5::executor&     h5_exec,
                                   observe_run_t       run)
    : h5_path_(std::move(h5_path)), h5_exec_(h5_exec), run_(std::move(run)) {}

observe_writer_t::~observe_writer_t() {
    stop_and_join();
}

void observe_writer_t::start() {
    if (started_.exchange(true)) {
        return;
    }

    // Pre-create the parent directory on the calling (sim) thread. Pure
    // std::filesystem, no HDF5 calls — safe here.
    if (h5_path_.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(h5_path_.parent_path(), ec);
    }

    // Open the HDF5 file on the executor's worker thread. If creation
    // fails we mark the writer stopped so subsequent enqueue() calls
    // become no-ops; we do not throw across the start() boundary because
    // observe failure should not abort the run.
    h5_file_ = h5_exec_.submit_sync([&] {
        if (run_.from_step) {
            std::string why;
            if (const hid_t id = open_to_continue(h5_path_, run_, why); id >= 0) {
                COAST_INFO("observe", "{}: run {} continues from step {}; its rows from that step on "
                           "were dropped", h5_path_.string(), run_.run_id, *run_.from_step);
                return hdf5::file_h{id, h5_exec_};
            }
            COAST_WARN("observe", "{}: run {} continues from step {}, but {}; the file is created "
                       "afresh, without the rows before that step", h5_path_.string(), run_.run_id,
                       *run_.from_step, why);
        }
        hid_t id = H5Fcreate(h5_path_.string().c_str(),
                             H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (id < 0) {
            COAST_ERROR("observe", "H5Fcreate failed for {}", h5_path_.string());
            return hdf5::file_h{};
        }
        // Family stamp lives in a /metadata group, uniform with the buffer
        // logger: a reader identifies any COAST log by /metadata@type.
        hid_t md = H5Gcreate2(id, "metadata", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (md >= 0) {
            herr_t arc = 0;
            arc |= hdf5::write_attr(md, "type", std::string_view{"observe_log"});
            arc |= hdf5::write_attr(md, "schema_version", std::string_view{"1.0"});
            const std::string ts =
                core::format_local_time(std::chrono::system_clock::now(), core::k_ctime_format);
            arc |= hdf5::write_attr(md, "created", std::string_view{ts});
            if (!run_.run_id.empty())
                arc |= hdf5::write_attr(md, "run_id", std::string_view{run_.run_id});
            // WARN, not abort — deliberately unlike state_logger, which fails the commit on
            // the same condition. A CHECKPOINT with missing metadata is unresumable, so
            // publishing one is worse than not writing it; a LOG with missing metadata
            // still carries all its sample data and is worth keeping. The divergence is a
            // severity judgement, not an oversight.
            if (arc < 0) COAST_WARN("observe", "observe_log: one or more metadata attributes failed to write");
            H5Gclose(md);
        }
        return hdf5::file_h{id, h5_exec_};
    });
}

bool observe_writer_t::open_existing(observe_site_t* site)
{
    const std::string value_path = "/" + flat_dataset_name(site->full_id);
    const std::string step_path  = value_path + std::string(k_step_suffix);
    if (!hdf5::link_exists(h5_file_.id(), value_path.c_str())) return false;
    // A site whose datasets cannot be continued starts afresh in this file: its old
    // datasets are unlinked, so the create path can make them under the same names.
    const auto drop = [&] {
        for (const auto& p : { value_path, step_path })
            if (hdf5::link_exists(h5_file_.id(), p.c_str()))
                H5Ldelete(h5_file_.id(), p.c_str(), H5P_DEFAULT);
        return false;
    };

    const hdf5::id_guard dapl(value_dataset_access(site->value_bytes), H5Pclose);
    hdf5::dataset_h value{H5Dopen2(h5_file_.id(), value_path.c_str(), dapl.id()), h5_exec_};
    hdf5::dataset_h steps{H5Dopen2(h5_file_.id(), step_path.c_str(), H5P_DEFAULT), h5_exec_};
    if (!value || !steps) {
        COAST_ERROR("observe", "{}: cannot reopen '{}' to continue it; the site starts afresh",
                    h5_path_.string(), value_path);
        return drop();
    }
    // The site must be what the file holds: same element type and row shape.
    const hdf5::id_guard type(H5Dget_type(value.id()), H5Tclose);
    const hdf5::id_guard space(H5Dget_space(value.id()), H5Sclose);
    const int rank = H5Sget_simple_extent_ndims(space.id());
    std::vector<hsize_t> dims(static_cast<std::size_t>(std::max(rank, 0)));
    if (rank > 0) H5Sget_simple_extent_dims(space.id(), dims.data(), nullptr);
    const bool fits = H5Tequal(type.id(), site->h5_dtype) > 0 &&
                      rank == static_cast<int>(1 + site->shape.size()) &&
                      std::equal(site->shape.begin(), site->shape.end(), dims.begin() + 1);
    if (!fits) {
        COAST_ERROR("observe", "{}: '{}' holds another type or shape than the site; its rows "
                    "are dropped and the site starts afresh", h5_path_.string(), value_path);
        return drop();
    }
    site->h5_dataset      = std::move(value);
    site->h5_step_dataset = std::move(steps);
    site->rows_written    = dims[0];
    return true;
}

void observe_writer_t::enqueue(observe_site_t* site, const void* value,
                               std::size_t value_bytes, std::uint64_t step) {
    if (stopped_.load(std::memory_order_acquire)) {
        return;
    }
    if (site == nullptr) {
        return;
    }

    // Copy the caller's value into an owned buffer. The lambda below captures
    // this buffer by move so the caller is free to mutate the source after
    // enqueue() returns. value_bytes is the site's locked size.
    //
    // This heap-allocates once per site per active step. Accepted for now: it is
    // paid only while a group is capturing, and submit() already heap-allocates
    // the type-erased task closure per record regardless, so a buffer pool here
    // would remove only one of two allocations while the HDF5 extend+write
    // dominates both. Revisit with a recycling pool only if a profile of a
    // high-rate capture tier (e.g. dendrite) actually shows this as hot.
    std::vector<std::byte> bytes(value_bytes);
    if (value_bytes > 0 && value != nullptr) {
        std::memcpy(bytes.data(), value, value_bytes);
    }

    h5_exec_.submit(
        [this, site, step, bytes = std::move(bytes)] {
            // All HDF5 work for this record runs on the executor's
            // worker thread. The whole sequence (lazy open + extend +
            // hyperslab select + write) is bundled in this one lambda
            // so it executes atomically w.r.t. other submissions —
            // important for the value/step pair, which must always
            // advance together.

            if (!h5_file_) {
                // File creation failed at start(); silently drop.
                return;
            }

            // ---- lazy dataset open ----
            // A continued file (start) already holds this site's datasets: they
            // are reopened, and the site appends after their kept rows.
            if (!site->h5_dataset && open_existing(site)) {
                opened_sites_.push_back(site);
            }
            if (!site->h5_dataset) {
                const std::string flat_name  = flat_dataset_name(site->full_id);
                const std::string value_path = "/" + flat_name;
                const std::string step_path  = "/" + flat_name + std::string(k_step_suffix);

                const std::size_t n_extra = site->shape.size();

                // Value dataset dims: [0, shape...]; maxdims:
                // [UNLIMITED, shape...]; chunk: [chunk_rows, shape...].
                //
                // chunk_rows is sized from a byte budget (not the historical
                // hardcoded 1024) so one chunk stays near 1 MiB and fits the
                // dataset chunk cache — see observe_chunk.h for why this
                // matters (cache-bypass thrash on large rows).
                const std::uint64_t chunk_rows =
                    observe::pick_chunk_rows(site->value_bytes);

                std::vector<hsize_t> dims   (1 + n_extra, 0);
                std::vector<hsize_t> maxdims(1 + n_extra, 0);
                std::vector<hsize_t> chunk  (1 + n_extra, 0);
                dims[0]    = 0;
                maxdims[0] = H5S_UNLIMITED;
                chunk[0]   = static_cast<hsize_t>(chunk_rows);
                for (std::size_t i = 0; i < n_extra; ++i) {
                    dims   [1 + i] = site->shape[i];
                    maxdims[1 + i] = site->shape[i];
                    chunk  [1 + i] = site->shape[i];
                }

                {
                    hid_t space = H5Screate_simple(static_cast<int>(dims.size()),
                                                   dims.data(), maxdims.data());
                    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
                    H5Pset_chunk(dcpl, static_cast<int>(chunk.size()), chunk.data());

                    // Per-dataset chunk cache sized to hold (at least) one full
                    // chunk, so the single-row appends accumulate in cache
                    // instead of triggering a read-modify-write per row. w0=1.0:
                    // append-only access means fully-written chunks are the best
                    // eviction candidates (we never re-read them).
                    hid_t dapl = value_dataset_access(site->value_bytes);

                    hid_t id = H5Dcreate2(h5_file_.id(), value_path.c_str(),
                                          site->h5_dtype, space,
                                          H5P_DEFAULT, dcpl, dapl);
                    H5Pclose(dapl);
                    H5Pclose(dcpl);
                    H5Sclose(space);
                    if (id < 0) {
                        COAST_ERROR("observe", "failed to create value dataset '{}'",
                                    value_path);
                        return;
                    }
                    site->h5_dataset = hdf5::dataset_h{id, h5_exec_};
                }

                {
                    hsize_t  s_dims[1]    = {0};
                    hsize_t  s_maxdims[1] = {H5S_UNLIMITED};
                    hsize_t  s_chunk[1]   = {1024};
                    hid_t space = H5Screate_simple(1, s_dims, s_maxdims);
                    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
                    H5Pset_chunk(dcpl, 1, s_chunk);
                    hid_t id = H5Dcreate2(h5_file_.id(), step_path.c_str(),
                                          H5T_NATIVE_UINT64, space,
                                          H5P_DEFAULT, dcpl, H5P_DEFAULT);
                    H5Pclose(dcpl);
                    H5Sclose(space);
                    if (id < 0) {
                        COAST_ERROR("observe", "failed to create step dataset '{}'",
                                    step_path);
                        // Reset the value dataset we just created so the
                        // pair stays consistent (both opened or neither).
                        site->h5_dataset.reset(true);
                        return;
                    }
                    site->h5_step_dataset = hdf5::dataset_h{id, h5_exec_};
                }

                opened_sites_.push_back(site);
            }

            // ---- append row ----
            // Value and step row `row` (hdf5::append_rows). A failed append
            // leaves rows_written unchanged, so the site's next row retries the
            // same index and the two datasets never disagree on the row count by
            // more than that one retried row (counting a failed row would leave a
            // gap and shift every later row against its step).
            const hsize_t row = static_cast<hsize_t>(site->rows_written);
            if (hdf5::append_rows(site->h5_dataset.id(), row, 1, site->shape,
                                  site->h5_dtype, bytes.data()) < 0) {
                COAST_ERROR("observe", "appending a value row failed for site '{}'",
                            site->full_id);
                return;
            }
            const std::uint64_t step_val = step;
            if (hdf5::append_rows(site->h5_step_dataset.id(), row, 1, {},
                                  H5T_NATIVE_UINT64, &step_val) < 0) {
                COAST_ERROR("observe", "appending a step row failed for site '{}'",
                            site->full_id);
                return;
            }

            site->rows_written += 1;
        });
}

void observe_writer_t::stop_and_join() {
    if (stopped_.exchange(true)) {
        return;
    }
    if (!started_.load(std::memory_order_acquire)) {
        return;
    }

    // Flush all pending writes by submitting a sync barrier — any
    // submit() calls already queued ahead of us will complete first.
    h5_exec_.submit_sync([this] {
        if (h5_file_) {
            H5Fflush(h5_file_.id(), H5F_SCOPE_LOCAL);
        }

        // Close all opened-site dataset handles, then the file, in that
        // order. Because reset() submits a fire-and-forget close and the
        // executor preserves FIFO order, every dataset close lands in the
        // queue before the file close — honouring HDF5's child-before-parent
        // requirement. A final submit_sync barrier ensures these closes have
        // executed before stop_and_join() returns.
        for (auto* s : opened_sites_) {
            if (s == nullptr) continue;
            s->h5_dataset.reset(true);
            s->h5_step_dataset.reset(true);
        }
        opened_sites_.clear();
        h5_file_.reset(true);

    });

    
}

} // namespace observe
