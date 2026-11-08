//
// State Logger implementation
//

#include "logging/state_logger.h"
#include "hdf5/native_type.h"
#include "core/local_time.h"

#include <cstring>
#include "logging/config_record.h"
#include "diag/diag.h"
#include "logging/mem_state_io.h"
#include "logging/state_checks.h"
#include "hdf5_parse.h"
#include "sim_core/sim_env.h"
#include "sim_core/io_entity.h"

#include "hdf5/executor.h"

#include <fcntl.h>
#include <unistd.h>

#include <functional>   // std::hash (per-writer tmp name below)
#include <thread>

#include <charconv>
#include <chrono>
#include <expected>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace logging {

namespace {

// The format version a checkpoint's metadata group records (schema_version),
// from the attribute's value `v` (empty when it is absent). "1.0", written
// before the version was an integer, is version 1. A missing or malformed
// value, or a version newer than this build reads, refuses the load: this build
// cannot know what a newer format changed.
std::expected<std::uint32_t, std::string> parse_format_version(std::string_view v)
{
    if (v.empty())
        return std::unexpected(std::string("the checkpoint records no format version (metadata/schema_version)"));
    if (v == "1.0")
        return 1u;
    std::uint32_t n = 0;
    const auto [end, ec] = std::from_chars(v.data(), v.data() + v.size(), n);
    if (ec != std::errc{} || end != v.data() + v.size() || n == 0)
        return std::unexpected(std::format("unreadable checkpoint format version \"{}\"", v));
    if (n > k_format_version)
        return std::unexpected(std::format("the checkpoint has format version {}, newer than this build "
                                           "reads ({}); load it with a newer build", n, k_format_version));
    return n;
}

// The format version is read from the checkpoint itself, before the reader the
// entities use is built with it (state_reader takes it at construction). A
// checkpoint without a metadata group is refused on every load path.
constexpr const char* k_missing_metadata = "Missing metadata group in state file";

std::expected<std::uint32_t, std::string> read_format_version(const mem_group& root)
{
    const auto md = root.groups.find("metadata");
    if (md == root.groups.end())
        return std::unexpected(std::string(k_missing_metadata));
    const auto v = md->second.attrs.find("schema_version");
    return parse_format_version(v == md->second.attrs.end() ? std::string_view{} : std::string_view{v->second});
}

std::expected<std::uint32_t, std::string> read_format_version(hdf5::file& file)
{
    if (!hdf5::link_exists(file.id(), "metadata"))
        return std::unexpected(std::string(k_missing_metadata));
    const auto md = hdf5::group::open(file.id(), "metadata");
    if (!md)
        return std::unexpected(std::string(k_missing_metadata));
    // Same read as hdf5_state_reader::read_attr: an absent attribute is empty.
    if (!hdf5::attr_exists(md.id(), "schema_version"))
        return parse_format_version({});
    return parse_format_version(hdf5::read_attr_string(md.id(), "schema_version").value_or(""));
}

} // namespace

// --- hdf5_state_writer implementation ---

hdf5_state_writer::hdf5_state_writer(hdf5::file& file)
    : file_(file)
{
}

hdf5_state_writer::~hdf5_state_writer()
{
    // Close any remaining groups
    group_stack_.clear();
}

hid_t hdf5_state_writer::current_loc() const
{
    if (group_stack_.empty()) {
        return file_.id();
    }
    return group_stack_.back().id();
}

void hdf5_state_writer::begin_group(std::string_view name)
{
    std::string name_str(name);
    auto g = hdf5::group::create(current_loc(), name_str.c_str());
    if (!g) ok_ = false;
    group_stack_.push_back(std::move(g));
}

void hdf5_state_writer::end_group()
{
    if (!group_stack_.empty()) {
        group_stack_.pop_back();
    }
}

// Scalars and arrays all go through these two, typed by hdf5::native_type<T>()
// (one C++ -> HDF5 table for the whole project).
namespace {

template <class T>
herr_t write_scalar(hid_t loc, std::string_view name, const T& value)
{
    const std::string n(name);
    auto space = hdf5::dataspace::scalar();
    auto ds = hdf5::dataset::create(loc, n.c_str(), hdf5::native_type<T>(), space.id());
    return ds.write(hdf5::native_type<T>(), H5S_ALL, H5S_ALL, &value);
}

// An empty array is not written (a reader treats the field as absent).
template <class T>
herr_t write_array(hid_t loc, std::string_view name, std::span<const T> values,
                   std::span<const hsize_t> dims)
{
    if (values.empty()) return 0;
    const std::string n(name);
    auto space = hdf5::dataspace::simple(dims);
    auto ds = hdf5::dataset::create(loc, n.c_str(), hdf5::native_type<T>(), space.id());
    return ds.write(hdf5::native_type<T>(), H5S_ALL, H5S_ALL, values.data());
}

template <class T>
herr_t write_vector(hid_t loc, std::string_view name, std::span<const T> values)
{
    const hsize_t dims[1] = { values.size() };
    return write_array(loc, name, values, std::span<const hsize_t>(dims));
}

} // namespace

void hdf5_state_writer::write(std::string_view n, double v)        { record(write_scalar(current_loc(), n, v)); }
void hdf5_state_writer::write(std::string_view n, float v)         { record(write_scalar(current_loc(), n, v)); }
void hdf5_state_writer::write(std::string_view n, std::int32_t v)  { record(write_scalar(current_loc(), n, v)); }
void hdf5_state_writer::write(std::string_view n, std::int64_t v)  { record(write_scalar(current_loc(), n, v)); }
void hdf5_state_writer::write(std::string_view n, std::uint32_t v) { record(write_scalar(current_loc(), n, v)); }
void hdf5_state_writer::write(std::string_view n, std::uint64_t v) { record(write_scalar(current_loc(), n, v)); }
// HDF5 has no native bool: int8 0/1 (read back by read_bool).
void hdf5_state_writer::write(std::string_view n, bool v)
{ record(write_scalar(current_loc(), n, std::int8_t{ v ? std::int8_t{1} : std::int8_t{0} })); }

void hdf5_state_writer::write(std::string_view n, std::span<const float> v)         { record(write_vector(current_loc(), n, v)); }
void hdf5_state_writer::write(std::string_view n, std::span<const double> v)        { record(write_vector(current_loc(), n, v)); }
void hdf5_state_writer::write(std::string_view n, std::span<const std::int32_t> v)  { record(write_vector(current_loc(), n, v)); }
void hdf5_state_writer::write(std::string_view n, std::span<const std::uint32_t> v) { record(write_vector(current_loc(), n, v)); }
void hdf5_state_writer::write(std::string_view n, std::span<const std::uint64_t> v) { record(write_vector(current_loc(), n, v)); }
void hdf5_state_writer::write(std::string_view n, std::span<const std::uint16_t> v) { record(write_vector(current_loc(), n, v)); }
void hdf5_state_writer::write(std::string_view n, std::span<const std::uint8_t> v)  { record(write_vector(current_loc(), n, v)); }

void hdf5_state_writer::write_2d(std::string_view name, std::span<const float> values,
                                  std::size_t rows, std::size_t cols)
{
    if (rows == 0 || cols == 0) return;
    const hsize_t dims[2] = { rows, cols };
    record(write_array(current_loc(), name, values, std::span<const hsize_t>(dims)));
}

void hdf5_state_writer::write_attr(std::string_view name, std::string_view value)
{
    std::string name_str(name);
    // Route through record() like every dataset write: an attribute failure
    // (e.g. ENOSPC landing on H5Awrite) must sink ok_ so the commit path aborts
    // rather than publishing a checkpoint with silently-missing metadata.
    record(hdf5::write_attr(current_loc(), name_str.c_str(), value));
}

// --- hdf5_state_reader implementation ---

hdf5_state_reader::hdf5_state_reader(hdf5::file& file, std::uint32_t format_version)
    : state_reader(format_version), file_(file)
{
}

hdf5_state_reader::~hdf5_state_reader()
{
    group_stack_.clear();
}

hid_t hdf5_state_reader::current_loc() const
{
    if (group_stack_.empty()) {
        return file_.id();
    }
    return group_stack_.back().id();
}

bool hdf5_state_reader::enter_group(std::string_view name)
{
    std::string name_str(name);
    if (!hdf5::link_exists(current_loc(), name_str.c_str())) {
        return false;
    }
    auto g = hdf5::group::open(current_loc(), name_str.c_str());
    if (!g) return false;
    group_stack_.push_back(std::move(g));
    return true;
}

void hdf5_state_reader::leave_group()
{
    if (!group_stack_.empty()) {
        group_stack_.pop_back();
    }
}

bool hdf5_state_reader::has_field(std::string_view name)
{
    std::string name_str(name);
    return hdf5::link_exists(current_loc(), name_str.c_str());
}

namespace {

// The field readers take a checkpoint's shapes on trust no further than the
// writer produces them: rank 0 for scalars, rank 1 for vectors, rank 2 for 2-D
// arrays (hdf5::require_rank, which also refuses H5S_NULL). A missing field is
// the reader's default; a field that exists but is not a readable dataset of
// that shape and a convertible type throws (#150).

std::optional<hdf5::dataset> open_field(hid_t loc, std::string_view name)
{
    const std::string name_str(name);
    if (!hdf5::link_exists(loc, name_str.c_str())) return std::nullopt;
    auto ds = hdf5::dataset::open(loc, name_str.c_str());
    if (!ds) {
        throw std::runtime_error("hdf5: " + hdf5::object_path(loc) + "/" + name_str +
                                 ": exists but is not a readable dataset");
    }
    return ds;
}

template <class T>
std::optional<T> read_scalar_field(hid_t loc, std::string_view name, hid_t mem_type)
{
    const auto ds = open_field(loc, name);
    if (!ds) return std::nullopt;
    return hdf5::read_array<T>(ds->id(), mem_type, 0).values[0];
}

template <class T>
std::vector<T> read_vec_field(hid_t loc, std::string_view name, hid_t mem_type)
{
    const auto ds = open_field(loc, name);
    if (!ds) return {};
    return hdf5::read_array<T>(ds->id(), mem_type, 1).values;
}

} // namespace

double hdf5_state_reader::read_double(std::string_view name, double default_value)
{
    return read_scalar_field<double>(current_loc(), name, H5T_NATIVE_DOUBLE).value_or(default_value);
}

float hdf5_state_reader::read_float(std::string_view name, float default_value)
{
    return read_scalar_field<float>(current_loc(), name, H5T_NATIVE_FLOAT).value_or(default_value);
}

std::int32_t hdf5_state_reader::read_int32(std::string_view name, std::int32_t default_value)
{
    return read_scalar_field<std::int32_t>(current_loc(), name, H5T_NATIVE_INT32).value_or(default_value);
}

std::int64_t hdf5_state_reader::read_int64(std::string_view name, std::int64_t default_value)
{
    return read_scalar_field<std::int64_t>(current_loc(), name, H5T_NATIVE_INT64).value_or(default_value);
}

std::uint32_t hdf5_state_reader::read_uint32(std::string_view name, std::uint32_t default_value)
{
    return read_scalar_field<std::uint32_t>(current_loc(), name, H5T_NATIVE_UINT32).value_or(default_value);
}

std::uint64_t hdf5_state_reader::read_uint64(std::string_view name, std::uint64_t default_value)
{
    return read_scalar_field<std::uint64_t>(current_loc(), name, H5T_NATIVE_UINT64).value_or(default_value);
}

bool hdf5_state_reader::read_bool(std::string_view name, bool default_value)
{
    // HDF5 has no native bool; the writer stores int8.
    const auto v = read_scalar_field<std::int8_t>(current_loc(), name, H5T_NATIVE_INT8);
    return v ? *v != 0 : default_value;
}

std::vector<float> hdf5_state_reader::read_float_vec(std::string_view name)
{
    return read_vec_field<float>(current_loc(), name, H5T_NATIVE_FLOAT);
}

std::vector<double> hdf5_state_reader::read_double_vec(std::string_view name)
{
    return read_vec_field<double>(current_loc(), name, H5T_NATIVE_DOUBLE);
}

std::vector<std::int32_t> hdf5_state_reader::read_int32_vec(std::string_view name)
{
    return read_vec_field<std::int32_t>(current_loc(), name, H5T_NATIVE_INT32);
}

std::vector<std::uint32_t> hdf5_state_reader::read_uint32_vec(std::string_view name)
{
    return read_vec_field<std::uint32_t>(current_loc(), name, H5T_NATIVE_UINT32);
}

std::vector<std::uint64_t> hdf5_state_reader::read_uint64_vec(std::string_view name)
{
    return read_vec_field<std::uint64_t>(current_loc(), name, H5T_NATIVE_UINT64);
}

std::vector<std::uint16_t> hdf5_state_reader::read_uint16_vec(std::string_view name)
{
    return read_vec_field<std::uint16_t>(current_loc(), name, H5T_NATIVE_UINT16);
}

std::vector<std::uint8_t> hdf5_state_reader::read_uint8_vec(std::string_view name)
{
    return read_vec_field<std::uint8_t>(current_loc(), name, H5T_NATIVE_UINT8);
}

std::vector<float> hdf5_state_reader::read_float_2d(std::string_view name, array_2d_info& info)
{
    info = {};
    const auto ds = open_field(current_loc(), name);
    if (!ds) return {};
    auto a = hdf5::read_array<float>(ds->id(), H5T_NATIVE_FLOAT, 2);
    info.rows = a.dims[0];
    info.cols = a.dims[1];
    return std::move(a.values);
}

std::string hdf5_state_reader::read_attr(std::string_view name)
{
    const std::string name_str(name);
    return hdf5::read_attr_string(current_loc(), name_str.c_str()).value_or("");
}

std::vector<std::pair<std::string, scalar_value>> hdf5_state_reader::read_scalars()
{
    mem_group g;
    detail::parse_group(current_loc(), &g, /*recurse=*/false);
    return detail::scalars_of(g);
}

// --- state_logger: quiesced-state split ---
//
// The save / load monolith is decomposed into four pieces that respect the
// thread split the redesign needs (design §4–§5):
//
//   extract_state   sim thread   env state  -> in-memory tree   (no HDF5)
//   serialize_state executor     tree       -> HDF5 file        (HDF5 only)
//   parse_state     executor     HDF5 file  -> in-memory tree    (HDF5 only)
//   apply_state     sim thread   tree       -> env state        (no HDF5)
//
// The in-memory tree (mem_state_writer / mem_state_reader, mem_state_io.h) is
// the bridge: extraction touches the wgpu queue (GPU readbacks) and must run on
// the sim thread, while libhdf5 stays on the single executor worker. Runtime
// call sites compose these via sim_thread_controller::run_quiesced (extraction on
// the sim thread, serialize/parse on the executor) — see logging_manager. The
// save_state convenience below runs both halves on the caller's thread; it is
// used by build_env-free contexts and tests, and by the symmetric load_state
// monolith retained for the build_env startup exemption (design §5.2).
//
// While a serialize/parse job runs, observe writers and buffer_logger writes
// posted to the same executor wait their turn — the intended behaviour (see
// design/hdf5-executor.md §8). For typical checkpoint intervals (10⁴ steps) and
// save durations (~1s), the producer-side stall is negligible.

mem_group state_logger::extract_state(sim::env& env, std::uint64_t step_count)
{
    mem_state_writer writer;

    // Env-derived metadata. The save-event metadata (config, timestamp) is added
    // by serialize_state — it derives from the save config / wall clock, which
    // extraction does not have.
    writer.begin_group("metadata");
    writer.write_attr("schema_version", std::to_string(k_format_version));
    writer.write_attr("type", std::string_view("simulation_state"));
    writer.write("step_count", step_count);
    writer.write_attr("run_id", env.run_id());
    writer.end_group();

    // IO buffer read pages. snapshot() reads the read page for every variant (CPU
    // page, CPU mirror, or on-demand GPU readback), so there is no per-variant
    // branching. On failure (e.g. a transient GPU readback fault) record an empty
    // dataset so the save still completes — matching the prior monolith.
    writer.begin_group("io_buffers");
    for (auto& [id, buffer] : env.get_io_buffers()) {
        writer.begin_group(std::to_string(id));
        std::vector<float> tmp(buffer->size());
        if (!tmp.empty() && buffer->snapshot(tmp)) {
            writer.write("data", std::span<const float>(tmp));
        } else {
            writer.write("data", std::span<const float>{});
        }
        writer.end_group();
    }
    writer.end_group();

    // Entity state (entity.save_state includes GPU entities' own readbacks).
    writer.begin_group("entities");
    for (auto& entity : env.iterate_entities()) {
        writer.begin_group(std::to_string(entity.get_outp_id()));
        writer.write_attr("type", entity.type_id().name());
        entity.save_state(writer);
        writer.end_group();
    }
    writer.end_group();

    return writer.take_root();
}

std::future<log_result> state_logger::serialize_state(mem_group                tree,
                                                      const state_save_config& config,
                                                      hdf5::executor&          h5_exec)
{
    // Add the save-event metadata the extractor could not know. Both are plain
    // string attrs on the (already present) metadata group; no HDF5 here — this
    // runs on the caller's thread before the tree is handed off.
    {
        mem_group& meta = tree.groups["metadata"];
        if (!config.config_json.empty()) {
            meta.attrs.insert_or_assign("config", config.config_json);
        }
        meta.attrs.insert_or_assign("timestamp",
            core::format_local_time(std::chrono::system_clock::now(), core::k_ctime_format));
    }

    auto prom = std::make_shared<std::promise<log_result>>();
    auto fut  = prom->get_future();

    // The tree is owned by the job: moved into a shared_ptr the job captures (the
    // wrapping keeps the std::function copyable; design §5.1). After handoff the
    // sim thread never touches it; the worker only ever touches the moved tree.
    auto owned = std::make_shared<mem_group>(std::move(tree));
    std::string path = config.output_path.string();

    // submit() (fire-and-forget) gives the bounded-queue backpressure the design
    // wants. The executor's own try/catch only logs an escaping exception, so the
    // job must fulfil the promise on every path or fut.get() would hang.
    //
    // The HDF5 work is done in an inner scope so `file` (and `writer`) are fully
    // DESTROYED — i.e. H5Fclose has run — BEFORE the promise is fulfilled. A
    // waiter that blocks on the future (headless: record_savestate file_size())
    // must observe the final on-disk file, not the larger pre-close flushed image
    // (HDF5 compacts on close). The old monolithic submit_sync had this ordering
    // for free; serialize_state must reproduce it explicitly.
    h5_exec.submit([prom, owned, path] {
        // Atomic-durable write: build the file at <path>.tmp, then fsync it,
        // rename it over the target, and fsync the directory. After the rename the
        // newest COMPLETE state file is intact by construction — a crash mid-write
        // leaves only the stale .tmp (never a torn target), so a supervisor that
        // resumes from the newest file never resumes from a half-written one. The
        // fsync fence closes the ext4 empty-file-after-rename window (rename is
        // atomic vs concurrent readers, but not durable vs power loss on its own).
        //
        // The tmp name is PER-WRITER, not a fixed "<path>.tmp". Two processes
        // writing the same target — two coast runs sharing a --save-state path, an
        // operator re-running while a job is finishing — would otherwise share one
        // scratch file, and the second H5Fcreate(H5F_ACC_TRUNC) truncates the
        // first's in-flight file: both then rename garbage over the target, or the
        // loser's rename undoes the winner's. Checkpoints carry a timestamp+step in
        // their name and are effectively unique, but --save-state is a path the
        // USER fixes, so this is reachable without anything unusual happening.
        // (coastd/src/atomic_write.cpp fixed the same defect in its own primitive;
        // this sibling was not brought along. pid+tid needs no clock or RNG.)
        const auto tid_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());
        const std::string tmp = path + ".p" + std::to_string(::getpid()) + "t" +
                                std::to_string(tid_hash) + ".tmp";
        log_result result = [&]() -> log_result {
            try {
                {
                    auto file = hdf5::file::create(tmp);
                    if (!file) {
                        return log_result::error("Failed to create state file: " + tmp);
                    }
                    hdf5_state_writer writer(file);
                    replay(*owned, writer);
                    // A failed H5Dwrite/H5Gcreate (e.g. ENOSPC) returns an error
                    // that the individual writes swallow; consult the writer's
                    // sticky flag, then flush and close EXPLICITLY and check both
                    // — H5Fclose can surface a deferred write error too. Any
                    // failure here means the .tmp is truncated, so we must NOT
                    // rename it over the last-good target: remove it and report.
                    if (!writer.ok()) {
                        std::error_code ec;
                        std::filesystem::remove(tmp, ec);
                        return log_result::error("State write failed (incomplete): " + tmp);
                    }
                    if (file.flush() < 0) {
                        std::error_code ec;
                        std::filesystem::remove(tmp, ec);
                        return log_result::error("State flush failed: " + tmp);
                    }
                    if (file.close() < 0) {
                        std::error_code ec;
                        std::filesystem::remove(tmp, ec);
                        return log_result::error("State close failed (deferred write error): " + tmp);
                    }
                }   // H5Fclose already ran above (file.close()); destructor no-ops

                // fsync the tmp so the bytes are durable before the rename. An
                // open/fsync/close failure here is a data-durability failure, so
                // it aborts the commit (mirrors coastd::atomic_write).
                const int fd = ::open(tmp.c_str(), O_RDONLY);
                if (fd < 0) {
                    std::error_code ec;
                    std::filesystem::remove(tmp, ec);
                    return log_result::error("State fsync open failed: " + tmp);
                }
                const bool fsync_ok = (::fsync(fd) == 0);
                const bool close_ok = (::close(fd) == 0);
                if (!fsync_ok || !close_ok) {
                    std::error_code ec;
                    std::filesystem::remove(tmp, ec);
                    return log_result::error("State fsync failed: " + tmp);
                }
                std::error_code ec;
                std::filesystem::rename(tmp, path, ec);
                if (ec) {
                    // Preserve the rename error before the cleanup remove overwrites it.
                    std::error_code rm_ec;
                    std::filesystem::remove(tmp, rm_ec);
                    return log_result::error("State rename failed: " + ec.message());
                }
                // Directory fsync: the step that makes the RENAME durable, i.e. the
                // one that closes the empty-file-after-rename window this whole
                // sequence exists for. Its result is part of the guarantee, so it is
                // reported like every other step above rather than discarded —
                // ok() must not claim a durability that was not achieved.
                //
                // The file itself is already renamed into place at this point, so a
                // failure here is NOT a reason to delete it: the data is complete
                // and readable, only the directory entry's durability across an
                // unclean reboot is unproven. Report the failure and leave the file.
                std::filesystem::path dir = std::filesystem::path(path).parent_path();
                if (dir.empty()) dir = ".";
                const int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
                if (dfd < 0)
                    return log_result::error("State dir fsync open failed: " + dir.string()
                                             + " (" + std::strerror(errno) + ")");
                const bool dir_sync_ok = (::fsync(dfd) == 0);
                const int  dir_errno   = errno;
                ::close(dfd);
                if (!dir_sync_ok)
                    return log_result::error("State dir fsync failed: " + dir.string()
                                             + " (" + std::strerror(dir_errno) + ")");
                return log_result::ok();
            } catch (const std::exception& e) {
                std::error_code ec;
                std::filesystem::remove(tmp, ec);
                return log_result::error(std::string("State serialization failed: ") + e.what());
            } catch (...) {
                std::error_code ec;
                std::filesystem::remove(tmp, ec);
                return log_result::error("State serialization failed (unknown exception)");
            }
        }();   // file + writer closed here, before the promise is set
        prom->set_value(std::move(result));
    });

    return fut;
}

namespace {

// The file half of every load, on an open file: in startup's order, the version
// gate first, since a newer format may hold objects this build's parse refuses;
// then a malformed file (anything a state writer never produces, see
// parse_hdf5_to_tree) throws from the parse, and a metadata field of another rank
// (a rank-1 step_count) from its read. Older versions load without a warning;
// what differs is for each load_state to gate on (apply_state builds the reader
// with the version). Executor job only.
state_logger::parse_result parse_open_file(hdf5::file& file)
{
    state_logger::parse_result pr;
    const auto version = read_format_version(file);
    if (!version) {
        pr.result.error_message = version.error();
        return pr;
    }
    pr.tree = parse_hdf5_to_tree(file);
    hdf5_state_reader metadata(file, *version);
    metadata.enter_group("metadata");
    pr.result.restored_step = metadata.read_uint64("step_count", 0);
    pr.result.config_json   = metadata.read_attr("config");
    pr.result.success = true;
    return pr;
}

} // namespace

state_logger::parse_result state_logger::parse_state(const std::filesystem::path& path,
                                                     hdf5::executor&              h5_exec)
{
    return h5_exec.submit_sync([&]() -> parse_result {
        auto file = hdf5::file::open_read(path.string());
        if (!file) {
            parse_result pr;
            pr.result.error_message = "Failed to open state file: " + path.string();
            return pr;
        }
        try {
            return parse_open_file(file);
        } catch (const std::exception& e) {
            parse_result pr;
            pr.result.error_message = "state file " + path.string() + ": " + e.what();
            return pr;
        }
    });
}

state_load_result state_logger::apply_state(sim::env& env, const mem_group& tree)
{
    state_load_result result;

    // The format version and restored_step from the tree's metadata (parse_state
    // also checks them; apply re-reads so it is usable standalone, e.g. in tests).
    // A tree without metadata is refused, as parse_state refuses it.
    const auto version = read_format_version(tree);
    if (!version) {
        result.error_message = version.error();
        return result;
    }
    mem_state_reader reader(tree, *version);
    reader.enter_group("metadata");
    result.restored_step = reader.read_uint64("step_count", 0);
    result.config_json   = reader.read_attr("config");
    // Before any entity loads: a dataset continues its sequence only when this
    // load continues the run that wrote the checkpoint (io/dataset_run.h, SPEC.md
    // S.5.14). Absent before format version 6: never.
    const std::string checkpoint_run = reader.read_attr("run_id");
    env.maybe_adopt_run_id(checkpoint_run);
    const bool continues = !checkpoint_run.empty() && checkpoint_run == env.run_id();
    reader.set_continues_run(continues);
    env.set_continued_from_step(continues ? std::optional{ result.restored_step } : std::nullopt);
    if (continues)
        COAST_INFO("state", "run {} continues: datasets continue their sequences", env.run_id());
    else
        // Nothing resets them: at startup they are fresh, a load into a running
        // session keeps where it is.
        COAST_INFO("state", "checkpoint of run {} loaded into run {}: datasets keep their own "
                   "sequences (name the checkpoint's run_id in the config to continue)",
                   checkpoint_run.empty() ? std::string("(none recorded)") : checkpoint_run,
                   env.run_id());
    reader.leave_group();

    // IO buffers. restore() writes the read page directly for every variant (no
    // swap — design §5.3), so there is no per-variant branching and no reliance on
    // a later swap. A size mismatch or failed GPU upload is reported, not silently
    // dropped — the buffer then keeps its zero-init.
    if (reader.enter_group("io_buffers")) {
        for (auto& [id, buffer] : env.get_io_buffers()) {
            std::string id_str = std::to_string(id);
            if (reader.enter_group(id_str)) {
                auto data = reader.read_float_vec("data");
                if (data.size() != buffer->size()) {
                    result.warnings.push_back("Buffer " + id_str + " size mismatch");
                } else if (buffer->size() > 0 && !buffer->restore(data)) {
                    result.warnings.push_back("Buffer " + id_str + " restore failed");
                }
                reader.leave_group();
            }
        }
        reader.leave_group();
    }

    // Entity state. The groups of a checkpoint written before configuration
    // records existed are noted on the reader and reported in one notice (M.7.8).
    if (reader.enter_group("entities")) {
        for (auto& entity : env.iterate_entities()) {
            std::string id_str = std::to_string(entity.get_outp_id());
            if (reader.enter_group(id_str)) {
                entity.load_state(reader);
                reader.leave_group();
            }
        }
        reader.leave_group();
        logging::report_unrecorded_configs(reader);
        logging::report_unrecorded_channels(reader);
        logging::report_unchecked_dendrite_types(reader);
    }

    result.success = true;
    return result;
}

log_result state_logger::save_state(sim::env&                env,
                                    std::uint64_t            step_count,
                                    const state_save_config& config,
                                    hdf5::executor&        h5_exec)
{
    // Single source of truth for the tree shape: extract, then serialize and await
    // the write so the file exists on return. Step 4 splits these across the sim
    // thread (extract) and the executor (serialize); here both run on the caller.
    mem_group tree = extract_state(env, step_count);
    auto      fut  = serialize_state(std::move(tree), config, h5_exec);
    return fut.get();
}

state_load_result state_logger::load_state(sim::env&                    env,
                                           const std::filesystem::path& path,
                                           hdf5::executor&            h5_exec)
{
    // The startup load (build_env, before the sim thread steps): the runtime
    // path's parse_state + apply_state, run back to back in ONE executor job --
    // the startup exemption (design §5.2) -- so both paths refuse the same files
    // and apply them the same way, by construction.
    return h5_exec.submit_sync([&]() -> state_load_result {
        auto file = hdf5::file::open_read(path.string());
        if (!file) {
            state_load_result failed;
            failed.error_message = "Failed to open state file: " + path.string();
            return failed;
        }
        // A malformed file throws from the parse, before any entity is loaded.
        // An entity that refuses its state throws from apply_state, possibly
        // after others were loaded; at startup (the only caller) the half-loaded
        // env is discarded with the process. The message has no path: build_env
        // prefixes it.
        try {
            const parse_result pr = parse_open_file(file);
            if (!pr.result) return pr.result;
            return apply_state(env, pr.tree);
        } catch (const std::exception& e) {
            state_load_result failed;
            failed.error_message = e.what();
            return failed;
        }
    });
}

} // namespace logging
