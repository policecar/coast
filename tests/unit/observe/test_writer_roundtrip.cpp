// Test 1.5 — observe writer -> HDF5 round-trip.
//
// This is the single highest-value observe test: it proves that captured data
// actually lands on disk in the expected layout. It exercises the real
// hdf5::executor (its worker thread) and a temp .h5 file, then reads the file
// back with the HDF5 C API and checks the on-disk structure and contents.
//
// Tier note: strictly this is an *in-process integration* test (it spins the
// executor worker thread and touches the filesystem), not a pure unit test.
// It lives under tests/unit/ only because that is where the lightweight
// doctest executables are wired; it has no sim dependency and runs sub-second.
//
// HDF5 threading: libhdf5 is built Threadsafety:OFF (one thread in the library
// at a time, process-wide). The executor's worker does all writes; we read back
// on the main thread only AFTER the executor has been destroyed (worker joined),
// so the two never touch libhdf5 concurrently.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "observe/observe_writer.h"
#include "observe/observe_site.h"
#include "hdf5/executor.h"

#include <hdf5.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

TEST_CASE("observe writer: enqueue then readback round-trips values and step indices") {
    namespace fs = std::filesystem;

    const fs::path h5_path = fs::temp_directory_path() / "coast_observe_writer_roundtrip.h5";
    std::error_code ec;
    fs::remove(h5_path, ec);  // start clean (H5Fcreate also truncates)

    // Hand-built scalar-float site. The '/' in full_id exercises the
    // hierarchical-to-flat dataset-name mapping ('/' -> '__').
    observe::observe_site_t site;
    site.full_id     = "neuron[3]/activity";
    site.h5_dtype    = H5T_NATIVE_FLOAT;
    site.shape       = {};  // scalar
    site.value_bytes = sizeof(float);

    const std::vector<float>         values{0.25f, 0.5f, 0.75f};
    const std::vector<std::uint64_t> steps {10, 20, 30};

    {
        hdf5::executor exec;
        observe::observe_writer_t writer(h5_path, exec);
        writer.start();
        for (std::size_t i = 0; i < values.size(); ++i) {
            // The accessor hands the value pointer straight to the writer,
            // which copies it into an owned buffer before returning.
            writer.enqueue(&site, &values[i], sizeof(float), steps[i]);
        }
        writer.stop_and_join();                // synchronous drain + close
    } // writer + executor destroyed; worker joined

    CHECK(site.rows_written == values.size());

    // ---- read the file back with the HDF5 C API ----
    REQUIRE(fs::exists(h5_path));
    const hid_t file = H5Fopen(h5_path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);

    // family stamp lives in /metadata, uniform with the buffer and probe
    // loggers (readers key on /metadata@type)
    {
        const hid_t md = H5Gopen2(file, "metadata", H5P_DEFAULT);
        REQUIRE(md >= 0);

        char        buf[32] = {};
        const hid_t attr    = H5Aopen(md, "type", H5P_DEFAULT);
        REQUIRE(attr >= 0);
        const hid_t atype = H5Aget_type(attr);
        REQUIRE(H5Aread(attr, atype, buf) >= 0);
        CHECK(std::string_view{buf} == "observe_log");
        H5Tclose(atype);
        H5Aclose(attr);

        char        vbuf[16] = {};
        const hid_t vattr    = H5Aopen(md, "schema_version", H5P_DEFAULT);
        REQUIRE(vattr >= 0);
        const hid_t vtype = H5Aget_type(vattr);
        REQUIRE(H5Aread(vattr, vtype, vbuf) >= 0);
        CHECK(std::string_view{vbuf} == "1.0");
        H5Tclose(vtype);
        H5Aclose(vattr);

        H5Gclose(md);
    }

    // value dataset: '/' in full_id became '__'
    {
        const hid_t dset = H5Dopen2(file, "/neuron[3]__activity", H5P_DEFAULT);
        REQUIRE(dset >= 0);

        const hid_t sp = H5Dget_space(dset);
        hsize_t dims[1] = {0};
        H5Sget_simple_extent_dims(sp, dims, nullptr);
        CHECK(dims[0] == values.size());

        std::vector<float> read(static_cast<std::size_t>(dims[0]), 0.0f);
        REQUIRE(H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                        read.data()) >= 0);
        for (std::size_t i = 0; i < values.size(); ++i) {
            CHECK(read[i] == doctest::Approx(values[i]));
        }
        H5Sclose(sp);
        H5Dclose(dset);
    }

    // sibling step-index dataset
    {
        const hid_t sdset = H5Dopen2(file, "/neuron[3]__activity__step_index", H5P_DEFAULT);
        REQUIRE(sdset >= 0);

        const hid_t sp = H5Dget_space(sdset);
        hsize_t dims[1] = {0};
        H5Sget_simple_extent_dims(sp, dims, nullptr);
        CHECK(dims[0] == steps.size());

        std::vector<std::uint64_t> read(static_cast<std::size_t>(dims[0]), 0);
        REQUIRE(H5Dread(sdset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                        read.data()) >= 0);
        for (std::size_t i = 0; i < steps.size(); ++i) {
            CHECK(read[i] == steps[i]);
        }
        H5Sclose(sp);
        H5Dclose(sdset);
    }

    H5Fclose(file);
    fs::remove(h5_path, ec);
}

// SPEC.md S.5.14: a run that continues from a checkpoint continues its observe
// files. Its rows for the steps from the checkpoint on are dropped (those steps run
// again) and the new rows follow the rest; a file of another run, or none, or an
// unreadable one, is created afresh.
namespace {

namespace fs = std::filesystem;

observe::observe_site_t scalar_site()
{
    observe::observe_site_t site;
    site.full_id     = "g/x";
    site.h5_dtype    = H5T_NATIVE_FLOAT;
    site.value_bytes = sizeof(float);
    return site;
}

// Writes one row per step in [first, last), the value equal to the step.
void write_steps(const fs::path& path, observe::observe_run_t run,
                 std::uint64_t first, std::uint64_t last)
{
    observe::observe_site_t site = scalar_site();   // a fresh process: no open handles
    hdf5::executor exec;
    observe::observe_writer_t writer(path, exec, std::move(run));
    writer.start();
    for (std::uint64_t s = first; s < last; ++s) {
        const float v = static_cast<float>(s);
        writer.enqueue(&site, &v, sizeof v, s);
    }
    writer.stop_and_join();
}

struct rows { std::vector<std::uint64_t> steps; std::vector<float> values; std::string run_id; };

rows read_rows(const fs::path& path)
{
    rows r;
    const hid_t file = H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    const auto read = [&](const char* name, hid_t type, auto& out) {
        const hid_t d = H5Dopen2(file, name, H5P_DEFAULT);
        REQUIRE(d >= 0);
        const hid_t sp = H5Dget_space(d);
        hsize_t n = 0;
        H5Sget_simple_extent_dims(sp, &n, nullptr);
        out.resize(static_cast<std::size_t>(n));
        if (n > 0) REQUIRE(H5Dread(d, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) >= 0);
        H5Sclose(sp);
        H5Dclose(d);
    };
    read("/g__x__step_index", H5T_NATIVE_UINT64, r.steps);
    read("/g__x", H5T_NATIVE_FLOAT, r.values);
    const hid_t md = H5Gopen2(file, "metadata", H5P_DEFAULT);
    if (H5Aexists(md, "run_id") > 0) {
        const hid_t a = H5Aopen(md, "run_id", H5P_DEFAULT);
        const hid_t t = H5Aget_type(a);
        std::string s(H5Tget_size(t), '\0');
        H5Aread(a, t, s.data());
        r.run_id = s.c_str();
        H5Tclose(t);
        H5Aclose(a);
    }
    H5Gclose(md);
    H5Fclose(file);
    return r;
}

std::vector<std::uint64_t> iota(std::uint64_t first, std::uint64_t last)
{
    std::vector<std::uint64_t> v;
    for (auto s = first; s < last; ++s) v.push_back(s);
    return v;
}

} // namespace

TEST_CASE("observe writer: a continued run keeps its rows before the checkpoint and appends") {
    const fs::path path = fs::temp_directory_path() / "coast_observe_writer_continue.h5";
    std::error_code ec;
    fs::remove(path, ec);

    // First launch: steps 100..129; checkpoint at 120; killed after step 129.
    write_steps(path, { .run_id = "r1", .from_step = std::nullopt }, 100, 130);
    CHECK(read_rows(path).run_id == "r1");
    // The relaunch continues from 120: 120..129 run again, then on to 140.
    write_steps(path, { .run_id = "r1", .from_step = 120 }, 120, 140);
    auto r = read_rows(path);
    CHECK(r.steps == iota(100, 140));                  // no gap, no repeated step
    REQUIRE(r.values.size() == r.steps.size());
    for (std::size_t i = 0; i < r.steps.size(); ++i) CHECK(r.values[i] == static_cast<float>(r.steps[i]));

    // Another run continuing from 120 does not continue r1's file: created afresh.
    write_steps(path, { .run_id = "r2", .from_step = 120 }, 120, 125);
    r = read_rows(path);
    CHECK(r.steps == iota(120, 125));
    CHECK(r.run_id == "r2");

    // No continuation: afresh, as before, even for the same run.
    write_steps(path, { .run_id = "r2", .from_step = std::nullopt }, 0, 3);
    CHECK(read_rows(path).steps == iota(0, 3));

    // An unreadable file (as a crash can leave it): created afresh, no throw.
    { std::ofstream(path, std::ios::trunc) << "not hdf5"; }
    write_steps(path, { .run_id = "r2", .from_step = 1 }, 1, 4);
    CHECK(read_rows(path).steps == iota(1, 4));
    fs::remove(path, ec);
}

TEST_CASE("observe writer: a continued file whose site changed type starts that site afresh") {
    const fs::path path = fs::temp_directory_path() / "coast_observe_writer_retype.h5";
    std::error_code ec;
    fs::remove(path, ec);

    write_steps(path, { .run_id = "r1", .from_step = std::nullopt }, 0, 10);   // float rows
    {
        // The same run continues from step 5, but the site now records doubles.
        observe::observe_site_t site = scalar_site();
        site.h5_dtype    = H5T_NATIVE_DOUBLE;
        site.value_bytes = sizeof(double);
        hdf5::executor exec;
        observe::observe_writer_t writer(path, exec, { .run_id = "r1", .from_step = 5 });
        writer.start();
        for (std::uint64_t s = 5; s < 8; ++s) {
            const double v = static_cast<double>(s);
            writer.enqueue(&site, &v, sizeof v, s);
        }
        writer.stop_and_join();
    }
    // The old rows are gone and the site holds the new rows only, as doubles.
    const auto r = read_rows(path);
    CHECK(r.steps == iota(5, 8));
    CHECK(r.values == std::vector<float>{ 5.f, 6.f, 7.f });
    const hid_t file = H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    const hid_t d = H5Dopen2(file, "/g__x", H5P_DEFAULT);
    const hid_t t = H5Dget_type(d);
    CHECK(H5Tequal(t, H5T_NATIVE_DOUBLE) > 0);
    H5Tclose(t);
    H5Dclose(d);
    H5Fclose(file);
    fs::remove(path, ec);
}
