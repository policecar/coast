// Unit test — observe chunk-geometry policy.
//
// Two parts:
//   1. pick_chunk_rows() boundary behaviour (pure integer math, no HDF5).
//   2. A writer round-trip on a wide (array-shaped) site, reading the on-disk
//      H5Pget_chunk() back to prove the time-axis chunk actually shrank from
//      the historical 1024 — i.e. the geometry the writer wrote matches the
//      policy, so a chunk stays near the 1 MiB budget instead of thrashing.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "observe/observe_chunk.h"
#include "observe/observe_writer.h"
#include "observe/observe_site.h"
#include "hdf5/executor.h"

#include <hdf5.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <system_error>
#include <vector>

TEST_CASE("pick_chunk_rows: scalar/small rows clamp to max_rows (historical 1024)") {
    // 4-byte scalar float: 1MiB/4 = 262144, clamped to 1024 -> unchanged.
    CHECK(observe::pick_chunk_rows(sizeof(float)) == 1024);
    // 1-byte row: 1MiB, clamped to 1024.
    CHECK(observe::pick_chunk_rows(1) == 1024);
    // Just under the point where 1024 rows would exceed 1 MiB:
    // 1MiB/1024 = 1024 bytes/row is the break-even; <=1024 keeps 1024 rows.
    CHECK(observe::pick_chunk_rows(1024) == 1024);
}

TEST_CASE("pick_chunk_rows: large rows shrink so chunk stays near the budget") {
    // 16 KiB row (e.g. 4096 floats): 1MiB/16KiB = 64 rows.
    CHECK(observe::pick_chunk_rows(4096 * sizeof(float)) == 64);
    // 2 KiB row: 1MiB/2KiB = 512 rows (below the 1024 clamp).
    CHECK(observe::pick_chunk_rows(2048) == 512);
    // Row larger than the whole budget: floor to 1 row.
    CHECK(observe::pick_chunk_rows((std::size_t{1} << 20) + 1) == 1);
    CHECK(observe::pick_chunk_rows(std::size_t{4} << 20) == 1);
}

TEST_CASE("pick_chunk_rows: degenerate zero row_bytes returns max_rows (no div-by-zero)") {
    CHECK(observe::pick_chunk_rows(0) == 1024);
}

TEST_CASE("pick_chunk_rows: explicit target/max args honoured") {
    // 256 KiB budget, 1 KiB row -> 256 rows.
    CHECK(observe::pick_chunk_rows(1024, std::size_t{256} << 10) == 256);
    // max_rows clamp below the byte-budget result.
    CHECK(observe::pick_chunk_rows(sizeof(float), std::size_t{1} << 20, 100) == 100);
}

TEST_CASE("observe writer: a wide site is created with the shrunk time-axis chunk") {
    namespace fs = std::filesystem;

    const fs::path h5_path = fs::temp_directory_path() / "coast_observe_chunk_geom.h5";
    std::error_code ec;
    fs::remove(h5_path, ec);

    // 4096-float row = 16 KiB. Historical geometry would chunk at [1024, 4096]
    // = 16 MiB/chunk (>> 1 MiB cache). Policy should pick 64 rows.
    constexpr std::size_t k_width = 4096;
    observe::observe_site_t site;
    site.full_id     = "population/out";
    site.h5_dtype    = H5T_NATIVE_FLOAT;
    site.shape       = {static_cast<hsize_t>(k_width)};
    site.value_bytes = k_width * sizeof(float);

    const std::vector<float> row(k_width, 1.0f);

    {
        hdf5::executor exec;
        observe::observe_writer_t writer(h5_path, exec);
        writer.start();
        // A few rows is enough; we only inspect the stored chunk geometry.
        // enqueue() takes the value + its byte size and copies both into an
        // owned buffer on the executor thread.
        for (std::uint64_t s = 0; s < 3; ++s) {
            writer.enqueue(&site, row.data(), site.value_bytes, s);
        }
        writer.stop_and_join();
    }

    REQUIRE(fs::exists(h5_path));
    const hid_t file = H5Fopen(h5_path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);

    const hid_t dset = H5Dopen2(file, "/population__out", H5P_DEFAULT);
    REQUIRE(dset >= 0);

    const hid_t dcpl = H5Dget_create_plist(dset);
    REQUIRE(dcpl >= 0);

    hsize_t chunk_dims[2] = {0, 0};
    const int rank = H5Pget_chunk(dcpl, 2, chunk_dims);
    CHECK(rank == 2);                              // [chunk_rows, width]
    CHECK(chunk_dims[0] == observe::pick_chunk_rows(site.value_bytes)); // 64
    CHECK(chunk_dims[0] < 1024);                   // proves it shrank from old geometry
    CHECK(chunk_dims[1] == k_width);

    H5Pclose(dcpl);
    H5Dclose(dset);
    H5Fclose(file);
    fs::remove(h5_path, ec);
}
