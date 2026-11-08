// Unit test — observe N-D (multi-dimensional) value capture.
//
// The new observe() overload accepts an explicit shape descriptor; the writer
// must create the value dataset with per-row shape [d0, d1, ...] so the on-disk
// dataset is [rows, d0, d1, ...]. This drives the writer with a hand-built 2-D
// (shape [2,3]) site and reads the file back, asserting both the dataset rank/
// extents and the values.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "observe/observe_writer.h"
#include "observe/observe_site.h"
#include "hdf5/executor.h"

#include <hdf5.h>

#include <cstdint>
#include <filesystem>
#include <system_error>
#include <vector>

TEST_CASE("observe N-D: a [2,3] site round-trips as an HDF5 [rows,2,3] dataset") {
    namespace fs = std::filesystem;

    const fs::path h5_path = fs::temp_directory_path() / "coast_observe_nd.h5";
    std::error_code ec;
    fs::remove(h5_path, ec);

    observe::observe_site_t site;
    site.full_id     = "grp/tensor";
    site.h5_dtype    = H5T_NATIVE_FLOAT;
    site.shape       = {2, 3};                       // per-row shape
    site.value_bytes = 6 * sizeof(float);

    const std::vector<std::vector<float>> rows{
        {1, 2, 3, 4, 5, 6},
        {7, 8, 9, 10, 11, 12},
    };
    const std::vector<std::uint64_t> steps{5, 6};

    {
        hdf5::executor exec;
        observe::observe_writer_t writer(h5_path, exec);
        writer.start();
        for (std::size_t i = 0; i < rows.size(); ++i) {
            writer.enqueue(&site, rows[i].data(), site.value_bytes, steps[i]);
        }
        writer.stop_and_join();
    }

    CHECK(site.rows_written == rows.size());

    REQUIRE(fs::exists(h5_path));
    const hid_t file = H5Fopen(h5_path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);

    const hid_t dset = H5Dopen2(file, "/grp__tensor", H5P_DEFAULT);
    REQUIRE(dset >= 0);

    const hid_t sp = H5Dget_space(dset);
    CHECK(H5Sget_simple_extent_ndims(sp) == 3);      // [rows, 2, 3]
    hsize_t dims[3] = {0, 0, 0};
    H5Sget_simple_extent_dims(sp, dims, nullptr);
    CHECK(dims[0] == rows.size());
    CHECK(dims[1] == 2);
    CHECK(dims[2] == 3);

    std::vector<float> read(static_cast<std::size_t>(dims[0] * dims[1] * dims[2]), 0.0f);
    REQUIRE(H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, read.data()) >= 0);
    for (std::size_t r = 0; r < rows.size(); ++r)
        for (std::size_t k = 0; k < 6; ++k)
            CHECK(read[r * 6 + k] == doctest::Approx(rows[r][k]));

    H5Sclose(sp);
    H5Dclose(dset);
    H5Fclose(file);
    fs::remove(h5_path, ec);
}
