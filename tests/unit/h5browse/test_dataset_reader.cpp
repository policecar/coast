// h5browse dataset_reader — read_block + depth-2 read_pipeline correctness
// against a known value pattern, including multi-batch streaming, partial
// ranges, 2-D shape-matched reads, and sticky error behavior.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "h5browse/dataset_reader.h"
#include "h5browse/file_model.h"
#include "hdf5/executor.h"
#include "logging/hdf5_utils.h"

#include <hdf5.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string_view>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Raw chunked file with the observe pairing convention so open_and_map
// yields real dataset_descs:
//   /act              f32[3M],       chunk 1024, act[i] = i % 977
//   /act__step_index  u64[3M],       chunk 1024, steps[i] = i
//   /vec              f32[500 x 8],  chunk [16,8], vec[r][c] = r*10 + c
constexpr std::uint64_t n_rows_1d = 3'000'000; // 12 MB -> 3 pipeline batches
constexpr std::uint64_t n_rows_2d = 500;

fs::path make_test_file(hdf5::executor& exec)
{
    const fs::path  p = fs::temp_directory_path() / "h5b_reader_test.h5";
    std::error_code ec;
    fs::remove(p, ec);

    exec.submit_sync([&] {
        const hid_t fid = H5Fcreate(p.string().c_str(), H5F_ACC_TRUNC,
                                    H5P_DEFAULT, H5P_DEFAULT);
        REQUIRE(fid >= 0);

        // Uniform /metadata@type stamp open_and_map keys on (observe convention).
        {
            const hid_t md = H5Gcreate2(fid, "metadata", H5P_DEFAULT, H5P_DEFAULT,
                                        H5P_DEFAULT);
            REQUIRE(hdf5::write_attr(md, "type", std::string_view{"observe_log"}) >= 0);
            REQUIRE(hdf5::write_attr(md, "schema_version", std::string_view{"1.0"}) >= 0);
            H5Gclose(md);
        }

        {
            const hsize_t dims[1]  = {n_rows_1d};
            const hsize_t chunk[1] = {1024};
            const hid_t   sp   = H5Screate_simple(1, dims, nullptr);
            const hid_t   dcpl = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(dcpl, 1, chunk);

            std::vector<float> vals(n_rows_1d);
            for (std::uint64_t i = 0; i < n_rows_1d; ++i) {
                vals[i] = static_cast<float>(i % 977);
            }
            const hid_t d = H5Dcreate2(fid, "act", H5T_NATIVE_FLOAT, sp,
                                       H5P_DEFAULT, dcpl, H5P_DEFAULT);
            H5Dwrite(d, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     vals.data());
            H5Dclose(d);

            std::vector<std::uint64_t> steps(n_rows_1d);
            std::iota(steps.begin(), steps.end(), 0);
            const hid_t d2 = H5Dcreate2(fid, "act__step_index",
                                        H5T_NATIVE_UINT64, sp, H5P_DEFAULT,
                                        dcpl, H5P_DEFAULT);
            H5Dwrite(d2, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     steps.data());
            H5Dclose(d2);
            H5Pclose(dcpl);
            H5Sclose(sp);
        }
        {
            const hsize_t dims[2]  = {n_rows_2d, 8};
            const hsize_t chunk[2] = {16, 8};
            const hid_t   sp   = H5Screate_simple(2, dims, nullptr);
            const hid_t   dcpl = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(dcpl, 2, chunk);
            std::vector<float> vals(n_rows_2d * 8);
            for (std::uint64_t r = 0; r < n_rows_2d; ++r) {
                for (std::uint64_t c = 0; c < 8; ++c) {
                    vals[r * 8 + c] = static_cast<float>(r * 10 + c);
                }
            }
            const hid_t d = H5Dcreate2(fid, "vec", H5T_NATIVE_FLOAT, sp,
                                       H5P_DEFAULT, dcpl, H5P_DEFAULT);
            H5Dwrite(d, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     vals.data());
            H5Dclose(d);
            const hid_t d2 = H5Dcreate2(fid, "vec__step_index",
                                        H5T_NATIVE_UINT64, sp, H5P_DEFAULT,
                                        dcpl, H5P_DEFAULT);
            H5Dclose(d2);
            H5Pclose(dcpl);
            H5Sclose(sp);
        }
        H5Fclose(fid);
    });
    return p;
}

const h5b::dataset_desc& desc_of(const h5b::file_model& m,
                                 std::string_view display)
{
    for (const auto& d : m.datasets) {
        if (d.display_name == display) return d;
    }
    REQUIRE(false);
    std::unreachable();
}

} // namespace

TEST_CASE("dataset_reader: pipeline streams 3M rows in multiple exact batches")
{
    hdf5::executor exec;
    const fs::path path = make_test_file(exec);

    const auto m = h5b::open_and_map(exec, path);
    REQUIRE(m.has_value());
    const auto& act = desc_of(*m, "act");
    CHECK(act.rows == n_rows_1d);

    const auto fh = h5b::open_file_readonly(exec, path);
    REQUIRE(fh.has_value());

    {
        h5b::read_pipeline pipe(exec, *fh, act, 0, act.rows);
        CHECK(pipe.rows_per_batch() % 1024 == 0); // whole chunks
        CHECK(pipe.rows_per_batch() * sizeof(float) >=
              h5b::read_pipeline::min_job_bytes);

        std::uint64_t seen = 0, batches = 0;
        while (true) {
            const auto b = pipe.next();
            REQUIRE(b.has_value());
            if (!b->has_value()) break;
            const auto& batch = **b;
            CHECK(batch.first_row == seen);
            const auto* f =
                reinterpret_cast<const float*>(batch.data.data());
            // Spot-check the pattern at both ends of every batch.
            CHECK(f[0] == static_cast<float>(batch.first_row % 977));
            CHECK(f[batch.n_rows - 1] ==
                  static_cast<float>((batch.first_row + batch.n_rows - 1) % 977));
            seen += batch.n_rows;
            ++batches;
        }
        CHECK(seen == n_rows_1d);
        CHECK(batches >= 3); // 12 MB / 4 MB jobs
    }

    h5b::close_file_handle(exec, *fh);
    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("dataset_reader: partial range, 2-D rows, and read_block")
{
    hdf5::executor exec;
    const fs::path path = make_test_file(exec);
    const auto     m    = h5b::open_and_map(exec, path);
    REQUIRE(m.has_value());
    const auto fh = h5b::open_file_readonly(exec, path);
    REQUIRE(fh.has_value());

    const auto& act = desc_of(*m, "act");
    const auto& vec = desc_of(*m, "vec");

    // Partial 1-D range not aligned to anything.
    {
        h5b::read_pipeline pipe(exec, *fh, act, 100, 1000);
        const auto         b = pipe.next();
        REQUIRE(b.has_value());
        REQUIRE(b->has_value());
        CHECK((*b)->first_row == 100);
        CHECK((*b)->n_rows == 1000);
        const auto* f = reinterpret_cast<const float*>((*b)->data.data());
        CHECK(f[0] == static_cast<float>(100 % 977));
        CHECK(f[999] == static_cast<float>(1099 % 977));
        const auto done = pipe.next();
        REQUIRE(done.has_value());
        CHECK_FALSE(done->has_value());
    }

    // 2-D rows through the shape-matched path.
    {
        h5b::read_pipeline pipe(exec, *fh, vec, 0, vec.rows);
        const auto         b = pipe.next();
        REQUIRE(b.has_value());
        REQUIRE(b->has_value());
        CHECK((*b)->n_rows == n_rows_2d);
        const auto* f = reinterpret_cast<const float*>((*b)->data.data());
        CHECK(f[0] == 0.0f);
        CHECK(f[7] == 7.0f);
        CHECK(f[123 * 8 + 5] == static_cast<float>(123 * 10 + 5));
    }

    // read_block: chunk-aligned block, tail block, out-of-range block.
    {
        const std::uint64_t rpb = h5b::default_block_rows(vec);
        CHECK(rpb == 16); // vec's chunk[0]

        const auto blk = h5b::read_block(exec, *fh, vec, 1, 2, rpb);
        REQUIRE(blk.has_value());
        CHECK((*blk)->first_row == 32);
        CHECK((*blk)->n_rows == 16);
        const auto* f =
            reinterpret_cast<const float*>((*blk)->bytes.data());
        CHECK(f[0] == 320.0f); // row 32, col 0

        // Tail block is short: 500 rows / 16 -> last block has 4 rows.
        const auto tail = h5b::read_block(exec, *fh, vec, 1, 31, rpb);
        REQUIRE(tail.has_value());
        CHECK((*tail)->first_row == 496);
        CHECK((*tail)->n_rows == 4);

        const auto oob = h5b::read_block(exec, *fh, vec, 1, 32, rpb);
        REQUIRE_FALSE(oob.has_value());
        CHECK(oob.error().category == h5b::error_category::read);
    }

    h5b::close_file_handle(exec, *fh);
    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("dataset_reader: pipeline error on a vanished dataset is sticky")
{
    hdf5::executor exec;
    const fs::path path = make_test_file(exec);
    const auto     m    = h5b::open_and_map(exec, path);
    REQUIRE(m.has_value());
    const auto fh = h5b::open_file_readonly(exec, path);
    REQUIRE(fh.has_value());

    h5b::dataset_desc bogus = desc_of(*m, "act");
    bogus.h5_path           = "/no_such_dataset";

    h5b::read_pipeline pipe(exec, *fh, bogus, 0, bogus.rows);
    const auto         e1 = pipe.next();
    REQUIRE_FALSE(e1.has_value());
    CHECK(e1.error().category == h5b::error_category::read);
    const auto e2 = pipe.next(); // sticky
    REQUIRE_FALSE(e2.has_value());

    h5b::close_file_handle(exec, *fh);
    std::error_code ec;
    fs::remove(path, ec);
}
