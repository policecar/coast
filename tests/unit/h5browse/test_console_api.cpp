// h5browse console_api — the range-style façade over the read layer that an
// embedded interpreter reaches through `open()`. Exercises discovery, streaming scan,
// random access, column/step extraction, dtype→double widening, the typed
// as<T>() escape hatch, and the throwing error surface — all against a real
// executor and a real (temp) HDF5 file.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "h5browse/console_api.h"
#include "hdf5/executor.h"
#include "logging/hdf5_utils.h"

#include <hdf5.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Observe-convention file so open_and_map yields real dataset_descs:
//   /act              f32[2000],      chunk 256,  act[i] = i % 100
//   /act__step_index  u64[2000],                  steps[i] = 10*i
//   /vec              i32[300 x 4],   chunk [64,4], vec[r][c] = r*10 + c
constexpr std::uint64_t n_act = 2000;
constexpr std::uint64_t n_vec = 300;

fs::path make_test_file(hdf5::executor& exec)
{
    const fs::path  p = fs::temp_directory_path() / "h5b_console_api_test.h5";
    std::error_code ec;
    fs::remove(p, ec);

    exec.submit_sync([&] {
        const hid_t fid = H5Fcreate(p.string().c_str(), H5F_ACC_TRUNC,
                                    H5P_DEFAULT, H5P_DEFAULT);
        REQUIRE(fid >= 0);
        {
            const hid_t md = H5Gcreate2(fid, "metadata", H5P_DEFAULT, H5P_DEFAULT,
                                        H5P_DEFAULT);
            REQUIRE(hdf5::write_attr(md, "type", std::string_view{"observe_log"}) >= 0);
            REQUIRE(hdf5::write_attr(md, "schema_version", std::string_view{"1.0"}) >= 0);
            H5Gclose(md);
        }
        { // /act + /act__step_index
            const hsize_t dims[1]  = {n_act};
            const hsize_t chunk[1] = {256};
            const hid_t   sp   = H5Screate_simple(1, dims, nullptr);
            const hid_t   dcpl = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(dcpl, 1, chunk);

            std::vector<float> vals(n_act);
            for (std::uint64_t i = 0; i < n_act; ++i) {
                vals[i] = static_cast<float>(i % 100);
            }
            const hid_t d = H5Dcreate2(fid, "act", H5T_NATIVE_FLOAT, sp,
                                       H5P_DEFAULT, dcpl, H5P_DEFAULT);
            H5Dwrite(d, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals.data());
            H5Dclose(d);

            std::vector<std::uint64_t> steps(n_act);
            for (std::uint64_t i = 0; i < n_act; ++i) steps[i] = 10 * i;
            const hid_t d2 = H5Dcreate2(fid, "act__step_index", H5T_NATIVE_UINT64,
                                        sp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
            H5Dwrite(d2, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     steps.data());
            H5Dclose(d2);
            H5Pclose(dcpl);
            H5Sclose(sp);
        }
        { // /vec  (2-D, i32)
            const hsize_t dims[2]  = {n_vec, 4};
            const hsize_t chunk[2] = {64, 4};
            const hid_t   sp   = H5Screate_simple(2, dims, nullptr);
            const hid_t   dcpl = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(dcpl, 2, chunk);
            std::vector<std::int32_t> vals(n_vec * 4);
            for (std::uint64_t r = 0; r < n_vec; ++r) {
                for (std::uint64_t c = 0; c < 4; ++c) {
                    vals[r * 4 + c] = static_cast<std::int32_t>(r * 10 + c);
                }
            }
            const hid_t d = H5Dcreate2(fid, "vec", H5T_NATIVE_INT32, sp,
                                       H5P_DEFAULT, dcpl, H5P_DEFAULT);
            H5Dwrite(d, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, vals.data());
            H5Dclose(d);
            const hid_t d2 = H5Dcreate2(fid, "vec__step_index", H5T_NATIVE_UINT64,
                                        sp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
            H5Dclose(d2);
            H5Pclose(dcpl);
            H5Sclose(sp);
        }
        H5Fclose(fid);
    });
    return p;
}

// Find a dataset_handle by display name, tolerating whatever order the mapper
// assigns.
h5b::console::dataset_handle by_name(const h5b::console::file_console& fc,
                                     std::string_view want)
{
    return fc.dataset(want);
}

} // namespace

TEST_CASE("console_api: discovery, metadata, and unknown-dataset throw")
{
    hdf5::executor exec;
    const fs::path path = make_test_file(exec);

    h5b::console::file_console h5(exec, path);

    CHECK(h5.count() >= 2);
    const auto names = h5.list();
    CHECK(std::find(names.begin(), names.end(), "act") != names.end());
    CHECK(std::find(names.begin(), names.end(), "vec") != names.end());
    CHECK(h5.family() == std::string_view{"observe"});

    auto act = h5.dataset("act");
    CHECK(act.name() == "act");
    CHECK(act.rows() == n_act);
    CHECK(act.width() == 1);
    CHECK(act.dtype() == h5b::scalar_type::f32);

    CHECK_THROWS_AS((void)h5.dataset("does_not_exist"), std::runtime_error);

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("console_api: streaming scan widens to double and covers all rows")
{
    hdf5::executor exec;
    const fs::path path = make_test_file(exec);
    h5b::console::file_console h5(exec, path);

    auto          act = by_name(h5, "act");
    std::uint64_t seen = 0;
    double        sum  = 0.0;
    std::uint64_t last_index = 0;
    for (auto row : act.scan()) {
        CHECK(row.index() == seen);      // rows arrive in order
        CHECK(row.at(0) == doctest::Approx(static_cast<double>(seen % 100)));
        sum += row.at(0);
        last_index = row.index();
        ++seen;
    }
    CHECK(seen == n_act);
    CHECK(last_index == n_act - 1);
    // 20 full cycles of 0..99 -> each value appears 20 times; sum = 20*sum(0..99).
    CHECK(sum == doctest::Approx(20.0 * (99.0 * 100.0 / 2.0)));

    // Partial range.
    std::uint64_t partial = 0;
    for (auto row : act.scan(500, 100)) {
        CHECK(row.index() == 500 + partial);
        ++partial;
    }
    CHECK(partial == 100);

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("console_api: random row(), 2-D shape, as<T>() and column()")
{
    hdf5::executor exec;
    const fs::path path = make_test_file(exec);
    h5b::console::file_console h5(exec, path);

    auto vec = by_name(h5, "vec");
    CHECK(vec.rows() == n_vec);
    CHECK(vec.width() == 4);
    CHECK(vec.dtype() == h5b::scalar_type::i32);

    // Random access to an arbitrary row; value pattern r*10 + c.
    auto r123 = vec.row(123);
    CHECK(r123.index() == 123);
    CHECK(r123.width() == 4);
    CHECK(r123.shape().size() == 1);
    CHECK(r123.shape()[0] == 4);
    CHECK(r123.at(0) == doctest::Approx(1230.0));
    CHECK(r123.at(3) == doctest::Approx(1233.0));

    // Zero-copy typed view matches; wrong type throws.
    auto span = r123.as<std::int32_t>();
    CHECK(span.size() == 4);
    CHECK(span[2] == 1232);
    CHECK_THROWS_AS((void)r123.as<float>(), std::runtime_error);

    // as_copy<T>() owns its data: same values, same dtype check, but detached
    // from the row_view (and thus from the cache block) it came from.
    const auto copy = r123.as_copy<std::int32_t>();
    CHECK(copy.size() == 4);
    CHECK(std::equal(copy.begin(), copy.end(), span.begin()));
    CHECK(static_cast<const void*>(copy.data()) != static_cast<const void*>(span.data()));
    CHECK_THROWS_AS((void)r123.as_copy<float>(), std::runtime_error);

    // Out-of-range accessors throw.
    CHECK_THROWS_AS((void)r123.at(4), std::runtime_error);
    CHECK_THROWS_AS((void)vec.row(n_vec), std::runtime_error);
    CHECK_THROWS_AS((void)vec.column(4), std::runtime_error);

    // column() gathers one column across all rows.
    const auto col1 = vec.column(1);
    REQUIRE(col1.size() == n_vec);
    CHECK(col1[0] == doctest::Approx(1.0));      // row 0, col 1
    CHECK(col1[7] == doctest::Approx(71.0));     // row 7: 70 + 1
    CHECK(col1[n_vec - 1] == doctest::Approx(static_cast<double>((n_vec - 1) * 10 + 1)));

    std::error_code ec;
    fs::remove(path, ec);
}

// The reason as_copy<T>() exists: a row's values commonly need to outlive the
// row_view (a fixed reference vector compared against a long scan). as<T>() spans
// die with the block that backs them — here we free the entire cache by dropping
// the file_console, which is the observable, sanitizer-checkable extreme of the
// LRU eviction that bites in real use. Doing this with as<T>() is a use-after-free.
TEST_CASE("console_api: as_copy<T>() outlives the file_console it came from")
{
    hdf5::executor exec;
    const fs::path path = make_test_file(exec);

    std::vector<std::int32_t> copy;
    {
        h5b::console::file_console h5(exec, path);
        copy = by_name(h5, "vec").row(123).as_copy<std::int32_t>();
    } // cache_ destroyed here; every row_block backing a span is freed

    REQUIRE(copy.size() == 4);
    CHECK(copy[0] == 1230);
    CHECK(copy[3] == 1233);

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("console_api: steps() reads the step-index sibling")
{
    hdf5::executor exec;
    const fs::path path = make_test_file(exec);
    h5b::console::file_console h5(exec, path);

    const auto steps = by_name(h5, "act").steps();
    REQUIRE(steps.size() == n_act);
    CHECK(steps[0] == 0);
    CHECK(steps[1] == 10);
    CHECK(steps[n_act - 1] == 10 * (n_act - 1));

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("console_api: open() returns a shared handle; multiple coexist")
{
    hdf5::executor exec;
    const fs::path path = make_test_file(exec);
    h5b::console::set_executor(exec);

    // Two independent handles to the SAME file coexist (multiple read-only opens
    // are allowed): each is a self-contained file_console over the shared exec.
    std::shared_ptr<h5b::console::file_console> a = h5b::console::open(path);
    std::shared_ptr<h5b::console::file_console> b = h5b::console::open(path);
    REQUIRE(a);
    REQUIRE(b);
    CHECK(a.get() != b.get());
    CHECK(a->dataset("act").rows() == n_act);
    CHECK(b->dataset("act").rows() == n_act);

    // A shared handle keeps the file alive after the "opener" reference drops.
    a.reset();
    CHECK(b->dataset("act").rows() == n_act);

    CHECK_THROWS_AS((void)h5b::console::open("/no/such/file.h5"), std::runtime_error);

    std::error_code ec;
    fs::remove(path, ec);
}

TEST_CASE("console_api: var_name_for derives a valid identifier from the path stem")
{
    using h5b::console::var_name_for;
    CHECK(var_name_for("run/mnist_seed0.h5") == "f_mnist_seed0");
    CHECK(var_name_for("/tmp/cifar-run.h5") == "f_cifar_run");   // '-' -> '_'
    CHECK(var_name_for("123.h5") == "f_123");                    // leading digit is fine after f_
    CHECK(var_name_for("a.b.h5") == "f_a_b");                    // stem keeps inner dot -> '_'
}
